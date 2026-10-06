"""ZHOUYI: the Zhouyi V3 NPU (AIPU V3), through the kernel driver's /dev/aipu.

Kernels are OpenCL C compiled by the installed vendor toolchain (`support/compiler_zhouyi.py`) and run as jobs of TEC tasks built in
`support/zhouyi/`. The fp16 4x4x4 `mma` of the TEC matrix unit is a TensorCore, reached through inline assembly."""
from __future__ import annotations
import ctypes, functools, math, struct, time
from tinygrad.codegen.opt.tc import TensorCore
from tinygrad.device import Compiled, Allocator, BufferSpec
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import getenv, DEBUG, to_mv, mv_address
from tinygrad.renderer.cstyle import ClangRenderer, uops_to_dtypes, wmma_args
from tinygrad.uop.ops import Ops, PatternMatcher, UPat
from tinygrad.uop.decompositions import TRANSCENDENTAL_DTYPES, UOp, _lazy_map_numbers, cody_waite_reduction
from tinygrad.uop.decompositions import sin_poly, sin_poly_small
from tinygrad.runtime.support.compiler_zhouyi import ZhouyiCompiler
from tinygrad.runtime.support.zhouyi import ZhouyiError, dev as _dev, hangid as _hid
from tinygrad.runtime.support.zhouyi.dev import Buffer as ZhouyiBuffer, RawDevice, MM_REUSE, MM_TCB, TCB_LEN
from tinygrad.runtime.support.zhouyi.launch import Launch, launch_args, pack_blobs, stack_frame_bytes, unpack_blob

ZHOUYI_TECS, ZHOUYI_CORES = 4, getenv("ZHOUYI_CORES", 3)

# A: lane 4m+k, B: lane 4n+k; the accumulator is a 16-float struct (TC_ACC) held in a register pair.
ZHOUYI_TC = TensorCore(dims=(4, 4, 4), threads=1, elements_per_thread=(16, 16, 16), dtype_in=dtypes.half, dtype_out=dtypes.float,
                       opts=("u0", "u0", "u1", "u1"),
                       swizzle=(((), ("r0", "r1", "u2", "u3"), ("u0", "u1")), ((), ("r0", "r1", "u0", "u1"), ("u2", "u3"))))
TC_ACC = "typedef struct { float v[16]; } __attribute__((aligned(32))) float16;"

# 2/pi in 12-bit digits: 2/pi = sum(TWO_OVER_PI_12[k] * 2**(-12*(k+1)))
TWO_OVER_PI_12 = (2607, 2435, 1764, 3652, 338, 2556, 629, 2001, 3923, 1245, 3085, 2914, 2393, 2364, 1081, 65)

def reduce32(x:UOp) -> tuple[UOp, UOp]:
  """float32 x >= 0 -> (x mod pi/2 in [-pi/4, pi/4], quadrant), in float32 and int32 only: upstream's Payne-Hanek needs 64-bit
  integers, which this target emulates slowly. x = hi + lo with 12 significant bits each, so every hi*digit and lo*digit
  product is exact; each contributes its integer part mod 4 to the quadrant and its fraction to the remainder.
  Rounding is by magic-number addition (sleef's rempisub): the vendor compiler wraps every float->int cvt in control-register
  writes to switch the rounding mode, which made the kernels huge and, unrolled 4-wide, hang."""
  M1, M2 = 1.5 * 2**23, 1.5 * 2**27   # t + M1 rounds t to the nearest integer, t + M2 to the nearest multiple of 16
  def whole_frac(t:UOp) -> tuple[UOp, UOp]:
    """|t| < 2**26 -> (integer part mod 4, fraction in [-0.5, 0.5])"""
    u = t - ((t + M2) - M2)   # in [-8, 8], exact
    n = u + M1                # mantissa = 2**22 + round(u), so its low bits are round(u) mod 4
    return n.bitcast(dtypes.int32) & 3, u - (n - M1)
  hi = (x.bitcast(dtypes.int32) & -4096).bitcast(dtypes.float32)
  q, f = x.const_like(0).cast(dtypes.int32), x.const_like(0.0)
  for h in (hi, x - hi):
    for k, c in enumerate(TWO_OVER_PI_12):
      s = 64 if k >= 8 else 0   # keep the small digits out of float32 denormals
      t = (h * 2.0**-s) * (c * 2.0**(s-12*(k+1)))
      qi, fi = whole_frac((t.abs() < 2.0**26).where(t, 0.0))   # bigger products are multiples of 8
      q, f = q + qi, f + fi
  qf, r = whole_frac(f)
  return r * (math.pi / 2), (q + qf) & 3

def xsin(d:UOp, switch_over:float=30.0) -> UOp:
  """upstream `xsin` with `reduce32` in place of Payne-Hanek above `switch_over`; an odd quadrant takes cos(r) = sin(pi/2-|r|), so
  the polynomial never sees more than pi/2."""
  x = _lazy_map_numbers(d, d.const_like(0.0), d.const_like(0.0), d.const_like(0.0), d)
  x_sign = x.ne(0).where((x<0).where(x.const_like(-1), x.const_like(1)), x.const_like(0))
  x_abs = x * x_sign
  r, q = reduce32(x_abs.cast(dtypes.float32))
  r_small, q_small = cody_waite_reduction(x_abs)
  r = (q & 1).ne(0).where(math.pi/2 - r.abs(), r).cast(d.dtype)
  large = sin_poly(r) * (q & 2).ne(0).where(r.const_like(-1), r.const_like(1))
  result = (x_abs < switch_over).where(sin_poly_small(r_small, q_small), large) * x_sign
  return _lazy_map_numbers(d, d.const_like(math.nan), d.const_like(math.nan), d.const_like(math.nan), result)

# btaipucc lowers a 16-bit int <-> half bitcast through a stack slot and schedules the reload before the store (even volatile),
# so the half reads stale stack. A 32-bit bitcast is a register move: go through float32 and rebuild the 16-bit pattern.
def bits_to_half(x:UOp) -> UOp:
  h = x.cast(dtypes.uint32) & 0xFFFF
  s, e, m = (h & 0x8000) << 16, (h >> 10) & 31, h & 1023
  sub = (m.cast(dtypes.float32) * 2.0**-24).bitcast(dtypes.uint32) | s
  bits = e.eq(0).where(sub, e.eq(31).where(s | 0x7F800000 | (m << 13), s | ((e + 112) << 23) | (m << 13)))
  return bits.bitcast(dtypes.float32).cast(dtypes.half)

def half_to_bits(x:UOp, dt:DType) -> UOp:
  f = x.cast(dtypes.float32)
  b = f.bitcast(dtypes.uint32)
  s, e, m = (b >> 16) & 0x8000, (b >> 23) & 255, b & 0x7FFFFF
  sub = s | (f.abs() * 2.0**24).cast(dtypes.uint32)
  bits = e.eq(255).where(s | 0x7C00 | (m >> 13), (e < 113).where(sub, s | ((e - 112) << 10) | (m >> 13)))
  return bits.cast(dt)

class ZhouyiRenderer(ClangRenderer):
  device = "ZHOUYI"
  # THREAD becomes a `core_id` argument, one task per TEC. Generated kernels stay on one core's 4 TECs: tasks on different
  # cores storing into the same 64-byte line lose one core's bytes, and a thread split does not align its boundaries.
  has_local, has_threads, has_shared = False, True, False
  global_max, local_max = (ZHOUYI_TECS, 1, 1), None
  supports_float4 = False
  extra_args: list[str] = []
  code_for_workitem: dict = {}
  tensor_cores = [ZHOUYI_TC]
  string_rewrite = PatternMatcher([
    (UPat(Ops.GEP, name="x", src=(UPat(dtype=dtypes.float.vec(16)),)), lambda ctx, x: f"{ctx[x.src[0]]}.v[{x.arg[0]}]"),
    (UPat(Ops.CONST, dtype=dtypes.float.vec(16), name="x"), lambda ctx, x: "(float16){" + ",".join([f"{x.arg}f"]*16) + "}"),
  ]) + ClangRenderer.string_rewrite
  # SIN is lowered before the dtype decompositions: its large-argument reduction uses 64-bit integers, which this target emulates
  pre_matcher = PatternMatcher([(UPat(Ops.SIN, dtype=TRANSCENDENTAL_DTYPES, src=(UPat.var("d"),)), xsin),
    (UPat(Ops.SIN, dtype=dtypes.bfloat16, src=(UPat.var("d"),)), lambda d: xsin(d.cast(dtypes.float32)).cast(dtypes.bfloat16))])
  extra_matcher = PatternMatcher([
    (UPat(Ops.BITCAST, dtype=dtypes.half, src=(UPat.var("x", dtype=(dtypes.uint16, dtypes.int16)),)), bits_to_half),
    (UPat(Ops.BITCAST, dtype=(dtypes.uint16, dtypes.int16), src=(UPat.var("x", dtype=dtypes.half),), name="b"),
     lambda x, b: half_to_bits(x, b.dtype)),
  ]) + ClangRenderer.extra_matcher
  # OpenCL C 1.2 without the builtin header: address spaces and `half` exist, `uint`/`uchar` and the math builtins do not.
  kernel_typedef, buffer_prefix, smem_prefix, buffer_suffix = "__kernel void", "__global ", "", " restrict"
  type_map = {dtypes.bool: "bool", dtypes.half: "half", dtypes.int8: "char", dtypes.uint8: "unsigned char",
              dtypes.int16: "short", dtypes.uint16: "unsigned short", dtypes.uint32: "unsigned int"}

  # __builtin_truncf goes through an int32 convert that saturates, and its (int)truncf(x) chain is scheduled wrongly in loops
  # (a stale register read); |x| >= 2**23 (and inf, nan) is already integral
  code_for_op = {**ClangRenderer.code_for_op,
                 Ops.TRUNC: lambda x, dtype: f"((({x})<8388608.0f&&({x})>-8388608.0f)?__builtin_truncf({x}):({x}))"}

  def __init__(self): self.compiler = ZhouyiCompiler()
  def render_vector_prefix(self, dt:DType) -> str: return TC_ACC if dt == dtypes.float.vec(16) else super().render_vector_prefix(dt)
  def wmma_prefix(self, name:str) -> str:
    # the accumulator pair t30/t31 cannot be requested by a constraint, so it is named, loaded and stored around the mma
    return f"""static float16 __{name}(half16 a, half16 b, float16 c) {{
  __asm volatile("{{ ld t30, [%0+0]; ld t31, [%0+32]; }}\\n{{ mma t30.fp32, t31.fp32, %1.fp16, %2.fp16; }}\\n"
                 "{{ st t30, [%0+0]; }}\\n{{ st t31, [%0+32]; }}"
                 :: "r"(&c), "t"(a), "t"(b) : "memory", "t30", "t31");
  return c;
}}"""
  def _render_defines(self, uops) -> list[str]:
    return ["#pragma OPENCL EXTENSION cl_khr_fp16 : enable"] + [self.render_vector_prefix(dt) for dt in uops_to_dtypes(uops) if dt.count > 1] + \
           [self.wmma_prefix(name) for name, *_ in wmma_args(uops)]

class ZhouyiAllocator(Allocator['ZhouyiDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> ZhouyiBuffer:
    # external_ptr: an NPU-view address owned elsewhere (e.g. a window slot re-pointed at weight memory by the KMD's SLOT_MAP):
    # nothing is allocated or freed here, and the host has no mapping of it (va 0)
    if options.external_ptr is not None: return ZhouyiBuffer(options.external_ptr, 0, size, 0)
    try: return self.dev.raw.req_buf(size, MM_REUSE)
    except OSError as e:
      if e.errno == 12: raise MemoryError(f"ZHOUYI: out of device memory allocating {size} B") from e
      raise
  def _free(self, opaque:ZhouyiBuffer, options:BufferSpec):
    if options.external_ptr is None: self.dev.raw.free_buf(opaque)
  # host reads first drop the CPU's stale clean lines of the buffer: the device writes DDR behind the kernel's cacheable alias
  def _as_buffer(self, src:ZhouyiBuffer) -> memoryview:
    self.dev.raw.cache_invalidate(src)
    return to_mv(src.va, src.nbytes)
  def _copyin(self, dest:ZhouyiBuffer, src:memoryview):
    self.dev.raw.drain()   # a job left in flight (the graph's async tail) may read dest
    ctypes.memmove(dest.va, mv_address(src), src.nbytes)
  def _copyout(self, dest:memoryview, src:ZhouyiBuffer):
    self.dev.raw.cache_invalidate(src)
    ctypes.memmove(mv_address(dest), src.va, dest.nbytes)
  def _transfer(self, dest:ZhouyiBuffer, src:ZhouyiBuffer, sz:int, src_dev:ZhouyiDevice, dest_dev:ZhouyiDevice):
    src_dev.raw.cache_invalidate(src)   # ZHOUYI:n are one NPU: a device-to-device copy is a host memmove between mapped buffers
    ctypes.memmove(dest.va, src.va, sz)
  def _offset(self, buf:ZhouyiBuffer, size:int, offset:int):
    return ZhouyiBuffer(buf.pa+offset, buf.dev_offset+offset, size, buf.va+offset, buf.offset+offset)

def submit_wait(raw:RawDevice, subs) -> float:
  st = time.perf_counter()
  raw.wait_jobs([raw.submit(*s) for s in subs])
  return time.perf_counter() - st

def _job_split(ntasks:int) -> list[int]:
  # a job is one group on one core: ceil(ntasks/4) jobs, balanced
  q, r = divmod(ntasks, njobs := -(-ntasks // ZHOUYI_TECS))
  return [q + (1 if j < r else 0) for j in range(njobs)]

class ZhouyiProgram:
  def __init__(self, dev:ZhouyiDevice, name:str, lib:bytes, runtimevars:dict[str, int]|None=None, **kwargs):
    self.dev, self.name, self.lib, self.runtimevars = dev, name, lib, runtimevars or {}
    img, rodata = unpack_blob(lib)
    self.launch = Launch(dev.raw, img, rodata)
    self.text, self.cnst = self.launch.text, self.launch.cnst
    if _hid.LEVEL:   # ZHOUYI_HANG_ID: name this image for the failure report (frame: the stack bytes below sp it may use)
      _hid.register_text(dev.raw.dev_addr(self.text), name, frame=stack_frame_bytes(img), lib=_hid.lib_id(lib), stamps=True)
    self.ntasks, self._chain_gen, self._param_gen = 0, 0, 0
    self._param_key: tuple[int, ...]|None = None
    self._blobs: dict[tuple[int, ...], list[bytes]] = {}
    self._one: tuple[int, int, int]|None = None

  def _chain(self, ntasks:int):
    """Per-task buffers and one TCB chain per job for `ntasks` tasks; `_chain_gen` tells graphs they moved."""
    if ntasks == self.ntasks: return
    self._chain_gen += 1
    self._param_key, self._blobs, self._one = None, {}, None
    old = list(self.launch.params) if _hid.LEVEL >= 2 else ()
    self.launch.chain(_job_split(ntasks))
    lay = self.launch
    if _hid.LEVEL >= 2: _hid.note_params(self.dev.raw, lay.params, forget=old)   # their stamp words are poisoned before each submit
    self.stacks, self.params, self.privs = lay.stacks, lay.params, lay.privs
    self.ntasks = ntasks

  # the launch's own TCB chains, built on first use (a graph's clone runs its tasks only inside the graph's chains: it has none)
  @property
  def chains(self): return self.launch.chains
  @property
  def tcb_bufs(self): return self.launch.tcb_bufs

  def group_slots_per_core(self) -> list:
    a = self.dev.raw.dev_addr
    slots = self.launch.task_slots()
    return [_dev.GroupSlot(spc=a(self.text), cp=a(self.cnst), tasks=slots[ZHOUYI_TECS*c:ZHOUYI_TECS*(c+1)])
            for c in range(-(-self.ntasks//ZHOUYI_TECS))]

  def group_slot(self) -> _dev.GroupSlot:
    if not 0 < self.ntasks <= ZHOUYI_TECS: raise RuntimeError(f"ZHOUYI: {self.name} has {self.ntasks} tasks; a chain group is one core")
    return self.group_slots_per_core()[0]

  def _one_job(self) -> tuple[int, int, int]:
    """A multi-core launch as ONE job: the per-core groups run concurrently and a barrier task closes the chain (the driver
    retires a job on its last TCB, so the chain must not end on a group another core may still be running)."""
    if self._one is None:
      raw, g = self.dev.raw, self.group_slots_per_core()
      if getattr(self, "_one_tcbs", None) is None: self._one_tcbs = raw.req_buf(_dev.chain_tcbs(ZHOUYI_CORES*ZHOUYI_TECS + 2)*TCB_LEN, MM_TCB)
      _dev.build_group_tcbs(raw, self._one_tcbs, g + [barrier_group(raw)], dep=[0]*len(g) + [_dev.DEP_PRE_ALL])
      pa, total = self._one_tcbs.pa, self.ntasks + 1
      self._one = (pa, pa + _dev.TASK_INDEX*TCB_LEN, pa + (_dev.TASK_INDEX + total - 1)*TCB_LEN)
    return self._one

  def _launch_args(self, bufs, vals, ntasks:int) -> tuple[list[int], int|None]:
    try: return launch_args(self.dev.raw, bufs, vals, ntasks, self.runtimevars)
    except ZhouyiError as e: raise RuntimeError(str(e)) from e
  def _pack_blobs(self, args:list[int], cid:int|None) -> list[bytes]: return pack_blobs(args, cid, len(self.params))
  def write_params(self, blobs, nb:int) -> None:
    self._param_gen, self._param_key = self._param_gen + 1, None
    for p, b in zip(self.params, blobs): ctypes.memmove(p.va, b, nb)

  def stage(self, bufs, global_size=(1,1,1), local_size=(1,1,1), vals:tuple[int, ...]=()) -> bool:
    """Everything a launch does short of submitting (a fused chain stages several Programs and submits once)."""
    if tuple(local_size) != (1, 1, 1) or tuple(global_size)[1:] != (1, 1) or not 1 <= global_size[0] <= ZHOUYI_CORES*ZHOUYI_TECS:
      raise RuntimeError(f"ZHOUYI launch {global_size}x{local_size}: only up to {ZHOUYI_CORES*ZHOUYI_TECS} `core_id` threads are dispatchable")
    self._chain(global_size[0])
    args, cid = self._launch_args(bufs, vals, self.ntasks)
    if self._param_key != (key := tuple(args)):
      if (blobs := self._blobs.get(key)) is None:
        blobs = self._pack_blobs(args, cid)
        if len(self._blobs) < 1024: self._blobs[key] = blobs
      self.write_params(blobs, len(args)*4)
      self._param_key = key
    return True

  def exception_vectors(self) -> list[int]:
    """Per task, the exception vector it took (0: none): the image's vectors record it in the private buffer's word 0."""
    return [ctypes.c_uint32.from_address(p.va).value for p in self.privs]

  def __call__(self, *bufs:ZhouyiBuffer, global_size=(1,1,1), local_size=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    self.stage(bufs, global_size, local_size, vals)
    for p in self.privs: ctypes.memset(p.va, 0, 4)
    try: et = submit_wait(self.dev.raw, [self._one_job()] if self.ntasks > ZHOUYI_TECS else self.chains)
    except ZhouyiError as e:
      hid = ""
      if _hid.LEVEL:
        hid = "\n".join([f"\n[ZHOUYI_HANG_ID] plain launch of {self.name}, {self.ntasks} task(s), args:"] + _hid.describe_bufs(self.dev.raw, bufs))
        _hid.log(hid)
      raise ZhouyiError(f"{e}\n  in {self.name}: exception vector per task {self.exception_vectors()} (0: none). An access the "
                        "SMMU refuses is not an NPU exception: the TEC waits forever; `dmesg | grep F_TRANSLATION` names the address" + hid) from None
    return et if wait else None

class ZhouyiDevice(Compiled):
  def __init__(self, device:str=""):
    try: self.raw = RawDevice()
    except ZhouyiError as e: raise RuntimeError(str(e)) from e
    if DEBUG >= 1: print(f"ZHOUYI: {self.raw}")
    if self.raw.tec_cnt != ZHOUYI_TECS or not 1 <= ZHOUYI_CORES <= self.raw.core_cnt:
      raise RuntimeError(f"ZHOUYI: {self.raw.core_cnt} cores x {self.raw.tec_cnt} TECs, expected {ZHOUYI_CORES} x {ZHOUYI_TECS}")
    from tinygrad.runtime.graph.zhouyi import ZhouyiGraph
    super().__init__(device, ZhouyiAllocator(self), [ZhouyiRenderer], functools.partial(ZhouyiProgram, self), ZhouyiGraph)
  def synchronize(self): self.raw.drain()   # the graphs' async tail: wait for the job left in flight

# custom-op arguments `ZhouyiGraph` may fuse into its chains; `extra.zhouyi.ops` registers them
CHAIN_MEMBER_ARGS: set[str] = set()

# A do-nothing task image (vector table, task prologue, epilogue) closing a fused chain.
_BARRIER_TEXT = bytes.fromhex(
  "22008001ff7f0c40ff7f0c40ff7f0cc01e008001ff7f0c40ff7f0c40ff7f0cc01c008001ff7f0c40ff7f0c40ff7f0cc01a008001ff7f0c40ff7f0c40ff7f0cc0"
  "18008001ff7f0c40ff7f0c40ff7f0cc016008001ff7f0c40ff7f0c40ff7f0cc014008001ff7f0c40ff7f0c40ff7f0cc012008001ff7f0c40ff7f0c40ff7f0cc0"
  "10008001ff7f0c40ff7f0c40ff7f0cc00e008001ff7f0c40ff7f0c40ff7f0cc00c008001ff7f0c40ff7f0c40ff7f0cc00a008001ff7f0c40ff7f0c40ff7f0cc0"
  "08008001ff7f0c40ff7f0c40ff7f0cc006008001ff7f0c40ff7f0c40ff7f0cc004008001ff7f0c40ff7f0c40ff7f0cc002008001ff7f0c40ff7f0c40ff7f0cc0"
  "00008001ff7f0c40ff7f0c40ff7f0cc049800401ff7f0c40ff7f0c40ff7f0cc0ff7f0c40297de041ff7f0c40ff7f0cc049800001ff7f0c40ff7f0c40ff7f0cc0"
  "09820401ff7f0c40ff7f0c40ff7f0cc0ff7f0c40297de041ff7f0c40ff7f0cc009820001ff7f0c40ff7f0c40ff7f0cc0980204015a6b0d401c00d0001700f6fd"
  "ff7f0c405a6fe2411c40ff001dd310f0ff7f0c40ff7f0c4019e31070ff7f0cc0ff7f0c40ff7f0c401bf31070170311f042100401ff7f0c40ff7f0c40ff7f0cc0"
  "ff7f0c4042901040ff7f0c40ff7f0cc042100001ff7f0c40ff7f0c40ff7f0cc003120401ff7f0c40ff7f0c40ff7f0cc0ff7f0c40ff7f0c40a32fe000ff7f0cc0"
  "03120001ff7f0c40ff7f0c40ff7f0cc042100401ff7f0c40ff7f0c40ff7f0cc0ff7f0c4042101040ff7f0c40ff7f0cc082ff7f01ff7f0c40ff7f0c40ff7f0cc0"
  "00004c01ff7f0c40ff7f0c40ff7f0cc000804f01ff7f0c40ff7f0c40ff7f0cc0"
)

_BARRIERS: dict = {}

def barrier_group(raw) -> _dev.GroupSlot:
  if (lay := _BARRIERS.get(id(raw))) is None:
    lay = _BARRIERS[id(raw)] = Launch(raw, _BARRIER_TEXT, b"")
    lay.chain([1])
    if _hid.LEVEL: _hid.register_text(raw.dev_addr(lay.text), "barrier (do-nothing task)")
  a = raw.dev_addr
  return _dev.GroupSlot(spc=a(lay.text), cp=a(lay.cnst), tasks=[_dev.TaskSlot(sp=a(lay.stacks[0])+lay.stacks[0].nbytes, pp=a(lay.params[0]),
                                                                              dp=a(lay.privs[0]))])

def clone_member(prg):
  """A fresh runner for the same custom op: a chain member holds one launch state, so repeats in a graph get their own."""
  if hasattr(prg, "clone"): r = prg.clone()            # launch state of its own, the rest shared (no re-init)
  else:
    cls, cf, device = prg._clone_args
    r = cls(cf, device)
  r._clone_args = prg._clone_args
  return r
