"""ZHOUYI custom ops (outside the tinygrad backend): the hand-written TEC GEMMs (`gemm_gs`; `ZHOUYI_GM=1` runs them with the
partials in GM), the E4M3 weight stream (`e4m3_stream`) and hand-written C kernels (`register_csrc` / `csrc_call`), lowered from
`Ops.CUSTOM_FUNCTION` nodes by extending `engine.realize.si_lowerer` (a module-level PatternMatcher looked up by name, so no core
patch is needed). Each op is an ordinary job through the same `Launch` / `SCHEDULE_JOB` path as codegen kernels, and a graph chain
member. Importing this module installs the lowerings."""
from tinygrad.device import Buffer
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import Runner
from tinygrad.helpers import prod
from tinygrad.helpers import to_mv
from tinygrad.runtime.support.zhouyi import dev as _dev
from extra.zhouyi import dma
from extra.zhouyi import image as _image
from extra.zhouyi import timing as _timing
from tinygrad.runtime.support.zhouyi.launch import Launch
from tinygrad.runtime.support.zhouyi.dev import MM_REUSE
from tinygrad.uop.ops import Ops
from tinygrad.uop.ops import PatternMatcher
from tinygrad.uop.ops import UOp
from tinygrad.uop.ops import UPat
import ctypes
import os
import struct
import time
from tinygrad.runtime.ops_zhouyi import CHAIN_MEMBER_ARGS, ZHOUYI_CORES, ZHOUYI_TECS


_LOWERING_INSTALLED = False
# The runner cache. Runners are cached per (op key, shape, device): a Runner owns its `Launch`es and descriptor buffers, and the
# lowerer runs once per schedule item (per call); uncached, re-lowering per call leaks a Launch per op and exhausts device memory.
# (The name is historical; measurement tools clear it between runs.)
_AIFF_RUNNERS: dict = {}


def _cached_runner(cls, cf:UOp, device:str):
  # a runner reads some environment variables once, when built (`ENV_KEYS`): they are part of what it is, so they are part
  # of the key -- otherwise a later call under another value (a ZHOUYI_GEMM_TIMING diagnostic, a ZHOUYI_GEMM_DRAIN arm)
  # silently reuses the runner built under the first
  k = (cls.__name__, device, tuple(int(s.arg) for s in cf.src), tuple(os.environ.get(e) for e in getattr(cls, "ENV_KEYS", ())))
  r = _AIFF_RUNNERS.get(k)
  if r is None: r = _AIFF_RUNNERS[k] = cls(cf, device); r._clone_args = (cls, cf, device)
  return r


GEMM_GS_ARG = "zhouyi_gemm_gs"  # fp16 2D-blocked GEMM on the TEC matrix unit (kern_tpc.k_gemm_gs / k_gemm_gm); chain member
E4M3_STREAM_ARG = "zhouyi_e4m3_stream"  # E4M3 -> fp16 pass over a weight stream (kern_tpc.k_e4m3_stream); chain member


def _install_lowering():
  global _LOWERING_INSTALLED
  if _LOWERING_INSTALLED: return
  from tinygrad.engine import realize as _R
  _R.si_lowerer = _R.si_lowerer + PatternMatcher([
    (UPat(Ops.CUSTOM_FUNCTION, arg=GEMM_GS_ARG, name="cf"), lambda ctx, cf: _cached_runner(GemmGMRunner if os.environ.get("ZHOUYI_GM", "0") == "1" else GemmGSRunner, cf, ctx[0].device)),
    (UPat(Ops.CUSTOM_FUNCTION, arg=E4M3_STREAM_ARG, name="cf"), lambda ctx, cf: _cached_runner(E4M3StreamRunner, cf, ctx[0].device)),
    (UPat(Ops.CUSTOM_FUNCTION, arg=CSRC_ARG, name="cf"), lambda ctx, cf: _csrc_runner(cf, ctx[0].device)),
  ])
  _LOWERING_INSTALLED = True


# ------------------------------------------------------------------ hand-written C kernels ----
CSRC_ARG = "zhouyi_csrc"
_CSRC: dict[int, tuple[str, str, int]] = {}


def register_csrc(name:str, src:str, ntasks:int=ZHOUYI_CORES*ZHOUYI_TECS) -> int:
  """Register a hand-written TEC kernel in the renderer's C dialect; returns the key `csrc_call` takes.

  Signature: `__kernel void name(__global T* data0 /* output */, ..., const int core_id)`, with `core_id`
  the task index 0..ntasks-1. Compiled by the device compiler and run as an ordinary Program, so it chains
  in the JIT's fused jobs like any codegen kernel (e.g. the `float8` / `half16` kernels in `vec_f16.py`)."""
  key = len(_CSRC) + 1
  _CSRC[key] = (name, src, ntasks)
  return key


def csrc_call(key:int, out, *ins):
  """`out` (a `Tensor.empty` the kernel writes) after the kernel ran on `(out, *ins)`."""
  from tinygrad.tensor import Tensor
  ins = [i.contiguous() for i in ins]
  fn = UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(UOp.const(dtypes.int, key), UOp.const(dtypes.int, 1 + len(ins))), arg=CSRC_ARG)
  return Tensor(out.uop.after(fn.call(out.uop, *[i.uop for i in ins])))


def _csrc_runner(cf:UOp, device:str):
  from tinygrad.renderer import ProgramSpec
  from tinygrad.engine.realize import CompiledRunner
  key, nbufs = int(cf.src[0].arg), int(cf.src[1].arg)
  k = ("csrc", device, key)
  r = _AIFF_RUNNERS.get(k)
  if r is None:
    name, src, ntasks = _CSRC[key]
    spec = ProgramSpec(name, src, device, UOp(Ops.SINK, dtypes.void, ()), global_size=[ntasks, 1, 1], local_size=[1, 1, 1],
                       vars=[UOp.variable("core_id", 0, ntasks - 1)], globals=list(range(nbufs)), outs=[0], ins=list(range(1, nbufs)))
    r = _AIFF_RUNNERS[k] = CompiledRunner(spec)
    from tinygrad.runtime.support.zhouyi import hangid as _hid
    if _hid.LEVEL: _hid.note_csrc(r.p.lib, key)        # ZHOUYI_HANG_ID: a hung job's report names the csrc key (and the caller's tag)
  return r


class GemmGSRunner(Runner):
  """`C = A @ B^T` in fp16 with fp32 accumulation on the TEC matrix unit (`kern_tpc.k_gemm_gs`).

  Six 16-column strips' B panels are resident in LSRAM, 12-row A blocks stream through two halves, and C
  blocks live in the task's 64 KiB of GSRAM. Operand layouts (`gemm_fp16` / `vec_f16`): A = `pack_a_slices`
  (K = 4*ks*nslices, M = 12*nrb), B = `pack_b_group` panels (ngroups of ns strips, contiguous), C = the
  `[group][rb][s]` fp32 tiles. `heads` batches at (a / b / c) strides. Each (head, row piece, run of groups)
  is one task; tasks fill 13-task launches (12 + a barrier), all joining a fused chain as this member's
  groups. `rawbufs` = (c, a, b)."""
  chain_member = True
  chain_cycles = False
  GSRAM = 0xF8000000
  ENV_KEYS = ('ZHOUYI_GEMM_DRAIN', 'ZHOUYI_GEMM_TIMING')   # read in __init__: part of the runner cache key (_cached_runner)
  def __init__(self, cf:UOp, device:str):
    from extra.zhouyi import kern_tpc as _KT, gemm_fp16 as _G
    self.ks, self.ns, self.nrb, self.nslices, self.ngroups, self.heads, self.a_stride, self.b_stride, self.c_stride, self.a_off, self.b_off, self.c_off, piece, self.lin48, self.rowmax, self.b8, self.bdiv = (int(x.arg) for x in cf.src[:17])
    self.bscale = int(cf.src[17].arg) if len(cf.src) > 17 else 0
    self.rows = int(cf.src[18].arg) if len(cf.src) > 18 else 0     # decode: only row block 0's first `rows` rows are computed
    # bscale with GGUF Q8_0 (int8 codes, a scale per 32 weights): 1 = subnormal expand + per-slice correction and scale (ks 8),
    # 2 = the weights built in fp16 during the expand (ks a multiple of 8: no per-slice scaling)
    self.q8 = int(cf.src[19].arg) if len(cf.src) > 19 else 0
    assert not self.bscale or (self.b8 and not self.lin48 and not self.rowmax and (self.ks == 32 if not self.q8 else self.ks == 8 if self.q8 == 1 else self.ks % 8 == 0)), \
      "gemm_gs(bscale): b8, ks=32 (q8 1: 8, q8 2: a multiple of 8), no lin48/rowmax"
    # the E4M3 block-scale table's layout: what the pack says (`gemm_gs(scales=)`, src[20]: 0 dup / 1 single); the kernel is built
    # for it. The caller passes the layout its stream was packed in
    self.scales = _G.SCALE_LAYOUTS[int(cf.src[20].arg)] if len(cf.src) > 20 else "dup"
    assert self.scales == "dup" or (self.bscale and not self.q8), "gemm_gs(scales='single'): the E4M3 block-scale stream (bscale, not q8)"
    # lin48: the 48-deep, 3-strip kernel over the same 24-deep / 6-strip A and C layouts (faster at large K).
    # Each 6-strip group is two half-groups h, weight panels repacked `[h][G][K/192][3][48 kk]`
    # (`gemm_fp16.repack_panels_48x3`); a task = (head, half, row piece, run of groups).
    # Row piece per task: even, and piece x strips x 768 B of C must fit 64 KiB of GSRAM. Default is half the
    # row blocks, or for lin48 the largest even divisor up to 28 (28 x 3 x 768 = 64512 B); larger pieces cut B
    # bytes per unit.
    self.nrbh = piece if piece else (max(p_ for p_ in range(2, 29, 2) if self.nrb % p_ == 0) if self.lin48 else self.nrb // 2)
    assert self.nrb % self.nrbh == 0 and self.nrbh % 2 == 0 and self.nrbh * (3 if self.lin48 else self.ns) * 768 <= 65536, (self.nrb, self.nrbh, self.ns)
    self.npieces = self.nrb // self.nrbh
    assert not self.rows or (self.npieces == 1 and not self.lin48 and not self.rowmax and 1 <= self.rows <= 12), "gemm_gs(rows): one row piece, 1..12 rows"
    self.raw = Device[device].raw
    # rowmax: each row block's C row (ns x 768 B) is followed by its rows' maxima (192 B; `k_gemm_gs(rowmax=True)`)
    assert not (self.rowmax and self.lin48), "rowmax is the attention scores' kernel (24 x 6)"
    self.crow = self.ns * 768 + (192 if self.rowmax else 0)
    self.gstride = self.nrb * self.crow  # C group stride (bytes) over the full M
    if self.lin48:
      assert self.ks == 24 and self.ns == 6 and self.nslices % 2 == 0, "lin48 runs over the 24-deep / 6-strip layouts with K a multiple of 192"
      # b8: B is E4M3 codes in the 48 x 3 panel order (half the bytes, no fp16 panel pass); C comes out scaled by 2^-8 (its expand is k_gemm_gs's
      # zip / asr / and form, internal to the 48 x 3 kernel)
      self.text = _image.text_only(_KT.k_gemm_gs(48, 3, self.nrbh, c_stride=self.gstride, drain="dma", a_rb_stride=24 * 96, c_rb_stride=6 * 768, b8=bool(self.b8)))
      desc = _G.descriptors_gs48(self.nrb, bool(self.b8))
    else:
      # drain="dma": GSRAM touched only between K slices (faster); ZHOUYI_GEMM_DRAIN=tec selects the TEC drain
      drain = "dma" if self.rowmax else os.environ.get("ZHOUYI_GEMM_DRAIN", "dma")
      pf_ok = bool(self.rows) and -(-self.rows // 4) <= 2 and bool(self.b8) and bool(self.bscale) and not os.environ.get("ZHOUYI_GEMM_TIMING")   # the rows-mode prefetch path
      assert not self.b8 or drain == "dma"
      self.text = _image.text_only(_KT.k_gemm_gs(self.ks, self.ns, self.nrbh, c_stride=self.gstride, drain=drain, rowmax=bool(self.rowmax), b8=bool(self.b8), kw=4 if self.bscale else 3, bscale=bool(self.bscale),
                                                   rows=self.rows or None, timing=(os.environ.get("ZHOUYI_GEMM_TIMING") or None) if self.rows else None,
                                                   q8=self.q8 == 1, q8f=self.q8 == 2, scales=self.scales))                                    # the scale table's layout, the pack's
      self.a_slice = self.ks * 32 * -(-self.rows // 4) if self.rows else self.nrb * self.ks * 96   # A bytes a K-slice (rows: compact)
      desc = _G.descriptors_gs(self.ks, self.ns, bool(self.rowmax), bool(self.b8), bool(self.bscale), a_bytes=self.ks * 32 * -(-self.rows // 4) if self.rows else None,
                               prefetch=pf_ok, q8=self.q8, scales=self.scales)
    self.desc = self.raw.req_buf(len(desc), MM_REUSE); ctypes.memmove(self.desc.va, desc, len(desc))
    # Plan: (head, piece, g0, ng) per task. Groups are split into r runs, with r chosen so tasks fill 12-task
    # launches with the least idle work (each launch waits for its slowest task).
    halves = 2 if self.lin48 else 1
    hp = self.heads * self.npieces * halves; best = None
    for r in range(1, self.ngroups + 1):
      runs = [self.ngroups // r + (1 if i < self.ngroups % r else 0) for i in range(r)]
      ntask = hp * r; nl = -(-ntask // 12)
      cost = nl * (max(runs) + 2)  # launches x (longest run + ~2 groups of per-launch overhead: C zeroing, barrier, setup)
      if best is None or cost < best[0]: best = (cost, runs)
    runs = best[1]
    self.tasks = []
    for h in range(self.heads):
      for hf in range(halves):
        for pc in range(self.npieces):
          g = 0
          for n in runs: self.tasks.append((h, pc, g, n, hf)); g += n
    self.nl = -(-len(self.tasks) // 12)
    self.launches = []
    for _ in range(self.nl):
      lay = Launch(self.raw, self.text, b""); lay.chain([13], group_widths=[4, 4, 4, 1], dep=[0, 0, 0, _dev.DEP_PRE_ALL]); self.launches.append(lay)
    self._chain_gen = 0
    self._staged_key = None
    super().__init__(f"TEC gemm_gs{'+rowmax' if self.rowmax else ''}{'+b8' if self.b8 else ''}{'+bscale' if self.bscale else ''}{'+sc1' if self.scales == 'single' else ''}{'+q8' if self.q8 else ''}{f'+rows{self.rows}' if self.rows else ''} M={12*self.nrb} K={4*self.ks*self.nslices} N={16*self.ns*self.ngroups} x{self.heads} ({len(self.tasks)} tasks, {self.nl} launches)", device)
  def group_slots(self) -> list:
    a = self.raw.dev_addr; out = []
    for lay in self.launches:
      slots = [_dev.TaskSlot(sp=a(s)+s.nbytes, pp=a(p), dp=a(d)) for s, p, d in zip(lay.stacks, lay.params, lay.privs)]
      out += [_dev.GroupSlot(spc=a(lay.text), cp=a(lay.cnst), tasks=slots[4*c:4*c+4]) for c in range(3)] +              [_dev.GroupSlot(spc=a(lay.text), cp=a(lay.cnst), tasks=slots[12:13])]
    return out
  def chain_deps(self) -> list: return [_dev.DEP_PRE_ALL, 0, 0, _dev.DEP_PRE_ALL] * self.nl
  def stage_chain(self, rawbufs:list[Buffer]) -> None: self._prepare(rawbufs)
  def _prepare(self, rawbufs:list[Buffer]):
    key = tuple(self.raw.dev_addr(b._buf) for b in rawbufs[:3])
    if self._staged_key == key: return
    c, a, b = key; a += self.a_off; b += self.b_off; c += self.c_off; d = self.raw.dev_addr(self.desc)
    from extra.zhouyi import gemm_fp16 as _gfp
    bpk = 64 if self.b8 else 128  # B bytes per strip k-step (E4M3 codes / fp16)
    gb = self.nslices * (self.ns * self.ks * bpk + (_gfp.gs_scale_bytes(self.ns, self.ks, self.q8, self.scales) if self.bscale else 0))  # bytes per B group (+ the scale tables)
    for li, lay in enumerate(self.launches):
      for t in range(13):
        i = li * 12 + t
        if t < 12 and i < len(self.tasks):
          h, pc, g0, ng, hf = self.tasks[i]
          if self.lin48:  # gb48 = a half-group's panels (K/192 slices x 3 strips x 48 kk x bpk B); A row blocks 2304 B apart (24-deep layout)
            gb48 = (self.nslices // 2) * 3 * 48 * bpk
            args = (a + h * self.a_stride + pc * self.nrbh * 2304, b + (h // self.bdiv) * self.b_stride + (hf * self.ngroups + g0) * gb48,
                    c + h * self.c_stride + hf * 2304 + g0 * self.gstride + pc * self.nrbh * self.ns * 768, d, self.nslices // 2, self.nrbh, self.nrb * 48 * 96,
                    self.GSRAM + 65536 * (t % 4), ng)
          else:
            args = (a + h * self.a_stride + pc * self.nrbh * self.ks * 96, b + (h // self.bdiv) * self.b_stride + g0 * gb,
                    c + h * self.c_stride + g0 * self.gstride + pc * self.nrbh * self.crow, d, self.nslices, self.nrbh, self.a_slice,
                    self.GSRAM + 65536 * (t % 4), ng)
        else: args = (a, b, c, d, 0, self.nrbh, self.nrb * self.ks * 96, self.GSRAM, 0)  # NSLICES = 0: the task exits (barrier and spare tasks)
        ctypes.memmove(lay.params[t].va, struct.pack("<9I", *args), 36)
    self._staged_key = key
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False) -> float|None:
    self._prepare(rawbufs)
    st = time.perf_counter()
    for lay in self.launches: _timing.submit_and_wait(self.raw, lay.chains)
    return time.perf_counter() - st if wait else None


GM_WINDOW, GM_A_CAP, GM_C_BASE, GM_C_REG = 4 << 20, 3276800, 3276800, 73728  # the window; A chunks below GM_A_CAP; 12 C regions above


def gm_window(raw):
  """The device's GM window: a 4 MiB-aligned 4 MiB of IOVA, allocated once and never used as DDR.

  Returns (its PA, its ASID0 offset)."""
  w = getattr(raw, "_gm_win", None)
  if w is None:
    buf = raw.req_buf(2 * GM_WINDOW, MM_REUSE)
    pa = (buf.pa + GM_WINDOW - 1) // GM_WINDOW * GM_WINDOW
    w = raw._gm_win = (buf, pa, raw.dev_addr(buf) + (pa - buf.pa))
  return w[1], w[2]


class GemmGMRunner(Runner):
  """`gemm_gs` with the partials (and the linears' A) in GM (`ZHOUYI_GM=1`; `kern_tpc.k_gemm_gm`).

  The job runs with the grid TCB's GM remap on. Linears (lin48, E4M3 B) run per row chunk of `nrbc` row
  blocks: a `k_gm_stage` launch copies the chunk's A into GM, then a `k_gemm_gm` launch whose 12 tasks split
  the columns (2 halves x 6 runs of groups). Attention GEMMs keep `GemmGSRunner`'s plan and read A from DDR.
  Every launch is [barrier][4][4][4][barrier] (14 tasks) and starts after everything before it, so chunks,
  stage/GEMM pairs and the per-task C regions (GM_C_BASE + task x GM_C_REG) never overlap. `rawbufs` = (c, a, b)."""
  chain_member = True
  chain_cycles = False
  uses_gm = True
  def __init__(self, cf:UOp, device:str):
    from extra.zhouyi import kern_tpc as _KT, gemm_fp16 as _G
    self.ks, self.ns, self.nrb, self.nslices, self.ngroups, self.heads, self.a_stride, self.b_stride, self.c_stride, self.a_off, self.b_off, self.c_off, piece, self.lin48, self.rowmax, self.b8, self.bdiv = (int(x.arg) for x in cf.src[:17])
    self.raw = Device[device].raw
    self.gm_pa, self.gm0 = gm_window(self.raw)
    self.staged = bool(self.lin48 and self.b8)
    T = lambda k: _image.text_only(k)
    mk = lambda buf_bytes: self.raw.req_buf(len(buf_bytes), MM_REUSE)
    self.plan = []  # (text, [args of the 12 work tasks]) per launch, filled by _prepare
    if self.staged:
      assert self.heads == 1 and self.ks == 24 and self.ns == 6 and self.nslices % 2 == 0
      self.s48 = self.nslices // 2
      self.nrbc = max(p_ for p_ in range(2, 29, 2) if self.nrb % p_ == 0 and p_ * self.s48 * 4608 <= GM_A_CAP)
      self.gstride = self.nrb * 6 * 768
      # k_gm_stage copies a chunk's A through LSRAM into GM ([slice48][rb][4608 B])
      self.texts = {"stage": T(_KT.k_gm_stage()), "gemm": T(_KT.k_gemm_gm(self.nrbc, c_stride=self.gstride, c_rb_stride=4608, b8=True))}
      descs = {"gemm": _G.descriptors_gm(48, 3, True), "stage": _G.descriptors_gm_stage(self.nrb)}
      G = self.ngroups; nr = min(6, G)
      runs = [G // nr + (1 if i < G % nr else 0) for i in range(nr)]
      self.cols, g = [], 0
      for n in runs: self.cols.append((g, n)); g += n
      nl = 2 * (self.nrb // self.nrbc)
      what = f"M={12*self.nrb} K={96*self.nslices} N={96*G} staged in {self.nrb // self.nrbc} chunks of {12*self.nrbc} rows"
    else:
      self.nrbh = piece if piece else (max(p_ for p_ in range(2, 29 if self.lin48 else 15, 2) if self.nrb % p_ == 0) if self.lin48 else self.nrb // 2)
      self.crow = (3 if self.lin48 else self.ns) * 768 + (192 if self.rowmax else 0)
      assert self.nrb % self.nrbh == 0 and self.nrbh % 2 == 0 and self.nrbh * self.crow <= GM_C_REG, (self.nrb, self.nrbh)
      self.npieces = self.nrb // self.nrbh
      if self.lin48:
        self.gstride = self.nrb * self.ns * 768
        self.texts = {"gemm": T(_KT.k_gemm_gm(self.nrbh, 48, 3, c_stride=self.gstride, c_rb_stride=6 * 768, b8=False, a_rb_stride=2304))}
        descs = {"gemm": _G.descriptors_gm(48, 3, False, a_gather_nrb=self.nrb)}
      else:
        self.gstride = self.nrb * self.crow
        self.texts = {"gemm": T(_KT.k_gemm_gm(self.nrbh, self.ks, self.ns, c_stride=self.gstride, c_rb_stride=self.crow, b8=False, rowmax=bool(self.rowmax)))}
        descs = {"gemm": _G.descriptors_gm(self.ks, self.ns, False, rowmax=bool(self.rowmax))}
      halves = 2 if self.lin48 else 1
      hp = self.heads * self.npieces * halves; best = None
      for r in range(1, self.ngroups + 1):
        runs = [self.ngroups // r + (1 if i < self.ngroups % r else 0) for i in range(r)]
        cost = -(-(hp * r) // 12) * (max(runs) + 2)
        if best is None or cost < best[0]: best = (cost, runs)
      self.tasks = []
      for h in range(self.heads):
        for hf in range(halves):
          for pc in range(self.npieces):
            g = 0
            for n in best[1]: self.tasks.append((h, pc, g, n, hf)); g += n
      nl = -(-len(self.tasks) // 12)
      what = f"M={12*self.nrb} K={4*self.ks*self.nslices} N={16*self.ns*self.ngroups} x{self.heads} ({len(self.tasks)} tasks)"
    self.descs = {}
    for k, d in descs.items():
      self.descs[k] = mk(d); ctypes.memmove(self.descs[k].va, d, len(d))
    self.launches = []
    for i in range(nl):
      name = ("stage" if i % 2 == 0 else "gemm") if self.staged else "gemm"
      lay = Launch(self.raw, self.texts[name], b""); lay.chain([14], group_widths=[1, 4, 4, 4, 1], dep=[0, 0, 0, 0, _dev.DEP_PRE_ALL])
      _dev.gm_fields(to_mv(lay.tcb_bufs[0].va, 128), self.gm_pa)
      self.launches.append((name, lay))
    self._chain_gen = 0
    self._staged_key = None
    super().__init__(f"TEC gemm_gm{'+rowmax' if self.rowmax else ''} {what}, {nl} launches", device)
  def group_slots(self) -> list:
    a = self.raw.dev_addr; out = []
    for _, lay in self.launches:
      sl = [_dev.TaskSlot(sp=a(s)+s.nbytes, pp=a(p), dp=a(d)) for s, p, d in zip(lay.stacks, lay.params, lay.privs)]
      out += [_dev.GroupSlot(spc=a(lay.text), cp=a(lay.cnst), tasks=sl[lo:hi]) for lo, hi in ((0, 1), (1, 5), (5, 9), (9, 13), (13, 14))]
    return out
  def chain_deps(self) -> list: return [_dev.DEP_PRE_ALL, 0, 0, 0, _dev.DEP_PRE_ALL] * len(self.launches)
  def stage_chain(self, rawbufs:list[Buffer]) -> None: self._prepare(rawbufs)
  def _prepare(self, rawbufs:list[Buffer]):
    key = tuple(self.raw.dev_addr(b._buf) for b in rawbufs[:3])
    if self._staged_key == key: return
    c, a, b = key; a += self.a_off; b += self.b_off; c += self.c_off
    dg = self.raw.dev_addr(self.descs["gemm"]); A_GM, C_GM = self.gm0, self.gm0 + GM_C_BASE
    work = []
    if self.staged:
      ds = self.raw.dev_addr(self.descs["stage"]); gb48 = self.s48 * 3 * 48 * 64
      for ch in range(self.nrb // self.nrbc):
        rb0 = ch * self.nrbc
        st = []
        for t in range(12):
          n = len(range(t, self.s48, 12))
          st.append((a + rb0 * 2304 + t * 2 * self.nrb * 2304, A_GM + t * self.nrbc * 4608, ds, n, self.nrbc, 12 * 2 * self.nrb * 2304, 12 * self.nrbc * 4608))
        gm = []
        for t in range(12):
          hf, r = divmod(t, 6)
          if r >= len(self.cols): gm.append(None); continue
          g0, n = self.cols[r]
          gm.append((A_GM, b + (hf * self.ngroups + g0) * gb48, c + hf * 2304 + g0 * self.gstride + rb0 * 4608, dg, self.s48, self.nrbc, self.nrbc * 4608, C_GM + t * GM_C_REG, n))
        work += [st, gm]
    else:
      gbk = self.nslices * self.ns * self.ks * 128
      for li in range(len(self.launches)):
        gm = []
        for t in range(12):
          i = li * 12 + t
          if i >= len(self.tasks): gm.append(None); continue
          h, pc, g0, n, hf = self.tasks[i]
          if self.lin48:
            gb48 = (self.nslices // 2) * 3 * 48 * 128
            gm.append((a + h * self.a_stride + pc * self.nrbh * 2304, b + (h // self.bdiv) * self.b_stride + (hf * self.ngroups + g0) * gb48,
                       c + h * self.c_stride + hf * 2304 + g0 * self.gstride + pc * self.nrbh * self.ns * 768, dg, self.nslices // 2, self.nrbh,
                       self.nrb * 48 * 96, C_GM + t * GM_C_REG, n))
          else:
            gm.append((a + h * self.a_stride + pc * self.nrbh * self.ks * 96, b + (h // self.bdiv) * self.b_stride + g0 * gbk,
                       c + h * self.c_stride + g0 * self.gstride + pc * self.nrbh * self.crow, dg, self.nslices, self.nrbh,
                       self.nrb * self.ks * 96, C_GM + t * GM_C_REG, n))
        work.append(gm)
    for (name, lay), args in zip(self.launches, work):
      idle = (0,) * 9  # NSLICES / the stage's slice count = 0: the task exits
      for t in range(14):
        x = args[t - 1] if 1 <= t <= 12 and args[t - 1] is not None else idle
        ctypes.memmove(lay.params[t].va, struct.pack("<%dI" % len(x), *x), 4 * len(x))
    self._staged_key = key
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False) -> float|None:
    self._prepare(rawbufs)
    st = time.perf_counter()
    for _, lay in self.launches: _timing.submit_and_wait(self.raw, lay.chains)
    return time.perf_counter() - st if wait else None


class E4M3StreamRunner(Runner):
  """E4M3 bytes -> fp16 halves, in order, exactly, split over 12 tasks. `rawbufs` = (out, src).

  `kern_tpc.k_e4m3_stream`: 8 KiB chunks DMA'd into LSRAM, closed-form unpack, 16 KiB DMA'd out (DDR-bound).
  `src` must be a multiple of 8 KiB."""
  chain_member = True
  chain_cycles = False
  def __init__(self, cf:UOp, device:str):
    import numpy as np
    from extra.zhouyi import kern_tpc as _KT
    self.nbytes = int(cf.src[0].arg); ch = _KT.E4M3_STREAM_CHUNK
    assert self.nbytes % ch == 0, "e4m3_stream: %d B is not a multiple of the %d B chunk" % (self.nbytes, ch)
    self.nchunks = self.nbytes // ch; self.ch = ch
    self.raw = Device[device].raw
    desc = bytes(np.asarray(dma.desc_words(ch), np.uint32).tobytes()) + bytes(8) + bytes(np.asarray(dma.desc_words(2 * ch), np.uint32).tobytes())
    self.desc = self.raw.req_buf(64, MM_REUSE); ctypes.memmove(self.desc.va, desc, len(desc))
    self.launch = Launch(self.raw, _image.text_only(_KT.k_e4m3_stream()), b"")
    self.launch.chain([13], group_widths=[4, 4, 4, 1], dep=[0, 0, 0, _dev.DEP_PRE_ALL])
    per, rem = divmod(self.nchunks, 12)
    self.plan = []; c0 = 0
    for t in range(12): n = per + (1 if t < rem else 0); self.plan.append((c0, n)); c0 += n
    self._chain_gen = 0
    self._staged_key = None
    super().__init__(f"TEC e4m3 -> fp16 {self.nbytes >> 20} MiB", device)
  def group_slots(self) -> list:
    a, lay = self.raw.dev_addr, self.launch
    slots = [_dev.TaskSlot(sp=a(s)+s.nbytes, pp=a(p), dp=a(d)) for s, p, d in zip(lay.stacks, lay.params, lay.privs)]
    return [_dev.GroupSlot(spc=a(lay.text), cp=a(lay.cnst), tasks=slots[4*c:4*c+4]) for c in range(3)] +            [_dev.GroupSlot(spc=a(lay.text), cp=a(lay.cnst), tasks=slots[12:13])]
  def chain_deps(self) -> list: return [_dev.DEP_PRE_ALL, 0, 0, _dev.DEP_PRE_ALL]
  def stage_chain(self, rawbufs:list[Buffer]) -> None: self._prepare(rawbufs)
  def _prepare(self, rawbufs:list[Buffer]):
    key = tuple(self.raw.dev_addr(b._buf) for b in rawbufs[:2])
    if self._staged_key == key: return
    out, src = key; d = self.raw.dev_addr(self.desc)
    for t in range(13):
      c0, n = self.plan[t] if t < 12 else (0, 0)
      ctypes.memmove(self.launch.params[t].va, struct.pack("<4I", src + c0 * self.ch, d, out + 2 * c0 * self.ch, n), 16)
    self._staged_key = key
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False) -> float|None:
    self._prepare(rawbufs)
    st = time.perf_counter()
    _timing.submit_and_wait(self.raw, self.launch.chains)
    return time.perf_counter() - st if wait else None


def host_invalidate(t):
  """Invalidate the CPU caches' stale clean lines for `t`'s pages (`RawDevice.cache_invalidate`) and return `t`.

  Call after the job that wrote `t` completes and before the host reads it; without it host reads of
  device-written data can be stale, even on repeated reads. Realizes `t` first. Skips tensors with no device
  buffer (e.g. a fused view, which reaches the host through its realized source). For a view the driver
  invalidates the whole owning block, which is harmless. Raises if the buffer belongs to another fd."""
  t.realize()  # ensure the buffer exists and its producing job has run
  try: b = t.uop.buffer._buf  # `.buffer` asserts on a uop that is not one (a fused view)
  except (AssertionError, AttributeError, RuntimeError): b = None  # not a BUFFER, or a non-contiguous view
  if b is None or not hasattr(b, "dev_offset"):
    # Not silent: a skipped read carries no guarantee. `host_invalidate.skipped` counts skips, the first one
    # warns, and callers gating on exactness can assert the count is zero.
    host_invalidate.skipped += 1
    if host_invalidate.skipped == 1:
      import warnings; warnings.warn("host_invalidate: no device buffer resolved; this read is NOT verified "
                                     "Further skips are counted in host_invalidate.skipped.", RuntimeWarning)
    return t
  Device[t.device].raw.cache_invalidate(b)
  return t


host_invalidate.skipped = 0


def e4m3_stream(src, out=None):
  """`src` (uint8 E4M3 codes, a multiple of 8 KiB) -> the same values as fp16 (uint16 halves), in order, exactly.

  NaN codes become fp16 NaN. Weights pre-packed in `gemm_fp16.pack_b_group_e4m3`'s panel order come out as
  `pack_b_group`'s fp16 panels, ready for `gemm_gs`."""
  from tinygrad.tensor import Tensor
  src = src.contiguous(); n = int(prod(src.shape))
  if out is None: out = Tensor.empty(n, device=src.device, dtype=dtypes.uint16)
  fn = UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(UOp.const(dtypes.int, n),), arg=E4M3_STREAM_ARG)
  return Tensor(out.uop.after(fn.call(out.uop, src.uop)))


def gemm_gs(a, b, *, ks:int, ns:int, nrb:int, nslices:int, ngroups:int, heads:int=1, a_stride:int=0, b_stride:int=0, c_stride:int=0, a_off:int=0, b_off:int=0, c_off:int=0, piece:int=0, lin48:bool=False, rowmax:bool=False, b8:bool=False, b_head_div:int=1, bscale:bool=False, rows:int=0, q8:int=0, scales:str="dup", out=None):
  """`C = A @ B^T` on the TEC matrix unit (`GemmGSRunner`); returns the fp32 C tiles.

    a: A layout (uint16 halves, `vec_f16` / `gemm_fp16.pack_a_slices`: 12*nrb rows x 4*ks*nslices).
    b: B panels (uint16, `ngroups` groups of `ns` strips).
    out: fp32 tiles `[heads][group][rb][s][192]` (`heads * ngroups * nrb * ns * 192` floats).
    heads: batch count; strides are in elements of each buffer (halves, halves, floats).
    a_off/b_off (halves), c_off (floats): operand starts inside their buffers (avoids a slicing copy).
    piece: row blocks per task (0: half of nrb).
    rowmax: each row block's C row (`[s][192]`) is followed by its 12 rows' lane-wise maxima over the group
      (48 floats, `[i][h][8]`: row 4i + 2h + lane // 4); a group's C block is `[rb][ns * 192 + 48]`.
    b8: `b` holds E4M3 codes (uint8) in panel order (`pack_b_group_e4m3`, or its 48 x 3 repack with lin48),
      expanded in LSRAM; C comes out scaled by 2^-8 exactly. `b_stride`/`b_off` then count bytes.
    b_head_div: head h reads B at `(h // b_head_div) * b_stride` (grouped-query attention).
    bscale (b8, ks=32): `b` is `pack_b_group_bscale`'s stream (each K-slice's codes followed by its per-column scales);
      C comes out fully scaled.
    q8 (bscale, ks=8): `b` is `pack_b_group_q8`'s stream (int8 codes, a scale per 32 weights: GGUF Q8_0).
    q8=2 (bscale, ks a multiple of 8): `pack_b_group_q8f`'s stream (the same weights, built in fp16 in the kernel: no per-slice scaling).
    scales (bscale, E4M3): the scale table's layout the stream was packed in -- "dup" (`pack_b_group_bscale`'s default: each
      scale twice, 128 B a strip and slice) or "single" (`scales="single"`: once, 64 B; the kernel rebuilds the lanes, C
      bit-identical). The pack's marker decides (the caller passes it here).
    rows (1..12, one row piece): only the first `rows` rows are computed (decode). A is then the compact layout
      [slice][k][rt tiles][32 B] (rt = ceil(rows / 4): the decode producers write it; no padding crosses DDR); C keeps
      its layout, the other rows zeros in row block 0 and untouched elsewhere."""
  from tinygrad.tensor import Tensor
  a = a.contiguous(); b = b.contiguous()
  if out is None: out = Tensor.empty(heads * ngroups * nrb * (ns * 192 + (48 if rowmax else 0)), device=a.device, dtype=dtypes.float32)
  assert scales in ("dup", "single"), scales
  fn = UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=tuple(UOp.const(dtypes.int, int(v)) for v in (ks, ns, nrb, nslices, ngroups, heads, 2 * a_stride, (1 if b8 else 2) * b_stride, 4 * c_stride, 2 * a_off, (1 if b8 else 2) * b_off, 4 * c_off, piece, int(lin48), int(rowmax), int(b8), int(b_head_div), int(bscale), int(rows), int(q8), int(scales == "single"))), arg=GEMM_GS_ARG)
  return Tensor(out.uop.after(fn.call(out.uop, a.uop, b.uop)))


CHAIN_MEMBER_ARGS.update({CSRC_ARG, GEMM_GS_ARG, E4M3_STREAM_ARG})
_install_lowering()
