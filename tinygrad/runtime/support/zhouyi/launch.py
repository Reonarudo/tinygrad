"""Allocation shape of one ZHOUYI launch: the buffers a dispatch needs and the TCB chain(s) over them.

A launch is one text buffer, one constant pool, a stack, param and priv buffer per task, and one TCB
chain per job. Used by `ops_zhouyi.ZhouyiProgram` and the custom-op runners, and usable by a simulator harness: it
needs no tinygrad device.

Not here (tinygrad-side policy in `ops_zhouyi.py`): the job split (`_job_split`, `ZHOUYI_TECS`), the
param memo and the chain/param generations.
"""
from __future__ import annotations

import ctypes, struct
from typing import Sequence

from . import ZhouyiError
from . import dev as _dev
from .dev import TaskSlot, build_tcbs, chain_tcbs
from .dev import TCB_LEN, MM_REUSE, MM_RODATA, MM_STACK, MM_STATIC, MM_TCB, MM_TEXT

# Compiled blob: length-prefixed (text, rodata) so the constant pool travels with the text in the single
# bytes object the Compiler returns and the disk cache stores. Both constants feed
# `compiler_zhouyi._cache_key`, so changing them invalidates cached blobs.
BLOB_HEADER = "<II"          # (len(text), len(rodata)), then text, then rodata
BLOB_VERSION = 3             # bump on ANY change to the layout above

# Per-task buffer sizes. Stack and private data must not be shared between tasks (silent corruption);
# the param buffer (`pp`) is per task too since it carries the task index (see `dev.TaskSlot`).
STACK_BYTES, PARAM_BYTES, PRIV_BYTES = 8192, 4096, 4096

# Per-task cycle stamps. `ctrl0[0xd1]` is the TEC's free-running 32-bit cycle counter. The image reads it
# after the FP enable and before each task epilogue (`image.cycle_stamp`) and stores both values in the
# last two words of the task's param buffer (`pp` is recoverable from `tcbp+56`); `launch_args` keeps
# kernel args out of these words.
# Stamps are poisoned before every submit; poison after `wait_jobs` means the image did not stamp.
# Deltas are `(end - start) & 0xFFFFFFFF` (the counter wraps every few seconds).
CYCLE_STAMP_BYTES = 8
CYCLE_STAMP_OFF = PARAM_BYTES - CYCLE_STAMP_BYTES       # 4088: start at +0, end at +4
CYCLE_STAMP_POISON = 0xFFFFFFFF

def stack_frame_bytes(text: bytes) -> int:
  """Bytes below the TCB sp the image's kernel can use: every `sub sp, sp, imm` and `sub sp, sp, rN` (rN from the last
  `mov`/`movh rN` before it) summed, + 31 for the prologue's `andh sp, sp, 31` (0: no frame). btaipucc's prologue, e.g.
  `sub sp,sp,48` (args past r3), `sub sp,sp,8` (fp, r4), `mov r20,1048; sub sp,sp,r20`, `andh sp,sp,31`; the epilogues give it
  back with `add sp` / `sub sp, fp, k`, which are not counted. Encodings from extra/zhouyi/enc.py (bit 31 = bundle end)."""
  regs, below = {}, 0
  for (w,) in struct.iter_unpack("<I", text[:len(text) // 4 * 4]):
    w &= 0x7FFFFFFF
    if w & 0x7FE00000 == 0x00C00000: regs[w & 31] = ((w >> 5) & 0xFFFF) - (0x10000 if w & (1 << 20) else 0)      # mov rd, simm16
    elif w & 0x7FE00000 == 0x00E00000: regs[w & 31] = (regs.get(w & 31, 0) & 0xFFFF) | (((w >> 5) & 0xFFFF) << 16)  # movh rd, imm16
    elif w & 0x7FF003FF == 0x405003BD: below += (w >> 10) & 0x3FF                                                  # sub sp, sp, imm
    elif w & 0x7FFF83FF == 0x400083BD: below += regs.get((w >> 10) & 31, 1 << 20)                                  # sub sp, sp, rN
  return below + 31 if below else 0


def unpack_blob(lib: bytes) -> tuple[bytes, bytes]:
  """`(text, rodata)` out of a compiled blob. The one reader of `BLOB_HEADER`'s layout."""
  ntext, nrodata = struct.unpack_from(BLOB_HEADER, lib, 0)
  hdr = struct.calcsize(BLOB_HEADER)
  return lib[hdr:hdr + ntext], lib[hdr + ntext:hdr + ntext + nrodata]


class Launch:
  """Every buffer one dispatch of one kernel needs, and the chain(s) that dispatch it.

  Built in two phases: the image (text + constant pool) lives as long as the program; the per-task
  buffers and TCBs depend on the task count and can be rebuilt with `chain`.

      lay = Launch(raw, *unpack_blob(lib))
      ...                                    # the caller's own allocations, if any
      lay.chain(job_split)

  `raw` is any device with `req_buf`/`free_buf`/`dev_addr`: `dev.RawDevice` on hardware, or a simulator harness's
  stand-in (which replaces the kernel driver, so it does not exercise it).
  """

  def __init__(self, raw, text: bytes, rodata: bytes):
    self.raw = raw
    self.text = raw.req_buf(len(text), MM_TEXT)
    ctypes.memmove(self.text.va, text, len(text))
    # `cp`, the constant pool. At least one page: every TCB carries a `cp`, even with no rodata.
    self.cnst = raw.req_buf(max(4096, len(rodata)), MM_STATIC)
    if rodata: ctypes.memmove(self.cnst.va, rodata, len(rodata))
    self.ntasks, self.ngroups, self.param_bytes = 0, 1, PARAM_BYTES
    self.widths: list[int] = []
    self.stacks: list = []
    self.params: list = []
    self.privs: list = []
    self.tcb_bufs: list = []
    self.chains: list = []
    self.owned: list = []

  def task_slots(self) -> list[TaskSlot]:
    """Every task's (sp, pp, dp), task order."""
    a = self.raw.dev_addr
    return [TaskSlot(sp=a(s) + s.nbytes, pp=a(p), dp=a(d))
            for s, p, d in zip(self.stacks, self.params, self.privs)]

  def chain(self, widths: Sequence[int], *, ngroups: int = 1, stack_bytes: int = STACK_BYTES,
            param_bytes: int = PARAM_BYTES, priv_bytes: int = PRIV_BYTES,
            dep=_dev.DEP_PRE_ALL, grid_end: bool = False, group_widths=None) -> "Launch":
    """Allocate the per-task buffers and build one TCB chain per entry of `widths`.

    `widths`: tasks per job, e.g. `[4]` for one core, `[4,4,4]` for three. Each job is a separate
    `SCHEDULE_JOB` with its own head, SegMMU pair and task TCBs; stacks/params/privs are one per task.

    `ngroups` (groups per job) is not `len(widths)`; the default 1 is the shipping configuration. With
    several groups, dep mode 0 can report DONE with work unfinished and non-zero modes serialise the
    groups, so extra cores are used via concurrent jobs instead; `ngroups > 1` is for experiments.

    Re-callable: previous per-task buffers are freed first. Note: callers must invalidate anything
    derived from them (captured addresses, param memo keys) before calling; a TCB pointing at freed
    memory produces wrong results, not a fault.
    """
    if not widths or any(n <= 0 for n in widths):
      raise ZhouyiError(f"a dispatch of {list(widths)} job width(s) has nothing to run")
    ntasks = sum(widths)
    if group_widths is None and ntasks % ngroups:
      raise ZhouyiError(f"{ntasks} task(s) do not divide into {ngroups} group(s)")
    for b in self.owned: self.raw.free_buf(b)
    a = self.raw.dev_addr
    self.stacks = [self.raw.req_buf(stack_bytes, MM_STACK) for _ in range(ntasks)]
    self.params = [self.raw.req_buf(param_bytes, MM_RODATA) for _ in range(ntasks)]
    self.privs = [self.raw.req_buf(priv_bytes, MM_REUSE) for _ in range(ntasks)]
    self.owned = [*self.stacks, *self.params, *self.privs]
    self._poison_stamps()
    self.chains, self.tcb_bufs, base = [], [], 0
    slots = self.task_slots()
    for n in widths:
      self.tcb_bufs.append(tcbs := self.raw.req_buf(chain_tcbs(n) * TCB_LEN, MM_TCB))
      build_tcbs(self.raw, tcbs, spc=a(self.text), cp=a(self.cnst), ngroups=ngroups, dep=dep, grid_end=grid_end, group_widths=group_widths,
                 tasks=slots[base:base + n])
      self.chains.append((tcbs.pa, tcbs.pa + _dev.TASK_INDEX * TCB_LEN,
                          tcbs.pa + (_dev.TASK_INDEX + n - 1) * TCB_LEN))
      self.owned.append(tcbs)
      base += n
    self.ntasks, self.widths, self.ngroups, self.param_bytes = ntasks, list(widths), ngroups, param_bytes
    return self

  def _stamp_words(self, p) -> "ctypes.Array":
    return (ctypes.c_uint32 * 2).from_address(p.va + CYCLE_STAMP_OFF)

  def _poison_stamps(self) -> None:
    for p in self.params:
      w = self._stamp_words(p)
      w[0] = w[1] = CYCLE_STAMP_POISON

  def read_cycles(self, poison: bool = True) -> list[tuple[int, int]]:
    """`(start, end)` cycle stamps of every task of the last launch, read from the param tails.

    Call only after `wait_jobs` (the epilogue's d-cache writeback lands them in DDR). A word still at
    `CYCLE_STAMP_POISON` means the image did not stamp. `poison=True` re-arms the tails for the next launch."""
    out = []
    for p in self.params:
      w = self._stamp_words(p)
      out.append((int(w[0]), int(w[1])))
      if poison: w[0] = w[1] = CYCLE_STAMP_POISON
    return out

  # Accessors for the first chain, for single-chain callers.
  @property
  def tcbs(self): return self.tcb_bufs[0]

  @property
  def head_pa(self) -> int: return self.chains[0][0]

  @property
  def task_pa(self) -> int: return self.chains[0][1]

  @property
  def last_task_pa(self) -> int: return self.chains[0][2]

  def named_buffers(self) -> list[tuple[str, object]]:
    """Every buffer this launch owns, named by role (e.g. for the simulator's per-region access reports).

    The kernel's own data buffers belong to the caller and are not included.
    """
    out = [("text", self.text), ("const", self.cnst)]
    for kind, bufs in (("stack", self.stacks), ("param", self.params), ("priv", self.privs)):
      out += [(f"{kind}{t}", b) for t, b in enumerate(bufs)]
    return out + [(f"tcb{j}", b) for j, b in enumerate(self.tcb_bufs)]


def launch_args(raw, bufs, vals, ntasks: int, runtimevars: dict) -> tuple[list[int], int | None]:
  """The flat u32 kernel-arg array of a launch, plus the index of the `core_id` slot (None if absent).

  Args live at `pp` (the prologue loads `ld r0,[pp+0]`, `ld r1,[pp+4]`, ...), one array per task that
  differs only in the `core_id` slot (the task index; tinygrad passes None there, as for `ops_cpu.py`).
  The slot is set to 0 here so the unbound-arg check is independent of per-task packing.
  """
  args = [raw.dev_addr(b) for b in bufs] + list(vals)
  # The last CYCLE_STAMP_BYTES of the param buffer hold the cycle stamps and would overwrite args there.
  if len(args) * 4 > CYCLE_STAMP_OFF:
    raise ZhouyiError(f"{len(args)} kernel args exceed the {CYCLE_STAMP_OFF} B of the {PARAM_BYTES} B param "
                    f"buffer a kernel may use (the last {CYCLE_STAMP_BYTES} B carry the cycle stamps)")
  cid = runtimevars.get("core_id")
  if cid is not None: cid += len(bufs)
  elif ntasks != 1: raise ZhouyiError(f"ZHOUYI launch wants {ntasks} threads but the kernel has no core_id")
  if cid is not None: args[cid] = 0
  if any(x is None for x in args): raise ZhouyiError(f"ZHOUYI kernel arg {args.index(None)} is unbound")
  return args, cid


def pack_blobs(args: list[int], cid: int | None, ntasks: int) -> list[bytes]:
  """`args` packed per task: one blob per task, identical except the `core_id` slot holds the task index."""
  blobs = []
  for t in range(ntasks):
    if cid is not None: args[cid] = t
    blobs.append(struct.pack("<%dI" % len(args), *[a & 0xFFFFFFFF for a in args]))
  return blobs
