"""Raw `/dev/aipu` submit path (the ioctls from `autogen/aipu.py`); no vendor runtime.

On Zhouyi V3 a job is a TCB chain in DMA memory (the start-PC descriptor fields are v1/v2 only):
`head/first_task/last/tail_tcb_pa` are full 64-bit PAs, and every pointer inside a TCB is a 32-bit
offset from the ASID0 base.

    open("/dev/aipu", O_RDWR)
    QUERY_CAP        -> asid_base[0], isa version, cluster/core/tec counts
    ALLOC_GRID_ID    -> grid_id
    REQ_BUF   x N    -> {pa, dev_offset, bytes};  mmap(fd, dev_offset) for the VA
    ...write text / TCBs / params...
    SCHEDULE_JOB     -> struct aipu_job_desc (TCB chain + asid0_base)
    QUERY_STATUS     -> poll; state DONE(1) / EXCEPTION(2)
    FREE_BUF  x N

Note: `REQ_BUF` mappings are uncached (Normal-NC), not coherent, so host reads over them are slow.
The pages also have a cacheable alias (the kernel linear map) whose stale clean lines can be served to
host reads after the device writes: call `RawDevice.cache_invalidate` before reading device output.
"""
import ctypes, mmap, os, struct, time
from typing import NamedTuple
from tinygrad.runtime.autogen import aipu, libc
from . import ZhouyiError, mapped_extent
from . import hangid as _hid
MM_TEXT, MM_RODATA, MM_STACK, MM_STATIC, MM_REUSE, MM_TCB = (aipu.AIPU_MM_DATA_TYPE_TEXT, aipu.AIPU_MM_DATA_TYPE_RODATA, aipu.AIPU_MM_DATA_TYPE_STACK,
                                                             aipu.AIPU_MM_DATA_TYPE_STATIC, aipu.AIPU_MM_DATA_TYPE_REUSE, aipu.AIPU_MM_DATA_TYPE_TCB)
TCB_LEN = 128
EXEC_FLAG_DBG_DISPATCH = 1 << 5   # the KMD UAPI's AIPU_JOB_EXEC_FLAG_DBG_DISPATCH: [zhouyi v3 only] schedule on the named core


# Default `wait_job` timeout. Kept short: a fault presents as a hang (exception vectors spin), so the
# timeout is the only fault detector. Genuinely long kernels can raise it with `AIPU_JOB_TIMEOUT`.
JOB_TIMEOUT_S = 10.0


def job_timeout(timeout_s: float|None = None) -> float:
  """Timeout for `wait_job`: explicit argument, else `AIPU_JOB_TIMEOUT`, else `JOB_TIMEOUT_S`."""
  return timeout_s if timeout_s is not None else float(os.environ.get("AIPU_JOB_TIMEOUT", JOB_TIMEOUT_S))


class Buffer:
  """One mapped `REQ_BUF` allocation. `pa` is the driver-reported physical address, `va` our mapping;
  the device sees `pa - asid0` as a u32.

  `nbytes` is the logical size requested; `map_bytes` is the page-rounded extent allocated and mapped,
  which `munmap` and `FREE_BUF` must receive. For a view (`_offset`) `map_bytes` defaults to `nbytes`;
  a view owns no mapping and must never be unmapped or freed."""
  def __init__(self, pa: int, dev_offset: int, nbytes: int, va: int, offset: int = 0, map_bytes: int|None = None):
    self.pa, self.dev_offset, self.nbytes, self.va, self.offset = pa, dev_offset, nbytes, va, offset
    self.map_bytes = nbytes if map_bytes is None else map_bytes
  @property
  def size(self): return self.nbytes


class RawDevice:
  def __init__(self, path: str = "/dev/aipu"):
    self.fd = os.open(path, os.O_RDWR)
    self.cap = cap = aipu.AIPU_IOCTL_QUERY_CAP(self.fd)
    self.asid0, self.isa_version = cap.asid_base[0], cap.partition_cap.version
    if self.isa_version != aipu.AIPU_ISA_VERSION_ZHOUYI_V3:
      raise ZhouyiError(f"expected Zhouyi V3 (isa {aipu.AIPU_ISA_VERSION_ZHOUYI_V3}), got {self.isa_version}")
    self.cluster_cnt = cap.partition_cap.cluster_cnt
    self.core_cnt = cap.partition_cap.clusters[0].core_cnt
    self.tec_cnt = cap.partition_cap.clusters[0].tec_cnt
    # a u32 payload the generated helper cannot size: pass it as a 1-element array (a zero c_uint32 would read as no payload)
    self.grid_id = aipu.AIPU_IOCTL_ALLOC_GRID_ID(self.fd, __payload=(ctypes.c_uint32 * 1)())[0]
    self.bufs: list[Buffer] = []
    self._job_id = 0x4660000
    # Completions dequeued by a poll that was waiting for other jobs; see `wait_jobs`.
    self._banked: dict[int, aipu.struct_aipu_job_status_desc] = {}
    # The async tail (`submit_async`): jobs submitted and not yet waited for, the tag of their submitter and a callable that
    # describes them if they fail. Everything that could race them -- the next submit, a host read or write of device memory, a
    # free, a slot map -- calls `drain()` first, so one job is on the device at a time as before.
    self.inflight: list[int] = []
    self.inflight_tag: object = None
    self._inflight_report = None
    if _hid.LEVEL: _hid.wrap_submit(self)    # ZHOUYI_HANG_ID: record each job's chain for the failure report (hangid.py)

  def __repr__(self):
    return (f"ZHOUYI isa v{self.isa_version}, {self.cluster_cnt} cluster(s), {self.core_cnt} core(s), "
            f"{self.tec_cnt} TEC(s), asid0=0x{self.asid0:x}, grid={self.grid_id}")

  def next_job_id(self) -> int:
    self._job_id += 1
    return self._job_id
  def dev_addr(self, b: Buffer) -> int: return b.pa - self.asid0   # every device pointer is a u32 ASID0 offset

  # A 16-byte vector load reads 16 bytes past its end on silicon (the simulator models exactly 16; scalar loads, 32-byte loads
  # and stores stay in bounds): at a buffer's end the overhang lands in the next page -- unmapped, the SMMU faults it and the
  # TEC hangs with no NPU exception (dmesg: F_TRANSLATION at the buffer's end); across the window's top (0xC0000000) the load
  # returns garbage. So every mapping carries LOAD_OVERHANG readable bytes after the logical size (a page only when the
  # buffer ends within 16 B of a page boundary). Measured on the device.
  LOAD_OVERHANG = 16

  def req_buf(self, nbytes: int, data_type: int = MM_REUSE) -> Buffer:
    # Clamp a zero-byte request once, before both sizes are derived from it.
    nbytes = max(nbytes, 1)
    rq = aipu.AIPU_IOCTL_REQ_BUF(self.fd, align_in_page=1, bytes=mapped_extent(nbytes + self.LOAD_OVERHANG), data_type=data_type,
                                 asid=0, region=0)
    # Note: when the ASID0 window is exhausted the driver can return rc=0 with an all-zero descriptor
    # (the mmap would then fail with EINVAL). Check the descriptor, not just the return code.
    if rq.desc.bytes == 0 or rq.desc.pa == 0:
      raise ZhouyiError(f"REQ_BUF returned rc=0 with a ZEROED descriptor for {nbytes} B "
                      f"(extent {mapped_extent(nbytes + self.LOAD_OVERHANG)}) — the ASID0 window refused and reported success. "
                      "This is exhaustion, not an mmap fault.")
    va = libc.mmap(None, rq.desc.bytes, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, self.fd, rq.desc.dev_offset)
    if va is None or va == ctypes.c_void_p(-1).value: raise ZhouyiError("mmap of a zhouyi buffer failed")
    ctypes.memset(va, 0, rq.desc.bytes)
    # Record both the logical size and the rounded extent (`rq.desc.bytes`, used by munmap/FREE_BUF).
    b = Buffer(rq.desc.pa, rq.desc.dev_offset, max(nbytes, 1), va, map_bytes=rq.desc.bytes)
    self.bufs.append(b)
    return b

  def drain(self) -> None:
    """Wait for the jobs `submit_async` left in flight (no-op when none). Single-threaded: call it from the submitting thread."""
    if not self.inflight: return
    ids, rep = self.inflight, self._inflight_report
    self.inflight, self.inflight_tag, self._inflight_report = [], None, None
    try: self.wait_jobs(ids)
    except ZhouyiError as e:
      if rep is None: raise
      raise ZhouyiError(f"{e}\n{rep()}") from None

  def submit_async(self, head_tcb_pa: int, first_task_tcb_pa: int, last_task_tcb_pa: int, tag: object = None, report=None) -> int:
    """`submit` and return without waiting: the job stays in flight until the next `drain()` (which every later submit, host
    read or write, free and slot map does). `tag`: who submitted it (`inflight_tag`); `report`: a callable whose text is added if
    the job fails."""
    jid = self.submit(head_tcb_pa, first_task_tcb_pa, last_task_tcb_pa)
    self.inflight.append(jid)
    self.inflight_tag, self._inflight_report = tag, report
    return jid

  def view(self, b: Buffer, offset: int, nbytes: int) -> Buffer:
    """A Buffer for [offset, offset + nbytes) of allocation `b` (owns no mapping: never unmap or free it)."""
    return Buffer(b.pa + offset, b.dev_offset + offset, nbytes, b.va + offset, b.offset + offset)

  def cache_invalidate(self, b: Buffer):
    """Make the device's writes to `b` visible to host reads.

    Invalidates the cacheable kernel linear-map alias of the buffer's pages, whose stale clean lines
    would otherwise be served to reads of the non-cacheable mapping. Note: requires the patched driver;
    in the stock driver this ioctl walks the non-cacheable alias and has no effect. `BUF_CACHE_FLUSH`
    is not needed (the stale lines are clean). Waits for a job left in flight first (`drain`).
    """
    self.drain()
    aipu.AIPU_IOCTL_BUF_CACHE_INVALID(self.fd, pa=b.pa, dev_offset=b.dev_offset, bytes=b.map_bytes)

  def free_buf(self, b: Buffer):
    self.drain()
    libc.munmap(b.va, b.map_bytes)
    aipu.AIPU_IOCTL_FREE_BUF(self.fd, pa=b.pa, dev_offset=b.dev_offset, bytes=b.map_bytes)
    if b in self.bufs: self.bufs.remove(b)

  # Zero-copy weights (a KMD extension's WBUF_* / SLOT_* ioctls, numbers 30-34; V3 custom-IOVA hardware): weight memory lives outside the NPU
  # window in 2 MiB chunks, filled by the process through a write-combine mmap; a window slot is re-pointed at a range of it
  # by SMMU block mappings (~20 us for a 400 MB layer) instead of copying the range in. An older driver refuses the ioctls
  # (ENOTTY): callers keep copying. Everything is freed with the fd; a slot must not be re-mapped while a job reads it.
  WSLOT_CHUNK = 2 << 20
  def _wslot(self, nr: int, fmt: str, *vals: int, write_only: bool = False) -> tuple[int, ...]:
    import fcntl
    b = bytearray(struct.pack(fmt, *vals))
    fcntl.ioctl(self.fd, ((1 if write_only else 3) << 30) | (len(b) << 16) | (ord(aipu.AIPU_IOCTL_MAGIC) << 8) | nr, b, True)
    return struct.unpack(fmt, b)
  def wbuf_alloc(self, nbytes: int) -> tuple[int, mmap.mmap]:
    """Weight memory of `nbytes` (whole 2 MiB chunks, zeroed): (id, its write-combine mapping). ENOMEM: no contiguous chunks."""
    _, wid, off = self._wslot(30, "<QQQ", nbytes, 0, 0)
    return wid, mmap.mmap(self.fd, -(-nbytes // self.WSLOT_CHUNK) * self.WSLOT_CHUNK, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=off)
  def wbuf_free(self, wid: int):
    self.drain()
    self._wslot(31, "<Q", wid, write_only=True)      # EBUSY while mmap()ed or mapped in a slot
  def slot_alloc(self, nbytes: int) -> tuple[int, int]:
    """A 2 MiB-aligned window range with nothing mapped: (id, its NPU-view address)."""
    _, sid, pa = self._wslot(32, "<QQQ", nbytes, 0, 0)
    return sid, pa
  def slot_free(self, sid: int):
    self.drain()
    self._wslot(33, "<Q", sid, write_only=True)
  def slot_map(self, sid: int, wid: int, offset: int, nbytes: int, drain: bool = True):
    """Point slot `sid` at weight buffer `wid`'s [offset, offset + nbytes) (offset 2 MiB-aligned; nbytes 0 unmaps).
    `drain=False`: do not wait for a job left in flight -- only when the caller knows that job does not read this slot (and
    from a thread other than the submitting one, always)."""
    if drain: self.drain()
    self._wslot(34, "<QQQQ", sid, wid, offset, nbytes, write_only=True)

  def submit(self, head_tcb_pa: int, first_task_tcb_pa: int, last_task_tcb_pa: int,
             dbg_core: int|None = None) -> int:
    """SCHEDULE_JOB over an already-built TCB chain. Returns the job id.

    `dbg_core` pins the job to a core. Note: `last_task_tcb_pa` must be exact (no default). The driver
    completes a job when `tail_tcbp == last_task_tcb_pa - asid0_base` and attributes interrupts by
    `first_task_tcb_pa <= tail_tcbp <= last_task_tcb_pa`; a wrong value wedges the device for the rest
    of the process. A job left in flight by `submit_async` is waited for first: two jobs in flight run concurrently, not
    in order (Distributing.md 4)."""
    if self.inflight: self.drain()
    jd = aipu.struct_aipu_job_desc()
    jd.job_id, jd.aipu_arch, jd.aipu_version = self.next_job_id(), 0, self.isa_version
    jd.exec_flag = aipu.AIPU_JOB_EXEC_FLAG_QOS_SLOW | aipu.AIPU_JOB_EXEC_FLAG_MULTI_GROUP
    if dbg_core is not None:
      # AIPU_JOB_EXEC_FLAG_DBG_DISPATCH + core_id: run this job on the named core (range-checked by the KMD).
      jd.exec_flag |= aipu.AIPU_JOB_EXEC_FLAG_DBG_DISPATCH
      jd.core_id = dbg_core
    jd.enable_poll_opt, jd.profile_fd, jd.asid0_base = 1, -1, self.asid0
    jd.head_tcb_pa = head_tcb_pa
    jd.first_task_tcb_pa = first_task_tcb_pa
    jd.last_task_tcb_pa = jd.tail_tcb_pa = last_task_tcb_pa
    aipu.AIPU_IOCTL_SCHEDULE_JOB(self.fd, __payload=jd)
    return jd.job_id

  QUERY_SLOTS = 16

  def wait_jobs(self, job_ids, timeout_s: float|None = None) -> dict[int, aipu.struct_aipu_job_status_desc]:
    """Wait for every id in `job_ids`. Returns `{job_id: status}`.

    `QUERY_STATUS` dequeues completions, so completions of other jobs are banked in `self._banked`
    (as copies; the status array is reused by the next ioctl) rather than discarded, and ids already
    banked are not polled for. Up to `QUERY_SLOTS` completions are read per poll. An EXCEPTION on any
    job aborts, even one this call is not waiting for."""
    timeout_s = job_timeout(timeout_s)
    pending, done = set(job_ids), {}
    for j in list(pending):
      if (banked := self._banked.pop(j, None)) is not None:
        pending.discard(j)
        done[j] = banked
    st = (aipu.struct_aipu_job_status_desc * self.QUERY_SLOTS)()
    q = aipu.struct_aipu_job_status_query()
    q.max_cnt, q.of_this_thread = self.QUERY_SLOTS, 1
    q.status = ctypes.cast(st, ctypes.POINTER(aipu.struct_aipu_job_status_desc))  # type: ignore[arg-type]
    t0 = time.perf_counter()
    while pending and time.perf_counter() - t0 < timeout_s:
      q.poll_cnt = 0
      aipu.AIPU_IOCTL_QUERY_STATUS(self.fd, __payload=q)
      for k in range(q.poll_cnt):
        rec = aipu.struct_aipu_job_status_desc.from_buffer_copy(st[k])
        if rec.state != aipu.AIPU_JOB_STATE_DONE:
          raise ZhouyiError(f"ZHOUYI job {rec.job_id:#x} state={rec.state} (EXCEPTION)"
                          + ("" if rec.job_id in pending else f" — a sibling of {sorted(hex(j) for j in pending)}")
                          + (_hid.job_report(self, rec.job_id, list(job_ids), done, pending, f"job {rec.job_id:#x} state={rec.state}",
                                             time.perf_counter() - t0) if _hid.LEVEL else ""))
        if rec.job_id in pending:
          pending.discard(rec.job_id)
          done[rec.job_id] = rec
        else: self._banked[rec.job_id] = rec
    if pending:
      raise ZhouyiError(f"ZHOUYI job(s) {sorted(hex(j) for j in pending)} timed out after {timeout_s}s "
                      f"(banked meanwhile: {sorted(hex(j) for j in self._banked)}). "
                      "A fault presents as a hang by design — the exception vectors spin. If the kernel is "
                      "genuinely long rather than faulting, raise AIPU_JOB_TIMEOUT."
                      + (_hid.job_report(self, None, list(job_ids), done, pending, "host timeout", time.perf_counter() - t0) if _hid.LEVEL else ""))
    return done

  def wait_job(self, job_id: int, timeout_s: float|None = None):
    """Wait for a single job (via `wait_jobs`)."""
    return self.wait_jobs((job_id,), timeout_s)[job_id]


# ***************** the TCB chain *****************
# The chain layout the vendor's user-mode driver builds, field for field. The head is three TCBs:
#
#   [0] grid/INIT   flag = (3<<16)|0  <- the (3<<16) announces 2 SegMMU TCBs
#   [1] SegMMU      all zero except byte 24 = 0x31 and byte 88 = 0x31
#   [2] SegMMU      identical
#   [3..] TASK      one per task; flag = TASK(1), GROUP_END(1<<6) on a group's last
#
# The SegMMU TCBs are consumed positionally after the grid TCB. Without them, fetch and loads work but
# stores never become visible.
# A group of N tasks is N task TCBs chained through `next`, each with its own `sp`/`pp`/`dp`/`tcbp`;
# `group_dim_x` is derived from the task count.
GRID_SEGMMU_COUNT = 3
HEAD_TCBS = 3                  # grid + 2 SegMMU, before the first task
TASK_INDEX = HEAD_TCBS         # index of the FIRST task TCB
ASID_RD, ASID_WR = 1 << 6, 1 << 5

FLAG_TASK, FLAG_GROUP_END = 1, 1 << 6
# GRID_END (bit 7): the vendor sets it on the last TCB of multi-subgraph jobs. It does not prevent a dep=0
# multi-group job from being retired early (the KMD retires on the last task TCB's interrupt); a barrier
# group (see `build_tcbs`) does. Available to reproduce the vendor layout; off by default.
FLAG_GRID_END = 1 << 7
INTERRUPT_EN = 0x73d           # EN_INTERRUPT_ALL_TYPE_V3 | TEC | CORE | CLUSTER, as the vendor sets it

# ***************** the dependency field: `flag[5:4]` *****************
# Set on the first task TCB of each group (the vendor's dependency types NONE / IMMEDIATE / PRE_ALL).
# Only DEP_PRE_ALL is safe: it serialises groups and is a store barrier at the group boundary.
# DEP_IMMEDIATE serialises but is not a barrier (the tail stores of earlier tasks can be lost).
# DEP_NONE overlaps groups but DONE can fire while a group is still running (work lost).
DEP_NONE, DEP_IMMEDIATE, DEP_PRE_ALL = 0, 1 << 4, 2 << 4
DEP_MASK = 0x30
DEP_MODES = (DEP_NONE, DEP_IMMEDIATE, DEP_PRE_ALL)

# Constants copied from the vendor, which writes them on every task TCB; not understood, so not
# parameters. `GROUP_ID_X` is 1 even for group 0. Removing the TEC bit from `INTERRUPT_EN` hangs the job.
GRID_DIM = (1, 1, 1)
GROUP_ID_X = 1

# Task-TCB field offsets of the V3 TCB layout.
T_FLAG, T_NEXT, T_SPC, T_INTERRUPT_EN = 0, 4, 12, 16
T_GROUP_ID, T_GRID_ID, T_TASK_ID = 20, 22, 26
T_GRID_DIM, T_GROUP_DIM, T_GROUP_ID_X, T_TASK_ID_X = 28, 34, 40, 46
T_SP, T_PP, T_DP, T_CP = 52, 56, 60, 64
T_CORE_ID, T_CLUSTER_ID, T_TEC_ID = 76, 78, 82
T_TCBP = 108


class TaskSlot(NamedTuple):
  """Per-task pointers (not shared with sibling tasks).

  `sp` (stack) and `dp` (private data) must be private. `pp` (kernel arg array) is per task so that each
  task's args can carry its task index (the prologue loads `pp` from its own TCB). `spc`/`cp` are shared."""
  sp: int
  pp: int
  dp: int


def chain_tcbs(ntasks: int) -> int:
  """How many TCBs a chain of `ntasks` tasks occupies, head included."""
  return HEAD_TCBS + ntasks


class GroupSlot(NamedTuple):
  """One group of a chain: a kernel landing on one core, its tasks on that core's TECs.

  Each group has its own `spc`/`cp` (task fields in the TCB), so different groups may run different kernels."""
  spc: int
  cp: int
  tasks: list  # list[TaskSlot]


def build_tcbs(dev: RawDevice, tcbs: Buffer, spc: int, tasks: list[TaskSlot], cp: int,
               ngroups: int = 1, asid1_base: int|None = None, dep=DEP_PRE_ALL,
               grid_end: bool = False, group_widths=None) -> bytes:
  """Serialise a chain of groups sharing one image (`spc`/`cp`) into `tcbs`. Returns the bytes.

  `tasks`: one entry per task TCB, group-major; split evenly into `ngroups`, or by `group_widths`.
  Tasks run on TECs and groups on cores (1 cluster x 3 cores x 4 TECs = 12 on the tested part).

  `dep`: dependency mode in `flag[5:4]` of the first task TCB of every group after the first (the first
  group has no predecessor). See the DEP_* notes: only DEP_PRE_ALL is safe, and it serialises groups.
  Bare dep=0 lets DONE fire with work unfinished or lost, so the backend uses one group per job and
  concurrent jobs for multiple cores. A safe overlapping shape is a barrier group: work groups at dep 0
  plus a trivial final group at DEP_PRE_ALL (`dep=[0, 0, 0, DEP_PRE_ALL]`, `group_widths=[4, 4, 4, 1]`).

  `asid1_base` defaults to `dev.asid0`: AIFF fetches weights through ASID1, so a zero base there faults.
  """
  if not tasks: raise ZhouyiError("a chain needs at least one task")
  # `group_widths`: explicit tasks-per-group (the BARRIER shape is [4, 4, 4, 1]); else an equal split.
  if group_widths is not None:
    if sum(group_widths) != len(tasks) or any(w <= 0 for w in group_widths):
      raise ZhouyiError(f"group widths {list(group_widths)} do not tile {len(tasks)} tasks")
    bounds, acc = [], 0
    for w in group_widths:
      bounds.append((acc, acc + w))
      acc += w
  else:
    if ngroups < 1 or len(tasks) % ngroups: raise ZhouyiError(f"{len(tasks)} tasks do not divide into {ngroups} group(s)")
    per_group = len(tasks)//ngroups
    bounds = [(g*per_group, (g+1)*per_group) for g in range(ngroups)]
  return build_group_tcbs(dev, tcbs, [GroupSlot(spc, cp, list(tasks[a:b])) for a, b in bounds],
                          asid1_base=asid1_base, dep=dep, grid_end=grid_end)


def gm_fields(grid: bytearray|memoryview, gm_pa: int) -> None:
  """Enable GM remap in a grid TCB (as the vendor's user-mode driver does): region 0 on as a plain remap, region 1
  unused, window at `gm_pa` (DDR PA, 4 MiB aligned). DMA whose external address falls in the window is
  served by the cluster's GM. Note: with this on, the TECs' GSRAM window (0xF8000000) is empty."""
  struct.pack_into("<I", grid, 16, 1)
  struct.pack_into("<II", grid, 24, 0, 3 << 30)
  struct.pack_into("<QQ", grid, 32, gm_pa, 0)


def build_group_tcbs(dev: RawDevice, tcbs: Buffer, groups: list[GroupSlot],
                     asid1_base: int|None = None, dep=DEP_PRE_ALL, grid_end: bool = False, gm_pa: int|None = None) -> bytes:
  """Serialise a chain of groups into `tcbs`; each group may have its own kernel (`spc`/`cp`) and width.

  `dep`: one mode for every group after the first, or a list with one entry per group (entry 0 ignored).
  For dependent kernels the mode must serialise and be a store barrier, which only DEP_PRE_ALL (the
  default) is. `gm_pa` enables GM remap (see `gm_fields`)."""
  if not groups or any(not g.tasks for g in groups): raise ZhouyiError("a chain needs at least one task in every group")
  # Barrier shape [0, 0, 0, DEP_PRE_ALL]: work groups overlap and the final trivial group waits for all of
  # them, so the chain's last TCB (on which the KMD retires the job) completes last.
  deps = list(dep) if isinstance(dep, (list, tuple)) else [dep] * len(groups)
  if len(deps) != len(groups): raise ZhouyiError(f"{len(deps)} dep entries for {len(groups)} groups")
  for d in deps:
    if d not in DEP_MODES: raise ZhouyiError(f"dependency mode {d:#x} is not one of {[hex(x) for x in DEP_MODES]}")
  ntasks = sum(len(g.tasks) for g in groups)
  need = chain_tcbs(ntasks)*TCB_LEN
  if tcbs.nbytes < need: raise ZhouyiError(f"TCB buffer holds {tcbs.nbytes} B, need {need}")
  a1 = dev.asid0 if asid1_base is None else asid1_base
  t = bytearray(need)
  base = dev.dev_addr(tcbs)
  def task_pa_rel(i): return base + (TASK_INDEX + i)*TCB_LEN

  # [0] grid/INIT: carries the ASID bases and skips over the SegMMU TCBs.
  struct.pack_into("<II", t, 0, (GRID_SEGMMU_COUNT << 16) | 0, task_pa_rel(0))
  struct.pack_into("<I", t, 20, dev.grid_id << 16)
  struct.pack_into("<QQ", t, 48, dev.asid0 | ASID_RD | ASID_WR, a1 | ASID_RD | ASID_WR)
  if gm_pa is not None: gm_fields(t, gm_pa)
  # [1],[2] SegMMU
  for k in (1, 2):
    struct.pack_into("<I", t, k*TCB_LEN + 24, 0x31)
    struct.pack_into("<I", t, k*TCB_LEN + 88, 0x31)
  # [3..] one TASK TCB per task, chained group-major. `next` is an ASID0 offset like every other
  # pointer inside a TCB; only the job descriptor's head/first/last/tail are full 64-bit PAs.
  i = 0
  for group, g in enumerate(groups):
    for idx, slot in enumerate(g.tasks):
      o, last = (TASK_INDEX + i)*TCB_LEN, idx == len(g.tasks) - 1
      # the dependency goes on the first task of each group that has a predecessor
      entry_dep = deps[group] if idx == 0 and group > 0 else 0
      # `grid_end`: set FLAG_GRID_END on the chain's final TCB, as the vendor does (off by default).
      end_flags = (FLAG_GROUP_END if last else 0) | (FLAG_GRID_END if grid_end and i == ntasks-1 else 0)
      struct.pack_into("<II", t, o+T_FLAG, FLAG_TASK | end_flags | entry_dep,
                       0 if i == ntasks-1 else task_pa_rel(i+1))
      struct.pack_into("<I", t, o+T_SPC, g.spc)
      struct.pack_into("<I", t, o+T_INTERRUPT_EN, INTERRUPT_EN)
      struct.pack_into("<HH", t, o+T_GROUP_ID, group, dev.grid_id)
      struct.pack_into("<H", t, o+T_TASK_ID, idx)
      struct.pack_into("<HHH", t, o+T_GRID_DIM, *GRID_DIM)
      struct.pack_into("<HHH", t, o+T_GROUP_DIM, len(g.tasks), 1, 1)
      struct.pack_into("<H", t, o+T_GROUP_ID_X, GROUP_ID_X)
      struct.pack_into("<H", t, o+T_TASK_ID_X, idx)
      struct.pack_into("<IIII", t, o+T_SP, slot.sp, slot.pp, slot.dp, g.cp)
      struct.pack_into("<I", t, o+T_TCBP, task_pa_rel(i))
      i += 1
  ctypes.memmove(tcbs.va, bytes(t), len(t))
  assert struct.unpack_from("<I", t, 0)[0] >> 16 == GRID_SEGMMU_COUNT, "grid TCB must announce 2 SegMMU TCBs"
  return bytes(t)


def task_placement(tcbs: Buffer, ntasks: int) -> list[tuple[int, int, int]]:
  """(cluster_id, core_id, tec_id) per task, read back from the task TCBs after a job.

  Note: the hardware does not fill these in; they read (0, 0, 0) on the tested part; kept so a firmware
  that populates them can be noticed."""
  out = []
  for i in range(ntasks):
    raw = ctypes.string_at(tcbs.va + (TASK_INDEX+i)*TCB_LEN + T_CORE_ID, T_TEC_ID + 2 - T_CORE_ID)
    core, cluster, _rsvd, tec = struct.unpack_from("<HHHH", raw, 0)
    out.append((cluster, core, tec))
  return out
