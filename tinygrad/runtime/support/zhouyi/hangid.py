"""Opt-in hung-job identification: `ZHOUYI_HANG_ID=1|2[:path]` (default 0: off; then nothing in this module runs).

A TEC that never finishes shows up as the KMD's deadline (`state=2 (EXCEPTION)`) or as our own timeout, and the error names only
a job id. With the switch on, `RawDevice.wait_jobs` appends a report of every job of that submission, decoded from the job's own
TCB chain as the hardware read it:

  * per group (one kernel on one core): the kernel's name (a `register_csrc` kernel's tag when the caller supplies
    `CSRC_TAGS`), its task count and dependency mode;
  * the first task's args read back from its param buffer: buffer names (`NAMER`), sizes, device addresses, the residue
    mod 8 KiB and the L2 sets they cover (set = address bits [12:6]);
  * per task: the stack pointer with its residue and the L2 sets of its frame (or LSRAM), the exception vector the image
    recorded in its priv word 0, and with level 2 its start / end cycle stamps;
  * which jobs of the submission completed, and the context the caller set (`CONTEXT`).

`ZhouyiGraph` adds where the job sits in the frozen graph (step, its kernels, the previous and next kernels, the steps that
completed in this call) and `ZhouyiProgram` the launch's own buffers.

  ZHOUYI_HANG_ID=1         the report only. Nothing on the device side changes: no image, buffer, address or submission differs
                           from the switch off; the report is host reads of our own TCB / param / priv buffers (after the
                           host-cache invalidate) and runs only on the failure path.
  ZHOUYI_HANG_ID=2         + every compiled image (codegen and register_csrc kernels, not the extra/zhouyi runners) stores the
                           TEC cycle counter at its start and before each task epilogue into the last 8 B of its param buffer
                           (launch.CYCLE_STAMP_OFF), and both words are poisoned before every submit. An end stamp still at
                           poison means the task never reached its epilogue: in a fused chain the first group with such tasks
                           is the kernel that hung. A start stamp is a cached store, written back only by the epilogue or an
                           eviction, so poison there proves nothing. Costs: a different image (everything recompiles once),
                           ~30 bundles and one dirty line a task, ~1 us a task a submit.
  ZHOUYI_HANG_ID=1:path    (or 2:path) the level as above, and also append every report to this file.

Set by callers (all optional; read only when a report is built):
  CONTEXT    dict printed with the report (e.g. phase / layer / iteration)
  NAMER      zero-arg callable returning (name, Tensor | tinygrad Buffer | zhouyi Buffer) pairs
  CSRC_TAGS  register_csrc key -> a tag (e.g. the examples' `Kernels.key` name with its geometry)
"""
from __future__ import annotations
import bisect, ctypes, os, re, struct, sys, time, zlib

def parse(v: str|None) -> tuple[int, str|None]:
  """`ZHOUYI_HANG_ID`'s value -> (level, report log path or None): "" / "0" off, "1" / "2" the level, "<level>:<path>" + the log."""
  lv, _, path = (v or "").partition(":")
  return int(lv or "0"), path or None

LEVEL, LOG = parse(os.environ.get("ZHOUYI_HANG_ID"))
CONTEXT: dict = {}
NAMER = None
CSRC_TAGS: dict = {}

POISON = 0xFFFFFFFF
LSRAM_LO = 0xFA000000
L2_WAY = 8192                          # L2: 8 ways of 8 KiB, 64 B lines -> 128 sets

_TEXTS: dict[int, dict] = {}           # image text device address -> {name, frame, lib, stamps}
_LIBKEY: dict[int, int] = {}           # crc32 of a compiled lib -> its register_csrc key
_JOBS: dict[int, tuple[int, int, int]] = {}   # job id -> (head, first task, last task) TCB PAs
_PP: dict[int, int] = {}               # level 2: param buffer device address -> va, for the images that stamp
JOB_MEMORY = 4096


# ***************** registration (cheap; only called when LEVEL) *****************
def lib_id(lib: bytes) -> int: return zlib.crc32(lib)

def register_text(addr: int, name: str, **info) -> None:
  """Name the image whose text sits at device address `addr` (a Program, a runner's launch, the barrier)."""
  if addr not in _TEXTS or info: _TEXTS[addr] = dict(_TEXTS.get(addr, {}), name=name, **info)

def note_csrc(lib: bytes, key: int) -> None: _LIBKEY[lib_id(lib)] = key

def note_params(raw, params, forget=()) -> None:
  """Level 2: the param buffers of a stamping image (Programs), so `submit` can poison their stamp words."""
  for p in forget: _PP.pop(raw.dev_addr(p), None)
  for p in params: _PP[raw.dev_addr(p)] = p.va

def wrap_submit(raw) -> None:
  """Record every job's TCB chain (and at level 2 poison the stamps of its tasks) around `raw.submit`; keep `raw._hid_bypa`
  (allocation start PA -> Buffer) exact by wrapping `req_buf` / `free_buf`. Installed on a fresh RawDevice (no buffers yet)."""
  sub, req, free = raw.submit, raw.req_buf, raw.free_buf
  raw._hid_bypa = {b.pa: b for b in raw.bufs}
  def submit(head_tcb_pa, first_task_tcb_pa, last_task_tcb_pa, dbg_core=None):
    if LEVEL >= 2: _poison(raw, head_tcb_pa, first_task_tcb_pa, last_task_tcb_pa)
    jid = sub(head_tcb_pa, first_task_tcb_pa, last_task_tcb_pa, dbg_core)
    _JOBS[jid] = (head_tcb_pa, first_task_tcb_pa, last_task_tcb_pa)
    _JOBS.pop(jid - JOB_MEMORY, None)
    return jid
  def req_buf(nbytes, *a, **kw):
    b = req(nbytes, *a, **kw); raw._hid_bypa[b.pa] = b; return b
  def free_buf(b):
    if raw._hid_bypa.get(b.pa) is b: del raw._hid_bypa[b.pa]
    free(b)
  raw.submit, raw.req_buf, raw.free_buf = submit, req_buf, free_buf

def _poison(raw, head_pa: int, first_pa: int, last_pa: int) -> None:
  from .dev import TCB_LEN, T_PP
  from .launch import CYCLE_STAMP_OFF
  if (b := raw._hid_bypa.get(head_pa)) is None: return        # every chain starts its own TCB allocation
  va = b.va + (first_pa - b.pa)
  for i in range((last_pa - first_pa) // TCB_LEN + 1):
    if (pva := _PP.get(ctypes.c_uint32.from_address(va + i * TCB_LEN + T_PP).value)) is not None:
      w = (ctypes.c_uint32 * 2).from_address(pva + CYCLE_STAMP_OFF); w[0] = w[1] = POISON


# ***************** address description *****************
class _Where:
  """Allocations (from `raw.bufs`, current at report time) and caller names, by device address."""
  def __init__(self, raw):
    self.raw, self.asid0 = raw, raw.asid0
    self.allocs = sorted(((raw.dev_addr(b), b) for b in raw.bufs), key=lambda t: t[0])
    self.starts = [a for a, _ in self.allocs]
    self.names: list[tuple[int, int, str]] = []
    if NAMER is not None:
      try:
        for name, o in NAMER():
          if (r := obj_range(raw, o)) is not None: self.names.append((r[0], r[0] + r[1], str(name)))
      except Exception as e: self.names.append((-1, -1, f"<NAMER failed: {e!r}>"))
  def alloc(self, a: int):
    i = bisect.bisect_right(self.starts, a) - 1
    if i >= 0 and a < self.allocs[i][0] + self.allocs[i][1].map_bytes: return self.allocs[i]
    return None
  def named(self, a: int) -> tuple[int, int, str]|None:
    """The smallest named range holding `a`."""
    best = None
    for s, e, n in self.names:
      if s <= a < e and (best is None or e - s < best[1] - best[0]): best = (s, e, n)
    return best
  def name(self, a: int) -> str|None:
    if (best := self.named(a)) is None: return None
    return best[2] if a == best[0] else f"{best[2]}+{a - best[0]:#x}"
  def known(self, a: int) -> bool: return self.alloc(a) is not None or self.name(a) is not None
  def __call__(self, a: int, n: int|None = None) -> str:
    s = f"{a:#010x} r{a % L2_WAY:#06x}"
    if n: s += f" {n} B {sets(a, n)}"
    if (nm := self.name(a)) is not None:
      s += f" '{nm}'"
      if not n and (b := self.named(a)) is not None: s += f" [{b[1] - b[0]} B {sets(b[0], b[1] - b[0])}]"
    if (al := self.alloc(a)) is not None:
      s += f" (alloc {al[0]:#x}{f'+{a - al[0]:#x}' if a != al[0] else ''} of {al[1].nbytes} B)"
    elif a >= LSRAM_LO: s += " (LSRAM/GSRAM window)"
    else: s += " (no live allocation of ours: a weight slot or freed)"
    return s
  def read_u32(self, a: int, n: int, invalidate: bool = True) -> list[int]|None:
    """`n` words at device address `a` from our mapping, after the host-cache invalidate (device-written memory)."""
    al = self.alloc(a)
    if al is None or not al[1].va: return None
    if invalidate:
      try: self.raw.cache_invalidate(al[1])
      except Exception: pass
    return list((ctypes.c_uint32 * n).from_address(al[1].va + (a - al[0])))

def sets(a: int, n: int) -> str:
  """The L2 sets [a, a + n) covers (set = address bits [12:6])."""
  if n >= L2_WAY: return "all sets"
  s0, s1 = (a >> 6) & 127, ((a + n - 1) >> 6) & 127
  return f"set {s0}" if s0 == s1 else f"sets {s0}-{s1}" if s0 < s1 else f"sets {s0}-127,0-{s1}"

def obj_range(raw, o) -> tuple[int, int]|None:
  """(device address, bytes) of a Tensor, a tinygrad Buffer or a zhouyi Buffer; None if not allocated."""
  try:
    if hasattr(o, "uop"):
      u = o.uop
      try: o = u.buffer
      except Exception: o = u.base.buffer
    if hasattr(o, "is_allocated"):
      if not o.is_allocated(): return None
      n, o = o.nbytes, o._buf
    else: n = o.nbytes
    return o.pa - raw.asid0, n
  except Exception: return None

def kernel_label(addr: int) -> str:
  t = _TEXTS.get(addr)
  if t is None: return f"<unregistered image @{addr:#x}>"
  s = t["name"]
  if (key := _LIBKEY.get(t.get("lib", -1))) is not None: s += f" [csrc #{key}{' ' + CSRC_TAGS[key] if key in CSRC_TAGS else ''}]"
  return s

def describe_bufs(raw, bufs, w: _Where|None = None, indent: str = "      ") -> list[str]:
  """One line per arg buffer (tinygrad Buffers or zhouyi Buffers)."""
  w = w or _Where(raw); out = []
  for i, b in enumerate(bufs):
    r = obj_range(raw, b) if b is not None else None
    dt = getattr(b, "dtype", None)
    out.append(f"{indent}buf{i}: " + (w(r[0], r[1]) + (f" {dt}" if dt is not None else "") if r is not None else "<unallocated>"))
  return out

def runner_label(ji) -> str:
  """An ExecItem's kernel: the Program's name, launch width and csrc tag, or a runner's display name."""
  prg = ji.prg
  p = getattr(prg, "p", None)
  if p is not None:
    s = f"{p.name} gs={p.global_size[0] if p.global_size else '?'}"
    lib = getattr(prg, "lib", None)
    if lib is not None and (key := _LIBKEY.get(lib_id(lib))) is not None: s += f" [csrc #{key}{' ' + CSRC_TAGS[key] if key in CSRC_TAGS else ''}]"
    return s
  return re.sub("\x1b\\[(K|.*?m)", "", str(getattr(prg, "display_name", type(prg).__name__)))


# ***************** the job report (from wait_jobs' failure path) *****************
_DEP = {0: "none", 0x10: "IMMEDIATE", 0x20: "PRE_ALL"}

def decode_job(raw, jid: int, w: _Where) -> list[str]:
  from .dev import TCB_LEN, TASK_INDEX, T_FLAG, T_SPC, T_GROUP_ID, T_TASK_ID, T_GROUP_DIM, T_SP
  from .launch import CYCLE_STAMP_OFF
  if (sub := _JOBS.get(jid)) is None: return [f"    job {jid:#x}: its chain was not recorded (submitted before ZHOUYI_HANG_ID was on?)"]
  head, first, last = sub
  hdev = head - raw.asid0
  al = w.alloc(hdev)
  if al is None or not al[1].va: return [f"    job {jid:#x}: TCB chain at pa {head:#x} is not a live allocation"]
  try: raw.cache_invalidate(al[1])
  except Exception: pass
  base = al[1].va + (hdev - al[0])
  ntasks = (last - first) // TCB_LEN + 1
  grid = ctypes.string_at(base, TCB_LEN)
  gm = struct.unpack_from("<I", grid, 16)[0] == 1
  tasks = []
  for i in range(ntasks):
    t = ctypes.string_at(base + (TASK_INDEX + i) * TCB_LEN, TCB_LEN)
    flag = struct.unpack_from("<I", t, T_FLAG)[0]
    spc = struct.unpack_from("<I", t, T_SPC)[0]
    gid = struct.unpack_from("<H", t, T_GROUP_ID)[0]
    tid = struct.unpack_from("<H", t, T_TASK_ID)[0]
    gdim = struct.unpack_from("<H", t, T_GROUP_DIM)[0]
    sp, pp, dp, cp = struct.unpack_from("<IIII", t, T_SP)
    tasks.append(dict(i=i, flag=flag, spc=spc, gid=gid, tid=tid, gdim=gdim, sp=sp, pp=pp, dp=dp))
  groups: list[list[dict]] = []
  for t in tasks:
    if not groups or groups[-1][0]["gid"] != t["gid"]: groups.append([t])
    else: groups[-1].append(t)
  out = [f"    job {jid:#x}: {ntasks} task(s) in {len(groups)} group(s), TCBs at pa {head:#x}{', GM remap on' if gm else ''}"]
  first_unfinished = None
  for gi, g in enumerate(groups):
    t0 = g[0]; info = _TEXTS.get(t0["spc"], {})
    out.append(f"    g{gi} (TCB group {t0['gid']}): {kernel_label(t0['spc'])} -- {len(g)} task(s), dep {_DEP.get(t0['flag'] & 0x30, hex(t0['flag'] & 0x30))}")
    words = w.read_u32(t0["pp"], 24, invalidate=False) or []
    while words and words[-1] == 0: words.pop()
    args = [w(v) if v >= 0x10000 and w.known(v) else str(v) for v in words]
    if args: out.append("      args (task 0): " + " | ".join(f"a{k}={s}" for k, s in enumerate(args)))
    stamps = LEVEL >= 2 and info.get("stamps", False)
    unfinished = 0
    for t in g:
      frame = info.get("frame") or 0
      if t["sp"] >= LSRAM_LO: st = f"sp {t['sp']:#x} (LSRAM)"
      else: st = f"sp {t['sp']:#010x} r{t['sp'] % L2_WAY:#06x}" + (f" frame {frame} B {sets(t['sp'] - frame, frame)}" if frame else "")
      exc = w.read_u32(t["dp"], 1)
      s = f"      t{t['tid']}: {st}; exception vector {exc[0] if exc else '?'}"
      if stamps:
        cy = w.read_u32(t["pp"] + CYCLE_STAMP_OFF, 2)
        if cy:
          end_ok = cy[1] != POISON
          unfinished += not end_ok
          s += f"; stamps start {'poison' if cy[0] == POISON else cy[0]} end {'POISON (no epilogue)' if not end_ok else cy[1]}"
          if end_ok and cy[0] != POISON: s += f" ({(cy[1] - cy[0]) & 0xFFFFFFFF} cycles)"
      out.append(s)
    if stamps and unfinished:
      out.append(f"      -> {unfinished}/{len(g)} task(s) of g{gi} never reached the epilogue")
      if first_unfinished is None: first_unfinished = gi
  if LEVEL >= 2:
    out.append("    => first group with unfinished tasks: " + (f"g{first_unfinished} {kernel_label(groups[first_unfinished][0]['spc'])}"
                                                               if first_unfinished is not None else "none among the stamping images"))
  return out

def job_report(raw, failed: int|None, job_ids, done, pending, why: str, waited_s: float) -> str:
  """The failure report of one submission (`job_ids`), appended to wait_jobs' error."""
  try:
    w = _Where(raw)
    lines = [f"[ZHOUYI_HANG_ID] {why} after {waited_s:.1f} s of waiting, {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"  context: {CONTEXT}" if CONTEXT else "  context: (none set)",
             f"  submission: {len(job_ids)} job(s) {[hex(j) for j in job_ids]}; completed {sorted(hex(j) for j in done)}; "
             f"not completed {sorted(hex(j) for j in pending)}; banked from other submissions {sorted(hex(j) for j in raw._banked)}"]
    for j in list(job_ids) + ([failed] if failed is not None and failed not in job_ids else []):
      tag = "FAILED" if j == failed else "completed" if j in done else "not completed"
      lines.append(f"  job {j:#x} [{tag}]")
      lines += decode_job(raw, j, w)
    text = "\n".join(lines)
  except Exception as e: text = f"[ZHOUYI_HANG_ID] report failed: {e!r}"
  log(text)
  return "\n" + text

def log(text: str) -> None:
  if LOG:
    try:
      with open(LOG, "a") as f: f.write(text + "\n")
    except OSError as e: print(f"ZHOUYI_HANG_ID log {LOG}: {e}", file=sys.stderr)
