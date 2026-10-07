"""Kernel text in the cluster's GM: the default placement of every image (`ZHOUYI_TEXT_GM`, default 1; 0 disables).

The TEC fetches its instructions through the GM window like its loads: a cold 64-B line costs ~67 cycles from GM against ~300 from
DDR, and the I-cache (64 KiB a core) is empty at every task start, so every launch pays its text cold (measured
on the board: 64-B lines, no prefetch, one I-cache a core shared by its 4 TECs). So:

  * **Placement.** A new image starts in an ordinary DDR text buffer (its "home", kept for good). Images are promoted into a text
    arena at the start of the device's GM window ([0, TEXT_GM_CAP), 3 MiB) in batches (`_promote`): at a text job once no new image
    has appeared for PROMOTE_QUIET jobs (compiling has paused), or PROMOTE_MAX jobs after the last batch -- the most launched
    first. A batch writes the window's DDR backing, re-points each image's Buffer and patches its task TCBs, and the next job that
    fetches from the arena is preceded by ONE SYNC_TO_GM preload of the arena (GM.md §3c): a preload per batch, not per new kernel
    (a model compiles hundreds of kernels between its first jobs). When the arena is full the rest stay in DDR (correct, fetched
    from DDR) and the arena is re-packed now and then by launch count (`_rank`), hottest at the lowest address, so the hot decode
    kernels are the ones in GM. `stats` counts everything; DEBUG >= 1 (or ZHOUYI_TEXT_GM_DEBUG=1) prints when the arena fills and
    re-packs, and the summary (`repr`) at exit.
  * **Window layout** (one window a device, `RawDevice.gm_window`):
        [0, 3 MiB)          the text arena
        [3, 3.75 MiB)       k_gemm_gs's C partials, 64 KiB a task (`gemm_c_base`; GSRAM is empty in a GM-on job)
        [0, 4 MiB)          a GM-DATA job's (`GemmGMRunner`, ZHOUYI_GM=1): A chunks and C regions over the whole window
  * **Job modes**, decided per chain in `build_group_tcbs` (`job`):
        text  GM on, the text arena fetched from GM. Every job not below.
        off   GM off: the job holds a GSRAM-using image (its GSRAM would be empty under GM, GM.md §3d). Arena images are fetched
              from the window's DDR backing (the same bytes); k_gemm_gs's C partials land in that backing (correct, slower).
        data  GM on for data (a GM member, `gm_pa`): the data owns all of GM, so every arena image of the job runs from its DDR
              home (`_ddr_addr`: the same bytes), and after the job GM no longer holds the arena -- the next text job preloads it.
    A job holding both a GSRAM user and a GM-data member is refused (ZhouyiError): it cannot be right either way.
  * **GSRAM users** are marked: by a static scan of the image (`gsram_refs`: a `movh` of a GSRAM high half 0xF800..0xF803, or a
    `dma` with a GSRAM end), or by the caller (`Launch(..., gsram=True)`: an image that gets a GSRAM address at run time). A marked
    image never enters the arena and every job holding it runs GM off. The one production user, k_gemm_gs's C partials (an
    argument), moves into the window instead (`gemm_c_base`). test/unit/test_zhouyi_textgm.py keeps the list of GSRAM users in
    the sources closed, so a new one cannot slip in unmarked.

The policy is invisible to callers: `Launch` asks `place` for its text buffer, `build_group_tcbs` asks `job` for the chain's GM
fields, `RawDevice.submit` calls `before_submit`, `RawDevice.free_buf` calls `freed`. An image's Buffer object is the one handle
everybody holds: a move writes the bytes, re-points that object and patches every live task TCB whose `spc` held the old address.
An image's DDR home is never freed or rewritten, so a TCB that still names it (a GM-data job's) stays right.
"""
from __future__ import annotations
import ctypes, os, struct, sys
from . import ZhouyiError, hangid as _hid
from .dev import Buffer, TCB_LEN, TASK_INDEX, T_SPC, MM_TEXT, PAGE_BYTES, GM_WINDOW, gm_fields

TEXT_GM = os.environ.get("ZHOUYI_TEXT_GM", "1") == "1"
GEMM_C_OFF, GEMM_C_REG, GEMM_C_TASKS = 3 << 20, 65536, 12        # k_gemm_gs's C regions: 12 x 64 KiB from 3 MiB (to 3.75 MiB)
TEXT_GM_CAP = int(os.environ.get("ZHOUYI_TEXT_GM_CAP", str(GEMM_C_OFF)), 0)   # the arena's size (a smaller one: tests of the full arena)
TEXT_GM_ALIGN = 256
RANK = os.environ.get("ZHOUYI_TEXT_GM_RANK", "1") == "1"           # re-pack a full arena by launch counts (0: first come, first placed)
RANK_FIRST, RANK_GROWTH = 64, 4                                    # re-pack after 64 text jobs counted full, then 256, 1024, ... (x4)
PROMOTE_QUIET, PROMOTE_MAX = 4, 64                                 # promote new images after 4 jobs without a new one, or 64 jobs
DEBUG = int(os.environ.get("ZHOUYI_TEXT_GM_DEBUG", os.environ.get("DEBUG", "0")) or 0)
GSRAM_LO, GSRAM_HI = 0xF8000000, 0xF8040000                         # GSRAM: 256 KiB a core (GSRAM.md)
assert 0 < TEXT_GM_CAP <= GEMM_C_OFF and GEMM_C_OFF + GEMM_C_TASKS * GEMM_C_REG <= GM_WINDOW
SYNC_TO_GM = 1 << 30


# ***************** the GSRAM scan *****************
def gsram_refs(text: bytes) -> list[str]:
  """Where `text` names GSRAM: a `movh rd, imm16` with imm16 in GSRAM's high halves (0xF800..0xF803: how a constant 0xF80xxxxx is
  built), or a two-word `dma` whose internal or external space is GSRAM (8). [] when none. Conservative: an fp32 constant whose high
  half happens to be 0xF800..0xF803 is reported too (the image then runs GM off, which is correct, only slower). Not seen: a GSRAM
  address that arrives as an argument -- such an image must be marked by its caller (`Launch(..., gsram=True)`)."""
  n = len(text) // 4
  if n == 0: return []
  try:
    import numpy as np
    w = np.frombuffer(text, dtype="<u4", count=n)
    hi = (w >> 5) & 0xFFFF
    mv = np.nonzero(((w & 0x7FE00000) == 0x00E00000) & (hi >= 0xF800) & (hi <= 0xF803))[0].tolist()
    w0 = w[0:n - 1:4] if n > 1 else w[:0]
    w1 = w[1:n:4][:len(w0)]
    isdma = ((w0 & 0x80000000) != 0) & ((w1 & 0xFFFF0000) == 0x40010000)
    sp = isdma & ((((w0 >> 23) & 0xF) == 8) | (((w0 >> 19) & 0xF) == 8))
    dm = (np.nonzero(sp)[0] * 4).tolist()
    words = [int(x) for x in w[mv]] if mv else []
  except ImportError:
    ws = struct.unpack_from("<%dI" % n, text)
    mv = [i for i, x in enumerate(ws) if x & 0x7FE00000 == 0x00E00000 and 0xF800 <= (x >> 5) & 0xFFFF <= 0xF803]
    dm = [i for i in range(0, n - 1, 4) if ws[i] & 0x80000000 and ws[i + 1] & 0xFFFF0000 == 0x40010000 and 8 in ((ws[i] >> 23) & 0xF, (ws[i] >> 19) & 0xF)]
    words = [ws[i] for i in mv]
  return [f"movh r{x & 31}, {(x >> 5) & 0xFFFF:#x} (bundle {i // 4})" for i, x in zip(mv, words)] + [f"dma with a GSRAM end (bundle {i // 4})" for i in dm]


def gemm_c_base(raw, t: int) -> int:
  """k_gemm_gs's C-partials base for task `t` of a launch: the GM window's C region t (with text in GM every job but a GSRAM user's
  runs GM on, and GSRAM is empty then), else the task's TEC's 64 KiB of GSRAM. A TEC's vld / vst in GM costs what GSRAM's does
  (GM.md §6)."""
  if TEXT_GM:
    assert 0 <= t < GEMM_C_TASKS, t
    return raw.gm_window()[1] + GEMM_C_OFF + GEMM_C_REG * t
  return GSRAM_LO + 65536 * (t % 4)


# ***************** the bookkeeping *****************
class _Img:
  """One image: `buf` is THE Buffer object its Launches (and Programs) hold -- promotion and re-packing re-point it in place."""
  __slots__ = ("data", "buf", "loc", "off", "count", "seq", "home", "refs", "fresh")
  def __init__(self, data: bytes, seq: int):
    self.data, self.seq, self.count = data, seq, 0
    self.buf: Buffer|None = None
    self.loc = ""                       # "arena" | "ddr" (not promoted yet, or no room) | "fixed" (asked for DDR) | "gsram" (a GSRAM user)
    self.off = -1                       # its offset in the arena
    self.home: Buffer|None = None       # its owned DDR copy: never freed or rewritten, so an address of it stays valid for good
    self.refs: list = []                # (job, byte offset of a task TCB's spc) for every text / off job that runs it
    self.fresh = True                   # never promoted yet (a candidate of `_promote`)


class _Job:
  __slots__ = ("mode", "tcbs", "imgs", "alive")
  def __init__(self, mode: str, tcbs: Buffer, imgs: list):
    self.mode, self.tcbs, self.imgs, self.alive = mode, tcbs, imgs, True


def _retarget(b: Buffer, to: Buffer, off: int) -> None:
  b.pa, b.dev_offset, b.va, b.offset = to.pa + off, to.dev_offset + off, to.va + off, to.offset + off
  b.map_bytes = b.nbytes


class TextGM:
  """The per-device state of the policy (`RawDevice.text_gm`, None with ZHOUYI_TEXT_GM=0)."""
  def __init__(self, raw):
    self.raw = raw
    self.imgs: list[_Img] = []
    self.at: dict[int, _Img] = {}       # device address -> image (current addresses)
    self.gsram: set[int] = set()        # device addresses of GSRAM users
    self.jobs: dict[int, _Job] = {}     # head TCB PA -> the chain built there last
    self.top = 0                        # arena bytes in use
    self.in_gm = False                  # GM holds the arena as it is now (preloaded since the last change; no GM-data or unknown job since)
    self.full = False                   # an image did not fit: ranking on
    self.fresh = 0                      # images never promoted
    self.since_new = self.since_promote = 0
    self.counted, self.next_rank = 0, RANK_FIRST
    self.stats = dict(arena_images=0, arena_bytes=0, ddr_images=0, ddr_bytes=0, fixed_images=0, gsram_images=0, text_jobs=0, off_jobs=0,
                      data_jobs=0, unknown_jobs=0, promotions=0, preloads=0, preload_pages=0, preload_s=0.0, ranks=0, moved=0, data_copies=0)
    if DEBUG >= 1:
      import atexit; atexit.register(lambda: print(f"ZHOUYI_TEXT_GM at exit: {self!r}", file=sys.stderr))

  def __repr__(self):
    s = self.stats
    return (f"text in GM: arena {self.top} of {TEXT_GM_CAP} B, {s['arena_images']} images in GM, {s['ddr_images']} in DDR ({s['ddr_bytes']} B"
            f"{', the arena full' if self.full else ''}), {s['gsram_images']} GSRAM users, {s['fixed_images']} fixed in DDR; jobs text "
            f"{s['text_jobs']} off {s['off_jobs']} data {s['data_jobs']} unknown {s['unknown_jobs']}; {s['promotions']} promotions, "
            f"{s['preloads']} preloads ({s['preload_pages']} pages, {s['preload_s'] * 1e3:.1f} ms), {s['ranks']} re-packs ({s['moved']} moves)")

  def _say(self, msg: str) -> None:
    if DEBUG >= 1: print(f"ZHOUYI_TEXT_GM: {msg}", file=sys.stderr)

  # ---- placement ----
  def _arena(self) -> tuple[Buffer, int]:
    """(the window's backing allocation, the window's offset in it)."""
    pa, _ = self.raw.gm_window(); wb = self.raw._gm_win[0]
    return wb, pa - wb.pa

  def _fit(self, top: int, n: int) -> int|None:
    off = -(-top // TEXT_GM_ALIGN) * TEXT_GM_ALIGN
    return off if off + n + self.raw.LOAD_OVERHANG <= TEXT_GM_CAP else None

  def _new_home(self, im: _Img) -> Buffer:
    if im.home is None:
      im.home = self.raw.req_buf(len(im.data), MM_TEXT)
      ctypes.memmove(im.home.va, im.data, len(im.data))
    return im.home

  def place(self, text: bytes, gsram: bool|None = None, where: str = "auto") -> Buffer:
    """The text Buffer for a new image, written: its DDR home (a candidate of the next promotion into the arena), or for good in
    DDR (`where="ddr"`, or a GSRAM user)."""
    hits = gsram_refs(text) if gsram is None else (["marked by its caller"] if gsram else [])
    im = _Img(text, len(self.imgs)); self.imgs.append(im)
    n = len(text)
    im.loc = "gsram" if hits else "fixed" if where == "ddr" else "ddr"
    im.buf = self.raw.view(self._new_home(im), 0, n)
    addr = self.raw.dev_addr(im.buf)
    self.at[addr] = im
    if im.loc == "gsram":
      self.gsram.add(addr); self.stats["gsram_images"] += 1; im.fresh = False
      self._say(f"a GSRAM-using image ({n} B: {', '.join(hits[:3])}): its jobs run GM off, its text from DDR")
    elif im.loc == "fixed": self.stats["fixed_images"] += 1; im.fresh = False
    else:
      self.fresh += 1; self.since_new = 0
      self.stats["ddr_images"] += 1; self.stats["ddr_bytes"] += n
    return im.buf

  def _ddr_addr(self, spc: int) -> int:
    """For a GM-data job: an arena image's DDR home (the same bytes), else the address itself."""
    im = self.at.get(spc)
    if im is None or im.loc != "arena": return spc
    assert im.home is not None
    if _hid.LEVEL: _hid.alias_text(self.raw.dev_addr(im.home), spc)
    self.stats["data_copies"] += 1
    return self.raw.dev_addr(im.home)

  # ---- jobs ----
  def job(self, tcbs: Buffer, groups: list, gm_pa: int|None) -> tuple[list, int|None]:
    """The chain being built in `tcbs`: its mode, its groups (GM-data jobs: the DDR homes of arena text) and its GM window (None: off)."""
    if (old := self.jobs.get(tcbs.pa)) is not None: old.alive = False
    gs = [g.spc for g in groups if g.spc in self.gsram]
    if gm_pa is not None:
      if gs: raise ZhouyiError(f"a job with a GM-data member (GM on for data) holds GSRAM-using image(s) at {[hex(a) for a in gs]}: GSRAM is empty "
                               "while GM is on (GM.md §3d), so this job cannot be right; run them in separate jobs")
      if gm_pa != self.raw.gm_window()[0]: raise ZhouyiError("a GM-data job must use the device's GM window (RawDevice.gm_window)")
      self.jobs[tcbs.pa] = _Job("data", tcbs, [])
      return [g._replace(spc=self._ddr_addr(g.spc)) for g in groups], gm_pa
    mode = "off" if gs else "text"
    imgs = [self.at.get(g.spc) for g in groups]
    j = self.jobs[tcbs.pa] = _Job(mode, tcbs, imgs)
    i = 0
    for g, im in zip(groups, imgs):
      if im is not None and im.loc in ("arena", "ddr"):
        if len(im.refs) >= 64 and len(im.refs) & (len(im.refs) - 1) == 0: im.refs = [r for r in im.refs if r[0].alive]
        for k in range(len(g.tasks)): im.refs.append((j, (TASK_INDEX + i + k) * TCB_LEN + T_SPC))
      i += len(g.tasks)
    return groups, (None if gs else self.raw.gm_window()[0])

  def freed(self, b: Buffer) -> None:
    if (j := self.jobs.pop(b.pa, None)) is not None: j.alive = False

  def before_submit(self, head_pa: int) -> None:
    """Called by `submit` (nothing in flight): count the job's launches, promote / re-pack when due, and preload the arena into GM
    before a text job if GM does not hold it as it is; note a GM-data job (GM no longer holds the arena)."""
    j = self.jobs.get(head_pa)
    if j is None:                       # a chain built elsewhere: its GM use is unknown -- assume it may have overwritten GM
      self.stats["unknown_jobs"] += 1; self.in_gm = False
      return
    if j.mode == "data":
      self.stats["data_jobs"] += 1; self.in_gm = False
      return
    self.stats[j.mode + "_jobs"] += 1
    for im in j.imgs:
      if im is not None: im.count += 1
    self.since_new += 1; self.since_promote += 1
    if self.fresh and (self.since_new >= PROMOTE_QUIET or self.since_promote >= PROMOTE_MAX): self._promote()
    if self.full and RANK:
      self.counted += 1
      if self.counted >= self.next_rank:
        self.next_rank *= RANK_GROWTH
        self._rank()
    if j.mode == "text" and not self.in_gm: self.preload()

  def preload(self) -> None:
    """SYNC_TO_GM: the arena's pages [0, top) from the window's DDR backing into GM, by a one-task job (the barrier image, in DDR)."""
    pages = -(-self.top // PAGE_BYTES)
    if pages:
      import time
      t0 = time.perf_counter()
      head, first, last, tcbs = self.raw._text_sync_job(self.raw)
      grid = memoryview((ctypes.c_char * TCB_LEN).from_address(tcbs.va)).cast("B")
      gm_fields(grid, self.raw.gm_window()[0])
      struct.pack_into("<I", grid, 24, SYNC_TO_GM | pages)                # gm_rgnx_ctrl[0] = SYNC_TO_GM | pages (GM.md §3c)
      self.raw.wait_jobs((self.raw._submit(head, first, last),))
      self.stats["preloads"] += 1; self.stats["preload_pages"] += pages; self.stats["preload_s"] += time.perf_counter() - t0
    self.in_gm = True

  # ---- promotion and ranking ----
  def _move(self, plan: list) -> None:
    """Carry out (image, arena offset or None = its DDR home) moves: the bytes from the host's copies (no overlap hazard), each
    image's Buffer re-pointed, every live task TCB that held its old address patched, its hang-ID name moved. GM no longer holds
    the arena as it is (a preload before the next text job)."""
    wb, w0 = self._arena()
    for im, off in plan:
      old = self.raw.dev_addr(im.buf)
      if off is not None:
        ctypes.memmove(wb.va + w0 + off, im.data, len(im.data))
        _retarget(im.buf, wb, w0 + off); im.loc, im.off = "arena", off
      else:
        _retarget(im.buf, self._new_home(im), 0); im.loc, im.off = "ddr", -1
      new = self.raw.dev_addr(im.buf)
      if self.at.get(old) is im: del self.at[old]
      self.at[new] = im
      live = []
      for jo in im.refs:
        if not jo[0].alive: continue
        live.append(jo)
        w = ctypes.c_uint32.from_address(jo[0].tcbs.va + jo[1])
        if w.value == old: w.value = new
      im.refs = live
      if _hid.LEVEL: _hid.move_text(old, new)
    movable = [im for im in self.imgs if im.loc in ("arena", "ddr")]
    s = self.stats
    s["arena_images"] = sum(1 for im in movable if im.loc == "arena"); s["arena_bytes"] = sum(len(im.data) for im in movable if im.loc == "arena")
    s["ddr_images"] = sum(1 for im in movable if im.loc == "ddr"); s["ddr_bytes"] = sum(len(im.data) for im in movable if im.loc == "ddr")
    s["moved"] += len(plan)
    self.in_gm = False

  def _promote(self) -> None:
    """The images never promoted, into the arena's free space after the ones there (the most launched first); what does not fit
    stays in DDR and the arena is full (ranking takes over)."""
    cands = sorted((im for im in self.imgs if im.fresh), key=lambda im: (-im.count, im.seq))
    plan, top = [], self.top
    for im in cands:
      im.fresh = False
      if (off := self._fit(top, len(im.data))) is not None: plan.append((im, off)); top = off + len(im.data) + self.raw.LOAD_OVERHANG
      elif not self.full:
        self.full = True
        self._say(f"the arena is full ({top} of {TEXT_GM_CAP} B): a {len(im.data)}-B image stays in DDR; the arena is now re-packed by "
                  f"launch counts{'' if RANK else ' (off: ZHOUYI_TEXT_GM_RANK=0)'}")
    self.fresh, self.since_promote = 0, 0
    if plan:
      self.top = top
      self._move(plan); self.stats["promotions"] += 1

  def _rank(self) -> None:
    """Re-pack a full arena: the movable images by launch count (then age), hottest at the lowest address, as many as fit; the rest in
    DDR. Counts halve after each re-pack, so the ranking follows recent use."""
    movable = [im for im in self.imgs if im.loc in ("arena", "ddr")]
    want: dict[int, int] = {}
    top = 0
    for im in sorted(movable, key=lambda im: (-im.count, im.seq)):
      if (off := self._fit(top, len(im.data))) is not None: want[id(im)] = off; top = off + len(im.data) + self.raw.LOAD_OVERHANG
    plan = [(im, want.get(id(im))) for im in movable if want.get(id(im), -1) != (im.off if im.loc == "arena" else -1)]
    for im in movable: im.count //= 2
    if not plan: return
    plan.sort(key=lambda p: p[1] is not None)                             # evictions first (their homes exist before the arena changes)
    self.top = top
    self._move(plan); self.stats["ranks"] += 1
    self._say(f"re-packed the arena by launch count: {len(plan)} images moved, {self.stats['arena_images']} in GM ({top} B), "
              f"{self.stats['ddr_images']} in DDR")
