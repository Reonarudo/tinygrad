"""TEC resources, checked: the four sync flags, and the LSRAM / GSRAM / descriptor-table layouts.

Silicon facts (measured on the device):

- A TEC has FOUR sync flags, 0-3, private to the TEC. A `dma`'s sync word selects its flag by `lo & 0x1F`; 31 is a
  broadcast (raises every flag); **4-30 silently land on flag 0** (not N mod 4). The simulator models eight independent
  flags, so a kernel on flag 4 passes the simulator and races on the part.
- `wfe <mask reg>, 1` waits for the flags whose mask bit is CLEAR (the `~(1 << flag)` convention); **a flag nobody raised
  returns at once** -- a silent race, not a hang. Mode 2 waits until no flag is busy, whatever the mask. Mode 0 waits on
  another event class (the task epilogue's `wfe r0, 0`). **Modes 3-31 raise exception vector 2.**
- A 16-byte vector load reads 16 B past its end; DMA descriptors are 32-B aligned; the top 64 B of LSRAM hold the
  cycle stamps; a GSRAM overrun fails silently.

This module gives the kernels one way to emit a request's sync word and a wait (`sync_flag`, `wait_flags`), a lint that
refuses an image breaking the flag rules (`lint_sync`, run by `image.text_only` / `image.build_encoded`), and arenas that
place buffers with the capacity / alignment / overhang checks done once (`Arena`, `Lsram`, `GsramWindow`, `DescTable`).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from tinygrad.runtime.support.zhouyi import ZhouyiError
from . import enc as E

FLAGS = (0, 1, 2, 3)
BROADCAST = 31
VLD_OVERHANG = 16          # a 16-B vector load reads this far past its end (RawDevice.LOAD_OVERHANG)


def _bundle(asm, op): return [(asm, op)]


# ***************** the sync flags *****************
def sync_word(n: int) -> int:
  """The request's sync word for flag `n` (0-3) or the broadcast (31): `(n << 8) | n`, the vendor idiom (the high byte is
  inert)."""
  if n not in FLAGS and n != BROADCAST:
    raise ZhouyiError(f"sync flag {n}: a TEC has flags 0-3 (31 = broadcast); selectors 4-30 silently land on flag 0")
  return (n << 8) | n


def sync_flag(n: int, reg: str = "r7") -> list:
  """One bundle: the sync word of flag `n` into `reg` (the `dma`'s rsync operand)."""
  w = sync_word(n)
  return [_bundle("mov %s, %d" % (reg, w), E.mov(reg, w))]


def wait_flags(*ns: int, mode: int = 1, reg: str = "r7") -> list:
  """The bundles of a wait. Mode 1 (default): until every flag in `ns` (0-3) is idle -- the mask `~OR(1 << n)` built in
  `reg` by one `sub reg, zero, m + 1`, then `wfe reg, 1`. Mode 2: until no flag is busy (a barrier; `ns` must be empty,
  the mask is ignored by the part, `zero` is passed)."""
  if mode == 2:
    if ns: raise ZhouyiError("wait_flags(mode=2) waits on every flag; do not name any")
    return [_bundle("wfe zero, 2", E.wfe("zero", 2))]
  if mode != 1: raise ZhouyiError(f"wfe mode {mode}: only 1 (named flags) and 2 (all idle) are allowed here; 3-31 raise exception vector 2")
  if not ns: raise ZhouyiError("wait_flags(mode=1) names no flag: such a wait returns at once")
  bad = [n for n in ns if n not in FLAGS]
  if bad: raise ZhouyiError(f"wait on flag(s) {bad}: a TEC has flags 0-3; a mode-1 wait on any other returns at once")
  m = 0
  for n in ns: m |= 1 << n
  return [_bundle("sub %s, zero, %d" % (reg, m + 1), E.sub(reg, "zero", m + 1)), _bundle("wfe %s, 1" % reg, E.wfe(reg, 1))]


# ***************** the image lint *****************
_NO_DEST = {"dma", "wfe", "st", "sth", "stb", "vst", "vsts", "cbnz", "cbz", "b", "br", "bl", "exit", "nop", "mtctrl0", "mtctrl1",
            "break", "aiff", "sync", "fence", "barrier"}
_INT = re.compile(r"^-?(0x[0-9a-fA-F]+|\d+)$")


def _ops(asm: str) -> tuple[str, list[str]]:
  asm = asm.split("//")[0].split(";")[0].strip()
  if not asm: return "", []
  parts = asm.split(None, 1)
  return parts[0].lower(), ([x.strip() for x in parts[1].split(",")] if len(parts) > 1 else [])


def _imm(s: str):
  return int(s, 0) if _INT.match(s) else None


@dataclass
class SyncReport:
  raised: set = field(default_factory=set)        # flags some request raises (31 -> all four)
  waited: set = field(default_factory=set)        # flags some mode-1 wait names
  unknown_requests: int = 0                       # requests whose sync word could not be traced
  unknown_waits: int = 0


def lint_sync(bundles, labels=(), name: str = "kernel") -> SyncReport:
  """Refuse an image that breaks the flag rules, from its assembly text (`bundles` = lists of (asm, op) pairs):

  - a `wfe` of mode 3-31;
  - a mode-1 `wfe` whose mask register holds a traced value naming a flag above 3, or naming none;
  - a `dma` whose sync register holds a traced value selecting 4-30;
  - a mode-1 wait on a flag that no request in the image raises (checked only when every request's word was traced).

  Register values are traced through `mov` / `movh` / `add|sub rd, zero, imm` within straight-line code: every other
  write to a register, and every branch target, forgets what is known. An untraced operand is not checked."""
  known: dict[str, int] = {"zero": 0, "r31": 0}
  rep = SyncReport(); errs = []; targets = set(labels)
  for i, b in enumerate(bundles):
    if i in targets: known = {"zero": 0, "r31": 0}
    writes = {}
    for asm, _op in b:
      mn, ops = _ops(asm)
      if not mn: continue
      if mn == "wfe":
        mode = _imm(ops[1]) if len(ops) > 1 else 0
        if mode is None or mode > 2: errs.append(f"bundle {i}: `{asm}`: wfe mode {ops[1] if len(ops) > 1 else '?'} (modes 3-31 raise exception vector 2)")
        elif mode == 1:
          v = known.get(ops[0])
          if v is None: rep.unknown_waits += 1
          else:
            named = [f for f in range(32) if not (v >> f) & 1]
            if not named: errs.append(f"bundle {i}: `{asm}`: the mask {v & 0xFFFFFFFF:#x} names no flag: the wait returns at once")
            elif any(f > 3 for f in named):
              errs.append(f"bundle {i}: `{asm}`: the mask {v & 0xFFFFFFFF:#x} names flag(s) {[f for f in named if f > 3][:4]}: a TEC has 0-3, and a wait on another returns at once (Sync.md 3)")
            else: rep.waited |= set(named)
      elif mn in ("dma", "aiff"):                 # both raise the flag their sync word selects
        v = known.get(ops[1]) if len(ops) > 1 else None
        if v is None: rep.unknown_requests += 1
        else:
          sel = v & 0x1F
          if 4 <= sel <= 30: errs.append(f"bundle {i}: `{asm}`: sync word {v:#x} selects {sel}, which silently lands on flag 0")
          else: rep.raised |= set(FLAGS) if sel == BROADCAST else {sel}
      if mn in _NO_DEST or not ops or not re.match(r"^(r\d+|[a-z]+p|fp|sp|ra|zero)$", ops[0]): continue
      rd = ops[0]
      if mn == "mov" and len(ops) == 2 and _imm(ops[1]) is not None: writes[rd] = _imm(ops[1]) & 0xFFFFFFFF
      elif mn == "movh" and len(ops) == 2 and _imm(ops[1]) is not None and rd in known:
        writes[rd] = (known[rd] & 0xFFFF) | ((_imm(ops[1]) & 0xFFFF) << 16)
      elif mn in ("add", "sub") and len(ops) == 3 and ops[1] in ("zero", "r31") and _imm(ops[2]) is not None:
        writes[rd] = (_imm(ops[2]) if mn == "add" else -_imm(ops[2])) & 0xFFFFFFFF
      else: writes[rd] = None
    for rd, v in writes.items():                 # a bundle's writes land after its reads
      if rd in ("zero", "r31"): continue
      if v is None: known.pop(rd, None)
      else: known[rd] = v
  if not rep.unknown_requests:
    for f in sorted(rep.waited - rep.raised):
      errs.append(f"a mode-1 wait names flag {f}, which no request in the image raises: that wait returns at once")
  if errs: raise ZhouyiError(f"{name}: the sync-flag lint refused the image:\n  " + "\n  ".join(errs))
  return rep


# ***************** layouts *****************
@dataclass
class Region:
  name: str
  off: int
  nbytes: int
  vld_tail: bool
  @property
  def end(self): return self.off + self.nbytes


class Arena:
  """Buffers placed in a fixed-size memory, checked once: capacity (`size - reserve_top`), alignment, no overlap, and for a
  region whose last bytes a 16-B `vld` reads (`vld_tail`) the VLD_OVERHANG bytes after it still inside the memory (the
  over-read lands in the next region or the reserved top -- harmless reads -- never past the end). `alloc` places at the
  first aligned offset after the last region (or at `at=`); `dump()` prints the map."""
  def __init__(self, name: str, size: int, reserve_top: int = 0, align: int = 32):
    self.name, self.size, self.reserve_top, self.align = name, size, reserve_top, align
    self.limit = size - reserve_top; self.regions: list[Region] = []; self.cursor = 0
  def alloc(self, name: str, nbytes: int, align: int | None = None, vld_tail: bool = False, at: int | None = None) -> int:
    a = self.align if align is None else align
    if nbytes < 0: raise ZhouyiError(f"{self.name}: {name}: {nbytes} B")
    off = (-(-self.cursor // a) * a) if at is None else at
    if off % a: raise ZhouyiError(f"{self.name}: {name} at {off:#x} is not {a}-B aligned")
    if off + nbytes > self.limit:
      raise ZhouyiError(f"{self.name}: {name} ({nbytes} B at {off:#x}) ends at {off + nbytes:#x}, past the usable {self.limit:#x} "
                        f"({self.size} B - {self.reserve_top} B reserved)\n{self.dump()}")
    if vld_tail and off + nbytes + VLD_OVERHANG > self.size:
      raise ZhouyiError(f"{self.name}: {name} ends {self.size - off - nbytes} B before the end of the memory; a 16-B vld of its last "
                        f"bytes reads {VLD_OVERHANG} B past it\n{self.dump()}")
    for r in self.regions:
      if off < r.end and r.off < off + nbytes: raise ZhouyiError(f"{self.name}: {name} [{off:#x}, {off + nbytes:#x}) overlaps {r.name} [{r.off:#x}, {r.end:#x})")
    self.regions.append(Region(name, off, nbytes, vld_tail)); self.cursor = max(self.cursor, off + nbytes)
    return off
  def __getitem__(self, name: str) -> int: return next(r.off for r in self.regions if r.name == name)
  @property
  def used(self) -> int: return self.cursor
  def dump(self) -> str:
    lines = [f"{self.name}: {self.size} B, {self.reserve_top} B reserved at the top, {self.limit - self.cursor} B free"]
    for r in sorted(self.regions, key=lambda r: r.off):
      lines.append(f"  {r.off:#07x} .. {r.end:#07x}  {r.nbytes:6d} B  {r.name}{'  (vld tail)' if r.vld_tail else ''}")
    return "\n".join(lines)


class Lsram(Arena):
  """A TEC's 32 KiB of LSRAM; the top 64 B hold the cycle stamps (`GB_STAMP`)."""
  def __init__(self, name: str = "", size: int = 32768, reserve_top: int = 64): super().__init__(f"LSRAM {name}".strip(), size, reserve_top)


class GsramWindow(Arena):
  """A TEC's 64 KiB GSRAM window (0xF8000000 + 64 KiB x the TEC's index in its core). An overrun of the window writes the
  neighbour's silently, so the window is the capacity."""
  def __init__(self, name: str = "", size: int = 65536): super().__init__(f"GSRAM window {name}".strip(), size, 0)


class DescTable(Arena):
  """A DMA descriptor table in DDR: 32-B aligned slots (a descriptor is six words; `DESC(d, i)` / `r3 + 32 i` address them)."""
  def __init__(self, name: str = "descriptors", nslots: int = 16, slot: int = 32):
    super().__init__(name, nslots * slot, 0, align=32); self.slot = slot
  def slot_at(self, name: str, byte_off: int) -> int: return self.alloc(name, 24, at=byte_off)
