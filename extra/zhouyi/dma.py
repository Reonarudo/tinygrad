"""The TEC's DMA engine: descriptors, sync flags, and the scratchpad map.

Everything a staged kernel needs to move data DDR->LSRAM ahead of the arithmetic that reads it.
The module knows nothing about uops: it emits C fragments and integers, so it serves both
hand-written kernels and the renderer's `pre_matcher`.

Note: GSRAM is fine to fill (DMA rate matches LSRAM) but slow to compute from: ~7x the cycles per
`vld`/`vst` of LSRAM, ~10x the load-to-use latency, and no bank structure to exploit.
"""
from __future__ import annotations

from typing import NamedTuple

from tinygrad.runtime.support.zhouyi import ZhouyiError

# ---------------------------------------------------------------- the part
# Address-space codes for the DMA builtin, and the flat scratchpad bases.
SPACE_DDR, SPACE_LSRAM, SPACE_GSRAM = 0, 4, 8
LSRAM_BASE, LSRAM_SIZE = 0xFA000000, 32 * 1024    # per TEC, private: the staging target
# GSRAM aperture is 256 KiB (the GM control register's 4 MiB is a different object).
# Note: past the end nothing faults: reads return garbage that varies between jobs, and a DMA
# into it reports success while the data is lost.
GSRAM_BASE, GSRAM_SIZE = 0xF8000000, 256 * 1024

# Four sync flags. Out-of-range flags do not fault but alias onto flag 0 (see check_flag).
SYNC_FLAGS = 4

# `init_dma_desc(desc, width, size)` (the vendor's kernel C library): six words; the outer pairs are the
# ext and int configurations, the middle pair holds the stride high bits (zero when contiguous).
DESC_WORDS = 6
# 64 B per slot so two TECs' descriptors never share a cache line.
DESC_SLOT = 64
# Ints reserved for a descriptor on a task stack (stack_slot / DEFINE_REG). The engine fetches
# descriptors at line granularity, so a 24 B `int[6]` near the frame top makes it read past the
# task's initial `sp` (an SMMU fault if that page is unmapped). 128 B covers any line-aligned
# window over the six words ([base-60, base+84)); the extra words are never interpreted.
DESC_REG_WORDS = 32


SYNC_BROADCAST = 31        # the selector value that raises every flag


def check_flag(flag: int) -> int:
  """Validate a sync flag (0-3, or SYNC_BROADCAST).

  The selector is the low 5 bits of the sync word's low half: 0-3 are flags, 31 broadcasts, and
  4-30 all fall back to flag 0 (not `flag % 4`). An out-of-range flag is a race: the request
  raises flag 0, the wait on the named flag returns immediately, and the transfer is still in
  flight."""
  if flag == SYNC_BROADCAST:
    return flag
  if not 0 <= flag < SYNC_FLAGS:
    raise ZhouyiError(
        f"sync flag {flag} is outside the TEC's {SYNC_FLAGS} (0-{SYNC_FLAGS - 1}). On silicon the "
        f"selector is the low 5 bits and {flag} & 0x1f = {flag & 0x1f}, which "
        + (f"is the BROADCAST code {SYNC_BROADCAST} -- pass {SYNC_BROADCAST} explicitly if that is "
           "what you meant"
           if (flag & 0x1f) == SYNC_BROADCAST else
           "falls back to FLAG 0. The request would still run; the wait would name a flag nothing "
           "raised and return immediately, leaving you reading the destination while the transfer "
           "is still in flight -- a wrong answer with a DONE job behind it, not a fault")
        )
  return flag


def best_width(size: int) -> int:
  """The vendor's `DMA_GET_BEST_WIDTH` rule."""
  if size < 32: return size
  if size < 512: return 32
  return min(size, 32768)


# The descriptor's field widths: 16-bit width, 24-bit stride split across two words, 24-bit
# transfer size. Checked rather than masked, since the packing `&` would truncate silently.
DMA_MAX_WIDTH, DMA_MAX_STRIDE, DMA_MAX_TRANS_SIZE = 0xFFFF, 0xFFFFFF, 0xFFFFFF


def desc_words(size: int, width: int | None = None, ext_stride: int | None = None,
               int_stride: int | None = None) -> tuple[int, ...]:
  """The six words `init_dma_desc(desc, width, ext_stride, int_stride, trans_size)` writes.

  Strides default to `width` (contiguous rows). With a stride > width the engine gathers/scatters
  `size/width` sub-blocks at that pitch in one request, at no bandwidth cost vs contiguous.
  `num_of_trans` stays 1; the sub-block count is implicit (`size/width`), so a strided `size`
  must be a multiple of `width`."""
  if size <= 0: raise ZhouyiError(f"a DMA of {size} B has nothing to transfer")
  w = best_width(size) if width is None else width
  es, ins = (w if ext_stride is None else ext_stride), (w if int_stride is None else int_stride)
  if not 0 < w <= DMA_MAX_WIDTH:
    raise ZhouyiError(f"a {w} B DMA width is outside the vendor's {DMA_MAX_WIDTH} B field and would be truncated")
  if not (0 < es <= DMA_MAX_STRIDE and 0 < ins <= DMA_MAX_STRIDE):
    raise ZhouyiError(f"DMA strides ({es}, {ins}) B are outside the vendor's {DMA_MAX_STRIDE} B field "
                    "and would be truncated into a transfer from an address nobody chose")
  if size > DMA_MAX_TRANS_SIZE:
    raise ZhouyiError(f"a {size} B DMA is outside the vendor's {DMA_MAX_TRANS_SIZE} B transfer-size field")
  # Strided only: contiguous transfers (e.g. best_width's 32 B for 32..512 B) may end in a short
  # sub-block, which the engine handles.
  if (es != w or ins != w) and size % w:
    raise ZhouyiError(f"a {size} B strided DMA does not divide into {w} B sub-blocks — the engine derives "
                    "their count as trans_size/width, so a remainder is a sub-block nobody sized")
  nt = (1 << 24) | size
  return ((w << 16) | (es & 0xFFFF), nt, (es & 0xFFFF0000) << 8,
          (ins & 0xFFFF0000) << 8, (w << 16) | (ins & 0xFFFF), nt)


# ---------------------------------------------------------------- the other three modes
# The descriptor layout of the vendor's `init_dma_desc`; field meanings for transpose/upsample are the
# vendor's claim, not verified on hardware.
DMA_MAX_TRANS_NUM = 0xFF     # 8 bit: `num_of_trans`
DMA_MAX_GAP = 0xFFFFFF       # 24 bit: the low bits of descriptor words 2 and 3

# The `enum DmaMode` values, so callers need not import `enc`.
MODE_DIRECT, MODE_TRANSPOSE, MODE_UPSAMPLE, MODE_MEMSET = 0, 1, 2, 3


def desc_words_full(ext_width: int, int_width: int, ext_stride: int, int_stride: int,
                    ext_gap: int, int_gap: int, ext_num_of_trans: int, int_num_of_trans: int,
                    ext_trans_size: int, int_trans_size: int) -> tuple[int, ...]:
  """The ten-argument `init_dma_desc`, field for field.

  Needed for asymmetric ext/int geometry (transpose, upsample, memset). The sides are ext/int,
  not src/dst: the instruction's `dir` operand (`kExt2Int`/`kInt2Ext`) decides which is read."""
  for n, v, w in (("ext_width", ext_width, DMA_MAX_WIDTH), ("int_width", int_width, DMA_MAX_WIDTH),
                  ("ext_stride", ext_stride, DMA_MAX_STRIDE), ("int_stride", int_stride, DMA_MAX_STRIDE),
                  ("ext_gap", ext_gap, DMA_MAX_GAP), ("int_gap", int_gap, DMA_MAX_GAP),
                  ("ext_num_of_trans", ext_num_of_trans, DMA_MAX_TRANS_NUM),
                  ("int_num_of_trans", int_num_of_trans, DMA_MAX_TRANS_NUM),
                  ("ext_trans_size", ext_trans_size, DMA_MAX_TRANS_SIZE),
                  ("int_trans_size", int_trans_size, DMA_MAX_TRANS_SIZE)):
    if not 0 <= v <= w:
      raise ZhouyiError(f"{n}={v} is outside the vendor's {w} field and the `&` that packs it "
                      "truncates silently — a truncated descriptor is a transfer from an address "
                      "nobody chose")
  return ((ext_width << 16) | (ext_stride & 0xFFFF),
          (ext_num_of_trans << 24) | ext_trans_size,
          ((ext_stride & 0xFFFF0000) << 8) | ext_gap,
          ((int_stride & 0xFFFF0000) << 8) | int_gap,
          (int_width << 16) | (int_stride & 0xFFFF),
          (int_num_of_trans << 24) | int_trans_size)


def transpose_desc(col: int, row: int, col_stride: int, row_stride: int,
                   src_gap: int = 0, dst_gap: int = 0, trans_num: int = 1) -> tuple[int, ...]:
  """`DMATranspose`'s descriptor: mode 1, `kExt2Int` (src external, dst internal).

  All arguments are in bytes. The element size comes from the instruction's `data_unit` operand
  (e.g. `du=kHalf` for 2-byte elements); it must match or bytes within elements get transposed."""
  return desc_words_full(col, row, col_stride, row_stride, src_gap, dst_gap,
                         trans_num, trans_num, 0, 0)


def upsample_desc(h_scale: int, w_scale: int, c: int, w: int, src_c_stride: int,
                  dst_c_stride: int, dst_w_stride: int) -> tuple[int, ...]:
  """`GEN_DMA_UPSAMPLE_EXT2INT`'s descriptor: mode 2, src external, dst internal.

  Note: which axis `h_scale`/`w_scale` actually scale is unverified; the vendor header's docs
  contradict the names. Names follow the vendor's parameters."""
  return desc_words_full(c, 0, src_c_stride, dst_c_stride, 0, dst_w_stride * dst_c_stride,
                         w_scale, h_scale, c * w, 0)


def memset_desc(trans_size: int, width: int | None = None) -> tuple[int, ...]:
  """`MemsetSRAM`'s descriptor: mode 3, `kExt2Int` into the internal side.

  The fill value is not in the descriptor; it travels in the instruction's external address
  operand, so the external half of the descriptor is zero. Note: `width` defaults to
  `best_width(trans_size)`, whereas the vendor macro rounds non-multiples of 512 down."""
  w = best_width(trans_size) if width is None else width
  return desc_words_full(0, w, 0, w, 0, 0, 1, 1, 0, trans_size)


def wfe_mask(flag: int) -> int:
  """The `__builtin_aipu_wfe_inter` mask: every flag bit but this one, as a signed 32-bit int
  (the vendor clang prototype takes signed; an unsigned literal selects a different overload)."""
  m = ~(1 << check_flag(flag)) & 0xFFFFFFFF
  return m - (1 << 32) if m >> 31 else m


# ---------------------------------------------------------------- where the descriptor lives
class DescriptorSlot(NamedTuple):
  """Where the descriptor words live while the DMA engine reads them.

  Stack vs DDR placement has no measurable performance difference; `stack_slot` is preferred
  because it needs no extra buffer or kernel argument.

  `decl` goes at the top of the kernel body, `words` is the `int[]` lvalue the words are written
  through, and `addr` is the plain `int` passed to the builtin. A `pre_matcher` may supply `{n}`
  placeholders for all three and let `Ops.CUSTOM` substitute them."""
  decl: str
  words: str
  addr: str


def stack_slot(name: str = "aipu_dma_desc") -> DescriptorSlot:
  """The descriptor as a local array: the default placement, per-TEC by construction.

  Sized `DESC_REG_WORDS`, not `DESC_WORDS`, so the engine's line-granular fetch stays inside the
  array rather than reading past the task's initial `sp`."""
  return DescriptorSlot(decl=f"int {name}[{DESC_REG_WORDS}];", words=name, addr=f"(int){name}")


def ddr_slot(base: str, tec: str = "core_id", off: int = 0) -> DescriptorSlot:
  """A per-TEC `DESC_SLOT`-byte slot in DDR at `base + off + tec*DESC_SLOT`.

  Note: `base + off` must clear everything the kernel writes; a descriptor overwritten by the
  kernel's own stores silently corrupts the transfer."""
  addr = f"((int){base} + {off} + ({tec})*{DESC_SLOT})"
  return DescriptorSlot(decl="", words=f"((__global int*){addr})", addr=addr)


# ---------------------------------------------------------------- the C the kernel gets
def init_c(slot: DescriptorSlot, size: int, width: int | None = None, ext_stride: int | None = None,
           int_stride: int | None = None) -> str:
  """C that writes the six words. Emitted once per descriptor shape and hoisted above the tile
  loop, since every tile of a sweep (strided or not) has the same size and pitch."""
  return " ".join(f"{slot.words}[{i}]={w};"
                  for i, w in enumerate(desc_words(size, width, ext_stride, int_stride)))


def request_c(slot: DescriptorSlot, dst: str, src: str, flag: int | str = 0,
              space: int = SPACE_LSRAM) -> str:
  """One `__builtin_aipu_dma_inter`: request `dst <- src` and raise `flag` when it lands.

  `flag` may be an int (checked) or a C expression (unchecked); the builtin accepts a runtime
  flag and destination, so a tile loop can double-buffer on `i & 1` without unrolling.
  Arguments: `(0, flag<<8|flag, 0, space, 0, 1, 0, desc, dst, src)`, per the vendor DSL headers."""
  fl = f"((({flag})<<8)|({flag}))" if isinstance(flag, str) else (check_flag(flag) << 8) | flag
  return f"__builtin_aipu_dma_inter(0, {fl}, 0, {space}, 0, 1, 0, {slot.addr}, {dst}, {src});"


def wait_c(flag: int | str) -> str:
  """C that blocks until `flag` is raised. An int flag folds the mask here; an expression flag
  computes `~(1 << f)` in C."""
  return (f"__builtin_aipu_wfe_inter(~(1 << ({flag})), 1);" if isinstance(flag, str)
          else f"__builtin_aipu_wfe_inter({wfe_mask(flag)}, 1);")


# ---------------------------------------------------------------- what a sweep actually reads
def halves(tile: int, reserve: int = 0) -> tuple[int, int]:
  """The two double-buffered halves' LSRAM addresses, with `reserve` bytes held back at the base.

  Raises if `reserve + 2*tile` exceeds LSRAM; an overrun would wrap silently rather than fault."""
  if reserve + 2*tile > LSRAM_SIZE:
    raise ZhouyiError(f"two {tile} B halves plus {reserve} B reserved need {reserve + 2*tile} B of a "
                    f"{LSRAM_SIZE} B LSRAM — the second half would start past the scratchpad and the "
                    "address would wrap rather than fault")
  return LSRAM_BASE + reserve, LSRAM_BASE + reserve + tile


def clamp_size(i: int, nbytes: int, tile: int) -> int:
  """The size of the `i`-th transfer of a sweep so nothing is read past `nbytes`.

  Returns 0 for a tile wholly past the end; skip that request (`desc_words` refuses 0 B, whose
  engine behaviour is undefined)."""
  return max(0, min(tile, nbytes - i*tile))
