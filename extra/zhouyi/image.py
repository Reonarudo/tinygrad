"""Image writer for hand-written kernels: one loadable NPU text image (no linker). The body is (asm text, encoder op)
pairs; `enc.py` emits the bytes (`text_only`), and `build_encoded` also has the installed assembler (`btaipuas`) assemble the
same text so the whole image can be byte-diffed. (Codegen kernels are built by `tinygrad/runtime/support/compiler_zhouyi.py`.)

Prologue/epilogue are defined once as `(asm, op)` pairs; the assembly text is derived from them.

Layout:

    _entry:  { b .Lmain; }        <- exception-vector slot 0 (= the start PC)
             { b .Lhang; } x15    <- every other vector spins
    .Lhang:  { b .Lhang; }
    .Lmain:  <FP enable, if the kernel uses vector arithmetic>
             <kernel body>
             <task epilogue>

Unused vectors must spin: if they held `exit`, a faulting job would report a clean DONE.
"""
import struct

from tinygrad.runtime.support.zhouyi import ZhouyiError
from . import enc as E
from . import toolchain as TC
from tinygrad.runtime.support.zhouyi.launch import CYCLE_STAMP_OFF

VECTORS = 16           # exception-vector slots
HANG_AT = VECTORS      # bundle index of .Lhang
MAIN_AT = VECTORS + 1


def I(asm, op): return (asm, op)
def BUNDLE(*ins): return list(ins)


# ***************** prologue / epilogue, defined once *****************
def fp_enable(scratch="r9"):
  """Enable the FP unit. Without it any vector arithmetic (fp32 `fma`, matrix `mml`/`mma`) traps to
  exception vector 3, which presents as a hang; vector load/store and scalar integer code do not need it.

  The sequence the vendor compiler emits, with r9 as scratch instead of its r24 so `tcbp` survives."""
  return [BUNDLE(I(f"mfctrl1 {scratch}, 2", E.mfctrl1(scratch, 2))),
          BUNDLE(I(f"bfs {scratch}, {scratch}, 31, 0", E.bfs(scratch, scratch, 31, 0))),
          BUNDLE(I(f"mtctrl1 {scratch}, 2", E.mtctrl1(scratch, 2))),
          BUNDLE(I(f"mfctrl1 {scratch}, 0x10", E.mfctrl1(scratch, 0x10))),
          BUNDLE(I(f"bfs {scratch}, {scratch}, 31, 0", E.bfs(scratch, scratch, 31, 0))),
          BUNDLE(I(f"mtctrl1 {scratch}, 0x10", E.mtctrl1(scratch, 0x10)))]


def prologue(nargs, fp=True):
  """The vendor's task prologue (clang -O2). `movh p7, 0` arms the all-lanes predicate used by
  `fma ...,p7.w`; kernel args are a flat u32 array at `pp` (`ld r0,[pp+0]`, `ld r1,[pp+4]`, ...).

  `fp=False` for AIFF kernels: AIFF is an integer fixed-function engine and needs no enable."""
  bs = (fp_enable() if fp else []) + [
    BUNDLE(I("xor sdp, sdp, sdp", E.xor("sdp", "sdp", "sdp")),
           I("mov lp, -32768", E.mov("lp", -32768)),
           I("movh p7, 0", E.movh("p7", 0)),
           I("mfctrl0 tcbp, 20", E.mfctrl0("tcbp", 20))),
    BUNDLE(I("movh lp, -1536", E.movh("lp", -1536)),
           I("bfs sdp, sdp, 27, 4", E.bfs("sdp", "sdp", 27, 4)),
           I("ld sp, [tcbp+52]", E.ld("sp", "tcbp", 52))),
    BUNDLE(I("ld pp, [tcbp+56]", E.ld("pp", "tcbp", 56)))]
  for i in range(0, nargs, 2):
    ins = [I("ld r%d, [pp+%d]" % (i, i*4), E.ld("r%d" % i, "pp", i*4))]
    if i + 1 < nargs:
      ins.append(I("ld r%d, [pp+%d]" % (i+1, (i+1)*4), E.ld("r%d" % (i+1), "pp", (i+1)*4)))
    bs.append(BUNDLE(*ins))
  bs.append(BUNDLE(I("ld dp, [tcbp+60]", E.ld("dp", "tcbp", 60)),
                   I("ld cp, [tcbp+64]", E.ld("cp", "tcbp", 64))))
  return bs


EPILOGUE_FLUSH_LABEL_AT = 6   # index within epilogue() of the .Lflush target


def epilogue(label=".Lflush"):
  """Task epilogue: a task terminates rather than returns. Byte-identical to the vendor's.

      ctrl0x82 |= 4         -- request dcache writeback
      ctrl0x90 |= 0x017D<<16
      poll ctrl0x82 & 4     -- drain
      wfe / exit

  Note: without the writeback the job still reports DONE but stores never reach DDR."""
  return [BUNDLE(I("mfctrl0 r2, 0x82", E.mfctrl0("r2", 0x82))),
          BUNDLE(I("orl r2, r2, 4", E.orl("r2", "r2", 4))),
          BUNDLE(I("mtctrl0 r2, 0x82", E.mtctrl0("r2", 0x82))),
          BUNDLE(I("mfctrl0 r3, 0x90", E.mfctrl0("r3", 0x90))),
          BUNDLE(I("movh r3, 381", E.movh("r3", 381))),
          BUNDLE(I("mtctrl0 r3, 0x90", E.mtctrl0("r3", 0x90))),
          # label goes here
          BUNDLE(I("mfctrl0 r2, 0x82", E.mfctrl0("r2", 0x82))),
          BUNDLE(I("andl r2, r2, 4", E.andl("r2", "r2", 4))),
          BUNDLE(I(f"cbnz r2, {label}", E.cbnz("r2", -2))),
          BUNDLE(I("wfe r0, 0", E.wfe("r0", 0))),
          BUNDLE(I("exit", E.exit_()))]


# The vendor's 11 epilogue words; checked against our encoder at import time.
VENDOR_EPILOGUE = [0x01041042, 0x40109042, 0x01001042, 0x01041203, 0x00E02FA3, 0x01001203,
                   0x01041042, 0x40101042, 0x017FFF82, 0x014C0000, 0x014F8000]
_OURS = [op[1] for b in epilogue() for _asm, op in b]
assert _OURS == VENDOR_EPILOGUE, ("the encoder no longer reproduces the vendor task epilogue:\n"
                                  f"  ours   {[hex(x) for x in _OURS]}\n"
                                  f"  vendor {[hex(x) for x in VENDOR_EPILOGUE]}")


# ***************** the cycle stamps, defined once *****************
def cycle_stamp(slot:int):
  """Store `ctrl0[0xd1]` (TEC cycle counter) into word `slot` of the task's param tail
  (`launch.CYCLE_STAMP_OFF`). Slot 0 goes after the FP enable, slot 1 before each task epilogue
  (whose writeback lands it in DDR).

  Uses r2/r3, which the vendor epilogue clobbers anyway. `tcbp` is re-read since the body may not
  preserve it. `add`'s immediate is 10-bit unsigned, so the offset goes through `mov`."""
  return [BUNDLE(I("mfctrl0 r2, 20", E.mfctrl0("r2", 20))),                    # tcbp, as a GPR
          BUNDLE(I("ld r2, [r2+56]", E.ld("r2", "r2", 56))),                    # pp
          BUNDLE(I(f"mov r3, {CYCLE_STAMP_OFF}", E.mov("r3", CYCLE_STAMP_OFF))),
          BUNDLE(I("add r2, r2, r3", E.add("r2", "r2", "r3"))),                 # pp + CYCLE_STAMP_OFF
          BUNDLE(I("mfctrl0 r3, 209", E.mfctrl0("r3", 0xd1))),                  # the cycle counter
          BUNDLE(I(f"st r3, [r2+{4*slot}]", E.st("r3", "r2", 4*slot)))]


# `btaipuas` output for `cycle_stamp(0)`, checked against our encoder at import time. The last word
# encodes the slot's byte offset in bits [18:10].
VENDOR_CYCLE_STAMP = [0x01040282, 0x7010E042, 0x00C1FF03, 0x40000C42, 0x01041A23, 0x30100043]
for _slot in (0, 1):
  _ours = [op[1] for b in cycle_stamp(_slot) for _asm, op in b]
  _want = VENDOR_CYCLE_STAMP[:-1] + [VENDOR_CYCLE_STAMP[-1] | (4*_slot) << 10]
  assert _ours == _want, ("the encoder no longer reproduces the probed cycle stamp:\n"
                          f"  ours   {[hex(x) for x in _ours]}\n  vendor {[hex(x) for x in _want]}")


# ***************** the image *****************
def vector_table():
  """Slot 0 branches to main; every other slot spins forever."""
  bs = [BUNDLE(I("b .Lmain", E.b(MAIN_AT - 0)))]
  bs += [BUNDLE(I("b .Lhang", E.b(HANG_AT - i))) for i in range(1, VECTORS)]
  bs.append(BUNDLE(I("b .Lhang", E.b(0))))       # .Lhang: spin
  return bs


B_BASE = 0x01800000
CBNZ_BASE, CBNZ_SHIFT, CBNZ_BITS = 0x01700000, 6, 14


def _sx(v:int, bits:int) -> int: return v - (1 << bits) if v >= 1 << (bits-1) else v


def branch_target(asm:str, op, at:int) -> int|None:
  """Bundle index a branch at bundle `at` lands on, decoded from the emitted word; None if not a branch.

  Labels in the oracle's assembly are derived from these targets rather than placed by hand."""
  # `aiff` and `dma` are two-word instructions (tuple payload) and never branches.
  if not isinstance(word := op[1], int): return None
  if asm.split(None, 1)[0] == "b" and word & 0xFFF00000 == B_BASE:
    return at + _sx(word & 0xFFFF, 16) // 2      # `b`'s immediate unit is 8 bytes, a bundle is 2
  if word & 0xFFF00000 == CBNZ_BASE:
    return at + _sx((word >> CBNZ_SHIFT) & ((1 << CBNZ_BITS) - 1), CBNZ_BITS)
  return None


def _label_map(bundles) -> dict[int, str]:
  """target bundle -> label, named by the first branch to it; a name reused for a different target
  (e.g. several `.Lflush` copies) is uniquified."""
  labels: dict[int, str] = {}
  taken: set[str] = set()
  for i, b in enumerate(bundles):
    for asm, op in b:
      tgt = branch_target(asm, op, i)
      if tgt is None: continue
      if not 0 <= tgt < len(bundles):
        raise ZhouyiError(f"branch at bundle {i} lands outside the image (bundle {tgt}): {asm}")
      if tgt in labels: continue
      want = asm.rsplit(",", 1)[-1].strip() if "," in asm else asm.split(None, 1)[1].strip()
      name = want if want not in taken else f"{want}_{tgt}"
      taken.add(name)
      labels[tgt] = name
  return labels


def emit_asm(bundles) -> str:
  """The image as assembly text, each branch rewritten to the label its encoding points at (oracle input)."""
  labels = _label_map(bundles)
  lines = ["\t.text", "\t.globl _entry", "\t.p2align 4", "_entry:"]
  for i, b in enumerate(bundles):
    if i in labels: lines.append(labels[i] + ":")
    lines.append("\t{")
    # An empty bundle must be written `nop`: `btaipuas` emits nothing for `{ }`, while `enc.bundle()`
    # emits four NOPs; `{ nop }` assembles to the same four words.
    if not b: lines.append("\t\tnop")
    for asm, op in b:
      if (tgt := branch_target(asm, op, i)) is not None:
        head = asm.rsplit(",", 1)[0] + ", " if "," in asm else asm.split(None, 1)[0] + " "
        asm = head + labels[tgt]
      lines.append("\t\t" + asm)
    lines.append("\t}")
  return "\n".join(lines) + "\n"


def lint(bundles, name: str = "kernel"):
  """The build-time gates that need the assembly text: the sync-flag rules (`tec_res.lint_sync`)."""
  from . import tec_res
  return tec_res.lint_sync(bundles, _label_map(bundles).keys(), name)


def text_only(kernel_bundles, n_epilogues: int = 1, sync_lint: bool = True) -> bytes:
  """Our encoder's image for `kernel_bundles` behind the spinning vector table, without the toolchain
  oracle (for kernels already verified via `build_encoded`). Still runs `check_image`, and the sync-flag
  lint unless `sync_lint=False` (only for a probe that breaks the flag rules on purpose)."""
  bundles = vector_table() + list(kernel_bundles)
  if sync_lint: lint(bundles)
  text = E.text([[op for _asm, op in b] for b in bundles])
  check_image(text, n_epilogues)
  return text


def build_encoded(kernel_bundles, vectors=None, sync_lint: bool = True) -> tuple[bytes, bytes, str]:
  """(our bytes, the oracle's bytes, the assembly text). The caller checks ours == theirs.

  `vectors` overrides the spinning vector table. `sync_lint`: as `text_only`."""
  bundles = (vector_table() if vectors is None else list(vectors)) + list(kernel_bundles)
  if sync_lint: lint(bundles)
  ours = E.text([[op for _asm, op in b] for b in bundles])
  asm_text = emit_asm(bundles)
  return ours, TC.asm(asm_text), asm_text


# ***************** the gates every image must pass *****************
def real_words(text:bytes) -> list[int]:
  """An image's instructions with padding nops stripped (both plain and end-of-bundle bit 31 forms)."""
  return [x for x in struct.unpack_from("<%dI" % (len(text)//4), text, 0) if x not in (E.NOP, E.NOP | 0x80000000)]


def check_image(text:bytes, n_epilogues:int = 1):
  """Structural checks for every image: slot 0 branches, one `exit` per epilogue, vendor-exact epilogue."""
  w = struct.unpack_from("<%dI" % (len(text)//4), text, 0)
  if w[0] & 0xFFF00000 != 0x01800000: raise ZhouyiError("slot0 of the vector table is not a branch")
  real = real_words(text)
  if real.count(0x014F8000) != n_epilogues:   # one `exit` per spliced epilogue
    raise ZhouyiError(f"expected {n_epilogues} task epilogue(s), found {real.count(0x014F8000)} exit instructions")
  if real[-11:] != VENDOR_EPILOGUE:
    raise ZhouyiError(f"task epilogue is not byte-exact vs the vendor:\n  ours   {[hex(x) for x in real[-11:]]}\n"
                    f"  vendor {[hex(x) for x in VENDOR_EPILOGUE]}")
