#!/usr/bin/env python3
"""Encoder for a subset of the Zhouyi X2 VLIW ISA (arch=zhouyi, mcpu=X2_1204).

Emits bytes directly; no vendor toolchain is called here. Encodings are verified against the
vendor assembler `btaipuas` (see `toolchain.py`).

## Encoding

**Word.** 32-bit little-endian. **Bundle.** Exactly 4 words (the `short-bundle` feature is
disabled). Unused slots hold `NOP`. Bit 31 of word3 marks the end of the bundle, except for the
two-word `aiff`/`dma` forms, which carry the marker in bit 31 of their own word0.

**Slots.** Position in the bundle selects the functional unit.

| slots | unit | instructions |
|---|---|---|
| 0, 1 | two ALU issue slots | `add` `sub` `max` `min` `mul` `mulh` `mml` `mma` `fma` (either); `xor` `and` `or` `lsl` `lsr` `asr` `bfs` `orl` `andl` (slot1 only) |
| 0 | control + compare | `cmp.lt` `cbnz` `br` `b` `loop` `loopend` `wfe` `exit` `mfctrl*` `mtctrl*` |
| 2, 3 | two load/store slots | `ld` `ldb` `st` `stb` `vld` `vst` |
| 2 | move | `mov` `movh` (GPR dest), `mov.pre` |
| 3 | predicate move | `mov`/`movh` with a `p0`-`p7` dest |
| 0+1 | two-word forms, alone in the bundle | `aiff` `dma` |
| 0, 1 | vector integer arithmetic | `qdpa` `dpa` `dot` `qdot` vector `mul` `mulh` `add` `sub` `max` `min` |
| 2 | vector data movement, shifts, logic (ONE per bundle) | `zip*` `sxt*` `nsr.*` `replic` `sld.*` `perm` `sel` `ext*` `shfl` `insert`, vector `lsl` `lsr` `asr` `and` `or` `xor` `andl` `orl` |
| 2, 3 / 3 | vector immediate moves | `movf` / `mov4` |

**Registers.** 32 GPRs and 32 256-bit vector ("t") registers, both in 5-bit fields,
disambiguated by opcode. Predicate regs `p0`-`p7` encode as 16-23 in the same GPR field,
disambiguated by slot.

**Common field layout.** `rd`[4:0], `rs1`[9:5], `rs2`[14:10].
"""
import struct

from tinygrad.runtime.support.zhouyi import ZhouyiError

NOP = 0x400C7FFF

GPR = {"r%d" % i: i for i in range(32)}
GPR.update(fp=22, cp=23, tcbp=24, pp=25, sdp=26, dp=27, lp=28, sp=29, ra=30, zero=31)
PREG = {"p%d" % i: 16 + i for i in range(8)}   # predicate regs share the same 5-bit field
VREG = {"t%d" % i: i for i in range(32)}       # 32 x 256-bit vector file

# slot-candidate classes
A  = (0, 1)   # general ALU: slot0 preferred, slot1 spare
B  = (1,)     # slot1-only ALU (xor, bfs, orl, andl)
C  = (0,)     # slot0-only: control + compare
M  = (2, 3)   # load/store unit: two slots
S2 = (2,)     # slot2-only moves (mov / movh with GPR dest, mov.pre)
P  = (3,)     # slot3-only (predicate-register moves)

AIFF_MARK = "aiff2"   # slot marker for the two-word `aiff` form
DMA_MARK  = "dma2"    # slot marker for the two-word `dma` form
TWO_WORD  = (AIFF_MARK, DMA_MARK)


def _g(x):
    if isinstance(x, int):
        return x
    if x in GPR:
        return GPR[x]
    if x in PREG:
        return PREG[x]
    raise KeyError("unknown register %r" % x)


def _v(x):
    if isinstance(x, int):
        return x
    if x in VREG:
        return VREG[x]
    raise KeyError("unknown vector register %r" % x)


def _u(v, w, name):
    if not (0 <= v < (1 << w)):
        raise ValueError("%s: %d out of unsigned %d-bit range" % (name, v, w))
    return v


def _s(v, w, name):
    lo, hi = -(1 << (w - 1)), (1 << (w - 1)) - 1
    if not (lo <= v <= hi):
        raise ValueError("%s: %d out of signed %d-bit range" % (name, v, w))
    return v & ((1 << w) - 1)


# ---- scalar ALU -------------------------------------------------------------
def _alu(base, rd, rs1, rs2):                 # rd[4:0] rs1[9:5] rs2[14:10]
    return base | _g(rd) | (_g(rs1) << 5) | (_g(rs2) << 10)


def _alui(base, rd, rs1, imm):                # imm[19:10], unsigned 10-bit
    return base | _g(rd) | (_g(rs1) << 5) | (_u(imm, 10, "imm") << 10)


def add(rd, rs1, x):
    return (A, _alui(0x40400000, rd, rs1, x) if isinstance(x, int) else _alu(0x40000000, rd, rs1, x))


def sub(rd, rs1, x):
    return (A, _alui(0x40500000, rd, rs1, x) if isinstance(x, int) else _alu(0x40008000, rd, rs1, x))


def mx(rd, rs1, x):                           # max, register form
    return (A, _alu(0x40060000, rd, rs1, x))


def mn(rd, rs1, x):                           # min, register form
    return (A, _alu(0x40068000, rd, rs1, x))


def mul(rd, rs1, x):
    """Low 32 bits of the product. There is no immediate form; materialise constants with
    `mov`/`movh` first."""
    return (A, _alu(0x40080000, rd, rs1, x))


def mulh(rd, rs1, x):                         # high 32 bits of the signed product
    return (A, _alu(0x40088000, rd, rs1, x))


def xor(rd, rs1, x):
    return (B, _alu(0x400D0000, rd, rs1, x))


# Register forms of and/or; distinct instructions from the immediate `andl`/`orl` below.
# Trailing underscore avoids the Python keywords.
def and_(rd, rs1, x):
    return (B, _alu(0x400C0000, rd, rs1, x))


def or_(rd, rs1, x):
    return (B, _alu(0x400C8000, rd, rs1, x))


def lsl(rd, rs1, x):
    """Register form only. The immediate form (`lsl r5, r6, 3` = `0x40178cc5`) uses a different
    immediate field that is not yet located, so it is not encoded."""
    return (B, _alu(0x400F8000, rd, rs1, x))


def lsr(rd, rs1, x):
    return (B, _alu(0x400F0000, rd, rs1, x))


def asr(rd, rs1, x):
    return (B, _alu(0x400E0000, rd, rs1, x))


def andl(rd, rs1, imm):
    """Slot1-only. The immediate is 5 bits (0-31) in the rs2 field [14:10], not `_alui`'s [19:10]."""
    return (B, 0x40100000 | _g(rd) | (_g(rs1) << 5) | (_u(imm, 5, "imm") << 10))


def orl(rd, rs1, imm):
    """5-bit immediate at [14:10], as `andl`."""
    return (B, 0x40108000 | _g(rd) | (_g(rs1) << 5) | (_u(imm, 5, "imm") << 10))


def bfs(rd, rs, p1, p2):
    return (B, 0x41E00000 | _g(rd) | (_g(rs) << 5)
            | (_u(p1, 5, "p1") << 10) | (_u(p2, 5, "p2") << 15))


def cmp_lt(rd, rs1, x):
    """cmp.lt rd, rs1, imm8 -- the immediate is split: i[4:0]@10, i[6:5]@19, i[7]@21,
    around the condition code at [16:15]."""
    if not isinstance(x, int):
        return (C, 0x42418000 | _g(rd) | (_g(rs1) << 5) | (_g(x) << 10))
    i = _s(x, 8, "imm")
    w = 0x42018000 | _g(rd) | (_g(rs1) << 5)
    w |= (i & 0x1F) << 10
    w |= ((i >> 5) & 0x3) << 19
    w |= ((i >> 7) & 0x1) << 21
    return (C, w)


# ---- scalar load / store ----------------------------------------------------
def _mem(base, r, breg, off, scale):
    if off % scale:
        raise ValueError("offset %d not %d-aligned" % (off, scale))
    return (M, base | _g(r) | (_g(breg) << 5) | (_s(off, 9, "off") << 10))


def ld(rd, b, off=0):
    return _mem(0x70100000, rd, b, off, 4)


def ldb(rd, b, off=0):
    return _mem(0x70000000, rd, b, off, 1)


def st(rs, b, off=0):
    return _mem(0x30100000, rs, b, off, 4)


def stb(rs, b, off=0):
    return _mem(0x30000000, rs, b, off, 1)


# ---- vector load / store ----------------------------------------------------
# Base = scalar ld/st base | 0x00680000 (32-byte width). Offset is an unsigned 9-bit byte offset
# at [18:10] (0..511); no alignment constraint is enforced by the assembler.
def _vmem(base, vr, breg, off):
    return (M, base | _v(vr) | (_g(breg) << 5) | (_u(off, 9, "voff") << 10))


def vld(vd, b, off=0):
    return _vmem(0x70780000, vd, b, off)


def vst(vs, b, off=0):
    # Slot 2 only: at most one vector store per bundle, and not beside a slot-2 vector op.
    slots, w = _vmem(0x30780000, vs, b, off)
    return (S2, w)


# ---- cache maintenance ----------------------------------------------------------------------
# `cacheop [rB+off], mode`: a store-family word (base 0x30380000) issued from the memory slots.
# mode (0..31) in [4:0], base register at [9:5], byte offset at [18:10].
# Note: the assembler refuses offsets above 31; keep the offset 0 and point the base at the line.
def cacheop(b, mode, off=0):
    return (M, 0x30380000 | _u(mode, 5, "mode") | (_g(b) << 5) | (_u(off, 5, "off") << 10))


# ---- the matrix unit --------------------------------------------------------
# `mml td.fp32, td+1.fp32, tx.f16, ty.f16`   -- 4x4x4 mixed-precision matmul
# `mma ...`                                  -- same, accumulating into {td,td+1}
# dest is a register PAIR, encoded as the low register in [4:0];
# x is [9:5] (row-major 4x4), y is [14:10] (column-major 4x4).
MML_FP16 = 0x63C10000
BIT_BF16 = 0x00008000     # both inputs bf16 instead of fp16; mixing is rejected
BIT_ACC  = 0x00020000     # mml -> mma
FMA_FP32 = 0x63380000


def _mm(base, vd, vx, vy):
    d = _v(vd)
    if d >= 31:
        raise ValueError("mml/mma dest pair {t%d,t%d} runs off the file" % (d, d + 1))
    return (A, base | d | (_v(vx) << 5) | (_v(vy) << 10))


def mml(vd, vx, vy, bf16=False):
    return _mm(MML_FP16 | (BIT_BF16 if bf16 else 0), vd, vx, vy)


def mma(vd, vx, vy, bf16=False):
    return _mm(MML_FP16 | BIT_ACC | (BIT_BF16 if bf16 else 0), vd, vx, vy)


def fma(vd, va, vb):
    """`fma td.fp32, ta.fp32, tb.fp32, p7.w` -- td += ta * tb, 8 lanes.

    The p7.w predicate is folded into the opcode base; p7 itself is set by the prologue's `movh p7, 0`."""
    return (A, FMA_FP32 | _v(vd) | (_v(va) << 5) | (_v(vb) << 10))


# ---- the vector integer unit -----------------------------------------------------------------
# Each base word is an assembler output word with its operand fields XORed out.
#
# Fields: rd [4:0], rs1 [9:5], rs2 [14:10], predicate [21:19]; element size .b/.h/.w = 0/1/2 at
# [16:15] on register-register forms; a 5-bit immediate at [14:10]; the 8-bit immediate of
# `add`/`sub`/`max`/`min` split low-5 at [14:10] and high-3 at [21:19]; `movf`/`mov4` carry a
# signed 8-bit immediate at [12:5].
#
# Slot classes: arithmetic (`qdpa` `dpa` `dot` `qdot` `mul` `mulh` `add` `sub` `max` `min`) takes
# slots 0/1, two per bundle. Everything else (`zip*` `sxt*` `uxtl`, shifts, logic, `nsr.*` `replic`
# `sld.*` `perm` `sel` `ext*` `shfl` `insert`) is slot 2 only, one per bundle, shared with the first
# `ld`. `movf` is slot 2/3, `mov4` slot 3. A `vst` cannot share a bundle with a slot-2 op.
#
# The assembler canonicalises the two sources of the predicated signed `qdpa`/`dpa`/`dot`/`qdot`/
# `mul`/`mulh` forms by register number (lower register in [9:5]); the `.uw` multiply forms put the
# higher register in [9:5]. The encoders below follow the same rule.
VEC_A = (0, 1)   # vector arithmetic: two per bundle
VEC_S = (2,)     # vector data movement / shifts / logic: slot 2 only
VEC_M = (2, 3)   # `movf`
VEC_M3 = (3,)    # `mov4`
VSIZE = {"b": 0, "h": 1, "w": 2}


def _vfields(rd, rs1=0, rs2=0, p=0):
    return _v(rd) | (_v(rs1) << 5) | (_v(rs2) << 10) | (p << 19)


def _pred(p):
    if isinstance(p, str):
        if p not in PREG: raise KeyError("predicate %r" % p)
        return PREG[p] - 16
    return _u(p, 3, "pred")


def _canon_signed(a, b):
    a, b = _v(a), _v(b)
    return (a, b) if a <= b else (b, a)


def _canon_unsigned(a, b):
    a, b = _v(a), _v(b)
    return (a, b) if a >= b else (b, a)


# reference words for (rd, rs1, rs2, p) = (t3, t5, t9, p7): 48f824a3 qdpa / 48b824a3 dpa / 487c24a3
# qdot / 487824a3 dot; the base is the word with those fields removed
_DOT_FIELDS = _vfields("t3", "t5", "t9", 7)
QDPA_BASE = 0x48f824a3 ^ _DOT_FIELDS     # 0x48c00000: bit 23 accumulate, bit 22 quad
DPA_BASE  = 0x48b824a3 ^ _DOT_FIELDS     # 0x48800000
QDOT_BASE = 0x487c24a3 ^ _DOT_FIELDS     # 0x48440000
DOT_BASE  = 0x487824a3 ^ _DOT_FIELDS     # 0x48400000


def _dotlike(base, td, ta, tb, p):
    a, b = _canon_signed(ta, tb)
    return (VEC_A, base | _vfields(td, a, b, _pred(p)))


def qdpa(td, ta, tb, p="p7"):
    """`qdpa td.w, ta.b, tb.b, p7.b` -- td[i] += sum_{k<4} ta[4i+k] * tb[4i+k], 8 int32 lanes
    from 32 int8 lanes (quad dot product, accumulating). Slot 0/1."""
    return _dotlike(QDPA_BASE, td, ta, tb, p)


def qdot(td, ta, tb, p="p7"):
    return _dotlike(QDOT_BASE, td, ta, tb, p)


def dpa(td, ta, tb, p="p7"):
    """`dpa td.h, ta.b, tb.b, p7.b` -- td[i] += ta[2i]*tb[2i] + ta[2i+1]*tb[2i+1], 16 int16 lanes."""
    return _dotlike(DPA_BASE, td, ta, tb, p)


def dot(td, ta, tb, p="p7"):
    return _dotlike(DOT_BASE, td, ta, tb, p)


# multiplies: 483924a3 mul.w / 483d24a3 mulh.w (canonical low-first); 483b1523 mul.uw /
# 483f1523 mulh.uw (canonical high-first: [9:5]=9, [14:10]=5)
VMUL_W_BASE  = 0x483924a3 ^ _DOT_FIELDS
VMULH_W_BASE = 0x483d24a3 ^ _DOT_FIELDS
VMUL_UW_BASE  = 0x483b1523 ^ _vfields("t3", "t9", "t5", 7)
VMULH_UW_BASE = 0x483f1523 ^ _vfields("t3", "t9", "t5", 7)


def vmul(td, ta, tb, p="p7", unsigned=False):
    """`mul td.w, ta.w, tb.w, p7.w` -- the low 32 bits of the lane product, 8 lanes."""
    a, b = (_canon_unsigned if unsigned else _canon_signed)(ta, tb)
    return (VEC_A, (VMUL_UW_BASE if unsigned else VMUL_W_BASE) | _vfields(td, a, b, _pred(p)))


def vmulh(td, ta, tb, p="p7", unsigned=False):
    """`mulh td.w, ta.w, tb.w, p7.w` -- the high 32 bits of the signed lane product."""
    a, b = (_canon_unsigned if unsigned else _canon_signed)(ta, tb)
    return (VEC_A, (VMULH_UW_BASE if unsigned else VMULH_W_BASE) | _vfields(td, a, b, _pred(p)))


# ---- the byte / half WIDENING multiplies (words as btaipuas assembles them, for t3, t5, t9, p7) --
# `mul td.h, ta.b, tb.b, p7.b` 483824a3: the 16-bit products of byte lanes 0-15 as 16 int16 lanes; `mulh` 483c24a3: lanes 16-31.
# The source size sits at [16:15] (the .w forms above are | 0x10000; `mul td.w, ta.h, tb.h, p7.h` 48389523). Signedness rides on
# the operand ORDER and bit 17: signed x signed = the lower register in [9:5]; one unsigned source = THAT register in [9:5], bit 17
# clear when it is the higher-numbered (`mul t3.h, t5.b, t9.ub` 48381523) and set when the lower (`mul t3.h, t5.ub, t9.b`
# 483a24a3); both unsigned = bit 17 and the higher register first (`mul t3.uw, t5.uw, t9.uw` 483b1523). The vendor disassembler
# renders the mixed forms loosely (48381523 as `t9.b, t5.b`, 483a24a3 as `.uh, .ub, .ub`); what each word DOES is the model's
# reading, kern_tpc.VEC_PROBE_RUNS["mulb"]: lanes 0-15 / 16-31, and the four sign mixes.
VMULB_BASE = 0x483824a3 ^ _vfields("t3", "t5", "t9", 7)      # 0x48000000
VMULHB_BASE = 0x483c24a3 ^ _vfields("t3", "t5", "t9", 7)     # 0x48040000


def _vmulw(base, td, ta, tb, ua, ub, p, size):
    a, b = _v(ta), _v(tb); u17 = 0
    if ua and ub: a, b, u17 = max(a, b), min(a, b), 0x20000
    elif ua or ub:
        a, b = (a, b) if ua else (b, a)                  # the unsigned source in [9:5]
        if a < b: u17 = 0x20000
    else: a, b = min(a, b), max(a, b)
    return (VEC_A, base | (VSIZE[size] << 15) | u17 | _vfields(td, a, b, _pred(p)))


def vmulb(td, ta, tb, ua=False, ub=False, p="p7", high=False, size="b"):
    """`mul td.h, ta.b, tb.b, p7.b` -- the widening products of byte lanes 0-15 as 16 int16 lanes (`high`: `mulh`, lanes 16-31);
    `ua` / `ub`: that source's lanes read unsigned (spelled `.ub`). `size="h"`: half lanes 0-7 / 8-15 into 8 int32 lanes."""
    return _vmulw(VMULHB_BASE if high else VMULB_BASE, td, ta, tb, ua, ub, p, size)


# add / sub / max / min: register forms are not canonicalised (`add t3.w, t5.w, t9.w` 434124a3;
# with p7 453924a3; .h 4340a4a3; .b 434024a3). Immediate forms carry an 8-bit immediate split
# [14:10] + [21:19]: unsigned for add/sub, signed for max/min (-3 -> 447974a3).
_RR = _vfields("t3", "t5", "t9")
VADD_BASE  = 0x434024a3 ^ _RR       # + VSIZE << 15
VSUB_BASE  = 0x434224a3 ^ _RR       # .b form (sub.w 434324a3)
VADDP_BASE = 0x453924a3 ^ _vfields("t3", "t5", "t9", 7)      # add.w, predicated
VMAXP_BASE = 0x433924a3 ^ _vfields("t3", "t5", "t9", 7)      # max.w / .uw = | 0x20000
VMINP_BASE = 0x433d24a3 ^ _vfields("t3", "t5", "t9", 7)      # min.w / .uw = | 0x20000
VADDI_BASE = 0x44010ca3 ^ _vfields("t3", "t5", 3)            # add.w imm
VSUBI_BASE = 0x44030ca3 ^ _vfields("t3", "t5", 3)            # sub.w imm
VMAXI_BASE = 0x44410ca3 ^ _vfields("t3", "t5", 3)            # max.w imm (signed)
VMINUI_BASE = 0x44470ca3 ^ _vfields("t3", "t5", 3)           # min.uw imm


# The `*_BASE` words below come from `.w` forms and already carry `VSIZE["w"] << 15`;
# `_sized` replaces that with the requested size.
def _sized(base, size):
    return (base & ~(3 << 15)) | (VSIZE[size] << 15)


def _imm8(base, td, ta, imm, signed, size="w"):
    v = (_s(imm, 8, "imm") if signed else _u(imm, 8, "imm"))
    return _sized(base, size) | _vfields(td, ta) | ((v & 0x1f) << 10) | ((v >> 5) << 19)


def vadd(td, ta, tb, size="w"):
    """`add td.w, ta.w, tb.w` (unpredicated). `size` b/h/w."""
    return (VEC_A, VADD_BASE | (VSIZE[size] << 15) | _vfields(td, ta, tb))


def vsub(td, ta, tb, size="w"):
    return (VEC_A, VSUB_BASE | (VSIZE[size] << 15) | _vfields(td, ta, tb))


def vaddi(td, ta, imm, size="w"):
    """`add td.w, ta.w, imm` -- imm 0..255."""
    return (VEC_A, _imm8(VADDI_BASE, td, ta, imm, False, size))


def vsubi(td, ta, imm, size="w"):
    return (VEC_A, _imm8(VSUBI_BASE, td, ta, imm, False, size))


def vmax(td, ta, tb, p="p7", unsigned=False, size="w"):
    return (VEC_A, _sized(VMAXP_BASE, size) | (0x20000 if unsigned else 0) | _vfields(td, ta, tb, _pred(p)))


def vmin(td, ta, tb, p="p7", unsigned=False, size="w"):
    return (VEC_A, _sized(VMINP_BASE, size) | (0x20000 if unsigned else 0) | _vfields(td, ta, tb, _pred(p)))


def vmaxi(td, ta, imm, size="w"):
    """`max td.w, ta.w, imm` -- imm signed −128..127."""
    return (VEC_A, _imm8(VMAXI_BASE, td, ta, imm, True, size))


def vminui(td, ta, imm, size="w"):
    """`min td.uw, ta.uw, imm` -- imm 0..255, unsigned lanes."""
    return (VEC_A, _imm8(VMINUI_BASE, td, ta, imm, False, size))


VMINI_BASE = 0x44450ca3 ^ _vfields("t3", "t5", 3)             # min.w imm (signed: -3 -> 447d74a3)
VMAXUI_BASE = 0x44430ca3 ^ _vfields("t3", "t5", 3)            # max.uw imm


def vmini(td, ta, imm, size="w"):
    """`min td.w, ta.w, imm` -- imm signed −128..127."""
    return (VEC_A, _imm8(VMINI_BASE, td, ta, imm, True, size))


def vmaxui(td, ta, imm, size="w"):
    return (VEC_A, _imm8(VMAXUI_BASE, td, ta, imm, False, size))


# ---- the vector FP32 unit ------------------------------------------------------------------------
# Same fields as the integer forms: td [4:0], ta [9:5], tb [14:10], predicate [21:19].
# All slot 0/1; `bcast` slot 3 only. There is no `cmp` on .fp32.
VMAXF_BASE = 0x62840000     # max td.fp32, ta.fp32, tb.fp32, p.w
VMINF_BASE = 0x62860000
VADDF_BASE = 0x62800000
VSUBF_BASE = 0x62820000     # td = ta - tb
VMULF_BASE = 0x63800000
VFMAF_BASE = 0x63000000     # td += ta * tb (`fma` above is this with p7)
VRINTF_BASE = 0x62c60800    # rint td.fp32, ta.fp32, p.w: round to nearest (even)
VSCAL2_BASE = 0x62c20000    # scal2 td.fp32, ta.fp32, tb.w, p.w: ta * 2^tb
CVT_W_F32 = 0x627e4000      # cvt td.w, ta.fp32 (no predicate)
CVT_F32_W = 0x627e4400      # cvt td.fp32, ta.w
CVTD_F16 = 0x62640000       # cvt.d td.fp16, ta.fp32, tb.fp32: lanes 0-7 from ta, 8-15 from tb
BCAST_W = 0x7dc14000        # bcast td.w, rs (slot 3)


def _f3(base, td, ta, tb=0, p="p7"): return (VEC_A, base | _vfields(td, ta, tb, _pred(p)))
def vmaxf(td, ta, tb, p="p7"): return _f3(VMAXF_BASE, td, ta, tb, p)
def vminf(td, ta, tb, p="p7"): return _f3(VMINF_BASE, td, ta, tb, p)
def vaddf(td, ta, tb, p="p7"): return _f3(VADDF_BASE, td, ta, tb, p)
def vsubf(td, ta, tb, p="p7"): return _f3(VSUBF_BASE, td, ta, tb, p)
def vmulf(td, ta, tb, p="p7"): return _f3(VMULF_BASE, td, ta, tb, p)
def vfmaf(td, ta, tb, p="p7"): return _f3(VFMAF_BASE, td, ta, tb, p)
# the 16-lane fp16 forms: the fp32 opcode | 1 << 16, the same fields (predicate as .h); oracle words (t3, t5, t9, p7) from
# as btaipuas assembles them: add.fp16 62b924a3, sub.fp16 62bb24a3, mul.fp16 63b924a3 (fp32: 62b824a3, 62ba24a3, 63b824a3)
F16_BIT = 0x00010000
def vaddf16(td, ta, tb, p="p7"): return _f3(VADDF_BASE | F16_BIT, td, ta, tb, p)
def vsubf16(td, ta, tb, p="p7"): return _f3(VSUBF_BASE | F16_BIT, td, ta, tb, p)
def vmulf16(td, ta, tb, p="p7"): return _f3(VMULF_BASE | F16_BIT, td, ta, tb, p)
def vrintf(td, ta, p="p7"): return _f3(VRINTF_BASE, td, ta, 0, p)
def vscal2(td, ta, tb, p="p7"): return _f3(VSCAL2_BASE, td, ta, tb, p)
def cvt_w_f32(td, ta): return (VEC_A, CVT_W_F32 | _vfields(td, ta))
def cvt_f32_w(td, ta): return (VEC_A, CVT_F32_W | _vfields(td, ta))
def cvtd_f16(td, ta, tb): return (VEC_A, CVTD_F16 | _vfields(td, ta, tb))
def bcast(td, rs): return (VEC_M3, BCAST_W | _v(td) | (_g(rs) << 5))


# ---- the slot-2 class ---------------------------------------------------------------------------
# reference words with (t3, t5, t9): zipl.b 044824a3  ziph.b 044a24a3  zipe.b 044c24a3  zipo.b 044e24a3
# (.h forms | 0x8000); or.w 040324a3 and.w 040124a3 xor.w 040524a3 (xor.b 040424a3); lsl.w reg
# 047b24a3 lsr.w reg 047324a3 asr.w reg 047124a3; perm.b 045624a3; extl.b 044024a3 exth.b 044224a3
# exte.b 044424a3 exto.b 044624a3; sel.w 057d24a3 (p7); replic.w 048324a3 (rs2 is a GPR)
ZIPL_BASE = 0x044824a3 ^ _RR
ZIPH_BASE = 0x044a24a3 ^ _RR
ZIPE_BASE = 0x044c24a3 ^ _RR
ZIPO_BASE = 0x044e24a3 ^ _RR
VOR_BASE  = 0x040224a3 ^ _RR       # .b form; .w = | 0x10000
VAND_BASE = 0x040024a3 ^ _RR
VXOR_BASE = 0x040424a3 ^ _RR
VLSL_BASE = 0x047a24a3 ^ _RR       # .b form; .w = | 0x10000
VLSR_BASE = 0x047224a3 ^ _RR
VASR_BASE = 0x047024a3 ^ _RR
PERM_BASE = 0x045624a3 ^ _RR
EXTL_BASE, EXTH_BASE, EXTE_BASE, EXTO_BASE = (w ^ _RR for w in (0x044024a3, 0x044224a3, 0x044424a3, 0x044624a3))
SEL_BASE = 0x057d24a3 ^ _vfields("t3", "t5", "t9", 7)      # sel.w
RPADD_BASE = 0x43ff20a3 ^ _vfields("t3", "t5", 0, 7)        # rpadd.w (rd, rs1, p): slot 0/1
REPLIC_BASE = 0x048224a3 ^ _RR    # .b form; .h | 0x8000, .w | 0x10000


def _s2(base, td, ta, tb, size="b"):
    return (VEC_S, base | (VSIZE[size] << 15) | _vfields(td, ta, tb))


def zipl(td, ta, tb, size="b"): return _s2(ZIPL_BASE, td, ta, tb, size)
def ziph(td, ta, tb, size="b"): return _s2(ZIPH_BASE, td, ta, tb, size)
def zipe(td, ta, tb, size="b"): return _s2(ZIPE_BASE, td, ta, tb, size)
def zipo(td, ta, tb, size="b"): return _s2(ZIPO_BASE, td, ta, tb, size)
def vor(td, ta, tb, size="w"): return _s2(VOR_BASE, td, ta, tb, size)
def vand(td, ta, tb, size="w"): return _s2(VAND_BASE, td, ta, tb, size)
def vxor(td, ta, tb, size="w"): return _s2(VXOR_BASE, td, ta, tb, size)
def vlsl(td, ta, tb, size="w"): return _s2(VLSL_BASE, td, ta, tb, size)
def vlsr(td, ta, tb, size="w"): return _s2(VLSR_BASE, td, ta, tb, size)
def vasr(td, ta, tb, size="w"): return _s2(VASR_BASE, td, ta, tb, size)
# rounding shifts by a vector amount: lsrr 047d24a3, asrr 047524a3 (t3, t5, t9)
VLSRR_BASE = 0x047c24a3 ^ _RR
VASRR_BASE = 0x047424a3 ^ _RR
def vlsrr(td, ta, tb, size="w"): return _s2(VLSRR_BASE, td, ta, tb, size)   # (a + 2^(b-1)) >> b, 33-bit
def vasrr(td, ta, tb, size="w"): return _s2(VASRR_BASE, td, ta, tb, size)
def perm(td, ta, tb): return _s2(PERM_BASE, td, ta, tb, "b")
# ext*: .w = .b | 0x10000 (exte.w 044524a3, exto.w 044724a3, extl.w 044124a3, exte.h 0444a4a3)
def extl(td, ta, tb, size="b"): return _s2(EXTL_BASE, td, ta, tb, size)
def exth(td, ta, tb, size="b"): return _s2(EXTH_BASE, td, ta, tb, size)
def exte(td, ta, tb, size="b"): return _s2(EXTE_BASE, td, ta, tb, size)
def exto(td, ta, tb, size="b"): return _s2(EXTO_BASE, td, ta, tb, size)


def rpadd(td, ta, p="p7"):
    """`rpadd td.w, ta.w, p7.w` -- pairwise reduce-add."""
    return (VEC_A, RPADD_BASE | _vfields(td, ta, 0, _pred(p)))


def sel(td, ta, tb, p="p7", size="w"):
    """`sel td.w, ta.w, tb.w, p7.w` -- per-lane select between `ta` and `tb` by predicate `p`."""
    return (VEC_S, _sized(SEL_BASE, size) | _vfields(td, ta, tb, _pred(p)))


# `cmp.<cond> pd.<sz>, ta.<sz>, tb.<sz>, p7.<sz>`: vector compare writing a predicate register
# (consumed by `sel`). Reference (p1, t5, t9, p7, .b): eq 42f824a1, ne 42f824a9, ge 42fa24a1,
# gt 42fa24a9, le 42fa24b1, lt 42fa24b9; p2 dest 42f824a2; .h | 0x8000, .w | 0x10000.
# Dest predicate at [2:0], condition at [4:3] plus bit 17; bit 18 (in the base) marks a register
# second operand. Only the register form is encoded.
CMP_COND = {"eq": (0, 0), "ne": (0, 1), "ge": (1, 0), "gt": (1, 1), "le": (1, 2), "lt": (1, 3)}
CMP_BASE = 0x42c00000


def vcmp(cond, pd, ta, tb, p="p7", size="w"):
    if cond not in CMP_COND: raise ZhouyiError("cmp condition %r is not one of %s" % (cond, sorted(CMP_COND)))
    if pd not in PREG: raise ZhouyiError("cmp writes a predicate register, got %r" % (pd,))
    ordered, sel_ = CMP_COND[cond]
    return (VEC_A, CMP_BASE | (VSIZE[size] << 15) | (ordered << 17) | (sel_ << 3)
                 | (PREG[pd] - 16) | (_v(ta) << 5) | (_v(tb) << 10) | (_pred(p) << 19))


def replic(td, ts, rs, size="w"):
    """`replic td.w, ts.w, rN` -- rs is a GPR (encoded in the rs2 field)."""
    return (VEC_S, REPLIC_BASE | (VSIZE[size] << 15) | _v(td) | (_v(ts) << 5) | (_g(rs) << 10))


# immediate shifts / logic, reference (t3, t5, 3): lsl.w 3c858ca3 lsr.w 3c818ca3 asr.w 3c808ca3
# lsrr.w 3c868ca3 asrr.w 3c828ca3 andl.w 3cc80ca3 orl.w 3cc88ca3 shfl.b 3ccc0ca3; sld.l.b 3ccd0ca3
# sld.r 3cdd0ca3 sld.dl 3ced0ca3 sld.dr 3cfd0ca3 sld.fr 3cdd8ca3 sld.dfr 3cfd8ca3; imm is 5 bits
_RI = _vfields("t3", "t5", 3)
VLSLI_BASE, VLSRI_BASE, VASRI_BASE = 0x3c858ca3 ^ _RI, 0x3c818ca3 ^ _RI, 0x3c808ca3 ^ _RI
VLSRRI_BASE, VASRRI_BASE = 0x3c868ca3 ^ _RI, 0x3c828ca3 ^ _RI
VANDLI_BASE, VORLI_BASE = 0x3cc80ca3 ^ _RI, 0x3cc88ca3 ^ _RI
SHFLI_BASE = 0x3ccc0ca3 ^ _RI
SLD_BASE = {"l": 0x3ccd0ca3 ^ _RI, "r": 0x3cdd0ca3 ^ _RI, "dl": 0x3ced0ca3 ^ _RI, "dr": 0x3cfd0ca3 ^ _RI,
            "fr": 0x3cdd8ca3 ^ _RI, "dfr": 0x3cfd8ca3 ^ _RI}


def _s2i(base, td, ta, imm):
    return (VEC_S, base | _vfields(td, ta) | (_u(imm, 5, "imm") << 10))


# Shift immediates: a one-hot size marker sits in [15:13] and the immediate occupies the bits
# below it, so the immediate is 3 bits at .b, 4 at .h, 5 at .w.
# Reference: lsl.b/.h/.w = 3c853ca3 / 3c855ca3 / 3c859ca3.
def _s2i_sz(base, td, ta, imm, size):
    sz = VSIZE[size]
    hi = 1 << (3 + sz)
    if not 0 <= imm < hi:
        raise ZhouyiError("a .%s shift immediate is %d bits: %d is outside 0..%d" % (size, 3 + sz, imm, hi - 1))
    return (VEC_S, (base & ~(7 << 13)) | _vfields(td, ta) | ((hi | imm) << 10))


def vlsli(td, ta, imm, size="w"): return _s2i_sz(VLSLI_BASE, td, ta, imm, size)
def vlsri(td, ta, imm, size="w"): return _s2i_sz(VLSRI_BASE, td, ta, imm, size)
def vasri(td, ta, imm, size="w"): return _s2i_sz(VASRI_BASE, td, ta, imm, size)
def vlsrri(td, ta, imm, size="w"): return _s2i_sz(VLSRRI_BASE, td, ta, imm, size)    # logical shift right, rounding
def vasrri(td, ta, imm, size="w"): return _s2i_sz(VASRRI_BASE, td, ta, imm, size)    # arithmetic shift right, rounding
def vandli(td, ta, imm): return _s2i(VANDLI_BASE, td, ta, imm)
def vorli(td, ta, imm): return _s2i(VORLI_BASE, td, ta, imm)
def shfli(td, ta, imm): return _s2i(SHFLI_BASE, td, ta, imm)


def sld(kind, td, ta, imm):
    """`sld.<kind> td.b, ta.b, imm` -- kind in l/r/dl/dr/fr/dfr (lane slide)."""
    return _s2i(SLD_BASE[kind], td, ta, imm)


# narrowing shifts: `nsr.as t3.bw, 3` 3c878e43 -- one register (in place), imm at [14:10]
_NI = _v("t3") | (3 << 10)
NSR_BASE = {("as", "bw"): 0x3c878e43 ^ _NI, ("asr", "bw"): 0x3c878e63 ^ _NI, ("a", "bw"): 0x3c878e03 ^ _NI,
            ("l", "bw"): 0x3c878f03 ^ _NI, ("as", "hw"): 0x3c878c43 ^ _NI, ("as", "bh"): 0x3c874c43 ^ _NI}


def nsr(mode, sizes, td, imm):
    """`nsr.<mode> td.<sizes>, imm` -- narrow td in place by a right shift of `imm`: mode a (arithmetic)
    / as (arithmetic, saturating) / asr (a, saturating, rounding) / l (logical); sizes bw (word->byte),
    hw (word->half), bh (half->byte)."""
    return (VEC_S, NSR_BASE[(mode, sizes)] | _v(td) | (_u(imm, 5, "imm") << 10))


# widening: sxtl.h.b 045e80a3  sxth.h.b 045e84a3  sxtl.w.h 045f00a3  sxth.w.h 045f04a3  uxtl.h.b 045e88a3
_RS = _vfields("t3", "t5")
SXT_BASE = {("l", "hb"): 0x045e80a3 ^ _RS, ("h", "hb"): 0x045e84a3 ^ _RS, ("l", "wh"): 0x045f00a3 ^ _RS,
            ("h", "wh"): 0x045f04a3 ^ _RS}
UXTL_HB_BASE = 0x045e88a3 ^ _RS


def sxt(half, sizes, td, ta):
    """`sxtl td.h, ta.b` / `sxth` / `sxtl td.w, ta.h` / `sxth td.w, ta.h`: sign-extend the low/high half."""
    return (VEC_S, SXT_BASE[(half, sizes)] | _vfields(td, ta))


def uxtl_hb(td, ta): return (VEC_S, UXTL_HB_BASE | _vfields(td, ta))


# insert: `insert t3.b, r5, 3` 3cce0ca3 (imm 5 bits) / `insert t3.w, r5, 3` 3cc62ca3 (imm 3 bits)
INSB_BASE = 0x3cce0ca3 ^ (_v("t3") | (_g("r5") << 5) | (3 << 10))
INSW_BASE = 0x3cc62ca3 ^ (_v("t3") | (_g("r5") << 5) | (3 << 10))


def insert(td, rs, lane, size="w"):
    if size == "w": return (VEC_S, INSW_BASE | _v(td) | (_g(rs) << 5) | (_u(lane, 3, "lane") << 10))
    return (VEC_S, INSB_BASE | _v(td) | (_g(rs) << 5) | (_u(lane, 5, "lane") << 10))


# moves: `movf t3.w, 3, p7.w` 7d7f6063 (slot 2/3; signed 8-bit imm at [12:5]); `mov4 t3, 3` fdf00063 (slot 3)
MOVF_BASE = 0x7d7f6063 ^ (_v("t3") | (3 << 5) | (7 << 19))
MOV4_BASE = 0x7df00063 ^ (_v("t3") | (3 << 5))


def movf(td, imm, p="p7"):
    """`movf td.w, imm, p7.w` -- imm signed −128..127 into every (predicated) lane."""
    return (VEC_M, MOVF_BASE | _v(td) | (_s(imm, 8, "imm") << 5) | (_pred(p) << 19))


def mov4(td, imm):
    return (VEC_M3, MOV4_BASE | _v(td) | (_s(imm, 8, "imm") << 5))


# ---- moves ------------------------------------------------------------------
def _ispred(r):
    return isinstance(r, str) and r in PREG


def mov(rd, imm):
    if _ispred(rd):
        return (P, 0x7DF60000 | _g(rd) | (_u(imm, 5, "imm") << 5))
    return (S2, 0x00C00000 | _g(rd) | (_s(imm, 16, "imm") << 5))


def movh(rd, imm):
    if _ispred(rd):
        return (P, 0x7DF60000 | _g(rd) | (_u(imm, 5, "imm") << 5))
    return (S2, 0x00E00000 | _g(rd) | (_s(imm, 16, "imm") << 5))


def mov_pre(rd, rs):
    return (S2, 0x7DFF0000 | _g(rd) | (_g(rs) << 5))


# ---- control ----------------------------------------------------------------
def br(rs):
    return (C, 0x01400000 | _g(rs))


def b(delta_bundles):
    """Unconditional PC-relative branch from this instruction's own bundle.

    The field is 20 bits signed ([19:0]) in 8-byte units, so one bundle (16 B) is 2."""
    return (C, 0x01800000 | (_s(delta_bundles * 2, 20, "branch delta") & 0xFFFFF))


def loop(rs):
    return (C, 0x01500000 | _g(rs))


def loopend():
    return (C, 0x01508000)


def cbnz(rs, delta):
    """delta is in bundles, relative to this instruction's own bundle. The field is s14@[19:6]."""
    return (C, 0x01700000 | _g(rs) | (_s(delta, 14, "bundle delta") << 6))


def wfe(rs, imm=0):
    """`wfe <rs mask>, <imm5 mode>` -- mask register at [4:0], mode (5 bits) at [9:5].

    Mode 1 waits for the flags whose mask bit is clear; mode 2 until no flag is busy (mask ignored); mode 0
    on another event class (the epilogue's `wfe r0, 0`); modes 3-31 raise exception vector 2
    (measured). `tec_res.wait_flags` emits the checked forms. (The vendor
    ISA XML's operand layout for WFE does not match X2.)"""
    return (C, 0x014C0000 | _g(rs) | (_u(imm, 5, "imm") << 5))


# Control-slot terminator family: one opcode, a 2-bit selector at [6:5]. No operands.
#
#     exit   0x014F8000    the task ends
#     irtn   0x014F8020    interrupt return: resume the faulting context
#     break  0x014F8040    raise an exception deliberately
#     ---    0x014F8060    encodable; no assembler mnemonic
def exit_():
    return (C, 0x014F8000)


def irtn():
    """Interrupt return. Encoding only; runtime behaviour after return is unverified."""
    return (C, 0x014F8020)


def break_():
    """Raise an exception deliberately (e.g. to exercise the vector table)."""
    return (C, 0x014F8040)


# Control-register moves: the index is 10 bits at [14:5] (1024 per space), the space at [16:15]
# (0-2; the fourth value 0x01058000 has no mnemonic). How many indices the hardware decodes is
# separate from what the encoding can name.
def mfctrl0(rd, i):
    return (C, 0x01040000 | _g(rd) | (_u(i, 10, "ctrl") << 5))


def mtctrl0(rs, i):
    return (C, 0x01000000 | _g(rs) | (_u(i, 10, "ctrl") << 5))


def mfctrl1(rd, i):
    """ctrl-space 1 read. The FP enable lives here (ctrl1[2] and ctrl1[0x10]); without it every
    vector-arithmetic instruction traps to exception vector 3."""
    return (C, 0x01048000 | _g(rd) | (_u(i, 10, "ctrl") << 5))


def mtctrl1(rs, i):
    return (C, 0x01008000 | _g(rs) | (_u(i, 10, "ctrl") << 5))


def mfctrl2(rd, i):
    return (C, 0x01050000 | _g(rd) | (_u(i, 10, "ctrl") << 5))


def mtctrl2(rs, i):
    """ctrl-space 2 write. Not the AIFF kick; that is the `aiff` instruction."""
    return (C, 0x01010000 | _g(rs) | (_u(i, 10, "ctrl") << 5))


# ---- AIFF -------------------------------------------------------------------
def aiff(mode, sync, desp_ctrl, ctrl, param, act):
    """`aiff <imm6 mode>, <rs sync>, <imm4 desp_ctrl>, <rs ctrl>, <rs param>, <rs act>`

        word0 = 0x80000000 | desp_ctrl << 27 | ctrl << 10 | param << 5 | act
        word1 = 0x40020000 | mode      << 10 | sync

    A two-word instruction filling bundle slots 0 and 1. (`aiff 0,zero,0,zero,zero,zero`
    = `80007fff 4002001f`; `zero` is GPR 31.)"""
    w0 = (0x80000000
          | (_u(desp_ctrl, 4, "desp_ctrl") << 27)
          | (_g(ctrl) << 10)
          | (_g(param) << 5)
          | _g(act))
    w1 = (0x40020000
          | (_u(mode, 6, "mode") << 10)
          | _g(sync))
    return (AIFF_MARK, (w0, w1))


# ---- DMA --------------------------------------------------------------------
# Operand names follow the vendor C builtin `__dma(...)`.
# Note: operand 5 is `dir` (transfer direction, bit 18), not `num_of_trans` (a descriptor field).
DMA_GLOBAL, DMA_GLOBAL1, DMA_GLOBAL2, DMA_GLOBAL3, DMA_LSRAM, DMA_SHARED = 0, 1, 2, 3, 4, 8
DMA_INT2EXT, DMA_EXT2INT = 0, 1                        # enum DMADirection
DMA_DIRECT, DMA_TRANSPOSE, DMA_UPSAMPLE, DMA_MEMSET = 0, 1, 2, 3   # enum DmaMode
DMA_USELESS, DMA_BYTE, DMA_HALF, DMA_WORD = 0, 0, 1, 2             # typedef DmaDataUnit


def dma(mode, rsync, desp_ctrl, int_base, ext_base, dir_, data_unit, desp, int_addr, ext_addr):
    """`dma <imm6 mode>, <rs rsync>, <imm4 desp_ctrl>, <imm4 int_base>, <imm4 ext_base>,
           <imm1 dir>, <imm3 data_unit>, <rs desp>, <rs int_addr>, <rs ext_addr>`

        word0 = 0x80000000 | desp_ctrl << 27 | int_base << 23 | ext_base << 19
                           | dir << 18 | data_unit << 15 | desp << 10 | int_addr << 5 | ext_addr
        word1 = 0x40010000 | mode << 10 | rsync

    A two-word instruction filling bundle slots 0 and 1, like `aiff` (end-of-bundle marker in
    bit 31 of word0, word3 a plain NOP).

    Note: `ext_base` must be a global aperture. An SRAM code there (e.g. GSRAM -> LSRAM) encodes
    but the transfer never completes and the `wfe` hangs; SRAM-to-SRAM goes through the TEC.
    `rsync` is a register holding `(flag << 8) | flag`, not an immediate flag number; flag
    allocation is owned by `dma.py`.
    """
    w0 = (0x80000000
          | (_u(desp_ctrl, 4, "desp_ctrl") << 27)
          | (_u(int_base, 4, "int_base") << 23)
          | (_u(ext_base, 4, "ext_base") << 19)
          | (_u(dir_, 1, "dir") << 18)
          | (_u(data_unit, 3, "data_unit") << 15)
          | (_g(desp) << 10)
          | (_g(int_addr) << 5)
          | _g(ext_addr))
    w1 = (0x40010000
          | (_u(mode, 6, "mode") << 10)
          | _g(rsync))
    return (DMA_MARK, (w0, w1))


# ---- bundle assembly --------------------------------------------------------
def bundle(*ops):
    """Pack ops into one 4-word bundle, honouring each op's slot class.

    Two-word forms (`aiff`, `dma`) must be alone: the assembler silently drops any companion
    instruction in their bundle, so one is refused here. (A DMA transfer still overlaps later
    bundles; only the issue slot is exclusive.)"""
    w = [NOP, NOP, NOP, NOP]
    used = set()
    if any(o[0] in TWO_WORD for o in ops):
        if len(ops) != 1:
            raise ValueError("%s must be alone in its bundle: btaipuas silently discards "
                             "companion instructions"
                             % next(o[0] for o in ops if o[0] in TWO_WORD).rstrip("2"))
        w[0], w[1] = ops[0][1]
        # Two-word bundles carry the end-of-bundle marker in bit 31 of word0, not word3.
        return struct.pack("<4I", *w)
    for cands, word in ops:
        for s in cands:
            if s not in used:
                used.add(s)
                w[s] = word
                break
        else:
            raise ValueError("no free slot in %s" % (cands,))
    w[3] |= 0x80000000
    return struct.pack("<4I", *w)


def text(bundles):
    """bundles: list of tuples of ops (each op is the (slots, word) pair above)."""
    return b"".join(bundle(*b) for b in bundles)
