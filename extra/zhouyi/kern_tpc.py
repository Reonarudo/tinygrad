#!/usr/bin/env python3
"""TPC kernels (matrix unit, fp32 vector FPU, scalar, DMA, LSRAM/GSRAM probes) written once as
(assembly text, encoder call) pairs, so the text given to the btaipuas assembler and the bytes our
encoder produces cannot drift apart; the gate is a whole-image byte diff.

The prologue carries the FP enable (`image.fp_enable`): without it every vector-arithmetic
instruction (fp32 `fma` as well as `mml`/`mma`) traps to exception vector 3, which presents as a hang.
"""
import struct
from typing import NamedTuple

from extra.zhouyi import enc as E
from extra.zhouyi import dma as D
from extra.zhouyi.tec_res import sync_flag, wait_flags, Lsram, GsramWindow
from extra.zhouyi.image import I, BUNDLE, fp_enable, prologue, epilogue, EPILOGUE_FLUSH_LABEL_AT


# ------------------------------------------------------------- kernel: mm ----
def k_matmul(bf16=False):
    """C[4x4] (fp32) = C + sum over NT tiles of A_tile(4x4 fp16) @ B_tile(4x4 fp16).

    args: r0=A, r1=B, r2=C, r3=NT. A tiles are 4x4 row-major fp16, B tiles 4x4 column-major fp16,
    32 B each, contiguous. C is 16 fp32 row-major (64 B), both bias and destination.
    """
    mm = E.mma
    n = "mma"
    q = "bf16" if bf16 else "fp16"
    body = [
        BUNDLE(I("ld t0, [r2+0]", E.vld("t0", "r2", 0)),
               I("ld t1, [r2+32]", E.vld("t1", "r2", 32))),
        BUNDLE(I("sub r10, r3, 1", E.sub("r10", "r3", 1))),
        BUNDLE(I("max r10, r10, zero", E.mx("r10", "r10", "zero"))),
        BUNDLE(I("loop r10", E.loop("r10"))),
        # .Lbody:
        BUNDLE(I("ld t10, [r0+0]", E.vld("t10", "r0", 0)),
               I("ld t11, [r1+0]", E.vld("t11", "r1", 0))),
        BUNDLE(I("add r0, r0, 32", E.add("r0", "r0", 32)),
               I("add r1, r1, 32", E.add("r1", "r1", 32))),
        BUNDLE(I("%s t0.fp32, t1.fp32, t10.%s, t11.%s" % (n, q, q),
                 mm("t0", "t10", "t11", bf16=bf16))),
        BUNDLE(I("loopend", E.loopend())),
        # vector stores do NOT dual-issue (only the two loads do) -- one per bundle
        BUNDLE(I("st t0, [r2+0]", E.vst("t0", "r2", 0))),
        BUNDLE(I("st t1, [r2+32]", E.vst("t1", "r2", 32))),
    ]
    return prologue(4) + body + epilogue()


# ------------------------------------------------ kernel: scalar integer -----
def k_intmul():
    """OUT[i] = (A[i] * B[i] >> 3) + max(A[i], B[i]), i < N, int32 throughout.

    args: r0=A, r1=B, r2=OUT, r3=N. The shift and max make the result depend on the product's low
    bits and on both operands independently. Integer-only, so no FP enable (`fp=False`). `mul`/`asr`
    have no immediate form here, so the shift distance lives in r9.
    """
    body = [
        # r9 = 3, the shift distance
        BUNDLE(I("mov r9, 3", E.mov("r9", 3))),
        # .Lbody: 6 bundles, branch delta -6
        BUNDLE(I("ld r4, [r0+0]", E.ld("r4", "r0", 0)),
               I("ld r5, [r1+0]", E.ld("r5", "r1", 0))),
        BUNDLE(I("mul r6, r4, r5", E.mul("r6", "r4", "r5")),
               I("add r0, r0, 4", E.add("r0", "r0", 4))),
        BUNDLE(I("max r7, r4, r5", E.mx("r7", "r4", "r5")),
               I("asr r6, r6, r9", E.asr("r6", "r6", "r9"))),
        BUNDLE(I("add r8, r6, r7", E.add("r8", "r6", "r7")),
               I("add r1, r1, 4", E.add("r1", "r1", 4))),
        BUNDLE(I("st r8, [r2+0]", E.st("r8", "r2", 0))),
        BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1)),
               I("add r2, r2, 4", E.add("r2", "r2", 4))),
        BUNDLE(I("cbnz r3, .Lbody", E.cbnz("r3", -6))),
    ]
    return prologue(4, fp=False) + body + epilogue()


# ------------------------------------------------- kernels: throughput -------
# The benches share one loop shape (8 body bundles, 2 ops each, then loopend); only the
# instruction under test differs.
def _bench_common_head():
    """args: r0=X (32 B operand block), r1=OUT, r2=LOOPS"""
    return prologue(3) + [
        BUNDLE(I("ld t16, [r0+0]", E.vld("t16", "r0", 0)),
               I("ld t17, [r0+32]", E.vld("t17", "r0", 32))),
    ] + [
        # init the 16 accumulators from OUT (host pre-fills)
        BUNDLE(I("ld t%d, [r1+%d]" % (i, i * 32), E.vld("t%d" % i, "r1", i * 32)),
               I("ld t%d, [r1+%d]" % (i + 1, (i + 1) * 32), E.vld("t%d" % (i + 1), "r1", (i + 1) * 32)))
        for i in range(0, 16, 2)
    ] + [
        BUNDLE(I("sub r10, r2, 1", E.sub("r10", "r2", 1))),
        BUNDLE(I("max r10, r10, zero", E.mx("r10", "r10", "zero"))),
        BUNDLE(I("loop r10", E.loop("r10"))),
    ]


def _bench_tail():
    return [
        BUNDLE(I("loopend", E.loopend())),
    ] + [
        BUNDLE(I("st t%d, [r1+%d]" % (i, i * 32), E.vst("t%d" % i, "r1", i * 32)))
        for i in range(16)
    ] + epilogue()


def k_bench_mma(bf16=False):
    """16 mma per iteration over 8 independent accumulator PAIRS (t0..t15),
    2 per bundle, so each pair is reused only every 4 bundles."""
    q = "bf16" if bf16 else "fp16"
    body = []
    for rep in range(2):
        for i in range(0, 16, 4):
            body.append(BUNDLE(
                I("mma t%d.fp32, t%d.fp32, t16.%s, t17.%s" % (i, i + 1, q, q),
                  E.mma("t%d" % i, "t16", "t17", bf16=bf16)),
                I("mma t%d.fp32, t%d.fp32, t16.%s, t17.%s" % (i + 2, i + 3, q, q),
                  E.mma("t%d" % (i + 2), "t16", "t17", bf16=bf16))))
    assert len(body) == 8
    return _bench_common_head() + body + _bench_tail()


def k_bench_fma():
    """16 fp32 fma per iteration over 16 independent accumulators (t0..t15),
    2 per bundle -- the same 8 body bundles as k_bench_mma."""
    body = []
    for i in range(0, 16, 2):
        body.append(BUNDLE(
            I("fma t%d.fp32, t16.fp32, t17.fp32, p7.w" % i, E.fma("t%d" % i, "t16", "t17")),
            I("fma t%d.fp32, t16.fp32, t17.fp32, p7.w" % (i + 1), E.fma("t%d" % (i + 1), "t16", "t17"))))
    assert len(body) == 8
    return _bench_common_head() + body + _bench_tail()


# ------------------------------------------------------------ bisect aids ---


def k_bench_nop():
    """8 empty bundles per iteration: the loop's own cost."""
    return _bench_common_head() + [BUNDLE() for _ in range(8)] + _bench_tail()


def k_bench_mma1(bf16=False):
    """The same 16 mma per iteration as k_bench_mma, one per bundle (16 bundles): distinguishes
    instruction-limited from bundle-limited matrix issue."""
    q = "bf16" if bf16 else "fp16"
    body = []
    for rep in range(2):
        for i in range(0, 16, 2):
            body.append(BUNDLE(
                I("mma t%d.fp32, t%d.fp32, t16.%s, t17.%s" % (i, i + 1, q, q),
                  E.mma("t%d" % i, "t16", "t17", bf16=bf16))))
    assert len(body) == 16
    return _bench_common_head() + body + _bench_tail()


# ------------------------------------------ locality probes ----
# Residency probes: do a producer's DDR lines survive into a consumer task, and does a TEC's LSRAM
# survive to a later task on the same TEC? Built to be defeated by a cache, not to go fast.

# LSRAM is 32 KiB per TEC at 0xFA000000, materialised as the prologue does `lp`: `mov` low half,
# `movh` high half (0xFA00 does not fit a signed 16-bit immediate, hence -1536). The prologue leaves
# `lp` at 0xFA008000, the top of the window. The task stack is in DDR (sp comes from the TCB), so
# all 32 KiB is available to a staged kernel.
LSRAM_BASE_LO, LSRAM_BASE_HI = 0, -1536     # 0xFA000000
LSRAM_PROBE_WORDS = 8                       # 32 B, well inside `st`'s 252 B immediate reach
GATE_ENDS = 8                               # words gated at EACH end by `k_dma_mode(gate="ends")`


def _lsram_base(reg):
    """Two bundles that leave `reg` holding 0xFA000000."""
    return [BUNDLE(I("mov %s, %d" % (reg, LSRAM_BASE_LO), E.mov(reg, LSRAM_BASE_LO))),
            BUNDLE(I("movh %s, %d" % (reg, LSRAM_BASE_HI), E.movh(reg, LSRAM_BASE_HI)))]


def k_chase():
    """Dependent pointer chase over DDR. args: r0=BASE, r1=STEPS, r2=OUT.

    BASE holds a permutation cycle of byte offsets, so each load address is the previous load's
    value: no prefetch or reordering can hide latency. The final offset (stored to OUT) is
    host-predictable and reachable only if every load happened.
    """
    body = [
        BUNDLE(I("mov r4, 0", E.mov("r4", 0))),
        # .Lchase: 4 bundles -- address, dependent load, counter, branch
        BUNDLE(I("add r5, r0, r4", E.add("r5", "r0", "r4"))),
        BUNDLE(I("ld r4, [r5+0]", E.ld("r4", "r5", 0))),
        BUNDLE(I("sub r1, r1, 1", E.sub("r1", "r1", 1))),
        BUNDLE(I("cbnz r1, .Lchase", E.cbnz("r1", -3))),
        BUNDLE(I("st r4, [r2+0]", E.st("r4", "r2", 0))),
    ]
    return prologue(3, fp=False) + body + epilogue()


def k_stream():
    """Sequential read of `r1` words from `r0`, summed into `r2[0]`. args: r0=BASE, r1=N, r2=OUT.

    Cache-evicting interlude; the sum is exact int32 for the host's seed, so a skipped run fails."""
    body = [
        BUNDLE(I("mov r4, 0", E.mov("r4", 0))),
        # .Lstream: 3 bundles
        BUNDLE(I("ld r5, [r0+0]", E.ld("r5", "r0", 0)),
               I("add r0, r0, 4", E.add("r0", "r0", 4))),
        BUNDLE(I("add r4, r4, r5", E.add("r4", "r4", "r5")),
               I("sub r1, r1, 1", E.sub("r1", "r1", 1))),
        BUNDLE(I("cbnz r1, .Lstream", E.cbnz("r1", -2))),
        BUNDLE(I("st r4, [r2+0]", E.st("r4", "r2", 0))),
    ]
    return prologue(3, fp=False) + body + epilogue()


def k_lsram_stamp():
    """Write a per-task signature into this TEC's LSRAM. args: r0=PATTERN, r1=OUT.

    Stores PATTERN+i (i < 8) at 0xFA000000 + 4i, then echoes PATTERN to OUT[0] in DDR so "never
    dispatched" differs from "LSRAM did not persist". Ascending values distinguish a smear from a stamp."""
    body = _lsram_base("r5")
    for i in range(LSRAM_PROBE_WORDS):
        body += [BUNDLE(I("add r6, r0, %d" % i, E.add("r6", "r0", i))),
                 BUNDLE(I("st r6, [r5+%d]" % (4 * i), E.st("r6", "r5", 4 * i)))]
    body += [BUNDLE(I("st r0, [r1+0]", E.st("r0", "r1", 0)))]
    return prologue(2, fp=False) + body + epilogue()


# ------------------------------------------ the DMA round-trip ----
def k_dma_roundtrip(wait=True, wait_flag=0, dma_flag=0):
    """DDR -> LSRAM (by DMA) -> DDR (by the scalar unit). Scalar-only (`fp=False`), word at a time.

    args: r0=SRC, r1=DESC, r2=OUT, r3=NWORDS, r4=POISON, r5=PROBE

    * LSRAM is filled with ascending POISON by this kernel right before the request (LSRAM persists
      across jobs, so stale correct data is possible otherwise).
    * PROBE records LSRAM[0] one bundle after the request, before any wait; in the waiting arm it
      should still read POISON.
    * `wait=False` omits the `wfe`.
    * `dma_flag` x `wait_flag` checks the `rsync` operand. A mismatched pair returns early (a wrong
      answer, not a hang): `wfe(~(1 << f), 1)` waits for flag f to go idle, and an unraised flag
      is already idle.

    The descriptor is built by the host into a DDR buffer (`dma.py`'s `ddr_slot` placement) and
    passed in r1.
    """
    # Both flags go through `dma.check_flag`: an out-of-range flag can silently alias onto flag 0
    # and make the wait return early.
    D.check_flag(dma_flag); D.check_flag(wait_flag)
    body = _lsram_base("r6") + [
        # ---- poison the LSRAM window: [r8] = r9, r9 ascending, NWORDS times
        BUNDLE(I("add r7, r3, 0", E.add("r7", "r3", 0)),
               I("add r9, r4, 0", E.add("r9", "r4", 0))),
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0))),
        # .Lpoison: 4 bundles. The store and the increment of its address register are kept in
        # separate bundles, so nothing relies on VLIW read-before-write.
        BUNDLE(I("st r9, [r8+0]", E.st("r9", "r8", 0))),
        BUNDLE(I("add r8, r8, 4", E.add("r8", "r8", 4)),
               I("add r9, r9, 1", E.add("r9", "r9", 1))),
        BUNDLE(I("sub r7, r7, 1", E.sub("r7", "r7", 1))),
        BUNDLE(I("cbnz r7, .Lpoison", E.cbnz("r7", -3))),
    ]
    # ---- the request: ext(DDR, r0) -> int(LSRAM, r6), descriptor at r1.
    # `rsync` is a register holding (flag << 8) | flag; for flag 0 `zero` serves (byte-identical to
    # the vendor compiler's output).
    if dma_flag:
        rs, rsv = "r11", (dma_flag << 8) | dma_flag
        body += [BUNDLE(I("mov r11, %d" % rsv, E.mov("r11", rsv)))]
    else:
        rs = "zero"
    body += [
        BUNDLE(I("dma 0, %s, 0, 4, 0, 1, 0, r1, r6, r0" % rs,
                 E.dma(E.DMA_DIRECT, rs, 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT,
                       E.DMA_USELESS, "r1", "r6", "r0"))),
        # ---- PROBE: LSRAM[0] one bundle after the request, before any wait
        BUNDLE(I("ld r10, [r6+0]", E.ld("r10", "r6", 0))),
        BUNDLE(I("st r10, [r5+0]", E.st("r10", "r5", 0))),
    ]
    if wait:
        # Wait mask ~(1 << flag) (`dma.wfe_mask`), materialised as `sub` from `zero` since
        # ~x == -x - 1. `sub`'s immediate is unsigned 10-bit, covering flags 0..9.
        body += [BUNDLE(I("sub r7, zero, %d" % ((1 << wait_flag) + 1),
                          E.sub("r7", "zero", (1 << wait_flag) + 1))),
                 BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    body += [
        # ---- copy the staged window back out to DDR, word at a time
        BUNDLE(I("add r7, r3, 0", E.add("r7", "r3", 0))),
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0))),
        # .Lcopy: 5 bundles
        BUNDLE(I("ld r9, [r8+0]", E.ld("r9", "r8", 0))),
        BUNDLE(I("add r8, r8, 4", E.add("r8", "r8", 4))),
        BUNDLE(I("st r9, [r2+0]", E.st("r9", "r2", 0))),
        BUNDLE(I("add r2, r2, 4", E.add("r2", "r2", 4)),
               I("sub r7, r7, 1", E.sub("r7", "r7", 1))),
        BUNDLE(I("cbnz r7, .Lcopy", E.cbnz("r7", -4))),
    ]
    return prologue(6, fp=False) + body + epilogue()


def k_dma_mode(mode=E.DMA_DIRECT, int_base=E.DMA_LSRAM, ext_base=E.DMA_GLOBAL,
               dir_=E.DMA_EXT2INT, data_unit=E.DMA_USELESS, dst_off=0, flagprobe=False,
               flag=0, wait_flag=None, int_ddr=False, gate="full", swap_addr=False):
    """One `dma` request into LSRAM in any of the four modes, then a scalar copy-out.

    args: r0=SRC, r1=DESC, r2=OUT, r3=NWORDS, r4=POISON, r5=FLAGPROBE

    `r0` is the external-side operand; in `kMemset` it is the fill VALUE, not an address (the vendor's
    `MemsetSRAM` passes `value` where the direct copy passes `src_addr`). Descriptors for transpose,
    upsample and memset follow the vendor's ten-argument `init_dma_desc` and are built on the host
    by `dma.mode_desc_words`.

    NWORDS of LSRAM get ascending poison before the request and are all copied back, so unwritten,
    shifted and smeared bytes are all distinguishable. `gate="ends"` poisons and copies back only
    GATE_ENDS words at each end (constant cost, still detects truncation).

    `flagprobe` stores three `mfctrl0 0x30` reads (before the request, after it, after the `wfe`) to
    r5; index 0x30 is what the vendor's `GET_IDLE_FLAG()` reads. Read-only.

    `flag` deliberately skips `dma.check_flag` so an out-of-range flag's aliasing is observable via
    `flagprobe`. `wait_flag` defaults to `flag`.
    """
    wf = flag if wait_flag is None else wait_flag
    if gate not in ("full", "ends"):
        raise ValueError(f"gate must be 'full' or 'ends', not {gate!r}")
    if int_ddr:
        # Internal end in DDR, with no TEC-side poison: TEC stores land in the d-cache and the
        # epilogue's writeback after the transfer would overwrite the DMA output. The host poisons.
        body = [BUNDLE(I("add r6, r2, 0", E.add("r6", "r2", 0)))]
        if dst_off:
            body += [BUNDLE(I("add r6, r6, %d" % dst_off, E.add("r6", "r6", dst_off)))]
    else:
        body = _lsram_base("r6")
        if dst_off:
            body += [BUNDLE(I("add r6, r6, %d" % dst_off, E.add("r6", "r6", dst_off)))]
        if gate == "ends":
            # r10 = r6 + (NWORDS - GATE_ENDS)*4, the address of the last GATE_ENDS words (x4 as two doublings).
            body += [
                BUNDLE(I("sub r10, r3, %d" % GATE_ENDS, E.sub("r10", "r3", GATE_ENDS))),
                BUNDLE(I("add r10, r10, r10", E.add("r10", "r10", "r10"))),
                BUNDLE(I("add r10, r10, r10", E.add("r10", "r10", "r10"))),
                BUNDLE(I("add r10, r6, r10", E.add("r10", "r6", "r10"))),
                BUNDLE(I("add r9, r4, 0", E.add("r9", "r4", 0))),
            ]
            for i in range(GATE_ENDS):
                body += [BUNDLE(I("st r9, [r6+%d]" % (4 * i), E.st("r9", "r6", 4 * i))),
                         BUNDLE(I("add r9, r9, 1", E.add("r9", "r9", 1)))]
            for i in range(GATE_ENDS):
                body += [BUNDLE(I("st r9, [r10+%d]" % (4 * i), E.st("r9", "r10", 4 * i))),
                         BUNDLE(I("add r9, r9, 1", E.add("r9", "r9", 1)))]
        else:
            body += [
                # ---- poison the whole destination window: [r8] = r9, r9 ascending, NWORDS times
                BUNDLE(I("add r7, r3, 0", E.add("r7", "r3", 0)),
                       I("add r9, r4, 0", E.add("r9", "r4", 0))),
                BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0))),
                # .Lpoison: 4 bundles
                BUNDLE(I("st r9, [r8+0]", E.st("r9", "r8", 0))),
                BUNDLE(I("add r8, r8, 4", E.add("r8", "r8", 4)),
                       I("add r9, r9, 1", E.add("r9", "r9", 1))),
                BUNDLE(I("sub r7, r7, 1", E.sub("r7", "r7", 1))),
                BUNDLE(I("cbnz r7, .Lpoison", E.cbnz("r7", -3))),
            ]
    if flagprobe:
        body += [BUNDLE(I("mfctrl0 r12, 0x30", E.mfctrl0("r12", 0x30))),
                 BUNDLE(I("st r12, [r5+0]", E.st("r12", "r5", 0)))]
    # `rsync` = (flag << 8) | flag in a register; `zero` for flag 0.
    if flag:
        rs, rsv = "r11", (flag << 8) | flag
        body += [BUNDLE(I("mov r11, %d" % rsv, E.mov("r11", rsv)))]
    else:
        rs = "zero"
    # `swap_addr` exchanges the INT and EXT address operands. To write into DDR the vendor uses
    # `kInt2Ext` with DDR on the external side (`MemsetDDR`, `GEN_DMA_UPSAMPLE_INT2EXT`). DDR on the
    # internal side works for kDirect/kTranspose (symmetric descriptor halves) but times out for
    # kMemset/kUpsample.
    ia, ea = ("r0", "r6") if swap_addr else ("r6", "r0")
    body += [
        BUNDLE(I("dma %d, %s, 0, %d, %d, %d, %d, r1, %s, %s"
                 % (mode, rs, int_base, ext_base, dir_, data_unit, ia, ea),
                 E.dma(mode, rs, 0, int_base, ext_base, dir_, data_unit, "r1", ia, ea))),
    ]
    if flagprobe:
        body += [BUNDLE(I("mfctrl0 r12, 0x30", E.mfctrl0("r12", 0x30))),
                 BUNDLE(I("st r12, [r5+4]", E.st("r12", "r5", 4)))]
    body += [
        # Wait mask ~(1 << wf), via `sub` from `zero` since ~x == -x - 1.
        BUNDLE(I("sub r7, zero, %d" % ((1 << wf) + 1), E.sub("r7", "zero", (1 << wf) + 1))),
        BUNDLE(I("wfe r7, 1", E.wfe("r7", 1))),
    ]
    if flagprobe:
        body += [BUNDLE(I("mfctrl0 r12, 0x30", E.mfctrl0("r12", 0x30))),
                 BUNDLE(I("st r12, [r5+8]", E.st("r12", "r5", 8)))]
    if not int_ddr and gate == "ends":
        # Gate both ends, not a prefix: a truncated transfer writes a correct head and no tail.
        # Cost is 2*GATE_ENDS words regardless of size.
        for i in range(GATE_ENDS):
            body += [BUNDLE(I("ld r9, [r6+%d]" % (4 * i), E.ld("r9", "r6", 4 * i))),
                     BUNDLE(I("st r9, [r2+%d]" % (4 * i), E.st("r9", "r2", 4 * i)))]
        for i in range(GATE_ENDS):
            body += [BUNDLE(I("ld r9, [r10+%d]" % (4 * i), E.ld("r9", "r10", 4 * i))),
                     BUNDLE(I("st r9, [r2+%d]" % (4 * (GATE_ENDS + i)),
                              E.st("r9", "r2", 4 * (GATE_ENDS + i))))]
    elif not int_ddr:
        body += [
            # ---- copy the whole destination window out to DDR, word at a time
            BUNDLE(I("add r7, r3, 0", E.add("r7", "r3", 0))),
            BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0))),
            # .Lcopy: 5 bundles
            BUNDLE(I("ld r9, [r8+0]", E.ld("r9", "r8", 0))),
            BUNDLE(I("add r8, r8, 4", E.add("r8", "r8", 4))),
            BUNDLE(I("st r9, [r2+0]", E.st("r9", "r2", 0))),
            BUNDLE(I("add r2, r2, 4", E.add("r2", "r2", 4)),
                   I("sub r7, r7, 1", E.sub("r7", "r7", 1))),
            BUNDLE(I("cbnz r7, .Lcopy", E.cbnz("r7", -4))),
        ]
    return prologue(6, fp=False) + body + epilogue()


# ------------------------------------------ the LSRAM feed rate ----
# Vector-load rate out of LSRAM with no address arithmetic in the loop's critical path. Every arm
# DMA-fills one LSRAM window from a seeded DDR source (host-built descriptor, flag 0) and then times
# a counted loop over it; all arms are gated.
LSRAM_WINDOW_MASK = 0x3FFF     # the vld arms fill 16 KiB and wrap their offsets inside it
GEMM_A_PANEL, GEMM_B_PANEL = 6144, 8192   # blocking B's halves: 24x128 and 32x128 fp16


def _dma_fill_and_wait(space="lsram"):
    """ext(DDR, r0) -> int(SRAM, r6), descriptor at r1, flag 0, then wait on flag 0.
    `space="gsram"` changes only int_base (LSRAM 4 -> shared 8)."""
    ib = E.DMA_SHARED if space == "gsram" else E.DMA_LSRAM
    return [
        BUNDLE(I("dma 0, zero, 0, %d, 0, 1, 0, r1, r6, r0" % ib,
                 E.dma(E.DMA_DIRECT, "zero", 0, ib, E.DMA_GLOBAL, E.DMA_EXT2INT,
                       E.DMA_USELESS, "r1", "r6", "r0"))),
        BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))),     # ~(1 << 0)
        BUNDLE(I("wfe r7, 1", E.wfe("r7", 1))),
    ]


def k_lsram_vld(per_bundle=2):
    """Vector loads out of a DMA-filled 16 KiB LSRAM window, `per_bundle` (1 or 2) per bundle.

    args: r0=SRC (16 KiB, seeded), r1=DESC (host-built, 16384 B, flag 0), r2=OUT, r3=LOOPS

    No address arithmetic in the loads' path: each 8-bundle half loads off one base register with
    immediates, while a 3-op chain in its first three bundles computes the other half's base
    (`r9 = (r9 + step) & 0x3FFF; other = r6 + r9`). A base is written 5 bundles before first use and
    3 after last use, so nothing relies on VLIW read-before-write.

    Gate: the second half's registers (the last window loaded) are stored to OUT; the host expects
    SRC at offset `((2*LOOPS - 1) * step) & 0x3FFF`, `step = 256 * per_bundle`. Proves the last loads only.
    """
    if per_bundle not in (1, 2):
        raise ValueError("per_bundle must be 1 or 2 (two load slots per bundle)")
    step = 256 * per_bundle
    nl = 8 * per_bundle
    body = _lsram_base("r6") + _dma_fill_and_wait() + [
        BUNDLE(I("mov r11, %d" % LSRAM_WINDOW_MASK, E.mov("r11", LSRAM_WINDOW_MASK))),
        BUNDLE(I("mov r9, 0", E.mov("r9", 0))),
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0))),
        BUNDLE(I("sub r10, r3, 1", E.sub("r10", "r3", 1))),
        BUNDLE(I("max r10, r10, zero", E.mx("r10", "r10", "zero"))),
        BUNDLE(I("loop r10", E.loop("r10"))),
    ]

    def half(base, first_t, other):
        chain = [I("add r9, r9, %d" % step, E.add("r9", "r9", step)),
                 I("and r9, r9, r11", E.and_("r9", "r9", "r11")),
                 I("add %s, r6, r9" % other, E.add(other, "r6", "r9"))]
        bs = []
        for b in range(8):
            ops = []
            for k in range(per_bundle):
                t, off = first_t + b * per_bundle + k, (b * per_bundle + k) * 32
                ops.append(I("ld t%d, [%s+%d]" % (t, base, off), E.vld("t%d" % t, base, off)))
            if b < 3:
                ops.append(chain[b])
            bs.append(BUNDLE(*ops))
        return bs

    body += half("r8", 0, "r12") + half("r12", nl, "r8")
    body += [BUNDLE(I("loopend", E.loopend()))]
    body += [BUNDLE(I("st t%d, [r2+%d]" % (nl + i, i * 32), E.vst("t%d" % (nl + i), "r2", i * 32)))
             for i in range(nl)]
    return prologue(4) + body + epilogue()


def k_gemm_inner(mma=True):
    """The fed GEMM's k-step: the 3x4 register block's inner loop over one staged pair of panels,
    with (`mma=True`) or without (`mma=False`, same 13 bundles with the `mma`s removed) the matrix unit.

    args: r0=SRC (A panel 6144 B then B panel 8192 B, packed in mma-block order),
          r1=DESC (host-built, 14336 B, flag 0), r2=OUT (C, 12 pairs x 64 B = 768 B, host
          pre-zeroed), r3=REPS

    C is 12 register pairs t0..t23 (pair (i, j) at t(2*(4i+j))), A's three 4x4 blocks in t24..t26,
    B's four in t27..t30. Per k-step: 7 `vld`, 12 `mma`, 13 bundles (12 would need an `mma` beside
    the loads that overwrite its operands). Each `mma` is at least one bundle after its later
    operand's load; the base updates ride in slot 1 of bundles 9 and 10.

    Hardware loop of 64 k-steps; the outer `cbnz` repeats REPS times, so C accumulates
    REPS * sum_k A_k @ B_k, exact for small-integer fp16 inputs. Not software-pipelined.
    """
    def A(i): return "t%d" % (24 + i)
    def Bv(j): return "t%d" % (27 + j)
    def MMA(i, j):
        c = 2 * (4 * i + j)
        return [I("mma t%d.fp32, t%d.fp32, %s.fp16, %s.fp16" % (c, c + 1, A(i), Bv(j)),
                  E.mma("t%d" % c, A(i), Bv(j)))] if mma else []
    def VA(i): return I("ld %s, [r8+%d]" % (A(i), 32 * i), E.vld(A(i), "r8", 32 * i))
    def VB(j): return I("ld %s, [r13+%d]" % (Bv(j), 32 * j), E.vld(Bv(j), "r13", 32 * j))
    kstep = [
        BUNDLE(VA(0), VB(0)),
        BUNDLE(VA(1), VB(1), *MMA(0, 0)),
        BUNDLE(VA(2), VB(2), *MMA(0, 1)),
        BUNDLE(VB(3), *MMA(1, 0)),
        BUNDLE(*MMA(1, 1)),
        BUNDLE(*MMA(0, 2)),
        BUNDLE(*MMA(1, 2)),
        BUNDLE(*MMA(2, 0)),
        BUNDLE(*MMA(2, 1)),
        BUNDLE(*MMA(2, 2), I("add r8, r8, 96", E.add("r8", "r8", 96))),
        BUNDLE(*MMA(0, 3), I("add r13, r13, 128", E.add("r13", "r13", 128))),
        BUNDLE(*MMA(1, 3)),
        BUNDLE(*MMA(2, 3)),
    ]
    assert len(kstep) == 13

    # C lives at OUT[0:768]; `vld`'s immediate reaches 512 B, so the upper third goes via r14.
    def cbase(t): return ("r2", t * 32) if t < 16 else ("r14", (t - 16) * 32)
    def cload(t):
        b, off = cbase(t)
        return I("ld t%d, [%s+%d]" % (t, b, off), E.vld("t%d" % t, b, off))
    def cstore(t):
        b, off = cbase(t)
        return I("st t%d, [%s+%d]" % (t, b, off), E.vst("t%d" % t, b, off))

    body = _lsram_base("r6") + _dma_fill_and_wait() + [
        BUNDLE(I("add r14, r2, 512", E.add("r14", "r2", 512))),
    ]
    body += [BUNDLE(cload(t), cload(t + 1)) for t in range(0, 24, 2)]
    body += [BUNDLE(I("mov r15, %d" % GEMM_A_PANEL, E.mov("r15", GEMM_A_PANEL)))]
    # .Louter: setup, loop, 13 k-step bundles, loopend, sub, cbnz -> the branch is 17 back
    body += [
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0)),
               I("add r13, r6, r15", E.add("r13", "r6", "r15")),
               I("mov r10, 63", E.mov("r10", 63))),
        BUNDLE(I("loop r10", E.loop("r10"))),
    ] + kstep + [
        BUNDLE(I("loopend", E.loopend())),
        BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1))),
        BUNDLE(I("cbnz r3, .Louter", E.cbnz("r3", -17))),
    ]
    body += [BUNDLE(cstore(t)) for t in range(24)]
    return prologue(4) + body + epilogue()


# ------------------------------------------------- pipelined k-steps ----
PIPE_A_PANEL = 6144                    # 64 k-steps x 3 tiles x 32 B
PIPE_B_OFF = PIPE_A_PANEL + 64         # B panel base: +64 so every (A_i, B_i) pair straddles bit 6


def k_gemm_pipe33(iters=16):
    """Pipelined 3x3 k-step: two `mma` per bundle, paired loads straddling bit 6, loads for step
    k+1 issued during step k's `mma`s, one base update per four steps.

    args: r0=SRC (16 KiB: A panel at 0, B panel at 6208, each 64 steps x 96 B),
          r1=DESC (16384 B, flag 0), r2=OUT (C, 9 pairs x 64 B = 576 B, host pre-zeroed), r3=REPS

    C pairs t0..t17 (pair (i,j) at t(2*(3i+j))); operand set 0 = t18..t23, set 1 = t24..t29, each
    A0 A1 A2 B0 B1 B2; steps alternate sets. Body: 18 bundles of {mma, mma [, vld, vld]} then
    {add rA, 384 ; add rB, 384}; loads for step s+1 land >= 2 bundles before its first `mma`.
    The last iteration's lookahead loads read 96 B past the A panel (inside LSRAM, never consumed).
    Each rep re-arms the bases and reloads set 0 for step 0."""
    def A(s, i): return "t%d" % (18 + 6 * s + i)
    def B(s, j): return "t%d" % (18 + 6 * s + 3 + j)
    def MMA(s, i, j):
        c = 2 * (3 * i + j)
        return I("mma t%d.fp32, t%d.fp32, %s.fp16, %s.fp16" % (c, c + 1, A(s, i), B(s, j)),
                 E.mma("t%d" % c, A(s, i), B(s, j)))
    def VA(s, k, i): return I("ld %s, [r8+%d]" % (A(s, i), 96 * k + 32 * i), E.vld(A(s, i), "r8", 96 * k + 32 * i))
    def VB(s, k, j): return I("ld %s, [r13+%d]" % (B(s, j), 96 * k + 32 * j), E.vld(B(s, j), "r13", 96 * k + 32 * j))

    # the 36 mma of four steps, in order; bundle b holds mma 2b and 2b+1
    mmas = [MMA(k % 2, i, j) for k in range(4) for i in range(3) for j in range(3)]
    # loads for step k+1 (k+1 == 4 is the next iteration's step 0, offset 384) go into set (k+1) % 2,
    # in the three bundles after step k-1's last read of that set
    start = {0: 0, 1: 5, 2: 9, 3: 14}
    loads = {}
    for k in range(4):
        s = (k + 1) % 2
        for i in range(3):
            loads[start[k] + i] = (VA(s, k + 1, i), VB(s, k + 1, i))
    kbody = []
    for b in range(18):
        ops = [mmas[2 * b], mmas[2 * b + 1]]
        if b in loads:
            ops += list(loads[b])
        kbody.append(BUNDLE(*ops))
    assert len(kbody) == 18
    kbody.append(BUNDLE(I("add r8, r8, 384", E.add("r8", "r8", 384)),
                        I("add r13, r13, 384", E.add("r13", "r13", 384))))

    def cbase(t): return ("r2", t * 32) if t < 16 else ("r14", (t - 16) * 32)
    def cload(t):
        b, off = cbase(t); return I("ld t%d, [%s+%d]" % (t, b, off), E.vld("t%d" % t, b, off))
    def cstore(t):
        b, off = cbase(t); return I("st t%d, [%s+%d]" % (t, b, off), E.vst("t%d" % t, b, off))

    body = _lsram_base("r6") + _dma_fill_and_wait() + [
        BUNDLE(I("add r14, r2, 512", E.add("r14", "r2", 512))),
        BUNDLE(I("mov r15, %d" % PIPE_B_OFF, E.mov("r15", PIPE_B_OFF))),
    ]
    body += [BUNDLE(cload(t), cload(t + 1)) for t in range(0, 18, 2)]
    # .Louter: re-arm the bases, load set 0 for step 0, hardware loop
    outer = [
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0)), I("add r13, r6, r15", E.add("r13", "r6", "r15"))),
        BUNDLE(VA(0, 0, 0), VB(0, 0, 0)),
        BUNDLE(VA(0, 0, 1), VB(0, 0, 1)),
        BUNDLE(VA(0, 0, 2), VB(0, 0, 2)),
        BUNDLE(I("mov r10, %d" % (iters - 1), E.mov("r10", iters - 1))),
        BUNDLE(I("loop r10", E.loop("r10"))),
    ] + kbody + [
        BUNDLE(I("loopend", E.loopend())),
        BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1))),
    ]
    outer.append(BUNDLE(I("cbnz r3, .Louter", E.cbnz("r3", -len(outer)))))
    body += outer
    body += [BUNDLE(cstore(t)) for t in range(18)]
    return prologue(4) + body + epilogue()


PIPE34_B_OFF = PIPE_A_PANEL + 64       # 3x4: B panel (128 B/step) base with bit 6 set
PIPE34_STEPS = 63                      # 3-step body x 21: the 9-bit immediate reaches step 3's loads


def k_gemm_pipe34(iters=PIPE34_STEPS // 3):
    """The 3x4 block pipelined in 31 registers. With no spare operand set, the order of the 12 `mma`
    frees each operand register early enough to reload it for the next step. Constraints: two mma
    per bundle; reload after last use and >= 2 bundles before first use; <= 2 loads per bundle;
    paired loads straddle bit 6 (allowing only (A0,A2), (B0,B2), (B0,B3), (B1,B2), (B1,B3)):

      b0: mma(0,0) mma(0,2)  vld B1,B3 (this step)      b3: mma(2,0) mma(2,2)  vld A0 (next)
      b1: mma(1,0) mma(1,2)  vld A2   (this step)      b4: mma(1,1) mma(1,3)  vld B0,B2 (next)
      b2: mma(0,1) mma(0,3)                            b5: mma(2,1) mma(2,3)  vld A1 (next)

    3-step body (18 bundles) + {add rA,288 ; add rB,384}.
    args: r0=SRC (16 KiB: A at 0, 63 x 96 B; B at 6208, 63 x 128 B), r1=DESC, r2=OUT (C, 12
    pairs = 768 B, host pre-zeroed), r3=REPS. C pairs t0..t23, A0..A2 = t24..t26, B0..B3 =
    t27..t30. Each rep re-arms the bases and preloads A0, A1, B0, B2 for step 0."""
    ORDER = [(0, 0), (0, 2), (1, 0), (1, 2), (0, 1), (0, 3), (2, 0), (2, 2), (1, 1), (1, 3), (2, 1), (2, 3)]
    THIS = {0: ["B1", "B3"], 1: ["A2"]}                       # loads for the current step
    NEXT = {3: ["A0"], 4: ["B0", "B2"], 5: ["A1"]}            # loads for the next step
    REG = {"A0": "t24", "A1": "t25", "A2": "t26", "B0": "t27", "B1": "t28", "B2": "t29", "B3": "t30"}
    def MMA(i, j):
        c = 2 * (4 * i + j)
        return I("mma t%d.fp32, t%d.fp32, %s.fp16, %s.fp16" % (c, c + 1, REG["A%d" % i], REG["B%d" % j]),
                 E.mma("t%d" % c, REG["A%d" % i], REG["B%d" % j]))
    def LD(x, k):
        if x[0] == "A": base, off = "r8", 96 * k + 32 * int(x[1])
        else:           base, off = "r13", 128 * k + 32 * int(x[1])
        return I("ld %s, [%s+%d]" % (REG[x], base, off), E.vld(REG[x], base, off))
    kbody = []
    for k in range(3):
        for b in range(6):
            ops = [MMA(*ORDER[2 * b]), MMA(*ORDER[2 * b + 1])]
            ops += [LD(x, k) for x in THIS.get(b, [])]
            ops += [LD(x, k + 1) for x in NEXT.get(b, [])]
            kbody.append(BUNDLE(*ops))
    kbody.append(BUNDLE(I("add r8, r8, 288", E.add("r8", "r8", 288)),
                        I("add r13, r13, 384", E.add("r13", "r13", 384))))

    def cbase(t): return ("r2", t * 32) if t < 16 else ("r14", (t - 16) * 32)
    def cload(t):
        b, off = cbase(t); return I("ld t%d, [%s+%d]" % (t, b, off), E.vld("t%d" % t, b, off))
    def cstore(t):
        b, off = cbase(t); return I("st t%d, [%s+%d]" % (t, b, off), E.vst("t%d" % t, b, off))
    body = _lsram_base("r6") + _dma_fill_and_wait() + [
        BUNDLE(I("add r14, r2, 512", E.add("r14", "r2", 512))),
        BUNDLE(I("mov r15, %d" % PIPE34_B_OFF, E.mov("r15", PIPE34_B_OFF))),
    ]
    body += [BUNDLE(cload(t), cload(t + 1)) for t in range(0, 24, 2)]
    outer = [
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0)), I("add r13, r6, r15", E.add("r13", "r6", "r15"))),
        BUNDLE(LD("B0", 0), LD("B2", 0)),
        BUNDLE(LD("A0", 0)),
        BUNDLE(LD("A1", 0)),
        BUNDLE(I("mov r10, %d" % (iters - 1), E.mov("r10", iters - 1))),
        BUNDLE(I("loop r10", E.loop("r10"))),
    ] + kbody + [
        BUNDLE(I("loopend", E.loopend())),
        BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1))),
    ]
    outer.append(BUNDLE(I("cbnz r3, .Louter", E.cbnz("r3", -len(outer)))))
    body += outer
    body += [BUNDLE(cstore(t)) for t in range(24)]
    return prologue(4) + body + epilogue()


HALF = 16384


def k_gemm_pipe34_db(fill_bytes, prefill_upper=False):
    """`k_gemm_pipe34` computing on one 16 KiB half while a DMA of `fill_bytes` fills the other.
    One flag: the fill is requested before the body and waited on after it, then the halves swap;
    the next rep computes on data only that DMA could have written, so the C gate proves every fill.

    args: r0=SRC (16 KiB; A at 0, B at 6208), r1=DESC (six words for the 16 KiB prologue fill,
    and at r1+64 six words for `fill_bytes`), r2=OUT (C, 768 B), r3=REPS.
    r6 = LSRAM base, r9 = compute half, r11 = other half, r16 = r1 + 64."""
    ORDER = [(0, 0), (0, 2), (1, 0), (1, 2), (0, 1), (0, 3), (2, 0), (2, 2), (1, 1), (1, 3), (2, 1), (2, 3)]
    THIS = {0: ["B1", "B3"], 1: ["A2"]}
    NEXT = {3: ["A0"], 4: ["B0", "B2"], 5: ["A1"]}
    REG = {"A0": "t24", "A1": "t25", "A2": "t26", "B0": "t27", "B1": "t28", "B2": "t29", "B3": "t30"}
    def MMA(i, j):
        c = 2 * (4 * i + j)
        return I("mma t%d.fp32, t%d.fp32, %s.fp16, %s.fp16" % (c, c + 1, REG["A%d" % i], REG["B%d" % j]),
                 E.mma("t%d" % c, REG["A%d" % i], REG["B%d" % j]))
    def LD(x, k):
        if x[0] == "A": base, off = "r8", 96 * k + 32 * int(x[1])
        else:           base, off = "r13", 128 * k + 32 * int(x[1])
        return I("ld %s, [%s+%d]" % (REG[x], base, off), E.vld(REG[x], base, off))
    kbody = []
    for k in range(3):
        for b in range(6):
            ops = [MMA(*ORDER[2 * b]), MMA(*ORDER[2 * b + 1])]
            ops += [LD(x, k) for x in THIS.get(b, [])] + [LD(x, k + 1) for x in NEXT.get(b, [])]
            kbody.append(BUNDLE(*ops))
    kbody.append(BUNDLE(I("add r8, r8, 288", E.add("r8", "r8", 288)), I("add r13, r13, 384", E.add("r13", "r13", 384))))
    def cbase(t): return ("r2", t * 32) if t < 16 else ("r14", (t - 16) * 32)
    def cload(t):
        b, off = cbase(t); return I("ld t%d, [%s+%d]" % (t, b, off), E.vld("t%d" % t, b, off))
    def cstore(t):
        b, off = cbase(t); return I("st t%d, [%s+%d]" % (t, b, off), E.vst("t%d" % t, b, off))
    body = _lsram_base("r6") + _dma_fill_and_wait() + [            # lower half filled, flag 0 idle
        BUNDLE(I("add r14, r2, 512", E.add("r14", "r2", 512))),
        BUNDLE(I("mov r15, %d" % PIPE34_B_OFF, E.mov("r15", PIPE34_B_OFF))),
        BUNDLE(I("add r9, r6, 0", E.add("r9", "r6", 0))),           # compute half
        BUNDLE(I("mov r11, %d" % HALF, E.mov("r11", HALF))),
        BUNDLE(I("add r11, r6, r11", E.add("r11", "r6", "r11"))),  # other half
        BUNDLE(I("add r16, r1, 64", E.add("r16", "r1", 64))),      # the in-loop descriptor
    ]
    if prefill_upper:   # a partial in-loop fill must be content-preserving: fill the upper half once, fully
        body += [BUNDLE(I("dma 0, zero, 0, 4, 0, 1, 0, r1, r11, r0",
                          E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT, E.DMA_USELESS, "r1", "r11", "r0"))),
                 BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))),
                 BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    body += [BUNDLE(cload(t), cload(t + 1)) for t in range(0, 24, 2)]
    outer = [
        BUNDLE(I("dma 0, zero, 0, 4, 0, 1, 0, r16, r11, r0",
                 E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT, E.DMA_USELESS, "r16", "r11", "r0"))),
        BUNDLE(I("add r8, r9, 0", E.add("r8", "r9", 0)), I("add r13, r9, r15", E.add("r13", "r9", "r15"))),
        BUNDLE(LD("B0", 0), LD("B2", 0)),
        BUNDLE(LD("A0", 0)),
        BUNDLE(LD("A1", 0)),
        BUNDLE(I("mov r10, %d" % (PIPE34_STEPS // 3 - 1), E.mov("r10", PIPE34_STEPS // 3 - 1))),
        BUNDLE(I("loop r10", E.loop("r10"))),
    ] + kbody + [
        BUNDLE(I("loopend", E.loopend())),
        BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))),
        BUNDLE(I("wfe r7, 1", E.wfe("r7", 1))),                    # the other half has landed
        BUNDLE(I("add r17, r9, 0", E.add("r17", "r9", 0))),         # swap halves
        BUNDLE(I("add r9, r11, 0", E.add("r9", "r11", 0))),
        BUNDLE(I("add r11, r17, 0", E.add("r11", "r17", 0))),
        BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1))),
    ]
    outer.append(BUNDLE(I("cbnz r3, .Louter", E.cbnz("r3", -len(outer)))))
    body += outer
    body += [BUNDLE(cstore(t)) for t in range(24)]
    return prologue(4) + body + epilogue()


PIPE34V3_STEP = 256      # one base register: A0 A1 A2 at +0 +32 +64, B0..B3 at +128..+224, per step


def k_gemm_pipe34_v3():
    """`k_gemm_pipe34` with the base update in the loopend bundle and one base register (A and B
    interleaved in a 256 B step). Negative result: produces wrong answers, because an op beside
    `loopend` executes once, on loop exit, so r8 never advances inside the loop.
    args: r0=SRC (63 x 256 B), r1=DESC (16384 B), r2=OUT (C, 768 B), r3=REPS."""
    ORDER = [(0, 0), (0, 2), (1, 0), (1, 2), (0, 1), (0, 3), (2, 0), (2, 2), (1, 1), (1, 3), (2, 1), (2, 3)]
    THIS = {0: ["B1", "B3"], 1: ["A2"]}
    NEXT = {3: ["A0"], 4: ["B0", "B2"], 5: ["A1"]}
    REG = {"A0": "t24", "A1": "t25", "A2": "t26", "B0": "t27", "B1": "t28", "B2": "t29", "B3": "t30"}
    OFF = {"A0": 0, "A1": 32, "A2": 64, "B0": 128, "B1": 160, "B2": 192, "B3": 224}
    def MMA(i, j):
        c = 2 * (4 * i + j)
        return I("mma t%d.fp32, t%d.fp32, %s.fp16, %s.fp16" % (c, c + 1, REG["A%d" % i], REG["B%d" % j]),
                 E.mma("t%d" % c, REG["A%d" % i], REG["B%d" % j]))
    def LD(x, k): return I("ld %s, [r8+%d]" % (REG[x], PIPE34V3_STEP * k + OFF[x]), E.vld(REG[x], "r8", PIPE34V3_STEP * k + OFF[x]))
    kbody = []
    for b in range(6):
        ops = [MMA(*ORDER[2 * b]), MMA(*ORDER[2 * b + 1])]
        ops += [LD(x, 0) for x in THIS.get(b, [])] + [LD(x, 1) for x in NEXT.get(b, [])]
        kbody.append(BUNDLE(*ops))
    def cbase(t): return ("r2", t * 32) if t < 16 else ("r14", (t - 16) * 32)
    def cload(t):
        b, off = cbase(t); return I("ld t%d, [%s+%d]" % (t, b, off), E.vld("t%d" % t, b, off))
    def cstore(t):
        b, off = cbase(t); return I("st t%d, [%s+%d]" % (t, b, off), E.vst("t%d" % t, b, off))
    body = _lsram_base("r6") + _dma_fill_and_wait() + [BUNDLE(I("add r14, r2, 512", E.add("r14", "r2", 512)))]
    body += [BUNDLE(cload(t), cload(t + 1)) for t in range(0, 24, 2)]
    outer = [
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0))),
        BUNDLE(LD("B0", 0), LD("B2", 0)),
        BUNDLE(LD("A0", 0)),
        BUNDLE(LD("A1", 0)),
        BUNDLE(I("mov r10, %d" % (PIPE34_STEPS - 1), E.mov("r10", PIPE34_STEPS - 1))),
        BUNDLE(I("loop r10", E.loop("r10"))),
    ] + kbody + [
        BUNDLE(I("loopend", E.loopend()), I("add r8, r8, %d" % PIPE34V3_STEP, E.add("r8", "r8", PIPE34V3_STEP))),
        BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1))),
    ]
    outer.append(BUNDLE(I("cbnz r3, .Louter", E.cbnz("r3", -len(outer)))))
    body += outer
    body += [BUNDLE(cstore(t)) for t in range(24)]
    return prologue(4) + body + epilogue()


# ------------------------------------- LSRAM load-rate variants ----
# Each variant changes one property of the vector-load loop (load form, base registers, address
# pattern) to locate the 1 load/cycle limit; no scalar chain, fixed immediates only.
PORT_SCRATCH = ["r4", "r5", "r9", "r11", "r14", "r15", "r16", "r17", "r18", "r19", "r20", "r21"]


def k_lsram_port(variant, stride=0, space="lsram"):
    """16-bundle hardware loop of loads from a DMA-filled 16 KiB LSRAM window, one property varied.

    args: r0=SRC (16 KiB), r1=DESC (16384 B, flag 0), r2=OUT, r3=LOOPS. r8 = window base,
    r12 = r8 + stride (second base).
      v1     one `vld`/bundle, t(b) <- [r8+32b]
      v2     two `vld`/bundle, one base: t(2b),t(2b+1) <- [r8+64b],[r8+64b+32]
             (bundles 8..15 off r12 = r8+512: the immediate stops at 511)
      v2s    two bases: t(2b) <- [r8+32b], t(2b+1) <- [r12+32b]
      v2same same address twice
      s2     two scalar `ld` (4 B) into the PORT_SCRATCH GPRs, cycling
      vs     `vld` + scalar `ld`: t(b) <- [r8+32b], gpr(b%12) <- [r12+4b]
    OUT: vector registers (32 B each), then scratch GPRs at OUT+1024."""
    if variant not in ("v1", "v2", "v2s", "v2same", "s2", "vs"):
        raise ValueError(variant)
    body = (_gsram_base("r6") if space == "gsram" else _lsram_base("r6")) + _dma_fill_and_wait(space)
    if stride >= 16384:   # second load reads the upper half: fill it from the same SRC
        body += [BUNDLE(I("mov r11, 16384", E.mov("r11", 16384))),
                 BUNDLE(I("add r9, r6, r11", E.add("r9", "r6", "r11"))),
                 BUNDLE(I("dma 0, zero, 0, 4, 0, 1, 0, r1, r9, r0",
                          E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT,
                                E.DMA_USELESS, "r1", "r9", "r0"))),
                 BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))),
                 BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    body += [
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0))),
        BUNDLE(I("mov r11, %d" % stride, E.mov("r11", stride))),
        BUNDLE(I("add r12, r8, r11", E.add("r12", "r8", "r11"))),
        BUNDLE(I("sub r10, r3, 1", E.sub("r10", "r3", 1))),
        BUNDLE(I("max r10, r10, zero", E.mx("r10", "r10", "zero"))),
        BUNDLE(I("loop r10", E.loop("r10"))),
    ]

    def V(t, b, off): return I("ld t%d, [%s+%d]" % (t, b, off), E.vld("t%d" % t, b, off))
    def S(r, b, off): return I("ld %s, [%s+%d]" % (r, b, off), E.ld(r, b, off))

    nvec, nscalar = 0, 0
    for b in range(16):
        if variant == "v1":
            body.append(BUNDLE(V(b, "r8", 32 * b))); nvec = 16
        elif variant == "v2":
            base, o = ("r8", 64 * b) if b < 8 else ("r12", 64 * (b - 8))
            body.append(BUNDLE(V(2 * b, base, o), V(2 * b + 1, base, o + 32))); nvec = 32
        elif variant == "v2s":
            body.append(BUNDLE(V(2 * b, "r8", 32 * b), V(2 * b + 1, "r12", 32 * b))); nvec = 32
        elif variant == "v2same":
            body.append(BUNDLE(V(2 * b, "r8", 32 * b), V(2 * b + 1, "r8", 32 * b))); nvec = 32
        elif variant == "s2":
            i = 2 * b
            body.append(BUNDLE(S(PORT_SCRATCH[i % 12], "r8", 4 * i),
                               S(PORT_SCRATCH[(i + 1) % 12], "r8", 4 * (i + 1)))); nscalar = 12
        elif variant == "vs":
            body.append(BUNDLE(V(b, "r8", 32 * b), S(PORT_SCRATCH[b % 12], "r12", 4 * b)))
            nvec, nscalar = 16, 12
    body += [BUNDLE(I("loopend", E.loopend()))]
    body += [BUNDLE(I("st t%d, [r2+%d]" % (t, 32 * t), E.vst("t%d" % t, "r2", 32 * t)))
             for t in range(min(nvec, 16))]
    if nvec > 16:   # immediate stops at 511: t16..t31 go via r13
        body += [BUNDLE(I("add r13, r2, 512", E.add("r13", "r2", 512)))]
        body += [BUNDLE(I("st t%d, [r13+%d]" % (t, 32 * (t - 16)), E.vst("t%d" % t, "r13", 32 * (t - 16))))
                 for t in range(16, nvec)]
    if nscalar:   # `add` immediate is 10-bit: 1024 goes via r13 (not a scratch GPR; those hold results)
        body += [BUNDLE(I("mov r13, 1024", E.mov("r13", 1024))),
                 BUNDLE(I("add r13, r2, r13", E.add("r13", "r2", "r13")))]
        body += [BUNDLE(I("st %s, [r13+%d]" % (PORT_SCRATCH[k], 4 * k), E.st(PORT_SCRATCH[k], "r13", 4 * k)))
                 for k in range(nscalar)]
    return prologue(4) + body + epilogue()


UPPER = 16384          # upper 16 KiB of LSRAM: never DMA-filled by the prologue
ST_FAR = 1024 + 64     # second store region: non-overlapping and bit 6 differs


# Mixed load-stream patterns: cost of same-bank pairs among other bundles, with optional slot-0
# companions. 16 bundle codes, o = 32*(b%4), one base, 9-bit immediate:
#   c  same-bank pair [r8+o],[r8+o+128]   s  straddling pair [r8+o],[r8+o+64]
#   1  single load [r8+o]                 0  empty bundle
# Loads write t(i%16) in issue order; readback stores t0..t15. mma companions accumulate into
# pairs t16..t23 (reuse distance 4) from t28/t29 loaded once.
LSRAM_PATTERNS = {
    "cc":   "c" * 16,  "ss": "s" * 16,  "11": "1" * 16,
    "cs":   "cs" * 8,  "sc": "sc" * 8,
    "ccss": "ccss" * 4, "csss": "csss" * 4, "sssc": "sssc" * 4,
    "c1":   "c1" * 8,  "s1": "s1" * 8,  "c0": "c0" * 8,  "c11": "c11" * 5 + "c",
    "c1s1": "c1s1" * 4, "cs11": "cs11" * 4, "c111": "c111" * 4, "cssc": "cssc" * 4,
    "00":   "0" * 16,      # no loads: ALU-only controls
}


def k_lsram_pattern(name, companion=None):
    """Load stream from `LSRAM_PATTERNS[name]` with an optional ALU companion per bundle.
    args: r0=SRC (16 KiB), r1=DESC, r2=OUT, r3=LOOPS."""
    pat = LSRAM_PATTERNS[name]
    #   add/addrot  `add r14, r15, 1` per bundle (addrot rotates r14/r16/r17/r18 to avoid WAW)
    #   xor/xorrot  `xor r14, r15, r15` (slot-1-only op)
    #   mma/mmaadd  one mma per bundle over 4 accumulator pairs (+ add in slot 1)
    #   mma8/mmaadd8  mma over 8 pairs t12..t27; loads then write t(i%12), readback t0..t11
    #   mma2x8  two mma per bundle over the 8 pairs
    assert len(pat) == 16 and companion in (None, "add", "addrot", "xor", "xorrot", "mma", "mmaadd", "mma8", "mmaadd8", "mma2x8")
    nreg = 12 if companion in ("mma8", "mmaadd8", "mma2x8") else 16
    def V(t, off): return I("ld t%d, [r8+%d]" % (t, off), E.vld("t%d" % t, "r8", off))
    body = _lsram_base("r6") + _dma_fill_and_wait() + [BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0)))]
    if companion in ("mma", "mmaadd", "mma8", "mmaadd8", "mma2x8"):
        body += [BUNDLE(I("ld t28, [r8+0]", E.vld("t28", "r8", 0)), I("ld t29, [r8+32]", E.vld("t29", "r8", 32)))]
        lo = 12 if nreg == 12 else 16
        body += [BUNDLE(V(t, 64), V(t + 1, 96)) for t in range(lo, 28, 2)]     # accumulators: any finite data
    body += [BUNDLE(I("mov r15, 0", E.mov("r15", 0))),
             BUNDLE(I("sub r10, r3, 1", E.sub("r10", "r3", 1))),
             BUNDLE(I("max r10, r10, zero", E.mx("r10", "r10", "zero"))),
             BUNDLE(I("loop r10", E.loop("r10")))]
    i = 0
    for b, code in enumerate(pat):
        o = 32 * (b % 4)
        offs = {"c": (o, o + 128), "s": (o, o + 64), "1": (o,), "0": ()}[code]
        ops = []
        for off in offs:
            ops.append(V(i % nreg, off)); i += 1
        if companion == "add":
            ops.append(I("add r14, r15, 1", E.add("r14", "r15", 1)))
        elif companion == "addrot":
            rd = ("r14", "r16", "r17", "r18")[b % 4]
            ops.append(I("add %s, r15, 1" % rd, E.add(rd, "r15", 1)))
        elif companion == "xor":
            ops.append(I("xor r14, r15, r15", E.xor("r14", "r15", "r15")))
        elif companion == "xorrot":
            rd = ("r14", "r16", "r17", "r18")[b % 4]
            ops.append(I("xor %s, r15, r15" % rd, E.xor(rd, "r15", "r15")))
        elif companion == "mma2x8":
            for k in (0, 1):
                acc = 12 + 2 * ((2 * b + k) % 8)
                ops.append(I("mma t%d.fp32, t%d.fp32, t28.fp16, t29.fp16" % (acc, acc + 1), E.mma("t%d" % acc, "t28", "t29")))
        elif companion in ("mma", "mmaadd", "mma8", "mmaadd8"):
            acc = (16 + 2 * (b % 4)) if nreg == 16 else (12 + 2 * (b % 8))
            ops.append(I("mma t%d.fp32, t%d.fp32, t28.fp16, t29.fp16" % (acc, acc + 1), E.mma("t%d" % acc, "t28", "t29")))
            if companion == "mmaadd":
                ops.append(I("add r14, r15, 1", E.add("r14", "r15", 1)))
            elif companion == "mmaadd8":
                rd = ("r14", "r16", "r17", "r18")[b % 4]
                ops.append(I("add %s, r15, 1" % rd, E.add(rd, "r15", 1)))
        body.append(BUNDLE(*ops))
    body += [BUNDLE(I("loopend", E.loopend()))]
    body += [BUNDLE(I("st t%d, [r2+%d]" % (t, 32 * t), E.vst("t%d" % t, "r2", 32 * t))) for t in range(nreg)]
    return prologue(4) + body + epilogue()


def k_lsram_loop(n):
    """Hardware loop of `n` empty bundles: isolates `loop`/`loopend` overhead.
    args as k_lsram_port; stores nothing (timing control)."""
    body = _lsram_base("r6") + _dma_fill_and_wait() + [
        BUNDLE(I("sub r10, r3, 1", E.sub("r10", "r3", 1))),
        BUNDLE(I("max r10, r10, zero", E.mx("r10", "r10", "zero"))),
        BUNDLE(I("loop r10", E.loop("r10")))]
    body += [BUNDLE() for _ in range(n)]
    body += [BUNDLE(I("loopend", E.loopend()))]
    return prologue(4) + body + epilogue()


def k_lsram_loop_add(n):
    """`k_lsram_loop(n)` with `add r14, r15, 1` in the loopend bundle's slot 1 (timing control)."""
    body = _lsram_base("r6") + _dma_fill_and_wait() + [
        BUNDLE(I("mov r15, 0", E.mov("r15", 0))),
        BUNDLE(I("sub r10, r3, 1", E.sub("r10", "r3", 1))),
        BUNDLE(I("max r10, r10, zero", E.mx("r10", "r10", "zero"))),
        BUNDLE(I("loop r10", E.loop("r10")))]
    body += [BUNDLE() for _ in range(n)]
    body += [BUNDLE(I("loopend", E.loopend()), I("add r14, r15, 1", E.add("r14", "r15", 1)))]
    return prologue(4) + body + epilogue()


def k_lsram_loop_cnt(where):
    """Counter `add r14, r14, 1` in the loopend bundle ("end") or the last body bundle ("body");
    OUT[0] = r14, which equals LOOPS iff it executed every iteration."""
    body = _lsram_base("r6") + _dma_fill_and_wait() + [
        BUNDLE(I("mov r14, 0", E.mov("r14", 0))),
        BUNDLE(I("sub r10, r3, 1", E.sub("r10", "r3", 1))),
        BUNDLE(I("max r10, r10, zero", E.mx("r10", "r10", "zero"))),
        BUNDLE(I("loop r10", E.loop("r10")))]
    body += [BUNDLE() for _ in range(15)]
    if where == "body":
        body += [BUNDLE(I("add r14, r14, 1", E.add("r14", "r14", 1))), BUNDLE(I("loopend", E.loopend()))]
    else:
        body += [BUNDLE(), BUNDLE(I("loopend", E.loopend()), I("add r14, r14, 1", E.add("r14", "r14", 1)))]
    body += [BUNDLE(I("st r14, [r2+0]", E.st("r14", "r2", 0)))]
    return prologue(4) + body + epilogue()


def k_lsram_gpr_raw(D):
    """GPR write -> load-base RAW distance. Four groups per iteration:
    {add r14, r15, 32g}, D-1 empty bundles, {ld t(26+g), [r14+0]}. A stale r14 yields the
    previous group's tile in t(26+g); t26..t29 go to OUT."""
    def V(t, off): return I("ld t%d, [r8+%d]" % (t, off), E.vld("t%d" % t, "r8", off))
    body = _lsram_base("r6") + _dma_fill_and_wait() + [
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0))),
        BUNDLE(I("add r15, r6, 0", E.add("r15", "r6", 0))),
        BUNDLE(I("add r14, r6, 0", E.add("r14", "r6", 0))),
        BUNDLE(I("sub r10, r3, 1", E.sub("r10", "r3", 1))),
        BUNDLE(I("max r10, r10, zero", E.mx("r10", "r10", "zero"))),
        BUNDLE(I("loop r10", E.loop("r10")))]
    for g in range(4):
        body.append(BUNDLE(I("add r14, r15, %d" % (32 * g), E.add("r14", "r15", 32 * g))))
        body += [BUNDLE() for _ in range(D - 1)]
        body.append(BUNDLE(I("ld t%d, [r14+0]" % (26 + g), E.vld("t%d" % (26 + g), "r14", 0))))
    body += [BUNDLE(I("loopend", E.loopend()))]
    body += [BUNDLE(I("st t%d, [r2+%d]" % (26 + g, 32 * g), E.vst("t%d" % (26 + g), "r2", 32 * g))) for g in range(4)]
    return prologue(4) + body + epilogue()


def k_lsram_latency(D, space="lsram"):
    """vld -> mma load-to-use latency. Four groups per iteration: {vld tX,[r8+o]}, D-1 empty
    bundles, {mma acc, tX.fp16, t30.fp16}. With latency L a group costs D + 2 + max(0, L - D).
    X rotates t26..t29, acc over pairs t16..t23, t30 constant. args as k_lsram_port."""
    assert 1 <= D <= 6
    def V(t, off): return I("ld t%d, [r8+%d]" % (t, off), E.vld("t%d" % t, "r8", off))
    body = (_gsram_base("r6") if space == "gsram" else _lsram_base("r6")) + _dma_fill_and_wait(space) + [
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0))),
        BUNDLE(I("ld t30, [r8+0]", E.vld("t30", "r8", 0))),
    ]
    body += [BUNDLE(V(t, 64), V(t + 1, 96)) for t in range(16, 24, 2)]
    body += [BUNDLE(I("sub r10, r3, 1", E.sub("r10", "r3", 1))),
             BUNDLE(I("max r10, r10, zero", E.mx("r10", "r10", "zero"))),
             BUNDLE(I("loop r10", E.loop("r10")))]
    for g in range(4):
        x, acc = 26 + g, 16 + 2 * g
        body.append(BUNDLE(V(x, 32 * g)))
        body += [BUNDLE() for _ in range(D - 1)]
        body.append(BUNDLE(I("mma t%d.fp32, t%d.fp32, t%d.fp16, t30.fp16" % (acc, acc + 1, x),
                             E.mma("t%d" % acc, "t%d" % x, "t30"))))
    body += [BUNDLE(I("loopend", E.loopend()))]
    body += [BUNDLE(I("st t%d, [r2+%d]" % (26 + g, 32 * g), E.vst("t%d" % (26 + g), "r2", 32 * g))) for g in range(4)]
    return prologue(4) + body + epilogue()


def k_lsram_dma_contend(variant, space="lsram"):
    """Load stream over the lower LSRAM half while a DMA fills the upper half (DMA/port contention).

    args: r0=SRC (16 KiB), r1=DESC (16384 B), r2=OUT, r3=OUTER. Outer `cbnz` loop, OUTER times:
      [dma]  SRC -> UPPER half (r9 = base + 16384), flag 0          (dma arms)
      inner  64 x 16 bundles of bit-6-apart `vld` pairs (64 B/cycle); empty bundles for dmaonly
      [wfe]  wait for flag 0                                         (dma arms)
    Arms: dmaload, loadonly, dmaonly; *1 = one `vld`/bundle; *_<pat> = 's'/'1' code body.
    OUT[0:1024]: load registers; dma arms add the upper half's first 512 B at OUT[1024:1536]."""
    pat = None
    if "_" in variant:
        variant, pat = variant.split("_", 1)
        pat = {"s1": "s1" * 8, "ss1": "ss1" * 5 + "s", "s11": "s11" * 5 + "s", "sss1": "sss1" * 4}[pat]
    if variant not in ("dmaload", "loadonly", "dmaonly", "dmaload1", "loadonly1"):
        raise ValueError(variant)
    has_dma, has_ld = variant in ("dmaload", "dmaonly", "dmaload1"), variant != "dmaonly"
    single = variant.endswith("1")

    def V(t, b, off): return I("ld t%d, [%s+%d]" % (t, b, off), E.vld("t%d" % t, b, off))
    def W(t, b, off): return I("st t%d, [%s+%d]" % (t, b, off), E.vst("t%d" % t, b, off))
    body = (_gsram_base("r6") if space == "gsram" else _lsram_base("r6")) + _dma_fill_and_wait(space) + [
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0))),
        BUNDLE(I("add r12, r6, 64", E.add("r12", "r6", 64))),
        BUNDLE(I("mov r11, %d" % UPPER, E.mov("r11", UPPER))),
        BUNDLE(I("add r9, r6, r11", E.add("r9", "r6", "r11"))),
    ]
    outer = []
    if has_dma:
        # the in-loop request must target `space` too, or the readback sees stale bytes
        _ib = E.DMA_SHARED if space == "gsram" else E.DMA_LSRAM
        outer += [BUNDLE(I("dma 0, zero, 0, %d, 0, 1, 0, r1, r9, r0" % _ib,
                           E.dma(E.DMA_DIRECT, "zero", 0, _ib, E.DMA_GLOBAL, E.DMA_EXT2INT,
                                 E.DMA_USELESS, "r1", "r9", "r0")))]
    outer += [BUNDLE(I("mov r10, 63", E.mov("r10", 63))), BUNDLE(I("loop r10", E.loop("r10")))]
    n = 0
    for b in range(16):
        if has_ld and pat is not None:
            if pat[b] == "s":
                outer.append(BUNDLE(V(n % 32, "r8", 32 * b), V((n + 1) % 32, "r12", 32 * b))); n += 2
            else:
                outer.append(BUNDLE(V(n % 32, "r8", 32 * b))); n += 1
        elif has_ld and single:
            outer.append(BUNDLE(V(b, "r8", 32 * b)))
        elif has_ld:
            outer.append(BUNDLE(V(2 * b, "r8", 32 * b), V(2 * b + 1, "r12", 32 * b)))
        else:
            outer.append(BUNDLE())      # empty bundle; emitted as `nop`
    outer += [BUNDLE(I("loopend", E.loopend()))]
    if has_dma:
        outer += [BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))),
                  BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    outer += [BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1)))]
    outer += [BUNDLE(I("cbnz r3, .Louter", E.cbnz("r3", -len(outer))))]
    body += outer
    if has_ld:
        body += [BUNDLE(W(t, "r2", 32 * t)) for t in range(16)]
    if has_ld and (pat is not None or not single):
        body += [BUNDLE(I("add r13, r2, 512", E.add("r13", "r2", 512)))]
        body += [BUNDLE(W(t, "r13", 32 * (t - 16))) for t in range(16, 32)]
    if has_dma:
        body += [BUNDLE(V(i, "r9", 32 * i)) for i in range(16)]
        body += [BUNDLE(I("mov r13, 1024", E.mov("r13", 1024))),
                 BUNDLE(I("add r13, r2, r13", E.add("r13", "r2", "r13")))]
        body += [BUNDLE(W(i, "r13", 32 * i)) for i in range(16)]
    return prologue(4) + body + epilogue()


def k_lsram_port2(variant, far=0, space="lsram"):
    """LSRAM stores, mixed load/store, a dependent-load chase, and the upper 16 KiB.
    ABI and loop shape as `k_lsram_port` (r0=SRC, r1=DESC, r2=OUT, r3=LOOPS).

    Store variants load t0..t15 from the filled lower half, store into the never-filled upper
    half (r13 = base + 16384), then read the stored region back to OUT.
      st1    one `vst`/bundle: [r13+32b] <- t(b)
      st2    two `vst`/bundle, same 64 B line (b>=8 off r14 = r13+512)
      st2x   two `vst`/bundle, bit 6 differs, r14 = r13 + far (default 1088)
      ldst   `vst` + `vld`, same bank: t(b) <- [r8+32b]; [r13+32b] <- t((b+8)%16)
      ldstx  as ldst, store to r14 = r13 + far
      chase  16 dependent `ld r4, [r4+0]` over absolute addresses in SRC; OUT[0] = r4
      v1hi   `k_lsram_port("v1")` with fill and loads in the upper half
      st1s64/st1s128  one `vst`/bundle striding S: t(4k+j) -> [r(13+k) + j*S], r(13+k) = r13 + 4kS
      st1same one `vst`/bundle, all to [r13+0] (t15 wins)
    Note: btaipuas rejects two `vst` in one bundle, so st2/st2x do not assemble."""
    if variant not in ("st1", "st2", "st2x", "ldst", "ldstx", "chase", "v1hi", "st1s64", "st1s128", "st1same"):
        raise ValueError(variant)
    strided = variant in ("st1s64", "st1s128")
    S = 64 if variant == "st1s64" else 128
    hi = [BUNDLE(I("mov r11, %d" % UPPER, E.mov("r11", UPPER)))]
    body = (_gsram_base("r6") if space == "gsram" else _lsram_base("r6"))
    if variant == "v1hi":
        body += hi + [BUNDLE(I("add r6, r6, r11", E.add("r6", "r6", "r11")))]
    body += _dma_fill_and_wait(space) + [BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0)))]

    def V(t, b, off): return I("ld t%d, [%s+%d]" % (t, b, off), E.vld("t%d" % t, b, off))
    def W(t, b, off): return I("st t%d, [%s+%d]" % (t, b, off), E.vst("t%d" % t, b, off))

    if variant in ("st1", "st2", "st2x", "ldst", "ldstx", "st1s64", "st1s128", "st1same"):
        # `far`: second store region offset from r13 (default ST_FAR)
        far = (far or ST_FAR) if variant in ("st2x", "ldstx") else 512
        body += [BUNDLE(V(t, "r8", 32 * t)) for t in range(16)]        # payload
        body += hi + [BUNDLE(I("add r13, r6, r11", E.add("r13", "r6", "r11"))),
                      BUNDLE(I("mov r11, %d" % far, E.mov("r11", far))),
                      BUNDLE(I("add r14, r13, r11", E.add("r14", "r13", "r11")))]
        if strided:   # four bases 4S apart keep every immediate under 512
            body += [BUNDLE(I("mov r11, %d" % (4 * S), E.mov("r11", 4 * S))),
                     BUNDLE(I("add r14, r13, r11", E.add("r14", "r13", "r11"))),
                     BUNDLE(I("add r15, r14, r11", E.add("r15", "r14", "r11"))),
                     BUNDLE(I("add r16, r15, r11", E.add("r16", "r15", "r11")))]
    elif variant == "chase":
        body += [BUNDLE(I("add r4, r6, 0", E.add("r4", "r6", 0)))]
    body += [BUNDLE(I("sub r10, r3, 1", E.sub("r10", "r3", 1))),
             BUNDLE(I("max r10, r10, zero", E.mx("r10", "r10", "zero"))),
             BUNDLE(I("loop r10", E.loop("r10")))]
    for b in range(16):
        if variant == "st1":
            body.append(BUNDLE(W(b, "r13", 32 * b)))
        elif variant == "st2":
            base, o = ("r13", 64 * b) if b < 8 else ("r14", 64 * (b - 8))
            body.append(BUNDLE(W((2 * b) % 16, base, o), W((2 * b + 1) % 16, base, o + 32)))
        elif variant == "st2x":
            body.append(BUNDLE(W((2 * b) % 16, "r13", 32 * b), W((2 * b + 1) % 16, "r14", 32 * b)))
        # Note: store first -- btaipuas packs vst+vld as {slot2: st, slot3: ld}, in given order.
        elif variant == "ldst":
            body.append(BUNDLE(W((b + 8) % 16, "r13", 32 * b), V(b, "r8", 32 * b)))
        elif variant == "ldstx":
            body.append(BUNDLE(W((b + 8) % 16, "r14", 32 * b), V(b, "r8", 32 * b)))
        elif variant == "chase":
            body.append(BUNDLE(I("ld r4, [r4+0]", E.ld("r4", "r4", 0))))
        elif variant == "v1hi":
            body.append(BUNDLE(V(b, "r8", 32 * b)))
        elif strided:
            body.append(BUNDLE(W(b, "r%d" % (13 + b // 4), S * (b % 4))))
        elif variant == "st1same":
            body.append(BUNDLE(W(b, "r13", 0)))
    body += [BUNDLE(I("loopend", E.loopend()))]

    if variant == "chase":
        body += [BUNDLE(I("st r4, [r2+0]", E.st("r4", "r2", 0)))]
        return prologue(4) + body + epilogue()
    if variant == "v1hi":
        body += [BUNDLE(W(t, "r2", 32 * t)) for t in range(16)]
        return prologue(4) + body + epilogue()
    # read the stored region back to OUT
    if strided:
        body += [BUNDLE(V(b, "r%d" % (13 + b // 4), S * (b % 4))) for b in range(16)]
        body += [BUNDLE(W(i, "r2", 32 * i)) for i in range(16)]
        return prologue(4) + body + epilogue()
    if variant == "st1same":
        body += [BUNDLE(V(0, "r13", 0)), BUNDLE(W(0, "r2", 0))]
        return prologue(4) + body + epilogue()
    rb = {"st1": [("r13", 16)], "ldst": [("r13", 16)], "st2": [("r13", 16), ("r14", 16)],
          "st2x": [("r13", 16), ("r14", 16)], "ldstx": [("r14", 16)]}[variant]
    t = 0
    for base, n in rb:
        body += [BUNDLE(V(t + i, base, 32 * i)) for i in range(n)]; t += n
    body += [BUNDLE(W(i, "r2", 32 * i)) for i in range(min(t, 16))]
    if t > 16:
        body += [BUNDLE(I("add r13, r2, 512", E.add("r13", "r2", 512)))]
        body += [BUNDLE(W(i, "r13", 32 * (i - 16))) for i in range(16, t)]
    return prologue(4) + body + epilogue()


def k_lsram_probe():
    """Copy this TEC's LSRAM signature words (`0xFA000000 + 4i`) to `OUT[i]`. args: r0=OUT.

    No on-TEC comparison: the host needs the raw words to tell "wrong TEC" from "content lost"."""
    body = _lsram_base("r5")
    for i in range(LSRAM_PROBE_WORDS):
        body += [BUNDLE(I("ld r6, [r5+%d]" % (4 * i), E.ld("r6", "r5", 4 * i))),
                 BUNDLE(I("st r6, [r0+%d]" % (4 * i), E.st("r6", "r0", 4 * i)))]
    return prologue(1, fp=False) + body + epilogue()


# ===================== ALU characterisation ==================================
# Register-resident kernels. `nchain` is the reuse distance of a dependency chain: cycles per
# bundle stay flat until nchain drops below the unit's latency.
# Operand block at r0:
#   +0   16 x fp16 1.0   +32  16 x fp16 1.0    (mma/mml operands: each mma adds 4.0)
#   +64   8 x fp32 0.0   +96   8 x fp32 1.0    (accumulator zero; fma operands: adds 1.0)
#   +128 16 x bf16 1.0   +160 16 x bf16 1.0    (bf16 operands)
ALU_BODY = 8                                                    # body bundles per iteration
ALU_GPR = ["r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18"]   # 8 scalar chains, no ABI reg
ALU_SCALAR = ("add", "sub", "mul", "mulh", "mx", "mn", "xor", "lsl", "lsr", "asr", "andl", "orl")
# slot-1-only ops (enc.B): at most one per bundle
ALU_SLOT1 = ("xor", "lsl", "lsr", "asr", "andl", "orl")
# per-chain seed: keeps the stored word exact and, where possible, distinct per chain
ALU_SEED = {"add": lambda c: 0, "sub": lambda c: 0, "mul": lambda c: 1, "mulh": lambda c: c + 1}


def _alu_preamble(scalar, bf16, op=None):
    """Operands into t28/t29 (matrix), t30 (fp32 ones), t31 (fp32 zero); accumulators zeroed.
    Scalar: r19=0, r20=1, r21=1023; with `op`, chains seeded per `ALU_SEED` (default c+1)."""
    ao, bo = (128, 160) if bf16 else (0, 32)
    body = [BUNDLE(I("ld t28, [r0+%d]" % ao, E.vld("t28", "r0", ao)),
                   I("ld t29, [r0+%d]" % bo, E.vld("t29", "r0", bo))),
            BUNDLE(I("ld t30, [r0+96]", E.vld("t30", "r0", 96)),
                   I("ld t31, [r0+64]", E.vld("t31", "r0", 64)))]
    if scalar:
        body += [BUNDLE(I("mov r19, 0", E.mov("r19", 0))), BUNDLE(I("mov r20, 1", E.mov("r20", 1)))]
        if op is not None:
            body += [BUNDLE(I("mov r21, 1023", E.mov("r21", 1023)))]     # `mn` preserving operand
            seed = ALU_SEED.get(op, lambda c: c + 1)
            body += [BUNDLE(I("mov %s, %d" % (g, seed(c)), E.mov(g, seed(c))))
                     for c, g in enumerate(ALU_GPR)]
    else:   # accumulators = fp32 0.0 via loads (no vector register move)
        body += [BUNDLE(I("ld t%d, [r0+64]" % i, E.vld("t%d" % i, "r0", 64)),
                        I("ld t%d, [r0+64]" % (i + 1), E.vld("t%d" % (i + 1), "r0", 64)))
                 for i in range(0, 16, 2)]
    return body


def _alu_loop_open():
    return [BUNDLE(I("sub r10, r3, 1", E.sub("r10", "r3", 1))),
            BUNDLE(I("max r10, r10, zero", E.mx("r10", "r10", "zero"))),
            BUNDLE(I("loop r10", E.loop("r10")))]


def _alu_stores(scalar):
    if scalar:
        return [BUNDLE(I("st %s, [r2+%d]" % (g, 4 * c), E.st(g, "r2", 4 * c))) for c, g in enumerate(ALU_GPR)]
    return [BUNDLE(I("st t%d, [r2+%d]" % (i, 32 * i), E.vst("t%d" % i, "r2", 32 * i))) for i in range(16)]


def _alu_op(op, c, q, bf16):
    """One `op` on chain `c`: a RAW chain on its own destination, value-preserving or counting."""
    if op == "add":                                   # rX += 1: counts executions
        g = ALU_GPR[c]; return I("add %s, %s, 1" % (g, g), E.add(g, g, 1))
    if op == "mul":                                   # rX *= 1
        g = ALU_GPR[c]; return I("mul %s, %s, r20" % (g, g), E.mul(g, g, "r20"))
    if op == "xor":                                   # rX ^= 0 (slot-1 only)
        g = ALU_GPR[c]; return I("xor %s, %s, r19" % (g, g), E.xor(g, g, "r19"))
    if op == "sub":                                   # rX -= 1: counts down, wraps
        g = ALU_GPR[c]; return I("sub %s, %s, 1" % (g, g), E.sub(g, g, 1))
    if op == "mulh":                                  # high 32 of rX*1 = 0 for a small rX
        g = ALU_GPR[c]; return I("mulh %s, %s, r20" % (g, g), E.mulh(g, g, "r20"))
    if op == "mx":                                    # max(rX, 0)
        g = ALU_GPR[c]; return I("max %s, %s, r19" % (g, g), E.mx(g, g, "r19"))
    if op == "mn":                                    # min(rX, 1023)
        g = ALU_GPR[c]; return I("min %s, %s, r21" % (g, g), E.mn(g, g, "r21"))
    if op in ("lsl", "lsr", "asr"):                   # shift by 0
        g = ALU_GPR[c]; fn = {"lsl": E.lsl, "lsr": E.lsr, "asr": E.asr}[op]
        return I("%s %s, %s, r19" % (op, g, g), fn(g, g, "r19"))
    if op == "andl":                                  # & 31: seeds are <= 8
        g = ALU_GPR[c]; return I("andl %s, %s, 31" % (g, g), E.andl(g, g, 31))
    if op == "orl":                                   # | 0
        g = ALU_GPR[c]; return I("orl %s, %s, 0" % (g, g), E.orl(g, g, 0))
    if op == "fma":                                   # tX += 1.0 * 1.0
        return I("fma t%d.fp32, t30.fp32, t30.fp32, p7.w" % c, E.fma("t%d" % c, "t30", "t30"))
    if op in ("mma", "mml"):                          # {t2c,t2c+1} += 4.0 per output (mml: = 4.0)
        a = 2 * c; fn = E.mma if op == "mma" else E.mml
        return I("%s t%d.fp32, t%d.fp32, t28.%s, t29.%s" % (op, a, a + 1, q, q), fn("t%d" % a, "t28", "t29", bf16=bf16))
    raise ValueError(op)


def k_alu(op, nchain, per_bundle=1, waw=False, bf16=False):
    """`ALU_BODY` bundles of `per_bundle` x `op`, round-robin over `nchain` dependency chains.

    `waw=True` emits `add r11, r19, 1` instead: same destination, independent source (pure WAW,
    the control for nchain=1).
    """
    scalar = op in ALU_SCALAR
    assert nchain >= 1 and (nchain <= 8 if (scalar or op in ("mma", "mml")) else nchain <= 16)
    assert not (op in ALU_SLOT1 and per_bundle > 1), "%s is slot-1 only: two per bundle is unencodable" % op
    q = "bf16" if bf16 else "fp16"
    body = _alu_preamble(scalar, bf16, op if scalar else None) + _alu_loop_open()
    k = 0
    for _ in range(ALU_BODY):
        ops = []
        for _ in range(per_bundle):
            c = k % nchain; k += 1
            ops.append(I("add r11, r19, 1", E.add("r11", "r19", 1)) if waw else _alu_op(op, c, q, bf16))
        body.append(BUNDLE(*ops))
    body += [BUNDLE(I("loopend", E.loopend()))]
    return prologue(4) + body + _alu_stores(scalar) + epilogue()


def k_alu_mix(op0, op1):
    """`op0` in slot 0 and `op1` in slot 1 every bundle; four chains per side on disjoint registers.

    mma/mml: pairs {t0,t1}..{t6,t7}; fma: t8..t11 when paired with a vector op0; scalar: r11..r14
    (slot 1 takes r15..r18 when both are scalar). Unused stored registers must read their seed.
    """
    assert op0 not in ("xor",), "xor cannot take slot 0"
    both_vec = op0 not in ALU_SCALAR and op1 not in ALU_SCALAR
    any_vec = op0 not in ALU_SCALAR or op1 not in ALU_SCALAR
    any_sca = op0 in ALU_SCALAR or op1 in ALU_SCALAR
    body = _alu_preamble(not any_vec, False)
    if any_sca:
        if any_vec:     # vector preamble skipped the scalar constants
            body += [BUNDLE(I("mov r19, 0", E.mov("r19", 0))), BUNDLE(I("mov r20, 1", E.mov("r20", 1)))]
        # distinct seeds (c+1) so an all-zero readback cannot pass
        body += [BUNDLE(I("mov %s, %d" % (g, c + 1), E.mov(g, c + 1))) for c, g in enumerate(ALU_GPR)]
    body += _alu_loop_open()
    for b in range(ALU_BODY):
        c = b % 4
        a = _alu_op(op0, c, "fp16", False)
        # slot 1 gets its own register block: fma at t8 (mma owns pairs t0..t7), scalar r15..r18
        d = c + 8 if (both_vec and op1 == "fma") else (c + 4 if (op0 in ALU_SCALAR and op1 in ALU_SCALAR) else c)
        body.append(BUNDLE(a, _alu_op(op1, d, "fp16", False)))
    body += [BUNDLE(I("loopend", E.loopend()))]
    stores = _alu_stores(False) if any_vec else []
    if any_sca:
        # scalars at OUT+512, after the vector registers; `st` offset is signed 9-bit, so via r21
        stores += [BUNDLE(I("add r21, r2, 512", E.add("r21", "r2", 512)))]
        stores += [BUNDLE(I("st %s, [r21+%d]" % (g, 4 * c), E.st(g, "r21", 4 * c)))
                   for c, g in enumerate(ALU_GPR)]
    return prologue(4) + body + stores + epilogue()


# Named ALU arms, shared by the image builder and the host harness.
ALU_ARMS = {
    # --- issue width (per_bundle 1 vs 2)
    "add_x1": lambda: k_alu("add", 8, 1),   "add_x2": lambda: k_alu("add", 8, 2),
    "mul_x1": lambda: k_alu("mul", 8, 1),   "mul_x2": lambda: k_alu("mul", 8, 2),
    "xor_x1": lambda: k_alu("xor", 8, 1),
    "fma_x1": lambda: k_alu("fma", 16, 1),  "fma_x2": lambda: k_alu("fma", 16, 2),
    "mma_x1": lambda: k_alu("mma", 8, 1),   "mma_x2": lambda: k_alu("mma", 8, 2),
    "mml_x1": lambda: k_alu("mml", 8, 1),   "mml_x2": lambda: k_alu("mml", 8, 2),
    "mmabf_x2": lambda: k_alu("mma", 8, 2, bf16=True),
    "mmlbf_x2": lambda: k_alu("mml", 8, 2, bf16=True),
    # --- latency (reuse-distance sweep)
    **{"add_n%d" % n: (lambda n=n: k_alu("add", n, 1)) for n in (1, 2, 3, 4, 6, 8)},
    **{"mul_n%d" % n: (lambda n=n: k_alu("mul", n, 1)) for n in (1, 2, 3, 4, 6, 8)},
    **{"xor_n%d" % n: (lambda n=n: k_alu("xor", n, 1)) for n in (1, 2, 4)},
    **{"fma_n%d" % n: (lambda n=n: k_alu("fma", n, 1)) for n in (1, 2, 3, 4, 5, 6, 7, 8)},
    **{"mma_n%d" % n: (lambda n=n: k_alu("mma", n, 1)) for n in (1, 2, 3, 4, 6)},
    # --- WAW control vs a dependent chain
    "add_waw": lambda: k_alu("add", 1, 1, waw=True),
    # --- remaining scalar ops
    **{"%s_n1" % o: (lambda o=o: k_alu(o, 1, 1)) for o in
       ("sub", "mulh", "mx", "mn", "lsl", "lsr", "asr", "andl", "orl")},
    **{"%s_n8" % o: (lambda o=o: k_alu(o, 8, 1)) for o in
       ("sub", "mulh", "mx", "mn", "lsl", "lsr", "asr", "andl", "orl")},
    **{"%s_x2" % o: (lambda o=o: k_alu(o, 8, 2)) for o in ("sub", "mulh", "mx", "mn")},
    # --- mixed packings (4 chains a side, disjoint registers)
    "mix_mma_fma": lambda: k_alu_mix("mma", "fma"),
    "mix_mma_add": lambda: k_alu_mix("mma", "add"),
    "mix_mma_xor": lambda: k_alu_mix("mma", "xor"),
    "mix_fma_add": lambda: k_alu_mix("fma", "add"),
    "mix_fma_xor": lambda: k_alu_mix("fma", "xor"),
    "mix_add_xor": lambda: k_alu_mix("add", "xor"),
    "mix_mul_xor": lambda: k_alu_mix("mul", "xor"),
}


def k_movpre_probe():
    """Semantics probe for `mov.pre rd, rs` (plain move vs pre-incrementing address form).

    Trials 0..2 (rs = 256, 64, 0): rs, rd at OUT+8i. Trial 3 (rs = LSRAM base): words loaded
    via rs/rd at OUT+24/+28, rd - rs at OUT+32.
    args: r0=SRC, r1=DESC, r2=OUT, r3 unused."""
    body = []
    for i, (val, src, dst) in enumerate([(256, "r5", "r6"), (64, "r7", "r9"), (0, "r11", "r12")]):
        body += [BUNDLE(I("mov %s, %d" % (src, val), E.mov(src, val))),
                 BUNDLE(I("mov %s, -1" % dst, E.mov(dst, -1))),
                 BUNDLE(I("mov.pre %s, %s" % (dst, src), E.mov_pre(dst, src))),
                 BUNDLE(I("st %s, [r2+%d]" % (src, 8 * i), E.st(src, "r2", 8 * i))),
                 BUNDLE(I("st %s, [r2+%d]" % (dst, 8 * i + 4), E.st(dst, "r2", 8 * i + 4)))]
    # trial 3: load through rs (r14 = LSRAM base) and rd (r15)
    body += _lsram_base("r6") + _dma_fill_and_wait()      # fill helper reads the base from r6
    body += [BUNDLE(I("add r14, r6, 0", E.add("r14", "r6", 0))),
             BUNDLE(I("mov r15, -1", E.mov("r15", -1))),
             BUNDLE(I("mov.pre r15, r14", E.mov_pre("r15", "r14"))),
             BUNDLE(I("ld r16, [r14+0]", E.ld("r16", "r14", 0))),
             BUNDLE(I("ld r17, [r15+0]", E.ld("r17", "r15", 0))),
             BUNDLE(I("sub r18, r15, r14", E.sub("r18", "r15", "r14"))),   # increment, if any
             BUNDLE(I("st r16, [r2+24]", E.st("r16", "r2", 24))),
             BUNDLE(I("st r17, [r2+28]", E.st("r17", "r2", 28))),
             BUNDLE(I("st r18, [r2+32]", E.st("r18", "r2", 32)))]
    return prologue(4) + body + epilogue()


def k_gemm_pipeN(ksteps=7, nabase=2, nbbase=2, total=PIPE34_STEPS):
    """`k_gemm_pipe34` with `ksteps` k-steps per base window, spread over several bases.

    `vld` offsets are 9-bit unsigned; bases sit 512 B apart, so offset `off` uses base
    `off//512` at `off%512` (bank bit 6 unchanged). Cost per k-step is
    (12*ksteps + ceil(nbases/2)) / ksteps; ksteps=7 with 2+2 bases minimises it within the
    10-bit `add` immediate. Same schedule, registers and gate as `k_gemm_pipe34`.
    """
    assert total % ksteps == 0
    ORDER = [(0, 0), (0, 2), (1, 0), (1, 2), (0, 1), (0, 3), (2, 0), (2, 2), (1, 1), (1, 3), (2, 1), (2, 3)]
    THIS = {0: ["B1", "B3"], 1: ["A2"]}
    NEXT = {3: ["A0"], 4: ["B0", "B2"], 5: ["A1"]}
    REG = {"A0": "t24", "A1": "t25", "A2": "t26", "B0": "t27", "B1": "t28", "B2": "t29", "B3": "t30"}
    ABASE = ["r8", "r16", "r18", "r20"][:nabase]
    BBASE = ["r13", "r17", "r19", "r21"][:nbbase]
    assert 96 * ksteps <= 1023 and 128 * ksteps <= 1023, "add immediate is 10-bit unsigned"

    def MMA(i, j):
        c = 2 * (4 * i + j)
        return I("mma t%d.fp32, t%d.fp32, %s.fp16, %s.fp16" % (c, c + 1, REG["A%d" % i], REG["B%d" % j]),
                 E.mma("t%d" % c, REG["A%d" % i], REG["B%d" % j]))

    def LD(x, k):
        stride, bases = (96, ABASE) if x[0] == "A" else (128, BBASE)
        off = stride * k + 32 * int(x[1])
        bi, ro = off // 512, off % 512
        assert bi < len(bases), "k-step %d of %s needs base %d, only %d given" % (k, x, bi, len(bases))
        return I("ld %s, [%s+%d]" % (REG[x], bases[bi], ro), E.vld(REG[x], bases[bi], ro))

    kbody = []
    for k in range(ksteps):
        for b in range(6):
            ops = [MMA(*ORDER[2 * b]), MMA(*ORDER[2 * b + 1])]
            ops += [LD(x, k) for x in THIS.get(b, [])]
            ops += [LD(x, k + 1) for x in NEXT.get(b, [])]
            kbody.append(BUNDLE(*ops))
    adv = ([I("add %s, %s, %d" % (r, r, 96 * ksteps), E.add(r, r, 96 * ksteps)) for r in ABASE] +
           [I("add %s, %s, %d" % (r, r, 128 * ksteps), E.add(r, r, 128 * ksteps)) for r in BBASE])
    for i in range(0, len(adv), 2):
        kbody.append(BUNDLE(*adv[i:i + 2]))

    def cbase(t): return ("r2", t * 32) if t < 16 else ("r14", (t - 16) * 32)
    def cload(t):
        b, off = cbase(t); return I("ld t%d, [%s+%d]" % (t, b, off), E.vld("t%d" % t, b, off))
    def cstore(t):
        b, off = cbase(t); return I("st t%d, [%s+%d]" % (t, b, off), E.vst("t%d" % t, b, off))
    body = _lsram_base("r6") + _dma_fill_and_wait() + [
        BUNDLE(I("add r14, r2, 512", E.add("r14", "r2", 512))),
        BUNDLE(I("mov r15, %d" % PIPE34_B_OFF, E.mov("r15", PIPE34_B_OFF))),
    ]
    body += [BUNDLE(cload(t), cload(t + 1)) for t in range(0, 24, 2)]
    arm = [BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0)),
                  I("add r13, r6, r15", E.add("r13", "r6", "r15")))]
    arm += [BUNDLE(I("add %s, %s, 512" % (ABASE[i], ABASE[i - 1]), E.add(ABASE[i], ABASE[i - 1], 512)))
            for i in range(1, nabase)]
    arm += [BUNDLE(I("add %s, %s, 512" % (BBASE[i], BBASE[i - 1]), E.add(BBASE[i], BBASE[i - 1], 512)))
            for i in range(1, nbbase)]
    outer = arm + [
        BUNDLE(LD("B0", 0), LD("B2", 0)),
        BUNDLE(LD("A0", 0)),
        BUNDLE(LD("A1", 0)),
        BUNDLE(I("mov r10, %d" % (total // ksteps - 1), E.mov("r10", total // ksteps - 1))),
        BUNDLE(I("loop r10", E.loop("r10"))),
    ] + kbody + [
        BUNDLE(I("loopend", E.loopend())),
        BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1))),
    ]
    outer.append(BUNDLE(I("cbnz r3, .Louter", E.cbnz("r3", -len(outer)))))
    body += outer
    body += [BUNDLE(cstore(t)) for t in range(24)]
    return prologue(4) + body + epilogue()


# ===================== GSRAM characterisation ================================
GSRAM_BASE_LO, GSRAM_BASE_HI = 0, -2048     # 0xF8000000, 256 KiB, per core, shared by its TECs


def _gsram_base(reg):
    """Two bundles that set `reg` = 0xF8000000 (GSRAM base)."""
    return [BUNDLE(I("mov %s, %d" % (reg, GSRAM_BASE_LO), E.mov(reg, GSRAM_BASE_LO))),
            BUNDLE(I("movh %s, %d" % (reg, GSRAM_BASE_HI), E.movh(reg, GSRAM_BASE_HI)))]


def k_sram_reach(via="vld", space="gsram"):
    """SRAM reachability: DMA DDR -> SRAM, then back to DDR by DMA (`via="dma"`) or by 8
    `vld` + `vst` from the TEC (`via="vld"`, 256 B). `space="lsram"` is the control.

    Note: a TEC fault spins (unused exception vectors), so it shows as a job timeout.
    args: r0=SRC (DDR), r1=DESC, r2=OUT (DDR), r3 unused."""
    assert via in ("dma", "vld") and space in ("gsram", "lsram")
    base = _gsram_base("r6") if space == "gsram" else _lsram_base("r6")
    ib = E.DMA_SHARED if space == "gsram" else E.DMA_LSRAM
    body = base + [
        BUNDLE(I("dma 0, zero, 0, %d, 0, 1, 0, r1, r6, r0" % ib,
                 E.dma(E.DMA_DIRECT, "zero", 0, ib, E.DMA_GLOBAL, E.DMA_EXT2INT,
                       E.DMA_USELESS, "r1", "r6", "r0"))),
        BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))),
        BUNDLE(I("wfe r7, 1", E.wfe("r7", 1))),
    ]
    if via == "dma":
        body += [
            BUNDLE(I("dma 0, zero, 0, %d, 0, 0, 0, r1, r6, r2" % ib,
                     E.dma(E.DMA_DIRECT, "zero", 0, ib, E.DMA_GLOBAL, E.DMA_INT2EXT,
                           E.DMA_USELESS, "r1", "r6", "r2"))),
            BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))),
            BUNDLE(I("wfe r7, 1", E.wfe("r7", 1))),
        ]
    else:
        body += [BUNDLE(I("ld t%d, [r6+%d]" % (i, 32 * i), E.vld("t%d" % i, "r6", 32 * i)),
                        I("ld t%d, [r6+%d]" % (i + 1, 32 * (i + 1)), E.vld("t%d" % (i + 1), "r6", 32 * (i + 1))))
                 for i in range(0, 8, 2)]
        body += [BUNDLE(I("st t%d, [r2+%d]" % (i, 32 * i), E.vst("t%d" % i, "r2", 32 * i))) for i in range(8)]
    return prologue(4) + body + epilogue()


def k_sram_share(space="gsram"):
    """Shared-SRAM load stream for multi-TEC contention; r4 selects overlapping (0 for all
    tasks) or disjoint (`t * region`) windows.

    args: r0=SRC, r1=DESC, r2=OUT, r3=LOOPS, r4=per-task byte offset from the window base.
    16 bundles x 1 `vld`, t(b) <- [r8+32b]; t0..t15 stored to OUT."""
    base = _gsram_base("r6") if space == "gsram" else _lsram_base("r6")
    body = base + _dma_fill_and_wait(space) + [
        BUNDLE(I("add r8, r6, r4", E.add("r8", "r6", "r4"))),      # per-task window
        BUNDLE(I("sub r10, r3, 1", E.sub("r10", "r3", 1))),
        BUNDLE(I("max r10, r10, zero", E.mx("r10", "r10", "zero"))),
        BUNDLE(I("loop r10", E.loop("r10")))]
    body += [BUNDLE(I("ld t%d, [r8+%d]" % (b, 32 * b), E.vld("t%d" % b, "r8", 32 * b))) for b in range(16)]
    body += [BUNDLE(I("loopend", E.loopend()))]
    body += [BUNDLE(I("st t%d, [r2+%d]" % (b, 32 * b), E.vst("t%d" % b, "r2", 32 * b))) for b in range(16)]
    return prologue(5) + body + epilogue()


def k_sram_to_sram(via="dma"):
    """GSRAM -> LSRAM move after DDR -> GSRAM, by DMA (`via="dma"`) or TEC vld/vst (`via="tec"`).

    DMA `int_base`: 4 = LSRAM, 8 = GSRAM; `ext_base` 0..3 = DDR, 8 = GSRAM (4-bit fields).
    args: r0=SRC, r1=DESC, r2=OUT, r3 unused. Reads back 16 x 32 B = 512 B of LSRAM."""
    assert via in ("dma", "tec")
    body = _gsram_base("r6") + [BUNDLE(I("mov r5, %d" % LSRAM_BASE_LO, E.mov("r5", LSRAM_BASE_LO))),
                                BUNDLE(I("movh r5, %d" % LSRAM_BASE_HI, E.movh("r5", LSRAM_BASE_HI)))]
    # 0. poison LSRAM's first 512 B from SRC+512 (a non-answer pattern): LSRAM persists
    #    across jobs, so stale correct bytes would otherwise mask a missing move
    body += [BUNDLE(I("add r9, r0, 512", E.add("r9", "r0", 512)))]
    body += [BUNDLE(I("ld t%d, [r9+%d]" % (b, 32 * b), E.vld("t%d" % b, "r9", 32 * b))) for b in range(16)]
    body += [BUNDLE(I("st t%d, [r5+%d]" % (b, 32 * b), E.vst("t%d" % b, "r5", 32 * b))) for b in range(16)]
    # 1. DDR -> GSRAM
    body += [BUNDLE(I("dma 0, zero, 0, 8, 0, 1, 0, r1, r6, r0",
                      E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_SHARED, E.DMA_GLOBAL, E.DMA_EXT2INT,
                            E.DMA_USELESS, "r1", "r6", "r0"))),
             BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))),
             BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    if via == "dma":
        # 2. GSRAM -> LSRAM: internal = LSRAM (4), external = GSRAM (8)
        body += [BUNDLE(I("dma 0, zero, 0, 4, 8, 1, 0, r1, r5, r6",
                          E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_LSRAM, E.DMA_SHARED, E.DMA_EXT2INT,
                                E.DMA_USELESS, "r1", "r5", "r6"))),
                 BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))),
                 BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    else:
        # 2'. same move through the TEC: 16 vld from GSRAM, 16 vst to LSRAM
        body += [BUNDLE(I("ld t%d, [r6+%d]" % (b, 32 * b), E.vld("t%d" % b, "r6", 32 * b))) for b in range(16)]
        body += [BUNDLE(I("st t%d, [r5+%d]" % (b, 32 * b), E.vst("t%d" % b, "r5", 32 * b))) for b in range(16)]
    # 3. LSRAM -> OUT
    body += [BUNDLE(I("ld t%d, [r5+%d]" % (b, 32 * b), E.vld("t%d" % b, "r5", 32 * b))) for b in range(16)]
    body += [BUNDLE(I("st t%d, [r2+%d]" % (b, 32 * b), E.vst("t%d" % b, "r2", 32 * b))) for b in range(16)]
    return prologue(4) + body + epilogue()



# SRAM <-> SRAM DMA descriptor shapes. Note: internal LSRAM + external GSRAM at its absolute
# address (as in `k_sram_to_sram`) hangs.
SRAM_DMA_SHAPES = {
    # name: (source, int_base, ext_base, dir, int addr form, ext addr form); form "abs" = TEC window, "off" = 0, "ddr" = r0
    "g2l_ig_ol":  ("gsram", 8, 4, 0, "abs", "abs"),   # GSRAM int -> LSRAM ext (INT2EXT)
    "g2l_ig_olo": ("gsram", 8, 4, 0, "abs", "off"),   # same, LSRAM as offset
    "g2l_il_ogo": ("gsram", 4, 8, 1, "abs", "off"),   # LSRAM int, GSRAM ext (EXT2INT), GSRAM as offset
    "g2l_il_og":  ("gsram", 4, 8, 1, "abs", "abs"),   # same, GSRAM absolute; hangs (negative control)
    "l2g_ig_ol":  ("lsram", 8, 4, 1, "abs", "abs"),   # LSRAM ext -> GSRAM int (EXT2INT)
    "l2g_ig_olo": ("lsram", 8, 4, 1, "abs", "off"),
    "l2g_il_og":  ("lsram", 4, 8, 0, "abs", "abs"),   # LSRAM int -> GSRAM ext (INT2EXT)
    "l2g_il_ogo": ("lsram", 4, 8, 0, "abs", "off"),
    # positive controls: DDR (global aperture at r0) -> destination
    "ctl_d2l":    ("gsram", 4, 0, 1, "abs", "ddr"),   # DDR -> LSRAM
    "ctl_d2g":    ("lsram", 8, 0, 1, "abs", "ddr"),   # DDR -> GSRAM
}


def k_sram_dma(shape):
    """Test one SRAM -> SRAM DMA shape on 512 B: poison the destination (TEC copy of SRC + 512), fill the source from DDR by a
    known-good DMA, run the candidate DMA, read the destination back to OUT. Pass = OUT holds the source data.
    Note: a refused shape never raises flag 0 and the job hangs until the KMD timeout; run each shape alone.
    args: r0 = SRC (512 B data + 512 B poison), r1 = DESC (512-B slot), r2 = OUT (512 B)."""
    src, ib, eb, dr, iform, eform = SRAM_DMA_SHAPES[shape]
    body = _gsram_base("r6") + [BUNDLE(I("mov r5, %d" % LSRAM_BASE_LO, E.mov("r5", LSRAM_BASE_LO))),
                                BUNDLE(I("movh r5, %d" % LSRAM_BASE_HI, E.movh("r5", LSRAM_BASE_HI)))]
    sreg, dreg = ("r6", "r5") if src == "gsram" else ("r5", "r6")
    body += [BUNDLE(I("add r9, r0, 512", E.add("r9", "r0", 512)))]
    body += [BUNDLE(I("ld t%d, [r9+%d]" % (b, 32 * b), E.vld("t%d" % b, "r9", 32 * b))) for b in range(16)]
    body += [BUNDLE(I("st t%d, [%s+%d]" % (b, dreg, 32 * b), E.vst("t%d" % b, dreg, 32 * b))) for b in range(16)]
    sib = E.DMA_SHARED if src == "gsram" else E.DMA_LSRAM
    body += [BUNDLE(I("dma 0, zero, 0, %d, 0, 1, 0, r1, %s, r0" % (sib, sreg),
                      E.dma(E.DMA_DIRECT, "zero", 0, sib, E.DMA_GLOBAL, E.DMA_EXT2INT, E.DMA_USELESS, "r1", sreg, "r0"))),
             BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))), BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    # candidate DMA: each end's address is absolute (TEC window), 0, or DDR
    iside = "gsram" if ib == E.DMA_SHARED else "lsram"; eside = "gsram" if eb == E.DMA_SHARED else "lsram"
    def areg(side, form, scratch):
        if form == "off": return [BUNDLE(I("mov %s, 0" % scratch, E.mov(scratch, 0)))], scratch
        if form == "ddr": return [], "r0"
        return [], ("r6" if side == "gsram" else "r5")
    pi, ri = areg(iside, iform, "r10"); pe, re_ = areg(eside, eform, "r11")
    body += pi + pe + [BUNDLE(I("dma 0, zero, 0, %d, %d, %d, 0, r1, %s, %s" % (ib, eb, dr, ri, re_),
                                E.dma(E.DMA_DIRECT, "zero", 0, ib, eb, dr, E.DMA_USELESS, "r1", ri, re_))),
                       BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))), BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    body += [BUNDLE(I("ld t%d, [%s+%d]" % (b, dreg, 32 * b), E.vld("t%d" % b, dreg, 32 * b))) for b in range(16)]
    body += [BUNDLE(I("st t%d, [r2+%d]" % (b, 32 * b), E.vst("t%d" % b, "r2", 32 * b))) for b in range(16)]
    return prologue(3) + body + epilogue()


def k_gm_remap():
    """Test the GM remap (grid TCB gm_ctrl / gm_rgnx_*): does a DMA to/from the global aperture at window W hit GSRAM?
      read:  A -> GSRAM+0 (TEC), poison LSRAM+0, DMA W+0 -> LSRAM+0 -> OUT+0. Routed: OUT[0:512] = A.
      write: B -> LSRAM+1024 (TEC), poison GSRAM+1024, DMA LSRAM+1024 -> W+1024, GSRAM+1024 -> OUT+512. Routed: OUT[512:1024] = B.
    args: r0 = W (ASID0 offset), r1 = OUT (1 KiB), r2 = DESC (512-B slot), r3 = PATTERNS (A | B | - | poison)."""
    body = _gsram_base("r6") + [BUNDLE(I("mov r5, %d" % LSRAM_BASE_LO, E.mov("r5", LSRAM_BASE_LO))),
                                BUNDLE(I("movh r5, %d" % LSRAM_BASE_HI, E.movh("r5", LSRAM_BASE_HI)))]
    cp = lambda src, soff, dst, doff: ([BUNDLE(I("ld t%d, [%s+%d]" % (b, src, soff + 32 * b), E.vld("t%d" % b, src, soff + 32 * b))) for b in range(16)] +
                                       [BUNDLE(I("st t%d, [%s+%d]" % (b, dst, doff + 32 * b), E.vst("t%d" % b, dst, doff + 32 * b))) for b in range(16)])
    body += [BUNDLE(I("add r9, r3, 512", E.add("r9", "r3", 512))), BUNDLE(I("mov r7, 1536", E.mov("r7", 1536))), BUNDLE(I("add r10, r3, r7", E.add("r10", "r3", "r7")))]
    body += cp("r3", 0, "r6", 0) + cp("r10", 0, "r5", 0)                          # A -> GSRAM+0; poison -> LSRAM+0
    body += [BUNDLE(I("dma 0, zero, 0, 4, 0, 1, 0, r2, r5, r0", E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT, E.DMA_USELESS, "r2", "r5", "r0"))),
             BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))), BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    body += cp("r5", 0, "r1", 0)
    body += [BUNDLE(I("mov r7, 1024", E.mov("r7", 1024))), BUNDLE(I("add r11, r5, r7", E.add("r11", "r5", "r7"))), BUNDLE(I("add r12, r6, r7", E.add("r12", "r6", "r7"))),
             BUNDLE(I("add r13, r0, r7", E.add("r13", "r0", "r7")))]
    body += cp("r9", 0, "r11", 0) + cp("r10", 0, "r12", 0)                         # B -> LSRAM+1024; poison -> GSRAM+1024
    body += [BUNDLE(I("dma 0, zero, 0, 4, 0, 0, 0, r2, r11, r13", E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_INT2EXT, E.DMA_USELESS, "r2", "r11", "r13"))),
             BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))), BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    body += [BUNDLE(I("add r14, r1, 512", E.add("r14", "r1", 512)))] + cp("r12", 0, "r14", 0)
    return prologue(4) + body + epilogue()


def k_gsram_dump(nbytes=8192):
    """The TEC's view of GSRAM [0, nbytes) -> OUT (vld / vst, 512 B a step). args: r0 = OUT."""
    body = _gsram_base("r6") + [BUNDLE(I("add r1, r0, 0", E.add("r1", "r0", 0)))]
    for step in range(nbytes // 512):
        body += [BUNDLE(I("ld t%d, [r6+%d]" % (b, 32 * b), E.vld("t%d" % b, "r6", 32 * b))) for b in range(16)]
        body += [BUNDLE(I("st t%d, [r1+%d]" % (b, 32 * b), E.vst("t%d" % b, "r1", 32 * b))) for b in range(16)]
        body += [BUNDLE(I("add r6, r6, 512", E.add("r6", "r6", 512)), I("add r1, r1, 512", E.add("r1", "r1", 512)))]
    return prologue(1) + body + epilogue()


def k_gm_roundtrip():
    """Round-trip 512 B through the GM remap window: PATTERN (r3+512) -> LSRAM+0, DMA -> global r0, poison LSRAM+1024,
    DMA global r0 -> LSRAM+1024 -> OUT. With the remap covering r0 the bytes never reach DDR.
    args: r0 = window address (ASID0 offset), r1 = OUT (512 B), r2 = DESC (512-B slot), r3 = PATTERNS."""
    body = [BUNDLE(I("mov r5, %d" % LSRAM_BASE_LO, E.mov("r5", LSRAM_BASE_LO))), BUNDLE(I("movh r5, %d" % LSRAM_BASE_HI, E.movh("r5", LSRAM_BASE_HI)))]
    cp = lambda src, dst: ([BUNDLE(I("ld t%d, [%s+%d]" % (b, src, 32 * b), E.vld("t%d" % b, src, 32 * b))) for b in range(16)] +
                           [BUNDLE(I("st t%d, [%s+%d]" % (b, dst, 32 * b), E.vst("t%d" % b, dst, 32 * b))) for b in range(16)])
    body += [BUNDLE(I("add r9, r3, 512", E.add("r9", "r3", 512))), BUNDLE(I("mov r7, 1536", E.mov("r7", 1536))), BUNDLE(I("add r10, r3, r7", E.add("r10", "r3", "r7"))),
             BUNDLE(I("mov r7, 1024", E.mov("r7", 1024))), BUNDLE(I("add r11, r5, r7", E.add("r11", "r5", "r7")))]
    body += cp("r9", "r5") + cp("r10", "r11")
    wait = [BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))), BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    body += [BUNDLE(I("dma 0, zero, 0, 4, 0, 0, 0, r2, r5, r0", E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_INT2EXT, E.DMA_USELESS, "r2", "r5", "r0")))] + wait
    body += [BUNDLE(I("dma 0, zero, 0, 4, 0, 1, 0, r2, r11, r0", E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT, E.DMA_USELESS, "r2", "r11", "r0")))] + wait
    body += cp("r11", "r1")
    return prologue(4) + body + epilogue()


def k_gm_alias():
    """Size GM by aliasing: X -> window r0, Y -> window r4, then read r0 -> OUT+0 and r4 -> OUT+512 (via LSRAM, DMA both ways).
    If r0 reads Y, the window wraps at r4 - r0.
    args: r0 = window address, r1 = OUT (1 KiB), r2 = DESC, r3 = PATTERNS (X | Y), r4 = second window address."""
    body = [BUNDLE(I("mov r5, %d" % LSRAM_BASE_LO, E.mov("r5", LSRAM_BASE_LO))), BUNDLE(I("movh r5, %d" % LSRAM_BASE_HI, E.movh("r5", LSRAM_BASE_HI)))]
    cp = lambda src, dst: ([BUNDLE(I("ld t%d, [%s+%d]" % (b, src, 32 * b), E.vld("t%d" % b, src, 32 * b))) for b in range(16)] +
                           [BUNDLE(I("st t%d, [%s+%d]" % (b, dst, 32 * b), E.vst("t%d" % b, dst, 32 * b))) for b in range(16)])
    wait = [BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))), BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    dma = lambda ls, g, d: [BUNDLE(I("dma 0, zero, 0, 4, 0, %d, 0, r2, %s, %s" % (d, ls, g), E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_LSRAM, E.DMA_GLOBAL, d, E.DMA_USELESS, "r2", ls, g)))] + wait
    body += [BUNDLE(I("add r9, r3, 512", E.add("r9", "r3", 512))), BUNDLE(I("mov r7, 1024", E.mov("r7", 1024))), BUNDLE(I("add r11, r5, r7", E.add("r11", "r5", "r7"))),
             BUNDLE(I("add r12, r1, 512", E.add("r12", "r1", 512)))]
    body += cp("r3", "r5") + dma("r5", "r0", E.DMA_INT2EXT)            # X -> window r0
    body += cp("r9", "r5") + dma("r5", "r4", E.DMA_INT2EXT)            # Y -> window r4
    body += dma("r11", "r0", E.DMA_EXT2INT) + cp("r11", "r1")          # window r0 -> OUT + 0
    body += dma("r11", "r4", E.DMA_EXT2INT) + cp("r11", "r12")         # window r4 -> OUT + 512
    return prologue(5) + body + epilogue()


def k_dma_bw(direction="in"):
    """DMA bandwidth between LSRAM+0 and global r0 + (i mod r4) * r5: r3 chunks of the descriptor's size, each waited.
    direction "in": global -> LSRAM, "out": LSRAM -> global.
    args: r0 = base, r1 = STAMP (4 B: cycles), r2 = DESC, r3 = chunks, r4 = chunks in the set, r5 = stride."""
    d = E.DMA_EXT2INT if direction == "in" else E.DMA_INT2EXT
    body = [BUNDLE(I("mov r6, %d" % LSRAM_BASE_LO, E.mov("r6", LSRAM_BASE_LO))), BUNDLE(I("movh r6, %d" % LSRAM_BASE_HI, E.movh("r6", LSRAM_BASE_HI))),
            BUNDLE(I("mfctrl0 r20, 209", E.mfctrl0("r20", 0xd1))), BUNDLE(I("add r8, r0, 0", E.add("r8", "r0", 0))), BUNDLE(I("add r9, r4, 0", E.add("r9", "r4", 0)))]
    loop = [BUNDLE(I("dma 0, zero, 0, 4, 0, %d, 0, r2, r6, r8" % d, E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_LSRAM, E.DMA_GLOBAL, d, E.DMA_USELESS, "r2", "r6", "r8"))),
            BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))), BUNDLE(I("wfe r7, 1", E.wfe("r7", 1))),
            BUNDLE(I("add r8, r8, r5", E.add("r8", "r8", "r5"))), BUNDLE(I("sub r9, r9, 1", E.sub("r9", "r9", 1)))]
    wrap = [BUNDLE(I("cbnz r9, .Lnowrap", E.cbnz("r9", 3))), BUNDLE(I("add r8, r0, 0", E.add("r8", "r0", 0))), BUNDLE(I("add r9, r4, 0", E.add("r9", "r4", 0)))]
    loop += wrap + [BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1)))]
    loop.append(BUNDLE(I("cbnz r3, .Lchunk", E.cbnz("r3", -len(loop)))))
    body += loop + [BUNDLE(I("mfctrl0 r21, 209", E.mfctrl0("r21", 0xd1))), BUNDLE(I("sub r21, r21, r20", E.sub("r21", "r21", "r20"))), BUNDLE(I("st r21, [r1+0]", E.st("r21", "r1", 0)))]
    return prologue(6) + body + epilogue()


def k_dma_pipe(depth=1):
    """A pipelined DMA stream DDR -> LSRAM the way a staged kernel runs it: `depth` requests in flight (1-4), each on its own
    flag into its own LSRAM tile; the loop waits the OLDEST and re-issues on its flag at once (k_gemm_gs's prefetch shape,
    not k_dma_flight's batch-and-wait-all). r3 requests of the descriptor's size (a multiple of `depth`), sources
    r0 + (i mod r4) x r5 so nothing is re-read within a set; the TEC's cycle counter around the whole stream.
    args: r0 = SRC base, r1 = STAMP (4 B: cycles), r2 = DESC (one slot; every flag uses it), r3 = requests, r4 = sources in
    the set, r5 = source stride (>= the tile), r6 = tile bytes (LSRAM pitch of the in-flight tiles; depth x tile <= 32 KiB)."""
    assert 1 <= depth <= 4, depth
    body = _lsram_base("r8") + [BUNDLE(I("add r9, r0, 0", E.add("r9", "r0", 0))), BUNDLE(I("add r10, r4, 0", E.add("r10", "r4", 0)))]
    body += [BUNDLE(I("mov r20, 0", E.mov("r20", 0)))]
    for f in range(1, depth): body += [BUNDLE(I("mov r%d, %d" % (20 + f, (f << 8) | f), E.mov("r%d" % (20 + f), (f << 8) | f)))]   # sync words
    body += [BUNDLE(I("add r16, r8, 0", E.add("r16", "r8", 0)))]
    for f in range(1, depth): body += [BUNDLE(I("add r%d, r%d, r6" % (16 + f, 15 + f), E.add("r%d" % (16 + f), "r%d" % (15 + f), "r6")))]   # tiles
    def req(f):
        return [BUNDLE(I("dma 0, r%d, 0, 4, 0, 1, 0, r2, r%d, r9" % (20 + f, 16 + f), E.dma(E.DMA_DIRECT, "r%d" % (20 + f), 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT, E.DMA_USELESS, "r2", "r%d" % (16 + f), "r9"))),
                BUNDLE(I("add r9, r9, r5", E.add("r9", "r9", "r5"))), BUNDLE(I("sub r10, r10, 1", E.sub("r10", "r10", 1))),
                BUNDLE(I("cbnz r10, .Lnowrap", E.cbnz("r10", 3))), BUNDLE(I("add r9, r0, 0", E.add("r9", "r0", 0))), BUNDLE(I("add r10, r4, 0", E.add("r10", "r4", 0)))]
    body += [BUNDLE(I("mfctrl0 r12, 209", E.mfctrl0("r12", 0xd1)))]
    for f in range(depth): body += req(f)
    body += [BUNDLE(I("sub r3, r3, %d" % depth, E.sub("r3", "r3", depth)))]
    loop = []
    for f in range(depth): loop += wait_flags(f) + req(f)
    loop += [BUNDLE(I("sub r3, r3, %d" % depth, E.sub("r3", "r3", depth)))]
    loop.append(BUNDLE(I("cbnz r3, .Lpipe", E.cbnz("r3", -len(loop)))))
    body += [BUNDLE(I("cbnz r3, .Lgo", E.cbnz("r3", 2))), BUNDLE(I("b .Ldrain", E.b(len(loop) + 1)))] + loop
    body += wait_flags(*range(depth))
    body += [BUNDLE(I("mfctrl0 r13, 209", E.mfctrl0("r13", 0xd1))), BUNDLE(I("sub r13, r13, r12", E.sub("r13", "r13", "r12"))), BUNDLE(I("st r13, [r1+0]", E.st("r13", "r1", 0)))]
    return prologue(7) + body + epilogue()


def k_gm_d2g():
    """DDR -> GM in one DMA (both ends global: DDR source internal, GM window external, INT2EXT), read back via LSRAM -> OUT.
    args: r0 = window address, r1 = OUT (512 B), r2 = DESC (512-B slot), r3 = SRC (DDR, 512 B)."""
    body = [BUNDLE(I("mov r5, %d" % LSRAM_BASE_LO, E.mov("r5", LSRAM_BASE_LO))), BUNDLE(I("movh r5, %d" % LSRAM_BASE_HI, E.movh("r5", LSRAM_BASE_HI)))]
    wait = [BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))), BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    body += [BUNDLE(I("dma 0, zero, 0, 0, 0, 0, 0, r2, r3, r0", E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_GLOBAL, E.DMA_GLOBAL, E.DMA_INT2EXT, E.DMA_USELESS, "r2", "r3", "r0")))] + wait
    body += [BUNDLE(I("dma 0, zero, 0, 4, 0, 1, 0, r2, r5, r0", E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT, E.DMA_USELESS, "r2", "r5", "r0")))] + wait
    body += [BUNDLE(I("ld t%d, [r5+%d]" % (b, 32 * b), E.vld("t%d" % b, "r5", 32 * b))) for b in range(16)]
    body += [BUNDLE(I("st t%d, [r1+%d]" % (b, 32 * b), E.vst("t%d" % b, "r1", 32 * b))) for b in range(16)]
    return prologue(4) + body + epilogue()


def k_dma_d2d_bw():
    """DDR -> global (DDR or the GM window) streaming: r3 chunks, each one int-side DDR -> ext-side global DMA of the descriptor's
    size, src r0 + i x r5, dst r4 + (i mod r6) x r5, waited one by one.  args: r0 = SRC, r1 = STAMP, r2 = DESC, r3 = chunks,
    r4 = DST, r5 = chunk bytes, r6 = chunks in the dst set."""
    body = [BUNDLE(I("mfctrl0 r20, 209", E.mfctrl0("r20", 0xd1))), BUNDLE(I("add r8, r0, 0", E.add("r8", "r0", 0))), BUNDLE(I("add r9, r4, 0", E.add("r9", "r4", 0))), BUNDLE(I("add r10, r6, 0", E.add("r10", "r6", 0)))]
    loop = [BUNDLE(I("dma 0, zero, 0, 0, 0, 0, 0, r2, r8, r9", E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_GLOBAL, E.DMA_GLOBAL, E.DMA_INT2EXT, E.DMA_USELESS, "r2", "r8", "r9"))),
            BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))), BUNDLE(I("wfe r7, 1", E.wfe("r7", 1))),
            BUNDLE(I("add r8, r8, r5", E.add("r8", "r8", "r5"))), BUNDLE(I("add r9, r9, r5", E.add("r9", "r9", "r5"))), BUNDLE(I("sub r10, r10, 1", E.sub("r10", "r10", 1)))]
    loop += [BUNDLE(I("cbnz r10, .Lnowrap", E.cbnz("r10", 3))), BUNDLE(I("add r9, r4, 0", E.add("r9", "r4", 0))), BUNDLE(I("add r10, r6, 0", E.add("r10", "r6", 0))),
             BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1)))]
    loop.append(BUNDLE(I("cbnz r3, .Lchunk", E.cbnz("r3", -len(loop)))))
    body += loop + [BUNDLE(I("mfctrl0 r21, 209", E.mfctrl0("r21", 0xd1))), BUNDLE(I("sub r21, r21, r20", E.sub("r21", "r21", "r20"))), BUNDLE(I("st r21, [r1+0]", E.st("r21", "r1", 0)))]
    return prologue(7) + body + epilogue()

# GSRAM rendezvous probe: do the four TECs of a core see each other's GSRAM writes?
GSRAM_SLOT, GSRAM_SLOTS = 1024, 4        # one 1 KiB slot per TEC of a core


def k_gsram_stamp():
    """Store PATTERN+i (i = 0..7) at 0xF8000000 + slot + 4i, then echo PATTERN to OUT[0] (GSRAM twin of `k_lsram_stamp`).
    The echo separates "never dispatched" from "GSRAM did not persist"; ascending values distinguish a smear from a stamp.
    args: r0 = PATTERN, r1 = OUT, r2 = slot byte offset."""
    body = _gsram_base("r5") + [BUNDLE(I("add r5, r5, r2", E.add("r5", "r5", "r2")))]
    for i in range(LSRAM_PROBE_WORDS):
        body += [BUNDLE(I("add r6, r0, %d" % i, E.add("r6", "r0", i))),
                 BUNDLE(I("st r6, [r5+%d]" % (4 * i), E.st("r6", "r5", 4 * i)))]
    body += [BUNDLE(I("st r0, [r1+0]", E.st("r0", "r1", 0)))]
    return prologue(3, fp=False) + body + epilogue()


def k_gsram_probe_all():
    """Copy all four GSRAM slots to OUT; the host does the comparison. args: r0 = OUT (32 words)."""
    body = _gsram_base("r5")
    n = 0
    for s in range(GSRAM_SLOTS):
        body += [BUNDLE(I("mov r7, %d" % (GSRAM_SLOT * s), E.mov("r7", GSRAM_SLOT * s))),
                 BUNDLE(I("add r8, r5, r7", E.add("r8", "r5", "r7")))]
        for i in range(LSRAM_PROBE_WORDS):
            body += [BUNDLE(I("ld r6, [r8+%d]" % (4 * i), E.ld("r6", "r8", 4 * i))),
                     BUNDLE(I("st r6, [r0+%d]" % (4 * n), E.st("r6", "r0", 4 * n)))]
            n += 1
    return prologue(1, fp=False) + body + epilogue()


# ===================== D-cache kernels ========
# Run on the simulator first: control-register writes or cacheops on unknown indices can hang silicon.
def _nowb_epilogue():
    """Task end without the dcache writeback (just `exit`); on silicon stores may then never reach DDR."""
    return [BUNDLE(I("exit", E.exit_()))]


CTRL_MF = {0: ("mfctrl0", E.mfctrl0), 1: ("mfctrl1", E.mfctrl1), 2: ("mfctrl2", E.mfctrl2)}
CTRL_MT = {0: ("mtctrl0", E.mtctrl0), 1: ("mtctrl1", E.mtctrl1), 2: ("mtctrl2", E.mtctrl2)}


def k_ctrl_dump(space=0):
    """Read all 256 registers of control space `space` into OUT. args: r0 = OUT (1 KiB).
    Note: simulator only; reading an unimplemented index on silicon may fault (hang).
    fp=False so ctrl1[2]/ctrl1[0x10] are read before the FP enable touches them."""
    name, mf = CTRL_MF[space]
    body = []
    for blk in range(4):                       # `st` reaches 252 B: one base per 64 words
        body.append(BUNDLE(I("add r5, r0, %d" % (256 * blk), E.add("r5", "r0", 256 * blk))))
        for j in range(64):
            i = 64 * blk + j
            body += [BUNDLE(I("%s r4, %d" % (name, i), mf("r4", i))),
                     BUNDLE(I("st r4, [r5+%d]" % (4 * j), E.st("r4", "r5", 4 * j)))]
    return prologue(1, fp=False) + body + epilogue()


def k_cacheop_probe(mode):
    """Touch three lines of BUF (two loads, one store), `cacheop mode` each, reload two. One kernel per mode 0..31.
    args: r0 = BUF (>= 256 B), r1 = OUT (16 B): [ld0, ld64, ld0 again, ld128 (the stored word)]."""
    body = [
        BUNDLE(I("ld r4, [r0+0]", E.ld("r4", "r0", 0))),
        BUNDLE(I("ld r5, [r0+64]", E.ld("r5", "r0", 64))),
        BUNDLE(I("mov r6, 4660", E.mov("r6", 4660))),                       # 0x1234
        BUNDLE(I("st r6, [r0+128]", E.st("r6", "r0", 128))),
        BUNDLE(I("add r7, r0, 64", E.add("r7", "r0", 64)),
               I("add r8, r0, 128", E.add("r8", "r0", 128))),
        BUNDLE(I("cacheop [r0+0], %d" % mode, E.cacheop("r0", mode))),
        BUNDLE(I("cacheop [r7+0], %d" % mode, E.cacheop("r7", mode))),
        BUNDLE(I("cacheop [r8+0], %d" % mode, E.cacheop("r8", mode))),
        BUNDLE(I("ld r9, [r0+0]", E.ld("r9", "r0", 0))),
        BUNDLE(I("ld r10, [r0+128]", E.ld("r10", "r0", 128))),
        BUNDLE(I("st r4, [r1+0]", E.st("r4", "r1", 0))),
        BUNDLE(I("st r5, [r1+4]", E.st("r5", "r1", 4))),
        BUNDLE(I("st r9, [r1+8]", E.st("r9", "r1", 8))),
        BUNDLE(I("st r10, [r1+12]", E.st("r10", "r1", 12))),
    ]
    return prologue(2, fp=False) + body + epilogue()


def k_store_nowb(nlines=4):
    """Store 0x5A00+i into line i of OUT (`nlines` 64 B lines), then exit without the dcache writeback.
    Shows whether stores reach DDR without the writeback. args: r0 = OUT."""
    body = []
    for i in range(nlines):
        body += [BUNDLE(I("mov r4, %d" % (0x5A00 + i), E.mov("r4", 0x5A00 + i))),
                 BUNDLE(I("st r4, [r0+%d]" % (64 * i), E.st("r4", "r0", 64 * i)))]
    return prologue(1, fp=False) + body + _nowb_epilogue()


# ----------------------------------------------------------- silicon kernels ----
# The simulator has no data cache (cacheop is a no-op), so it validates only control flow and stores here.
DCACHE_LINE = 64
CTRL_SIM_VALID = {                     # control indices the simulator implements, per space
    0: [0x00, 0x01, 0x10, 0x14, 0x30, 0x31, 0x32, 0x33, 0x40, 0x41, 0x44, 0x45, 0x80, 0x81, 0x82,
        0x90, 0x91, 0x92, 0x93, 0xd0, 0xd1, 0xe0, 0xe1, 0xe2, 0xe3, 0xe8, 0xe9, 0xea, 0xeb, 0xf0],
    1: [0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x80, 0x81, 0x82, 0x83],
    2: [0x00, 0x02, 0x03, 0x04, 0x30, 0x31, 0x40, 0x42, 0x43, 0x44, 0x70, 0x71],
}


def k_ctrl_read(space=0, indices=None):
    """Read the given control-register indices (default: `CTRL_SIM_VALID[space]`) into OUT, one word each.
    Read-only, safe on silicon. args: r0 = OUT."""
    name, mf = CTRL_MF[space]
    idx = list(CTRL_SIM_VALID[space] if indices is None else indices)
    body = [BUNDLE(I("add r5, r0, 0", E.add("r5", "r0", 0)))]
    for k, i in enumerate(idx):
        if k and k % 64 == 0:
            body.append(BUNDLE(I("add r5, r5, 256", E.add("r5", "r5", 256))))
        body += [BUNDLE(I("%s r4, %d" % (name, i), mf("r4", i))),
                 BUNDLE(I("st r4, [r5+%d]" % (4 * (k % 64)), E.st("r4", "r5", 4 * (k % 64))))]
    return prologue(1, fp=False) + body + epilogue()


MLP_BASES = ["r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18"]
MLP_DESTS = ["r4", "r5", "r6", "r7", "r9", "r19", "r20", "r21"]


def k_mlp(n):
    """Memory-level parallelism: `n` independent scalar loads per iteration (one per stream), then base advances, then
    accumulates, so no load stalls on another. Stream i starts at BASE + i * 4 MiB (r8, kernel-set).
    STRIDE=64 walks sequentially (exposes a next-line prefetcher); STRIDE=4160 (65 lines) defeats one.
    args: r0 = BASE, r1 = ITERS, r2 = OUT, r3 = STRIDE. OUT[0] = sum of all loaded words (exact for small ints)."""
    assert 1 <= n <= 8
    B, D = MLP_BASES[:n], MLP_DESTS[:n]
    body = [BUNDLE(I("mov r8, 0", E.mov("r8", 0))), BUNDLE(I("movh r8, 64", E.movh("r8", 64))),   # 4 MiB
            BUNDLE(I("mov r10, 0", E.mov("r10", 0))), BUNDLE(I("mov r22, 0", E.mov("r22", 0))),   # two accumulators
            BUNDLE(I("add %s, r0, 0" % B[0], E.add(B[0], "r0", 0)))]
    for i in range(1, n):
        body.append(BUNDLE(I("add %s, %s, r8" % (B[i], B[i - 1]), E.add(B[i], B[i - 1], "r8"))))
    loop = []
    for i in range(0, n, 2):                                    # loads, two per bundle
        ops = [I("ld %s, [%s+0]" % (D[i], B[i]), E.ld(D[i], B[i], 0))]
        if i + 1 < n: ops.append(I("ld %s, [%s+0]" % (D[i + 1], B[i + 1]), E.ld(D[i + 1], B[i + 1], 0)))
        loop.append(BUNDLE(*ops))
    for i in range(0, n, 2):                                    # advances, two per bundle
        ops = [I("add %s, %s, r3" % (B[i], B[i]), E.add(B[i], B[i], "r3"))]
        if i + 1 < n: ops.append(I("add %s, %s, r3" % (B[i + 1], B[i + 1]), E.add(B[i + 1], B[i + 1], "r3")))
        loop.append(BUNDLE(*ops))
    for i in range(0, n, 2):                                    # accumulates (the stall point)
        ops = [I("add r10, r10, %s" % D[i], E.add("r10", "r10", D[i]))]
        if i + 1 < n: ops.append(I("add r22, r22, %s" % D[i + 1], E.add("r22", "r22", D[i + 1])))
        loop.append(BUNDLE(*ops))
    loop.append(BUNDLE(I("sub r1, r1, 1", E.sub("r1", "r1", 1))))
    loop.append(BUNDLE(I("cbnz r1, .Lmlp", E.cbnz("r1", -len(loop)))))
    body += loop
    body += [BUNDLE(I("add r10, r10, r22", E.add("r10", "r10", "r22"))),
             BUNDLE(I("st r10, [r2+0]", E.st("r10", "r2", 0)))]
    return prologue(4, fp=False) + body + epilogue()


def k_vstream(inflight, window=0, pairs=True):
    """Sequential 32 B `vld` stream from DDR through the cache (vector twin of `k_mlp`): `inflight` loads issued before the
    first is consumed by fp32 `fma` into 8 rotating accumulators t16..t23 (x1.0 from t30); 32*inflight bytes per iteration.
    `window` > 0 confines the walk to BASE..BASE+window (4096: L1, 32768: L2); r1 then counts laps, each a hardware loop of
    window/(32*inflight) iterations. Note: uses nested loops, not register `and` masking, which hung silicon.
    `pairs=False` issues one vld per bundle.
    args: r0 = BASE (fp32 small ints), r1 = ITERS, r2 = OUT (256 B: t30 seed, then the 8 accumulators)."""
    assert 1 <= inflight <= 16
    if window: assert window % (32 * inflight) == 0
    step = 2 if pairs else 1
    body = [BUNDLE(I("ld t30, [r2+0]", E.vld("t30", "r2", 0)))]                # 8 x 1.0, host-seeded
    body += [BUNDLE(I("ld t%d, [r2+%d]" % (16 + i, 32 * (1 + i)), E.vld("t%d" % (16 + i), "r2", 32 * (1 + i)))) for i in range(8)]
    body += [BUNDLE(I("add r8, r0, 0", E.add("r8", "r0", 0)))]
    inner = []
    for i in range(0, inflight, step):
        ops = [I("ld t%d, [r8+%d]" % (i, 32 * i), E.vld("t%d" % i, "r8", 32 * i))]
        if pairs and i + 1 < inflight: ops.append(I("ld t%d, [r8+%d]" % (i + 1, 32 * (i + 1)), E.vld("t%d" % (i + 1), "r8", 32 * (i + 1))))
        inner.append(BUNDLE(*ops))
    inner.append(BUNDLE(I("add r8, r8, %d" % (32 * inflight), E.add("r8", "r8", 32 * inflight))))
    for i in range(0, inflight, 2):
        ops = [I("fma t%d.fp32, t%d.fp32, t30.fp32, p7.w" % (16 + i % 8, i), E.fma("t%d" % (16 + i % 8), "t%d" % i, "t30"))]
        if i + 1 < inflight:
            ops.append(I("fma t%d.fp32, t%d.fp32, t30.fp32, p7.w" % (16 + (i + 1) % 8, i + 1), E.fma("t%d" % (16 + (i + 1) % 8), "t%d" % (i + 1), "t30")))
        inner.append(BUNDLE(*ops))
    if window:
        n = window // (32 * inflight)
        loop = [BUNDLE(I("add r8, r0, 0", E.add("r8", "r0", 0))),
                BUNDLE(I("mov r10, %d" % (n - 1), E.mov("r10", n - 1))),
                BUNDLE(I("loop r10", E.loop("r10")))] + inner + [BUNDLE(I("loopend", E.loopend()))]
    else:
        loop = inner
    loop.append(BUNDLE(I("sub r1, r1, 1", E.sub("r1", "r1", 1))))
    loop.append(BUNDLE(I("cbnz r1, .Lvs", E.cbnz("r1", -len(loop)))))
    body += loop
    body += [BUNDLE(I("st t%d, [r2+%d]" % (16 + i, 32 * (1 + i)), E.vst("t%d" % (16 + i), "r2", 32 * (1 + i)))) for i in range(8)]
    return prologue(3) + body + epilogue()


def k_cop_loop(mode=None, do_chase=True):
    """R times: `cacheop mode` on each of NLINES consecutive lines of BUF, then one pointer-chase lap (NLINES+1 steps).
    mode=None keeps the loop without cacheop (control); do_chase=False runs the cacheop loop alone.
    args: r0 = BUF (a cycle over its NLINES lines), r1 = NLINES, r2 = OUT, r3 = R. OUT[0] = offset after the lap."""
    body = []
    rep = []
    # hardware loop over NLINES: {cacheop} {add r8, 64}
    rep += [BUNDLE(I("add r8, r0, 0", E.add("r8", "r0", 0))),
            BUNDLE(I("sub r10, r1, 1", E.sub("r10", "r1", 1))),
            BUNDLE(I("loop r10", E.loop("r10")))]
    rep.append(BUNDLE(I("cacheop [r8+0], %d" % mode, E.cacheop("r8", mode))) if mode is not None else BUNDLE())
    rep += [BUNDLE(I("add r8, r8, 64", E.add("r8", "r8", 64))),
            BUNDLE(I("loopend", E.loopend()))]
    if do_chase:
        rep += [BUNDLE(I("mov r4, 0", E.mov("r4", 0))),
                BUNDLE(I("add r11, r1, 1", E.add("r11", "r1", 1)))]
        chase = [BUNDLE(I("add r5, r0, r4", E.add("r5", "r0", "r4"))),
                 BUNDLE(I("ld r4, [r5+0]", E.ld("r4", "r5", 0))),
                 BUNDLE(I("sub r11, r11, 1", E.sub("r11", "r11", 1)))]
        chase.append(BUNDLE(I("cbnz r11, .Lchase", E.cbnz("r11", -len(chase)))))
        rep += chase
    rep.append(BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1))))
    rep.append(BUNDLE(I("cbnz r3, .Lrep", E.cbnz("r3", -len(rep)))))
    body += [BUNDLE(I("mov r4, 0", E.mov("r4", 0)))] + rep
    body += [BUNDLE(I("st r4, [r2+0]", E.st("r4", "r2", 0)))]
    return prologue(4, fp=False) + body + epilogue()


def k_cop_wb(mode=None, span=1024, every_line=False, wb=False):
    """Store 0x5A00+i at BUF+16i over `span` bytes, `cacheop mode` at BUF+0 (or every 64 B line), then exit without the
    dcache writeback unless `wb`. The words that reach DDR show the op's granule. args: r0 = BUF."""
    nw = span // 16
    body = [BUNDLE(I("add r8, r0, 0", E.add("r8", "r0", 0)))]
    for i in range(nw):
        if i and (16 * i) % 256 == 0:
            body.append(BUNDLE(I("add r8, r8, 256", E.add("r8", "r8", 256))))
        body += [BUNDLE(I("mov r4, %d" % (0x5A00 + i), E.mov("r4", 0x5A00 + i))),
                 BUNDLE(I("st r4, [r8+%d]" % ((16 * i) % 256), E.st("r4", "r8", (16 * i) % 256)))]
    if mode is not None:
        lines = range(0, span, 64) if every_line else [0]
        body.append(BUNDLE(I("add r8, r0, 0", E.add("r8", "r0", 0))))
        for k, off in enumerate(lines):
            if k: body.append(BUNDLE(I("add r8, r8, 64", E.add("r8", "r8", 64))))
            body.append(BUNDLE(I("cacheop [r8+0], %d" % mode, E.cacheop("r8", mode))))
    return prologue(1, fp=False) + body + (epilogue() if wb else _nowb_epilogue())


def k_dirty(wb=True):
    """Dirty NLINES consecutive lines of BUF (one word each), then end with or without the dcache writeback;
    the wb vs no-wb difference prices the drain per dirty line. args: r0 = BUF, r1 = NLINES, r2 = OUT (OUT[0] = NLINES)."""
    body = [BUNDLE(I("add r8, r0, 0", E.add("r8", "r0", 0))),
            BUNDLE(I("add r9, r1, 0", E.add("r9", "r1", 0))),
            BUNDLE(I("mov r4, 23130", E.mov("r4", 23130)))]                # 0x5A5A
    loop = [BUNDLE(I("st r4, [r8+0]", E.st("r4", "r8", 0))),
            BUNDLE(I("add r8, r8, 64", E.add("r8", "r8", 64)), I("sub r9, r9, 1", E.sub("r9", "r9", 1)))]
    loop.append(BUNDLE(I("cbnz r9, .Ldirty", E.cbnz("r9", -len(loop)))))
    body += loop + [BUNDLE(I("st r1, [r2+0]", E.st("r1", "r2", 0)))]
    return prologue(3, fp=False) + body + (epilogue() if wb else _nowb_epilogue())


def k_gemm_cachefed(space="cache", ksteps=PIPE34_STEPS):
    """`k_gemm_pipe34` (3x4 block, 12 mma + 7 vld per k-step, same schedule/registers) with operands read from DDR
    through the cache instead of DMA-filled LSRAM.
      space="cache":  every rep re-reads the same panel pair (21 k-steps = 4.7 KiB fits L1; 63 = 14.1 KiB fits L2)
      space="stream": every rep advances SRC by 16 KiB; the host seeds REPS identical panels so C is unchanged
    args: r0 = SRC (A at 0, B at PIPE34_B_OFF), r1 = unused, r2 = OUT (C, 768 B, pre-zeroed), r3 = REPS."""
    assert space in ("cache", "stream") and ksteps % 3 == 0
    iters = ksteps // 3
    ORDER = [(0, 0), (0, 2), (1, 0), (1, 2), (0, 1), (0, 3), (2, 0), (2, 2), (1, 1), (1, 3), (2, 1), (2, 3)]
    THIS = {0: ["B1", "B3"], 1: ["A2"]}
    NEXT = {3: ["A0"], 4: ["B0", "B2"], 5: ["A1"]}
    REG = {"A0": "t24", "A1": "t25", "A2": "t26", "B0": "t27", "B1": "t28", "B2": "t29", "B3": "t30"}
    def MMA(i, j):
        c = 2 * (4 * i + j)
        return I("mma t%d.fp32, t%d.fp32, %s.fp16, %s.fp16" % (c, c + 1, REG["A%d" % i], REG["B%d" % j]),
                 E.mma("t%d" % c, REG["A%d" % i], REG["B%d" % j]))
    def LD(x, k):
        if x[0] == "A": base, off = "r8", 96 * k + 32 * int(x[1])
        else:           base, off = "r13", 128 * k + 32 * int(x[1])
        return I("ld %s, [%s+%d]" % (REG[x], base, off), E.vld(REG[x], base, off))
    kbody = []
    for k in range(3):
        for b in range(6):
            ops = [MMA(*ORDER[2 * b]), MMA(*ORDER[2 * b + 1])]
            ops += [LD(x, k) for x in THIS.get(b, [])]
            ops += [LD(x, k + 1) for x in NEXT.get(b, [])]
            kbody.append(BUNDLE(*ops))
    kbody.append(BUNDLE(I("add r8, r8, 288", E.add("r8", "r8", 288)),
                        I("add r13, r13, 384", E.add("r13", "r13", 384))))

    def cbase(t): return ("r2", t * 32) if t < 16 else ("r14", (t - 16) * 32)
    def cload(t):
        b, off = cbase(t); return I("ld t%d, [%s+%d]" % (t, b, off), E.vld("t%d" % t, b, off))
    def cstore(t):
        b, off = cbase(t); return I("st t%d, [%s+%d]" % (t, b, off), E.vst("t%d" % t, b, off))
    body = [BUNDLE(I("add r6, r0, 0", E.add("r6", "r0", 0))),                  # panels in DDR, not LSRAM
            BUNDLE(I("add r14, r2, 512", E.add("r14", "r2", 512))),
            BUNDLE(I("mov r15, %d" % PIPE34_B_OFF, E.mov("r15", PIPE34_B_OFF))),
            BUNDLE(I("mov r16, %d" % 16384, E.mov("r16", 16384)))]
    body += [BUNDLE(cload(t), cload(t + 1)) for t in range(0, 24, 2)]
    outer = [
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0)), I("add r13, r6, r15", E.add("r13", "r6", "r15"))),
        BUNDLE(LD("B0", 0), LD("B2", 0)),
        BUNDLE(LD("A0", 0)),
        BUNDLE(LD("A1", 0)),
        BUNDLE(I("mov r10, %d" % (iters - 1), E.mov("r10", iters - 1))),
        BUNDLE(I("loop r10", E.loop("r10"))),
    ] + kbody + [
        BUNDLE(I("loopend", E.loopend())),
        BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1)),
               *([I("add r6, r6, r16", E.add("r6", "r6", "r16"))] if space == "stream" else [])),
    ]
    outer.append(BUNDLE(I("cbnz r3, .Louter", E.cbnz("r3", -len(outer)))))
    body += outer
    body += [BUNDLE(cstore(t)) for t in range(24)]
    return prologue(4) + body + epilogue()


# ------------------------------------------------- distributed GEMM ----------
# Like `k_gemm_cachefed` but with separate A and B bases, so several TECs can split A while sharing B.
GEMM_DIST_KSTEPS = 63                    # k-steps in one panel; 3x4 block -> 63*4 = 252 deep
GEMM_DIST_A_STEP = 96                    # 3 A tiles x 32 B per k-step
GEMM_DIST_B_STEP = 128                   # 4 B tiles x 32 B per k-step


def k_gemm_dist(ksteps=GEMM_DIST_KSTEPS, trace=False):
    """C[12x16] += SWEEPS x (A[12 x 4*ksteps*NPANEL] @ B[4*ksteps*NPANEL x 16]), fp16 in, fp32 out, with A and B on
    independent bases (each task may get its own A and share B). Body is `k_gemm_pipe34`'s schedule, unmodified.
    NPANEL sets the working set (each sweep rewinds both streams), SWEEPS the amount of work.

    args: r0 = A (NPANEL+1 panels of 96*ksteps B), r1 = B (NPANEL+1 panels of 128*ksteps B),
          r2 = OUT (C, 12 pairs x 64 B = 768 B, pre-zeroed), r3 = NPANEL, r4 = SWEEPS

    Note: allocate one panel of slack past each stream: the pipelined loop's last k-step loads 96 B past A's end and
    128 B past B's, and an SRAM over-read returns garbage without faulting.
    Exact: small-integer operands, fp32 accumulation; keep SWEEPS * NPANEL * 4*ksteps * 9 < 2^24.
    Registers: C pairs t0..t23, A0..A2 = t24..t26, B0..B3 = t27..t30; r5 panel counter, r6/r7 stream cursors,
    r8/r13 in-panel cursors, r14 upper C base, r15/r16 panel strides.
    """
    assert ksteps % 3 == 0, "the pipelined body advances three k-steps at a time"
    iters = ksteps // 3
    a_panel, b_panel = GEMM_DIST_A_STEP * ksteps, GEMM_DIST_B_STEP * ksteps
    # `mov` takes a signed 16-bit immediate.
    assert max(a_panel, b_panel) < 32768, "panel stride needs a 32-bit constant"

    ORDER = [(0, 0), (0, 2), (1, 0), (1, 2), (0, 1), (0, 3), (2, 0), (2, 2), (1, 1), (1, 3), (2, 1), (2, 3)]
    THIS = {0: ["B1", "B3"], 1: ["A2"]}
    NEXT = {3: ["A0"], 4: ["B0", "B2"], 5: ["A1"]}
    REG = {"A0": "t24", "A1": "t25", "A2": "t26", "B0": "t27", "B1": "t28", "B2": "t29", "B3": "t30"}

    def MMA(i, j):
        c = 2 * (4 * i + j)
        return I("mma t%d.fp32, t%d.fp32, %s.fp16, %s.fp16" % (c, c + 1, REG["A%d" % i], REG["B%d" % j]),
                 E.mma("t%d" % c, REG["A%d" % i], REG["B%d" % j]))

    def LD(x, k):
        if x[0] == "A": base, off = "r8", GEMM_DIST_A_STEP * k + 32 * int(x[1])
        else:           base, off = "r13", GEMM_DIST_B_STEP * k + 32 * int(x[1])
        return I("ld %s, [%s+%d]" % (REG[x], base, off), E.vld(REG[x], base, off))

    kbody = []
    for k in range(3):
        for b in range(6):
            ops = [MMA(*ORDER[2 * b]), MMA(*ORDER[2 * b + 1])]
            ops += [LD(x, k) for x in THIS.get(b, [])]
            ops += [LD(x, k + 1) for x in NEXT.get(b, [])]
            kbody.append(BUNDLE(*ops))
    kbody.append(BUNDLE(I("add r8, r8, %d" % (3 * GEMM_DIST_A_STEP), E.add("r8", "r8", 3 * GEMM_DIST_A_STEP)),
                        I("add r13, r13, %d" % (3 * GEMM_DIST_B_STEP), E.add("r13", "r13", 3 * GEMM_DIST_B_STEP))))

    def cbase(t): return ("r2", t * 32) if t < 16 else ("r14", (t - 16) * 32)
    def cload(t):
        b, off = cbase(t); return I("ld t%d, [%s+%d]" % (t, b, off), E.vld("t%d" % t, b, off))
    def cstore(t):
        b, off = cbase(t); return I("st t%d, [%s+%d]" % (t, b, off), E.vst("t%d" % t, b, off))

    body = []
    if trace:
        # Save the tag (r5) before the sweep loop reuses r5 as the panel counter.
        body += [BUNDLE(I("add r17, r5, 0", E.add("r17", "r5", 0)))]
        # Entry stamp in the same slot as the exit stamp, tag | GEMM_DIST_ENTRY_BIT:
        #   bit clear -> finished; bit set -> started, did not finish; older tag -> never started.
        body += [BUNDLE(I("add r18, r17, %d" % GEMM_DIST_ENTRY_BIT,
                          E.add("r18", "r17", GEMM_DIST_ENTRY_BIT)))]
        body += _gemm_dist_stamp("r18")
    body += [
        BUNDLE(I("add r14, r2, 512", E.add("r14", "r2", 512))),   # `vld`'s imm reaches 512 B
        # `mov` only issues in slot 2, so two cannot share a bundle.
        BUNDLE(I("mov r15, %d" % a_panel, E.mov("r15", a_panel))),
        BUNDLE(I("mov r16, %d" % b_panel, E.mov("r16", b_panel))),
    ]
    body += [BUNDLE(cload(t), cload(t + 1)) for t in range(0, 24, 2)]
    outer = [
        # .Louter: one panel of each stream per trip
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0)), I("add r13, r7, 0", E.add("r13", "r7", 0))),
        BUNDLE(LD("B0", 0), LD("B2", 0)),
        BUNDLE(LD("A0", 0)),
        BUNDLE(LD("A1", 0)),
        BUNDLE(I("mov r10, %d" % (iters - 1), E.mov("r10", iters - 1))),
        BUNDLE(I("loop r10", E.loop("r10"))),
    ] + kbody + [
        BUNDLE(I("loopend", E.loopend())),
        BUNDLE(I("add r6, r6, r15", E.add("r6", "r6", "r15")),
               I("add r7, r7, r16", E.add("r7", "r7", "r16"))),
        BUNDLE(I("sub r5, r5, 1", E.sub("r5", "r5", 1))),
    ]
    outer.append(BUNDLE(I("cbnz r5, .Louter", E.cbnz("r5", -len(outer)))))
    # .Lsweep: rewind both streams, then walk all NPANEL panels (fixes the working set).
    sweep = [
        BUNDLE(I("add r6, r0, 0", E.add("r6", "r0", 0)),          # A stream cursor
               I("add r7, r1, 0", E.add("r7", "r1", 0))),         # B stream cursor
        BUNDLE(I("add r5, r3, 0", E.add("r5", "r3", 0))),         # panels left in this sweep
    ] + outer + [
        BUNDLE(I("sub r4, r4, 1", E.sub("r4", "r4", 1))),
    ]
    sweep.append(BUNDLE(I("cbnz r4, .Lsweep", E.cbnz("r4", -len(sweep)))))
    body += sweep
    body += [BUNDLE(cstore(t)) for t in range(24)]
    if trace:
        body += _gemm_dist_stamp()
    return prologue(6 if trace else 5) + body + epilogue()


def _hold(nloops, reg="r9"):
    """Delay of 8*nloops cycles: a hardware loop of 8 empty bundles.
    Looped rather than straight-line so the delay is not dominated by instruction-fetch misses."""
    return _mov32(reg, nloops - 1) + [BUNDLE(I("loop %s" % reg, E.loop(reg)))] \
        + [BUNDLE() for _ in range(8)] + [BUNDLE(I("loopend", E.loopend()))]


def k_lsram_stamp_hold(nloops=300000):
    """`k_lsram_stamp` preceded by a ~8*nloops-cycle hold, so the placement probe runs as long as the jobs it models.
    Holds first, then stamps, so every task stays resident while the rest of the wave starts.
    args: r0 = PATTERN, r1 = OUT."""
    body = _hold(nloops) + _lsram_base("r5")
    for i in range(LSRAM_PROBE_WORDS):
        body += [BUNDLE(I("add r6, r0, %d" % i, E.add("r6", "r0", i))),
                 BUNDLE(I("st r6, [r5+%d]" % (4 * i), E.st("r6", "r5", 4 * i)))]
    body += [BUNDLE(I("st r0, [r1+0]", E.st("r0", "r1", 0)))]
    return prologue(2, fp=False) + body + epilogue()


def k_lsram_probe_hold(nloops=300000):
    """`k_lsram_probe` with the same hold, so the read-back wave is placed like the stamping wave. args: r0 = OUT."""
    body = _hold(nloops) + _lsram_base("r5")
    for i in range(LSRAM_PROBE_WORDS):
        body += [BUNDLE(I("ld r6, [r5+%d]" % (4 * i), E.ld("r6", "r5", 4 * i))),
                 BUNDLE(I("st r6, [r0+%d]" % (4 * i), E.st("r6", "r0", 4 * i)))]
    return prologue(1, fp=False) + body + epilogue()


GEMM_DIST_ENTRY_BIT = 0x10      # set in the tag by the ENTRY stamp, clear in the EXIT stamp


def _gemm_dist_stamp(tagreg="r17"):
    """LSRAM signature for `k_gemm_dist`, written after the C stores and before the epilogue. LSRAM is per-TEC and
    bypasses the dcache writeback path, so it distinguishes "ran to the end but C never reached DDR" from "never finished".
    Layout: word 0 = TAG (from `tagreg`, default r17: the tag arrives in r5, which the sweep loop reuses),
    word 1 = ctrl0[0x000] (TEC identity: (tec_id << 16) | (1 << 8) | task_index), words i >= 2 = TAG + i (ascending).
    Note: leaf kernels only; the prologue points `lp` at the top of the LSRAM window and nothing here pushes."""
    body = _lsram_base("r9")
    body += [BUNDLE(I("mfctrl0 r11, 0", E.mfctrl0("r11", 0))),
             BUNDLE(I("st %s, [r9+0]" % tagreg, E.st(tagreg, "r9", 0))),
             BUNDLE(I("st r11, [r9+4]", E.st("r11", "r9", 4)))]
    for i in range(2, LSRAM_PROBE_WORDS):
        body += [BUNDLE(I("add r12, %s, %d" % (tagreg, i), E.add("r12", tagreg, i))),
                 BUNDLE(I("st r12, [r9+%d]" % (4 * i), E.st("r12", "r9", 4 * i)))]
    return body


# Shared arm table for probe harnesses (simulator or device).
GEMM_DIST_ARMS = {
    "gdist63": (lambda: k_gemm_dist(63)),      # 14.1 KiB per panel pair: L2-sized
    "gdist21": (lambda: k_gemm_dist(21)),      #  4.7 KiB per panel pair: L1-sized
    # with the entry/exit LSRAM stamp
    "gdist63t": (lambda: k_gemm_dist(63, trace=True)),
    "gdist21t": (lambda: k_gemm_dist(21, trace=True)),
    # ~2 ms hold at 1.2 GHz, the same order as the GEMM jobs
    "stamphold": (lambda: k_lsram_stamp_hold(300000)),
    "probehold": (lambda: k_lsram_probe_hold(300000)),
}


def k_ctrl_delta(space=0, idx=0xd1, nbundles=1000):
    """Read control register `idx` three times, `nbundles` straight-line empty bundles apart: OUT = [first, second, third].
    Read-only. args: r0 = OUT."""
    name, mf = CTRL_MF[space]
    body = [BUNDLE(I("%s r4, %d" % (name, idx), mf("r4", idx)))]
    body += [BUNDLE() for _ in range(nbundles)]
    body += [BUNDLE(I("%s r5, %d" % (name, idx), mf("r5", idx)))]
    body += [BUNDLE() for _ in range(nbundles)]
    body += [BUNDLE(I("%s r6, %d" % (name, idx), mf("r6", idx))),
             BUNDLE(I("st r4, [r0+0]", E.st("r4", "r0", 0))),
             BUNDLE(I("st r5, [r0+4]", E.st("r5", "r0", 4))),
             BUNDLE(I("st r6, [r0+8]", E.st("r6", "r0", 8)))]
    return prologue(1, fp=False) + body + epilogue()


def k_cop_range(mode=None, delay=24000):
    """Store 0x5A00+i at BUF+16i over 4 KiB (64 lines), one `cacheop mode` at OPADDR, then `delay` empty bundles (time
    for an asynchronous op, no memory traffic), then exit without the writeback. The words that land show the op's range.
    args: r0 = BUF, r1 = OPADDR."""
    body = [BUNDLE(I("add r8, r0, 0", E.add("r8", "r0", 0)))]
    for i in range(256):
        if i and (16 * i) % 256 == 0:
            body.append(BUNDLE(I("add r8, r8, 256", E.add("r8", "r8", 256))))
        body += [BUNDLE(I("mov r4, %d" % (0x5A00 + i), E.mov("r4", 0x5A00 + i))),
                 BUNDLE(I("st r4, [r8+%d]" % ((16 * i) % 256), E.st("r4", "r8", (16 * i) % 256)))]
    if mode is not None:
        body.append(BUNDLE(I("cacheop [r1+0], %d" % mode, E.cacheop("r1", mode))))
    body += [BUNDLE(I("mov r10, %d" % (delay - 1), E.mov("r10", delay - 1))),
             BUNDLE(I("loop r10", E.loop("r10"))),
             BUNDLE(),
             BUNDLE(I("loopend", E.loopend()))]
    return prologue(2, fp=False) + body + _nowb_epilogue()


def k_ctrl_loopdelta(space=0, idx=0xd1, iters=1000, body=8, kind="nop"):
    """`k_ctrl_delta` with each delay a hardware loop of `iters` x `body` bundles (stays in the I-cache).
    kind="nop": empty bundles; otherwise {mma, mma} bundles (2 cycles each), telling a cycle counter from a bundle counter.
    OUT = [first, second, third]; (second - first) / (iters * body) = counter ticks per bundle. args: r0 = OUT."""
    name, mf = CTRL_MF[space]
    n = iters - 1
    lo = n & 0xFFFF; lo = lo - 0x10000 if lo >= 0x8000 else lo       # mov sign-extends its 16 bits
    hi = n >> 16                                                      # movh REPLACES the top 16
    def bundle_():
        if kind == "nop": return BUNDLE()
        # {mma, mma} costs two cycles: a cycle counter reads 16 per 8 bundles, a bundle counter 8.
        return BUNDLE(I("mma t0.fp32, t1.fp32, t16.fp16, t17.fp16", E.mma("t0", "t16", "t17")),
                      I("mma t2.fp32, t3.fp32, t16.fp16, t17.fp16", E.mma("t2", "t16", "t17")))
    def delay():
        return [BUNDLE(I("mov r10, %d" % lo, E.mov("r10", lo))),
                BUNDLE(I("movh r10, %d" % hi, E.movh("r10", hi))),
                BUNDLE(I("loop r10", E.loop("r10")))] + [bundle_() for _ in range(body)] + [BUNDLE(I("loopend", E.loopend()))]
    body_ = [BUNDLE(I("%s r4, %d" % (name, idx), mf("r4", idx)))] + delay()
    body_ += [BUNDLE(I("%s r5, %d" % (name, idx), mf("r5", idx)))] + delay()
    body_ += [BUNDLE(I("%s r6, %d" % (name, idx), mf("r6", idx))),
              BUNDLE(I("st r4, [r0+0]", E.st("r4", "r0", 0))),
              BUNDLE(I("st r5, [r0+4]", E.st("r5", "r0", 4))),
              BUNDLE(I("st r6, [r0+8]", E.st("r6", "r0", 8)))]
    return prologue(1, fp=(kind != "nop")) + body_ + epilogue()


def k_cop_range2(mode=None, nlines=512, delay=24000):
    """`k_cop_range` over `nlines` dirty lines (0x5A00+i at BUF+64i). 512 lines = 32 KiB fills every L2 set four ways
    deep, so a mode that cleans a subset (way, set, range) shows as the subset that lands. args: r0 = BUF, r1 = OPADDR."""
    body = [BUNDLE(I("add r8, r0, 0", E.add("r8", "r0", 0)))]
    for i in range(nlines):
        if i and i % 4 == 0:
            body.append(BUNDLE(I("add r8, r8, 256", E.add("r8", "r8", 256))))
        body += [BUNDLE(I("mov r4, %d" % (0x5A00 + i), E.mov("r4", 0x5A00 + i))),
                 BUNDLE(I("st r4, [r8+%d]" % (64 * (i % 4)), E.st("r4", "r8", 64 * (i % 4))))]
    if mode is not None:
        body.append(BUNDLE(I("cacheop [r1+0], %d" % mode, E.cacheop("r1", mode))))
    body += [BUNDLE(I("mov r10, %d" % (delay - 1), E.mov("r10", delay - 1))),
             BUNDLE(I("loop r10", E.loop("r10"))),
             BUNDLE(),
             BUNDLE(I("loopend", E.loopend()))]
    return prologue(2, fp=False) + body + _nowb_epilogue()


def k_gemm_cachefed33(space="cache", iters=16):
    """`k_gemm_pipe33` (3x3 block, two operand register sets so loads run a whole k-step ahead) fed from DDR through the
    cache. iters=16 = 64 k-steps = 12.3 KiB (L2-sized); iters=8 = 6.1 KiB (L1-sized).
    args: r0 = SRC (A at 0, B at PIPE_B_OFF, 96 B per step each), r1 = unused, r2 = OUT (576 B), r3 = REPS."""
    def A(s, i): return "t%d" % (18 + 6 * s + i)
    def B(s, j): return "t%d" % (18 + 6 * s + 3 + j)
    def MMA(s, i, j):
        c = 2 * (3 * i + j)
        return I("mma t%d.fp32, t%d.fp32, %s.fp16, %s.fp16" % (c, c + 1, A(s, i), B(s, j)),
                 E.mma("t%d" % c, A(s, i), B(s, j)))
    def VA(s, k, i): return I("ld %s, [r8+%d]" % (A(s, i), 96 * k + 32 * i), E.vld(A(s, i), "r8", 96 * k + 32 * i))
    def VB(s, k, j): return I("ld %s, [r13+%d]" % (B(s, j), 96 * k + 32 * j), E.vld(B(s, j), "r13", 96 * k + 32 * j))

    # the 36 mma of four steps, in order; bundle b holds mma 2b and 2b+1
    mmas = [MMA(k % 2, i, j) for k in range(4) for i in range(3) for j in range(3)]
    # step k+1's loads (k+1 == 4: next iteration's step 0, offset 384) go into set (k+1) % 2, in the three bundles
    # right after step k-1's last read of that set: k=0 -> b0, 1 -> b5, 2 -> b9, 3 -> b14
    start = {0: 0, 1: 5, 2: 9, 3: 14}
    loads = {}
    for k in range(4):
        s = (k + 1) % 2
        for i in range(3):
            loads[start[k] + i] = (VA(s, k + 1, i), VB(s, k + 1, i))
    kbody = []
    for b in range(18):
        ops = [mmas[2 * b], mmas[2 * b + 1]]
        if b in loads:
            ops += list(loads[b])
        kbody.append(BUNDLE(*ops))
    assert len(kbody) == 18
    kbody.append(BUNDLE(I("add r8, r8, 384", E.add("r8", "r8", 384)),
                        I("add r13, r13, 384", E.add("r13", "r13", 384))))

    def cbase(t): return ("r2", t * 32) if t < 16 else ("r14", (t - 16) * 32)
    def cload(t):
        b, off = cbase(t); return I("ld t%d, [%s+%d]" % (t, b, off), E.vld("t%d" % t, b, off))
    def cstore(t):
        b, off = cbase(t); return I("st t%d, [%s+%d]" % (t, b, off), E.vst("t%d" % t, b, off))

    assert space in ("cache", "stream")
    body = [BUNDLE(I("add r6, r0, 0", E.add("r6", "r0", 0))),                  # panels in DDR, not LSRAM
            BUNDLE(I("add r14, r2, 512", E.add("r14", "r2", 512))),
            BUNDLE(I("mov r15, %d" % PIPE_B_OFF, E.mov("r15", PIPE_B_OFF))),
            BUNDLE(I("mov r16, %d" % 16384, E.mov("r16", 16384)))]
    body += [BUNDLE(cload(t), cload(t + 1)) for t in range(0, 18, 2)]
    # .Louter: re-arm bases, load set 0 for step 0, then the hardware loop
    outer = [
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0)), I("add r13, r6, r15", E.add("r13", "r6", "r15"))),
        BUNDLE(VA(0, 0, 0), VB(0, 0, 0)),
        BUNDLE(VA(0, 0, 1), VB(0, 0, 1)),
        BUNDLE(VA(0, 0, 2), VB(0, 0, 2)),
        BUNDLE(I("mov r10, %d" % (iters - 1), E.mov("r10", iters - 1))),
        BUNDLE(I("loop r10", E.loop("r10"))),
    ] + kbody + [
        BUNDLE(I("loopend", E.loopend())),
        BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1)),
               *([I("add r6, r6, r16", E.add("r6", "r6", "r16"))] if space == "stream" else [])),
    ]
    outer.append(BUNDLE(I("cbnz r3, .Louter", E.cbnz("r3", -len(outer)))))
    body += outer
    body += [BUNDLE(cstore(t)) for t in range(18)]
    return prologue(4) + body + epilogue()


OP_PROBE = {          # op -> (encoder, a, b, expected a OP b); a/b chosen so 0 cannot pass
    "and":  (E.and_, 0x0F0F0F0F, 0x00FFFF00, 0x000F0F00),
    "or":   (E.or_,  0x0F0F0F0F, 0x00FFFF00, 0x0FFFFF0F),
    "xor":  (E.xor,  0x0F0F0F0F, 0x00FFFF00, 0x0FF0F00F),
    "andl": (lambda d, a, b: E.andl(d, a, 7), 0x0F0F0F0F, 7, 0x07),
}


def k_op_probe(op):
    """Execute ONE scalar op on two constants and store the result.  args: r0=OUT.

    Isolates a single register-form ALU op (one bundle, one op, one store) so that
    "the instruction hangs" is separable from a broken image."""
    enc, a, b, _want = OP_PROBE[op]
    lo = lambda v: (v & 0xFFFF) - 0x10000 if (v & 0xFFFF) >= 0x8000 else (v & 0xFFFF)
    body = [BUNDLE(I("mov r4, %d" % lo(a), E.mov("r4", lo(a)))),
            BUNDLE(I("movh r4, %d" % (a >> 16), E.movh("r4", a >> 16))),
            BUNDLE(I("mov r5, %d" % lo(b), E.mov("r5", lo(b)))),
            BUNDLE(I("movh r5, %d" % (b >> 16), E.movh("r5", b >> 16))),
            BUNDLE(I("%s r6, r4, %s" % (op, "7" if op.endswith("l") else "r5"), enc("r6", "r4", "r5"))),
            BUNDLE(I("st r6, [r0+0]", E.st("r6", "r0", 0)))]
    return prologue(1, fp=False) + body + epilogue()


# ---- DMA arms ----
# Single registry shared by the image builder and the host harness.
#   dma_*   VERIFY (`gate="full"`): whole destination window copied back; kept small because
#           the copy-back is O(N) scalar.
#   dmaf*   sync-flag register (`flagprobe`): ctrl0[0x30] before / between / after.
#   dmar*   RATE (`gate="ends"`): O(1) gate at head AND tail (a capped width writes a correct
#           head and no tail). Size comes from the descriptor, so one image serves every size.
DMA_ARMS = {
    # verify: one per mode, plus controls expected to come out wrong
    "dma_direct": lambda: k_dma_mode(),
    "dma_ms0":    lambda: k_dma_mode(mode=E.DMA_MEMSET, data_unit=0),
    "dma_ms1":    lambda: k_dma_mode(mode=E.DMA_MEMSET, data_unit=1),
    "dma_ms2":    lambda: k_dma_mode(mode=E.DMA_MEMSET, data_unit=2),
    "dma_tr":     lambda: k_dma_mode(mode=E.DMA_TRANSPOSE, data_unit=0),
    "dma_trh":    lambda: k_dma_mode(mode=E.DMA_TRANSPOSE, data_unit=1),
    "dma_up":     lambda: k_dma_mode(mode=E.DMA_UPSAMPLE, data_unit=0),
    # both ends kGlobal: DDR->DDR transpose
    "dma_ddr":    lambda: k_dma_mode(mode=E.DMA_TRANSPOSE, data_unit=0, int_base=E.DMA_GLOBAL,
                                     int_ddr=True),
    # flag register: request and wait on flag f, read ctrl0[0x30] three times
    **{"dmaf%d" % f: (lambda f=f: k_dma_mode(flagprobe=True, flag=f)) for f in range(5)},
    # rate: one image per mode, size carried by the descriptor
    **{"dmar%d" % m: (lambda m=m: k_dma_mode(mode=m, gate="ends")) for m in range(4)},
    # rate into DDR, for widths LSRAM's 32 KiB cannot hold
    "dmardw":     lambda: k_dma_mode(int_base=E.DMA_GLOBAL, int_ddr=True),
    # rate DDR->DDR, one per mode. Supersedes the LSRAM rate arms: LSRAM (32 KiB) cannot hold a
    # transfer large enough to rise above the ~20 us job floor; `trans_size` is 24-bit, so DDR can.
    **{"dmad%d" % m: (lambda m=m: k_dma_mode(mode=m, int_base=E.DMA_GLOBAL, int_ddr=True))
       for m in range(4)},
    # INT2EXT variant for the two modes with asymmetric descriptor halves (DDR on the external
    # side, value/source on the internal one). See `swap_addr`.
    **{"dmads%d" % m: (lambda m=m: k_dma_mode(mode=m, int_base=E.DMA_GLOBAL, int_ddr=True,
                                              dir_=E.DMA_INT2EXT, swap_addr=True))
       for m in (2, 3)},
}


def k_dma_flight(nflags=1, mode=E.DMA_DIRECT):
    """N DMA requests in flight at once on N distinct sync flags, then one wait for all N.
    Measures whether DMA throughput is bounded by outstanding requests rather than bandwidth.

    args: r0=SRC, r1=DESC, r2=OUT, r3=ITERS, r4=TILE bytes, r5=POISON

    One request per flag (a flag is the outstanding-request slot); `wfe(-(1 << nflags), 1)` waits
    for all N. `nflags` <= 4: flag 4 aliases onto flag 0 and stays raised. Each flag has its own
    descriptor, 64 B apart. Destination tiles (LSRAM) and sources (DDR) are disjoint; sources
    advance by `nflags * TILE` per iteration so nothing is re-read."""
    if not 1 <= nflags <= 4:
        raise ValueError(f"nflags must be 1..4 (flag 4 aliases onto flag 0); got {nflags}")
    INT, EXT, DSC, RSY = 16, 20, 12, 24     # register bases for the four per-flag quantities
    body = _lsram_base("r6")
    # int addrs: LSRAM + f*TILE, fixed for the whole run
    body += [BUNDLE(I("add r%d, r6, 0" % INT, E.add("r%d" % INT, "r6", 0)))]
    for f in range(1, nflags):
        body += [BUNDLE(I("add r%d, r%d, r4" % (INT + f, INT + f - 1),
                          E.add("r%d" % (INT + f), "r%d" % (INT + f - 1), "r4")))]
    # ext addrs: SRC + f*TILE, advanced by nflags*TILE each iteration
    body += [BUNDLE(I("add r%d, r0, 0" % EXT, E.add("r%d" % EXT, "r0", 0)))]
    for f in range(1, nflags):
        body += [BUNDLE(I("add r%d, r%d, r4" % (EXT + f, EXT + f - 1),
                          E.add("r%d" % (EXT + f), "r%d" % (EXT + f - 1), "r4")))]
    # descriptor addrs: DESC + f*64
    body += [BUNDLE(I("add r%d, r1, 0" % DSC, E.add("r%d" % DSC, "r1", 0)))]
    for f in range(1, nflags):
        body += [BUNDLE(I("add r%d, r%d, %d" % (DSC + f, DSC + f - 1, D.DESC_SLOT),
                          E.add("r%d" % (DSC + f), "r%d" % (DSC + f - 1), D.DESC_SLOT)))]
    # rsync values: (f << 8) | f. Flag 0 is 0, for which `zero` serves.
    for f in range(1, nflags):
        v = (f << 8) | f
        body += [BUNDLE(I("mov r%d, %d" % (RSY + f, v), E.mov("r%d" % (RSY + f), v)))]
    # r11 = nflags * TILE, the per-iteration source advance
    body += [BUNDLE(I("add r11, r4, 0", E.add("r11", "r4", 0)))]
    for _ in range(nflags - 1):
        body += [BUNDLE(I("add r11, r11, r4", E.add("r11", "r11", "r4")))]
    # r10 = the wait mask, ~((1 << nflags) - 1) == -(1 << nflags)
    body += [BUNDLE(I("sub r10, zero, %d" % (1 << nflags), E.sub("r10", "zero", 1 << nflags)))]

    loop = []
    for f in range(nflags):
        rs = "zero" if f == 0 else "r%d" % (RSY + f)
        loop += [BUNDLE(I("dma %d, %s, 0, %d, %d, %d, 0, r%d, r%d, r%d"
                          % (mode, rs, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT,
                             DSC + f, INT + f, EXT + f),
                          E.dma(mode, rs, 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT,
                                E.DMA_USELESS, "r%d" % (DSC + f), "r%d" % (INT + f),
                                "r%d" % (EXT + f))))]
    loop += [BUNDLE(I("wfe r10, 1", E.wfe("r10", 1)))]
    for f in range(nflags):
        loop += [BUNDLE(I("add r%d, r%d, r11" % (EXT + f, EXT + f),
                          E.add("r%d" % (EXT + f), "r%d" % (EXT + f), "r11")))]
    loop += [BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1)))]
    loop += [BUNDLE(I("cbnz r3, .Lflight", E.cbnz("r3", -len(loop))))]
    body += loop

    # gate: 8 words from the head of every tile, so a request that never landed shows
    for f in range(nflags):
        for i in range(8):
            body += [BUNDLE(I("ld r9, [r%d+%d]" % (INT + f, 4 * i), E.ld("r9", "r%d" % (INT + f), 4 * i))),
                     BUNDLE(I("st r9, [r2+%d]" % (4 * (8 * f + i)), E.st("r9", "r2", 4 * (8 * f + i))))]
    return prologue(6, fp=False) + body + epilogue()


DMA_ARMS.update({"dmaflight%d" % n: (lambda n=n: k_dma_flight(n)) for n in (1, 2, 3, 4)})


# ==== control registers / exception vectors ====
# Reading an unimplemented control index may fault; the diag vector table turns the fault into
# a readout. Convention for the kernels below:
#     r0 = PROG   poisoned progress buffer, written as the kernel goes; the end of the written
#                 prefix locates a fault.
#     r2 = VEC    one word; the diag handlers overwrite it with 0x1000+n (vector n).
CTRL_PROG_POISON = 0xDEADBEEF          # host fill value for PROG
CTRL_SENTINEL_HI = 0x5EED              # top half of the "read did not land" sentinel;
                                       # must be <= 0x7FFF (`movh` immediate is signed 16-bit)


def _ctrl_read_sentinel(name, mf, i):
    """Seed r4 with a per-index sentinel (`CTRL_SENTINEL_HI << 16 | i`), then read control index `i`.

    Required: an unimplemented index leaves the destination unwritten, so without the seed a
    refused read would store the previous index's value. Outcomes: the value, the sentinel
    (read did not land), or host poison (kernel faulted before the store)."""
    return [BUNDLE(I("mov r4, %d" % i, E.mov("r4", i))),
            BUNDLE(I("movh r4, %d" % CTRL_SENTINEL_HI, E.movh("r4", CTRL_SENTINEL_HI))),
            BUNDLE(I("%s r4, %d" % (name, i), mf("r4", i)))]


def k_ctrl_scan(space=0, lo=0, n=1024):
    """Read control indices `lo` .. `lo+n-1` of `space` (10-bit index, 1024 per space) into PROG.
    args: r0=PROG (4*n bytes, poisoned), r2=VEC.

    Read-then-store one index at a time: if index k faults, PROG[0..k-1] hold values and PROG[k]
    is poison (assumes a precise fault; confirm with `k_ctrl_one`). Reads only: writing an
    unknown control register can wedge the device."""
    name, mf = CTRL_MF[space]
    body = [BUNDLE(I("add r5, r0, 0", E.add("r5", "r0", 0)))]
    for k in range(n):
        if k and k % 64 == 0:                  # `st`'s offset reaches 252 B
            body.append(BUNDLE(I("add r5, r5, 256", E.add("r5", "r5", 256))))
        body += _ctrl_read_sentinel(name, mf, lo + k) + \
                [BUNDLE(I("st r4, [r5+%d]" % (4 * (k % 64)), E.st("r4", "r5", 4 * (k % 64))))]
    return prologue(3, fp=False) + body + epilogue()


def k_ctrl_one(space=0, idx=0, reps=2):
    """Read ONE control index `reps` times into PROG[0..reps-1].  args: r0=PROG, r2=VEC.

    Confirms a `k_ctrl_scan` fault boundary; with reps>=2 also distinguishes constant from
    changing registers."""
    name, mf = CTRL_MF[space]
    body = []
    for k in range(reps):
        body += _ctrl_read_sentinel(name, mf, idx) + \
                [BUNDLE(I("st r4, [r0+%d]" % (4 * k), E.st("r4", "r0", 4 * k)))]
    return prologue(3, fp=False) + body + epilogue()


def k_ctrl_delta_scan(space=0, lo=0, n=64, iters=1000):
    """Read every index of a range, run `iters` x 8 empty bundles, read them all again.
    args: r0=PROG (8*n bytes, poisoned), r2=VEC.  PROG[2i] = before, PROG[2i+1] = after.

    The before/after difference classifies registers as constant, free-running or
    work-proportional (e.g. the cycle counter ctrl0[0xd1]). The delay is a hardware loop, since
    straight-line empty bundles are dominated by instruction-fetch misses."""
    name, mf = CTRL_MF[space]
    idx = list(range(lo, lo + n))

    # a fresh base every 32 indices keeps every offset inside `st`'s 8-bit reach
    def reads(word_off):
        out = []
        for k, i in enumerate(idx):
            if k % 32 == 0:
                out.append(BUNDLE(I("add r5, r0, %d" % (256 * (k // 32)), E.add("r5", "r0", 256 * (k // 32)))))
            o = 8 * (k % 32) + 4 * word_off
            out += _ctrl_read_sentinel(name, mf, i) + \
                   [BUNDLE(I("st r4, [r5+%d]" % o, E.st("r4", "r5", o)))]
        return out

    delay = [BUNDLE(I("mov r6, %d" % iters, E.mov("r6", iters))),
             BUNDLE(I("loop r6", E.loop("r6")))] + \
            [BUNDLE() for _ in range(7)] + \
            [BUNDLE(I("loopend", E.loopend()))]
    return prologue(3, fp=False) + reads(0) + delay + reads(1) + epilogue()


# ---- fault arms ----
# One generator; arms differ only in the provocation. 0xB1 is stored before it and 0xB2 after:
# [0xB1,poison]+VEC set = caught fault; [0xB1,0xB2]+VEC poison = no effect; [poison,..] = not run.
FAULT_KINDS = ("none", "fp", "badaddr", "misalign", "break", "irtn")
WILD_HI = 0x7FF0           # r6 = 0x7FF00000, an unmapped device address


def k_fault(kind="none"):
    """args: r0=PROG (8 B, poisoned), r2=VEC (4 B, poisoned).  `kind` selects the provocation.

    `none` is the negative control (table must not fire spuriously); `fp` is the positive
    control (vector FP op without the FP enable traps to vector 3)."""
    assert kind in FAULT_KINDS, kind
    pre = [BUNDLE(I("mov r4, 177", E.mov("r4", 0xB1))),
           BUNDLE(I("st r4, [r0+0]", E.st("r4", "r0", 0)))]
    post = [BUNDLE(I("mov r5, 178", E.mov("r5", 0xB2))),
            BUNDLE(I("st r5, [r0+4]", E.st("r5", "r0", 4)))]
    if kind == "none":
        mid = []
    elif kind == "fp":
        # prologue uses fp=False, so the FP enable is not set
        mid = [BUNDLE(I("fma t0.fp32, t1.fp32, t2.fp32, p7.w", E.fma("t0", "t1", "t2")))]
    elif kind == "badaddr":
        mid = [BUNDLE(I("mov r6, 0", E.mov("r6", 0))),
               BUNDLE(I("movh r6, %d" % WILD_HI, E.movh("r6", WILD_HI))),
               BUNDLE(I("ld r7, [r6+0]", E.ld("r7", "r6", 0)))]
    elif kind == "misalign":
        mid = [BUNDLE(I("add r6, r0, 1", E.add("r6", "r0", 1))),
               BUNDLE(I("ld r7, [r6+0]", E.ld("r7", "r6", 0)))]
    elif kind == "break":
        mid = [BUNDLE(I("break", E.break_()))]
    else:
        mid = [BUNDLE(I("irtn", E.irtn()))]
    return prologue(3, fp=False) + pre + mid + post + epilogue()


# Control indices known safe to read; the only ones a fault handler may dump (a faulting read
# inside a handler is not assumed recoverable).
CTRL_SYNDROME_REGS = [(sp, i) for sp in (0, 1, 2) for i in CTRL_SIM_VALID[sp]]


def k_ctrl_list(regs):
    """Read an explicit list of (space, index) into PROG[1..], with PROG[0] left alone.
    args: r0=PROG, r2=VEC.

    Control for the syndrome arm; matches the handler's layout (vector at +0, dump from +4)."""
    mfs = {0: ("mfctrl0", E.mfctrl0), 1: ("mfctrl1", E.mfctrl1), 2: ("mfctrl2", E.mfctrl2)}
    body = [BUNDLE(I("add r6, r0, 4", E.add("r6", "r0", 4)))]
    for k, (sp, idx) in enumerate(regs):
        if k and k % 63 == 0:
            body.append(BUNDLE(I("add r6, r6, 252", E.add("r6", "r6", 252))))
        name, mf = mfs[sp]
        body += _ctrl_read_sentinel(name, mf, idx) + \
                [BUNDLE(I("st r4, [r6+%d]" % (4 * (k % 63)), E.st("r4", "r6", 4 * (k % 63))))]
    return prologue(3, fp=False) + body + epilogue()


def ctrl_arm(name):
    """Resolve a control/fault image name to (bundles, vector-table mode, syndrome regs).

    Parametric so any index (e.g. `cone0_0x1e3_4`) is buildable by name. Mode is "diag"
    (reporting table) or "spin" (shipped table; those arms are expected to hang)."""
    import re
    m = re.fullmatch(r"cscan(\d)_(\w+)_(\d+)", name)
    if m:
        return k_ctrl_scan(int(m[1]), int(m[2], 0), int(m[3])), "diag", ()
    m = re.fullmatch(r"cone(\d)_(\w+)_(\d+)", name)
    if m:
        return k_ctrl_one(int(m[1]), int(m[2], 0), int(m[3])), "diag", ()
    m = re.fullmatch(r"cdelta(\d)_(\w+)_(\d+)_(\d+)", name)
    if m:
        return k_ctrl_delta_scan(int(m[1]), int(m[2], 0), int(m[3]), int(m[4])), "diag", ()
    m = re.fullmatch(r"v(%s)(s?)" % "|".join(FAULT_KINDS), name)
    if m:
        return k_fault(m[1]), ("spin" if m[2] else "diag"), ()
    m = re.fullmatch(r"vret(%s)" % "|".join(FAULT_KINDS), name)
    if m:                                 # handlers end with `irtn`: is the fault recoverable?
        return k_fault(m[1]), "diagret", ()
    m = re.fullmatch(r"vfix(%s)" % "|".join(FAULT_KINDS), name)
    if m:                                 # handlers repair the cause before returning
        return k_fault(m[1]), "diagfix", ()
    m = re.fullmatch(r"vsyn(%s)" % "|".join(FAULT_KINDS), name)
    if m:                                 # syndrome arm: fault, handler dumps control regs
        return k_fault(m[1]), "diag", CTRL_SYNDROME_REGS
    if name == "vsyn":
        return k_fault("irtn"), "diag", CTRL_SYNDROME_REGS
    if name == "vsynclean":               # control: same reads, no fault
        return k_ctrl_list(CTRL_SYNDROME_REGS), "diag", ()
    return None


# ==== sync flags ====
# Reads the DMA busy register ctrl0[0x30] while varying flag number, the two sync-word halves,
# and the `wfe` mode. Register convention (compatible with the diag vector table):
#     r0 = SRC   r1 = DESC   r2 = VEC (fault handler writes 0x1000+n)   r4 = PROBE
SYNC_PROBE_WORDS = 4


def _mov32(reg, value):
    """Materialise an arbitrary 32-bit constant into `reg` (two bundles).

    `mov` takes a signed 16-bit immediate and `movh` replaces the top 16 bits, so each half is
    passed as a signed value. Used for masks/sync words beyond the 10-bit `add`/`sub` immediates."""
    lo, hi = value & 0xFFFF, (value >> 16) & 0xFFFF
    slo = lo - 0x10000 if lo >= 0x8000 else lo
    shi = hi - 0x10000 if hi >= 0x8000 else hi
    return [BUNDLE(I("mov %s, %d" % (reg, slo), E.mov(reg, slo))),
            BUNDLE(I("movh %s, %d" % (reg, shi), E.movh(reg, shi)))]


def k_sync_probe(sync_hi=0, sync_lo=0, wait_bits=(0,), wfe_mode=1, wait=True,
                 second=None, nwords=64):
    """One DMA request with an arbitrary sync word, then an arbitrary `wfe`.

    args: r0=SRC, r1=DESC, r2=VEC, r3=unused, r4=PROBE.
    PROBE[0..3] = ctrl0[0x30] before the request, before the wait, after the wait, and after a
    short delay (still raised vs. cleared late); PROBE[4..] = GATE_ENDS words of the destination.

    sync_hi/sync_lo: the two 8-bit halves of the sync word, settable independently.
    wait_bits: flags named in the `wfe` mask; the mask register is `~OR(1 << b)`.
    wait=False: no `wfe` (tests whether a raised flag survives into the next job).
    second=(hi, lo): issue a second request before the wait."""
    if not 0 <= wfe_mode < 32:
        raise ValueError("wfe mode is a 5-bit field: %d" % wfe_mode)
    mask = 0
    for b in wait_bits:
        mask |= 1 << b

    def probe(slot):
        return [BUNDLE(I("mfctrl0 r12, 0x30", E.mfctrl0("r12", 0x30))),
                BUNDLE(I("st r12, [r4+%d]" % (4 * slot), E.st("r12", "r4", 4 * slot)))]

    def request(hi, lo, rreg):
        out = []
        w = (hi << 8) | lo
        if w:
            out += _mov32(rreg, w)
            rs = rreg
        else:
            rs = "zero"          # sync word 0: use `zero`, as the vendor compiler does
        out += [BUNDLE(I("dma %d, %s, 0, %d, %d, %d, %d, r1, r6, r0"
                         % (E.DMA_DIRECT, rs, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT,
                            E.DMA_USELESS),
                         E.dma(E.DMA_DIRECT, rs, 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT,
                               E.DMA_USELESS, "r1", "r6", "r0")))]
        return out

    body = _lsram_base("r6")
    body += probe(0)
    body += request(sync_hi, sync_lo, "r11")
    if second is not None:
        # second descriptor 64 B along, so the requests do not share a descriptor
        body += [BUNDLE(I("add r1, r1, %d" % D.DESC_SLOT, E.add("r1", "r1", D.DESC_SLOT))),
                 # `add` immediate is 10-bit (max 1023); 512 B clears a 256 B default request
                 BUNDLE(I("add r6, r6, 512", E.add("r6", "r6", 512)))]
        body += request(second[0], second[1], "r13")
    body += probe(1)
    if wait:
        # wait mask = ~mask via _mov32; the vendor's `sub r7, zero, mask+1` only reaches 10 bits
        body += _mov32("r7", (~mask) & 0xFFFFFFFF)
        body += [BUNDLE(I("wfe r7, %d" % wfe_mode, E.wfe("r7", wfe_mode)))]
    body += probe(2)
    # short delay, then one more reading: "still raised" vs "cleared late"
    body += [BUNDLE(I("mov r8, 64", E.mov("r8", 64))),
             BUNDLE(I("loop r8", E.loop("r8")))] + [BUNDLE() for _ in range(7)] + \
            [BUNDLE(I("loopend", E.loopend()))]
    body += probe(3)
    # byte gate: copy GATE_ENDS destination words to PROBE[4..] so "flag raised" is separable
    # from "data arrived"
    body += _lsram_base("r9")
    for i in range(GATE_ENDS):
        body += [BUNDLE(I("ld r10, [r9+%d]" % (4 * i), E.ld("r10", "r9", 4 * i))),
                 BUNDLE(I("st r10, [r4+%d]" % (16 + 4 * i), E.st("r10", "r4", 16 + 4 * i)))]
    return prologue(5, fp=False) + body + epilogue()


def k_sync_bytec():
    """Each TEC raises the sync flag equal to its own index (`ctrl0[0] >> 16`), then reads the
    busy register.  args: r0=SRC, r1=DESC, r2=VEC, r3=unused, r4=PROBE.

        private flags -> TEC t sees only bit t
        shared flags  -> every TEC sees all four bits

    The sync word's high byte is inert, so the index is passed to `dma` unshifted as the low byte.
    PROBE: [0..2] ctrl0[0x30] readings, [3] TEC index, [4..] byte gate, [12] cycle counter."""
    body = [BUNDLE(I("mfctrl0 r11, 0", E.mfctrl0("r11", 0))),
            BUNDLE(I("mov r13, 16", E.mov("r13", 16))),
            BUNDLE(I("lsr r11, r11, r13", E.lsr("r11", "r11", "r13")))]   # r11 = this TEC's index
    body += _lsram_base("r6")
    body += [BUNDLE(I("mfctrl0 r12, 0x30", E.mfctrl0("r12", 0x30))),
             BUNDLE(I("st r12, [r4+0]", E.st("r12", "r4", 0)))]
    body += [BUNDLE(I("dma %d, r11, 0, %d, %d, %d, %d, r1, r6, r0"
                      % (E.DMA_DIRECT, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT, E.DMA_USELESS),
                      E.dma(E.DMA_DIRECT, "r11", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT,
                            E.DMA_USELESS, "r1", "r6", "r0")))]
    body += [BUNDLE(I("mfctrl0 r12, 0x30", E.mfctrl0("r12", 0x30))),
             BUNDLE(I("st r12, [r4+4]", E.st("r12", "r4", 4)))]
    # r7 = ~(1 << tec) = -(1 << tec) - 1. `lsl` is register-form only, hence the shift in r11.
    body += [BUNDLE(I("mov r12, 1", E.mov("r12", 1))),
             BUNDLE(I("lsl r12, r12, r11", E.lsl("r12", "r12", "r11"))),
             BUNDLE(I("sub r7, zero, r12", E.sub("r7", "zero", "r12"))),
             BUNDLE(I("sub r7, r7, 1", E.sub("r7", "r7", 1))),
             BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    body += [BUNDLE(I("mfctrl0 r12, 0x30", E.mfctrl0("r12", 0x30))),
             BUNDLE(I("st r12, [r4+8]", E.st("r12", "r4", 8))),
             BUNDLE(I("st r11, [r4+12]", E.st("r11", "r4", 12)))]   # PROBE[3] = which TEC this is
    body += _lsram_base("r9")
    for i in range(GATE_ENDS):
        body += [BUNDLE(I("ld r10, [r9+%d]" % (4 * i), E.ld("r10", "r9", 4 * i))),
                 BUNDLE(I("st r10, [r4+%d]" % (16 + 4 * i), E.st("r10", "r4", 16 + 4 * i)))]
    # PROBE[12] = cycle counter: the private/shared reading is only valid if the four tasks ran
    # concurrently, which close counter values confirm.
    body += [BUNDLE(I("mfctrl0 r10, 209", E.mfctrl0("r10", 0xd1))),
             BUNDLE(I("st r10, [r4+48]", E.st("r10", "r4", 48)))]
    return prologue(5, fp=False) + body + epilogue()


def sync_arm(name):
    """Resolve a sync-flag image name to (bundles, vector-table mode, syndrome regs). Names:

        sf<N>            request and wait on flag N          (N may exceed the 4-flag budget)
        sfhl<H>_<L>      sync word (H << 8) | L, wait on H   -- the asymmetric arm
        sfw<H>_<L>_<W>   ...and wait on flag W instead
        sfm<N>_<M>       flag N, `wfe` mode M
        sfwm<H>_<L>_<W>_<M>  hi, lo, wait flag, `wfe` mode
        sfe<M> / sfz<M>  `wfe` mode M with mask register 0xFFFFFFFF / 0
        sfbytec          per-TEC flag (`k_sync_bytec`)
        sfnw<N>          flag N, no wait
        sf2_<A>_<B>      two requests, flags A and B
        sfmask<N>_<B>    flag N, wait mask also names flag B
    """
    import re
    m = re.fullmatch(r"sf(\d+)", name)
    if m:
        f = int(m[1]); return k_sync_probe(f, f, (f,)), "diag", ()
    m = re.fullmatch(r"sfhl(\d+)_(\d+)", name)
    if m:
        h, l = int(m[1]), int(m[2]); return k_sync_probe(h, l, (h,)), "diag", ()
    m = re.fullmatch(r"sfw(\d+)_(\d+)_(\d+)", name)
    if m:
        h, l, w = int(m[1]), int(m[2]), int(m[3]); return k_sync_probe(h, l, (w,)), "diag", ()
    m = re.fullmatch(r"sfm(\d+)_(\d+)", name)
    if m:
        f, md = int(m[1]), int(m[2]); return k_sync_probe(f, f, (f,), wfe_mode=md), "diag", ()
    m = re.fullmatch(r"sfwm(\d+)_(\d+)_(\d+)_(\d+)", name)
    if m:
        h, l, w, md = (int(x) for x in m.groups())
        return k_sync_probe(h, l, (w,), wfe_mode=md), "diag", ()
    if name == "sfbytec":
        return k_sync_bytec(), "diag", ()
    m = re.fullmatch(r"sfe(\d+)", name)
    if m:                                 # mask register = 0xFFFFFFFF (no wait bits named)
        return k_sync_probe(0, 0, (), wfe_mode=int(m[1])), "diag", ()
    m = re.fullmatch(r"sfz(\d+)", name)
    if m:
        # mask register = 0: distinguishes "wait for all busy flags" (blocks) from
        # "wait while busy & mask" (returns immediately) for `wfe` mode 2
        return k_sync_probe(0, 0, tuple(range(32)), wfe_mode=int(m[1])), "diag", ()
    m = re.fullmatch(r"sfnw(\d+)", name)
    if m:
        f = int(m[1]); return k_sync_probe(f, f, (f,), wait=False), "diag", ()
    m = re.fullmatch(r"sf2_(\d+)_(\d+)", name)
    if m:
        a, b = int(m[1]), int(m[2])
        return k_sync_probe(a, a, (a, b), second=(b, b)), "diag", ()
    m = re.fullmatch(r"sfmask(\d+)_(\d+)", name)
    if m:
        f, b = int(m[1]), int(m[2]); return k_sync_probe(f, f, (f, b)), "diag", ()
    return None


# ==== GSRAM capacity ====
# Resolves 256 KiB (`dma.py`) vs 4 MiB (the GM control register / gm0_size).
# Note: reads before writes. A wild write can fault the SMMU; an out-of-range read only hangs
# the job. Writes only touch offsets a read has shown to be alive.
GSRAM_BASE = 0xF8000000
LSRAM_BASE = 0xFA000000
GS_WITNESS = 0xA5A50000          # marker written at offset 0
GS_STAMP   = 0xC0DE0000          # stamp at offset i is GS_STAMP | i


def k_gsram_read(offsets):
    """Read GSRAM at `base + off` for each offset, into OUT. args: r0=OUT, r2=VEC.

    Read-only. OUT is poisoned, so a hang is located by the end of the written prefix."""
    body = []
    for k, off in enumerate(offsets):
        body += _mov32("r6", (GSRAM_BASE + off) & 0xFFFFFFFF)
        body += [BUNDLE(I("ld r7, [r6+0]", E.ld("r7", "r6", 0))),
                 BUNDLE(I("st r7, [r0+%d]" % (4 * k), E.st("r7", "r0", 4 * k)))]
    return prologue(3, fp=False) + body + epilogue()


def k_gsram_stamp(offsets):
    """Write a distinct stamp (`GS_STAMP | k`) at each offset. args: r0=OUT (unused), r2=VEC.

    If the aperture wraps with period P, the last stamp written to a cell wins, so a later read
    reveals P. Read back in a separate job: no `cacheop` mode invalidates the d-cache, which is
    empty at task start."""
    body = []
    for k, off in enumerate(offsets):
        body += _mov32("r6", (GSRAM_BASE + off) & 0xFFFFFFFF)
        body += _mov32("r7", GS_STAMP | k)
        body += [BUNDLE(I("st r7, [r6+0]", E.st("r7", "r6", 0)))]
    return prologue(3, fp=False) + body + epilogue()


# Offset ladder: first within the known 256 KiB, then increasingly beyond it.
GS_OFFSETS = [0, 0x10000, 0x20000, 0x30000,            # 0, 64K, 128K, 192K -- known-good
              0x40000, 0x60000, 0x80000,               # 256K, 384K, 512K
              0x100000, 0x200000, 0x300000,            # 1M, 2M, 3M
              0x3F0000, 0x400000, 0x800000]            # ~4M, 4M, 8M


def k_gsram_rw(vector=False, off=0, write=False, lsram=False):
    """One cell at `base + off`, by scalar or vector access. args: r0=OUT, r2=VEC.

    Compares scalar `ld`/`st` against `vld`/`vst` at the same address (scalar GSRAM access is
    not known to work). `write=True` stores a stamp, otherwise reads it back; run write and read
    as separate jobs (no d-cache invalidate). `lsram=True` runs the same kernel on LSRAM, which
    persists across jobs, as a control."""
    body = _mov32("r6", ((LSRAM_BASE if lsram else GSRAM_BASE) + off) & 0xFFFFFFFF)
    if write:
        body += _mov32("r7", GS_STAMP | (off >> 12 & 0xFFF))
        if vector:
            # fill a vector register with 8 copies of the stamp, via OUT as scratch
            for i in range(8):
                body += [BUNDLE(I("st r7, [r0+%d]" % (4 * i), E.st("r7", "r0", 4 * i)))]
            body += [BUNDLE(I("ld t0, [r0+0]", E.vld("t0", "r0", 0))),
                     BUNDLE(I("st t0, [r6+0]", E.vst("t0", "r6", 0)))]
        else:
            body += [BUNDLE(I("st r7, [r6+0]", E.st("r7", "r6", 0)))]
    else:
        if vector:
            body += [BUNDLE(I("ld t0, [r6+0]", E.vld("t0", "r6", 0))),
                     BUNDLE(I("st t0, [r0+0]", E.vst("t0", "r0", 0)))]
        else:
            body += [BUNDLE(I("ld r7, [r6+0]", E.ld("r7", "r6", 0))),
                     BUNDLE(I("st r7, [r0+0]", E.st("r7", "r0", 0)))]
    return prologue(3, fp=False) + body + epilogue()


def k_gsram_roundtrip(vector=True, off=0):
    """Store a stamp into GSRAM and read it back in the same job. args: r0=OUT, r2=VEC.

    Fails if TEC stores to GSRAM do not land; passes if GSRAM merely does not persist across
    jobs. Note: a same-job read-back may be served by the d-cache, so a pass does not prove the
    bytes reached SRAM."""
    body = _mov32("r6", (GSRAM_BASE + off) & 0xFFFFFFFF)
    body += _mov32("r7", GS_STAMP | (off >> 12 & 0xFFF))
    if vector:
        for i in range(8):
            body += [BUNDLE(I("st r7, [r0+%d]" % (4 * i), E.st("r7", "r0", 4 * i)))]
        body += [BUNDLE(I("ld t0, [r0+0]", E.vld("t0", "r0", 0))),
                 BUNDLE(I("st t0, [r6+0]", E.vst("t0", "r6", 0))),
                 # scrub the scratch so the read-back cannot be the scratch itself
                 BUNDLE(I("st zero, [r0+0]", E.st("zero", "r0", 0)))]
        for i in range(1, 8):
            body += [BUNDLE(I("st zero, [r0+%d]" % (4 * i), E.st("zero", "r0", 4 * i)))]
        body += [BUNDLE(I("ld t1, [r6+0]", E.vld("t1", "r6", 0))),
                 BUNDLE(I("st t1, [r0+0]", E.vst("t1", "r0", 0)))]
    else:
        body += [BUNDLE(I("st r7, [r6+0]", E.st("r7", "r6", 0))),
                 BUNDLE(I("ld r8, [r6+0]", E.ld("r8", "r6", 0))),
                 BUNDLE(I("st r8, [r0+0]", E.st("r8", "r0", 0)))]
    return prologue(3, fp=False) + body + epilogue()


def k_gsram_reach_off(off=0, space="gsram"):
    """DMA-fill `base + off`, then `vld` 128 B back to OUT. args: r0=SRC, r1=DESC, r2=OUT.

    Capacity test that bypasses the d-cache: DMA writes, the TEC only reads. Inside the
    aperture OUT holds the DMA'd data. Note: reads past the end of GSRAM return drifting
    garbage, not a fault."""
    base = _gsram_base("r6") if space == "gsram" else _lsram_base("r6")
    body = base
    if off:
        body += _mov32("r8", off)
        body += [BUNDLE(I("add r6, r6, r8", E.add("r6", "r6", "r8")))]
    ib = E.DMA_SHARED if space == "gsram" else E.DMA_LSRAM
    body += [BUNDLE(I("dma 0, zero, 0, %d, 0, 1, 0, r1, r6, r0" % ib,
                      E.dma(E.DMA_DIRECT, "zero", 0, ib, E.DMA_GLOBAL, E.DMA_EXT2INT,
                            E.DMA_USELESS, "r1", "r6", "r0"))),
             BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))),
             BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    for i in range(4):
        body += [BUNDLE(I("ld t0, [r6+%d]" % (32 * i), E.vld("t0", "r6", 32 * i))),
                 BUNDLE(I("st t0, [r2+%d]" % (32 * i), E.vst("t0", "r2", 32 * i)))]
    return prologue(3, fp=False) + body + epilogue()


def gsram_arm(name):
    """`gsrN` reads the first N offsets of the ladder; `gswN` stamps them; `gsr1_<off>` reads one."""
    import re
    m = re.fullmatch(r"gsr(\d+)", name)
    if m:
        return k_gsram_read(GS_OFFSETS[:int(m[1])]), "diag", ()
    m = re.fullmatch(r"gsw(\d+)", name)
    if m:
        return k_gsram_stamp(GS_OFFSETS[:int(m[1])]), "diag", ()
    m = re.fullmatch(r"gsrt(s|v)_(\w+)", name)
    if m:
        return k_gsram_roundtrip(vector=(m[1] == "v"), off=int(m[2], 0)), "diag", ()
    m = re.fullmatch(r"rch_(gsram|lsram)_(\w+)", name)
    if m:
        return k_gsram_reach_off(off=int(m[2], 0), space=m[1]), "diag", ()
    m = re.fullmatch(r"reach_(vld|dma)_(gsram|lsram)", name)
    if m:                                 # known-good reachability arm, as a control
        return k_sram_reach(via=m[1], space=m[2]), "diag", ()
    m = re.fullmatch(r"(gs|ls)(s|v)(r|w)_(\w+)", name)
    if m:                                 # e.g. gssr_0 / lsvw_0: space, scalar|vector, read|write
        return (k_gsram_rw(vector=(m[2] == "v"), off=int(m[4], 0), write=(m[3] == "w"),
                           lsram=(m[1] == "ls")), "diag", ())
    m = re.fullmatch(r"gsr1_(\w+)", name)
    if m:
        return k_gsram_read([int(m[1], 0)]), "diag", ()
    return None


# ---- vector-integer probe ----
# `k_vec_probe(run)` loads eight vectors into t0..t7 and two GPRs (r9, r10), runs one named
# program, and stores all 32 vector registers back so every lane (including in-place source
# edits) can be checked against the numpy model.
#
# args: r0 = IN (8 x 32 B), r1 = OUT (32 x 32 B), r2 = two u32 for r9/r10.
def _vp(asm, op): return I(asm, op)


def _vcopy(td, ts):
    return _vp("or %s.w, %s.w, %s.w" % (td, ts, ts), E.vor(td, ts, ts))


VEC_PROBE_RUNS = {
    # dest register -> op(s) producing it; a two-op entry first copies a source into dest (in-place
    # forms / accumulator seed). Sources: t0 int8 ramp, t1 int8 pattern, t2/t3 int32, t4 int16,
    # t5 byte indices, t6 int32 accumulator seed, t7 bytes 0..31.
    "layout": [
        ("t8",  [_vp("zipl t8.b, t0.b, t7.b", E.zipl("t8", "t0", "t7"))]),
        ("t9",  [_vp("ziph t9.b, t0.b, t7.b", E.ziph("t9", "t0", "t7"))]),
        ("t10", [_vp("zipl t10.h, t0.h, t7.h", E.zipl("t10", "t0", "t7", "h"))]),
        ("t11", [_vp("ziph t11.h, t0.h, t7.h", E.ziph("t11", "t0", "t7", "h"))]),
        ("t12", [_vp("zipe t12.b, t0.b, t7.b", E.zipe("t12", "t0", "t7"))]),
        ("t13", [_vp("zipo t13.b, t0.b, t7.b", E.zipo("t13", "t0", "t7"))]),
        ("t14", [_vp("extl t14.b, t0.b, t7.b", E.extl("t14", "t0", "t7"))]),
        ("t15", [_vp("exth t15.b, t0.b, t7.b", E.exth("t15", "t0", "t7"))]),
        ("t16", [_vp("exte t16.b, t0.b, t7.b", E.exte("t16", "t0", "t7"))]),
        ("t17", [_vp("exto t17.b, t0.b, t7.b", E.exto("t17", "t0", "t7"))]),
        ("t18", [_vcopy("t18", "t6"), _vp("qdpa t18.w, t0.b, t1.b, p7.b", E.qdpa("t18", "t0", "t1"))]),
        ("t19", [_vp("qdot t19.w, t0.b, t1.b, p7.b", E.qdot("t19", "t0", "t1"))]),
        ("t20", [_vcopy("t20", "t4"), _vp("dpa t20.h, t0.b, t1.b, p7.b", E.dpa("t20", "t0", "t1"))]),
        ("t21", [_vp("dot t21.h, t0.b, t1.b, p7.b", E.dot("t21", "t0", "t1"))]),
        ("t22", [_vp("sxtl t22.h, t0.b", E.sxt("l", "hb", "t22", "t0"))]),
        ("t23", [_vp("sxth t23.h, t0.b", E.sxt("h", "hb", "t23", "t0"))]),
        ("t24", [_vp("sxtl t24.w, t4.h", E.sxt("l", "wh", "t24", "t4"))]),
        ("t25", [_vp("sxth t25.w, t4.h", E.sxt("h", "wh", "t25", "t4"))]),
        ("t26", [_vp("uxtl t26.h, t0.b", E.uxtl_hb("t26", "t0"))]),
        ("t27", [_vp("perm t27.b, t0.b, t5.b", E.perm("t27", "t0", "t5"))]),
        ("t28", [_vcopy("t28", "t0"), _vp("sld.l t28.b, t7.b, 3", E.sld("l", "t28", "t7", 3))]),
        ("t29", [_vcopy("t29", "t0"), _vp("sld.r t29.b, t7.b, 3", E.sld("r", "t29", "t7", 3))]),
        ("t30", [_vcopy("t30", "t0"), _vp("sld.dl t30.b, t7.b, 3", E.sld("dl", "t30", "t7", 3))]),
        ("t31", [_vcopy("t31", "t0"), _vp("sld.dr t31.b, t7.b, 3", E.sld("dr", "t31", "t7", 3))]),
    ],
    "arith": [
        ("t8",  [_vp("mul t8.w, t2.w, t3.w, p7.w", E.vmul("t8", "t2", "t3"))]),
        ("t9",  [_vp("mulh t9.w, t2.w, t3.w, p7.w", E.vmulh("t9", "t2", "t3"))]),
        ("t10", [_vp("mul t10.uw, t2.uw, t3.uw, p7.w", E.vmul("t10", "t2", "t3", unsigned=True))]),
        ("t11", [_vp("mulh t11.uw, t2.uw, t3.uw, p7.w", E.vmulh("t11", "t2", "t3", unsigned=True))]),
        ("t12", [_vp("add t12.w, t2.w, t3.w", E.vadd("t12", "t2", "t3"))]),
        ("t13", [_vp("sub t13.w, t2.w, t3.w", E.vsub("t13", "t2", "t3"))]),
        ("t14", [_vp("add t14.w, t2.w, 200", E.vaddi("t14", "t2", 200))]),
        ("t15", [_vp("sub t15.w, t2.w, 3", E.vsubi("t15", "t2", 3))]),
        ("t16", [_vp("max t16.w, t2.w, t3.w, p7.w", E.vmax("t16", "t2", "t3"))]),
        ("t17", [_vp("min t17.w, t2.w, t3.w, p7.w", E.vmin("t17", "t2", "t3"))]),
        ("t18", [_vp("min t18.uw, t2.uw, t3.uw, p7.w", E.vmin("t18", "t2", "t3", unsigned=True))]),
        ("t19", [_vp("max t19.w, t3.w, -3", E.vmaxi("t19", "t3", -3))]),
        ("t20", [_vp("min t20.uw, t3.uw, 255", E.vminui("t20", "t3", 255))]),
        ("t21", [_vp("lsl t21.w, t2.w, t3.w", E.vlsl("t21", "t2", "t3"))]),
        ("t22", [_vp("lsr t22.w, t2.w, t3.w", E.vlsr("t22", "t2", "t3"))]),
        ("t23", [_vp("asr t23.w, t2.w, t3.w", E.vasr("t23", "t2", "t3"))]),
        ("t24", [_vp("lsl t24.w, t2.w, 3", E.vlsli("t24", "t2", 3))]),
        ("t25", [_vp("lsr t25.w, t2.w, 3", E.vlsri("t25", "t2", 3))]),
        ("t26", [_vp("asr t26.w, t2.w, 3", E.vasri("t26", "t2", 3))]),
        ("t27", [_vp("lsrr t27.w, t2.w, 3", E.vlsrri("t27", "t2", 3))]),
        ("t28", [_vp("asrr t28.w, t2.w, 3", E.vasrri("t28", "t2", 3))]),
        ("t29", [_vp("andl t29.w, t2.w, 3", E.vandli("t29", "t2", 3))]),
        ("t30", [_vp("orl t30.w, t2.w, 3", E.vorli("t30", "t2", 3))]),
        ("t31", [_vp("xor t31.w, t2.w, t3.w", E.vxor("t31", "t2", "t3"))]),
    ],
    # Lane-sized min/max/shift/imm forms and cmp -> sel (which source a set predicate bit picks).
    "fp8": [
        ("t8",  [_vp("uxtl t8.h, t7.b", E.uxtl_hb("t8", "t7"))]),                       # 0..15 per halfword
        ("t9",  [_vp("mov4 t9, 4", E.mov4("t9", 4)), _vp("uxtl t9.h, t9.b", E.uxtl_hb("t9", "t9"))]),
        ("t10", [_vp("min t10.h, t8.h, t9.h, p7.h", E.vmin("t10", "t8", "t9", size="h"))]),
        ("t11", [_vp("max t11.h, t8.h, t9.h, p7.h", E.vmax("t11", "t8", "t9", size="h"))]),
        ("t12", [_vp("min t12.uh, t8.uh, t9.uh, p7.h", E.vmin("t12", "t8", "t9", unsigned=True, size="h"))]),
        ("t13", [_vp("min t13.h, t8.h, 4", E.vmini("t13", "t8", 4, size="h"))]),
        ("t14", [_vp("lsl t14.h, t8.h, 7", E.vlsli("t14", "t8", 7, size="h"))]),
        ("t15", [_vp("lsr t15.h, t8.h, 1", E.vlsri("t15", "t8", 1, size="h"))]),
        ("t16", [_vp("mov4 t16, 17", E.mov4("t16", 0x11))]),                            # 0x1111 per halfword
        ("t17", [_vp("mov4 t17, 34", E.mov4("t17", 0x22))]),                            # 0x2222
        ("t18", [_vp("mov4 t18, 0", E.mov4("t18", 0))]),
        ("t19", [_vp("cmp.eq p1.h, t8.h, t18.h, p7.h", E.vcmp("eq", "p1", "t8", "t18", size="h")),
                 _vp("sel t19.h, t16.h, t17.h, p1.h", E.sel("t19", "t16", "t17", "p1", size="h"))]),
        ("t20", [_vp("cmp.lt p2.h, t8.h, t9.h, p7.h", E.vcmp("lt", "p2", "t8", "t9", size="h")),
                 _vp("sel t20.h, t16.h, t17.h, p2.h", E.sel("t20", "t16", "t17", "p2", size="h"))]),
    ],
    "extra": [
        ("t8",  [_vp("exte t8.w, t2.w, t3.w", E.exte("t8", "t2", "t3", "w"))]),
        ("t9",  [_vp("exto t9.w, t2.w, t3.w", E.exto("t9", "t2", "t3", "w"))]),
        ("t10", [_vp("lsrr t10.w, t2.w, t6.w", E.vlsrr("t10", "t2", "t6"))]),      # shift amounts 10..80: >= 32 too
        ("t11", [_vp("asrr t11.w, t2.w, t6.w", E.vasrr("t11", "t2", "t6"))]),
        ("t12", [_vp("min t12.w, t3.w, -3", E.vmini("t12", "t3", -3))]),
        ("t13", [_vp("perm t13.b, t5.b, t0.b", E.perm("t13", "t5", "t0"))]),
        ("t14", [_vcopy("t14", "t6"), _vp("insert t14.w, r9, 0", E.insert("t14", "r9", 0, "w")), _vp("replic t14.w, t14.w, zero", E.replic("t14", "t14", "zero"))]),
        ("t15", [_vp("mov4 t15, 1", E.mov4("t15", 1)), _vp("lsr t15.w, t15.w, t5.w", E.vlsr("t15", "t15", "t5"))]),   # t5 words are big -> 0
        ("t16", [_vcopy("t16", "t2"), _vp("sld.l t16.b, t3.b, 4", E.sld("l", "t16", "t3", 4))]),
        ("t17", [_vp("lsrr t17.w, t2.w, t5.w", E.vlsrr("t17", "t2", "t5"))]),
        ("t18", [_vp("mov4 t18, 1", E.mov4("t18", 1)), _vp("and t18.w, t18.w, t2.w", E.vand("t18", "t18", "t2"))]),
        ("t19", [_vp("exte t19.h, t4.h, t4.h", E.exte("t19", "t4", "t4", "h"))]),
        ("t20", [_vp("lsl t20.w, t2.w, t6.w", E.vlsl("t20", "t2", "t6"))]),
        ("t21", [_vp("lsr t21.w, t2.w, t6.w", E.vlsr("t21", "t2", "t6"))]),
    ],
    # The byte-widening multiplies (enc.vmulb): which byte lanes `mul` / `mulh` take, and what the four sign mixes do; then the
    # E4M3 -> fp16 expand of k_gemm_gs (expand_v2) on t0's bytes as codes: 192 x c_signed - 64 x c_unsigned.
    "mulb": [
        ("t8",  [_vp("mul t8.h, t0.b, t1.b, p7.b", E.vmulb("t8", "t0", "t1"))]),
        ("t9",  [_vp("mulh t9.h, t0.b, t1.b, p7.b", E.vmulb("t9", "t0", "t1", high=True))]),
        ("t10", [_vp("mul t10.h, t0.b, t1.ub, p7.b", E.vmulb("t10", "t0", "t1", ub=True))]),
        ("t11", [_vp("mul t11.h, t0.ub, t1.b, p7.b", E.vmulb("t11", "t0", "t1", ua=True))]),
        ("t12", [_vp("mul t12.h, t0.ub, t1.ub, p7.b", E.vmulb("t12", "t0", "t1", ua=True, ub=True))]),
        ("t13", [_vp("mulh t13.h, t0.b, t1.ub, p7.b", E.vmulb("t13", "t0", "t1", ub=True, high=True))]),
        ("t14", [_vp("mulh t14.h, t0.ub, t1.b, p7.b", E.vmulb("t14", "t0", "t1", ua=True, high=True))]),
        ("t15", [_vp("mulh t15.h, t0.ub, t1.ub, p7.b", E.vmulb("t15", "t0", "t1", ua=True, ub=True, high=True))]),
        ("t16", [_vp("mov4 t16, -64", E.mov4("t16", -64))]),
        ("t17", [_vp("mov4 t17, 64", E.mov4("t17", 64))]),
        ("t18", [_vp("mul t18.h, t0.b, t16.ub, p7.b", E.vmulb("t18", "t0", "t16", ub=True))]),
        ("t19", [_vp("mul t19.h, t0.ub, t17.b, p7.b", E.vmulb("t19", "t0", "t17", ua=True))]),
        ("t20", [_vp("sub t20.h, t18.h, t19.h", E.vsub("t20", "t18", "t19", size="h"))]),
        ("t21", [_vp("mulh t21.h, t0.b, t16.ub, p7.b", E.vmulb("t21", "t0", "t16", ub=True, high=True))]),
        ("t22", [_vp("mulh t22.h, t0.ub, t17.b, p7.b", E.vmulb("t22", "t0", "t17", ua=True, high=True))]),
        ("t23", [_vp("sub t23.h, t21.h, t22.h", E.vsub("t23", "t21", "t22", size="h"))]),
        ("t24", [_vp("mul t24.w, t4.h, t4.h, p7.h", E.vmulb("t24", "t4", "t4", size="h"))]),
        ("t25", [_vp("mulh t25.w, t4.h, t4.h, p7.h", E.vmulb("t25", "t4", "t4", size="h", high=True))]),
    ],
    "narrow": [
        ("t8",  [_vcopy("t8", "t2"), _vp("nsr.as t8.bw, 3", E.nsr("as", "bw", "t8", 3))]),
        ("t9",  [_vcopy("t9", "t2"), _vp("nsr.asr t9.bw, 3", E.nsr("asr", "bw", "t9", 3))]),
        ("t10", [_vcopy("t10", "t2"), _vp("nsr.a t10.bw, 3", E.nsr("a", "bw", "t10", 3))]),
        ("t11", [_vcopy("t11", "t2"), _vp("nsr.l t11.bw, 3", E.nsr("l", "bw", "t11", 3))]),
        ("t12", [_vcopy("t12", "t2"), _vp("nsr.as t12.hw, 3", E.nsr("as", "hw", "t12", 3))]),
        ("t13", [_vcopy("t13", "t4"), _vp("nsr.as t13.bh, 0", E.nsr("as", "bh", "t13", 0))]),
        ("t14", [_vcopy("t14", "t3"), _vp("nsr.as t14.bw, 0", E.nsr("as", "bw", "t14", 0))]),
        ("t15", [_vcopy("t15", "t6"), _vp("nsr.as t15.bw, 0", E.nsr("as", "bw", "t15", 0))]),
        ("t16", [_vp("replic t16.w, t6.w, r9", E.replic("t16", "t6", "r9"))]),
        ("t17", [_vp("replic t17.b, t6.b, r9", E.replic("t17", "t6", "r9", "b"))]),
        ("t18", [_vp("replic t18.h, t6.h, r9", E.replic("t18", "t6", "r9", "h"))]),
        ("t19", [_vp("movf t19.w, -3, p7.w", E.movf("t19", -3))]),
        ("t20", [_vp("mov4 t20, 5", E.mov4("t20", 5))]),
        ("t21", [_vcopy("t21", "t6"), _vp("insert t21.w, r9, 3", E.insert("t21", "r9", 3, "w"))]),
        ("t22", [_vcopy("t22", "t7"), _vp("insert t22.b, r9, 5", E.insert("t22", "r9", 5, "b"))]),
        ("t23", [_vp("shfl t23.b, t7.b, 3", E.shfli("t23", "t7", 3))]),
        ("t24", [_vp("sel t24.w, t2.w, t3.w, p7.w", E.sel("t24", "t2", "t3"))]),
        ("t25", [_vp("rpadd t25.w, t2.w, p7.w", E.rpadd("t25", "t2"))]),
        ("t26", [_vp("and t26.w, t2.w, t3.w", E.vand("t26", "t2", "t3"))]),
        ("t27", [_vp("or t27.w, t2.w, t3.w", E.vor("t27", "t2", "t3"))]),
        ("t28", [_vcopy("t28", "t0"), _vp("sld.fr t28.b, t7.b, 3", E.sld("fr", "t28", "t7", 3))]),
        ("t29", [_vcopy("t29", "t0"), _vp("sld.dfr t29.b, t7.b, 3", E.sld("dfr", "t29", "t7", 3))]),
        ("t30", [_vp("replic t30.w, t6.w, r10", E.replic("t30", "t6", "r10"))]),
        ("t31", [_vp("lsl t31.w, t2.w, t30.w", E.vlsl("t31", "t2", "t30"))]),   # shift by a replicated GPR
    ],
}


def k_vec_probe(run):
    """Semantics probe for VEC_PROBE_RUNS[run]: each op alone in a bundle followed by three empty
    bundles, so latency/slot effects cannot affect the result."""
    body = [BUNDLE(I("ld t%d, [r0+%d]" % (i, 32 * i), E.vld("t%d" % i, "r0", 32 * i))) for i in range(8)]
    body += [BUNDLE(I("ld r9, [r2+0]", E.ld("r9", "r2", 0)), I("ld r10, [r2+4]", E.ld("r10", "r2", 4)))]
    body += [BUNDLE()] * 3
    for _dest, ops in VEC_PROBE_RUNS[run]:
        for op in ops:
            body += [BUNDLE(op), BUNDLE(), BUNDLE(), BUNDLE()]
    body += [BUNDLE(I("add r11, r1, 512", E.add("r11", "r1", 512)))]
    for i in range(32):
        b, off = ("r1", 32 * i) if i < 16 else ("r11", 32 * (i - 16))
        body += [BUNDLE(I("st t%d, [%s+%d]" % (i, b, off), E.vst("t%d" % i, b, off)))]
    return prologue(3) + body + epilogue()


# ---- vector-integer unit cost benchmarks ----
# Loop body of one op per bundle, round-robin over `nchain` RAW chains (nchain=8 throughput,
# nchain=1 latency); mixed arms put one op per class in a bundle. Timed as the slope between two
# loop counts; stored registers only prove the loop ran (timing only, no value check).
VEC_CHAIN = ["t%d" % i for i in range(8)]
VEC_CHAIN2 = ["t%d" % i for i in range(8, 16)]      # second op stream of a mixed arm: disjoint registers
VEC_CHAIN3 = ["t%d" % i for i in range(16, 24)]


def _vec_op(op, c, chain=None):
    x = (chain or VEC_CHAIN)[c]
    if op == "vadd": return I("add %s.w, %s.w, t31.w" % (x, x), E.vadd(x, x, "t31"))
    if op == "vadd_waw": return I("add t0.w, t30.w, t31.w", E.vadd("t0", "t30", "t31"))
    if op == "vsub": return I("sub %s.w, %s.w, t31.w" % (x, x), E.vsub(x, x, "t31"))
    if op == "vmul": return I("mul %s.w, %s.w, t30.w, p7.w" % (x, x), E.vmul(x, x, "t30"))
    if op == "vmulh": return I("mulh %s.w, %s.w, t30.w, p7.w" % (x, x), E.vmulh(x, x, "t30"))
    if op == "qdpa": return I("qdpa %s.w, t28.b, t31.b, p7.b" % x, E.qdpa(x, "t28", "t31"))
    if op == "qdot": return I("qdot %s.w, %s.b, t31.b, p7.b" % (x, x), E.qdot(x, x, "t31"))
    if op == "vminu": return I("min %s.uw, %s.uw, t30.uw, p7.w" % (x, x), E.vmin(x, x, "t30", unsigned=True))
    if op == "vmax0": return I("max %s.w, %s.w, 0" % (x, x), E.vmaxi(x, x, 0))
    if op == "zipl": return I("zipl %s.b, %s.b, t31.b" % (x, x), E.zipl(x, x, "t31"))
    if op == "ziph": return I("ziph %s.h, %s.h, t31.h" % (x, x), E.ziph(x, x, "t31", "h"))
    if op == "vlsl": return I("lsl %s.w, %s.w, t31.w" % (x, x), E.vlsl(x, x, "t31"))
    if op == "vlsri": return I("lsr %s.w, %s.w, 0" % (x, x), E.vlsri(x, x, 0))
    if op == "exte": return I("exte %s.w, %s.w, t31.w" % (x, x), E.exte(x, x, "t31", "w"))
    if op == "sld": return I("sld.l %s.b, t31.b, 0" % x, E.sld("l", x, "t31", 0))
    if op == "vand": return I("and %s.w, %s.w, t30.w" % (x, x), E.vand(x, x, "t30"))
    if op == "vxor": return I("xor %s.w, %s.w, t31.w" % (x, x), E.vxor(x, x, "t31"))
    if op == "vor": return I("or %s.w, %s.w, %s.w" % (x, x, x), E.vor(x, x, x))
    if op == "sxtl": return I("sxtl %s.h, %s.b" % (x, x), E.sxt("l", "hb", x, x))
    if op == "vld": return I("ld %s, [r0+%d]" % (x, 64 * (c & 1)), E.vld(x, "r0", 64 * (c & 1)))
    if op == "sadd": g = "r%d" % (11 + c); return I("add %s, %s, 1" % (g, g), E.add(g, g, 1))
    if op == "mfcyc": g = "r%d" % (11 + c); return I("mfctrl0 %s, 209" % g, E.mfctrl0(g, 0xd1))
    if op == "sld": g = "r%d" % (11 + c); return I("ld %s, [r0+%d]" % (g, 4 * c), E.ld(g, "r0", 4 * c))  # scalar DDR load, one line
    if op == "sldfar": g = "r%d" % (11 + c); return I("ld %s, [r0+%d]" % (g, 32 * c), E.ld(g, "r0", 32 * c))  # scalar DDR load, four lines
    if op == "sst": g = "r%d" % (11 + c); return I("st %s, [r2+%d]" % (g, 4 * c), E.st(g, "r2", 4 * c))  # scalar DDR store
    if op == "sldl": g = "r%d" % (11 + c); return I("ld %s, [r19+%d]" % (g, 4 * c), E.ld(g, "r19", 4 * c))  # scalar LSRAM load
    if op == "sstl": g = "r%d" % (11 + c); return I("st %s, [r19+%d]" % (g, 4 * c), E.st(g, "r19", 4 * c))
    if op == "vldd": return I("ld %s, [r0+%d]" % (x, 32 * (c & 3)), E.vld(x, "r0", 32 * (c & 3)))  # vector DDR load
    raise ValueError(op)


VEC_ARMS = {}
for _op in ("vadd", "vmul", "vmulh", "qdpa", "qdot", "vminu", "vmax0", "zipl", "ziph", "vlsl", "vlsri", "exte", "sld", "vand", "vxor", "sxtl", "vld"):
    VEC_ARMS["%s_n8" % _op] = ((_op,), 8)
    VEC_ARMS["%s_n1" % _op] = ((_op,), 1)
    VEC_ARMS["%s_n2" % _op] = ((_op,), 2)
    VEC_ARMS["%s_n4" % _op] = ((_op,), 4)
for _a, _b in (("vadd", "vadd"), ("qdpa", "qdpa"), ("vadd", "zipl"), ("qdpa", "zipl"), ("qdpa", "vlsl"), ("vadd", "vld"),
               ("zipl", "vld"), ("qdpa", "sadd"), ("zipl", "sadd"), ("vmul", "vmulh"), ("vadd", "exte"),
               ("vmul", "zipl"), ("vmul", "exte"), ("vmulh", "sld"), ("qdpa", "exte"), ("qdpa", "sld"), ("qdpa", "vand"),
               ("qdpa", "vld"), ("qdot", "zipl"), ("vminu", "zipl"), ("vmax0", "vlsri"), ("exte", "vld"), ("qdpa", "vxor")):
    VEC_ARMS["%s+%s" % (_a, _b)] = ((_a, _b), 8)
for _a, _b, _c in (("qdpa", "qdpa", "vld"), ("vadd", "vadd", "zipl"), ("qdpa", "qdpa", "zipl"), ("vmul", "vmulh", "exte")):
    VEC_ARMS["%s+%s+%s" % (_a, _b, _c)] = ((_a, _b, _c), 8)
VEC_ARMS["qdpa+qdpa+vld+vld"] = (("qdpa", "qdpa", "vld", "vld"), 8)
VEC_ARMS["vadd_waw"] = (("vadd_waw",), 8)  # WAW: same dest every bundle, independent source
VEC_ARMS["vld+vld"] = (("vld", "vld"), 8)
VEC_ARMS["mfcyc_n8"] = (("mfcyc",), 8)
for _op in ("sld", "sldfar", "sst", "sldl", "sstl", "vldd"):
    VEC_ARMS["%s_n8" % _op] = ((_op,), 8)
VEC_ARMS["sld+sst"] = (("sld", "sst"), 8)
for _n in (16, 32, 33, 34, 36, 40, 44, 48, 64, 80, 96, 128, 192, 256):  # loop-body length vs the per-TEC loop buffer
    VEC_ARMS["vadd_b%d" % _n] = (("vadd",), 8, _n)
    VEC_ARMS["mix_b%d" % _n] = (("vadd", "zipl", "vld"), 8, _n)


def k_vec_bench(ops, nchain, body_n=ALU_BODY):
    if ops and ops[0] == "le": return k_loop_entry(*ops[1:])
    if ops and ops[0] == "dmareq": return k_dma_req(*ops[1:])
    """args: r0 = operands (t28 bytes 0..31, t29 halves, t30 ones, t31 zeros: 4 x 32 B), r1 unused,
    r2 = OUT (8 x 32 B), r3 = LOOPS. `ops` is one op per bundle slot, each on its own DISJOINT chain
    set (t0-7, t8-15, t16-23); `body_n` bundles per iteration (the loop-body-length arms)."""
    body = [BUNDLE(I("ld t28, [r0+0]", E.vld("t28", "r0", 0)), I("ld t29, [r0+32]", E.vld("t29", "r0", 32))),
            BUNDLE(I("ld t30, [r0+64]", E.vld("t30", "r0", 64)), I("ld t31, [r0+96]", E.vld("t31", "r0", 96)))]
    body += [BUNDLE(I("ld t%d, [r0+0]" % i, E.vld("t%d" % i, "r0", 0)), I("ld t%d, [r0+64]" % (i + 1), E.vld("t%d" % (i + 1), "r0", 64)))
             for i in range(0, 24, 2)]
    body += [BUNDLE(I("mov r%d, %d" % (11 + c, c), E.mov("r%d" % (11 + c), c))) for c in range(8)]
    body += [BUNDLE(I("mov r19, 0", E.mov("r19", 0))), BUNDLE(I("movh r19, -1536", E.movh("r19", -1536)))]  # LSRAM base
    body += _alu_loop_open()
    k = 0
    chains = (VEC_CHAIN, VEC_CHAIN2, VEC_CHAIN3, VEC_CHAIN)
    for _ in range(body_n):
        c = k % nchain; k += 1
        b = [_vec_op(op, c, chains[j]) for j, op in enumerate(ops)]
        body.append(BUNDLE(*b))
    body += [BUNDLE(I("loopend", E.loopend()))]
    body += [BUNDLE(I("st t%d, [r2+%d]" % (i, 32 * i), E.vst("t%d" % i, "r2", 32 * i))) for i in range(8)]
    return prologue(4) + body + epilogue()


# Hardware-loop entry cost: `nloops` loops of `body_n` bundles x `inner` trips inside a cbnz outer
# loop of r3 trips. Slope over r3 = nloops x (inner x body_n + entry); inner=1 vs 8 isolates entry.
def k_loop_entry(body_n=32, inner=1, nloops=4):
    body = [BUNDLE(I("ld t28, [r0+0]", E.vld("t28", "r0", 0)), I("ld t31, [r0+96]", E.vld("t31", "r0", 96))),
            BUNDLE(I("mov r11, %d" % (inner - 1), E.mov("r11", inner - 1)))]
    outer = []
    for _ in range(nloops):
        outer += [BUNDLE(I("loop r11", E.loop("r11")))]
        outer += [BUNDLE(_vec_op("vadd", k % 8)) for k in range(body_n)]
        outer += [BUNDLE(I("loopend", E.loopend()))]
    outer += [BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1)))]
    outer += [BUNDLE(I("cbnz r3, .Louter", E.cbnz("r3", -len(outer))))]
    body += outer
    body += [BUNDLE(I("st t0, [r2+0]", E.vst("t0", "r2", 0)))]
    return prologue(4) + body + epilogue()


for _b in (8, 16, 32):
    for _i in (1, 8):
        VEC_ARMS["le_b%d_i%d" % (_b, _i)] = (("le", _b, _i), 1)
VEC_ARMS["le_b32_i1_n1"] = (("le", 32, 1, 1), 1)


# DMA request cost on the issuing TEC: `nreq` DDR (r0, descriptor at r0+256) -> LSRAM requests per
# r3 outer trip, each waited on flag 0. `wait=False` issues only (flags 0..3 round-robin, one wait
# at the end). NTASKS=4 contends four TECs on the core's single DMA engine.
def k_dma_req(nreq=8, wait=True):
    body = [BUNDLE(I("mov r19, 0", E.mov("r19", 0))), BUNDLE(I("movh r19, -1536", E.movh("r19", -1536))),
            BUNDLE(I("add r11, r0, 256", E.add("r11", "r0", 256))),  # descriptor
            BUNDLE(I("mov r12, 257", E.mov("r12", 257))), BUNDLE(I("mov r13, 514", E.mov("r13", 514))), BUNDLE(I("mov r14, 771", E.mov("r14", 771)))]
    outer = []
    for k in range(nreq):
        rs = ["zero", "r12", "r13", "r14"][k % 4] if not wait else "zero"
        outer += [BUNDLE(I("dma 0, %s, 0, 4, 0, 1, 0, r11, r19, r0" % rs,
                           E.dma(E.DMA_DIRECT, rs, 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT, E.DMA_USELESS, "r11", "r19", "r0")))]
        if wait: outer += [BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))), BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]
    if not wait: outer += [BUNDLE(I("wfe zero, 2", E.wfe("zero", 2)))]  # all outstanding
    outer += [BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1)))]
    outer += [BUNDLE(I("cbnz r3, .Louter", E.cbnz("r3", -len(outer))))]
    return prologue(4) + body + outer + epilogue()


VEC_ARMS["dmareq_w"] = (("dmareq", 8, True), 1)
VEC_ARMS["dmareq_nw"] = (("dmareq", 8, False), 1)


# ---- FP8 weights on the fp16 mma path ----
# E5M2 is fp16's top byte: `zipl td, tZERO, tPACKED` puts zero in the low byte of each halfword and
# the fp8 byte in the high byte (first source -> even bytes). E4M3 needs arithmetic (bias 7 -> 15,
# subnormal renormalisation, NaN), so it is a separate once-per-tile unpack kernel.
PACKEDB_STEPS = 64


def k_gemm_packedb(packed=True, steps=PACKEDB_STEPS):
    """3x4 register-block k-step with B loaded as fp16 (4 loads) or E5M2 (2 packed loads + 4 zips).
    Both arms share the same 16-bundle schedule and mma placement; only the B load path differs.

    args: r0=SRC (A panel at 0: `steps` x 96 B; B panel at 6144: `steps` x 128 B fp16 or x 64 B
          E5M2), r1=DESC (flag 0), r2=OUT (C, 12 pairs x 64 B, pre-zeroed), r3=REPS
    Registers: C t0..t23, A t24..t26, B t27..t30, t31 zero (all 32 used). Packed: P0 -> t28,
    P1 -> t30, zipl/ziph against t31 (ziph in place). Each mma is >= 2 bundles after its producers."""
    def A(i): return "t%d" % (24 + i)
    def Bv(j): return "t%d" % (27 + j)
    def MMA(i, j):
        c = 2 * (4 * i + j)
        return I("mma t%d.fp32, t%d.fp32, %s.fp16, %s.fp16" % (c, c + 1, A(i), Bv(j)),
                 E.mma("t%d" % c, A(i), Bv(j)))
    def VA(i): return I("ld %s, [r8+%d]" % (A(i), 32 * i), E.vld(A(i), "r8", 32 * i))
    def VB(j): return I("ld %s, [r13+%d]" % (Bv(j), 32 * j), E.vld(Bv(j), "r13", 32 * j))
    def VP(n, reg): return I("ld %s, [r13+%d]" % (reg, 32 * n), E.vld(reg, "r13", 32 * n))
    def ZL(td, tp): return I("zipl %s.b, t31.b, %s.b" % (td, tp), E.zipl(td, "t31", tp))
    def ZH(td, tp): return I("ziph %s.b, t31.b, %s.b" % (td, tp), E.ziph(td, "t31", tp))
    b_step = 64 if packed else 128
    if packed:
        head = [BUNDLE(VA(0), VP(0, "t28")),
                BUNDLE(VA(1), VP(1, "t30")),
                BUNDLE(ZL("t27", "t28"), VA(2)),  # B0; zip first: slot-2-only, ld takes slot 3
                BUNDLE(ZH("t28", "t28")),  # B1, in place
                BUNDLE(ZL("t29", "t30"), MMA(0, 0)),  # B2
                BUNDLE(ZH("t30", "t30"), MMA(1, 0))]  # B3, in place
    else:
        head = [BUNDLE(VA(0), VB(0)),
                BUNDLE(VA(1), VB(1)),
                BUNDLE(VA(2), VB(2)),
                BUNDLE(VB(3)),
                BUNDLE(MMA(0, 0)),
                BUNDLE(MMA(1, 0))]
    kstep = head + [
        BUNDLE(MMA(0, 1)),
        BUNDLE(MMA(1, 1)),
        BUNDLE(MMA(0, 2)),
        BUNDLE(MMA(1, 2)),
        BUNDLE(MMA(2, 0)),
        BUNDLE(MMA(2, 1), I("add r8, r8, 96", E.add("r8", "r8", 96))),
        BUNDLE(MMA(2, 2), I("add r13, r13, %d" % b_step, E.add("r13", "r13", b_step))),
        BUNDLE(MMA(0, 3)),
        BUNDLE(MMA(1, 3)),
        BUNDLE(MMA(2, 3)),
    ]
    assert len(kstep) == 16

    def cbase(t): return ("r2", t * 32) if t < 16 else ("r14", (t - 16) * 32)
    def cload(t):
        b, off = cbase(t); return I("ld t%d, [%s+%d]" % (t, b, off), E.vld("t%d" % t, b, off))
    def cstore(t):
        b, off = cbase(t); return I("st t%d, [%s+%d]" % (t, b, off), E.vst("t%d" % t, b, off))

    body = _lsram_base("r6") + _dma_fill_and_wait() + [
        BUNDLE(I("add r14, r2, 512", E.add("r14", "r2", 512))),
        BUNDLE(I("mov r15, %d" % GEMM_A_PANEL, E.mov("r15", GEMM_A_PANEL))),
        BUNDLE(I("mov4 t31, 0", E.mov4("t31", 0))),
    ]
    body += [BUNDLE(cload(t), cload(t + 1)) for t in range(0, 24, 2)]
    outer = [
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0)), I("add r13, r6, r15", E.add("r13", "r6", "r15")),
               I("mov r10, %d" % (steps - 1), E.mov("r10", steps - 1))),
        BUNDLE(I("loop r10", E.loop("r10"))),
    ] + kstep + [
        BUNDLE(I("loopend", E.loopend())),
        BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1))),
    ]
    outer.append(BUNDLE(I("cbnz r3, .Louter", E.cbnz("r3", -len(outer)))))
    body += outer
    body += [BUNDLE(cstore(t)) for t in range(24)]
    return prologue(4) + body + epilogue()


E4M3_LSRAM_OUT = 8192  # LSRAM: DMA fill in [0, 8192), unpacked output above


def k_e4m3_unpack(nvec=256, dst="ddr"):
    """E4M3 bytes -> fp16, exact (subnormals renormalised, signed NaN), 32 bytes per vector,
    on the integer SIMD side in 16-bit lanes.

    args: r0=SRC (`nvec` x 32 B E4M3, DMA-filled into LSRAM), r1=DESC (flag 0),
          r2=OUT (`nvec` x 64 B fp16, DDR), r3=REPS
    `nvec` is the inner hardware loop, fixed at build time so the LSRAM output window fits in
    32 KiB; r3 repeats it, so timing slopes are taken over r3.

    Per lane X: s = X & 0x80, e = (X >> 3) & 0xF, m = X & 7. normal = ((X & 0x7F) << 7) + 0x2000;
    subnormal = ((6 + k) << 10) | ((m << (10 - k)) & 0x3ff), k = floor(log2 m), 0 when m = 0;
    select subnormal where e = 0; NaN (X & 0x7f == 0x7f) -> 0x7E00; then | s << 8. Masks are
    0/0xFFFF lanes from `0 - bit`; k, "e != 0" and "not NaN" are bit folds (no lane-sized
    min/max/cmp used; see `k_e4m3_unpack_v2`). Constants t16..t31, working t0..t15.

    `dst` changes only where the 64 B per vector goes (same ops and bundles):
      "ddr"    stream to DDR, r2 advancing
      "hot"    r2 fixed: every store hits one L1 line (isolates DDR line traffic)
      "lsram"  stream to LSRAM; vector 0 copied to DDR after the loop (constant cost)"""
    # Constants: t16 zero, t17 0xFFFF, others one 8-bit value per h-lane (shifted where needed).
    Z, KFFFF = "t16", "t17"
    K = {0x80: "t18", 0x78: "t19", 0x07: "t20", 0x7F: "t21", 1: "t23", 2: "t24", 6: "t25",
         7: "t26", 8: "t27", 10: "t28", 3: "t31"}
    K2000, K3FF, K7E00 = "t22", "t29", "t30"

    def V(asm, op): return I(asm, op)
    def AND(td, ta, tb): return V("and %s.h, %s.h, %s.h" % (td, ta, tb), E.vand(td, ta, tb, "h"))
    def OR(td, ta, tb): return V("or %s.h, %s.h, %s.h" % (td, ta, tb), E.vor(td, ta, tb, "h"))
    def XOR(td, ta, tb): return V("xor %s.h, %s.h, %s.h" % (td, ta, tb), E.vxor(td, ta, tb, "h"))
    def LSL(td, ta, tb): return V("lsl %s.h, %s.h, %s.h" % (td, ta, tb), E.vlsl(td, ta, tb, "h"))
    def LSR(td, ta, tb): return V("lsr %s.h, %s.h, %s.h" % (td, ta, tb), E.vlsr(td, ta, tb, "h"))
    def ADD(td, ta, tb): return V("add %s.h, %s.h, %s.h" % (td, ta, tb), E.vadd(td, ta, tb, "h"))
    def SUB(td, ta, tb): return V("sub %s.h, %s.h, %s.h" % (td, ta, tb), E.vsub(td, ta, tb, "h"))
    def MOV4(td, imm): return V("mov4 %s, %d" % (td, imm), E.mov4(td, imm))
    def UXTL(td, ta): return V("uxtl %s.h, %s.b" % (td, ta), E.uxtl_hb(td, ta))
    def ZIPL(td, ta, tb): return V("zipl %s.b, %s.b, %s.b" % (td, ta, tb), E.zipl(td, ta, tb))
    def ZIPH(td, ta, tb): return V("ziph %s.b, %s.b, %s.b" % (td, ta, tb), E.ziph(td, ta, tb))

    def s8(v): return v - 256 if v >= 128 else v          # mov4's immediate is signed
    consts = [BUNDLE(MOV4(Z, 0)), BUNDLE(MOV4(KFFFF, -1))]
    for val, reg in K.items():                              # one 8-bit value per h-lane
        consts += [BUNDLE(MOV4("t0", s8(val))), BUNDLE(UXTL(reg, "t0"))]
    consts += [BUNDLE(MOV4("t0", 0x20)), BUNDLE(UXTL(K2000, "t0")), BUNDLE(LSL(K2000, K2000, K[8]))]
    consts += [BUNDLE(MOV4("t0", 0x7E)), BUNDLE(UXTL(K7E00, "t0")), BUNDLE(LSL(K7E00, K7E00, K[8]))]
    consts += [BUNDLE(MOV4("t0", -1)), BUNDLE(UXTL(K3FF, "t0")),     # 0x00FF
               BUNDLE(LSL(K3FF, K3FF, K[2])), BUNDLE(OR(K3FF, K3FF, K[3]))]   # 0x03FF

    def half(X, out):
        """16 lanes of X (zero-extended bytes) -> fp16 in `out`. Working: t2..t14."""
        S, Ee, M, N, a, b, k, zm, sh, mm, ex, es, tmp = ["t%d" % i for i in range(2, 15)]
        ops = [
            AND(S, X, K[0x80]), LSL(S, S, K[8]),                       # sign, to bit 15
            AND(Ee, X, K[0x78]),                                       # exp << 3
            AND(M, X, K[0x07]),                                        # mantissa
            AND(N, X, K[0x7F]), LSL(N, N, K[7]), ADD(N, N, K2000),     # normal path
            LSR(a, M, K[1]), LSR(b, M, K[2]),                          # a = m>>1, b = m>>2
            LSR(tmp, a, K[1]), OR(k, a, tmp), AND(k, k, K[1]), ADD(k, k, b),   # k = min(a,1) + b
            OR(zm, M, a), OR(zm, zm, b), AND(zm, zm, K[1]), SUB(zm, Z, zm),    # m != 0 mask
            SUB(sh, K[10], k),                                         # 10 - k
            LSL(mm, M, sh), AND(mm, mm, K3FF),                         # (m << (10-k)) & 0x3ff
            ADD(ex, K[6], k), LSL(ex, ex, K[10]),                      # (6 + k) << 10
            OR(ex, ex, mm), AND(ex, ex, zm),                           # subnormal, zero if m == 0
            LSR(tmp, Ee, K[2]), OR(es, Ee, tmp), LSR(tmp, es, K[1]), OR(es, es, tmp),
            LSR(es, es, K[3]), AND(es, es, K[1]), SUB(es, Z, es),      # e != 0 mask
            AND(N, N, es), XOR(tmp, es, KFFFF), AND(ex, ex, tmp), OR(out, N, ex),
            AND(tmp, X, K[0x7F]), XOR(tmp, tmp, K[0x7F]),              # 0 iff NaN
            LSR(a, tmp, K[2]), LSR(a, a, K[2]), OR(tmp, tmp, a),       # fold bits 0..6 into bit 0
            LSR(a, tmp, K[2]), OR(tmp, tmp, a), LSR(a, tmp, K[1]), OR(tmp, tmp, a),
            AND(tmp, tmp, K[1]), SUB(tmp, Z, tmp),                     # not-NaN mask
            AND(out, out, tmp), XOR(tmp, tmp, KFFFF), AND(tmp, tmp, K7E00), OR(out, out, tmp),
            OR(out, out, S),
        ]
        return [BUNDLE(o) for o in ops]

    loop = [BUNDLE(I("ld t0, [r8+0]", E.vld("t0", "r8", 0)))]
    loop += [BUNDLE(ZIPL("t1", "t0", Z))] + half("t1", "t15") + [BUNDLE(I("st t15, [r2+0]", E.vst("t15", "r2", 0)))]
    loop += [BUNDLE(ZIPH("t1", "t0", Z))] + half("t1", "t15") + [BUNDLE(I("st t15, [r2+32]", E.vst("t15", "r2", 32)))]
    bump = [I("add r8, r8, 32", E.add("r8", "r8", 32))]
    if dst != "hot": bump.append(I("add r2, r2, 64", E.add("r2", "r2", 64)))
    loop += [BUNDLE(*bump)]

    body = _lsram_base("r6") + _dma_fill_and_wait() + consts + [
        BUNDLE(I("add r5, r2, 0", E.add("r5", "r2", 0))),  # DDR destination, kept
        BUNDLE(I("mov r15, %d" % E4M3_LSRAM_OUT, E.mov("r15", E4M3_LSRAM_OUT))),
    ]
    # .Louter: re-arm pointers, inner hardware loop `nvec` times, repeat r3 times
    dest = [I("add r2, r6, r15", E.add("r2", "r6", "r15"))] if dst == "lsram" else \
           [I("add r2, r5, 0", E.add("r2", "r5", 0))]
    outer = [
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0)), *dest),
        BUNDLE(I("mov r10, %d" % (nvec - 1), E.mov("r10", nvec - 1))),
        BUNDLE(I("loop r10", E.loop("r10"))),
    ] + loop + [
        BUNDLE(I("loopend", E.loopend())),
        BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1))),
    ]
    outer.append(BUNDLE(I("cbnz r3, .Louter", E.cbnz("r3", -len(outer)))))
    body += outer
    if dst == "lsram":                       # constant cost, outside the slope
        body += [BUNDLE(I("add r7, r6, r15", E.add("r7", "r6", "r15"))),
                 BUNDLE(I("ld t0, [r7+0]", E.vld("t0", "r7", 0)), I("ld t1, [r7+32]", E.vld("t1", "r7", 32))),
                 BUNDLE(I("st t0, [r5+0]", E.vst("t0", "r5", 0))),
                 BUNDLE(I("st t1, [r5+32]", E.vst("t1", "r5", 32)))]
    return prologue(4) + body + epilogue()


# ---- the bundle scheduler: straight-line vector code packed under the slot and latency rules ----
# Slots: `qdpa`/`mul`/`add`/`min`/`max` take 0/1 (two per cycle); zips/shifts/logic/`exte` slot 2 only; loads 2/3, stores 2.
# Every op has a 2-cycle result latency (`qdpa` accumulates back to back); a `qdpa` beside a slot-2 op costs 2 cycles.
class Op(NamedTuple):
    asm: str
    enc: tuple
    dst: tuple
    src: tuple
    lat: int = -1          # -1: LAT["v"] at schedule time
    off: int = 0           # loads: the immediate offset, for the bank rule


# latencies: 2-cycle results, qdpa accumulates back to back, a qdpa never beside a slot-2 op; loads 3
LAT = {"ld": 3, "v": 2, "mul": 2, "dot": 1, "waw": 1, "dot_s2": 1, "mov": 0}   # mov: a slot-2-only op may displace a load to slot 3


def _sched(ops: list[Op]) -> list:
    """Greedy in-order-issue list scheduling of one straight-line block into bundles.

    An op lands in the earliest bundle where every source is ready, after the last read and write
    of every register it writes, with a slot of its class free, two loads only when they straddle a
    64 B bank (same base, offsets differing in bit 6), and never a `qdpa` beside a slot-2 vector op.
    Slots are taken in the first-free order `enc.bundle` re-derives, so the packed bytes match the schedule."""
    bundles: list[list] = []
    used: list[set] = []
    loads: dict = {}
    units: dict = {}
    movable: dict = {}     # bundle -> index of a load sitting in slot 2 while slot 3 is free: a slot-2-only op may displace it
    ready: dict = {}
    last_read: dict = {}
    last_write: dict = {}
    for op in ops:
        b = 0
        for r in op.src: b = max(b, ready.get(r, 0))
        for r in op.dst: b = max(b, last_read.get(r, -1) + 1, last_write.get(r, -1) + LAT["waw"])
        unit = "dot" if op.enc[0] == E.VEC_A and op.asm.startswith(("qdpa", "qdot")) else ("s2" if op.enc[0] == E.VEC_S else None)
        while True:
            while b >= len(bundles): bundles.append([]); used.append(set()); loads[len(bundles) - 1] = None; units[len(bundles) - 1] = set(); movable[len(bundles) - 1] = None
            free = [s for s in op.enc[0] if s not in used[b]]
            if not free and LAT["mov"] and op.enc[0] == E.VEC_S and movable[b] is not None and 3 not in used[b] and not (unit and LAT["dot_s2"] and "dot" in units[b]):
                # the load in slot 2 moves to slot 3 (enc.bundle packs first-free in order: this op goes before it)
                bundles[b].insert(movable[b], I(op.asm, op.enc)); used[b].add(3); movable[b] = None; units[b].add("s2")
                break
            if free and op.enc[0] == E.M and loads[b] is not None:
                ob, ooff = loads[b]
                if not (op.src and op.src[0] == ob and ((ooff ^ op.off) & 64)): free = []
            if free and unit and LAT["dot_s2"] and ({"dot", "s2"} - {unit}) & units[b]: free = []
            if free:
                used[b].add(free[0]); bundles[b].append(I(op.asm, op.enc))
                if op.enc[0] == E.M: loads[b] = (op.src[0], op.off)
                if op.enc[0] == E.M and free[0] == 2: movable[b] = len(bundles[b]) - 1
                elif free[0] == 3 or (op.enc[0] == E.VEC_S and free[0] == 2): movable[b] = None
                if unit: units[b].add(unit)
                break
            b += 1
        for r in op.src: last_read[r] = max(last_read.get(r, -1), b)
        for r in op.dst: last_write[r] = b; ready[r] = b + (LAT["v"] if op.lat < 0 else op.lat)
    return [BUNDLE(*b) for b in bundles]


# ---- E4M3 -> fp16, closed form ----
# `((x & 0x7f) << 7) + 0x2000` | sign is exact for all 238 normal codes; the 8 subnormals (e = 0)
# and 2 NaNs (0x7f, 0xff) are each fixed by a cmp + sel.
# Subnormal: sb = ((m << 10) >> k) + ((5 + k) << 10), k = floor(log2 m); the leading bit of
# m >> k supplies the 1024 the canonical form masks off (one op and constant fewer). m = 0 has
# its own select.
# Note: `sel td, ta, tb, p` takes `ta` where the predicate bit is SET.
E4M3_V2_CONSTS = ["t31", "t30", "t29", "t28", "t27", "t26", "t25"]      # Z K7F K7 K80 K78 K2000 K7E00
LAT_LD = 3  # vector ld latency, as LAT["ld"]


def _e4m3_v2_loop(mode, r_in, r_out, in_step=32, out_step=64):
    """Scheduled closed-form E4M3 -> fp16 loop body: 32 B at [r_in] -> 64 B at [r_out], cursors
    advanced. Scratch t0..t24; E4M3_V2_CONSTS (t25..t31) must be live. Used by `k_e4m3_unpack_v2` and `k_e4m3_stream`."""
    if mode not in ("exact", "ftz"): raise ValueError("mode %r" % mode)
    Z, K7F, K7, K80, K78, K2000, K7E00 = E4M3_V2_CONSTS

    def S(asm, enc, d, s): return Op(asm, enc, (d,), tuple(s))
    def AND(d, a, b): return S("and %s.h, %s.h, %s.h" % (d, a, b), E.vand(d, a, b, "h"), d, (a, b))
    def OR(d, a, b):  return S("or %s.h, %s.h, %s.h" % (d, a, b), E.vor(d, a, b, "h"), d, (a, b))
    def LSLI(d, a, i): return S("lsl %s.h, %s.h, %d" % (d, a, i), E.vlsli(d, a, i, size="h"), d, (a,))
    def LSRI(d, a, i): return S("lsr %s.h, %s.h, %d" % (d, a, i), E.vlsri(d, a, i, size="h"), d, (a,))
    def LSRR(d, a, b): return S("lsr %s.h, %s.h, %s.h" % (d, a, b), E.vlsr(d, a, b, "h"), d, (a, b))
    def ADD(d, a, b): return S("add %s.h, %s.h, %s.h" % (d, a, b), E.vadd(d, a, b, "h"), d, (a, b))
    def ADDI(d, a, i): return S("add %s.h, %s.h, %d" % (d, a, i), E.vaddi(d, a, i, size="h"), d, (a,))
    def MINI(d, a, i): return S("min %s.h, %s.h, %d" % (d, a, i), E.vmini(d, a, i, size="h"), d, (a,))
    def CMPEQ(p, a, b): return S("cmp.eq %s.h, %s.h, %s.h, p7.h" % (p, a, b), E.vcmp("eq", p, a, b, size="h"), p, (a, b))
    def SEL(d, a, b, p): return S("sel %s.h, %s.h, %s.h, %s.h" % (d, a, b, p), E.sel(d, a, b, p, size="h"), d, (a, b, p))
    def ZIPL(d, a, b): return S("zipl %s.b, %s.b, %s.b" % (d, a, b), E.zipl(d, a, b), d, (a, b))
    def ZIPH(d, a, b): return S("ziph %s.b, %s.b, %s.b" % (d, a, b), E.ziph(d, a, b), d, (a, b))

    def half(X, T, P, out):
        u, n, s, m, e8, a, b, w, se, sb, R = T
        pz, pe, pn = P
        ops = [AND(u, X, K7F), LSLI(n, u, 7), ADD(n, n, K2000),
               AND(s, X, K80), LSLI(s, s, 8)]
        if mode == "exact":
            ops += [AND(m, X, K7),
                    MINI(a, m, 4), LSRI(a, a, 2), MINI(b, m, 2), LSRI(b, b, 1), ADD(a, a, b),
                    LSLI(w, m, 10), LSRR(w, w, a),
                    ADDI(se, a, 5), LSLI(se, se, 10), ADD(sb, se, w),
                    CMPEQ(pz, m, Z), SEL(R, Z, sb, pz)]
        ops += [AND(e8, X, K78), CMPEQ(pe, e8, Z),
                SEL(R, R if mode == "exact" else Z, n, pe),
                CMPEQ(pn, u, K7F), SEL(R, K7E00, R, pn),
                OR(out, R, s)]
        return ops

    TA = ["t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12", "t13"]
    TB = ["t14", "t15", "t16", "t17", "t18", "t19", "t20", "t21", "t22", "t23", "t24"]
    ops = [Op("ld t0, [%s+0]" % r_in, E.vld("t0", r_in, 0), ("t0",), (r_in,), LAT_LD, 0),
           ZIPL("t1", "t0", Z), ZIPH("t2", "t0", Z)]
    ops += half("t1", TA, ("p1", "p2", "p3"), "t1")
    ops += half("t2", TB, ("p4", "p5", "p6"), "t2")
    ops += [Op("st t1, [%s+0]" % r_out, E.vst("t1", r_out, 0), (), ("t1", r_out)),
            Op("st t2, [%s+32]" % r_out, E.vst("t2", r_out, 32), (), ("t2", r_out)),
            Op("add %s, %s, %d" % (r_in, r_in, in_step), E.add(r_in, r_in, in_step), (r_in,), (r_in,))]
    if out_step: ops.append(Op("add %s, %s, %d" % (r_out, r_out, out_step), E.add(r_out, r_out, out_step), (r_out,), (r_out,)))
    return _sched(ops)


def _e4m3_v2_consts():
    """The seven constant registers of the closed form (t25..t31): 13 bundles."""
    Z, K7F, K7, K80, K78, K2000, K7E00 = E4M3_V2_CONSTS
    def MOV4(t, v): return BUNDLE(I("mov4 %s, %d" % (t, v), E.mov4(t, v)))
    def UXTL(t): return BUNDLE(I("uxtl %s.h, %s.b" % (t, t), E.uxtl_hb(t, t)))
    consts = [MOV4(Z, 0)]
    for t, v in ((K7F, 0x7F), (K7, 7), (K80, -128), (K78, 0x78)):
        consts += [MOV4(t, v), UXTL(t)]
    for t, v in ((K2000, 0x20), (K7E00, 0x7E)):
        consts += [MOV4(t, v), UXTL(t), BUNDLE(I("lsl %s.h, %s.h, 8" % (t, t), E.vlsli(t, t, 8, size="h")))]
    return consts


def k_e4m3_unpack_v2(nvec=256, dst="ddr", mode="exact"):
    """E4M3 bytes -> fp16 by the closed form, scheduled by `_sched`.

    args: as `k_e4m3_unpack` (r0=SRC, r1=DESC, r2=OUT, r3=REPS, `nvec` inner loop).
    mode="exact" matches `k_e4m3_unpack` bit for bit; mode="ftz" flushes the 8 subnormal codes to
    zero (deliberate; drops 13 of 24 ops per half, separate reference). The two halves are
    independent chains so the vector unit issues one bundle per cycle instead of every two."""
    if mode not in ("exact", "ftz"): raise ValueError("mode %r" % mode)
    Z, K7F, K7, K80, K78, K2000, K7E00 = E4M3_V2_CONSTS

    def S(asm, enc, d, s): return Op(asm, enc, (d,), tuple(s))
    def AND(d, a, b): return S("and %s.h, %s.h, %s.h" % (d, a, b), E.vand(d, a, b, "h"), d, (a, b))
    def OR(d, a, b):  return S("or %s.h, %s.h, %s.h" % (d, a, b), E.vor(d, a, b, "h"), d, (a, b))
    def LSLI(d, a, i): return S("lsl %s.h, %s.h, %d" % (d, a, i), E.vlsli(d, a, i, size="h"), d, (a,))
    def LSRI(d, a, i): return S("lsr %s.h, %s.h, %d" % (d, a, i), E.vlsri(d, a, i, size="h"), d, (a,))
    def LSRR(d, a, b): return S("lsr %s.h, %s.h, %s.h" % (d, a, b), E.vlsr(d, a, b, "h"), d, (a, b))
    def ADD(d, a, b): return S("add %s.h, %s.h, %s.h" % (d, a, b), E.vadd(d, a, b, "h"), d, (a, b))
    def ADDI(d, a, i): return S("add %s.h, %s.h, %d" % (d, a, i), E.vaddi(d, a, i, size="h"), d, (a,))
    def MINI(d, a, i): return S("min %s.h, %s.h, %d" % (d, a, i), E.vmini(d, a, i, size="h"), d, (a,))
    def CMPEQ(p, a, b): return S("cmp.eq %s.h, %s.h, %s.h, p7.h" % (p, a, b), E.vcmp("eq", p, a, b, size="h"), p, (a, b))
    def SEL(d, a, b, p): return S("sel %s.h, %s.h, %s.h, %s.h" % (d, a, b, p), E.sel(d, a, b, p, size="h"), d, (a, b, p))
    def ZIPL(d, a, b): return S("zipl %s.b, %s.b, %s.b" % (d, a, b), E.zipl(d, a, b), d, (a, b))
    def ZIPH(d, a, b): return S("ziph %s.b, %s.b, %s.b" % (d, a, b), E.ziph(d, a, b), d, (a, b))

    def half(X, T, P, out):
        """One 16-lane half. `T` is 11 scratch registers, `P` three predicates."""
        u, n, s, m, e8, a, b, w, se, sb, R = T
        pz, pe, pn = P
        ops = [AND(u, X, K7F), LSLI(n, u, 7), ADD(n, n, K2000),
               AND(s, X, K80), LSLI(s, s, 8)]
        if mode == "exact":
            ops += [AND(m, X, K7),
                    MINI(a, m, 4), LSRI(a, a, 2), MINI(b, m, 2), LSRI(b, b, 1), ADD(a, a, b),
                    LSLI(w, m, 10), LSRR(w, w, a),
                    ADDI(se, a, 5), LSLI(se, se, 10), ADD(sb, se, w),
                    CMPEQ(pz, m, Z), SEL(R, Z, sb, pz)]
        ops += [AND(e8, X, K78), CMPEQ(pe, e8, Z),
                SEL(R, R if mode == "exact" else Z, n, pe),
                CMPEQ(pn, u, K7F), SEL(R, K7E00, R, pn),
                OR(out, R, s)]
        return ops

    TA = ["t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12", "t13"]
    TB = ["t14", "t15", "t16", "t17", "t18", "t19", "t20", "t21", "t22", "t23", "t24"]
    body_ops = [Op("ld t0, [r8+0]", E.vld("t0", "r8", 0), ("t0",), ("r8",), LAT_LD, 0),
                ZIPL("t1", "t0", Z), ZIPH("t2", "t0", Z)]
    body_ops += half("t1", TA, ("p1", "p2", "p3"), "t1")
    body_ops += half("t2", TB, ("p4", "p5", "p6"), "t2")
    body_ops += [Op("st t1, [r2+0]", E.vst("t1", "r2", 0), (), ("t1", "r2")),
                 Op("st t2, [r2+32]", E.vst("t2", "r2", 32), (), ("t2", "r2")),
                 Op("add r8, r8, 32", E.add("r8", "r8", 32), ("r8",), ("r8",))]
    if dst != "hot":
        body_ops.append(Op("add r2, r2, 64", E.add("r2", "r2", 64), ("r2",), ("r2",)))
    loop = _sched(body_ops)

    def MOV4(t, v): return BUNDLE(I("mov4 %s, %d" % (t, v), E.mov4(t, v)))
    def UXTL(t): return BUNDLE(I("uxtl %s.h, %s.b" % (t, t), E.uxtl_hb(t, t)))
    consts = [MOV4(Z, 0)]
    for t, v in ((K7F, 0x7F), (K7, 7), (K80, -128), (K78, 0x78)):
        consts += [MOV4(t, v), UXTL(t)]
    for t, v in ((K2000, 0x20), (K7E00, 0x7E)):
        consts += [MOV4(t, v), UXTL(t),
                   BUNDLE(I("lsl %s.h, %s.h, 8" % (t, t), E.vlsli(t, t, 8, size="h")))]

    body = _lsram_base("r6") + _dma_fill_and_wait() + consts + [
        BUNDLE(I("add r5, r2, 0", E.add("r5", "r2", 0))),
        BUNDLE(I("mov r15, %d" % E4M3_LSRAM_OUT, E.mov("r15", E4M3_LSRAM_OUT))),
    ]
    dest = [I("add r2, r6, r15", E.add("r2", "r6", "r15"))] if dst == "lsram" else \
           [I("add r2, r5, 0", E.add("r2", "r5", 0))]
    outer = [
        BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0)), *dest),
        BUNDLE(I("mov r10, %d" % (nvec - 1), E.mov("r10", nvec - 1))),
        BUNDLE(I("loop r10", E.loop("r10"))),
    ] + loop + [
        BUNDLE(I("loopend", E.loopend())),
        BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1))),
    ]
    outer.append(BUNDLE(I("cbnz r3, .Louter", E.cbnz("r3", -len(outer)))))
    body += outer
    if dst == "lsram":
        body += [BUNDLE(I("add r7, r6, r15", E.add("r7", "r6", "r15"))),
                 BUNDLE(I("ld t0, [r7+0]", E.vld("t0", "r7", 0)), I("ld t1, [r7+32]", E.vld("t1", "r7", 32))),
                 BUNDLE(I("st t0, [r5+0]", E.vst("t0", "r5", 0))),
                 BUNDLE(I("st t1, [r5+32]", E.vst("t1", "r5", 32)))]
    return prologue(4) + body + epilogue()


# ---- 2D-blocked fp16 GEMM: B panels of `ns` strips in LSRAM, A row blocks reused across strips, C in GSRAM ----
E4M3_STREAM2_CHUNK = 4096           # two 4 KiB fills + two 8 KiB drains = 24 KiB of LSRAM


def k_e4m3_stream2(mode="exact", chunk=E4M3_STREAM2_CHUNK):
    """Double-buffered `k_e4m3_stream`: while chunk i unpacks (slot s = i & 1), chunk i + 1 fills the
    other slot (flag s ^ 1) and chunk i - 1 drains from the other out slot (flag 2 + (s ^ 1)), hiding
    the DMA under the unpack. Per-slot registers (r15/r16 fill sync word / wait mask, r17/r18 drain
    sync / mask, r20 descriptor offset, r21 in offset, r22 out offset) flip by xor every chunk.
    args: r0 = SRC, r1 = DESC (flag f's descriptor at +64 f: fills of `chunk` bytes at +0 / +64,
    drains of 2 chunk at +128 / +192), r2 = OUT, r3 = NCHUNKS (0: exit). LSRAM: in slots at 0 and
    `chunk`, out slots at 2 chunk and 4 chunk."""
    C = chunk
    def X(rd, rs, imm):   # `xor` is register-form only: the immediate goes through r7
        return _mov32("r7", imm) + [BUNDLE(I("xor %s, %s, r7" % (rd, rs), E.xor(rd, rs, "r7")))]
    body = [BUNDLE(I("cbnz r3, .Lgo", E.cbnz("r3", 2))), BUNDLE(I("b .Ldone", E.b(0)))]
    skip = 1
    body += _lsram_base("r6") + _e4m3_v2_consts()
    body += [BUNDLE(I("mov r15, 0", E.mov("r15", 0))), BUNDLE(I("sub r16, zero, 2", E.sub("r16", "zero", 2))),          # slot 0: flag 0 fill, flag 2 drain
             BUNDLE(I("mov r17, 514", E.mov("r17", 514))), BUNDLE(I("sub r18, zero, 5", E.sub("r18", "zero", 5))),
             BUNDLE(I("mov r20, 0", E.mov("r20", 0))), BUNDLE(I("mov r21, 0", E.mov("r21", 0)))] + _mov32("r22", 2 * C)
    body += _mov32("r11", 2 * C) + _mov32("r12", C)
    # first fill (slot 0)
    body += [BUNDLE(I("dma 0, zero, 0, 4, 0, 1, 0, r1, r6, r0", E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT, E.DMA_USELESS, "r1", "r6", "r0"))),
             BUNDLE(I("add r0, r0, r12", E.add("r0", "r0", "r12")))]
    chunk_ = [BUNDLE(I("wfe r16, 1", E.wfe("r16", 1)))]  # this slot's fill landed
    # prefetch the next chunk into the other slot (if any)
    pre = [*X("r23", "r15", 257), *X("r24", "r20", 64), BUNDLE(I("add r24, r1, r24", E.add("r24", "r1", "r24")))]
    pre += X("r25", "r21", C) + [BUNDLE(I("add r25, r6, r25", E.add("r25", "r6", "r25"))),
            BUNDLE(I("dma 0, r23, 0, 4, 0, 1, 0, r24, r25, r0", E.dma(E.DMA_DIRECT, "r23", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT, E.DMA_USELESS, "r24", "r25", "r0"))),
            BUNDLE(I("add r0, r0, r12", E.add("r0", "r0", "r12")))]
    chunk_ += [BUNDLE(I("sub r9, r3, 1", E.sub("r9", "r3", 1))), BUNDLE(I("cbnz r9, .Lpre", E.cbnz("r9", 2))), BUNDLE(I("b .Lnopre", E.b(len(pre) + 1)))] + pre
    chunk_ += [BUNDLE(I("wfe r18, 1", E.wfe("r18", 1))),  # this out slot's previous drain done
               BUNDLE(I("add r8, r6, r21", E.add("r8", "r6", "r21"))), BUNDLE(I("add r13, r6, r22", E.add("r13", "r6", "r22"))), BUNDLE(I("add r14, r13, 0", E.add("r14", "r13", 0))),
               BUNDLE(I("mov r10, %d" % (C // 32 - 1), E.mov("r10", C // 32 - 1))), BUNDLE(I("loop r10", E.loop("r10")))]
    chunk_ += _e4m3_v2_loop(mode, "r8", "r13") + [BUNDLE(I("loopend", E.loopend()))]
    chunk_ += [BUNDLE(I("add r24, r1, r20", E.add("r24", "r1", "r20"))), BUNDLE(I("add r24, r24, 128", E.add("r24", "r24", 128))),
               BUNDLE(I("dma 0, r17, 0, 4, 0, 0, 0, r24, r14, r2", E.dma(E.DMA_DIRECT, "r17", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_INT2EXT, E.DMA_USELESS, "r24", "r14", "r2"))),
               BUNDLE(I("add r2, r2, r11", E.add("r2", "r2", "r11"))),
               *X("r15", "r15", 257), *X("r16", "r16", 3),
               *X("r17", "r17", 257), *X("r18", "r18", 12),
               *X("r20", "r20", 64)] + X("r21", "r21", C) + X("r22", "r22", 6 * C)
    chunk_ += [BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1)))]
    chunk_.append(BUNDLE(I("cbnz r3, .Lchunk", E.cbnz("r3", -len(chunk_)))))
    body += chunk_ + [BUNDLE(I("sub r7, zero, 13", E.sub("r7", "zero", 13))), BUNDLE(I("wfe r7, 1", E.wfe("r7", 1)))]   # both drains
    body[skip] = BUNDLE(I("b .Ldone", E.b(len(body) - skip)))
    return prologue(4) + body + epilogue()


def k_gemm_gs(ks=24, ns=6, nrb=14, stamps=False, c_stride=None, drain="tec", a_rb_stride=None, c_rb_stride=None, rowmax=False, timing=None, b8=False, kw=3, bscale=False,
              rows=None, poison=False, q8=False, q8f=False, scales="dup", tern=False, tkloop="orig", tscale="f32", tkloop2="orig"):
    """C[rb, s] += A[rb] @ B[s] over K-slices, fp16 in, fp32 out, for `nrb` 12-row blocks and `ns`
    16-column strips: A is reused across the strips instead of re-read per strip.

    Per K-slice of `ks` k-steps: the strips' fp16 B panels (`gemm_fp16.pack_b_group`:
    [slice][s][k][4 tiles][32 B], ns*ks*128 B) land in LSRAM by one DMA (flag 0); each row block's
    A slice (`pack_a_slices`: [slice][rb][k][3 tiles][32 B], ks*96 B) streams through two LSRAM
    halves by parity (flags 1 / 2) and runs the 3x4 body against every panel, its C block (768 B)
    loaded from / stored to GSRAM around each k-loop (cost amortised by `ks`). The GSRAM C region
    ([rb][s][768 B]) is zeroed at the start and drained to DDR by the TEC at the end.

    args: r0 = A (first row block), r1 = B groups (contiguous), r2 = C out (NGROUPS x nrb x ns x 768 B),
          r3 = DESC (A row-block slice +0, B group slice +32), r4 = NSLICES, r5 = NRB (even),
          r6 = A_SLICE_BYTES (all row blocks x ks x 96), r7 = GSRAM C base (0xF8000000 + 64 KiB x TEC
          index in core; nrb*ns*768 <= 64 KiB), r8 = NGROUPS (C zeroed, filled, drained per group),
          r9 = STAMP when `stamps` (16 B: total, DMA waits, k-loops, C loads+stores).
    `c_stride`: C out advance per group (default nrb x ns x 768); a task doing part of the row blocks
    passes the full-M stride so its blocks land in the [group][rb][s] layout."""
    # `kw`: k-steps per loop body (the operand pointers advance once per body); ks % kw == 0.
    # `bscale` (b8, drain="dma"): B's E4M3 codes carry a scale per (column, K-slice) -- a checkpoint quantised in
    # 128 x 128 blocks with ks = 32 -- so each K-slice's product is accumulated from zero, multiplied by the
    # slice's per-column fp32 scales (SCLB bytes after the slice's codes in the B stream: [s][j][8] = strip s,
    # tile j, [c0 c1 c2 c3 c0 c1 c2 c3], x 2^8 folded in) and added to the C block from GSRAM. A slice's codes and scales
    # arrive in ONE request (DESC +128, flag 0) into a codes+scales buffer (CB), expanded from there into the fp16 panels:
    # a TEC has four sync flags (0-3; a selector of 4-30 silently lands on flag 0, and a wait on a flag nobody raised returns
    # at once), and every flag already serves (0 B, 1 / 2 A halves, 3 drain). C comes out fully scaled.
    # `poison` (diagnostic): NaN over the scale table right before the request that fills it, so a read that beats the
    # data shows as NaN in C.
    # `q8` (with b8 + bscale): GGUF Q8_0 (int8 q, a scale d per 32 weights: ks = 8, a slice is one scale block). The packer
    # stores u = q + 128 (unsigned), so the expand is one zip: the byte in a 16-bit lane's low half IS the fp16 subnormal
    # u x 2^-24 (exact). The +128 comes back out per slice: a pre-pass over the A slice, `mma` against a constant tile of
    # 2^-17 (= 128 x 2^-24), gives each row's sum S' already in the C tiles' layout (one pair a row tile, LSRAM scratch);
    # after a strip's k-loop C -= S' (identical products in the same order: an all-zero block gives exactly 0), then x d.
    # The scale table is 16 fp16 a strip (d x 2^10: every non-zero fp16 d normal), widened in registers to fp32 d x 2^24.
    # `q8f` (with b8 + bscale): the same Q8_0 weights, each built in fp16 by the expand -- u = q + 128 zipped under a 0x64 byte is
    # the fp16 1024 + u, minus 1152 is q exactly, times the column's block scale d is the weight (one rounding, as fp16 weights) --
    # so a slice spans ks / 8 scale blocks and needs no correction or scale pass: the stream (`pack_b_group_q8f`) carries per slice
    # the codes, then [s][block][16 cols] fp16 d (32 B a strip and block).
    # The E4M3 -> fp16 expand (b8, not q8 / q8f, which have their own): the byte-widening multiplies (`expand_v2`: 16 bundles
    # per 128 codes, no slot-2 op but the stores). The 48-deep 3-strip kernel (`lin48`: ks 48, ns 3) keeps the zip / asr / and
    # expand (`expand`, 36 bundles per 128 codes): the widening-multiply expand has not been validated on that geometry.
    # `scales` (bscale, E4M3): the scale table's layout in the stream. "dup" = `pack_b_group_bscale`'s [s][j][c0 c1 c2 c3 c0 c1
    # c2 c3] fp32 (128 B a strip): each vector is loaded and multiplied as is. "single" = each scale once, [s][j][c0 c1 c2 c3]
    # (64 B a strip, `pack_b_group_bscale(scales="single")`): one 32-B load carries tiles j and j + 1, and `extl` / `exth` of
    # the register with itself rebuild the duplicated lanes (the q8 path's widening below) -- the same fp32 values in the same
    # multiplies, so C is bit-identical; a slice's request shrinks by ns x 64 B (6528 -> 6336 B at ns 3). The buffers follow SCLB.
    # `tern` (b8 + bscale + scales="single"): ternary g128 weights {-1, 0, +1}, a scale per (column, 128-wide K-slice), as
    # 2-bit codes c = w + 1 (`gemm_fp16.pack_b_group_tern`): per (strip, k-step pair) 32 B, byte 16 h + l (h the k-step, l the
    # tile lane 4 n + k) holding tile j's code in bits 7-2j:6-2j. `expand_tern` makes Y_j = 4^j X mod 256 (byte `add`s, which
    # wrap) and widens it (byte `mul` / `mulh` x 1, slots 0/1) into tile j's lanes as the fp16 subnormal u(Y_j) x 2^-24 =
    # (64 c_j + 16 c_(j+1) + 4 c_(j+2) + c_(j+3)) x 2^-24: no shift or mask, the stores are the only slot-2 ops. The mixing is
    # across the strip's COLUMN tiles, so after the k-loop C(j) -= C(j+1) / 4 (j = 0, 1, 2, in that order) leaves C(j) =
    # 64 x 2^-24 A c_j; the offset (c = w + 1) is q8's pre-pass (`sprep`, constant 64 x 2^-24) and C -= S'; the single-layout
    # table carries s x 2^18. Exact before the scale multiply (integer lanes, power-of-two steps). Rows 1-8 (rt <= 2) take the
    # fast path TF below: the expand fused into the k-loop (codes -> B tiles in registers, no fp16 panels: `strips_tf`), C
    # accumulated in LSRAM and scale-and-accumulate fused (acc = fp32(acc + C x s), one rounding a slice after the first); rows 9-12 and the prefill layout keep C in GSRAM with a separate multiply and add. (A duplicated
    # scale table would save the extl / exth, but they ride in the epilogue's slot-2 gaps: the same bundle count, so the
    # single layout -- ns x 64 B fewer a slice -- stays the only one.)
    # `tkloop` (tern TF, rt 1): "orig" = the k-loop below as it was (mma and expand bundles alternating:
    # 8 bundles a pair, every expand bundle the first vector bundle after an mma bundle); "mmfirst" = a pair's four mma
    # bundles first, then its expand bundles (9 a pair: the first after the mmas costs 3 cycles on the board, ALU.md 7,
    # the rest 1) -- one mma -> vector transition a pair instead of four. See strips_tf.
    # `tscale` (tern): the per (column, slice) scale table's format. "f32" = the single-layout fp32 table above
    # (64 B a strip). "f16" = 16 fp16 a strip (32 B: 2.125 bits a weight instead of 2.25), lane 2i = column i (tiles 0-1's
    # columns), lane 2i + 1 = column 8 + i (tiles 2-3's), the value s x 2^15 (`gemm_fp16.pack_b_group_tern(tscale="f16")`: a
    # normal fp16 for every fp16 group scale d down to the subnormals); widened by fmae / fmao (fp16 x fp16 -> fp32, exact)
    # against fp16 2^3, so the fp32 lanes hold s x 2^15 x 2^3 = s x 2^18 -- the f32
    # table's values exactly -- then the same extl / exth: C is bit-identical. Every tern path: TF (rows 1-8, both tkloops) in
    # the scheduled epilogue (the constant pair in r12: `ifz` clobbers r7), rows 9-12 and the prefill layout in the strip epilogue.
    # `tkloop2` (tern TF, rt 2 = rows 5-8): "orig" = the k-loop below as it was (the earlier order: 8 mma and 4 expand
    # bundles a pair interleaved, every expand bundle the first vector bundle after an mma bundle, each tile rewritten right after
    # its last read); "m8e5" / "m8e4" = a pair's eight mma bundles first, then its expand bundles (one mma -> vector transition a
    # pair instead of four). See strips_tf.
    assert tkloop in ("orig", "mmfirst"), tkloop
    assert tkloop2 in ("orig", "m8e5", "m8e4", "m8u", "m8w"), tkloop2
    assert tscale in ("f32", "f16") and (tscale == "f32" or tern), tscale
    F16S = tscale == "f16"
    assert scales in ("dup", "single"), scales
    SSC = scales == "single"
    assert not tern or (b8 and bscale and SSC and not q8 and not q8f and ks % 2 == 0 and kw == 4), "k_gemm_gs(tern): b8 + bscale + scales='single'"
    SP = q8 or tern                                # the A row-sum pre-pass and C -= S' (q8: the +128 offset; tern: c = w + 1)
    assert not SSC or (bscale and not q8 and not q8f), "k_gemm_gs(scales='single'): the E4M3 block-scale stream (bscale, not q8 / q8f)"
    V2 = b8 and not q8 and not q8f and not (ks == 48 and ns == 3)
    assert not q8 or (b8 and bscale), "k_gemm_gs(q8): b8 + bscale"
    assert not q8f or (b8 and bscale and not q8 and ks % 8 == 0), "k_gemm_gs(q8f): b8 + bscale, ks a multiple of 8"
    assert ks % kw == 0 and nrb % 2 == 0, (ks, nrb, ns, kw)
    GSW = GsramWindow(f"k_gemm_gs(nrb={nrb}, ns={ns})")     # the task's 64 KiB window: C blocks [rb][s][768 B]
    GSW.alloc("C blocks", nrb * ns * 768)
    assert not bscale or (b8 and drain == "dma" and not rowmax and not stamps), "k_gemm_gs(bscale): b8, drain=dma"
    # `rows` (decode: few real rows): only row block 0's first `rows` rows are computed -- rt = ceil(rows / 4) row tiles, one
    # row block, no row-block pairing. The layouts are unchanged (A over nrb blocks, C over nrb blocks); C rows past the rt
    # tiles are drained as zeros, the other row blocks' C is never written. For rt <= 2 the k-step is 2 rt `mma` bundles
    # (4 rt cycles) with the next step's operands in the other register set, against 24 cycles for two full row blocks.
    RT = 3 if rows is None else -(-rows // 4)
    SINGLE = rows is not None
    assert not SINGLE or (1 <= rows <= 12 and drain == "dma" and not rowmax and not stamps and timing in (None, "dmaonly", "nodma", "pfnodma", "pfnodma2")), "k_gemm_gs(rows): 1..12, drain=dma"
    assert timing not in ("dmaonly", "nodma", "pfnodma", "pfnodma2") or (SINGLE and b8), "timing dmaonly / nodma: rows mode, b8"
    # "pfnodma" (diagnostic, wrong results): the rows-mode PREFETCH path with its B / A requests
    # and their waits dropped (the C drain kept, as "nodma" keeps it) -- the TEC-only arm of the same text the full path runs
    # ("nodma" takes the non-prefetch path)
    # "pfnodma2" (diagnostic): "pfnodma" with the TF k-loop at twice its trip count (it reads past its
    # codes and A in LSRAM: garbage), so pfnodma2 - pfnodma is the k-loop's own board cost
    NODMA = timing in ("nodma", "pfnodma", "pfnodma2"); KX = 2 if timing == "pfnodma2" else 1
    def dmaq(x): return [] if NODMA else x         # a DMA request or its wait, dropped by "nodma"
    NC = 8 * RT                                    # C registers in use: t0 .. t(NC - 1)
    # rows mode reads a compact A: row block 0's RT tiles only, [slice][k][RT tiles][32 B] (AST B a k-step), so no padding
    # tile or row block crosses DDR (the caller's descriptor and slice pitch are ks x AST)
    AST = 32 * RT if SINGLE else 96
    SCLB = (ns * 32 * (ks // 8) if q8f else ns * 32 if q8 else ns * (32 if F16S else 64 if SSC else 128)) if bscale else 0
    # `rowmax` (drain="dma"): each row block's C row in DDR is followed by its 12 rows' lane-wise maxima over
    # the group's columns (192 B: [i][h] float8 per row pair 4i + 2h, lanes 0-3 / 4-7 its two rows), for the
    # softmax; taken from the last K slice's registers. Uses r28 (DDR C pointer) and r30 (maxima): no stamps.
    CROW = ns * 768 + (192 if rowmax else 0)
    # `b8`: B arrives as E4M3 codes in panel order (ns x ks x 64 B a slice, half the DDR bytes; the linears are
    # DMA-bandwidth bound), landing in the upper half of the B region and expanded in place once per slice: each
    # 16-bit lane becomes `((code << 8) asr 1) & 0xBFFF`, the fp16 of code x 2^-8, exact for all 254 non-NaN codes
    # (NaN codes 0x7f / 0xff read as +-1.875). C comes out x 2^-8; the caller's per-channel scale carries 2^8.
    # Row block 0's A request is issued before the expansion so they overlap.
    BH = ns * ks * (16 if tern else 64 if b8 else 128)   # B bytes a slice from DDR (tern: 2-bit codes)
    TU = next(u for u in (3, 2, 1) if (BH // 32) % u == 0) if tern else 0   # tern: 32-B code units an expand iteration
    # Diagnostic `timing` (wrong results, timing only): "noc" skips the middle slices' GSRAM C loads/stores;
    # "gs{x}{n}" adds n GSRAM accesses of t31 at r15 per 3-k-step body (x: "l" alternating ld/st, "o" loads, "s" stores).
    # "dmaonly" / "nodma" (rows mode, diagnostics, wrong results): only the DMAs and their waits / only the expansion and compute
    assert timing in (None, "noc", "dmaonly", "nodma", "pfnodma", "pfnodma2") or timing[:3] in ("gsl", "gso", "gss")
    GSL = int(timing[3:]) if timing and timing[:2] == "gs" else 0
    GSK = timing[2] if GSL else None
    c_stride = nrb * CROW if c_stride is None else c_stride
    # `drain="dma"`: GSRAM only between K slices. The first slice starts from zeroed registers (no zeroing pass),
    # the last stores C into an LSRAM staging buffer and each row block's ns x 768 B go to DDR in one DMA (flag 3,
    # double-buffered by row-block parity), avoiding slow TEC stores to DDR. DESC gains a slot at +64: the drain.
    assert drain in ("tec", "dma")
    # `a_rb_stride` / `c_rb_stride`: DDR distance between row blocks' A slices / C rows (default ks x 96 / ns x 768),
    # e.g. a 48-deep 3-strip kernel reading the 24-deep A layout and draining into the 6-strip C layout.
    # c_rb_stride needs drain="dma" and a power-of-two ratio to ns x 768.
    a_rb_stride = ks * 96 if a_rb_stride is None else a_rb_stride
    c_rb_ratio = 1 if c_rb_stride is None else c_rb_stride // (ns * 768)
    assert c_rb_stride is None or (drain == "dma" and c_rb_ratio * ns * 768 == c_rb_stride and c_rb_ratio & (c_rb_ratio - 1) == 0)
    DMA_DRAIN = drain == "dma"
    assert not rowmax or (DMA_DRAIN and c_rb_ratio == 1 and not stamps), "k_gemm_gs(rowmax): drain=dma, the plain C layout, no stamps"
    # LSRAM (tec_res.Lsram: capacity, 32-B alignment and the vld overhang checked; `LS.dump()` prints the map). Every region
    # is read by 16-B vld. The fp16 B panels come first (b8: a slice's codes land in their upper half and expand in place).
    LS = Lsram(f"k_gemm_gs(ks={ks}, ns={ns}, rt={RT})")
    BREG = LS.alloc("B panels (fp16)", ns * ks * 128, vld_tail=True)
    A0 = LS.alloc("A half 0", ks * 96, vld_tail=True); A1 = LS.alloc("A half 1", ks * 96, vld_tail=True)
    STG0 = LS.alloc("C staging 0", CROW, vld_tail=True); STG1 = LS.alloc("C staging 1", CROW, vld_tail=True)
    CBB = BH + (max(SCLB, 32 * TU) if tern else max(SCLB, 128) if V2 else SCLB)   # the codes + scales buffer: expand_v2's pipeline reads 128 B past the codes (a table under 128 B is padded)
    CB = LS.alloc("bscale: a slice's codes + scales", CBB, vld_tail=True) if bscale else STG1 + CROW
    TF = tern and SINGLE and RT <= 2 and timing in (None, "nodma", "pfnodma", "pfnodma2")
    SPB = (256 if TF else 64) * RT                 # TF: the k-loop's starting offsets (a register a C register) instead of S'
    SPO = LS.alloc("q8 / tern: the A slice's row sums x 2^-17 / 2^-18 (a pair a row tile; TF: the C start offsets)", SPB, vld_tail=True) if SP else None
    # rows mode, rt <= 2, block-scaled E4M3: the next slice's B (codes + scales, one DMA through DESC +128) and compact A are
    # requested while the current slice expands and computes. LSRAM: fp16 panels [0, 2 BH) | A x2 | one staging | B x2
    PF = SINGLE and RT <= 2 and b8 and bscale and timing in (None, "pfnodma", "pfnodma2")
    if PF:
        LP = Lsram(f"k_gemm_gs(ks={ks}, ns={ns}, rt={RT}) prefetch")
        if not TF: LP.alloc("B panels (fp16)", ns * ks * 128, vld_tail=True)   # TF expands in the k-loop: no panels
        PA0 = LP.alloc("A 0 (compact)", ks * AST, vld_tail=True); PA1 = LP.alloc("A 1 (compact)", ks * AST, vld_tail=True)
        PSTG = LP.alloc("C staging", CROW, vld_tail=True)
        PCB0 = LP.alloc("codes + scales 0", CBB, vld_tail=True); PCB1 = LP.alloc("codes + scales 1", CBB, vld_tail=True)
        if SP: SPO = LP.alloc("q8 / tern: the A slice's row sums x 2^-17 / 2^-18 (TF: the C start offsets)", SPB, vld_tail=True)
    # `tern` in rows mode at rt <= 2 (rows 1-8; not "dmaonly"): the C accumulators live in LSRAM, not GSRAM -- a strip's 8 rt
    # registers compact (256 rt B), a group's ns strips contiguous -- and the strip epilogue is list-scheduled (`strips_tf`).
    # The prefetch path's group block holds as many groups as the free LSRAM takes (TGBLK).
    TCS = 256 * RT                                 # tern fast path: a strip's accumulators in LSRAM
    if TF:
        LA = LP if PF else LS
        TGBLK = max(1, min(16, (LA.limit - (-(-LA.used // 32) * 32) - 16) // (ns * TCS))) if PF else 1
        TACC = LA.alloc("tern: the C accumulators (a group's ns strips x 8 rt registers)", TGBLK * ns * TCS, vld_tail=True)
        TSTG = PSTG if PF else STG0
    GB_STAMP = LS.limit
    ORDER = [(0, 0), (0, 2), (1, 0), (1, 2), (0, 1), (0, 3), (2, 0), (2, 2), (1, 1), (1, 3), (2, 1), (2, 3)]
    THIS = {0: ["B1", "B3"], 1: ["A2"]}
    NEXT = {3: ["A0"], 4: ["B0", "B2"], 5: ["A1"]}
    REG = {"A0": "t24", "A1": "t25", "A2": "t26", "B0": "t27", "B1": "t28", "B2": "t29", "B3": "t30"}
    def MMA(i, j):
        c = 2 * (4 * i + j)
        return I("mma t%d.fp32, t%d.fp32, %s.fp16, %s.fp16" % (c, c + 1, REG["A%d" % i], REG["B%d" % j]), E.mma("t%d" % c, REG["A%d" % i], REG["B%d" % j]))
    def LD(x, k):
        if x[0] == "A": base, off = "r8", 96 * k + 32 * int(x[1])
        else:           base, off = "r13", 128 * k + 32 * int(x[1])
        return I("ld %s, [%s+%d]" % (REG[x], base, off), E.vld(REG[x], base, off))
    kbody = []
    gsl = [(("st" if (GSK == "s" or (GSK == "l" and m % 2)) else "ld"), 32 * (m % 16)) for m in range(GSL)]
    GSLOT = [(k, b) for b in (2, 1, 3, 5) for k in range(3)]  # bundles with a free memory slot
    # vector-load offsets are 9 bits: when the next window's operands would sit at >= 512 (kw = 4), the pointer-advance
    # bundle moves from the end of the window to before the last k-step's final two bundles (b = 4, 5), whose loads (the
    # next window's B0 / B2 and A1) use the rebased offsets
    early = 128 * kw + 96 >= 512
    def LDN(x, k, b):          # the next k-step's operand
        if early and k + 1 == kw and b >= 4: return I("ld %s, [%s+%d]" % (REG[x], "r8" if x[0] == "A" else "r13", 32 * int(x[1])), E.vld(REG[x], "r8" if x[0] == "A" else "r13", 32 * int(x[1])))
        return LD(x, k + 1)
    advance = BUNDLE(I("add r8, r8, %d" % (96 * kw), E.add("r8", "r8", 96 * kw)), I("add r13, r13, %d" % (128 * kw), E.add("r13", "r13", 128 * kw)))
    for k in range(kw):
        for b in range(6):
            if early and k == kw - 1 and b == 4: kbody.append(advance)
            ops = [MMA(*ORDER[2 * b]), MMA(*ORDER[2 * b + 1])] + [LD(x, k) for x in THIS.get(b, [])] + [LDN(x, k, b) for x in NEXT.get(b, [])]
            for m, (op, off) in enumerate(gsl):
                if GSLOT[m % len(GSLOT)] == (k, b):
                    ops = ([I("st t31, [r15+%d]" % off, E.vst("t31", "r15", off))] if op == "st" else []) + ops + ([I("ld t31, [r15+%d]" % off, E.vld("t31", "r15", off))] if op == "ld" else [])
            kbody.append(BUNDLE(*ops))
    if not early: kbody.append(advance)
    if SINGLE and RT <= 2:
        # set 0 = the registers of the original schedule, set 1 = t16.. (free: C uses t0 .. t(8 RT - 1) <= t15)
        SETS = [{"A0": "t24", "A1": "t25", "B0": "t26", "B1": "t27", "B2": "t28", "B3": "t29"},
                {"A0": "t16", "A1": "t17", "B0": "t18", "B1": "t19", "B2": "t20", "B3": "t21"}]
        def MMA2(i, j, st): c = 2 * (4 * i + j); return I("mma t%d.fp32, t%d.fp32, %s.fp16, %s.fp16" % (c, c + 1, st["A%d" % i], st["B%d" % j]), E.mma("t%d" % c, st["A%d" % i], st["B%d" % j]))
        def LDS(x, st, kx, rebased):  # operand x of k-step kx into set st; rebased: the bases already advanced past this window
            base, stride = ("r8", AST) if x[0] == "A" else ("r13", 128); off = (0 if rebased else stride * kx) + 32 * int(x[1])
            return I("ld %s, [%s+%d]" % (st[x], base, off), E.vld(st[x], base, off))
        loads = ["A0", "B0", "B1", "B2", "B3"] + (["A1"] if RT == 2 else [])
        mmas = [(i, j) for i in range(RT) for j in range(4)]
        kbody = []
        for k in range(kw):
            cur, nxt = SETS[k % 2], SETS[(k + 1) % 2]; last = k == kw - 1
            if last: kbody.append(BUNDLE(I("add r8, r8, %d" % (AST * kw), E.add("r8", "r8", AST * kw)), I("add r13, r13, %d" % (128 * kw), E.add("r13", "r13", 128 * kw))))
            nb = max(2 * RT, 3)                              # rt 1: a third bundle carries the fifth load
            for b in range(nb):
                ops = [MMA2(*m, cur) for m in mmas[2 * b:2 * b + 2]] + [LDS(x, nxt, k + 1, last) for x in loads[2 * b:2 * b + 2]]
                kbody.append(BUNDLE(*ops))
    def cbase(t): return ("r15", t * 32) if t < 16 else ("r14", (t - 16) * 32)
    def cload(t): b, off = cbase(t); return I("ld t%d, [%s+%d]" % (t, b, off), E.vld("t%d" % t, b, off))
    def cload_into(r, t): b, off = cbase(t); return I("ld t%d, [%s+%d]" % (r, b, off), E.vld("t%d" % r, b, off))
    def cstore(t): b, off = cbase(t); return I("st t%d, [%s+%d]" % (t, b, off), E.vst("t%d" % t, b, off))
    def dma_in(sync, desc, lsram, ddr):
        return BUNDLE(I("dma 0, %s, 0, 4, 0, 1, 0, %s, %s, %s" % (sync, desc, lsram, ddr), E.dma(E.DMA_DIRECT, sync, 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT, E.DMA_USELESS, desc, lsram, ddr)))
    def CYC(r): return BUNDLE(I("mfctrl0 %s, 209" % r, E.mfctrl0(r, 0xd1)))
    def acc(reg, start_reg, end_reg):
        return [BUNDLE(I("sub %s, %s, %s" % (end_reg, end_reg, start_reg), E.sub(end_reg, end_reg, start_reg))),
                BUNDLE(I("add %s, %s, %s" % (reg, reg, end_reg), E.add(reg, reg, end_reg)))]
    def wait(flag):                                # tec_res: flags 0-3 only
        w = wait_flags(flag)
        return w if not stamps else [CYC("r30")] + w + [CYC("r7")] + acc("r20", "r30", "r7")
    def sync(flag): return sync_flag(flag)
    def add(rd, ra, x):
        if isinstance(x, int) and x > 1023:
            return _mov32("r7", x) + [BUNDLE(I("add %s, %s, r7" % (rd, ra), E.add(rd, ra, "r7")))]
        return [BUNDLE(I("add %s, %s, %s" % (rd, ra, x), E.add(rd, ra, x)))]

    def ifz(test, then, other):
        """`test` leaves r7 == 0 for THEN: [test] cbnz r7 -> other; then; b -> end; other."""
        return test + [BUNDLE(I("cbnz r7, .Lelse", E.cbnz("r7", len(then) + 2)))] + then + [BUNDLE(I("b .Lend", E.b(len(other) + 1)))] + other

    first_slice = [BUNDLE(I("sub r7, r18, r4", E.sub("r7", "r18", "r4")))]          # r18 counts slices down from r4
    last_slice = [BUNDLE(I("sub r7, r18, 1", E.sub("r7", "r18", 1)))]
    def zero_c(): return [BUNDLE(I("mov4 t%d, 0" % t, E.mov4("t%d" % t, 0))) for t in range(NC)]
    def stage_c():                 # C -> staging buffer at r3, r3 += 768 (rows past the RT tiles: zeros, from t30)
        st = add("r14", "r3", 512) + ([BUNDLE(I("mov4 t30, 0", E.mov4("t30", 0)))] if NC < 24 else [])
        for t in range(24):
            b_, off = ("r3", t * 32) if t < 16 else ("r14", (t - 16) * 32); src = t if t < NC else 30
            st += [BUNDLE(I("st t%d, [%s+%d]" % (src, b_, off), E.vst("t%d" % src, b_, off)))]
        return st + add("r3", "r3", 768)
    def rowmax_strip():            # after stage_c (t0-t23 free): max over the strip's 4 column tiles,
        ops = []                   # then against the running maxima at r30 (t24-t29 dead here)
        l1 = [(8 * i + h + a, 8 * i + h + a + 2) for i in range(3) for h in range(2) for a in (0, 4)]
        lds = [(24 + k, 32 * k) for k in range(6)]
        for n in range(6):
            b_ = [I("max t%d.fp32, t%d.fp32, t%d.fp32, p7.w" % (d, d, x), E.vmaxf("t%d" % d, "t%d" % d, "t%d" % x)) for d, x in l1[2 * n:2 * n + 2]]
            if n < 3: b_ += [I("ld t%d, [r30+%d]" % (r, o), E.vld("t%d" % r, "r30", o)) for r, o in lds[2 * n:2 * n + 2]]
            ops.append(BUNDLE(*b_))
        l2 = [8 * i + h for i in range(3) for h in range(2)]
        for n in range(3):
            ops.append(BUNDLE(*[I("max t%d.fp32, t%d.fp32, t%d.fp32, p7.w" % (d, d, d + 4), E.vmaxf("t%d" % d, "t%d" % d, "t%d" % (d + 4))) for d in l2[2 * n:2 * n + 2]]))
        for n in range(3):
            ops.append(BUNDLE(*[I("max t%d.fp32, t%d.fp32, t%d.fp32, p7.w" % (24 + k, 24 + k, l2[k]), E.vmaxf("t%d" % (24 + k), "t%d" % (24 + k), "t%d" % l2[k])) for k in (2 * n, 2 * n + 1)]))
        ops += [BUNDLE(I("st t%d, [r30+%d]" % (24 + k, 32 * k), E.vst("t%d" % (24 + k), "r30", 32 * k))) for k in range(6)]
        return ops

    def expand(src=None):          # [r25 + BH, r25 + 2 BH) (or the codes at register `src`) E4M3 -> [r25, r25 + 2 BH) fp16 x 2^-8
        e = [BUNDLE(I("mov4 t0, 0", E.mov4("t0", 0)))] + _mov32("r7", 0xBFFFBFFF) + [BUNDLE(I("bcast t1.w, r7", E.bcast("t1", "r7")))]
        e += (add("r13", "r25", BH) if src is None else add("r13", src, 0)) + add("r14", "r25", 0) + [BUNDLE(I("mov r10, %d" % (BH // 128 - 1), E.mov("r10", BH // 128 - 1))), BUNDLE(I("loop r10", E.loop("r10")))]
        X, H = ["t%d" % (2 + v) for v in range(4)], ["t%d" % (6 + j) for j in range(8)]
        body = [BUNDLE(I("ld %s, [r13+%d]" % (X[0], 0), E.vld(X[0], "r13", 0)), I("ld %s, [r13+%d]" % (X[1], 32), E.vld(X[1], "r13", 32))),
                BUNDLE(I("ld %s, [r13+%d]" % (X[2], 64), E.vld(X[2], "r13", 64)), I("ld %s, [r13+%d]" % (X[3], 96), E.vld(X[3], "r13", 96)))]
        for v in range(4):  # codes into the high byte of each 16-bit lane
            body += [BUNDLE(I("zipl %s.b, t0.b, %s.b" % (H[2 * v], X[v]), E.zipl(H[2 * v], "t0", X[v]))),
                     BUNDLE(I("ziph %s.b, t0.b, %s.b" % (H[2 * v + 1], X[v]), E.ziph(H[2 * v + 1], "t0", X[v])))]
        body[-1] = BUNDLE(*(list(body[-1]) + [I("add r13, r13, 128", E.add("r13", "r13", 128))]))
        for j in range(8): body.append(BUNDLE(I("asr %s.h, %s.h, 1" % (H[j], H[j]), E.vasri(H[j], H[j], 1, size="h"))))
        for j in range(8): body.append(BUNDLE(I("and %s.w, %s.w, t1.w" % (H[j], H[j]), E.vand(H[j], H[j], "t1"))))
        for j in range(8): body.append(BUNDLE(I("st %s, [r14+%d]" % (H[j], 32 * j), E.vst(H[j], "r14", 32 * j))))
        body[-1] = BUNDLE(*(list(body[-1]) + [I("add r14, r14, 256", E.add("r14", "r14", 256))]))
        return e + body + [BUNDLE(I("loopend", E.loopend()))]

    def expand_v2(src=None):       # the same bytes as expand(), by the byte-widening multiplies (enc.vmulb)
        # A code c read as a signed byte s and as an unsigned byte u: 192 s - 64 u = ((c & 0x7f) << 7) | (c & 0x80) << 8 in a
        # 16-bit lane, i.e. ((c << 8) asr 1) & 0xBFFF, the fp16 of code x 2^-8 (all 256 codes; the sign bit's two readings differ
        # by 256, and 64 x 256 = 2^14 is exactly the bit the asr duplicates). `mul` / `mulh` take byte lanes 0-15 / 16-31 straight
        # into 16-bit lanes (slots 0/1, two a bundle), so a 32-code register is six arithmetic ops and two stores and nothing on
        # slot 2 but the stores: 16 bundles per 128 codes against v1's 36 (8 zip + 8 asr + 8 and, one a bundle, + 8 st).
        # Software-pipelined: the next 128 codes load in this body's store-free bundles (a register is reloaded two bundles after
        # its last read; 2-cycle vector latency and the 3-cycle load-to-use honoured), so the last iteration reads 128 B past the
        # codes -- the scales that follow them (SCLB >= 128; a smaller table pads the buffer: CBB), or the A half after the B region without bscale: LSRAM either way.
        X = ["t%d" % (24 + v) for v in range(4)]; P = ["t%d" % (2 + j) for j in range(8)]; Q = ["t%d" % (10 + j) for j in range(8)]
        K192, K64 = "t30", "t0"    # the unsigned 192 (0xC0 bytes, mov4 -64) and the signed 64: the unsigned source of each product is the higher-numbered register
        e = [BUNDLE(I("mov4 t30, -64", E.mov4("t30", -64))), BUNDLE(I("mov4 t0, 64", E.mov4("t0", 64)))]
        e += (add("r13", "r25", BH) if src is None else add("r13", src, 0)) + add("r14", "r25", 0)
        def LDX(v): return I("ld %s, [r13+%d]" % (X[v], 32 * v), E.vld(X[v], "r13", 32 * v))
        e += [BUNDLE(LDX(0), LDX(1)), BUNDLE(LDX(2), LDX(3)), BUNDLE(I("add r13, r13, 128", E.add("r13", "r13", 128))),
              BUNDLE(I("mov r10, %d" % (BH // 128 - 1), E.mov("r10", BH // 128 - 1))), BUNDLE(I("loop r10", E.loop("r10")))]
        def PQ(j, which):          # the two products of output tile j: codes X[j // 2], lanes 0-15 (j even) or 16-31 (j odd)
            x, hi = X[j // 2], bool(j % 2); m = "mulh" if hi else "mul"
            if which == "P": return I("%s %s.h, %s.b, %s.ub, p7.b" % (m, P[j], x, K192), E.vmulb(P[j], x, K192, ub=True, high=hi))
            return I("%s %s.h, %s.ub, %s.b, p7.b" % (m, Q[j], x, K64), E.vmulb(Q[j], x, K64, ua=True, high=hi))
        def H(j): return I("sub %s.h, %s.h, %s.h" % (P[j], P[j], Q[j]), E.vsub(P[j], P[j], Q[j], size="h"))
        def ST(j): return I("st %s, [r14+%d]" % (P[j], 32 * j), E.vst(P[j], "r14", 32 * j))
        body = [BUNDLE(PQ(0, "P"), PQ(0, "Q")),                                    # b0
                BUNDLE(PQ(1, "P"), PQ(1, "Q")),                                    # b1
                BUNDLE(H(0), PQ(2, "P")),                                          # b2
                BUNDLE(PQ(2, "Q"), H(1), LDX(0)),                                  # b3   X0 last read at b1
                BUNDLE(PQ(3, "P"), PQ(3, "Q"), ST(0)),                             # b4
                BUNDLE(H(2), PQ(4, "P"), ST(1)),                                   # b5
                BUNDLE(PQ(4, "Q"), H(3), LDX(1)),                                  # b6   X1 last read at b4
                BUNDLE(PQ(5, "P"), PQ(5, "Q"), ST(2)),                             # b7
                BUNDLE(H(4), PQ(6, "P"), ST(3)),                                   # b8
                BUNDLE(PQ(6, "Q"), H(5), LDX(2)),                                  # b9   X2 last read at b7
                BUNDLE(PQ(7, "P"), PQ(7, "Q"), ST(4)),                             # b10
                BUNDLE(H(6), ST(5)),                                               # b11
                BUNDLE(H(7), LDX(3)),                                              # b12  X3 last read at b10
                BUNDLE(ST(6), I("add r13, r13, 128", E.add("r13", "r13", 128))),   # b13
                BUNDLE(ST(7), I("add r14, r14, 256", E.add("r14", "r14", 256))),   # b14
                BUNDLE(I("loopend", E.loopend()))]
        return e + body
    if V2: expand = expand_v2

    def expand_q8(src=None):       # u = q + 128 -> the fp16 subnormal u x 2^-24: the byte into each 16-bit lane's LOW half (one zip)
        e = [BUNDLE(I("mov4 t0, 0", E.mov4("t0", 0)))]
        e += (add("r13", "r25", BH) if src is None else add("r13", src, 0)) + add("r14", "r25", 0) + [BUNDLE(I("mov r10, %d" % (BH // 128 - 1), E.mov("r10", BH // 128 - 1))), BUNDLE(I("loop r10", E.loop("r10")))]
        X, H = ["t%d" % (2 + v) for v in range(4)], ["t%d" % (6 + j) for j in range(8)]
        body = [BUNDLE(I("ld %s, [r13+%d]" % (X[0], 0), E.vld(X[0], "r13", 0)), I("ld %s, [r13+%d]" % (X[1], 32), E.vld(X[1], "r13", 32))),
                BUNDLE(I("ld %s, [r13+%d]" % (X[2], 64), E.vld(X[2], "r13", 64)), I("ld %s, [r13+%d]" % (X[3], 96), E.vld(X[3], "r13", 96)))]
        for v in range(4):
            body += [BUNDLE(I("zipl %s.b, %s.b, t0.b" % (H[2 * v], X[v]), E.zipl(H[2 * v], X[v], "t0"))),
                     BUNDLE(I("ziph %s.b, %s.b, t0.b" % (H[2 * v + 1], X[v]), E.ziph(H[2 * v + 1], X[v], "t0")))]
        body[-1] = BUNDLE(*(list(body[-1]) + [I("add r13, r13, 128", E.add("r13", "r13", 128))]))
        for j in range(8): body.append(BUNDLE(I("st %s, [r14+%d]" % (H[j], 32 * j), E.vst(H[j], "r14", 32 * j))))
        body[-1] = BUNDLE(*(list(body[-1]) + [I("add r14, r14, 256", E.add("r14", "r14", 256))]))
        return e + body + [BUNDLE(I("loopend", E.loopend()))]

    def sprep(half_reg):           # q8: S'_i = the A slice's row sums x 2^-17 per row tile i (mml / mma against a constant tile) -> scratch
        c = _mov32("r7", 0x00400040 if tern else 0x00800080) + [BUNDLE(I("bcast t30.w, r7", E.bcast("t30", "r7")))]      # fp16 2^-17 (bits 0x0080; tern 2^-18, 0x0040) in every half
        c += add("r13", half_reg, 0)
        AR = ["t26", "t27", "t28"]
        for k in range(ks):
            for i0 in range(0, RT, 2):
                c.append(BUNDLE(*[I("ld %s, [r13+%d]" % (AR[i], 32 * i), E.vld(AR[i], "r13", 32 * i)) for i in range(i0, min(RT, i0 + 2))]))
            for i in range(RT):
                op, f = ("mml", E.mml) if k == 0 else ("mma", E.mma)
                c.append(BUNDLE(I("%s t%d.fp32, t%d.fp32, %s.fp16, t30.fp16" % (op, 2 * i, 2 * i + 1, AR[i]), f("t%d" % (2 * i), AR[i], "t30"))))
            c += add("r13", "r13", AST)
        c += _mov32("r7", SPO) + [BUNDLE(I("add r7, r25, r7", E.add("r7", "r25", "r7")))]
        if TF:                     # the C start offsets O_j = K_j S' (K = -85/64, -21/16, -5/4, -1): after the k-loop the triangle
            # C(j) - C(j + 1) / 4 then carries O_j - O_(j+1) / 4 = -S' (O_3 = -S'), so the epilogue has no subtract; [i][j][h] at 32 t
            for j, kv in enumerate((-85 / 64, -21 / 16, -5 / 4, -1.0)):
                c += _mov32("r10", struct.unpack("<I", struct.pack("<f", kv))[0]) + [BUNDLE(I("bcast t%d.w, r10" % (20 + j), E.bcast("t%d" % (20 + j), "r10")))]
            for i in range(RT):
                for j in range(4):
                    c.append(BUNDLE(*[I("mul t%d.fp32, t%d.fp32, t%d.fp32, p7.w" % (8 + 2 * j + h, 2 * i + h, 20 + j), E.vmulf("t%d" % (8 + 2 * j + h), "t%d" % (2 * i + h), "t%d" % (20 + j))) for h in (0, 1)]))
                for q in range(8):
                    c.append(BUNDLE(I("st t%d, [r7+%d]" % (8 + q, 256 * i + 32 * q), E.vst("t%d" % (8 + q), "r7", 256 * i + 32 * q))))
            return c
        for i in range(RT):
            c.append(BUNDLE(I("st t%d, [r7+%d]" % (2 * i, 64 * i), E.vst("t%d" % (2 * i), "r7", 64 * i))))
            c.append(BUNDLE(I("st t%d, [r7+%d]" % (2 * i + 1, 64 * i + 32), E.vst("t%d" % (2 * i + 1), "r7", 64 * i + 32))))
        return c
    def expand_q8f(src):           # codes at `src`, scales at src + BH -> the fp16 weights (1024 + u - 1152) x d in [r25, r25 + 2 BH)
        nb = ks // 8
        e = [BUNDLE(I("mov4 t0, 100", E.mov4("t0", 0x64)))] + _mov32("r7", 0x64806480) + [BUNDLE(I("bcast t1.w, r7", E.bcast("t1", "r7")))]   # 0x64 bytes; fp16 1152
        e += add("r13", src, 0) + add("r12", src, BH) + add("r14", "r25", 0)
        X, H, SC = ["t%d" % (2 + v) for v in range(4)], ["t%d" % (6 + j) for j in range(8)], ["t14", "t15", "t16", "t17"]
        for sb in range(ns * nb):  # per strip and scale block: its four tile scale vectors, then 8 k-steps (512 codes) in 4 passes
            e += [BUNDLE(I("ld t18, [r12+%d]" % (32 * sb), E.vld("t18", "r12", 32 * sb)))]
            e += [BUNDLE(I("zipl t19.h, t18.h, t18.h", E.zipl("t19", "t18", "t18", size="h"))), BUNDLE(I("ziph t20.h, t18.h, t18.h", E.ziph("t20", "t18", "t18", size="h"))),
                  BUNDLE(I("zipl %s.w, t19.w, t19.w" % SC[0], E.zipl(SC[0], "t19", "t19", size="w"))), BUNDLE(I("ziph %s.w, t19.w, t19.w" % SC[1], E.ziph(SC[1], "t19", "t19", size="w"))),
                  BUNDLE(I("zipl %s.w, t20.w, t20.w" % SC[2], E.zipl(SC[2], "t20", "t20", size="w"))), BUNDLE(I("ziph %s.w, t20.w, t20.w" % SC[3], E.ziph(SC[3], "t20", "t20", size="w")))]
            e += [BUNDLE(I("mov r10, 3", E.mov("r10", 3))), BUNDLE(I("loop r10", E.loop("r10")))]
            body = [BUNDLE(I("ld %s, [r13+0]" % X[0], E.vld(X[0], "r13", 0)), I("ld %s, [r13+32]" % X[1], E.vld(X[1], "r13", 32))),
                    BUNDLE(I("ld %s, [r13+64]" % X[2], E.vld(X[2], "r13", 64)), I("ld %s, [r13+96]" % X[3], E.vld(X[3], "r13", 96)))]
            for v in range(4):         # tile j = (2 v + 0 / 1) % 4: zipl / ziph of code register v
                body += [BUNDLE(I("zipl %s.b, %s.b, t0.b" % (H[2 * v], X[v]), E.zipl(H[2 * v], X[v], "t0"))),
                         BUNDLE(I("ziph %s.b, %s.b, t0.b" % (H[2 * v + 1], X[v]), E.ziph(H[2 * v + 1], X[v], "t0")))]
            body[-1] = BUNDLE(*(list(body[-1]) + [I("add r13, r13, 128", E.add("r13", "r13", 128))]))
            for j in range(0, 8, 2):
                body.append(BUNDLE(*[I("sub %s.fp16, %s.fp16, t1.fp16, p7.h" % (H[i], H[i]), E.vsubf16(H[i], H[i], "t1")) for i in (j, j + 1)]))
            for j in range(0, 8, 2):
                body.append(BUNDLE(*[I("mul %s.fp16, %s.fp16, %s.fp16, p7.h" % (H[i], H[i], SC[i % 4]), E.vmulf16(H[i], H[i], SC[i % 4])) for i in (j, j + 1)]))
            for j in range(8): body.append(BUNDLE(I("st %s, [r14+%d]" % (H[j], 32 * j), E.vst(H[j], "r14", 32 * j))))
            body[-1] = BUNDLE(*(list(body[-1]) + [I("add r14, r14, 256", E.add("r14", "r14", 256))]))
            e += body + [BUNDLE(I("loopend", E.loopend()))]
        return e
    def expand_tern(src):          # 2-bit codes at `src` (BH B) -> the fp16 panels [r25, r25 + 8 BH): tile j's lanes u(4^j X mod 256) x 2^-24
        # One iteration = TU 32-B code units (each 128 weights = a strip's k-step pair = 256 B of panels): per unit 6 byte `add`s
        # (X -> 2X -> 4X ... 64X, wrapping), 8 widening `mul` / `mulh` by 1 (Y_j unsigned) and 8 stores; scheduled below under the
        # measured rules (2 ALU ops a bundle, one store, loads beside a store or in pairs, 2-cycle vector latency, 3-cycle
        # load-to-use). The next iteration's codes load in this one (after the last read of X), so the last iteration reads
        # 32 TU B past the codes: the scale table (CBB pads it).
        CU = [tuple("t%d" % (23 + 3 * u + v) for v in range(3)) for u in range(TU)]   # X, T, Y per unit: t23 .. t31
        ONE = "t0"; POOL = ["t%d" % r for r in range(1, 23)]
        e = [BUNDLE(I("mov4 t0, 1", E.mov4("t0", 1)))] + add("r13", src, 0) + add("r14", "r25", 0) + add("r12", "r25", 512)
        def LDX(u): return I("ld %s, [r13+%d]" % (CU[u][0], 32 * u), E.vld(CU[u][0], "r13", 32 * u))
        e += [BUNDLE(*[LDX(u) for u in range(u0, min(TU, u0 + 2))]) for u0 in range(0, TU, 2)]
        e += [BUNDLE(I("add r13, r13, %d" % (32 * TU), E.add("r13", "r13", 32 * TU))),
              BUNDLE(I("mov r10, %d" % (BH // (32 * TU) - 1), E.mov("r10", BH // (32 * TU) - 1))), BUNDLE(I("loop r10", E.loop("r10")))]
        # ops of one iteration: name -> kind, deps (RAW: (name, latency)), after (WAR: issued in an earlier bundle), emitter
        ops = []
        def ADD(n, d, a, deps, after=()): ops.append(dict(n=n, k="alu", deps=deps, after=list(after), f=lambda d=d, a=a: I("add %s.b, %s.b, %s.b" % (d, a, a), E.vadd(d, a, a, size="b"))))
        for u in range(TU):
            X, T, Y = CU[u]
            ADD(f"t{u}0", T, X, [])                                          # T = 2X
            for j in range(4):
                yj, dep = (X, []) if j == 0 else (Y, [(f"y{u}{j}", 2)])
                for hi in (0, 1): ops.append(dict(n=f"m{u}{j}{hi}", k="mul", deps=dep, after=[], y=yj, hi=hi, u=u, j=j))
                if j == 3: break
                if j > 0: ADD(f"t{u}{j}", T, Y, [(f"y{u}{j}", 2)])          # T = 2 Y_j
                ADD(f"y{u}{j+1}", Y, T, [(f"t{u}{j}", 2)], [f"m{u}{j}0", f"m{u}{j}1", f"t{u}{j}"] if j else [])   # Y = 4 Y_j
        H = {}                                                               # priority: the longest path to a store (RAW latency, WAR 1)
        def height(n):
            if n not in H:
                H[n] = max([2] + [height(o["n"]) + l for o in ops for a, l in o["deps"] if a == n] + [height(o["n"]) + 1 for o in ops if n in o["after"]])
            return H[n]
        for o in ops: height(o["n"])                                         # before the sort (list.sort empties the list it sorts)
        ops.sort(key=lambda o: (-H[o["n"]], o["n"][1], o["n"]))
        done, sched, free_at, stores = {}, [], {r: 0 for r in POOL}, []
        pending = list(ops); cyc = 0; loads_left = list(range(TU)); ptr_done = False
        def ok(op): return all(a in done and done[a] + l <= cyc for a, l in op["deps"]) and all(a in done and done[a] < cyc for a in op["after"])
        while pending or stores or loads_left or not ptr_done:
            alu, mem = [], []
            feed = len(stores) < 2                                           # keep the store slot fed: products first when few are queued
            for op in sorted(pending, key=lambda o: (not (feed and o["k"] == "mul"), pending.index(o))):
                if len(alu) == 2: break
                if not ok(op): continue
                if op["k"] == "mul":
                    reg = next((r for r in POOL if free_at[r] <= cyc), None)
                    if reg is None: continue
                    free_at[reg] = 10 ** 9; u, j, hi = op["u"], op["j"], op["hi"]
                    alu.append(I("%s %s.h, %s.ub, %s.b, p7.b" % ("mulh" if hi else "mul", reg, op["y"], ONE), E.vmulb(reg, op["y"], ONE, ua=True, high=bool(hi))))
                    stores.append((cyc + 2, 256 * u + 128 * hi + 32 * j, reg))
                else: alu.append(op["f"]())
                done[op["n"]] = cyc; pending.remove(op)
            st = min((x for x in stores if x[0] <= cyc), default=None)       # a store: the oldest ready product
            if st:
                stores.remove(st); _, off, reg = st; base, o = ("r14", off) if off < 512 else ("r12", off - 512)
                mem.append(I("st %s, [%s+%d]" % (reg, base, o), E.vst(reg, base, o))); free_at[reg] = cyc + 1
            for u in list(loads_left):                                        # the next iteration's codes, once X is dead
                if len(mem) < 2 and all(n in done and done[n] < cyc for n in (f"t{u}0", f"m{u}00", f"m{u}01")): mem.append(LDX(u)); loads_left.remove(u)
            if not loads_left and not ptr_done and len(alu) < 2:              # r13 once the loads are issued
                alu.append(I("add r13, r13, %d" % (32 * TU), E.add("r13", "r13", 32 * TU))); ptr_done = True
            sched.append(BUNDLE(*alu, *mem)); cyc += 1
        sched.append(BUNDLE(I("add r14, r14, %d" % (256 * TU), E.add("r14", "r14", 256 * TU)), I("add r12, r12, %d" % (256 * TU), E.add("r12", "r12", 256 * TU))))   # the store bases, after every store
        # the next iteration's first use of X is >= 3 cycles after its load (body tail + loopend)
        last_ld = max(i for i, b in enumerate(sched) if any(x[0].startswith("ld ") for x in b))
        while len(sched) + 1 - last_ld < 3: sched.append(BUNDLE())
        return e + sched + [BUNDLE(I("loopend", E.loopend()))]
    if q8: expand = expand_q8
    if q8f: expand = expand_q8f
    if tern: expand = expand_tern

    def strips(half_reg, stg=None, scl=None, cratio=None):
        """All `ns` strips against the row block in `half_reg`; r13 walks the panels, r15 the C blocks.
        `stg` (drain=dma): the row block's LSRAM staging offset. `scl`: a register holding the slice's codes (scales at +BH)."""
        if TF: return strips_tf(half_reg, stg, scl or "r20")
        blk = add("r13", "r25", 0) + [BUNDLE(I("mov r27, %d" % ns, E.mov("r27", ns)))]
        if bscale: blk += add("r30", scl or "r20", BH)          # the slice's scales follow its codes
        if DMA_DRAIN:  # r3 = staging buffer, r12 = row block's C in DDR
            blk += _mov32("r7", stg) + [BUNDLE(I("add r3, r25, r7", E.add("r3", "r25", "r7")))]
            if rowmax:  # r12 = C row (r28), r30 = maxima in staging, -inf at the last slice
                blk += [BUNDLE(I("add r12, r28, 0", E.add("r12", "r28", 0)))] + add("r28", "r28", CROW)
                blk += _mov32("r7", stg + ns * 768) + [BUNDLE(I("add r30, r25, r7", E.add("r30", "r25", "r7")))]
                blk += ifz(last_slice, [BUNDLE(I("st t31, [r30+%d]" % (32 * k), E.vst("t31", "r30", 32 * k))) for k in range(6)], [])
            else:
                blk += [BUNDLE(I("sub r12, r15, r29", E.sub("r12", "r15", "r29")))]
                for _ in range(((cratio or c_rb_ratio).bit_length()) - 1): blk += [BUNDLE(I("add r12, r12, r12", E.add("r12", "r12", "r12")))]  # x C row ratio
                blk += [BUNDLE(I("add r12, r12, r2", E.add("r12", "r12", "r2")))]
        body = add("r14", "r15", 512)
        if stamps: body += [CYC("r30")]
        loads = [BUNDLE(cload(t), cload(t + 1)) for t in range(0, NC, 2)]
        if bscale: body += zero_c()
        else: body += ifz(first_slice, zero_c(), [] if timing == "noc" else loads) if DMA_DRAIN else loads
        if stamps: body += [CYC("r7")] + acc("r26", "r30", "r7")
        if SINGLE and RT <= 2:     # k-step 0's operands into set 0, >= 3 cycles before the first mma
            body += add("r8", half_reg, 0) + [BUNDLE(LDS("A0", SETS[0], 0, False), LDS("B0", SETS[0], 0, False)), BUNDLE(LDS("B1", SETS[0], 0, False), LDS("B2", SETS[0], 0, False)),
                                                BUNDLE(LDS("B3", SETS[0], 0, False), *([LDS("A1", SETS[0], 0, False)] if RT == 2 else [])),
                                                BUNDLE(I("mov r10, %d" % (ks // kw - 1), E.mov("r10", ks // kw - 1)))]
        else:
            body += add("r8", half_reg, 0) + [BUNDLE(LD("B0", 0), LD("B2", 0)), BUNDLE(LD("A0", 0)), BUNDLE(LD("A1", 0)),
                                                BUNDLE(I("mov r10, %d" % (ks // kw - 1), E.mov("r10", ks // kw - 1)))]
        if stamps: body += [CYC("r30")]
        body += [BUNDLE(I("loop r10", E.loop("r10")))] + kbody + [BUNDLE(I("loopend", E.loopend()))]
        if stamps: body += [CYC("r7")] + acc("r21", "r30", "r7") + [CYC("r30")]
        if tern:                   # C(j) -= C(j + 1) / 4 (j = 0, 1, 2: each reads the next tile before it is corrected), then C -= S'
            body += _mov32("r7", 0xBE800000) + [BUNDLE(I("bcast t31.w, r7", E.bcast("t31", "r7")))]     # -0.25 (t31: free after the k-loop at every RT)
            for i in range(RT):
                for j in range(3):
                    c, d = 2 * (4 * i + j), 2 * (4 * i + j + 1)
                    body.append(BUNDLE(*[I("fma t%d.fp32, t%d.fp32, t31.fp32, p7.w" % (c + h, d + h), E.vfmaf("t%d" % (c + h), "t%d" % (d + h), "t31")) for h in (0, 1)]))
            body += _mov32("r7", SPO) + [BUNDLE(I("add r7, r25, r7", E.add("r7", "r25", "r7")))]
            for i in range(RT):
                body.append(BUNDLE(I("ld t28, [r7+%d]" % (64 * i), E.vld("t28", "r7", 64 * i)), I("ld t29, [r7+%d]" % (64 * i + 32), E.vld("t29", "r7", 64 * i + 32))))
                for j in range(4):
                    c = 2 * (4 * i + j)
                    body.append(BUNDLE(I("sub t%d.fp32, t%d.fp32, t28.fp32, p7.w" % (c, c), E.vsubf("t%d" % c, "t%d" % c, "t28")),
                                       I("sub t%d.fp32, t%d.fp32, t29.fp32, p7.w" % (c + 1, c + 1), E.vsubf("t%d" % (c + 1), "t%d" % (c + 1), "t29"))))
        if bscale and q8:          # C -= S' (the codes' +128), then x the strip's 16 scales: fp16 d x 2^10 -> fp32 d x 2^24, lanes [c0..c3 c0..c3]
            body += _mov32("r7", SPO) + [BUNDLE(I("add r7, r25, r7", E.add("r7", "r25", "r7")))]
            for i in range(RT):
                body.append(BUNDLE(I("ld t28, [r7+%d]" % (64 * i), E.vld("t28", "r7", 64 * i)), I("ld t29, [r7+%d]" % (64 * i + 32), E.vld("t29", "r7", 64 * i + 32))))
                for j in range(4):
                    c = 2 * (4 * i + j)
                    body.append(BUNDLE(I("sub t%d.fp32, t%d.fp32, t28.fp32, p7.w" % (c, c), E.vsubf("t%d" % c, "t%d" % c, "t28")),
                                       I("sub t%d.fp32, t%d.fp32, t29.fp32, p7.w" % (c + 1, c + 1), E.vsubf("t%d" % (c + 1), "t%d" % (c + 1), "t29"))))
            body += [BUNDLE(I("ld t24, [r30+0]", E.vld("t24", "r30", 0)), I("mov4 t31, 0", E.mov4("t31", 0)))]
            body += _mov32("r7", 0x3F000000) + [BUNDLE(I("bcast t30.w, r7", E.bcast("t30", "r7")))]
            body += [BUNDLE(I("zipl t25.h, t24.h, t31.h", E.zipl("t25", "t24", "t31", size="h"))), BUNDLE(I("ziph t26.h, t24.h, t31.h", E.ziph("t26", "t24", "t31", size="h"))),
                     BUNDLE(I("lsl t25.w, t25.w, 13", E.vlsli("t25", "t25", 13, size="w"))), BUNDLE(I("lsl t26.w, t26.w, 13", E.vlsli("t26", "t26", 13, size="w"))),
                     BUNDLE(I("add t25.w, t25.w, t30.w", E.vadd("t25", "t25", "t30")), I("add t26.w, t26.w, t30.w", E.vadd("t26", "t26", "t30"))),
                     BUNDLE(I("extl t27.w, t25.w, t25.w", E.extl("t27", "t25", "t25", size="w"))), BUNDLE(I("exth t28.w, t25.w, t25.w", E.exth("t28", "t25", "t25", size="w"))),
                     BUNDLE(I("extl t29.w, t26.w, t26.w", E.extl("t29", "t26", "t26", size="w"))), BUNDLE(I("exth t24.w, t26.w, t26.w", E.exth("t24", "t26", "t26", size="w")))]
            SC = ["t27", "t28", "t29", "t24"]
            for i in range(RT):
                for j in range(4):
                    c = 2 * (4 * i + j)
                    body += [BUNDLE(*[I("mul t%d.fp32, t%d.fp32, %s.fp32, p7.w" % (r, r, SC[j]), E.vmulf("t%d" % r, "t%d" % r, SC[j])) for r in (c, c + 1)])]
        elif bscale and SSC and F16S:   # tern, fp16 table: [r30 + 0] = 16 fp16 (even lanes tiles 0-1's columns, odd 2-3's) x fp16 2^3 -> fp32
            # (fmae / fmao into zeroed registers, exact), then extl / exth as the fp32 table's; spaced by the result latencies (load 3,
            # vector 2, fp32 4)
            body += _mov32("r7", 0x48004800) + [BUNDLE(I("bcast t30.w, r7", E.bcast("t30", "r7")))]
            body += [BUNDLE(I("ld t24, [r30+0]", E.vld("t24", "r30", 0))), BUNDLE(I("mov4 t25, 0", E.mov4("t25", 0))), BUNDLE(I("mov4 t31, 0", E.mov4("t31", 0))),
                     BUNDLE(I("fmae t25.fp32, t24.fp16, t30.fp16, p7.h", E.vfmae("t25", "t24", "t30"))),
                     BUNDLE(I("fmao t31.fp32, t24.fp16, t30.fp16, p7.h", E.vfmao("t31", "t24", "t30"))), BUNDLE(), BUNDLE(),
                     BUNDLE(I("extl t26.w, t25.w, t25.w", E.extl("t26", "t25", "t25", size="w"))), BUNDLE(I("exth t27.w, t25.w, t25.w", E.exth("t27", "t25", "t25", size="w"))),
                     BUNDLE(I("extl t28.w, t31.w, t31.w", E.extl("t28", "t31", "t31", size="w"))), BUNDLE(I("exth t29.w, t31.w, t31.w", E.exth("t29", "t31", "t31", size="w")))]
            for i in range(RT):
                for j in range(4):
                    c = 2 * (4 * i + j)
                    body += [BUNDLE(*[I("mul t%d.fp32, t%d.fp32, t%d.fp32, p7.w" % (r, r, 26 + j), E.vmulf("t%d" % r, "t%d" % r, "t%d" % (26 + j))) for r in (c, c + 1)])]
        elif bscale and SSC:       # single layout: [r30 + 0] = tiles 0 | 1's quads, [r30 + 32] = tiles 2 | 3's; extl / exth (x, x) = the low / high quad twice
            body += [BUNDLE(I("ld t24, [r30+0]", E.vld("t24", "r30", 0)), I("ld t25, [r30+32]", E.vld("t25", "r30", 32)))]
            body += [BUNDLE(I("extl t26.w, t24.w, t24.w", E.extl("t26", "t24", "t24", size="w"))), BUNDLE(I("exth t27.w, t24.w, t24.w", E.exth("t27", "t24", "t24", size="w"))),
                     BUNDLE(I("extl t28.w, t25.w, t25.w", E.extl("t28", "t25", "t25", size="w"))), BUNDLE(I("exth t29.w, t25.w, t25.w", E.exth("t29", "t25", "t25", size="w")))]
            for i in range(RT):
                for j in range(4):
                    c = 2 * (4 * i + j)
                    body += [BUNDLE(*[I("mul t%d.fp32, t%d.fp32, t%d.fp32, p7.w" % (r, r, 26 + j), E.vmulf("t%d" % r, "t%d" % r, "t%d" % (26 + j))) for r in (c, c + 1)])]
        elif bscale and not q8f:   # C tile (i, j) = t(2(4i+j)), t(+1): rows x 4 columns; x the strip's scale vector j (r30 walks the table)
            body += [BUNDLE(I("ld t%d, [r30+%d]" % (24 + j, 32 * j), E.vld("t%d" % (24 + j), "r30", 32 * j))) for j in range(4)]
            for i in range(RT):
                for j in range(4):
                    c = 2 * (4 * i + j)
                    body += [BUNDLE(*[I("mul t%d.fp32, t%d.fp32, t%d.fp32, p7.w" % (r, r, 24 + j), E.vmulf("t%d" % r, "t%d" % r, "t%d" % (24 + j))) for r in (c, c + 1)])]
        if bscale:                 # both: + the C block of the previous slices
            adds = []              # + the C block of the previous slices (GSRAM), six registers at a time through t24-t29
            for c0 in range(0, NC, 6):
                qs = [q for q in range(6) if c0 + q < NC]
                adds += [BUNDLE(*[cload_into(24 + q, c0 + q) for q in qs[p:p + 2]]) for p in range(0, len(qs), 2)]
                adds += [BUNDLE(*[I("add t%d.fp32, t%d.fp32, t%d.fp32, p7.w" % (c0 + q, c0 + q, 24 + q), E.vaddf("t%d" % (c0 + q), "t%d" % (c0 + q), "t%d" % (24 + q))) for q in qs[p:p + 2]]) for p in range(0, len(qs), 2)]
            body += ifz(first_slice, [], adds) + add("r30", "r30", SCLB // ns)
        stores = [BUNDLE(cstore(t)) for t in range(NC)]
        body += ifz(last_slice, stage_c() + (rowmax_strip() if rowmax else []), [] if timing == "noc" else stores) if DMA_DRAIN else stores
        if stamps: body += [CYC("r7")] + acc("r26", "r30", "r7")
        body += add("r15", "r15", 768) + [BUNDLE(I("sub r27, r27, 1", E.sub("r27", "r27", 1)))]
        body.append(BUNDLE(I("cbnz r27, .Lstrip", E.cbnz("r27", -len(body)))))
        tail = []
        if DMA_DRAIN:  # last slice: wait the other buffer's drain (flag 3), drain this row block
            dr = wait_flags(3)
            dr += _mov32("r7", stg) + [BUNDLE(I("add r3, r25, r7", E.add("r3", "r25", "r7"))), BUNDLE(I("add r14, r23, 64", E.add("r14", "r23", 64))),
                                        BUNDLE(I("mov r7, 771", E.mov("r7", 771))),
                                        BUNDLE(I("dma 0, r7, 0, 4, 0, 0, 0, r14, r3, r12", E.dma(E.DMA_DIRECT, "r7", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_INT2EXT, E.DMA_USELESS, "r14", "r3", "r12")))]
            tail = ifz(last_slice, dr, [])
        return blk + body + tail

    # ---- the tern fast path (TF): a list scheduler for straight-line code, and the strip loops ----
    # An op: f(m) -> its encodings (alternatives, first that fits wins; m maps virtual registers "%x" to physical), the
    # registers it reads / writes, its result latency. Virtual registers are allocated in program order (a FIFO free list,
    # freed after the last use), then ops issue greedily by critical path under the dependences (RAW: the producer's latency;
    # WAW: the producer's latency; WAR: a later bundle) and the slots (two arithmetic 0/1, the store or a slot-2 op in 2, loads
    # in 2/3, mov4 / bcast in 3). Latencies: loads 3, fp32 4 (ALU.md: fma), vector integer / moves 2, scalar 1 (interlocked).
    def tz_op(f, rd=(), wr=(), lat=1): return dict(f=f, r=tuple(rd), w=tuple(wr), lat=lat)
    def tz_s(asm, enc, rd=(), wr=()): return tz_op(lambda m, a=asm, e=enc: [(a, e)], rd, wr, 1)
    def tz_ld(d, b_, off): return tz_op(lambda m: [("ld %s, [%s+%d]" % (m(d), b_, off), E.vld(m(d), b_, off))], (b_,), (d,), 3)
    def tz_st(x, b_, off): return tz_op(lambda m: [("st %s, [%s+%d]" % (m(x), b_, off), E.vst(m(x), b_, off))], (x, b_), (), 1)
    TZF = {"fma": E.vfmaf, "mul": E.vmulf, "sub": E.vsubf, "add": E.vaddf}
    def tz_f(op, d, a, b_):        # fp32 d = a op b (fma: d += a x b)
        return tz_op(lambda m: [("%s %s.fp32, %s.fp32, %s.fp32, p7.w" % (op, m(d), m(a), m(b_)), TZF[op](m(d), m(a), m(b_)))],
                     (d, a, b_) if op == "fma" else (a, b_), (d,), 4)
    def tz_ext(op, d, a): return tz_op(lambda m: [("%s %s.w, %s.w, %s.w" % (op, m(d), m(a), m(a)), getattr(E, op)(m(d), m(a), m(a), size="w"))], (a,), (d,), 2)
    def tz_zero(d): return tz_op(lambda m: [("sub %s.w, %s.w, %s.w" % (d, d, d), E.vsub(d, d, d, size="w")), ("mov4 %s, 0" % d, E.mov4(d, 0))], (d,), (d,), 2)
    def tz_fit(encs):              # most-constrained first: the order enc.bundle assigns slots in; None if they do not fit
        order = sorted(encs, key=lambda e: (len(e[1][0]), e[1][0])); used = set()
        for e in order:
            sl_ = next((x for x in e[1][0] if x not in used), None)
            if sl_ is None: return None
            used.add(sl_)
        return order
    def tz_sched(ops, pool=(), ready=None, prio="height"):     # prio: "height" (critical path first) or "order" (program order)
        last = {}
        for i, o in enumerate(ops):
            for x in o["r"] + o["w"]:
                if x.startswith("%"): last[x] = i
        free, amap = list(pool), {}
        for i, o in enumerate(ops):
            for x in o["w"]:
                if x.startswith("%") and x not in amap: amap[x] = free.pop(0)
            for x in dict.fromkeys(o["r"] + o["w"]):
                if x.startswith("%") and last[x] == i: free.append(amap[x])
        def m(x): return amap.get(x, x)
        R = [tuple(m(x) for x in o["r"]) for o in ops]; W = [tuple(m(x) for x in o["w"]) for o in ops]
        n = len(ops); deps = [[] for _ in range(n)]; lastw, rds = {}, {}
        for i in range(n):
            deps[i] += [(lastw[x], ops[lastw[x]]["lat"]) for x in R[i] + W[i] if x in lastw]
            deps[i] += [(j, 1) for x in W[i] for j in rds.get(x, []) if j != i]
            for x in R[i]: rds.setdefault(x, []).append(i)
            for x in W[i]: lastw[x] = i; rds[x] = []
        succ = [[] for _ in range(n)]
        for i in range(n):
            for j, d in deps[i]: succ[j].append((i, d))
        H = [0] * n
        for i in reversed(range(n)): H[i] = max([ops[i]["lat"]] + [H[k] + d for k, d in succ[i]])
        ready = ready or {}; issue = [None] * n; left = set(range(n)); out = []; cyc = 0
        while left:
            cand = sorted((i for i in left if all(issue[j] is not None and issue[j] + d <= cyc for j, d in deps[i])
                           and all(cyc >= ready.get(x, 0) for x in R[i] + W[i])), key=lambda i: (-H[i], i) if prio == "height" else i)
            chosen = []
            for i in cand:
                for alt in ops[i]["f"](m):
                    if tz_fit([e for _, e in chosen] + [alt]): chosen.append((i, alt)); break
            for i, _ in chosen: issue[i] = cyc; left.discard(i)
            out.append(BUNDLE(*tz_fit([e for _, e in chosen]))); cyc += 1
            assert cyc < 4096, "tz_sched: no progress"
        return out, issue

    def tz_vzero(d):               # d = 0 (any lanes): mov4 (slot 3) or sub (slots 0/1); the old value is not read
        return tz_op(lambda m: [("mov4 %s, 0" % m(d), E.mov4(m(d), 0)), ("sub %s.w, %s.w, %s.w" % (m(d), m(d), m(d)), E.vsub(m(d), m(d), m(d), size="w"))], (), (d,), 2)
    def tz_fm(op, d, a, b_):       # fmae / fmao: d.fp32 += a.fp16 x b.fp16 over the even / odd fp16 lanes (fp32 lane i <- fp16 lane 2i / 2i + 1)
        f = E.vfmae if op == "fmae" else E.vfmao
        return tz_op(lambda m: [("%s %s.fp32, %s.fp16, %s.fp16, p7.h" % (op, m(d), m(a), m(b_)), f(m(d), m(a), m(b_)))], (d, a, b_), (d,), 4)
    def tz_bc(d, r):               # bcast d.w, r (slot 3)
        return tz_op(lambda m: [("bcast %s.w, %s" % (m(d), r), E.bcast(m(d), r))], (r,), (d,), 2)

    def strips_tf(half_reg, stg, scl):
        """TF (tern, rows mode, rt <= 2): the group's ns strips against the A slice in `half_reg`; r15 walks the group's
        LSRAM accumulators (TCS B a strip), r30 the slice's scale table, r14 the row sums S'. Four strip loops, one per slice
        position (first: C x s -> acc; middle: acc + C x s -> acc, one `fma` (a single rounding); last: acc + C x s -> staging;
        the only slice: C x s -> staging), chosen once a group step; staging rows past the rt tiles are zeroed once at the
        start."""
        bo = [tz_s("add r13, %s, 0" % scl, E.add("r13", scl, 0), (scl,), ("r13",)), tz_s("mov r27, %d" % ns, E.mov("r27", ns), (), ("r27",)),
              tz_s("mov r8, %d" % BH, E.mov("r8", BH), (), ("r8",)), tz_s("add r30, %s, r8" % scl, E.add("r30", scl, "r8"), (scl, "r8"), ("r30",)),
              tz_s("mov r10, %d" % stg, E.mov("r10", stg), (), ("r10",)), tz_s("add r3, r25, r10", E.add("r3", "r25", "r10"), ("r25", "r10"), ("r3",)),
              tz_s("mov r12, %d" % SPO, E.mov("r12", SPO), (), ("r12",)), tz_s("add r14, r25, r12", E.add("r14", "r25", "r12"), ("r25", "r12"), ("r14",)),
              tz_s("mov r7, 0", E.mov("r7", 0), (), ("r7",)), tz_s("movh r7, %d" % (0xBE80 - 0x10000), E.movh("r7", 0xBE80 - 0x10000), ("r7",), ("r7",)),
              tz_op(lambda m: [("bcast t31.w, r7", E.bcast("t31", "r7"))], ("r7",), ("t31",), 2)]   # -0.25 (t31: the k-loop leaves it alone at rt <= 2)
        if F16S:                   # fp16 2^3 pairs in r12 (after its use for r14): the f16 table's widening constant, bcast per strip.
            # r7 does not survive blk (it is ifz's test register); r12 does, from blk to the drain
            bo += [tz_s("mov r12, %d" % (0x4800), E.mov("r12", 0x4800), (), ("r12",)), tz_s("movh r12, %d" % (0x4800), E.movh("r12", 0x4800), ("r12",), ("r12",))]
        assert BH < 32768 and stg < 32768 and SPO < 32768
        blk = tz_sched(bo)[0]
        # The k-loop expands as it goes (no fp16 panels): r13 walks the strip's codes (32 B a k-step pair, a strip's 16 pairs
        # contiguous, so the loop leaves r13 at the next strip's), r8 the A slice. A code unit X becomes the pair's 8 B tiles:
        # Y_0 = X, Y_(j+1) = Y_j << 2 (byte lanes, `lsl .b`, slot 2), tile j of k-step 2p / 2p + 1 = `mul` / `mulh` (Y_j.ub x 1)
        # (or `zipl` against zero), the fp16 subnormal u(Y_j) x 2^-24 -- the bytes the panel expand wrote. A pair's expand is
        # four 1-cycle bundles {lsl ; mul, mulh} beside its four (rt 1) / eight (rt 2) 2-cycle mma bundles, software-pipelined
        # one pair ahead. rt 1: two B sets (t8-15 / t16-23), A t24 / t25 (k-step a / b), X t26, Y t27 / t28, 1 t29, 0 t30.
        # rt 2: ONE B set t16-23, each tile rewritten right after its two mma bundles (the next pair's), A t24-27 (row tile 0 / 1 of
        # k-step a, then of b), Y t28 / t29 (the code is loaded into t28), 1 t30. t31 (-0.25) is left alone at both.
        TONE = "t29" if RT == 1 else "t30"
        def k_mma(c, a, b_): return I("mma t%d.fp32, t%d.fp32, %s.fp16, %s.fp16" % (c, c + 1, a, b_), E.mma("t%d" % c, a, b_))
        def k_mul(d, y, hi): return I("%s %s.h, %s.ub, %s.b, p7.b" % ("mulh" if hi else "mul", d, y, TONE), E.vmulb(d, y, TONE, ua=True, high=hi))
        def k_lsl(d, a): return I("lsl %s.b, %s.b, 2" % (d, a), E.vlsli(d, a, 2, size="b"))
        def k_ld(d, b_, off): assert 0 <= off < 512; return I("ld %s, [%s+%d]" % (d, b_, off), E.vld(d, b_, off))
        def k_add(r, x): return I("add %s, %s, %d" % (r, r, x), E.add(r, r, x))
        def KB(*ops): return BUNDLE(*sorted(ops, key=lambda e: len(e[1][0])))     # slot-2-only ops placed first
        def sets(base): return {"a": ["t%d" % (base + j) for j in range(4)], "b": ["t%d" % (base + 4 + j) for j in range(4)]}
        if RT == 1:
            S_ = [sets(8), sets(16)]; Aa, Ab, X, ya, yb = "t24", "t25", "t26", "t27", "t28"
            def part(p):
                cur, nxt = S_[p], S_[1 - p]; oA = 2 * AST * p
                return [KB(k_mma(0, Aa, cur["a"][0]), k_mma(2, Aa, cur["a"][1]), k_ld(Ab, "r8", oA + AST)),
                        KB(k_lsl(ya, X), k_mul(nxt["a"][0], X, False), k_mul(nxt["b"][0], X, True)),
                        KB(k_mma(4, Aa, cur["a"][2]), k_mma(6, Aa, cur["a"][3]), k_ld(X, "r13", 64 if p == 0 else 32)),
                        KB(k_lsl(yb, ya), k_mul(nxt["a"][1], ya, False), k_mul(nxt["b"][1], ya, True)),
                        KB(k_mma(0, Ab, cur["b"][0]), k_mma(2, Ab, cur["b"][1]), k_ld(Aa, "r8", oA + 2 * AST)),
                        KB(k_lsl(ya, yb), k_mul(nxt["a"][2], yb, False), k_mul(nxt["b"][2], yb, True)),
                        KB(k_mma(4, Ab, cur["b"][2]), k_mma(6, Ab, cur["b"][3])),
                        KB(I("zipl %s.b, %s.b, t30.b" % (nxt["a"][3], ya), E.zipl(nxt["a"][3], ya, "t30")), k_mul(nxt["b"][3], ya, True),
                           k_add("r13", 64) if p == 0 else k_add("r8", 4 * AST))]
            if tkloop == "mmfirst":
                # A pair's mma bundles first (M0-M3, reading `cur`), then five expand bundles writing `nxt` from X (the next pair's
                # codes): Y_j = X << 2j by independent byte shifts (no chain through ya / yb, so no 2-cycle wait between expand
                # bundles), t30 the third Y (the zipl and its zero register are gone: all eight tiles by mul / mulh). The fifth
                # bundle is needed because only tiles a0 / b0 come from X itself: Y_1 is ready two cycles after its shift.
                #   M0 {mma a0, a1 ; ld Ab}  M1 {mma a2, a3}  M2 {mma b0, b1 ; ld Aa'}  M3 {mma b2, b3}
                #   E0 {lsl y1 ; mul a0', mulh b0' (X)}  E1 {lsl y2 ; add r13 / r8}  E2 {lsl y3 ; mul a1', mulh b1' (y1)}
                #   E3 {mul a2', mulh b2' (y2) ; ld X'}  E4 {mul a3', mulh b3' (y3)}
                # Every `nxt` tile is rewritten >= 8 cycles after its last mma read (the previous pair's M group, then this pair's
                # four mma bundles), and read >= 2 cycles after its rewrite (a3' at E4 -> the next M1). X' (the pair after next)
                # loads at E3, after its last read (E2's shift), >= 8 cycles before the next E0. A: Ab at M0 (read from M2),
                # Aa' at M2 (after M1's last read of Aa).
                y1, y2, y3 = "t27", "t28", "t30"
                def k_lsln(d, a, n): return I("lsl %s.b, %s.b, %d" % (d, a, n), E.vlsli(d, a, n, size="b"))
                def part(p):
                    cur, nxt = S_[p], S_[1 - p]; oA = 2 * AST * p
                    return [KB(k_mma(0, Aa, cur["a"][0]), k_mma(2, Aa, cur["a"][1]), k_ld(Ab, "r8", oA + AST)),
                            KB(k_mma(4, Aa, cur["a"][2]), k_mma(6, Aa, cur["a"][3])),
                            KB(k_mma(0, Ab, cur["b"][0]), k_mma(2, Ab, cur["b"][1]), k_ld(Aa, "r8", oA + 2 * AST)),
                            KB(k_mma(4, Ab, cur["b"][2]), k_mma(6, Ab, cur["b"][3])),
                            KB(k_lsln(y1, X, 2), k_mul(nxt["a"][0], X, False), k_mul(nxt["b"][0], X, True)),
                            KB(k_lsln(y2, X, 4), k_add("r13", 64) if p == 0 else k_add("r8", 4 * AST)),
                            KB(k_lsln(y3, X, 6), k_mul(nxt["a"][1], y1, False), k_mul(nxt["b"][1], y1, True)),
                            KB(k_mul(nxt["a"][2], y2, False), k_mul(nxt["b"][2], y2, True), k_ld(X, "r13", 0 if p == 0 else 32)),
                            KB(k_mul(nxt["a"][3], y3, False), k_mul(nxt["b"][3], y3, True))]
            kb = part(0) + part(1)
            ent = {"S": S_[0], "A": [(Aa, 0)], "X": X}
        else:
            S1 = sets(16); A0a, A1a, A0b, A1b, ya, yb = "t24", "t25", "t26", "t27", "t28", "t29"
            def Ma(j): return [k_mma(2 * j, A0a, S1["a"][j]), k_mma(2 * (4 + j), A1a, S1["a"][j])]
            def Mb(j): return [k_mma(2 * j, A0b, S1["b"][j]), k_mma(2 * (4 + j), A1b, S1["b"][j])]
            def Ex(j, y, ny): return ([k_lsl(ny, y)] if j < 3 else []) + [k_mul(S1["a"][j], y, False), k_mul(S1["b"][j], y, True)]
            def part(p):
                kA, kB = 2 * p, 2 * p + 1
                # every load >= 3 bundles before its use, every lsl / mul >= 2 (counted in bundles, not cycles); one load a bundle
                return [KB(*Ma(0), k_ld(A1b, "r8", kB * AST + 32)), KB(*Ma(1)), KB(*Ma(2)), KB(*Mb(0)), KB(*Ex(0, ya, yb)),
                        KB(*Ma(3)), KB(*Mb(1), k_ld(A0a, "r8", (kA + 2) * AST)), KB(*Ex(1, yb, ya)),
                        KB(*Mb(2), k_ld(A1a, "r8", (kA + 2) * AST + 32)), KB(*Ex(2, ya, yb)),
                        KB(*Mb(3), k_ld(ya, "r13", 32 * (2 + p))), KB(*Ex(3, yb, None), k_ld(A0b, "r8", (kB + 2) * AST))]
            kb = part(0) + part(1) + [KB(k_add("r8", 4 * AST), k_add("r13", 64))]
            if tkloop2 != "orig":
                # A pair's eight mma bundles first (Ma0-3: row tiles 0 / 1 of k-step a x tile j, then Mb0-3: k-step b), then its
                # expand bundles writing the next pair's tiles from X (in ya): Y_1 = X << 2 into yb, Y_2 = X << 4 in place in ya,
                # Y_3 = Y_1 << 4 in place in yb (a shift reads its source in the bundle that rewrites it: the bundle's reads come
                # first). The first expand bundle after the mmas costs 3 cycles on the board (ALU.md 7), the rest 1.
                #  m8e5: M0 {Ma0 ; ld A1b}  M1-M3 {Ma1-3}  M4 {Mb0 ; ld A0a'}  M5 {Mb1 ; ld A1a'}  M6 M7 {Mb2, Mb3}
                #        E0 {lsl yb = ya << 2 ; mul a0', mulh b0' (ya) ; ld A0b'}  E1 {lsl ya <<= 4 ; (part 1: add r8 / r13)}
                #        E2 {mul a1', mulh b1' (yb) ; lsl yb <<= 4}  E3 {mul a2', mulh b2' (ya)}  E4 {mul a3', mulh b3' (yb) ; ld X''}
                #  m8e4: the first shift beside the last mma bundle (M7 {Mb3 ; lsl yb = ya << 2}: a slot-2 op beside {mma, mma}),
                #        then four expand bundles E0 {mul a0', mulh b0' (ya) ; lsl ya <<= 4 ; ld A0b'}  E1 {mul a1', mulh b1' (yb) ;
                #        lsl yb <<= 4}  E2 {mul a2', mulh b2' (ya)}  E3 {mul a3', mulh b3' (yb) ; ld X''}, the pointer adds in a bundle
                #        of their own after the second pair.
                # Every tile is rewritten >= 7 cycles after its last mma read and read >= 2 cycles after its rewrite; A0a' / A1a'
                # load after Ma3 (their last read), A0b' after Mb3, X'' after its last read; each C accumulates k-step a then b.
                def k_lsln(d, a, n): return I("lsl %s.b, %s.b, %d" % (d, a, n), E.vlsli(d, a, n, size="b"))
                def k_mulp(j, y): return [k_mul(S1["a"][j], y, False), k_mul(S1["b"][j], y, True)]
                def part(p):
                    kA, kB = 2 * p, 2 * p + 1; rb = 64 if p == 1 else 0
                    M = [KB(*Ma(0), k_ld(A1b, "r8", kB * AST + 32)), KB(*Ma(1)), KB(*Ma(2)), KB(*Ma(3)),
                         KB(*Mb(0), k_ld(A0a, "r8", (kA + 2) * AST)), KB(*Mb(1), k_ld(A1a, "r8", (kA + 2) * AST + 32)), KB(*Mb(2))]
                    if tkloop2 == "m8e5":
                        return M + [KB(*Mb(3)),
                                    KB(k_lsln(yb, ya, 2), *k_mulp(0, ya), k_ld(A0b, "r8", (kB + 2) * AST)),
                                    KB(k_lsln(ya, ya, 4), *([k_add("r8", 4 * AST), k_add("r13", 64)] if p == 1 else [])),
                                    KB(k_lsln(yb, yb, 4), *k_mulp(1, yb)), KB(*k_mulp(2, ya)),
                                    KB(*k_mulp(3, yb), k_ld(ya, "r13", 32 * (2 + p) - rb))]
                    return M + [KB(*Mb(3), k_lsln(yb, ya, 2)),
                                KB(k_lsln(ya, ya, 4), *k_mulp(0, ya), k_ld(A0b, "r8", (kB + 2) * AST)),
                                KB(k_lsln(yb, yb, 4), *k_mulp(1, yb)), KB(*k_mulp(2, ya)),
                                KB(*k_mulp(3, yb), k_ld(ya, "r13", 32 * (2 + p)))]
                kb = part(0) + part(1) + ([] if tkloop2 == "m8e5" else [KB(k_add("r8", 4 * AST), k_add("r13", 64))])
            ent = {"S": S1, "A": [(A0a, 0), (A1a, 32), (A0b, AST)], "X": ya}
            if tkloop2 == "m8u":
                # m8u: no ALU expand at all -- the tiles are widened by uxtl / uxth (slot 2: the low / high 16 bytes of Y_j zero-extended
                # to halfwords, the fp16 subnormals mul / mulh x 1 made), and 8 of a pair's 11 slot-2 ops ride beside its {mma, mma}
                # bundles. The mma bundles alternate k-step a / b per tile (Ma0 Mb0 Ma1 Mb1 ...: each C still accumulates a then b), so
                # tiles a_j / b_j are both read by bundle 2j + 1 and rewritten from bundle 2j + 2 on, and X / Y_1 / Y_2 (t28 / t29 / t30:
                # no ones register) suffice:
                #   M0 {Ma0 ; lsl Y1 = X << 2}  M1 {Mb0 ; lsl Y2 = X << 4}  M2 {Ma1 ; uxtl a0' (X)}  M3 {Mb1 ; uxth b0' (X)}
                #   M4 {Ma2 ; uxtl a1' (Y1) ; ld X''}  M5 {Mb2 ; uxth b1' (Y1)}  M6 {Ma3 ; uxtl a2' (Y2)}  M7 {Mb3 ; uxth b2' (Y2) ; ld A0a'}
                #   E0 {lsl Y3 = Y1 << 4 (in place) ; ld A1a'}  E1 {uxtl a3' (Y3) ; ld A0b'}  E2 {uxth b3' (Y3) ; ld A1b'}
                # (the pointer adds in a bundle of their own after the second pair). Every tile is rewritten >= 4 cycles after its last
                # mma read (the slot-2 op two mma bundles later) and read >= 6 cycles after it; A' load after their last read (M6 / M7)
                # and >= 3 cycles before their first; X'' after its last read (M3). The entry state: all four A, the first pair's tiles
                # (expanded as before, by mul / mulh against the ones in t30, which the k-loop then overwrites as Y_2), the second pair's code.
                Y2 = "t30"
                def k_ux(d, y, hi): return I("%s %s.h, %s.b" % ("uxth" if hi else "uxtl", d, y), (E.uxth_hb if hi else E.uxtl_hb)(d, y))
                def MM(j, h): A0, A1, t = (A0b, A1b, S1["b"][j]) if h else (A0a, A1a, S1["a"][j]); return [k_mma(2 * j, A0, t), k_mma(2 * (4 + j), A1, t)]
                def part(p):
                    kA, kB = 2 * p, 2 * p + 1
                    return [KB(*MM(0, 0), k_lsln(yb, ya, 2)), KB(*MM(0, 1), k_lsln(Y2, ya, 4)),
                            KB(*MM(1, 0), k_ux(S1["a"][0], ya, False)), KB(*MM(1, 1), k_ux(S1["b"][0], ya, True)),
                            KB(*MM(2, 0), k_ux(S1["a"][1], yb, False), k_ld(ya, "r13", 32 * (2 + p))), KB(*MM(2, 1), k_ux(S1["b"][1], yb, True)),
                            KB(*MM(3, 0), k_ux(S1["a"][2], Y2, False)), KB(*MM(3, 1), k_ux(S1["b"][2], Y2, True), k_ld(A0a, "r8", (kA + 2) * AST)),
                            KB(k_lsln(yb, yb, 4), k_ld(A1a, "r8", (kA + 2) * AST + 32)), KB(k_ux(S1["a"][3], yb, False), k_ld(A0b, "r8", (kB + 2) * AST)),
                            KB(k_ux(S1["b"][3], yb, True), k_ld(A1b, "r8", (kB + 2) * AST + 32))]
                kb = part(0) + part(1) + [KB(k_add("r8", 4 * AST), k_add("r13", 64))]
                ent = {"S": S1, "A": [(A0a, 0), (A1a, 32), (A0b, AST), (A1b, AST + 32)], "X": ya}
            if tkloop2 == "m8w":
                # m8w: one expand bundle a pair (19 cycles: 8 mma bundles + the first vector bundle after them). t31 (-0.25) is freed for
                # a ones register (the epilogue bcasts -0.25 from r7, set before the strip loop), so a pair's 11 expand ops fit 8
                # slot-2 slots beside the mmas + one bundle {uxtl ; mul, mulh}. The mma bundles are paired so that every C accumulates
                # k-step a before b, A0a / A1a are done early and A1b is first read at bundle 2:
                #   P0 {C00a, C10a x a0 ; uxth b3 (Y3, the previous pair's) ; ld A1b}  P1 {C00b x b0, C01a x a1 ; lsl Y1 = X << 2}
                #   P2 {C10b x b0, C11a x a1 ; uxtl a0' (X)}  P3 {C01b x b1, C02a x a2 ; lsl Y2 = X << 4}  P4 {C11b x b1, C12a x a2 ; uxth b0' (X)}
                #   P5 {C02b x b2, C03a x a3 ; uxtl a1' (Y1) ; ld X''}  P6 {C12b x b2, C13a x a3 ; uxth b1' (Y1) ; ld A0a'}
                #   P7 {C03b, C13b x b3 ; lsl Y3 = Y1 << 4 (in place) ; ld A1a'}  E {uxtl a2' (Y2) ; mulh b2' (Y2), mul a3' (Y3) ; ld A0b'}
                # X t28, Y1 / Y3 t29, Y2 t30, ones t31. Every tile is rewritten >= 2 bundles (4 cycles) after its last mma read and
                # before its next read (b3' in the next pair's P0, from Y3 kept in t29); the A' loads follow their last read and are
                # >= 3 cycles ahead of their first. Entry state: the first pair's tiles, Y3 of its code in t29, the second pair's code
                # in t28, A0a / A1a / A0b, the ones in t31.
                TONE = "t31"
                def k_ux(d, y, hi): return I("%s %s.h, %s.b" % ("uxth" if hi else "uxtl", d, y), (E.uxth_hb if hi else E.uxtl_hb)(d, y))
                AR = {"a": (A0a, A1a), "b": (A0b, A1b)}
                def mm(r, j, k): return k_mma(2 * (4 * r + j), AR[k][r], S1[k][j])
                def part(p):
                    kA, kB = 2 * p, 2 * p + 1; X, Y13, Y2 = "t28", "t29", "t30"
                    return [KB(mm(0, 0, "a"), mm(1, 0, "a"), k_ux(S1["b"][3], Y13, True), k_ld(A1b, "r8", kB * AST + 32)),
                            KB(mm(0, 0, "b"), mm(0, 1, "a"), k_lsln(Y13, X, 2)),
                            KB(mm(1, 0, "b"), mm(1, 1, "a"), k_ux(S1["a"][0], X, False)),
                            KB(mm(0, 1, "b"), mm(0, 2, "a"), k_lsln(Y2, X, 4)),
                            KB(mm(1, 1, "b"), mm(1, 2, "a"), k_ux(S1["b"][0], X, True)),
                            KB(mm(0, 2, "b"), mm(0, 3, "a"), k_ux(S1["a"][1], Y13, False), k_ld(X, "r13", 32 * (2 + p))),
                            KB(mm(1, 2, "b"), mm(1, 3, "a"), k_ux(S1["b"][1], Y13, True), k_ld(A0a, "r8", (kA + 2) * AST)),
                            KB(mm(0, 3, "b"), mm(1, 3, "b"), k_lsln(Y13, Y13, 4), k_ld(A1a, "r8", (kA + 2) * AST + 32)),
                            KB(k_ux(S1["a"][2], Y2, False), k_mul(S1["b"][2], Y2, True), k_mul(S1["a"][3], Y13, False), k_ld(A0b, "r8", (kB + 2) * AST))]
                kb = part(0) + part(1) + [KB(k_add("r8", 4 * AST), k_add("r13", 64))]
        U = RT == 2 and tkloop2 == "m8u"
        W = RT == 2 and tkloop2 == "m8w"
        def pre_ops():             # r8, the k-loop's entry state (A, the first pair expanded, the second's code, the constants), the
            # loop count, and C = the starting offsets (sprep's table at r14, a load a register). Strip 0's run before the strip
            # loop; strip s + 1's are scheduled into strip s's epilogue.
            po = [tz_s("add r8, %s, 0" % half_reg, E.add("r8", half_reg, 0), (half_reg,), ("r8",))]
            po += [tz_ld(r, "r8", o) for r, o in ent["A"]]
            po += [tz_op(lambda m, d=d, v=v: [("mov4 %s, %d" % (d, v), E.mov4(d, v))], (), (d,), 2) for d, v in ((TONE, 1),) + ((("t30", 0),) if RT == 1 else ())]
            if W:              # m8w: Y_j = Y_0 << 2j from the code in t30; Y_1 / Y_3 in t29 (Y_3 stays: the loop's P0 makes b3 from it), Y_2 in t28
                for j, y in enumerate(("t30", "t29", "t28", "t29")):
                    if j == 0: po.append(tz_ld(y, "r13", 0))
                    else: po.append(tz_op(lambda m, j=j, y=y: [("lsl %s.b, t30.b, %d" % (y, 2 * j), E.vlsli(y, "t30", 2 * j, size="b"))], ("t30",), (y,), 2))
                    for hi, d in ((False, ent["S"]["a"][j]), (True, ent["S"]["b"][j])):
                        po.append(tz_op(lambda m, y=y, hi=hi, d=d: [("%s %s.h, %s.ub, %s.b, p7.b" % ("mulh" if hi else "mul", d, y, TONE), E.vmulb(d, y, TONE, ua=True, high=hi))],
                                        (y, TONE), (d,), 2))
                po.append(tz_ld(ent["X"], "r13", 32))
                return po + [tz_s("mov r10, %d" % (KX * ks // kw - 1), E.mov("r10", KX * ks // kw - 1), (), ("r10",))] + [tz_ld("t%d" % t, "r14", 32 * t) for t in range(NC)]
            ys = [("t27", "t28") if RT == 1 else ("t28", "t29") if U else ("t27", "t29")][0] * 2     # Y_0 .. Y_3 alternating (fixed: no virtual among the entry state; m8u: t27 is A1b, t28 the code loaded after)
            po.append(tz_ld(ys[0], "r13", 0))
            for j in range(4):
                if j < 3: po.append(tz_op(lambda m, j=j: [("lsl %s.b, %s.b, 2" % (m(ys[j + 1]), m(ys[j])), E.vlsli(m(ys[j + 1]), m(ys[j]), 2, size="b"))], (ys[j],), (ys[j + 1],), 2))
                for hi, d in ((False, ent["S"]["a"][j]), (True, ent["S"]["b"][j])):
                    po.append(tz_op(lambda m, j=j, hi=hi, d=d: [("%s %s.h, %s.ub, %s.b, p7.b" % ("mulh" if hi else "mul", d, m(ys[j]), TONE), E.vmulb(d, m(ys[j]), TONE, ua=True, high=hi))],
                                    (ys[j], TONE), (d,), 2))
            po.append(tz_ld(ent["X"], "r13", 32))
            return po + [tz_s("mov r10, %d" % (KX * ks // kw - 1), E.mov("r10", KX * ks // kw - 1), (), ("r10",))] + [tz_ld("t%d" % t, "r14", 32 * t) for t in range(NC)]
        def fit_loads(b_, iss, ops, after):   # pad so every vector result (loads 3, vector 2) is ready at the loop (`after` bundles precede it)
            while max(c + o["lat"] for c, o in zip(iss, ops) if o["lat"] > 1) > len(b_) + after: b_.append(BUNDLE())
            return b_
        po0 = pre_ops() + ([tz_s("mov r7, 0", E.mov("r7", 0), (), ("r7",)), tz_s("movh r7, %d" % (0xBE80 - 0x10000), E.movh("r7", 0xBE80 - 0x10000), ("r7",), ("r7",))] if W else [])
        prew0 = {x for o in po0 for x in o["w"]}     # (m8w: r7 = -0.25 for the strips' epilogues; nothing in the strip loop writes r7)
        pre = fit_loads(*tz_sched(po0, pool=[r for r in ("t%d" % r for r in range(NC, 31)) if r not in prew0]), po0, 1)
        SC = ["%%S%d" % j for j in range(4)]
        def epi(v):
            eo = []
            if F16S:                   # fp16 table: even lanes = tiles 0-1's columns, odd = tiles 2-3's; fp32 = fp16 x 2^15 (exact), then extl / exth
                eo += [tz_bc("%K", "r12"), tz_ld("%h", "r30", 0), tz_vzero("%fe"), tz_vzero("%fo"), tz_fm("fmae", "%fe", "%h", "%K"), tz_fm("fmao", "%fo", "%h", "%K"),
                       tz_ext("extl", SC[0], "%fe"), tz_ext("exth", SC[1], "%fe"), tz_ext("extl", SC[2], "%fo"), tz_ext("exth", SC[3], "%fo")]
            else:
                eo += [tz_ld("%q0", "r30", 0), tz_ld("%q1", "r30", 32), tz_ext("extl", SC[0], "%q0"), tz_ext("exth", SC[1], "%q0"),   # the single
                       tz_ext("extl", SC[2], "%q1"), tz_ext("exth", SC[3], "%q1")]                                                     # layout's lanes
            eo.append(tz_s("add r30, r30, %d" % (SCLB // ns), E.add("r30", "r30", SCLB // ns), ("r30",), ("r30",)))
            QM = "t31"
            if W: QM = "%M"; eo.append(tz_bc(QM, "r7"))     # m8w: t31 is the k-loop's ones; -0.25 from r7
            for i in range(RT):        # per register, in column-tile order: C(j) -= C(j + 1) / 4 (j < 3: its source still uncorrected),
                for j in range(4):     # (-S' rides in the k-loop's starting offsets), x the column scale, + the previous slices'
                    for h in (0, 1):   # accumulator, store
                        t = 2 * (4 * i + j) + h; x = "t%d" % t
                        if j < 3: eo.append(tz_f("fma", x, "t%d" % (t + 2), QM))
                        if v in ("M", "L"): eo += [tz_ld("%%A%d" % t, "r15", 32 * t), tz_f("fma", "%%A%d" % t, x, SC[j])]; x = "%%A%d" % t   # acc += C x s, fused
                        else: eo.append(tz_f("mul", x, x, SC[j]))
                        eo.append(tz_st(x, "r3" if v in ("L", "FL") else "r15", 32 * t))
            eo.append(tz_s("add r15, r15, %d" % TCS, E.add("r15", "r15", TCS), ("r15",), ("r15",)))
            if v in ("L", "FL"): eo.append(tz_s("add r3, r3, 768", E.add("r3", "r3", 768), ("r3",), ("r3",)))
            eo.append(tz_s("sub r27, r27, 1", E.sub("r27", "r27", 1), ("r27",), ("r27",)))
            eo += pre_ops()
            prew = {x for o in pre_ops() for x in o["w"]}       # the pool: registers the next strip's entry state leaves alone first
            pool = sorted(["t%d" % r for r in range(NC, 31)], key=lambda r: r in prew)
            return min((fit_loads(*tz_sched(eo, pool=pool, ready={"t%d" % t: 3 for t in range(NC)}, prio=pr), eo, 2)
                        for pr in ("height", "order")), key=len)
        def loop(v):               # .Lstrip = the `loop` bundle (strip 0's operands and zeroing precede the loop)
            lp = [BUNDLE(I("loop r10", E.loop("r10")))] + kb + [BUNDLE(I("loopend", E.loopend()))] + epi(v)
            return pre + lp + [BUNDLE(I("cbnz r27, .Lstrip", E.cbnz("r27", -len(lp))))]
        body = ifz(first_slice, ifz(last_slice, loop("FL"), loop("F")), ifz(last_slice, loop("L"), loop("M")))
        dr = wait_flags(3)             # last slice: the other buffer's drain; this group's C row in DDR -> r12; drain
        if PF: dr += [BUNDLE(I("sub r12, r16, r28", E.sub("r12", "r16", "r28")))] + _mov32("r7", c_stride) + \
                     [BUNDLE(I("mul r12, r12, r7", E.mul("r12", "r12", "r7"))), BUNDLE(I("add r12, r12, r2", E.add("r12", "r12", "r2")))]
        else: dr += [BUNDLE(I("add r12, r2, 0", E.add("r12", "r2", 0)))]
        dr += _mov32("r7", stg) + [BUNDLE(I("add r3, r25, r7", E.add("r3", "r25", "r7"))), BUNDLE(I("add r14, r23, 64", E.add("r14", "r23", 64))),
                                    BUNDLE(I("mov r7, 771", E.mov("r7", 771))),
                                    BUNDLE(I("dma 0, r7, 0, 4, 0, 0, 0, r14, r3, r12", E.dma(E.DMA_DIRECT, "r7", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_INT2EXT, E.DMA_USELESS, "r14", "r3", "r12")))]
        return blk + body + ifz(last_slice, dr, [])

    def prefetch(half_reg, flag):
        req = sync(flag) + [dma_in("r7", "r23", half_reg, "r17")] + add("r17", "r17", a_rb_stride)
        return [BUNDLE(I("cbnz r16, .Lpf", E.cbnz("r16", 2))), BUNDLE(I("b .Lskip", E.b(len(req) + 1)))] + req

    body = [BUNDLE(I("cbnz r4, .Lgo", E.cbnz("r4", 2))), BUNDLE(I("b .Ldone", E.b(0)))]
    skip = 1
    body += [BUNDLE(I("add r29, r7, 0", E.add("r29", "r7", 0)))]  # GSRAM C base, before r7 is scratch
    body += [BUNDLE(I("add r31, r8, 0", E.add("r31", "r8", 0)))] if False else [BUNDLE(I("add r19, r8, 0", E.add("r19", "r8", 0)))]  # group count
    if stamps: body += [BUNDLE(I("add r28, r9, 0", E.add("r28", "r9", 0))), CYC("r30"), BUNDLE(I("st r30, [r28+0]", E.st("r30", "r28", 0))), BUNDLE(I("mov r20, 0", E.mov("r20", 0))), BUNDLE(I("mov r21, 0", E.mov("r21", 0))), BUNDLE(I("mov r26, 0", E.mov("r26", 0)))]
    if rowmax: body += _mov32("r7", 0xFF800000) + [BUNDLE(I("bcast t31.w, r7", E.bcast("t31", "r7")))]  # t31 = -inf
    body += _lsram_base("r12") + add("r25", "r12", BREG) + add("r9", "r12", A0) + add("r11", "r12", A1) + add("r23", "r3", 0) + add("r24", "r3", 32)
    if bscale and not PF:          # the codes + scales buffer and its descriptor (one request a slice, flag 0)
        body += add("r20", "r12", CB) + add("r26", "r3", 128)
    if PF:                         # buffers (r9 / r11 A, r20 / r21 B: current / next), r5 = B bytes a group, the first step's requests
        body += add("r9", "r12", PA0) + add("r11", "r12", PA1) + add("r20", "r12", PCB0) + add("r21", "r12", PCB1) + add("r26", "r3", 128)
        body += _mov32("r7", BH + SCLB) + [BUNDLE(I("mul r5, r4, r7", E.mul("r5", "r4", "r7")))]
        body += add("r24", "r1", 0) + add("r22", "r1", 0) + add("r17", "r0", 0)
        body += dmaq([dma_in("zero", "r26", "r21", "r24")] + sync(1) + [dma_in("r7", "r23", "r11", "r17")])
    if TF:                         # r29 = the LSRAM accumulators; the staging rows past the rt tiles are zeros for the whole task
        body += add("r29", "r12", TACC) + [BUNDLE(I("mov4 t0, 0", E.mov4("t0", 0)))] + add("r14", "r12", TSTG)
        for s_ in range(ns):
            body += add("r13", "r14", 512) + [BUNDLE(I("st t0, [%s+%d]" % (b_, o_), E.vst("t0", b_, o_))) for t in range(NC, 24)
                                              for b_, o_ in [("r14", 32 * t) if t < 16 else ("r13", 32 * (t - 16))]] + add("r14", "r14", 768)
    # .Lgroup: zero the C region (nrb x ns blocks)
    grp = [BUNDLE(I("mov4 t0, 0", E.mov4("t0", 0)))] + add("r15", "r29", 0) + add("r16", "r5", 0)
    body_head, body = body, grp
    if not DMA_DRAIN:
        for _ in range(ns - 1): body += [BUNDLE(I("add r16, r16, r5", E.add("r16", "r16", "r5")))]
        z = add("r14", "r15", 512) + [BUNDLE(I("st t0, [r15+%d]" % (32 * t), E.vst("t0", "r15", 32 * t))) for t in range(16)]
        z += [BUNDLE(I("st t0, [r14+%d]" % (32 * t), E.vst("t0", "r14", 32 * t))) for t in range(8)]
        z += add("r15", "r15", 768) + [BUNDLE(I("sub r16, r16, 1", E.sub("r16", "r16", 1)))]
        z.append(BUNDLE(I("cbnz r16, .Lzero", E.cbnz("r16", -len(z)))))
        body += z
    body += add("r18", "r4", 0) + add("r22", "r0", 0)
    # .Lslice
    if not b8:
        sl = sync(0)[:0] + [dma_in("zero", "r24", "r25", "r1")] + wait(0) + add("r1", "r1", BH)
        sl += add("r16", "r5", 0) + add("r17", "r22", 0) + add("r15", "r29", 0) + (add("r28", "r2", 0) if rowmax else [])
        sl += sync(1) + [dma_in("r7", "r23", "r9", "r17")] + add("r17", "r17", a_rb_stride)          # row block 0 -> half 0
    else:                          # codes -> upper half of B region; row block 0's A; then expand
        if bscale:                 # codes + scales (contiguous in DDR) -> CB in one request on flag 0
            sl = []
            if poison:
                sl += _mov32("r7", 0x7FC00000) + [BUNDLE(I("bcast t31.w, r7", E.bcast("t31", "r7")))] + add("r13", "r20", BH)
                sl += [BUNDLE(I("st t31, [r13+%d]" % (32 * i), E.vst("t31", "r13", 32 * i))) for i in range(SCLB // 32)]
            sl += dmaq([dma_in("zero", "r26", "r20", "r1")])
        else: sl = add("r14", "r25", BH) + dmaq([dma_in("zero", "r24", "r14", "r1")])
        sl += add("r1", "r1", BH + SCLB)
        sl += add("r16", "r5", 0) + add("r17", "r22", 0) + add("r15", "r29", 0) + (add("r28", "r2", 0) if rowmax else [])
        sl += dmaq(sync(1) + [dma_in("r7", "r23", "r9", "r17")]) + add("r17", "r17", a_rb_stride)          # row block 0 -> half 0
        sl += dmaq(wait(0)) + ([] if timing == "dmaonly" or TF else expand("r20" if bscale else None))
    if PF:
        # Group-blocked (rows mode): blocks of gb <= GBLK groups; per K-slice, A once for the block and B for each group, the
        # groups' C accumulators resident in GSRAM (group g at g x CROW). Registers: r1 block's B base (the next block's is
        # r1 + gb x r5, recomputed where needed: r31 is `zero` and r22-r30 already serve), r16 gb, r18 slices left, r28 groups
        # left in the slice, r19 groups left; r22 / r24 / r17 the last requested B slice base / B / A; r2 the block's first C
        # row in DDR. Each step swaps current / next buffers, then waits and requests.
        GBLK = TGBLK if TF else max(1, min(16, 65536 // CROW)); SLB = BH + SCLB
        if not TF: GsramWindow(f"k_gemm_gs(rows, ns={ns})").alloc("group block's C rows", GBLK * CROW)
        assert c_stride % CROW == 0 and (c_stride // CROW) & (c_stride // CROW - 1) == 0, "group-blocked C: c_stride a power-of-two multiple of CROW"
        def mv(d, s_): return [BUNDLE(I("add %s, %s, 0" % (d, s_), E.add(d, s_, 0)))]
        def swap(a, b_): return mv("r7", a) + mv(a, b_) + mv(b_, "r7")
        reqB = dmaq([dma_in("zero", "r26", "r21", "r24")])
        reqA = dmaq(sync(1) + [dma_in("r7", "r23", "r11", "r17")])
        last_block = [BUNDLE(I("sub r7, r19, r16", E.sub("r7", "r19", "r16")))]
        last_group = [BUNDLE(I("sub r7, r28, 1", E.sub("r7", "r28", 1)))]
        step = swap("r20", "r21") + dmaq(wait(0))
        nxt_blk = [BUNDLE(I("mul r7, r16, r5", E.mul("r7", "r16", "r5"))), BUNDLE(I("add r24, r1, r7", E.add("r24", "r1", "r7")))] + mv("r22", "r24")
        step += ifz(last_group, ifz(last_slice, ifz(last_block, [], nxt_blk + reqB), add("r22", "r22", SLB) + mv("r24", "r22") + reqB),
                    [BUNDLE(I("add r24, r24, r5", E.add("r24", "r24", "r5")))] + reqB)
        step += [] if TF else expand("r20")
        step += ifz(last_slice, wait_flags(3), [])
        step += strips("r9", PSTG - BREG, scl="r20", cratio=c_stride // CROW)
        step += [BUNDLE(I("sub r28, r28, 1", E.sub("r28", "r28", 1)))]
        step.append(BUNDLE(I("cbnz r28, .Lstep", E.cbnz("r28", -len(step)))))
        slc = mv("r28", "r16") + mv("r15", "r29") + swap("r9", "r11") + dmaq(wait(1))
        slc += ifz(last_slice, ifz(last_block, [], mv("r17", "r0") + reqA), [BUNDLE(I("add r17, r17, r6", E.add("r17", "r17", "r6")))] + reqA)
        slc += (sprep("r9") if SP else []) + step + [BUNDLE(I("sub r18, r18, 1", E.sub("r18", "r18", 1)))]
        slc.append(BUNDLE(I("cbnz r18, .Lslc", E.cbnz("r18", -len(slc)))))
        blk = _mov32("r7", GBLK) + [BUNDLE(I("mn r16, r19, r7", E.mn("r16", "r19", "r7")))]
        blk += mv("r18", "r4")
        blk += slc
        blk += _mov32("r7", c_stride) + [BUNDLE(I("mul r7, r16, r7", E.mul("r7", "r16", "r7"))), BUNDLE(I("add r2, r2, r7", E.add("r2", "r2", "r7")))]
        blk += [BUNDLE(I("mul r7, r16, r5", E.mul("r7", "r16", "r5"))), BUNDLE(I("add r1, r1, r7", E.add("r1", "r1", "r7")))]
        blk += [BUNDLE(I("sub r19, r19, r16", E.sub("r19", "r19", "r16")))]
        blk.append(BUNDLE(I("cbnz r19, .Lblock", E.cbnz("r19", -len(blk)))))
    if SINGLE:                     # row block 0 only; its one staging buffer: the previous group's drain must be done first
        # the previous group's drain (flag 3) -- none in "dmaonly", which drains nothing: no wait on a flag nobody raises
        rb = dmaq(wait(1)) + ([] if timing == "dmaonly" else ifz(last_slice, wait_flags(3), []))
        rb += [] if timing == "dmaonly" else (sprep("r9") if SP else []) + strips("r9", STG0 - BREG)
    else:
        rb = [BUNDLE(I("sub r16, r16, 2", E.sub("r16", "r16", 2)))]
        rb += wait(1) + sync(2) + [dma_in("r7", "r23", "r11", "r17")] + add("r17", "r17", a_rb_stride)  # odd row block -> half 1
        rb += (sprep("r9") if SP else []) + strips("r9", STG0 - BREG)
        rb += wait(2) + prefetch("r9", 1)
        rb += (sprep("r11") if SP else []) + strips("r11", STG1 - BREG)
        rb.append(BUNDLE(I("cbnz r16, .Lrb", E.cbnz("r16", -len(rb)))))
    if not PF:
        sl += rb
        sl += add("r22", "r22", "r6") + [BUNDLE(I("sub r18, r18, 1", E.sub("r18", "r18", 1)))]
        sl.append(BUNDLE(I("cbnz r18, .Lslice", E.cbnz("r18", -len(sl)))))
    body += sl
    # drain: C region GSRAM -> DDR by the TEC
    if not DMA_DRAIN:
        body += add("r15", "r29", 0) + add("r16", "r5", 0) + add("r17", "r2", 0)
        for _ in range(ns - 1): body += [BUNDLE(I("add r16, r16, r5", E.add("r16", "r16", "r5")))]
        d = add("r14", "r15", 512) + add("r13", "r17", 512)
        for t in range(0, 24, 2): d += [BUNDLE(cload(t), cload(t + 1))]
        for t in range(24):
            b_, off = ("r17", t * 32) if t < 16 else ("r13", (t - 16) * 32)
            d += [BUNDLE(I("st t%d, [%s+%d]" % (t, b_, off), E.vst("t%d" % t, b_, off)))]
        d += add("r15", "r15", 768) + add("r17", "r17", 768) + [BUNDLE(I("sub r16, r16, 1", E.sub("r16", "r16", 1)))]
        d.append(BUNDLE(I("cbnz r16, .Ldrain", E.cbnz("r16", -len(d)))))
        body += d
    # next group: C out advances, A rewinds (r0), B continues (r1)
    body += add("r2", "r2", c_stride) + [BUNDLE(I("sub r19, r19, 1", E.sub("r19", "r19", 1)))]
    body.append(BUNDLE(I("cbnz r19, .Lgroup", E.cbnz("r19", -len(body)))))
    body = body_head + (blk if PF else body)
    if stamps:  # total = now - entry stamp at [r28+0]
        body += [CYC("r30"), BUNDLE(I("ld r7, [r28+0]", E.ld("r7", "r28", 0))), BUNDLE(I("sub r30, r30, r7", E.sub("r30", "r30", "r7"))),
                 BUNDLE(I("st r30, [r28+0]", E.st("r30", "r28", 0))), BUNDLE(I("st r20, [r28+4]", E.st("r20", "r28", 4))),
                 BUNDLE(I("st r21, [r28+8]", E.st("r21", "r28", 8))), BUNDLE(I("st r26, [r28+12]", E.st("r26", "r28", 12)))]
    if DMA_DRAIN and timing != "dmaonly": body += wait_flags(3)  # last row block's drain
    body[skip] = BUNDLE(I("b .Ldone", E.b(len(body) - skip)))
    return prologue(10 if stamps else 9) + body + epilogue()


def k_mma32_bench(gs="none", ks=24):
    """Timing-only benchmark: does a GSRAM access into registers the mma does not touch overlap the matrix work?
    3 x 2 register block: C in t0..t11, an idle second C set Y in t12..t23, A t24..t26, B double-buffered
    t27/t28 | t29/t30; 6 `mma` per k-step in 3 bundles, loads two bundles ahead, one free memory slot per k-step.
      gs: "none"; "ld" / "st" (a Y load / store every 2 k-steps); "stld" (store then load, one per k-step);
          "ld1" (a load every k-step)
    args: r0 = GSRAM base, r1 = stamp out (4 B: cycles for all reps), r2 = reps. LSRAM contents arbitrary."""
    assert ks % 6 == 0 and gs in ("none", "ld", "st", "stld", "ld1")
    CX = lambda i, j: 2 * (2 * i + j)
    A = ["t24", "t25", "t26"]; B = [["t27", "t28"], ["t29", "t30"]]
    def mma(i, j, bs): c = CX(i, j); return I("mma t%d.fp32, t%d.fp32, %s.fp16, %s.fp16" % (c, c + 1, A[i], B[bs][j]), E.mma("t%d" % c, A[i], B[bs][j]))
    def ld(r, base, off): return I("ld %s, [%s+%d]" % (r, base, off), E.vld(r, base, off))
    def st(r, base, off): return I("st %s, [%s+%d]" % (r, base, off), E.vst(r, base, off))
    body = []
    for k in range(ks):
        bs, kk = k % 2, k % 3  # B set; kk: k-step within the 3-step pointer window
        nb, nk = 1 - bs, kk + 1  # next k-step's B set / window offset (3: next window)
        aoff = lambda kx, i: (96 * kx + 32 * i) if kx < 3 else None
        g = []
        if gs == "ld" and k % 2 == 0: g = [ld("t%d" % (12 + (k // 2) % 12), "r15", 32 * ((k // 2) % 12))]
        if gs == "st" and k % 2 == 0: g = [st("t%d" % (12 + (k // 2) % 12), "r15", 32 * ((k // 2) % 12))]
        if gs == "stld": j = (k // 2) % 12; g = [st("t%d" % (12 + j), "r15", 32 * j)] if k % 2 == 0 else [ld("t%d" % (12 + j), "r15", 32 * j)]
        if gs == "ld1": g = [ld("t%d" % (12 + k % 12), "r15", 32 * (k % 12))]
        # next k-step's operands: A0 / B0 in b1, A1 in b2, B1 in b0 (A2 in its own b0)
        def nxt(r, kind, idx):
            if nk < 3: return ld(r, "r8" if kind == "A" else "r13", (96 if kind == "A" else 128) * nk + 32 * idx)
            return ld(r, "r9" if kind == "A" else "r14", 32 * idx)  # next window: r9 = r8 + 288, r14 = r13 + 384
        b0 = [mma(0, 0, bs), mma(0, 1, bs), ld(A[2], "r8", 96 * kk + 64), nxt(B[nb][1], "B", 1)]
        b1 = [mma(1, 0, bs), mma(1, 1, bs), nxt(A[0], "A", 0), nxt(B[nb][0], "B", 0)]
        b2 = [mma(2, 0, bs), mma(2, 1, bs)] + [x for x in g if x[0].startswith("st")] + [nxt(A[1], "A", 1)] + [x for x in g if not x[0].startswith("st")]
        body += [BUNDLE(*b0), BUNDLE(*b1), BUNDLE(*b2)]
        if kk == 2:  # advance the pointer window (r9 / r14 one ahead)
            body.append(BUNDLE(I("add r8, r8, 288", E.add("r8", "r8", 288)), I("add r13, r13, 384", E.add("r13", "r13", 384))))
            body.append(BUNDLE(I("add r9, r8, 288", E.add("r9", "r8", 288)), I("add r14, r13, 384", E.add("r14", "r13", 384))))
    setup = _lsram_base("r12") + [BUNDLE(I("add r8, r12, 0", E.add("r8", "r12", 0)), I("add r15, r0, 0", E.add("r15", "r0", 0)))]
    setup += _mov32("r7", 16384) + [BUNDLE(I("add r13, r12, r7", E.add("r13", "r12", "r7")))]
    setup += [BUNDLE(I("add r9, r8, 288", E.add("r9", "r8", 288)), I("add r14, r13, 384", E.add("r14", "r13", 384)))]
    setup += [BUNDLE(ld(A[0], "r8", 0), ld(B[0][0], "r13", 0)), BUNDLE(ld(A[1], "r8", 32), ld(B[0][1], "r13", 32)), BUNDLE(ld(A[2], "r8", 64))]
    rep = [BUNDLE(I("add r8, r12, 0", E.add("r8", "r12", 0)), I("add r9, r12, 288", E.add("r9", "r12", 288)))] + _mov32("r7", 16384) + \
          [BUNDLE(I("add r13, r12, r7", E.add("r13", "r12", "r7"))), BUNDLE(I("add r14, r13, 384", E.add("r14", "r13", 384))),
           BUNDLE(I("mov r10, 0", E.mov("r10", 0))), BUNDLE(I("loop r10", E.loop("r10")))] + body + [BUNDLE(I("loopend", E.loopend()))]
    rep += [BUNDLE(I("sub r2, r2, 1", E.sub("r2", "r2", 1)))]
    rep.append(BUNDLE(I("cbnz r2, .Lrep", E.cbnz("r2", -len(rep)))))
    t = [BUNDLE(I("mfctrl0 r20, 209", E.mfctrl0("r20", 0xd1)))] + rep + [BUNDLE(I("mfctrl0 r21, 209", E.mfctrl0("r21", 0xd1))),
         BUNDLE(I("sub r21, r21, r20", E.sub("r21", "r21", "r20"))), BUNDLE(I("st r21, [r1+0]", E.st("r21", "r1", 0)))]
    return prologue(3) + setup + t + epilogue()


def k_gemm_gm(nrb=28, ks=48, ns=3, c_stride=None, c_rb_stride=None, b8=True, a_rb_stride=None, rowmax=False, timing=None):
    """`k_gemm_gs`'s 3x4 loop with the partials in GM instead of GSRAM. Per K slice: the strips' B panels
    resident in LSRAM (b8: E4M3 codes expanded in place); each row block's A slice DMA'd from GM (staged by
    `k_gm_stage`: [slice][rb][ks x 96 B]) through two LSRAM halves; its C row (ns x 768 B) DMA'd
    GM -> LSRAM staging -> registers -> staging -> GM around the k-loops (first slice starts from zero,
    last drains to the DDR C). The TEC only touches LSRAM; the C DMA runs behind the next row block.
    Flags: 0 = B, then staging 0's C; 1 / 2 = A halves; 3 = staging 1's C. Each staging buffer alternates
    out(rb) / in(rb + 2) on its flag, the in issued after strip 0 of the other buffer's row block.
    args: r0 = A (GM, slice 0 / row block 0), r1 = B groups, r2 = C out (DDR), r3 = DESC (`descriptors_gm`:
          +0 A slice, +32 B slice, +64 C row), r4 = NSLICES, r5 = NRB (even), r6 = A_SLICE (nrb x ks x 96),
          r7 = the task's GM C region (nrb x ns x 768 B), r8 = NGROUPS (C out advancing `c_stride`).
    `a_rb_stride`: distance between row blocks' A slices (default ks x 96); e.g. 2304 to read DDR's 24-deep
    layout directly. `rowmax`: as `k_gemm_gs`, each C row followed by its rows' maxima (192 B)."""
    assert ks % 3 == 0 and nrb % 2 == 0
    CR = ns * 768 + (192 if rowmax else 0)  # a C row: strips (+ maxima)
    a_rb_stride = ks * 96 if a_rb_stride is None else a_rb_stride
    c_stride = nrb * CR if c_stride is None else c_stride
    c_rb_stride = CR if c_rb_stride is None else c_rb_stride
    BH = ns * ks * (64 if b8 else 128)
    LS = Lsram(f"k_gemm_gm(ks={ks}, ns={ns})")     # every region is read by 16-B vld
    LS.alloc("B panels (fp16)", ns * ks * 128, vld_tail=True)
    A0 = LS.alloc("A half 0", ks * 96, vld_tail=True); A1 = LS.alloc("A half 1", ks * 96, vld_tail=True)
    STG0 = LS.alloc("C staging 0", CR, vld_tail=True); STG1 = LS.alloc("C staging 1", CR, vld_tail=True)
    ORDER = [(0, 0), (0, 2), (1, 0), (1, 2), (0, 1), (0, 3), (2, 0), (2, 2), (1, 1), (1, 3), (2, 1), (2, 3)]
    THIS = {0: ["B1", "B3"], 1: ["A2"]}
    NEXT = {3: ["A0"], 4: ["B0", "B2"], 5: ["A1"]}
    REG = {"A0": "t24", "A1": "t25", "A2": "t26", "B0": "t27", "B1": "t28", "B2": "t29", "B3": "t30"}
    def MMA(i, j):
        c = 2 * (4 * i + j)
        return I("mma t%d.fp32, t%d.fp32, %s.fp16, %s.fp16" % (c, c + 1, REG["A%d" % i], REG["B%d" % j]), E.mma("t%d" % c, REG["A%d" % i], REG["B%d" % j]))
    def LD(x, k):
        if x[0] == "A": base, off = "r8", 96 * k + 32 * int(x[1])
        else:           base, off = "r13", 128 * k + 32 * int(x[1])
        return I("ld %s, [%s+%d]" % (REG[x], base, off), E.vld(REG[x], base, off))
    kbody = []
    for k in range(3):
        for b in range(6):
            kbody.append(BUNDLE(*([MMA(*ORDER[2 * b]), MMA(*ORDER[2 * b + 1])] + [LD(x, k) for x in THIS.get(b, [])] + [LD(x, k + 1) for x in NEXT.get(b, [])])))
    kbody.append(BUNDLE(I("add r8, r8, 288", E.add("r8", "r8", 288)), I("add r13, r13, 384", E.add("r13", "r13", 384))))
    def add(rd, ra, x):
        if isinstance(x, int) and x > 1023: return _mov32("r7", x) + [BUNDLE(I("add %s, %s, r7" % (rd, ra), E.add(rd, ra, "r7")))]
        return [BUNDLE(I("add %s, %s, %s" % (rd, ra, x), E.add(rd, ra, x)))]
    def wait(flag): return wait_flags(flag)
    def dma(flag, desc, lsram, ext, into_lsram):
        d = E.DMA_EXT2INT if into_lsram else E.DMA_INT2EXT
        return sync_flag(flag) + [
                BUNDLE(I("dma 0, r7, 0, 4, 0, %d, 0, %s, %s, %s" % (d, desc, lsram, ext), E.dma(E.DMA_DIRECT, "r7", 0, E.DMA_LSRAM, E.DMA_GLOBAL, d, E.DMA_USELESS, desc, lsram, ext)))]
    def ifz(test, then, other):
        return test + [BUNDLE(I("cbnz r7, .Lelse", E.cbnz("r7", len(then) + 2)))] + then + [BUNDLE(I("b .Lend", E.b(len(other) + 1)))] + other
    def when_nz(reg, code): return [BUNDLE(I("cbnz %s, .Ldo" % reg, E.cbnz(reg, 2))), BUNDLE(I("b .Lskip", E.b(len(code) + 1)))] + code
    first_slice = [BUNDLE(I("sub r7, r18, r4", E.sub("r7", "r18", "r4")))]
    last_slice = [BUNDLE(I("sub r7, r18, 1", E.sub("r7", "r18", 1)))]
    def cbase(t): return ("r3", t * 32) if t < 16 else ("r14", (t - 16) * 32)
    PAIRS = [(t, t + 2) for t in range(24) if t % 4 in (0, 1)]  # two loads a bundle, 64 B apart (LSRAM banks)
    def loads(): return [BUNDLE(*[I("ld t%d, [%s+%d]" % (t, *cbase(t)), E.vld("t%d" % t, *cbase(t))) for t in pr]) for pr in PAIRS]
    def stores(): return [BUNDLE(I("st t%d, [%s+%d]" % (t, *cbase(t)), E.vst("t%d" % t, *cbase(t)))) for t in range(24)]
    def rowmax_strip():            # as k_gemm_gs: after the stores, maxima against the running ones at r30
        ops = []
        l1 = [(8 * i + h + a, 8 * i + h + a + 2) for i in range(3) for h in range(2) for a in (0, 4)]
        lds = [(24 + k, 32 * k) for k in range(6)]
        for n in range(6):
            b_ = [I("max t%d.fp32, t%d.fp32, t%d.fp32, p7.w" % (d, d, x), E.vmaxf("t%d" % d, "t%d" % d, "t%d" % x)) for d, x in l1[2 * n:2 * n + 2]]
            if n < 3: b_ += [I("ld t%d, [r30+%d]" % (r, o), E.vld("t%d" % r, "r30", o)) for r, o in lds[2 * n:2 * n + 2]]
            ops.append(BUNDLE(*b_))
        l2 = [8 * i + h for i in range(3) for h in range(2)]
        for n in range(3):
            ops.append(BUNDLE(*[I("max t%d.fp32, t%d.fp32, t%d.fp32, p7.w" % (d, d, d + 4), E.vmaxf("t%d" % d, "t%d" % d, "t%d" % (d + 4))) for d in l2[2 * n:2 * n + 2]]))
        for n in range(3):
            ops.append(BUNDLE(*[I("max t%d.fp32, t%d.fp32, t%d.fp32, p7.w" % (24 + k, 24 + k, l2[k]), E.vmaxf("t%d" % (24 + k), "t%d" % (24 + k), "t%d" % l2[k])) for k in (2 * n, 2 * n + 1)]))
        ops += [BUNDLE(I("st t%d, [r30+%d]" % (24 + k, 32 * k), E.vst("t%d" % (24 + k), "r30", 32 * k))) for k in range(6)]
        return ops
    def zero_c(): return [BUNDLE(I("mov4 t%d, 0" % t, E.mov4("t%d" % t, 0))) for t in range(24)]

    def expand():                  # as k_gemm_gs b8
        e = [BUNDLE(I("mov4 t0, 0", E.mov4("t0", 0)))] + _mov32("r7", 0xBFFFBFFF) + [BUNDLE(I("bcast t1.w, r7", E.bcast("t1", "r7")))]
        e += add("r13", "r25", BH) + add("r14", "r25", 0) + [BUNDLE(I("mov r10, %d" % (BH // 128 - 1), E.mov("r10", BH // 128 - 1))), BUNDLE(I("loop r10", E.loop("r10")))]
        X, H = ["t%d" % (2 + v) for v in range(4)], ["t%d" % (6 + j) for j in range(8)]
        body = [BUNDLE(I("ld %s, [r13+0]" % X[0], E.vld(X[0], "r13", 0)), I("ld %s, [r13+64]" % X[2], E.vld(X[2], "r13", 64))),
                BUNDLE(I("ld %s, [r13+32]" % X[1], E.vld(X[1], "r13", 32)), I("ld %s, [r13+96]" % X[3], E.vld(X[3], "r13", 96)))]
        for v in range(4):
            body += [BUNDLE(I("zipl %s.b, t0.b, %s.b" % (H[2 * v], X[v]), E.zipl(H[2 * v], "t0", X[v]))),
                     BUNDLE(I("ziph %s.b, t0.b, %s.b" % (H[2 * v + 1], X[v]), E.ziph(H[2 * v + 1], "t0", X[v])))]
        body[-1] = BUNDLE(*(list(body[-1]) + [I("add r13, r13, 128", E.add("r13", "r13", 128))]))
        for j in range(8): body.append(BUNDLE(I("asr %s.h, %s.h, 1" % (H[j], H[j]), E.vasri(H[j], H[j], 1, size="h"))))
        for j in range(8): body.append(BUNDLE(I("and %s.w, %s.w, t1.w" % (H[j], H[j]), E.vand(H[j], H[j], "t1"))))
        for j in range(8): body.append(BUNDLE(I("st %s, [r14+%d]" % (H[j], 32 * j), E.vst(H[j], "r14", 32 * j))))
        body[-1] = BUNDLE(*(list(body[-1]) + [I("add r14, r14, 256", E.add("r14", "r14", 256))]))
        return e + body + [BUNDLE(I("loopend", E.loopend()))]

    def strips(half_reg, stg_reg, hook):
        """The row block in `half_reg` against the ns strips; C in / out of the staging buffer at `stg_reg`; `hook` after strip 0."""
        blk = add("r13", "r25", 0) + [BUNDLE(I("mov r27, %d" % ns, E.mov("r27", ns))), BUNDLE(I("add r3, %s, 0" % stg_reg, E.add("r3", stg_reg, 0)))]
        if rowmax:  # r30 = maxima after the strips in staging, -inf at the last slice
            blk += add("r30", stg_reg, ns * 768) + ifz(last_slice, [BUNDLE(I("st t31, [r30+%d]" % (32 * k), E.vst("t31", "r30", 32 * k))) for k in range(6)], [])
        body = add("r14", "r3", 512)
        body += ifz(first_slice, zero_c(), loads())
        body += add("r8", half_reg, 0) + [BUNDLE(LD("B0", 0), LD("B2", 0)), BUNDLE(LD("A0", 0)), BUNDLE(LD("A1", 0)),
                                            BUNDLE(I("mov r10, %d" % (ks // 3 - 1), E.mov("r10", ks // 3 - 1)))]
        body += [BUNDLE(I("loop r10", E.loop("r10")))] + kbody + [BUNDLE(I("loopend", E.loopend()))]
        body += stores()
        if rowmax: body += ifz(last_slice, rowmax_strip(), [])
        body += ifz([BUNDLE(I("sub r7, r27, %d" % ns, E.sub("r7", "r27", ns)))], hook, [])
        body += add("r3", "r3", 768) + [BUNDLE(I("sub r27, r27, 1", E.sub("r27", "r27", 1)))]
        body.append(BUNDLE(I("cbnz r27, .Lstrip", E.cbnz("r27", -len(body)))))
        return blk + body

    def c_out(stg_reg, flag):  # C row to GM (r15), or to DDR C (r28) after the last slice
        return ifz(last_slice, [BUNDLE(I("add r14, r28, 0", E.add("r14", "r28", 0)))], [BUNDLE(I("add r14, r15, 0", E.add("r14", "r15", 0)))]) + \
               dma(flag, "r21", stg_reg, "r14", False) + add("r15", "r15", CR) + add("r28", "r28", c_rb_stride)

    c_next = add("r14", "r15", CR)  # next row block's C row in GM
    hook_even = wait(3) + ifz(first_slice, [], c_next + dma(3, "r21", "r29", "r14", True))  # odd rb's C -> STG1
    hook_odd = wait(0) + ifz(first_slice, [], when_nz("r16", c_next + dma(0, "r21", "r26", "r14", True)))  # next pair's even C -> STG0

    body = [BUNDLE(I("cbnz r4, .Lgo", E.cbnz("r4", 2))), BUNDLE(I("b .Ldone", E.b(0)))]
    skip = 1
    body += [BUNDLE(I("add r20, r7, 0", E.add("r20", "r7", 0))), BUNDLE(I("add r19, r8, 0", E.add("r19", "r8", 0)))]
    body += _lsram_base("r12") + add("r25", "r12", 0) + add("r9", "r12", A0) + add("r11", "r12", A1) + add("r26", "r12", STG0) + add("r29", "r12", STG1)
    body += add("r23", "r3", 0) + add("r24", "r3", 32) + add("r21", "r3", 64)
    if rowmax: body += _mov32("r7", 0xFF800000) + [BUNDLE(I("bcast t31.w, r7", E.bcast("t31", "r7")))]     # t31 = -inf
    head, body = body, []
    body += add("r18", "r4", 0) + add("r22", "r0", 0)
    sl = wait(0) + wait(3)  # previous slice's C traffic done; flag 0 is B's
    sl += add("r14", "r25", BH if b8 else 0) + dma(0, "r24", "r14", "r1", True) + add("r1", "r1", BH)
    sl += add("r16", "r5", 0) + add("r17", "r22", 0) + add("r15", "r20", 0) + add("r28", "r2", 0)
    sl += dma(1, "r23", "r9", "r17", True) + add("r17", "r17", a_rb_stride)  # A(rb 0) -> half 0
    sl += wait(0) + ifz(first_slice, [], dma(0, "r21", "r26", "r15", True))  # B landed; C(rb 0) -> STG0
    if b8 and timing != "noexp": sl += expand()  # timing="noexp": diagnostic, skips expansion (wrong results)
    rb = [BUNDLE(I("sub r16, r16, 2", E.sub("r16", "r16", 2)))]
    rb += wait(1) + dma(2, "r23", "r11", "r17", True) + add("r17", "r17", a_rb_stride)  # A(odd) -> half 1
    rb += wait(0) + strips("r9", "r26", hook_even) + c_out("r26", 0)
    rb += wait(2) + when_nz("r16", dma(1, "r23", "r9", "r17", True) + add("r17", "r17", a_rb_stride))  # A(next even) -> half 0
    rb += wait(3) + strips("r11", "r29", hook_odd) + c_out("r29", 3)
    rb.append(BUNDLE(I("cbnz r16, .Lrb", E.cbnz("r16", -len(rb)))))
    sl += rb + add("r22", "r22", "r6") + [BUNDLE(I("sub r18, r18, 1", E.sub("r18", "r18", 1)))]
    sl.append(BUNDLE(I("cbnz r18, .Lslice", E.cbnz("r18", -len(sl)))))
    body += sl + add("r2", "r2", c_stride) + [BUNDLE(I("sub r19, r19, 1", E.sub("r19", "r19", 1)))]
    body.append(BUNDLE(I("cbnz r19, .Lgroup", E.cbnz("r19", -len(body)))))
    body = head + body + wait(0) + wait(3)
    body[skip] = BUNDLE(I("b .Ldone", E.b(len(body) - skip)))
    return prologue(9) + body + epilogue()


def k_gm_stage(ks=48, rbw=2304):
    """Stage a row chunk DDR -> GM for `k_gemm_gm`: 24-deep A layout ([slice24][NRB][2304 B]) to
    [slice][rb][ks x 96 B]. Per (slice, row block): strided gather into LSRAM (flag p), contiguous DMA to GM
    (flag 2 + p), double-buffered. args: r0 = DDR source, r1 = GM dest, r2 = DESC (+0 gather, +32 out),
    r3 = slice count (0: exit), r4 = row blocks (even), r5 / r6 = source / dest step between slices."""
    UB = ks * 96
    def add(rd, ra, x):
        if isinstance(x, int) and x > 1023: return _mov32("r7", x) + [BUNDLE(I("add %s, %s, r7" % (rd, ra), E.add(rd, ra, "r7")))]
        return [BUNDLE(I("add %s, %s, %s" % (rd, ra, x), E.add(rd, ra, x)))]
    def wait(flag): return wait_flags(flag)
    def dma(flag, desc, lsram, ext, into):
        d = E.DMA_EXT2INT if into else E.DMA_INT2EXT
        return sync_flag(flag) + [
                BUNDLE(I("dma 0, r7, 0, 4, 0, %d, 0, %s, %s, %s" % (d, desc, lsram, ext), E.dma(E.DMA_DIRECT, "r7", 0, E.DMA_LSRAM, E.DMA_GLOBAL, d, E.DMA_USELESS, desc, lsram, ext)))]
    body = [BUNDLE(I("cbnz r3, .Lgo", E.cbnz("r3", 2))), BUNDLE(I("b .Ldone", E.b(0)))]
    skip = 1
    body += _lsram_base("r12") + add("r9", "r12", 0) + add("r11", "r12", UB) + add("r13", "r2", 32)
    sl = add("r10", "r0", 0) + add("r14", "r1", 0) + add("r16", "r4", 0)
    pair = [BUNDLE(I("sub r16, r16, 2", E.sub("r16", "r16", 2)))]
    pair += wait(2) + dma(0, "r2", "r9", "r10", True) + add("r10", "r10", rbw)
    pair += wait(3) + dma(1, "r2", "r11", "r10", True) + add("r10", "r10", rbw)
    pair += wait(0) + dma(2, "r13", "r9", "r14", False) + add("r14", "r14", UB)
    pair += wait(1) + dma(3, "r13", "r11", "r14", False) + add("r14", "r14", UB)
    pair.append(BUNDLE(I("cbnz r16, .Lpair", E.cbnz("r16", -len(pair)))))
    sl += pair + add("r0", "r0", "r5") + add("r1", "r1", "r6") + [BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1)))]
    sl.append(BUNDLE(I("cbnz r3, .Lslice", E.cbnz("r3", -len(sl)))))
    body += sl + wait(2) + wait(3)
    body[skip] = BUNDLE(I("b .Ldone", E.b(len(body) - skip)))
    return prologue(7) + body + epilogue()


E4M3_STREAM_CHUNK = 8192            # bytes of E4M3 per LSRAM fill (256 vectors) -> 16 KiB of fp16 out


def k_e4m3_stream(mode="exact"):
    """E4M3 -> fp16 over a DDR stream: 8 KiB chunks DMA'd into LSRAM (flag 0), unpacked by `_e4m3_v2_loop`
    into 16 KiB of fp16 in LSRAM, drained to DDR by DMA (flag 1; direct vector stores to DDR are far slower).
    Byte order is kept, so `pack_b_group_e4m3` input yields `pack_b_group` fp16 panels.
    args: r0 = SRC, r1 = DESC (8 KiB fill +0, 16 KiB drain +32), r2 = OUT, r3 = NCHUNKS (0: exit)."""
    body = [BUNDLE(I("cbnz r3, .Lgo", E.cbnz("r3", 2))), BUNDLE(I("b .Ldone", E.b(0)))]
    skip = 1
    body += _lsram_base("r6") + _e4m3_v2_consts() + [BUNDLE(I("add r9, r1, 32", E.add("r9", "r1", 32)))]
    body += _mov32("r11", 2 * E4M3_STREAM_CHUNK) + [BUNDLE(I("mov r12, %d" % E4M3_STREAM_CHUNK, E.mov("r12", E4M3_STREAM_CHUNK))), BUNDLE(I("add r12, r6, r12", E.add("r12", "r6", "r12")))]  # r12 = fp16 region
    chunk = [BUNDLE(I("dma 0, zero, 0, 4, 0, 1, 0, r1, r6, r0", E.dma(E.DMA_DIRECT, "zero", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT, E.DMA_USELESS, "r1", "r6", "r0"))),
             BUNDLE(I("sub r7, zero, 2", E.sub("r7", "zero", 2))), BUNDLE(I("wfe r7, 1", E.wfe("r7", 1))),
             BUNDLE(I("add r8, r6, 0", E.add("r8", "r6", 0))), BUNDLE(I("add r13, r12, 0", E.add("r13", "r12", 0))),
             BUNDLE(I("mov r10, %d" % (E4M3_STREAM_CHUNK // 32 - 1), E.mov("r10", E4M3_STREAM_CHUNK // 32 - 1))), BUNDLE(I("loop r10", E.loop("r10")))]
    chunk += _e4m3_v2_loop(mode, "r8", "r13") + [BUNDLE(I("loopend", E.loopend()))]
    chunk += [BUNDLE(I("mov r7, 257", E.mov("r7", 257))),
              BUNDLE(I("dma 0, r7, 0, 4, 0, 0, 0, r9, r12, r2", E.dma(E.DMA_DIRECT, "r7", 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_INT2EXT, E.DMA_USELESS, "r9", "r12", "r2"))),
              BUNDLE(I("sub r7, zero, 3", E.sub("r7", "zero", 3))), BUNDLE(I("wfe r7, 1", E.wfe("r7", 1))),
              BUNDLE(I("add r2, r2, r11", E.add("r2", "r2", "r11"))),
              BUNDLE(I("mov r7, %d" % E4M3_STREAM_CHUNK, E.mov("r7", E4M3_STREAM_CHUNK))), BUNDLE(I("add r0, r0, r7", E.add("r0", "r0", "r7"))),
              BUNDLE(I("sub r3, r3, 1", E.sub("r3", "r3", 1)))]
    chunk.append(BUNDLE(I("cbnz r3, .Lchunk", E.cbnz("r3", -len(chunk)))))
    body += chunk
    body[skip] = BUNDLE(I("b .Ldone", E.b(len(body) - skip)))
    return prologue(4) + body + epilogue()
