"""Host side of the TEC GEMMs (`kern_tpc.k_gemm_gs` / `k_gemm_gm`): the operand packings the kernels read, the C unpacking,
the DMA descriptor tables, the E4M3 table and a reference -- numpy only.

    A  fp16 `[MP, K]` (MP = 12 * nrb rows) -> `pack_a_slices`: [K/(4ks) slices][MP/12 row blocks][ks][3 tiles][4x4]
    B  `[N, K]` (one row per output column) -> `pack_b_group` (fp16 panels of ns 16-column strips), `pack_b_group_e4m3`
       (the same order as E4M3 codes), `pack_b_group_bscale` / `_q8` / `_q8f` (block-scaled streams), `repack_panels_48x3`
    C  the kernel's `[rb][s][3][4][4x4 fp32]` tiles -> `unpack_c_gs`: fp32 `[12 nrb, 16 ns]`
"""
import numpy as np
from extra.zhouyi import dma as D

_E4M3 = None


def e4m3_table() -> np.ndarray:
    """The 256 E4M3 (fn) values as float32 (NaN at 0x7f / 0xff)."""
    global _E4M3
    if _E4M3 is None:
        t = np.empty(256, np.float32)
        for b in range(256):
            s = -1.0 if b & 0x80 else 1.0; e = (b >> 3) & 0xF; m = b & 7
            t[b] = s * ((m / 8.0) * 2.0 ** -6 if e == 0 else (np.nan if (e == 15 and m == 7) else (1.0 + m / 8.0) * 2.0 ** (e - 7)))
        _E4M3 = t
    return _E4M3


def pack_a_slices(a_f16: np.ndarray, ks: int) -> np.ndarray:
    """A `[MP, K]` -> [K/(4ks) slices][MP/12 row blocks][ks][3 tiles][4x4]: a row block's slice is one contiguous DMA."""
    a = np.ascontiguousarray(a_f16).view(np.uint16) if a_f16.dtype != np.uint16 else a_f16
    MP, K = a.shape; assert MP % 12 == 0 and K % (4 * ks) == 0
    t = a.reshape(MP // 12, 3, 4, K // (4 * ks), ks, 4)                              # [rb][i][r][slice][kk][k]
    return np.ascontiguousarray(t.transpose(3, 0, 4, 1, 2, 5))                       # [slice][rb][kk][i][r][k]


def pack_b_group(b_e4m3: np.ndarray, group: int, ns: int, ks: int) -> np.ndarray:
    """B `[N, K]` E4M3 -> the fp16 panels of strips `group*ns .. +ns`: [slice][s][kk][4 tiles][4 cols][4 k] uint16
    (the host's exact unpack; the TEC pass `k_e4m3_unpack_v2` produces the same bytes)."""
    N, K = b_e4m3.shape; assert K % (4 * ks) == 0
    b = e4m3_table()[b_e4m3[16 * ns * group:16 * ns * (group + 1)]].astype(np.float16).view(np.uint16)   # [16 ns, K]
    t = b.reshape(ns, 4, 4, K // (4 * ks), ks, 4)                                    # [s][j][n][slice][kk][k]
    return np.ascontiguousarray(t.transpose(3, 0, 4, 1, 2, 5))                       # [slice][s][kk][j][n][k]


def unpack_c_gs(c_bytes: bytes, nrb: int, ns: int) -> np.ndarray:
    """The C region `[rb][s][3][4][4x4 fp32]` -> `[12 nrb, 16 ns]`."""
    c = np.frombuffer(c_bytes, np.float32)[:nrb * ns * 192].reshape(nrb, ns, 3, 4, 4, 4)
    return np.ascontiguousarray(c.transpose(0, 2, 4, 1, 3, 5)).reshape(12 * nrb, 16 * ns)


SCALE_LAYOUTS = ("dup", "single")   # the E4M3 block-scale table's layout: [s][j][c0 c1 c2 c3 c0 c1 c2 c3] (128 B a strip) / [s][j][c0 c1 c2 c3] (64 B)


def gs_scale_bytes(ns: int, ks: int, q8: int = 0, scales: str = "dup") -> int:
    """A K-slice's scale table in a block-scaled gemm_gs stream: E4M3 (q8 0) [s][4 tiles][8] fp32 = 128 B a strip (`scales`
    "dup": each scale stored twice, the kernel's lane layout) or [s][4 tiles][4] fp32 = 64 B a strip ("single": each once, the
    kernel widens it); Q8_0 with the correction kernel (q8 1, ks 8) 16 fp16 a strip; Q8_0 dequantised in fp16 (q8 2) 16 fp16 a
    strip and 32-weight block."""
    assert scales in SCALE_LAYOUTS, scales
    return ns * (128 if scales == "dup" else 64) if not q8 else ns * 32 if q8 == 1 else ns * 32 * (ks // 8)


def descriptors_gs(ks: int, ns: int, rowmax: bool = False, b8: bool = False, bscale: bool = False, a_bytes: int | None = None,
                   prefetch: bool = False, q8: int = 0, scales: str = "dup") -> bytes:
    """+0 the A row-block slice (`a_bytes`, default ks x 96; rows mode: the compact ks x 32 rt), +32 the B group slice (fp16, or
    the E4M3 codes with `b8`), +64 the row block's C drain (`k_gemm_gs(drain="dma")`; + its 192 B of row maxima with `rowmax`),
    +96 the slice's scale table (`bscale`, ns x 128 B, or ns x 64 B with `scales="single"`), +128 (`prefetch`: k_gemm_gs rows
    mode) a slice's codes and scales in one."""
    scl = gs_scale_bytes(ns, ks, q8, scales)                                     # a slice's scale table
    ws = [D.desc_words(ks * 96 if a_bytes is None else a_bytes), D.desc_words(ns * ks * (64 if b8 else 128)), D.desc_words(ns * 768 + (192 if rowmax else 0))] + ([D.desc_words(scl)] if bscale else [])
    if prefetch or bscale: ws.append(D.desc_words(ns * ks * 64 + scl))            # +128: a slice's codes + scales in one request
    out = bytearray(32 * len(ws))
    for i, w in enumerate(ws): out[32 * i:32 * i + 24] = np.asarray(w, np.uint32).tobytes()
    return bytes(out)


def pack_b_group_bscale(b_e4m3: np.ndarray, scale_inv: np.ndarray, group: int, ns: int, ks: int, scales: str = "dup") -> np.ndarray:
    """B `[N, K]` E4M3 codes with a scale per 128 x 128 block (`scale_inv[N/128, K/128]`, e.g. a `weight_scale_inv`) ->
    `k_gemm_gs(b8=True, bscale=True, ks=32)`'s stream for strips `group*ns .. +ns`: per K-slice (= one 128-wide scale block)
    the codes in `pack_b_group_e4m3`'s panel order (ns x ks x 64 B) then the strips' fp32 scale vectors ([s][j][c0 c1 c2 c3
    c0 c1 c2 c3] x 2^8, 128 B a strip; rows past N get 0). N may be short of a multiple of 16 ns; K must be a multiple of 128.
    `scales="single"` (`k_gemm_gs(scales="single")`): each scale once, [s][j][c0 c1 c2 c3] x 2^8, 64 B a strip -- the same
    values, the kernel rebuilds the duplicated lanes (`extl` / `exth`); a slice is ns x ks x 64 + ns x 64 B."""
    N, K = b_e4m3.shape; assert 4 * ks == 128 and K % 128 == 0 and scale_inv.shape == (-(-N // 128), K // 128), (b_e4m3.shape, scale_inv.shape)
    assert scales in SCALE_LAYOUTS, scales
    r0, r1 = 16 * ns * group, 16 * ns * (group + 1)
    codes = np.zeros((16 * ns, K), np.uint8); codes[:max(0, min(N, r1) - r0)] = b_e4m3[r0:min(N, r1)]
    panels = np.ascontiguousarray(codes.reshape(ns, 4, 4, K // 128, ks, 4).transpose(3, 0, 4, 1, 2, 5)).reshape(K // 128, -1)   # [slice][s kk j n k]
    rows = np.arange(r0, r1); valid = rows < N
    sc = np.where(valid[:, None], scale_inv[np.minimum(rows, N - 1) // 128], 0.0).astype(np.float32) * 256.0             # [16 ns, slices]
    sc = sc.T.reshape(K // 128, ns, 4, 4)                                                                                   # [slice][s][j][c]
    scv = np.ascontiguousarray(np.concatenate([sc, sc], -1) if scales == "dup" else sc).reshape(K // 128, -1).view(np.uint8)   # [slice][s j (c c)] / [slice][s j c] fp32
    return np.ascontiguousarray(np.concatenate([panels, scv], 1)).ravel()


def bscale_stream_single(stream: np.ndarray, ns: int, ks: int) -> np.ndarray:
    """A `pack_b_group_bscale` stream of whole groups in the "dup" layout (any number of groups and slices: a run of
    (ns ks 64 + ns 128)-B slice records) -> the same slices in the "single" layout (ns ks 64 + ns 64 B each): the codes
    untouched, the second copy of every scale quad dropped. Byte-identical to packing with `scales="single"` (the
    equality is `pack_b_group_bscale`'s unit check); the repack of an existing cache."""
    bh, dup, one = ns * ks * 64, ns * 128, ns * 64
    s = np.ascontiguousarray(stream).view(np.uint8).ravel(); assert s.size % (bh + dup) == 0, (s.size, bh + dup)
    rec = s.reshape(-1, bh + dup)
    out = np.empty((rec.shape[0], bh + one), np.uint8)
    np.copyto(out[:, :bh], rec[:, :bh])
    sc = rec[:, bh:].view(np.uint32).reshape(-1, ns * 4, 8)[:, :, :4]                 # [s j][c c] -> the first quad (32-bit lanes)
    out[:, bh:] = np.ascontiguousarray(sc).view(np.uint8).reshape(-1, one)
    return out.ravel()


def pack_b_group_q8(q: np.ndarray, d: np.ndarray, group: int, ns: int, ks: int = 8) -> np.ndarray:
    """B `[N, K]` int8 codes with a scale per 32 weights along K (`d[N, K/32]`: GGUF Q8_0) -> `k_gemm_gs(b8=True, bscale=True,
    q8=True, ks=8)`'s stream for strips `group*ns .. +ns`: per K-slice (32 wide = one Q8_0 block) the codes as u = q + 128 in the
    panel order (ns x ks x 64 B), then the strips' scales, 16 fp16 a strip = d x 2^10 (32 B; the kernel widens them to d x 2^24;
    rows past N get code 128 and scale 0)."""
    N, K = q.shape; assert 4 * ks == 32 and K % 32 == 0 and d.shape == (N, K // 32), (q.shape, d.shape)
    assert float(np.abs(d).max(initial=0)) * 1024 < 65504, "Q8_0 scales too large for fp16 x 2^10"
    r0, r1 = 16 * ns * group, 16 * ns * (group + 1); nsl = K // 32; nr = max(0, min(N, r1) - r0)
    codes = np.full((16 * ns, K), 128, np.uint8); codes[:nr] = (q[r0:r0 + nr].astype(np.int16) + 128).astype(np.uint8)
    panels = np.ascontiguousarray(codes.reshape(ns, 4, 4, nsl, ks, 4).transpose(3, 0, 4, 1, 2, 5)).reshape(nsl, -1)   # [slice][s kk j n k]
    sc = np.zeros((16 * ns, nsl), np.float32); sc[:nr] = d[r0:r0 + nr].astype(np.float32) * np.float32(1024.0)
    sc16 = np.ascontiguousarray(sc.T.astype(np.float16)).reshape(nsl, -1).view(np.uint8)                                    # [slice][s][16 cols]
    return np.ascontiguousarray(np.concatenate([panels, sc16], 1)).ravel()


def pack_b_group_q8f(q: np.ndarray, d: np.ndarray, group: int, ns: int, ks: int = 32) -> np.ndarray:
    """B `[N, K]` GGUF Q8_0 (int8 q, `d[N, K/32]`) -> `k_gemm_gs(b8=True, bscale=True, q8f=True, ks)`'s stream for strips
    `group*ns .. +ns`: per K-slice (4 ks wide, ks / 8 Q8_0 blocks) the codes as u = q + 128 in the panel order (ns x ks x 64 B),
    then the scales [s][block][16 cols] fp16 d (32 B a strip and block). The kernel builds each weight in fp16 -- 1024 + u by a
    zip under 0x64, - 1152 (exact: q), x d (one rounding) -- so a slice needs no scale or correction pass. Rows past N get code
    128 and scale 0."""
    N, K = q.shape; nb = ks // 8; assert ks % 8 == 0 and K % (4 * ks) == 0 and d.shape == (N, K // 32), (q.shape, d.shape)
    assert float(np.abs(d).max(initial=0)) < 65504, "Q8_0 scales beyond fp16"
    r0, r1 = 16 * ns * group, 16 * ns * (group + 1); nsl = K // (4 * ks); nr = max(0, min(N, r1) - r0)
    codes = np.full((16 * ns, K), 128, np.uint8); codes[:nr] = (q[r0:r0 + nr].astype(np.int16) + 128).astype(np.uint8)
    panels = np.ascontiguousarray(codes.reshape(ns, 4, 4, nsl, ks, 4).transpose(3, 0, 4, 1, 2, 5)).reshape(nsl, -1)   # [slice][s kk j n k]
    sc = np.zeros((16 * ns, K // 32), np.float16); sc[:nr] = d[r0:r0 + nr].astype(np.float16)
    sc16 = np.ascontiguousarray(sc.reshape(ns, 16, nsl, nb).transpose(2, 0, 3, 1)).reshape(nsl, -1).view(np.uint8)    # [slice][s][block][16 cols]
    return np.ascontiguousarray(np.concatenate([panels, sc16], 1)).ravel()


def reference_group(a_f16: np.ndarray, b_e4m3: np.ndarray, group: int, ns: int) -> np.ndarray:
    a = (a_f16.view(np.float16) if a_f16.dtype == np.uint16 else a_f16).astype(np.float32)
    b = e4m3_table()[b_e4m3[16 * ns * group:16 * ns * (group + 1)]].astype(np.float32)
    return a @ b.T


def pack_b_group_e4m3(b_e4m3: np.ndarray, group: int, ns: int, ks: int) -> np.ndarray:
    """The E4M3 bytes in `pack_b_group`'s panel order ([slice][s][kk][j][n][k]): `k_e4m3_stream`
    turns this stream into exactly `pack_b_group`'s fp16 panels."""
    N, K = b_e4m3.shape
    b = b_e4m3[16 * ns * group:16 * ns * (group + 1)]
    t = b.reshape(ns, 4, 4, K // (4 * ks), ks, 4)
    return np.ascontiguousarray(t.transpose(3, 0, 4, 1, 2, 5))


# ---- the 48-deep, 3-strip k_gemm_gs over the 24-deep / 6-strip layouts ----
def descriptors_gs48(nrb: int, b8: bool = False) -> bytes:
    """`k_gemm_gs(48, 3, ..., a_rb_stride=2304, c_rb_stride=4608)` over the ks-24 A layout: +0 a row block's 48-step
    A slice gathered from two consecutive 24-step slices (2 x 2304 B, `nrb` x 2304 B apart), +32 a 3-strip B
    slice (18432 B; `b8`: its E4M3 codes, 9216 B), +64 a row block's 3-strip C drain (2304 B)."""
    out = bytearray(96)
    for i, w in enumerate((D.desc_words(4608, width=2304, ext_stride=nrb * 2304, int_stride=2304), D.desc_words(3 * 48 * (64 if b8 else 128)), D.desc_words(3 * 768))):
        out[32 * i:32 * i + 24] = np.asarray(w, np.uint32).tobytes()
    return bytes(out)


def repack_panels_48x3(p: np.ndarray, K: int) -> np.ndarray:
    """Weight panels packed for ks 24 / 6 strips (`[G][K/96 slices][6 strips][24 kk][64 B]`, any element size in
    the 64 bytes: E4M3 codes or fp16 as bytes) -> the 48-deep / 3-strip order `k_gemm_gs(48, 3)` reads, the
    half-groups first: `[h][G][K/192 slices][3 strips][2 x 24 kk][64 B]` (strip 3 h + s of group G)."""
    unit = 64                                                # one (strip, k-step): 4 tiles x 4 columns x 4 k, E4M3 bytes
    G = p.size // (K // 96 * 6 * 24 * unit)
    t = p.reshape(G, K // 192, 2, 2, 3, 24, unit)             # [G][S48][half-slice][h][s][kk][u]
    return np.ascontiguousarray(t.transpose(3, 0, 1, 4, 2, 5, 6)).ravel()   # [h][G][S48][s][half-slice][kk][u]


def descriptors_gm(ks: int = 48, ns: int = 3, b8: bool = True, a_gather_nrb: int = 0, rowmax: bool = False) -> bytes:
    """`k_gemm_gm`: +0 a row block's A slice (ks x 96 B, contiguous in GM; `a_gather_nrb`: gathered from DDR's 24-deep layout,
    ks / 24 pieces of 2304 B that many row blocks apart), +32 a B slice (ns strips; E4M3 codes with `b8`), +64 a C row (ns x 768
    (+ 192 B of maxima with `rowmax`))."""
    out = bytearray(96)
    aw = D.desc_words(ks * 96, width=2304, ext_stride=a_gather_nrb * 2304, int_stride=2304) if a_gather_nrb and ks > 24 else D.desc_words(ks * 96)
    for i, w in enumerate((aw, D.desc_words(ns * ks * (64 if b8 else 128)), D.desc_words(ns * 768 + (192 if rowmax else 0)))):
        out[32 * i:32 * i + 24] = np.asarray(w, np.uint32).tobytes()
    return bytes(out)


def descriptors_gm_stage(nrb: int, ks: int = 48) -> bytes:
    """`k_gm_stage`: +0 the gather of a (slice, row block)'s ks / 24 pieces of 2304 B, nrb x 2304 B apart in the 24-deep A layout; +32 its ks x 96 B out."""
    out = bytearray(64)
    out[0:24] = np.asarray(D.desc_words(ks * 96, width=2304, ext_stride=nrb * 2304, int_stride=2304), np.uint32).tobytes()
    out[32:56] = np.asarray(D.desc_words(ks * 96), np.uint32).tobytes()
    return bytes(out)
