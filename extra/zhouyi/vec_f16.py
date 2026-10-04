"""Helpers for hand-written vector kernels in the renderer's C dialect (run through `ops.register_csrc` / `csrc_call`): the
fp32 / fp16 vector header (`H`, `FULL_H` with the DMA macros), the GEMM operand layouts and their numpy references, the DMA
descriptor-table builder (`_desc_slots`), and two generic kernels (`dma_copy_src`, `tec_sum_src`). fp16 by `cvtd` (two float8 ->
one half16, round-to-nearest-even: bit-exact vs numpy). Model kernels built on these live with their models (see the examples
repository).

Layouts (halves / floats; `gemm_fp16.pack_a_slices`, `pack_b_group`, `unpack_c_gs`):
  A  (a GEMM's left operand, fp16, `rows` padded to 12*nrb, K = 4 ks nslices):
       (m, j) -> ((((slice*nrb + rb)*ks + kq)*3 + i)*4 + r)*4 + k     slice = j // 4ks, kq = j % 4ks // 4, k = j % 4
                                                                   rb = m // 12, i = m % 12 // 4, r = m % 4
       a 4x4 tile (rows m..m+3 of one row group, k-quad kq) is 32 contiguous bytes.
  B  (the right operand as [N, K] rows, fp16, ns strips per group):
       (n, j) -> group*GB + ((((slice*ns + s)*ks + kq)*4 + jt)*4 + c)*4 + k    s = n // 16 % ns, jt = n % 16 // 4, c = n % 4
                                                                          GB = nslices*ns*ks*64 halves
  C  (the GEMM's output, fp32 tiles, every row block of the whole M, ns strips per group):
       (m, n) -> ((((group*nrb + rb)*ns + s)*12 + i*4 + jt)*16 + r*4 + c      group = n // 16 // ns
       four columns n..n+3 (c = 0..3) are one float4.
Rows split over `nt` tasks in chunks of 6 row groups (2 row blocks) so no two tasks write one 64-byte line
(cross-core stores to one line lose data: write-back per-core L2).
"""
import numpy as np

# The fp32 vector side's header (`float8` arithmetic, the `__builtin_aipu_*_tfp32_pw` builtins). Toolchain constraints: every fp32
# vector builtin takes a PASSTHROUGH first operand and an all-lanes predicate last (`(bool8)(1)`, `ext_vector_type(8) bool`; the
# `vector_size` spelling is refused). There is no exp / exp2 / tanh builtin (`exp2_f8` builds 2^t from `rint`, a degree-6
# polynomial on [-0.5, 0.5] and `scal2`). `log2_tfp32` returns the FLOOR exponent as int. A narrowing vector store (`trunc to
# v8i8`) crashes the backend, so bytes go out through in-place `nsr.as` narrows and `exte` merges.
NT = 4                                   # tasks: one core's four TECs, so the kernel can join a chain group of fused jobs
                                         # (a 12-task program spans three cores, see `graph/zhouyi._chainable`); `nt=12` for a solo launch
LOG2E = 1.4426950408889634

_H32 = r"""
typedef float float8 __attribute__((__vector_size__(32)));
typedef int int8v __attribute__((__vector_size__(32)));
typedef short short16 __attribute__((__vector_size__(32)));
typedef char char32 __attribute__((__vector_size__(32)));
typedef bool bool8 __attribute__((ext_vector_type(8)));
#define ALL ((bool8)(1))
#define VMAX(a, b) __builtin_aipu_max_tfp32_tfp32_pw((a), (a), (b), ALL)
#define VMIN(a, b) __builtin_aipu_min_tfp32_tfp32_pw((a), (a), (b), ALL)
#define VRCP(x)    __builtin_aipu_rcp_tfp32_pw((x), (x), ALL)
#define VRSQRT(x)  __builtin_aipu_rsqrt_tfp32_pw((x), (x), ALL)
#define VRINT(x)   __builtin_aipu_rint_tfp32_pw((x), (x), ALL)
#define VF(n)      __builtin_aipu_cvt_tfp32_tw(n)
#define VI(x)      __builtin_aipu_cvt_tw_tfp32(x)
#define VSCAL2(x, n) __builtin_aipu_scal2_tfp32_tfp32_pw((x), (x), (n), ALL)
#define BC(v)      ((float8){(v), (v), (v), (v), (v), (v), (v), (v)})
/* 2^t for t in [-126, 126]: k = rint(t), f = t - k in [-0.5, 0.5], 2^f by a degree-6 polynomial
   (cephes exp2f), then the exponent by scal2 */
static inline float8 exp2_f8(float8 t) {
  t = VMAX(t, BC(-126.0f)); t = VMIN(t, BC(126.0f));
  float8 k = VRINT(t); float8 f = t - k;
  float8 p = BC(1.535336188319500e-4f);
  p = p * f + BC(1.339887440266574e-3f); p = p * f + BC(9.618437357674640e-3f); p = p * f + BC(5.550332471162809e-2f);
  p = p * f + BC(2.402264791363012e-1f); p = p * f + BC(6.931472028550421e-1f); p = p * f + BC(1.0f);
  return VSCAL2(p, VI(k));
}
/* four int32 lane vectors (values within int8) -> 32 bytes, in place narrows + even extracts */
static inline char32 pack32(int8v w0, int8v w1, int8v w2, int8v w3) {
  short16 h01 = __builtin_aipu_exte_th_th(__builtin_aipu_nsras_thw_i(w0, 0), __builtin_aipu_nsras_thw_i(w1, 0));
  short16 h23 = __builtin_aipu_exte_th_th(__builtin_aipu_nsras_thw_i(w2, 0), __builtin_aipu_nsras_thw_i(w3, 0));
  return __builtin_aipu_exte_tb_tb(__builtin_aipu_nsras_tbh_i(h01, 0), __builtin_aipu_nsras_tbh_i(h23, 0));
}
/* rint(x) clipped to [-127, 127] as int32 lanes */
static inline int8v q8v(float8 x) { x = VRINT(x); x = VMAX(x, BC(-127.0f)); x = VMIN(x, BC(127.0f)); return VI(x); }
static inline float hsum8(float8 v) { return v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7]; }
static inline float hmax8(float8 v) { float m = v[0]; m = v[1] > m ? v[1] : m; m = v[2] > m ? v[2] : m; m = v[3] > m ? v[3] : m;
  m = v[4] > m ? v[4] : m; m = v[5] > m ? v[5] : m; m = v[6] > m ? v[6] : m; m = v[7] > m ? v[7] : m; return m; }
"""

H = _H32 + r"""
typedef float float4 __attribute__((__vector_size__(16)));
typedef half half8 __attribute__((__vector_size__(16)));
typedef half half16 __attribute__((__vector_size__(32)));
typedef half half4 __attribute__((__vector_size__(8)));
#define CVT16(lo, hi) __builtin_aipu_cvtd_tfp16_tfp32_tfp32((lo), (hi))
/* four rows' float8 (8 columns = 2 k-quads) -> the two 4x4 fp16 tiles (rows x k) of quads 0 and 1 */
#define TILES2(x0, x1, x2, x3, t0, t1) { half16 h01_ = CVT16((x0), (x1)), h23_ = CVT16((x2), (x3)); \
  t0 = __builtin_shufflevector(h01_, h23_, 0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27); \
  t1 = __builtin_shufflevector(h01_, h23_, 4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31); }
/* a C tile row's 8 columns (two adjacent tiles jt, jt+1 of one strip) as a float8 */
#define CT8(p) __builtin_shufflevector(*(__global float4*)(p), *(__global float4*)((p) + 16), 0,1,2,3,4,5,6,7)
/* the 4-row x 8-column block (x0..x3) transposed: column d's four rows as a half4 (V^T panels) */
#define COL4(h01, h23, d) __builtin_shufflevector((h01), (h23), (d), 8 + (d), 16 + (d), 24 + (d))
"""


# ---- index helpers (C expressions and numpy) ----


def a_layout_ref(x: np.ndarray, ks: int, nrb: int) -> np.ndarray:
    """numpy: fp32 `[rows, K]` (rows <= 12 nrb, padded with zeros) -> the A layout as uint16 halves."""
    from .gemm_fp16 import pack_a_slices
    xp = np.zeros((12 * nrb, x.shape[1]), np.float16); xp[:x.shape[0]] = x.astype(np.float16)
    return pack_a_slices(xp, ks).ravel()


def c_tiles_ref(c: np.ndarray, ns: int, nrb: int) -> np.ndarray:
    """numpy: fp32 `[12 nrb, N]` -> the C tile layout (float32, ravel)."""
    M, N = c.shape; ngroups = N // (16 * ns)
    t = c.reshape(nrb, 3, 4, ngroups, ns, 4, 4)                         # [rb][i][r][g][s][jt][cc]
    return np.ascontiguousarray(t.transpose(3, 0, 4, 1, 5, 2, 6)).ravel().astype(np.float32)


def c_from_tiles(ct: np.ndarray, ns: int, nrb: int, N: int) -> np.ndarray:
    ngroups = N // (16 * ns)
    t = ct.reshape(ngroups, nrb, ns, 3, 4, 4, 4)
    return np.ascontiguousarray(t.transpose(1, 3, 5, 0, 2, 4, 6)).reshape(12 * nrb, N)


def b_layout_ref(b: np.ndarray, ks: int, ns: int) -> np.ndarray:
    """numpy: fp32 `[N, K]` -> the B panels (uint16), all groups contiguous."""
    N, K = b.shape; ng = N // (16 * ns)
    t = b.astype(np.float16).view(np.uint16).reshape(ng, ns, 4, 4, K // (4 * ks), ks, 4)   # [g][s][jt][c][slice][kq][k]
    return np.ascontiguousarray(t.transpose(0, 4, 1, 5, 2, 3, 6)).ravel()


# ==== DMA-staged versions ====
# These stage every operand through LSRAM by DMA and skip TILES2's shuffles: a C tile ([r][c]) IS an A tile ([r][k])
# and a B tile ([c][k]), so conversion is `cvtd` of its two float8 halves. Descriptors: one 64-byte slot per (shape,
# flag) (`dma.desc_words`); at most one request in flight per flag (four flags). LSRAM: 32 KiB per TEC.
DMA_H = r"""
#define LSRAM_BASE 0xFA000000
#define LSF(off) ((__global float*)(LSRAM_BASE + (off)))
#define LSH(off) ((__global half*)(LSRAM_BASE + (off)))
#define DMA_FILL(flag, desc, ls_off, ext)  __builtin_aipu_dma_inter(0, (flag) * 257, 0, 4, 0, 1, 0, (desc), LSRAM_BASE + (ls_off), (ext))
#define DMA_DRAIN(flag, desc, ls_off, ext) __builtin_aipu_dma_inter(0, (flag) * 257, 0, 4, 0, 0, 0, (desc), LSRAM_BASE + (ls_off), (ext))
#define DMA_WAIT(flag) __builtin_aipu_wfe_inter(~(1 << (flag)), 1)
#define DMA_WAIT_ALL() __builtin_aipu_wfe_inter(0, 2)
#define DESC(d, i) ((int)(d) + 64 * (i))
/* a C tile's two halves (rows 0,1 / rows 2,3) -> the fp16 tile */
#define TILE16(p) CVT16(*(__global float8*)(p), *(__global float8*)((p) + 8))
/* rows (a, b) as a tile-half scalar vector: lanes 0..3 = a, 4..7 = b */
#define ROWS2(a, b) ((float8){(a), (a), (a), (a), (b), (b), (b), (b)})
"""
FULL_H = H + DMA_H


def _desc_slots(*shapes) -> np.ndarray:
    """int32 `[16 * n]`: slot i = `desc_words(*shapes[i])` (a (size[, width, ext_stride, int_stride]) tuple)."""
    from extra.zhouyi import dma as D
    out = np.zeros(16 * len(shapes), np.int32)
    for i, sh in enumerate(shapes): out[16 * i:16 * i + 6] = np.asarray(D.desc_words(*sh), np.uint32).view(np.int32)
    return out


def dup_quads(v: np.ndarray) -> np.ndarray:
    """fp32 `[K]` -> `[K/4, 8]`: each quad twice (the per-column factor of a tile's float8 half)."""
    q = v.astype(np.float32).reshape(-1, 4); return np.ascontiguousarray(np.concatenate([q, q], 1)).ravel()


def tile_rows(x: np.ndarray, nrb: int) -> np.ndarray:
    """fp32 `[rows, K]` (zero-padded to 12 nrb) -> `[3 nrb row groups][K / 4 quads][4][4]`: 16-float tiles (cos / sin order)."""
    xp = np.zeros((12 * nrb, x.shape[1]), np.float32); xp[:x.shape[0]] = x
    return np.ascontiguousarray(xp.reshape(3 * nrb, 4, x.shape[1] // 4, 4).transpose(0, 2, 1, 3)).ravel()


# ---- the host-visible copy ----


def tec_copy_descs() -> np.ndarray: return _desc_slots((8192,))


# Ints per task in `tec_sum`'s result: one 64-byte line each, so no two tasks share a line.
TEC_SUM_SLOT = 16


def tec_sum_src(nwords: int, nt: int = NT) -> str:
    """Per-task int32 sums of a buffer at `res[t * TEC_SUM_SLOT]`, the call's salt at `res[nt * TEC_SUM_SLOT]`: checksums
    a host read can be verified against. All reads by DMA (cached loads can be stale).
    Note: one 64-byte line per task is load-bearing (cross-core stores to one line lose data); do not pack.
    args: res (`(nt + 1) * TEC_SUM_SLOT` ints), src, desc (`tec_copy_descs`), salt (2048 ints)."""
    assert nwords % (2048 * nt) == 0
    per = nwords // nt; nch = per // 2048
    return FULL_H + f"""
__kernel void tec_sum(__global int* restrict res, __global int* restrict src, __global int* restrict desc, __global int* restrict salt, const int core_id) {{
  int acc = 0;
  DMA_FILL(0, DESC(desc, 0), 0, (int)(src + core_id * {per}));
  for (int c = 0; c < {nch}; c++) {{
    int s = c & 1;
    if (c + 1 < {nch}) DMA_FILL(s ^ 1, DESC(desc, 0), (s ^ 1) * 8192, (int)(src + core_id * {per} + (c + 1) * 2048));
    DMA_WAIT(s);
    __global int* p = (__global int*)LSF(s * 8192);
    for (int i = 0; i < 2048; i++) acc += p[i];
  }}
  DMA_WAIT_ALL();
  DMA_FILL(0, DESC(desc, 0), 16384, (int)salt); DMA_WAIT(0);
  res[core_id * {TEC_SUM_SLOT}] = acc; if (core_id == 0) res[{nt * TEC_SUM_SLOT}] = ((__global int*)LSF(16384))[0];
}}"""


def dma_copy_src(nbytes: int, out_off: int = 0, nt: int = NT) -> str:
    """`out[out_off:out_off + nbytes] = src[:nbytes]` through LSRAM in 8 KiB chunks. args: out, src, desc (`dma_copy_descs`)."""
    n, tail = divmod(nbytes, 8192)
    return FULL_H + f"""
__kernel void dma_copy(__global char* restrict out, __global char* restrict src, __global int* restrict desc, const int core_id) {{
  for (int u = core_id; u < {n + (1 if tail else 0)}; u += {nt}) {{
    int d = (u < {n}) ? 0 : 1;
    DMA_FILL(0, DESC(desc, d), 0, (int)(src + u * 8192)); DMA_WAIT_ALL();
    DMA_DRAIN(0, DESC(desc, d), 0, (int)(out + {out_off} + u * 8192)); DMA_WAIT_ALL();
  }}
}}"""


def dma_copy_descs(nbytes: int) -> np.ndarray:
    tail = nbytes % 8192
    assert tail % 16 == 0
    return _desc_slots((8192,), (tail or 16,))
