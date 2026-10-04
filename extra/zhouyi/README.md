# The Zhouyi NPU backend (`DEV=ZHOUYI`)

A tinygrad backend for the Zhouyi V3 NPU (AIPU V3: one cluster of 3 cores x 4 TECs on the tested part). Kernels are
tinygrad's generated OpenCL C, compiled by the vendor's AIPU compiler and run as TCB-chain jobs submitted straight to the
kernel driver's `/dev/aipu` (no vendor user-mode runtime). Consecutive launches in a JIT graph are fused into one job.

| where | what |
| --- | --- |
| `tinygrad/runtime/ops_zhouyi.py` | renderer (C dialect, the `mma` TensorCore, a 32-bit `sin` reduction), allocator, `ZhouyiProgram`, device |
| `tinygrad/runtime/graph/zhouyi.py` | `ZhouyiGraph`: fuses chainable launches into one TCB chain per job, frozen replay |
| `tinygrad/runtime/support/compiler_zhouyi.py` | `ZhouyiCompiler`: the installed toolchain in-process, the image's vector table and task epilogue, ELF lift |
| `tinygrad/runtime/support/zhouyi/` | `dev.py` raw ioctl submit path and TCB chains, `launch.py` per-launch buffers, `elf.py`, `hangid.py` |
| `tinygrad/runtime/autogen/__init__.py` | `aipu`: the KMD bindings, generated on first use (below) |
| `extra/zhouyi/ops.py` | custom ops, lowered from `Ops.CUSTOM_FUNCTION` (importing it installs them; each is a graph chain member): `gemm_gs` (hand-written fp16 / E4M3 / Q8_0 / ternary GEMM on the matrix unit, `GemmGSRunner`; `GemmGMRunner` with `ZHOUYI_GM=1`), `e4m3_stream`, `register_csrc` / `csrc_call` (a hand-written C kernel as an ordinary Program), `host_invalidate` |
| `extra/zhouyi/kern_tpc.py` | TEC kernels as (assembly, encoder) pairs: `k_gemm_gs` and its variants, `k_gemm_gm`, `k_gm_stage`, `k_e4m3_stream`; ALU / vector / LSRAM / GSRAM / GM / DMA / sync / control-register probes and benchmarks |
| `extra/zhouyi/gemm_fp16.py` | the GEMM operand packers (`pack_a_slices`, `pack_b_group*` incl. `pack_b_group_tern`, `repack_panels_48x3`, `bscale_stream_single`), `unpack_c_gs`, descriptor tables |
| `extra/zhouyi/gemm_plan.py`, `gemm_plan_coeffs.json` | an analytic cost model of `gemm_gs` and `plan` / `plan_piece`. The JSON holds per-operation costs (cycles per `mma`, per GSRAM vector, per DMA request, DDR rate, ...) fitted to device timings of 111 calibration configurations: a generic hardware cost model, not tied to any model's shapes |
| `extra/zhouyi/vec_f16.py` | helpers for hand-written vector kernels: the C header (`H`, `FULL_H`), the GEMM layouts and numpy references, `_desc_slots`, `dma_copy_src`, `tec_sum_src` |
| `extra/zhouyi/enc.py`, `image.py` | the VLIW encoder; the image writer (`text_only`) and `build_encoded`, which byte-diffs an image against the installed assembler |
| `extra/zhouyi/dma.py`, `tec_res.py` | DMA descriptors; checked sync flags, a build-time sync lint, LSRAM / GSRAM / descriptor arenas |
| `extra/zhouyi/timing.py`, `toolchain.py` | `submit_and_wait` and the process-wide timing `Recorder`; the toolchain as an assembler oracle |

Models built on this backend (and their model-specific kernels) live in the examples repository.

## `gemm_gs` weight formats

`gemm_gs(a, b, ...)` computes `C = A @ B^T` with fp16 A (the `pack_a_slices` layout, or the compact rows-mode layout) and fp32 C.
The weight stream `b` is one of these; the caller passes the flags its stream was packed with.

| format | packer | `gemm_gs` flags | bytes / weight | C |
| --- | --- | --- | ---: | --- |
| fp16 panels | `pack_b_group` (`repack_panels_48x3` for `lin48`) | (none) | 2 | fp32 accumulation |
| E4M3 codes, the caller scales | `pack_b_group_e4m3` | `b8=True` | 1 | x 2^-8 |
| E4M3, a scale per (column, 128-K block) | `pack_b_group_bscale` | `b8, bscale` (+ `scales="single"`: the 64 B-a-strip table) | 1.03 | fully scaled |
| GGUF Q8_0 (int8, a scale per 32) | `pack_b_group_q8` / `pack_b_group_q8f` | `b8, bscale, q8=1` (ks 8) / `q8=2` | 1.06-1.13 | fully scaled |
| **ternary g128** ({-1, 0, +1}, a scale per (column, 128-K group)) | `pack_b_group_tern(w, scale, group, ns, ks=32)` | `b8, bscale, scales="single", tern=True` (ks 32) | 0.28 | fully scaled |

**Ternary.** `w` is int8 `[N, K]` in {-1, 0, +1}, `scale` `[N, K / 128]` (fp16 values; any real scale the format carries, e.g.
a GGUF block scale). The stream stores the 2-bit codes `c = w + 1`, four per byte across a strip's four column tiles, then the
single-layout fp32 table `scale x 2^18` per K-slice. The kernel expands the codes into fp16 subnormal panels with byte adds and
widening multiplies (no shifts or masks), runs the ordinary fp16 k-loop, and after it undoes the tile mixing
(`C(j) -= C(j+1) / 4`) and the `+1` offset (a row-sum pre-pass, as Q8_0's) before the scale multiply. The codes and both
corrections are exact (integer fp16 lanes, power-of-two steps), so C is the matrix unit's fp32 product `A_slice @ w_slice` times
`fp32(scale)`, accumulated slice by slice in fp32, as in the E4M3 block-scaled path. Rows past N are packed as code 1 (w = 0)
with scale 0. Against the E4M3 stream (1.03 B a weight) the ternary one moves a quarter of the bytes, and on the device the GEMM
is bound by the expand rather than by DDR (K 5120, N 34 816, 8 rows: 5.75 ms against 8.53 ms). Any input transform the format implies (e.g. a rotation of the activations
before a rotated-basis weight) is the caller's: it belongs in the kernel that produces A.

## Requirements

* **Linux on the NPU's host (arm64) with the Zhouyi KMD loaded** (`/dev/aipu`). `RawDevice.cache_invalidate` relies on a
  KMD whose `BUF_CACHE_INVALID` maintains the buffer's cacheable linear-map alias; with a driver that does not, host reads
  of device output can be stale. `wbuf_alloc` / `slot_*` (zero-copy weights) need a KMD with the `WBUF_*` / `SLOT_*`
  ioctls (numbers 30-34) and are optional: an older driver answers `ENOTTY`.
* **The KMD's UAPI header `armchina_aipu.h`** (not bundled). It is looked up, in order, at `ZHOUYI_AIPU_HEADER`,
  `/usr/src/zhouyi-aipu-<version>/include/armchina_aipu.h` (the driver's DKMS source package; the newest version wins),
  `/usr/include/misc/armchina_aipu.h`, `/usr/include/linux/armchina_aipu.h`, and the vendor NPU SDK's
  `/usr/share/cix/include/npu/kmd/armchina_aipu.h`. Prefer the header of the driver you run.
* **libclang** (tinygrad's own bindings, `tinygrad/runtime/autogen/libclang.py`: `libclang` from LLVM 20 or the system
  `clang`), to generate the bindings once. Distributions that ship only a versioned library (Debian / Ubuntu `libclang-NN-dev`
  installs e.g. `/usr/lib/llvm-21/lib/libclang-21.so`, no plain `libclang.so`) need it named: `LIBCLANG_PATH=<that file>` for the
  first run (tinygrad's library lookup; the generated `aipu.py` is reused afterwards).
* **The vendor's AIPU compiler toolchain**: `libaiputoolchain.so` with its siblings `libaipu_toolchain_core.so` and
  `libaipu_buildtool.so` (`btaipucc` / `btaipuas` / `btaipuld`), as installed by the NPU SDK's `cix-npu-onnxruntime`
  package in `/usr/share/cix/lib/onnxruntime`; `ZHOUYI_TOOLCHAIN_DIR` points elsewhere. It is loaded in-process,
  `RTLD_LOCAL` (it carries its own LLVM 11).
* `numpy` for `extra/zhouyi`.

### The `aipu` bindings

`tinygrad/runtime/autogen/aipu.py` is generated by tinygrad's autogen (`autogen.load`, libclang) from the header above the
first time `tinygrad.runtime.autogen.aipu` is imported; `REGEN=1` regenerates it. Off Linux (e.g. to run the host tests on
a Mac), `linux/types.h` and `linux/ioctl.h` come from the Debian `linux-libc-dev` package, fetched as for the `pci` / `vfio`
bindings. **Deviation from upstream:** tinygrad commits its generated bindings; this one is `.gitignore`d, because the
header is the driver's (GPL-2.0 WITH Linux-syscall-note) and the bindings must match the driver installed.

## Environment variables

| var | default | effect |
| --- | --- | --- |
| `ZHOUYI_AIPU_HEADER` | lookup above | the KMD UAPI header the bindings are generated from |
| `ZHOUYI_TOOLCHAIN_DIR` | `/usr/share/cix/lib/onnxruntime` | the directory holding `libaiputoolchain.so` |
| `ZHOUYI_CORES` | 3 | cores a launch may use (12-task launches need 3) |
| `ZHOUYI_CHAIN_MAX` | 8 | launches fused into one job by `ZhouyiGraph`; 1 runs every launch as its own job (pins a hang to one kernel) |
| `AIPU_JOB_TIMEOUT` | 10 | seconds before a job counts as hung (a fault presents as a hang) |
| `ZHOUYI_HANG_ID` | 0 | hung-job report: `1` decodes the failed submission's chains, args and exception vectors (no device-side change); `2` also stamps every compiled image's start / end cycles, so the first kernel that never finished is named (recompiles everything once); `<level>:<path>` also appends each report to a file |
| `ZHOUYI_GM` | 0 | 1: `gemm_gs` lowers to `GemmGMRunner` (A chunks and partials in the cluster's GM) |
| `ZHOUYI_GEMM_DRAIN` | `dma` | `GemmGSRunner`'s C drain for fp16 B: `dma` (staged through LSRAM) or `tec` |
| `ZHOUYI_GEMM_TIMING` | unset | rows-mode diagnostic (`noc`, `dmaonly`, `nodma`, `gsl<n>` / `gso<n>` / `gss<n>`): **wrong results**, timing only |

A custom-op runner reads its switches once, when built; they are part of the runner cache key (`ENV_KEYS`).

## Host tests (no device)

```sh
python3 -m pytest test/testextra/test_zhouyi_gemm_plan.py test/testextra/test_zhouyi_runner_cache.py \
  test/testextra/test_zhouyi_tec_res.py test/testextra/test_zhouyi_timing.py test/unit/test_zhouyi_hangid.py
```

They import the bindings, so the first run needs the header (`ZHOUYI_AIPU_HEADER=/path/to/armchina_aipu.h` off the device).
