"""An analytic cost model of `gemm_gs` (`kern_tpc.k_gemm_gs` through `ops.GemmGSRunner`), and `plan()`, which picks the cheapest
feasible configuration for a call site.

The model is structural: every term is an operation COUNT read off the kernel generator for the configuration (k-steps,
GSRAM C vectors, E4M3 expand iterations, DMA bytes and requests, strips, row blocks, slices, groups, launches), times a
per-operation COST in cycles of the 1.2034 GHz TEC clock. The costs start at measured silicon rates (the `PRIOR` column)
and are FITTED to device timings only (`fit` over measured calls; never a simulator's timing). Per task (one TEC):

  compute   = kloop + gsram + expand + bscale + stage + overhead
  dma       = exposed B fills (the non-prefetch path waits for a slice's B right after requesting it; the prefetch path
              overlaps each step's request with the step's compute: max per step)
  task      = compute + dma                     (non-prefetch)   |   sum over steps of max(compute, dma)   (prefetch)

and per launch (<= 12 tasks, concurrent): max(slowest task, the launch's DDR bytes / the shared DDR rate); the call adds a
cost per launch and one per call. `predict()` reports every term (cycles of the critical task, and seconds), the max and
the sum, and names the BOUNDING term: the largest of {ddr, dma, kloop, gsram, expand, bscale, stage, overhead, launch}.
"""
from __future__ import annotations
import json, math, os
from dataclasses import dataclass, field, replace, asdict

CLOCK = 1.2034e9                       # TEC cycles per second (measured: the cycle counter against wall time)

# per-operation costs: name -> (prior, source). Units: cycles unless noted.
PRIOR = {
  "mma":      (1.046, "12.55 cycles per 3x4 k-step of 6 paired mma bundles (12 cycles of issue) -> x1.046"),
  "gs_vec":   (7.3,   "a 32-B vector to or from GSRAM at 4.4 B/cycle"),
  "exp_bun":  (1.0,   "the E4M3 expand loop: 26 bundles per 128 codes, inside the 32-bundle loop buffer"),
  "bs_bun":   (1.0,   "bscale: the scale loads, multiplies and adds, one bundle each"),
  "ls_vst":   (1.5,   "vst to LSRAM at 21.3 B/cycle (the C staging at the last slice)"),
  "ddr_vst":  (40.0,  "TEC vector stores to DDR (through the data cache) ~8x the unpack (drain='tec' only)"),
  "strip":    (24.0,  "per strip visit: pointer setup, loop entry, the C base bundles"),
  "rb":       (16.0,  "per row block (pair): waits, the A request, the half swap"),
  "slice":    (40.0,  "per K-slice: the B request, counters"),
  "group":    (120.0, "per group: C zeroing / setup, the C out advance"),
  "req":      (350.0, "a DMA request's fixed cost (~350 cycles at 64 B)"),
  "tec_rate": (13.2,  "one TEC's DMA fill rate, B/cycle"),
  "ddr_rate": (19.9,  "the device-wide DDR ceiling ~24 GB/s = 19.9 B/cycle, shared by every TEC"),
  "launch":   (1.0e-4, "a job boundary ~21-27 us, a group boundary 15-25 us; seconds per launch"),
  "call":     (5.0e-5, "the host's submit-and-wait around a runner call; seconds"),
}
TERMS = ("kloop", "gsram", "expand", "bscale", "stage", "overhead", "dma")


def coeffs(path: str | None = None) -> dict:
  """The fitted costs (gemm_plan_coeffs.json beside this file, written by `fit`), else the priors."""
  path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm_plan_coeffs.json")
  c = {k: v[0] for k, v in PRIOR.items()}
  if os.path.exists(path): c.update(json.load(open(path))["coeffs"])
  return c


@dataclass(frozen=True)
class Config:
  """One `gemm_gs` call as `GemmGSRunner` runs it. `ks`/`ns` are the operand LAYOUT's (the packers bake them in); for `lin48`
  the layout is 24 x 6 and the kernel runs 48 x 3 over half-groups. `piece` 0 = the runner's default. `scales`: the block-scale
  table's layout in the B stream (bscale): "dup" (128 B a strip and slice) or "single" (64 B), as `gemm_gs(scales=)`."""
  K: int; N: int; nrb: int
  ks: int = 24; ns: int = 6
  piece: int = 0
  b8: bool = False; bscale: bool = False; lin48: bool = False; rowmax: bool = False
  rows: int = 0
  drain: str = "dma"
  timing: str | None = None
  heads: int = 1
  scales: str = "dup"

  # ---- the runner's geometry (ops.GemmGSRunner.__init__), kept in step with it
  @property
  def kks(self): return 48 if self.lin48 else self.ks        # the kernel's ks / ns
  @property
  def kns(self): return 3 if self.lin48 else self.ns
  @property
  def kw(self): return 4 if self.bscale else 3
  @property
  def nslices(self): return self.K // (4 * self.kks)
  @property
  def ngroups(self): return self.N // (16 * self.ns) * (2 if self.lin48 else 1)   # kernel groups (lin48: half-groups)
  @property
  def nrbh(self):
    if self.piece: return self.piece
    return max(p for p in range(2, 29, 2) if self.nrb % p == 0) if self.lin48 else self.nrb // 2
  @property
  def rt(self): return -(-self.rows // 4) if self.rows else 3
  @property
  def single(self): return bool(self.rows)
  @property
  def pf(self): return self.single and self.rt <= 2 and self.b8 and self.bscale and self.timing is None
  @property
  def bh(self): return self.kns * self.kks * (64 if self.b8 else 128)
  @property
  def sclb(self): return self.kns * (128 if self.scales == "dup" else 64) if self.bscale else 0   # gemm_fp16.gs_scale_bytes (E4M3)
  @property
  def crow(self): return self.kns * 768 + (192 if self.rowmax else 0)
  @property
  def a_slice(self): return self.kks * 32 * self.rt if self.single else self.nrbh * self.kks * 96   # A bytes a task reads a slice
  def tasks(self):
    """[(groups in the task's run)] per task, and the launches (12 tasks each), as the runner plans them."""
    halves = 1                                   # lin48's halves are already counted in ngroups here (kernel groups)
    npieces = self.nrb // self.nrbh
    g_total = self.N // (16 * self.ns); hp = self.heads * npieces * (2 if self.lin48 else 1); best = None
    for r in range(1, g_total + 1):
      runs = [g_total // r + (1 if i < g_total % r else 0) for i in range(r)]
      nl = -(-(hp * r) // 12); cost = nl * (max(runs) + 2)
      if best is None or cost < best[0]: best = (cost, runs)
    runs = best[1]
    tasks = [g for _ in range(hp) for g in runs]
    return tasks, -(-len(tasks) // 12)

  def feasible(self) -> tuple[bool, str]:
    """The runner's asserts, then the kernel generator itself (tec_res's LSRAM / GSRAM arenas decide what fits)."""
    try:
      if self.bscale and not (self.b8 and not self.lin48 and not self.rowmax and self.ks == 32): return False, "bscale: b8, ks=32, no lin48/rowmax"
      if self.scales not in ("dup", "single") or (self.scales == "single" and not self.bscale): return False, "scales: 'dup', or 'single' with bscale"
      if self.lin48 and not (self.ks == 24 and self.ns == 6 and self.nslices % 2 == 0 and not self.rowmax): return False, "lin48: 24 x 6 layouts, K a multiple of 192"
      if self.nrb % self.nrbh or self.nrbh % 2: return False, f"piece {self.nrbh} does not divide nrb {self.nrb} evenly"
      if self.nrbh * self.kns * 768 > 65536: return False, "C blocks exceed the 64 KiB GSRAM window"
      if self.rows and (self.nrb // self.nrbh != 1 or self.lin48 or self.rowmax or not 1 <= self.rows <= 12): return False, "rows: one row piece, 1..12"
      if self.K % (4 * self.kks) or self.N % (16 * self.ns): return False, "K / N not multiples of the slice / group"
      from extra.zhouyi import kern_tpc as KT
      kw = dict(c_stride=self.nrb * self.crow, drain="dma" if self.lin48 else self.drain, rowmax=self.rowmax, b8=self.b8)
      if self.lin48: KT.k_gemm_gs(48, 3, self.nrbh, c_stride=self.nrb * 4 * 768, drain="dma", a_rb_stride=2304, c_rb_stride=4608, b8=self.b8)
      else: KT.k_gemm_gs(self.ks, self.ns, self.nrbh, kw=self.kw, bscale=self.bscale, rows=self.rows or None, timing=self.timing, scales=self.scales, **kw)
      return True, ""
    except Exception as e:
      return False, f"{type(e).__name__}: {str(e).splitlines()[0]}"


@dataclass
class Prediction:
  seconds: float
  bound: str
  terms_s: dict                    # seconds per term, summed over launches (critical task)
  launches: int
  tasks: int
  config: Config
  def line(self) -> str:
    t = " ".join(f"{k} {v * 1e3:.3f}" for k, v in sorted(self.terms_s.items(), key=lambda kv: -kv[1]) if v > 0)
    return f"{self.seconds * 1e3:8.3f} ms  bound by {self.bound:8s} | {t} (ms) | {self.launches} launches, {self.tasks} tasks"


def task_cycles(c: Config, G: int, ntask: int, k: dict) -> tuple[float, dict, float]:
  """One task with G groups, `ntask` tasks sharing the DDR: (cycles, terms in cycles, DDR bytes)."""
  S, ns, ks, kw, RT = c.nslices, c.kns, c.kks, c.kw, c.rt
  U = 1 if c.single else c.nrbh                                    # row-block visits per slice
  NC = 8 * RT
  visits = G * S * U * ns                                           # strip visits (C block x slice)
  # k-loop: per k-step, the mma bundles cost 2 cycles each (paired), the others 1; one pointer-advance bundle per kw k-steps
  n_mma = 4 * RT; n_bun = 6 if RT == 3 else max(2 * RT, 3)
  kstep = n_mma / 2 + n_bun
  compute_off = c.timing == "dmaonly"
  kloop = 0.0 if compute_off else visits * (ks * kstep * k["mma"] + ks / kw)
  # GSRAM C traffic: loads + stores of NC vectors per visit; drain="dma" skips the loads at the first slice (zeroed
  # registers) and replaces the stores at the last by the LSRAM staging; drain="tec" zeroes and drains through GSRAM.
  if c.drain == "dma" or c.lin48:
    gs_vec = G * U * ns * NC * (2 * S - 2)
    stage = G * U * ns * 24 * k["ls_vst"]
  else:
    gs_vec = G * U * ns * NC * (2 * S) + G * U * ns * 24 * 2      # + the zeroing stores and the drain loads
    stage = G * U * ns * 24 * k["ddr_vst"]
  gsram = 0.0 if compute_off else gs_vec * k["gs_vec"]
  stage = 0.0 if compute_off else stage
  expand = 0.0 if (compute_off or not c.b8) else G * S * (c.bh / 128) * 26 * k["exp_bun"]
  bscale = 0.0 if (compute_off or not c.bscale) else visits * (4 + 4 * RT + NC / 2) * k["bs_bun"]
  overhead = 0.0 if compute_off else visits * k["strip"] + G * S * U * k["rb"] / (1 if c.single else 2) + G * S * k["slice"] + G * k["group"]
  # DMA: B per (group, slice); A per (group, slice) for the task's rows (rows mode: compact A, once per block of groups when
  # prefetching); C out per group. A task pays its fills at its own engine's rate (`tec_rate`); the device-wide DDR ceiling
  # is applied once, per launch (`predict`): a TEC that computes leaves its share to the ones that fetch.
  a_bytes = (G / min(G, 16) if c.pf else G) * S * c.a_slice
  b_bytes = G * S * (c.bh + c.sclb)
  c_bytes = G * U * 12 * ns * 64 * (1 if not c.single else RT / 3)
  ddr_bytes = 0.0 if c.timing == "nodma" else a_bytes + b_bytes + c_bytes            # "nodma" issues no request at all
  rate = k["tec_rate"]
  nodma = c.timing == "nodma"
  comp = kloop + gsram + expand + bscale + stage + overhead
  steps = G * S
  fill = (c.bh + c.sclb) / rate + k["req"]                         # one step's B request
  if nodma: dma = 0.0
  elif c.pf: dma = max(0.0, steps * (fill + c.a_slice / rate / min(G, 16)) - comp)   # overlapped: sum of max(compute, fill) per step
  else:
    dma = steps * fill                                              # waited right after the request
    if c.single: dma += steps * max(0.0, c.a_slice / rate - expand / max(1, steps))
  terms = dict(kloop=kloop, gsram=gsram, expand=expand, bscale=bscale, stage=stage, overhead=overhead, dma=dma)
  return comp + dma, terms, ddr_bytes


def predict(c: Config, k: dict | None = None) -> Prediction:
  k = coeffs() if k is None else k
  tasks, nl = c.tasks()
  total, terms_s = 0.0, {t: 0.0 for t in TERMS + ("ddr", "launch")}
  for li in range(nl):
    lt = tasks[12 * li:12 * li + 12]
    per = [task_cycles(c, g, len(lt), k) for g in lt]
    crit = max(per, key=lambda p: p[0])
    ddr = sum(p[2] for p in per) / k["ddr_rate"]
    cyc = max(crit[0], ddr)
    total += cyc / CLOCK
    if ddr > crit[0]: terms_s["ddr"] += ddr / CLOCK        # the launch is DDR-bound: that term carries it
    else:
      for t, v in crit[1].items(): terms_s[t] += v / CLOCK
  terms_s["launch"] = nl * k["launch"] + k["call"]
  total += terms_s["launch"]
  bound = max(terms_s, key=terms_s.get)
  return Prediction(total, bound, terms_s, nl, len(tasks), c)


def candidates(base: Config, free=("piece", "drain")):
  """The configurations `plan` may pick among, varying only the runner-level knobs in `free` (the operand layout -- ks,
  ns, lin48, b8, bscale, rows, rowmax -- is the caller's)."""
  pieces = [base.piece] if "piece" not in free else [p for p in range(2, min(base.nrb, 28) + 1, 2) if base.nrb % p == 0]
  drains = [base.drain] if ("drain" not in free or base.lin48 or base.b8 or base.rowmax) else ["dma", "tec"]
  for p in pieces:
    for d in drains:
      cc = replace(base, piece=p, drain=d)
      ok, _ = cc.feasible()
      if ok: yield cc


def plan(M: int, K: int, N: int, dtype: str = "fp16", bscale: bool = False, rows_hint: int = 0, *, ks: int | None = None,
         ns: int | None = None, lin48: bool = False, rowmax: bool = False, heads: int = 1, scales: str = "dup", free=("piece", "drain"),
         k: dict | None = None):
  """The cheapest feasible configuration for C[M, N] = A[M, K] @ B[N, K]^T, and its predicted breakdown.

  dtype: "fp16" (B fp16 panels) or "e4m3" (B codes, expanded in LSRAM); bscale: E4M3 with 128 x 128 block scales (ks = 32), their
  table in the `scales` layout ("dup" / "single").
  rows_hint: the real rows when M is padded (1..12: rows mode). ks / ns: the operand layout (None: every layout that fits,
  for a call site whose operands are packed per call). Returns (Prediction of the pick, [Prediction of every candidate])."""
  nrb = -(-M // 12); nrb += nrb % 2
  layouts = [(ks, ns)] if ks is not None else [(k_, n_) for k_ in (24, 32, 36, 48) for n_ in (2, 3, 4, 6)]
  preds = []
  for ks_, ns_ in layouts:
    if bscale and ks_ != 32: continue
    base = Config(K=K, N=N, nrb=nrb, ks=ks_, ns=ns_, b8=dtype == "e4m3", bscale=bscale, lin48=lin48, rowmax=rowmax,
                  rows=rows_hint if 0 < rows_hint <= 12 and nrb <= 2 else 0, heads=heads, piece=nrb if (rows_hint and nrb <= 2) else 0,
                  scales=scales)
    if base.K % (4 * base.kks) or base.N % (16 * base.ns): continue
    for cc in candidates(base, free if not base.rows else ()):
      preds.append(predict(cc, k))
  if not preds: raise ValueError(f"no feasible gemm_gs configuration for M={M} K={K} N={N} {dtype} bscale={bscale}")
  preds.sort(key=lambda p: p.seconds)
  return preds[0], preds


_PIECES: dict = {}
def plan_piece(K: int, N: int, nrb: int, *, ks: int, ns: int, b8: bool = False, bscale: bool = False, lin48: bool = False,
               rowmax: bool = False, rows: int = 0, heads: int = 1, scales: str = "dup") -> int:
  """The row piece `plan` picks for a call whose operand layout is fixed (the knob a call site passes as `piece=`); cached.
  Rows mode has one piece (nrb). `scales`: the B stream's block-scale layout, as the call passes `gemm_gs(scales=)`."""
  key = (K, N, nrb, ks, ns, b8, bscale, lin48, rowmax, rows, heads, scales)
  if key not in _PIECES:
    if rows: _PIECES[key] = nrb
    else:
      base = Config(K=K, N=N, nrb=nrb, ks=ks, ns=ns, b8=b8, bscale=bscale, lin48=lin48, rowmax=rowmax, heads=heads, scales=scales)
      preds = [predict(c) for c in candidates(base, ("piece",))]
      if not preds: raise ValueError(f"gemm_plan: no feasible piece for {base}")
      _PIECES[key] = min(preds, key=lambda p: p.seconds).config.nrbh
  return _PIECES[key]


# ***************** fitting (board data only) *****************
FIT = ("mma", "gs_vec", "exp_bun", "bs_bun", "strip", "rb", "slice", "group", "req", "tec_rate", "ddr_rate", "launch", "call", "ls_vst", "ddr_vst")


def from_record(r: dict) -> Config:
  return Config(K=r["K"], N=r["N"], nrb=r["nrb"], ks=r["ks"], ns=r["ns"], piece=r["piece"], b8=r["b8"], bscale=r["bscale"],
                lin48=r["lin48"], rowmax=r["rowmax"], rows=r["rows"], drain=r["drain"], timing=r["timing"], scales=r.get("scales", "dup"))


# how far each cost may move from its prior, in natural-log units (a Gaussian prior on log(cost)): the silicon-measured rates
# are held close, the structural overheads are free. Without it the counts' collinearity lets the fit trade the matrix
# unit's cost for GSRAM's and a per-group constant -- as accurate on the fitted shapes, and wrong for any other ks.
SIGMA = {"mma": 0.05, "gs_vec": 0.35, "exp_bun": 0.5, "bs_bun": 1.0, "ls_vst": 0.5, "ddr_vst": 1.0, "strip": 2.0, "rb": 2.0,
         "slice": 2.0, "group": 2.0, "req": 1.0, "tec_rate": 0.5, "ddr_rate": 0.15, "launch": 1.5, "call": 1.5}


def fit(records: list[dict], iters: int = 200, names=FIT, k0: dict | None = None, prior_weight: float = 1.0) -> dict:
  """Levenberg-Marquardt on log(predicted / measured) over the costs in `names` (log-parametrised: positive), with the
  Gaussian prior SIGMA on each log-cost around PRIOR (weight `prior_weight`; 0 = an unregularised fit)."""
  import numpy as np
  k = dict(coeffs() if k0 is None else k0); cfgs = [from_record(r) for r in records]; meas = np.array([r["median"] for r in records])
  x0 = np.log([max(PRIOR[n][0], 1e-12) for n in names]); sig = np.array([SIGMA[n] for n in names])
  x = np.log([max(k[n], 1e-12) for n in names])
  def resid(x):
    kk = dict(k); kk.update({n: float(math.exp(v)) for n, v in zip(names, x)})
    data = np.log(np.array([predict(c, kk).seconds for c in cfgs]) / meas)
    return np.concatenate([data, prior_weight * (x - x0) / sig * 0.05])      # 0.05: a prior sigma of 1 weighs as a 5 % miss
  lam, r = 1e-2, resid(x)
  for _ in range(iters):
    J = np.stack([(resid(x + d) - r) / 1e-4 for d in np.eye(len(x)) * 1e-4], 1)
    A = J.T @ J; g = J.T @ r
    step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
    xn = x + step; rn = resid(xn)
    if (rn ** 2).sum() < (r ** 2).sum(): x, r, lam = xn, rn, lam / 3
    else: lam *= 4
    if np.abs(step).max() < 1e-6 or lam > 1e8: break
  k.update({n: float(math.exp(v)) for n, v in zip(names, x)})
  return k


def errors(records: list[dict], k: dict) -> list[tuple[dict, Prediction, float]]:
  out = []
  for r in records:
    p = predict(from_record(r), k); out.append((r, p, p.seconds / r["median"] - 1.0))
  return out
