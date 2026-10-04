"""ZHOUYI graph runner: fuses consecutive kernel launches into one job to cut per-launch host overhead.

Each run of chainable kernels becomes one TCB chain (`dev.build_group_tcbs`), one group per kernel, entered under `DEP_PRE_ALL`
(the only dependency mode that both serialises and is a store barrier), so a run costs one submit and one wait. A launch joins a
chain if it is a one-core Program (<= 4 tasks), a 12-task Program (one group per core, behind a barrier group) or a custom-op
runner registered in `CHAIN_MEMBER_ARGS`, with concrete launch dims, and its Program is not already in the chain (its param
buffers hold one argument state). Others run solo.

After the first call, each position's argument arrays are frozen: later calls only poke the words that change (JIT variables,
replaced input addresses), rewrite param buffers another position overwrote, and submit.
"""
from __future__ import annotations
import ctypes, struct, time
from typing import Any, cast
from tinygrad.device import Buffer
from tinygrad.engine.jit import GraphRunner
from tinygrad.engine.realize import CompiledRunner, ExecItem
from tinygrad.helpers import all_int, getenv
from tinygrad.runtime.ops_zhouyi import ZhouyiProgram, ZHOUYI_TECS, ZHOUYI_CORES, submit_wait
from tinygrad.runtime.support.zhouyi import ZhouyiError, dev as _dev, hangid as _hid
from tinygrad.runtime.support.zhouyi.dev import MM_TCB, TCB_LEN


CHAIN_MAX = getenv("ZHOUYI_CHAIN_MAX", 8)   # launches per fused job (1: every launch its own job, e.g. to pin a hang to one kernel)


class _Frozen:
  """One launch position's arg blobs, packed once for the frozen replay.

  Buffers are pinned by the jit_cache and launch dims are concrete, so only JIT vars and
  `input_replace` addresses change per call (`poke`).
  `gens` is the Program's `(_chain_gen, _param_gen)` after this position last wrote its params; they are
  trusted only while the Program's counters still match. Any other write (another position sharing the
  Program, a foreign `stage`, a `_chain` rebuild) bumps a counter; a poke sets `gens = None`."""
  __slots__ = ("blobs", "cblobs", "nb", "ntasks", "nglobals", "gens", "bufrefs")

  def __init__(self, blobs: list[bytearray], nb: int, ntasks: int, nglobals: int, bufrefs: list):
    self.blobs, self.nb, self.ntasks, self.nglobals, self.bufrefs = blobs, nb, ntasks, nglobals, bufrefs
    # ctypes views sharing the bytearrays' memory (pokes land in both), for a direct memmove.
    self.cblobs = [(ctypes.c_char*nb).from_buffer(b) for b in blobs]
    self.gens: tuple[int, int]|None = None

  def poke(self, offs: list[int], v: int) -> None:
    w = struct.pack("<I", v & 0xFFFFFFFF)
    for o in offs:
      for b in self.blobs: b[o:o+4] = w
    self.gens = None

  def ensure_written(self, prg: ZhouyiProgram) -> None:
    """`stage` minus the arg construction: rewrite params (and re-`_chain` to this width) if stale.

    A shared Program may have been staged at another width meanwhile, so `_chain(self.ntasks)` is needed,
    not just the param bytes; it is a no-op at equal width."""
    if self.gens != (prg._chain_gen, prg._param_gen):
      prg._chain(self.ntasks)
      prg.write_params(self.cblobs, self.nb)
      self.gens = (prg._chain_gen, prg._param_gen)


def plan_runs(keys: list, cap: int) -> list[list[int]]:
  """Partition launch positions into maximal chainable runs (pure function, testable without hardware).

  `keys[i]`: hashable identity of a chainable launch (`id()` of its Program), or None to run solo. A run
  breaks at an unchainable launch, at a Program already in the run, or at `cap` launches. Returns index
  lists in launch order; singletons are normally launched solo."""
  runs: list[list[int]] = []
  seg: list[int] = []
  seen: set = set()
  for i, k in enumerate(keys):
    if (k is None or k in seen or len(seg) >= cap) and seg:
      runs.append(seg)
      seg, seen = [], set()
    if k is None: runs.append([i])
    else:
      seg.append(i)
      seen.add(k)
  if seg: runs.append(seg)
  return runs


def _is_member(prg) -> bool:
  """True for a chain member that is not a Program: a runner exposing `group_slots()`/`chain_deps()`/
  `stage_chain()` (e.g. `extra/zhouyi/ops.py`'s GEMM runners). Its groups join the chain with their own dependency modes."""
  return getattr(prg, "chain_member", False)


from tinygrad.runtime.ops_zhouyi import barrier_group as _barrier_group


class _Chain:
  """One fused job: a run of one-core Programs and chain members, each a group (or a few) of a
  single TCB chain; a 12-task Program is three groups (one per core)."""

  def __init__(self, items: list[tuple[ExecItem, Any, int]]):
    self.items = items                       # (ExecItem, ZhouyiProgram or chain member, ntasks)
    self.tcbs: _dev.Buffer|None = None
    self.gens: list[int] = []
    self.build()

  def build(self) -> None:
    p0 = self.items[0][1]
    raw = p0.raw if _is_member(p0) else p0.dev.raw          # a member is a Runner: its `dev` is the device NAME
    groups, deps = [], []
    self.layout: list[int] = []                             # ZHOUYI_HANG_ID: TCB groups per item (the job report's g numbers)
    for ji, prg, n in self.items:
      g0 = len(groups)
      if _is_member(prg):
        if getattr(prg, "_njobs", 1) == 0: prg.stage_chain(ji.bufs)      # an unlaunched clone has no chain until staged
        groups += prg.group_slots()
        deps += prg.chain_deps()
        if _hid.LEVEL:
          for g in groups[g0:]: _hid.register_text(g.spc, _hid.runner_label(ji))
      elif n == ZHOUYI_CORES * ZHOUYI_TECS:
        # a 12-task Program: one group per core.
        # a do-nothing barrier group gated on everything before, then the core groups free behind it. Note:
        # [PRE_ALL g1][0 g2][0 g3] is not safe: a free group can start before g1's gate opens.
        prg._chain(n)
        g = prg.group_slots_per_core()
        groups += [_barrier_group(raw)] + g
        deps += [_dev.DEP_PRE_ALL] + [0] * len(g)
      else:
        prg._chain(n)
        groups.append(prg.group_slot())
        deps.append(_dev.DEP_PRE_ALL)
      self.layout.append(len(groups) - g0)
    if deps[-1] != _dev.DEP_PRE_ALL:
      # the chain's last TCB retires the job, so it must wait for every core: append a do-nothing barrier group
      groups.append(_barrier_group(raw))
      deps.append(_dev.DEP_PRE_ALL)
    self.closing_barrier = len(groups) > sum(self.layout)
    total = sum(len(g.tasks) for g in groups)
    if self.tcbs is None: self.tcbs = raw.req_buf(_dev.chain_tcbs(total)*TCB_LEN, MM_TCB)
    assert self.tcbs is not None
    # a chain containing a GM member runs with GM remap on in the grid TCB (all DMA into the window goes to GM)
    gm = next((p.gm_pa for _, p, _ in self.items if getattr(p, "uses_gm", False)), None)
    _dev.build_group_tcbs(raw, self.tcbs, groups, dep=deps, gm_pa=gm)   # DEP_PRE_ALL between kernels; a member's own modes inside it
    self.head_pa = self.tcbs.pa
    self.first_pa = self.tcbs.pa + _dev.TASK_INDEX*TCB_LEN
    self.last_pa = self.tcbs.pa + (_dev.TASK_INDEX + total - 1)*TCB_LEN
    self.gens = [prg._chain_gen for _, prg, _ in self.items]

  def ensure(self) -> None:
    """Rebuild if any member's per-task buffers changed since the build (`_chain_gen`); a stale chain
    would run on freed addresses and silently produce wrong results."""
    if [prg._chain_gen for _, prg, _ in self.items] != self.gens: self.build()


class ZhouyiGraph(GraphRunner):
  @staticmethod
  def supports_exec_item(batch_devs, new_call) -> bool:
    """Compiled kernels, plus the Zhouyi custom ops whose runners can join chains; otherwise the JIT would
    split the batch at every such op."""
    from tinygrad.uop.ops import Ops
    from tinygrad.runtime.ops_zhouyi import CHAIN_MEMBER_ARGS
    if new_call.src[0].op is Ops.CUSTOM_FUNCTION and new_call.src[0].arg in CHAIN_MEMBER_ARGS:
      return len(GraphRunner._all_devs(batch_devs, new_call)) == 1
    return GraphRunner.supports_exec_item(batch_devs, new_call)

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # A chain member or Program holds one launch state, so each repeat of the same runner in this graph
    # gets its own clone (members: `clone_member`; Programs: `CompiledRunner(p)`, same lib, new Launch).
    # Otherwise `plan_runs` would break the run at every repeat.
    from dataclasses import replace as _replace
    from tinygrad.runtime.ops_zhouyi import clone_member
    seen: set = set()
    for i, ji in enumerate(self.jit_cache):
      if _is_member(ji.prg) and hasattr(ji.prg, "_clone_args"):
        if id(ji.prg) in seen: self.jit_cache[i] = ji = _replace(ji, prg=clone_member(ji.prg))
        seen.add(id(ji.prg))
      elif isinstance(ji.prg, CompiledRunner) and self._chainable(ji) is not None:
        if id(ji.prg) in seen: self.jit_cache[i] = ji = _replace(ji, prg=CompiledRunner(ji.prg.p))
        seen.add(id(ji.prg))
    prgs = [self._chainable(ji) for ji in self.jit_cache]
    # the plan: ("chain", _Chain) | ("solo", ExecItem), in launch order
    self.plan: list[tuple[str, Any]] = []
    for run in plan_runs([id(p) if p is not None else None for p in prgs], CHAIN_MAX):
      # A lone 12-task Program is chained too. Note: as a solo launch it would be three concurrent jobs, which
      # occasionally drops one core's tasks; one job of per-core groups does not.
      lone_wide = len(run) == 1 and prgs[run[0]] is not None and not _is_member(prgs[run[0]]) and \
        cast(CompiledRunner, self.jit_cache[run[0]].prg).p.global_size[0] == ZHOUYI_CORES * ZHOUYI_TECS
      if (len(run) >= 2 or lone_wide) and prgs[run[0]] is not None:
        self.plan.append(("chain", _Chain([(ji := self.jit_cache[j], prgs[j], 13 if _is_member(prgs[j]) else
                                             cast(CompiledRunner, ji.prg).p.global_size[0]) for j in run])))
      else: self.plan.extend(("solo", self.jit_cache[j]) for j in run)

    self.n_chained = sum(len(c.items) for kind, c in self.plan if kind == "chain")

    # Frozen replay state, built after the first staged call (which provides a var binding).
    self._j_of = {id(ji): j for j, ji in enumerate(self.jit_cache)}
    self._steps: list[tuple]|None = None
    self._raw: Any = None
    self._var_pokes: dict[tuple[int, int], tuple[_Frozen, list[int]]] = {}
    self._in_pokes: list[tuple[_Frozen, list[int], int]] = []

  def _freezable(self, ji: ExecItem) -> ZhouyiProgram|None:
    """The Program if the frozen replay may pack this position: a non-fallback device Program with concrete,
    non-replaced launch dims (any width, unlike `_chainable`); else None."""
    if not isinstance(ji.prg, CompiledRunner) or not isinstance(prg := ji.prg._prg, ZhouyiProgram): return None
    if self._j_of[id(ji)] in self.launch_dims_replace: return None
    p = ji.prg.p
    if p.global_size is None or not all_int(tuple(p.global_size)): return None
    return prg

  def _freeze_one(self, ji: ExecItem, prg: ZhouyiProgram, var_vals: dict[str, int]) -> _Frozen:
    """Pack one position using the same `_launch_args`/`_pack_blobs` as `stage`. `bufrefs` keeps the arg
    Buffers alive while the frozen blobs hold their addresses."""
    cr = cast(CompiledRunner, ji.prg)
    vv = var_vals | ji.fixedvars
    gs, _ = cr.p.launch_dims(vv)
    bufrefs = [cast(Buffer, ji.bufs[i]) for i in cr.p.globals]
    vals = tuple(vv[k.expr] if k.expr not in cr.p.runtimevars else None for k in cr.p.vars)
    args, cid = prg._launch_args([b._buf for b in bufrefs], vals, gs[0])
    prg._chain(gs[0])   # usually a no-op; re-chains a Program last staged at another width
    blobs = [bytearray(b) for b in prg._pack_blobs(args, cid)]
    return _Frozen(blobs, len(args)*4, gs[0], len(cr.p.globals), bufrefs)

  def _freeze(self, var_vals: dict[str, int]) -> list[tuple]:
    """Build the frozen steps (one per plan entry) and poke tables. Runs after a staged call, when all
    chains and task buffers exist."""
    steps: list[tuple] = []
    frozen_by_j: dict[int, _Frozen] = {}
    for kind, entry in self.plan:
      if kind == "solo":
        ji: ExecItem = entry
        if (prg := self._freezable(ji)) is None:
          steps.append(("slow", ji))
          continue
        frozen_by_j[j := self._j_of[id(ji)]] = ps = self._freeze_one(ji, prg, var_vals)
        steps.append(("psolo", prg, ps))
      else:
        chain: _Chain = entry
        # `_chainable` implies `_freezable` today; if not, fall back to the staged path for this chain.
        if any(self._freezable(ji) is None for ji, _, _ in chain.items):
          steps.append(("chain_slow", chain))
          continue
        members = []
        for ji, prg, _ in chain.items:
          frozen_by_j[j := self._j_of[id(ji)]] = ps = self._freeze_one(ji, prg, var_vals)
          members.append((prg, ps))
        steps.append(("chain", chain, members))
    # Poke tables: the only words that change per call. The arg array is dev_addrs then vals (in `p.vars`
    # order), so var i sits at 4*(nglobals + i); an input buffer sits at every slot `p.globals` maps to it.
    self._var_pokes = {(j, i): (frozen_by_j[j], [4*(frozen_by_j[j].nglobals + i)])
                       for j, rep in self.var_vals_replace.items() if j in frozen_by_j for i, _ in rep}
    self._in_pokes = []
    for (j, i), input_idx in self.input_replace.items():
      if (fz := frozen_by_j.get(j)) is not None:
        if offs := [4*k for k, g in enumerate(cast(CompiledRunner, self.jit_cache[j].prg).p.globals) if g == i]:
          self._in_pokes.append((fz, offs, input_idx))
    return steps

  @staticmethod
  def _chainable(ji: ExecItem):
    """The chain-eligibility rules from the module docstring. Returns the Program (or chain member), or
    None to run `ji` solo."""
    if _is_member(ji.prg): return ji.prg
    if not isinstance(ji.prg, CompiledRunner) or not isinstance(prg := ji.prg._prg, ZhouyiProgram): return None
    p = ji.prg.p
    if p.global_size is None or not all_int(tuple(p.global_size)): return None
    if tuple(p.global_size)[1:] != (1, 1) or not (1 <= p.global_size[0] <= ZHOUYI_TECS or p.global_size[0] == ZHOUYI_CORES * ZHOUYI_TECS): return None
    if p.local_size is not None and tuple(p.local_size) != (1, 1, 1): return None
    return prg

  def _stage_members(self, chain: _Chain, var_vals: dict[str, int]) -> None:
    for ji, prg, _ in chain.items:
      if _is_member(prg):
        prg.stage_chain(ji.bufs)
        continue
      cr = cast(CompiledRunner, ji.prg)
      vv = var_vals | ji.fixedvars
      gs, ls = cr.p.launch_dims(vv)
      prg.stage([cast(Buffer, ji.bufs[i])._buf for i in cr.p.globals], tuple(gs), tuple(ls) if ls else (1, 1, 1),
                tuple(vv[k.expr] if k.expr not in cr.p.runtimevars else None for k in cr.p.vars))

  def _run_staged(self, var_vals: dict[str, int]) -> None:
    """Staged replay: stage every member; the first call, before freezing."""
    k = 0
    try:
      for k, (kind, entry) in enumerate(self.plan):
        if kind == "solo":
          entry.run(var_vals, wait=False, jit=True, do_update_stats=False)
          continue
        chain: _Chain = entry
        self._stage_members(chain, var_vals)
        chain.ensure()
        p0 = chain.items[0][1]
        raw = p0.raw if _is_member(p0) else p0.dev.raw
        submit_wait(raw, [(chain.head_pa, chain.first_pa, chain.last_pa)])
    except ZhouyiError as e:
      if not _hid.LEVEL: raise
      raise ZhouyiError(f"{e}{self._hang_report(k, 'staged')}") from None

  def _hang_report(self, k: int, mode: str) -> str:
    """ZHOUYI_HANG_ID: where plan step `k` (the one that raised) sits in this graph, its kernels and their buffers."""
    try:
      def jis_of(kind, entry): return [entry] if kind == "solo" else [ji for ji, _, _ in entry.items]
      def raw_of(ji):
        p = ji.prg
        return p.raw if _is_member(p) else p._prg.dev.raw if isinstance(p, CompiledRunner) else None
      kind, entry = self.plan[k]
      jis = jis_of(kind, entry)
      raw = self._raw or next((r for r in map(raw_of, jis) if r is not None), None)
      w = _hid._Where(raw) if raw is not None else None
      js = [self._j_of[id(ji)] for ji in jis]
      lines = [f"[ZHOUYI_HANG_ID] graph {id(self):#x}: {len(self.jit_cache)} launches in {len(self.plan)} steps ({self.n_chained} chained), "
               f"{mode} replay: step {k} [{kind}] raised; steps 0..{k - 1} completed in this call",
               f"  context: {_hid.CONTEXT}" if _hid.CONTEXT else "  context: (none set)"]
      layout = getattr(entry, "layout", None) if kind != "solo" else None
      g = 0
      for i, (ji, j) in enumerate(zip(jis, js)):
        grp = ""
        if layout is not None:
          wide = not _is_member(ji.prg) and entry.items[i][2] == ZHOUYI_CORES * ZHOUYI_TECS
          grp = (f" (job groups g{g}-g{g + layout[i] - 1}: a barrier, then one per core)" if wide else
                 f" (job group{'s' if layout[i] > 1 else ''} g{g}{f'-g{g + layout[i] - 1}' if layout[i] > 1 else ''})")
          g += layout[i]
        lines.append(f"  jit[{j}] {_hid.runner_label(ji)}{grp}")
        if raw is not None: lines += _hid.describe_bufs(raw, ji.bufs, w, "      ")
      if layout is not None and getattr(entry, "closing_barrier", False): lines.append(f"  job group g{g}: the closing barrier")
      lab = lambda j: _hid.runner_label(self.jit_cache[j])
      lines.append(f"  previous launch: jit[{min(js) - 1}] {lab(min(js) - 1)}" if min(js) > 0 else "  previous launch: (none: the graph's first)")
      lines.append(f"  next launch:     jit[{max(js) + 1}] {lab(max(js) + 1)}" if max(js) + 1 < len(self.jit_cache) else "  next launch: (none: the graph's last)")
      lines.append("  steps around it:")
      for kk in range(max(0, k - 3), min(len(self.plan), k + 4)):
        kd, en = self.plan[kk]
        lines.append(f"   {'>>' if kk == k else '  '} step {kk} [{kd}]{' done' if kk < k else ''}: " +
                     "; ".join(f"jit[{self._j_of[id(ji)]}] {_hid.runner_label(ji)}" for ji in jis_of(kd, en)))
      text = "\n".join(lines)
    except Exception as ex: text = f"[ZHOUYI_HANG_ID] graph report failed: {ex!r}"
    _hid.log(text)
    return "\n" + text

  def _run_frozen(self, input_buffers: list[Buffer], var_vals: dict[str, int]) -> None:
    """Frozen replay: poke changed words, rewrite only stale params, then submit."""
    for ps, offs, input_idx in self._in_pokes:
      ps.poke(offs, self._raw.dev_addr(input_buffers[input_idx]._buf))
    for j, i, v in self.updated_vars(var_vals):
      if (t := self._var_pokes.get((j, i))) is not None: t[0].poke(t[1], v)
    raw = self._raw
    steps = cast(list[tuple], self._steps)
    try:
      for step in steps:
        kind = step[0]
        if kind == "chain":
          _, chain, members = step
          for prg, ps in members: ps.ensure_written(prg)
          chain.ensure()
          submit_wait(raw, [(chain.head_pa, chain.first_pa, chain.last_pa)])
        elif kind == "psolo":
          _, prg, ps = step
          ps.ensure_written(prg)
          submit_wait(raw, [prg._one_job()] if prg.ntasks > ZHOUYI_TECS else prg.chains)
        elif kind == "chain_slow":
          chain = step[1]
          self._stage_members(chain, var_vals)
          chain.ensure()
          submit_wait(raw, [(chain.head_pa, chain.first_pa, chain.last_pa)])
        else: step[1].run(var_vals, wait=False, jit=True, do_update_stats=False)
    except ZhouyiError as e:
      if not _hid.LEVEL: raise
      raise ZhouyiError(f"{e}{self._hang_report(next(k for k, s_ in enumerate(steps) if s_ is step), 'frozen')}") from None

  def __call__(self, input_buffers: list[Buffer], var_vals: dict[str, int], wait=False) -> float|None:
    st = time.perf_counter()
    for (j, i), input_idx in self.input_replace.items(): self.jit_cache[j].bufs[i] = input_buffers[input_idx]
    if self._steps is not None:
      self._run_frozen(input_buffers, var_vals)
    else:
      self._run_staged(var_vals)
      if self._steps is None:
        self._steps = self._freeze(var_vals)
        self._raw = next((prg.dev.raw for s in self._steps if s[0] in ("psolo", "chain")
                          for prg in ([s[1]] if s[0] == "psolo" else [p for p, _ in s[2]])), None)
        if self._raw is None: self._steps = None   # nothing frozen: stay on the staged loop
    return time.perf_counter() - st
