"""Timing instrumentation for the custom ops' jobs (`ops.py`'s runners).

Host-time categories: `T_submit` (the SCHEDULE_JOB ioctls), `T_AIFF_exec` (the wait for the jobs -- the name predates the
TEC runners and is kept: callers read it), `wall`; and device cycles from the per-task stamps.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from tinygrad.runtime.support.zhouyi.launch import CYCLE_STAMP_POISON

SUBMIT = "T_submit"
AIFF_EXEC = "T_AIFF_exec"
WALL = "wall"
CATEGORIES = (SUBMIT, AIFF_EXEC, WALL)

# Not a category: the categories are host seconds, this is TEC cycles (ctrl0[0xd1]) from the per-task stamps
# `launch.read_cycles` returns. Reported separately from the categories.
DEVICE_CYCLES = "device_cycles"


@dataclass
class _Cycles:
  """Per-task device cycles over all recorded launches. `max` is the slowest task (a launch finishes with
  it); `total` sums all tasks. `unstamped` counts tasks whose stamps still hold the poison (built with
  without the stamps); they are excluded from the sums."""
  launches: int = 0
  tasks: int = 0
  total: int = 0
  max: int = 0
  unstamped: int = 0

  def add(self, stamps: Iterable[tuple[int, int]]) -> None:
    self.launches += 1
    for start, end in stamps:
      self.tasks += 1
      if start == CYCLE_STAMP_POISON or end == CYCLE_STAMP_POISON:
        self.unstamped += 1
        continue
      d = (end - start) & 0xFFFFFFFF   # free-running 32-bit counter: never assume end >= start
      self.total += d
      if d > self.max: self.max = d


@dataclass
class _Totals:
  """Running count and sum (no per-sample list, to keep overhead constant)."""
  count: int = 0
  seconds: float = 0.0

  def add(self, seconds: float) -> None:
    self.count += 1
    self.seconds += seconds

  @property
  def mean(self) -> float:
    return self.seconds / self.count if self.count else 0.0


class _Timer:
  """`with recorder.time(SUBMIT): ...`. Records on exit even if the block raises (e.g. a timeout)."""
  __slots__ = ("_r", "_c", "_t0")

  def __init__(self, r: "Recorder", category: str) -> None:
    self._r, self._c = r, category

  def __enter__(self) -> "_Timer":
    self._t0 = time.perf_counter()
    return self

  def __exit__(self, *exc) -> bool:
    self._r.add(self._c, time.perf_counter() - self._t0)
    return False


class Recorder:
  """Per-category running totals and device cycles. One process-wide instance: `DEFAULT`."""

  def __init__(self) -> None:
    self._totals: dict[str, _Totals] = {c: _Totals() for c in CATEGORIES}
    self._cycles = _Cycles()

  def add(self, category: str, seconds: float) -> None:
    self._totals[category].add(seconds)

  def add_cycles(self, stamps: Iterable[tuple[int, int]]) -> None:
    """One launch's per-task `(start, end)` stamps, as returned by `Launch.read_cycles()`."""
    self._cycles.add(stamps)

  @property
  def device_cycles(self) -> _Cycles:
    return self._cycles

  def time(self, category: str) -> _Timer:
    return _Timer(self, category)

  def report(self) -> dict:
    return {
      "categories": {c: {"count": t.count, "seconds": t.seconds, "mean_seconds": t.mean}
                     for c, t in self._totals.items()},
      DEVICE_CYCLES: {"launches": self._cycles.launches, "tasks": self._cycles.tasks,
                      "total": self._cycles.total, "max": self._cycles.max,
                      "unstamped": self._cycles.unstamped},
    }

  def reset(self) -> None:
    self.__init__()


DEFAULT = Recorder()


def submit_and_wait(raw, submissions: Iterable[tuple[int, int, int]],
                    recorder: Recorder = DEFAULT) -> float:
  """Submit every `(head_tcb_pa, first_task_tcb_pa, last_task_tcb_pa)` in `submissions`, then wait for all.

  Records `T_submit` (the SCHEDULE_JOB ioctls), `T_AIFF_exec` (the wait) and `wall`; returns the wall time."""
  t0 = time.perf_counter()
  job_ids = [raw.submit(*args) for args in submissions]
  t1 = time.perf_counter()
  raw.wait_jobs(job_ids)
  t2 = time.perf_counter()
  recorder.add(SUBMIT, t1 - t0)
  recorder.add(AIFF_EXEC, t2 - t1)
  recorder.add(WALL, t2 - t0)
  return t2 - t0
