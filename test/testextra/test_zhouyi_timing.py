"""extra/zhouyi/timing: the recorder, the timer and `submit_and_wait` (the runners' submit path).

Host-only throughout: no `/dev/aipu`, no board. `Recorder`/`_Timer`/`submit_and_wait` are pure Python + `time.perf_counter`,
checked against a `FakeRaw` double standing in for `RawDevice`.
"""
import time
import unittest

from extra.zhouyi import timing as TM


class TestRecorder(unittest.TestCase):
  def test_add_accumulates_count_and_seconds(self):
    r = TM.Recorder()
    r.add(TM.SUBMIT, 0.002)
    r.add(TM.SUBMIT, 0.003)
    cats = r.report()["categories"][TM.SUBMIT]
    self.assertEqual(cats["count"], 2)
    self.assertAlmostEqual(cats["seconds"], 0.005)
    self.assertAlmostEqual(cats["mean_seconds"], 0.0025)

  def test_categories_are_independent(self):
    r = TM.Recorder()
    r.add(TM.SUBMIT, 0.001)
    report = r.report()["categories"]
    for c in TM.CATEGORIES:
      if c != TM.SUBMIT:
        self.assertEqual(report[c]["count"], 0)
        self.assertEqual(report[c]["seconds"], 0.0)

  def test_mean_of_an_empty_category_is_zero_not_a_zerodiv(self):
    r = TM.Recorder()
    self.assertEqual(r.report()["categories"][TM.WALL]["mean_seconds"], 0.0)

  def test_reset_clears_totals_and_cycles(self):
    r = TM.Recorder()
    r.add(TM.SUBMIT, 0.5)
    r.add_cycles([(100, 350), (100, 600)])
    r.reset()
    report = r.report()
    self.assertEqual(report["categories"][TM.SUBMIT]["count"], 0)
    self.assertEqual(report[TM.DEVICE_CYCLES], {"launches": 0, "tasks": 0, "total": 0, "max": 0, "unstamped": 0})

  def test_cycles_wrap_and_skip_unstamped_tasks(self):
    from tinygrad.runtime.support.zhouyi.launch import CYCLE_STAMP_POISON
    r = TM.Recorder()
    r.add_cycles([(100, 350), (0xFFFFFF00, 0x100), (CYCLE_STAMP_POISON, 5)])
    c = r.report()[TM.DEVICE_CYCLES]
    self.assertEqual((c["launches"], c["tasks"], c["unstamped"]), (1, 3, 1))
    self.assertEqual((c["total"], c["max"]), (250 + 0x200, 0x200))

class TestTimerContextManager(unittest.TestCase):
  def test_records_one_sample_of_roughly_the_right_magnitude(self):
    r = TM.Recorder()
    with r.time(TM.WALL):
      time.sleep(0.01)
    cats = r.report()["categories"][TM.WALL]
    self.assertEqual(cats["count"], 1)
    self.assertGreaterEqual(cats["seconds"], 0.01)

  def test_records_even_when_the_block_raises(self):
    """A job that times out still spent real time before it failed -- the sample belongs in the report, not silently dropped."""
    r = TM.Recorder()
    with self.assertRaises(ValueError):
      with r.time(TM.SUBMIT):
        raise ValueError("boom")
    self.assertEqual(r.report()["categories"][TM.SUBMIT]["count"], 1)


class TestOverheadIsNegligible(unittest.TestCase):
  """Instrumentation must be free or its cost stated. A job submit costs milliseconds; this asserts the instrumentation's
  own overhead sits at least three orders of magnitude below that, so the measurement never becomes a meaningful fraction
  of what it measures."""

  def test_timer_overhead_is_microseconds_not_milliseconds(self):
    r = TM.Recorder()
    n = 20_000
    t0 = time.perf_counter()
    for _ in range(n):
      with r.time(TM.WALL):
        pass
    total = time.perf_counter() - t0
    per_call_overhead = total / n
    self.assertLess(per_call_overhead, 20e-6,
                    f"per-call overhead {per_call_overhead*1e6:.2f} us is not negligible "
                    "against a ~2.8-3.5 ms job")


class FakeRaw:
  """Structural stand-in for `RawDevice` -- only `submit`/`wait_jobs` matter here. `submit_delay`/`wait_delay` make the two
  spans distinguishable without sleeping for anything close to a real job's ms-scale cost."""
  def __init__(self, submit_delay=0.001, wait_delay=0.003):
    self.submit_delay, self.wait_delay = submit_delay, wait_delay
    self.submitted, self.waited = [], None

  def submit(self, *args) -> int:
    time.sleep(self.submit_delay)
    self.submitted.append(args)
    return len(self.submitted)

  def wait_jobs(self, job_ids):
    time.sleep(self.wait_delay)
    self.waited = list(job_ids)
    return {j: None for j in job_ids}


class TestSubmitAndWait(unittest.TestCase):
  def test_splits_submit_from_exec_and_sums_to_the_return_value(self):
    raw = FakeRaw(submit_delay=0.005, wait_delay=0.02)
    r = TM.Recorder()
    chains = [(1, 2, 3), (4, 5, 6)]
    wall = TM.submit_and_wait(raw, chains, recorder=r)
    cats = r.report()["categories"]
    self.assertEqual(cats[TM.SUBMIT]["count"], 1)
    self.assertEqual(cats[TM.AIFF_EXEC]["count"], 1)
    self.assertEqual(cats[TM.WALL]["count"], 1)
    # two submits at 5ms each dominate T_submit; one wait at 20ms is T_AIFF_exec (the wait category)
    self.assertGreaterEqual(cats[TM.SUBMIT]["seconds"], 0.010)
    self.assertGreaterEqual(cats[TM.AIFF_EXEC]["seconds"], 0.02)
    self.assertAlmostEqual(cats[TM.SUBMIT]["seconds"] + cats[TM.AIFF_EXEC]["seconds"], wall, places=2)

  def test_submits_every_chain_then_waits_on_every_job_id(self):
    raw = FakeRaw(submit_delay=0, wait_delay=0)
    chains = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    TM.submit_and_wait(raw, chains, recorder=TM.Recorder())
    self.assertEqual(raw.submitted, chains)
    self.assertEqual(raw.waited, [1, 2, 3])  # job ids `FakeRaw.submit` handed back, in order

  def test_default_recorder_is_used_when_none_is_passed(self):
    before = TM.DEFAULT.report()["categories"][TM.SUBMIT]["count"]
    TM.submit_and_wait(FakeRaw(submit_delay=0, wait_delay=0), [(1, 2, 3)])
    after = TM.DEFAULT.report()["categories"][TM.SUBMIT]["count"]
    self.assertEqual(after, before + 1)


if __name__ == "__main__":
  unittest.main()
