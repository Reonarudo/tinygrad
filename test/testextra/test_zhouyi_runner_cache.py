"""extra/zhouyi/ops._cached_runner: a runner built under one value of an environment variable it reads at construction
(`ENV_KEYS`) is not reused under another (host only)."""
import os, unittest
from tinygrad.uop.ops import UOp, Ops
from tinygrad import dtypes
from extra.zhouyi import ops as OA

class _Fake:
  ENV_KEYS = ("ZHOUYI_TEST_KNOB",)
  def __init__(self, cf, device): self.knob = os.environ.get("ZHOUYI_TEST_KNOB")

class TestRunnerCache(unittest.TestCase):
  def test_env_is_part_of_the_key(self):
    cf = UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(UOp.const(dtypes.int, 7),), arg="test")
    try:
      os.environ.pop("ZHOUYI_TEST_KNOB", None); a = OA._cached_runner(_Fake, cf, "ZHOUYI")
      os.environ["ZHOUYI_TEST_KNOB"] = "tec"; b = OA._cached_runner(_Fake, cf, "ZHOUYI")
      os.environ.pop("ZHOUYI_TEST_KNOB"); c = OA._cached_runner(_Fake, cf, "ZHOUYI")
    finally: os.environ.pop("ZHOUYI_TEST_KNOB", None)
    self.assertIsNot(a, b); self.assertEqual((a.knob, b.knob), (None, "tec")); self.assertIs(a, c)
  def test_declared_keys(self):
    self.assertEqual(set(OA.GemmGSRunner.ENV_KEYS), {"ZHOUYI_GEMM_DRAIN", "ZHOUYI_GEMM_TIMING"})
    self.assertEqual(tuple(getattr(OA.GemmGMRunner, "ENV_KEYS", ())), ())

if __name__ == "__main__": unittest.main()
