"""extra/zhouyi/gemm_plan: the model's structure and plan() (host only; the costs are the checked-in board fit)."""
import unittest
from extra.zhouyi import gemm_plan as P

class TestGemmPlan(unittest.TestCase):
  def test_coeffs_are_the_fit(self):
    k = P.coeffs()
    self.assertAlmostEqual(k["mma"], 1.045, places=2)                  # 12.55 cycles a 3x4 k-step
    self.assertTrue(18 < k["ddr_rate"] < 22)                            # ~24 GB/s at 1.2034 GHz
  def test_every_prediction_names_a_bound(self):
    p = P.predict(P.Config(K=5120, N=34848, nrb=2, ks=32, ns=3, piece=2, b8=True, bscale=True, rows=1))
    self.assertEqual(p.bound, "ddr")                                    # decode: the weights' bytes
    p = P.predict(P.Config(K=4608, N=13824, nrb=392, lin48=True, b8=True))
    self.assertIn(p.bound, P.TERMS + ("ddr", "launch"))
    self.assertAlmostEqual(sum(p.terms_s.values()) >= p.seconds * 0.99, True)
  def test_feasibility_uses_the_allocator(self):
    ok, why = P.Config(K=4608, N=10272, nrb=2, ks=24, ns=6, piece=2, b8=True, bscale=False).feasible()
    self.assertTrue(ok, why)
    ok, why = P.Config(K=5120, N=10272, nrb=2, ks=32, ns=6, piece=2, b8=True, bscale=True).feasible()
    self.assertFalse(ok); self.assertIn("LSRAM", why)                  # 32 x 6 block-scaled does not fit
    self.assertFalse(P.Config(K=5120, N=10272, nrb=6, ks=32, ns=3, piece=3, b8=True, bscale=True).feasible()[0])   # odd piece
  def test_plan_piece(self):
    # as measured on the device: one piece of all the row blocks is fastest for block-scaled E4M3 linears at K 5120 (up to
    # 28 row blocks), piece 12 for an fp16 576 -> 1728 projection over 1368 row blocks
    self.assertEqual([P.plan_piece(5120, 10272, n, ks=32, ns=3, b8=True, bscale=True) for n in (2, 4, 6, 8, 24)], [2, 4, 6, 8, 24])
    self.assertEqual(P.plan_piece(576, 1728, 1368, ks=24, ns=6), 12)
    self.assertEqual(P.plan_piece(5120, 10272, 2, ks=32, ns=3, b8=True, bscale=True, rows=1), 2)   # rows mode: one piece
  def test_scale_table_bytes_follow_the_layout(self):
    # the B stream's scale table: 128 B a strip and slice "dup", 64 B "single" (gemm_fp16.gs_scale_bytes, the runner's stride)
    from extra.zhouyi import gemm_fp16 as G
    for sc in ("dup", "single"):
      c = P.Config(K=5120, N=10272, nrb=2, ks=32, ns=3, piece=2, b8=True, bscale=True, rows=1, scales=sc)
      self.assertEqual(c.sclb, G.gs_scale_bytes(3, 32, 0, sc)); self.assertTrue(c.feasible()[0], c.feasible()[1])
    self.assertFalse(P.Config(K=4608, N=10272, nrb=2, ks=24, ns=6, piece=2, b8=True, scales="single").feasible()[0])   # single: bscale only
    self.assertEqual([P.plan_piece(5120, 10272, n, ks=32, ns=3, b8=True, bscale=True, scales="single") for n in (2, 8, 24)], [2, 8, 24])
  def test_plan_picks_the_cheapest(self):
    best, allp = P.plan(96, 5120, 10272, "e4m3", True, 0, ks=32, ns=3)
    self.assertEqual(best.config.nrbh, 8); self.assertEqual(best.seconds, min(p.seconds for p in allp))

if __name__ == "__main__": unittest.main()
