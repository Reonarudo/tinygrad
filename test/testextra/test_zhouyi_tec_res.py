"""extra/zhouyi/tec_res: the sync-flag helpers, the image lint and the layout arenas (host only, no device).

The hazards are silicon's: four flags, selectors 4-30 alias to flag 0, a mode-1
wait on an unraised flag returns at once, wfe modes 3-31 raise exception vector 2 -- and the simulator models none of it."""
import unittest
from tinygrad.runtime.support.zhouyi import ZhouyiError
from extra.zhouyi import tec_res as T, image as IM, kern_tpc as K, enc as E
from extra.zhouyi.image import I, BUNDLE

def dma(sync="r7"): return BUNDLE(I("dma 0, %s, 0, 4, 0, 1, 0, r3, r9, r1" % sync, E.dma(E.DMA_DIRECT, sync, 0, E.DMA_LSRAM, E.DMA_GLOBAL, E.DMA_EXT2INT, E.DMA_USELESS, "r3", "r9", "r1")))
def mov(reg, v): return BUNDLE(I("mov %s, %d" % (reg, v), E.mov(reg, v)))
def raw_wait(n, mode=1): return [BUNDLE(I("sub r7, zero, %d" % n, E.sub("r7", "zero", n))), BUNDLE(I("wfe r7, %d" % mode, E.wfe("r7", mode)))]
def lint(body): return T.lint_sync(IM.vector_table() + body)

class TestFlags(unittest.TestCase):
  def test_helpers_emit_the_closures_bytes(self):
    self.assertEqual(T.sync_flag(2), [[("mov r7, 514", E.mov("r7", 514))]])
    self.assertEqual(T.wait_flags(3), raw_wait(9))
    self.assertEqual(T.wait_flags(0, 2), raw_wait(6))              # ~(1 | 4)
    self.assertEqual(T.sync_word(31), 31 * 257)                   # the broadcast is legal
  def test_flag4_refused_by_the_helpers(self):
    for bad in (4, 5, 30, -1, 32):
      with self.assertRaisesRegex(ZhouyiError, "flags 0-3"): T.sync_flag(bad)
      with self.assertRaisesRegex(ZhouyiError, "flag"): T.wait_flags(bad)
  def test_wfe_mode3_refused_by_the_helper(self):
    with self.assertRaisesRegex(ZhouyiError, "exception vector 2"): T.wait_flags(0, mode=3)
    with self.assertRaisesRegex(ZhouyiError, "names no flag"): T.wait_flags()

class TestLint(unittest.TestCase):
  def test_clean(self):
    r = lint(T.sync_flag(1) + [dma()] + T.wait_flags(1) + T.wait_flags(mode=2))
    self.assertEqual((r.raised, r.waited), ({1}, {1}))
  def test_selector4_request_refused(self):
    with self.assertRaisesRegex(ZhouyiError, "selects 4, which silently lands on flag 0"): lint([mov("r7", (4 << 8) | 4), dma()] + raw_wait(17))
  def test_wait_on_flag4_refused(self):
    with self.assertRaisesRegex(ZhouyiError, r"names flag\(s\) \[4\]"): lint(T.sync_flag(0) + [dma()] + raw_wait((1 << 4) + 1))
  def test_wfe_mode3_refused(self):
    with self.assertRaisesRegex(ZhouyiError, "wfe mode 3"): lint(T.sync_flag(0) + [dma()] + raw_wait(2, mode=3))
  def test_wait_on_unraised_flag_refused(self):
    with self.assertRaisesRegex(ZhouyiError, "names flag 2, which no request in the image raises"): lint(T.sync_flag(0) + [dma()] + T.wait_flags(2))
  def test_aiff_kick_counts_as_a_request(self):
    r = lint([mov("r11", 257), BUNDLE(I("aiff 0, r11, 0, r5, r6, r7", E.aiff(0, "r11", 0, "r5", "r6", "r7")))] + T.wait_flags(1))
    self.assertEqual(r.raised, {1})
  def test_untraced_operand_not_checked(self):
    lint([BUNDLE(I("add r7, r4, r5", E.add("r7", "r4", "r5"))), dma()] + T.wait_flags(3))   # nothing known about r7: no verdict
  def test_bundle_reads_before_writes(self):
    # mov r7 in the same bundle as the dma: the dma sees r7's OLD value (flag 1), not 1028
    b = BUNDLE(I("mov r7, 1028", E.mov("r7", 1028)), I("dma 0, r7, 0, 4, 0, 1, 0, r3, r9, r1", dma()[0][1]))
    self.assertEqual(lint(T.sync_flag(1) + [b] + T.wait_flags(1)).raised, {1})
  def test_text_only_runs_the_lint(self):
    with self.assertRaisesRegex(ZhouyiError, "lint refused"): IM.text_only(T.sync_flag(0) + [dma()] + T.wait_flags(2))

class TestKernelsPassTheLint(unittest.TestCase):
  def test_gemm_configurations_in_use(self):
    for a, kw in [((48, 3, 28), dict(c_stride=28 * 4 * 4608, drain="dma", a_rb_stride=2304, c_rb_stride=4608, b8=True)),
                  ((24, 6, 14), dict(drain="dma")), ((24, 6, 14), dict(drain="tec")), ((24, 6, 12), dict(drain="dma", rowmax=True)),
                  ((32, 3, 2), dict(drain="dma", b8=True, kw=4, bscale=True)), ((32, 3, 4), dict(drain="dma", b8=True, kw=4, bscale=True)),
                  *[((32, 3, 2), dict(drain="dma", b8=True, kw=4, bscale=True, rows=r)) for r in (1, 4, 5, 8, 9, 12)]]:
      IM.text_only(K.k_gemm_gs(*a, **kw))
    IM.text_only(K.k_gemm_gm(28, c_stride=28 * 4608, c_rb_stride=4608, b8=True)); IM.text_only(K.k_gm_stage())

class TestArena(unittest.TestCase):
  def test_places_in_order(self):
    L = T.Lsram("t"); self.assertEqual([L.alloc("a", 100), L.alloc("b", 64), L.alloc("c", 32)], [0, 128, 192])
    self.assertIn("b", L.dump())
  def test_lsram_overflow(self):
    L = T.Lsram("t"); L.alloc("a", 32000)
    with self.assertRaisesRegex(ZhouyiError, "past the usable 0x7fc0"): L.alloc("b", 768)
  def test_gemm_gs_overflow_names_the_region(self):
    with self.assertRaisesRegex(ZhouyiError, "codes \\+ scales"): K.k_gemm_gs(24, 6, 2, drain="dma", b8=True, kw=4, bscale=True)
  def test_misaligned(self):
    with self.assertRaisesRegex(ZhouyiError, "not 32-B aligned"): T.DescTable().slot_at("A slice", 40)
    with self.assertRaisesRegex(ZhouyiError, "not 32-B aligned"): T.Lsram("t").alloc("x", 64, at=16)
  def test_overlap(self):
    L = T.Lsram("t"); L.alloc("a", 256)
    with self.assertRaisesRegex(ZhouyiError, "overlaps a"): L.alloc("b", 64, at=128)
  def test_vld_tail(self):
    A = T.Arena("no reserve", 1024); A.alloc("x", 1000)
    with self.assertRaisesRegex(ZhouyiError, "reads 16 B past"): A.alloc("tail", 24, align=8, vld_tail=True)
    T.Lsram("t").alloc("to the stamps", 32768 - 64, vld_tail=True)     # the overhang lands in the reserved top: allowed
  def test_gsram_window(self):
    G = T.GsramWindow(); G.alloc("C", 28 * 3 * 768)
    with self.assertRaisesRegex(ZhouyiError, "past the usable 0x10000"): T.GsramWindow().alloc("C", 16 * 6 * 768)

if __name__ == "__main__": unittest.main()
