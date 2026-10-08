"""Kernel text in GM (runtime/support/zhouyi/textgm.py) without hardware: the placement, the job modes, the preloads, the ranking
(re-packing a full arena and patching the TCBs), the GSRAM scan, and the closed list of GSRAM users in the sources."""
import ast, ctypes, pathlib, re, struct, unittest
from tinygrad.runtime.support.zhouyi import ZhouyiError, dev, textgm
from tinygrad.runtime.support.zhouyi.launch import Launch
from tinygrad.runtime import ops_zhouyi
from extra.zhouyi import kern_tpc as K, enc as E, image as IM

ROOT = pathlib.Path(__file__).resolve().parents[2]


class FakeRaw:
  """A RawDevice over host memory: the real `view` / `gm_window` / `submit` order (policy first), submits recorded."""
  LOAD_OVERHANG = dev.RawDevice.LOAD_OVERHANG
  view = dev.RawDevice.view
  gm_window = dev.RawDevice.gm_window
  _text_sync_job = staticmethod(ops_zhouyi._text_sync_job)
  def __init__(self):
    self.asid0, self.grid_id, self.bufs, self._mem, self._next, self._job, self.log = 0x1_0000_0000, 7, [], [], 0x20_0000, 0, []
    self.text_gm = textgm.TextGM(self)
  def req_buf(self, n, data_type=0):
    m = -(-(n + 16) // 4096) * 4096
    mem = (ctypes.c_uint8 * m)(); self._mem.append(mem)
    b = dev.Buffer(self.asid0 + self._next, 0, n, ctypes.addressof(mem), map_bytes=m); self._next += m + 4096
    self.bufs.append(b); return b
  def dev_addr(self, b): return b.pa - self.asid0
  def free_buf(self, b):
    self.text_gm.freed(b); self.bufs.remove(b)
  def _submit(self, head, first, last, dbg_core=None):
    self._job += 1
    grid = ctypes.string_at(self._va(head), dev.TCB_LEN)
    self.log.append((head, struct.unpack_from("<I", grid, 16)[0], struct.unpack_from("<I", grid, 24)[0], struct.unpack_from("<Q", grid, 32)[0]))
    return self._job
  def submit(self, head, first, last, dbg_core=None):
    self.text_gm.before_submit(head); return self._submit(head, first, last)
  def wait_jobs(self, ids, timeout_s=None): return {}
  def _va(self, pa):
    for b in self.bufs:
      if b.pa <= pa < b.pa + b.map_bytes: return b.va + (pa - b.pa)
    raise KeyError(hex(pa))
  # helpers
  def read(self, addr, n): return ctypes.string_at(self._va(addr + self.asid0), n)


def image(tag: int, nbundles: int = 40) -> bytes:
  """A distinct image: `mov r5, tag` bundles behind the vector table, then the epilogue."""
  return IM.text_only([[("mov r5, %d" % tag, E.mov("r5", tag))]] * nbundles + IM.epilogue(), sync_lint=False)


def chain(raw, imgs, gm_pa=None):
  """One job: a 1-task group per image (each its own Launch's buffers), built by the real build_group_tcbs."""
  lays = [Launch(raw, t, b"") for t in imgs]
  for l in lays: l.chain([1])
  a = raw.dev_addr
  groups = [dev.GroupSlot(spc=a(l.text), cp=a(l.cnst), tasks=l.task_slots()) for l in lays]
  tcbs = raw.req_buf(dev.chain_tcbs(len(groups)) * dev.TCB_LEN)
  dev.build_group_tcbs(raw, tcbs, groups, gm_pa=gm_pa)
  return lays, tcbs


def spcs(tcbs, n): return [struct.unpack_from("<I", ctypes.string_at(tcbs.va + (dev.TASK_INDEX + i) * dev.TCB_LEN + dev.T_SPC, 4))[0] for i in range(n)]
def grid(tcbs): return struct.unpack_from("<I", ctypes.string_at(tcbs.va + 16, 4))[0], struct.unpack_from("<Q", ctypes.string_at(tcbs.va + 32, 8))[0]


class TestTextGM(unittest.TestCase):
  def setUp(self):
    self.saved = (textgm.TEXT_GM_CAP, textgm.RANK_FIRST, textgm.PROMOTE_QUIET, textgm.PROMOTE_MAX)
    ops_zhouyi._SYNC.clear()
    from tinygrad.runtime.support.zhouyi import launch; launch._IMAGES.clear()
  def tearDown(self): textgm.TEXT_GM_CAP, textgm.RANK_FIRST, textgm.PROMOTE_QUIET, textgm.PROMOTE_MAX = self.saved

  def _win(self, raw): return raw.gm_window()[1]
  def _inarena(self, raw, lay): w = self._win(raw); return w <= raw.dev_addr(lay.text) < w + textgm.TEXT_GM_CAP

  def test_promotion_and_preloads(self):
    raw = FakeRaw(); tg = raw.text_gm
    a, b = image(1), image(2)
    lays, tcbs = chain(raw, [a, b])
    self.assertFalse(any(self._inarena(raw, l) for l in lays), "new images start in DDR")
    self.assertEqual(grid(tcbs), (1, raw.gm_window()[0]), "a text job runs GM on over the window")
    for _ in range(textgm.PROMOTE_QUIET - 1): raw.submit(tcbs.pa, 0, 0)
    self.assertEqual(len(raw.log), textgm.PROMOTE_QUIET - 1, "nothing promoted while new: no preload")
    raw.submit(tcbs.pa, 0, 0)                                        # quiet long enough: one batch, one preload
    self.assertEqual(tg.stats["promotions"], 1); self.assertEqual(raw.log[-2][2] >> 30, 1, "a SYNC_TO_GM preload first")
    self.assertEqual(raw.log[-2][2] & 0xFFFF, -(-tg.top // 4096))
    self.assertTrue(all(self._inarena(raw, l) for l in lays))
    self.assertEqual(spcs(tcbs, 2), [raw.dev_addr(l.text) for l in lays], "the TCBs follow their images")
    for l, t in zip(lays, (a, b)): self.assertEqual(raw.read(raw.dev_addr(l.text), len(t)), t)
    n = len(raw.log); raw.submit(tcbs.pa, 0, 0); self.assertEqual(len(raw.log), n + 1, "no preload: GM holds the arena")
    # a new image: DDR until the next batch, which preloads again
    lays2, tcbs2 = chain(raw, [image(3, 4)])
    self.assertFalse(self._inarena(raw, lays2[0]))
    for _ in range(textgm.PROMOTE_QUIET): raw.submit(tcbs2.pa, 0, 0)
    self.assertTrue(self._inarena(raw, lays2[0])); self.assertEqual(tg.stats["preloads"], 2)
    self.assertEqual(raw.read(raw.dev_addr(lays2[0].text), len(image(3, 4))), image(3, 4))
    self.assertEqual(spcs(tcbs2, 1), [raw.dev_addr(lays2[0].text)])

  def test_promote_max_while_compiling(self):
    textgm.PROMOTE_MAX = 3
    raw = FakeRaw(); tg = raw.text_gm
    for i in range(3):                                              # a new image before every job: never quiet
      lays, tcbs = chain(raw, [image(10 + i)]); raw.submit(tcbs.pa, 0, 0)
    self.assertEqual(tg.stats["promotions"], 1, "promoted after PROMOTE_MAX jobs all the same")

  def test_gsram_user_runs_gm_off(self):
    textgm.PROMOTE_QUIET = 1
    raw = FakeRaw(); tg = raw.text_gm
    g = IM.text_only(K.k_gsram_dump(), sync_lint=False)
    lays, tcbs = chain(raw, [image(1), g])
    self.assertEqual(grid(tcbs)[0], 0, "a job with a GSRAM user runs GM off")
    raw.submit(tcbs.pa, 0, 0)
    self.assertTrue(self._inarena(raw, lays[0]), "its plain image is promoted all the same (a GM-off job fetches the window's DDR backing)")
    self.assertFalse(self._win(raw) <= raw.dev_addr(lays[1].text) < self._win(raw) + dev.GM_WINDOW, "a GSRAM user's text stays in DDR")
    self.assertEqual(spcs(tcbs, 2), [raw.dev_addr(l.text) for l in lays])
    self.assertEqual(len(raw.log), 1, "no preload for a GM-off job"); self.assertEqual(tg.stats["off_jobs"], 1)
    lay = Launch(raw, image(9), b"", gsram=True)                    # marked by its caller: a GSRAM address given at run time
    self.assertIn(raw.dev_addr(lay.text), tg.gsram)

  def test_data_job(self):
    textgm.PROMOTE_QUIET = 1
    raw = FakeRaw(); tg = raw.text_gm
    a = image(1)
    lays, tcbs = chain(raw, [a, image(2)])
    raw.submit(tcbs.pa, 0, 0); self.assertTrue(self._inarena(raw, lays[0]))
    dl, dt = chain(raw, [a, image(5)], gm_pa=raw.gm_window()[0])
    w = self._win(raw)
    for x, t in zip(spcs(dt, 2), [a, image(5)]):
      self.assertFalse(w <= x < w + dev.GM_WINDOW, "a GM-data job's text runs from DDR")
      self.assertEqual(raw.read(x, len(t)), t)
    self.assertEqual(grid(dt), (1, raw.gm_window()[0]))
    n = len(raw.log); raw.submit(dt.pa, 0, 0); self.assertEqual(len(raw.log), n + 1, "no preload before a data job")
    self.assertFalse(tg.in_gm)
    raw.submit(tcbs.pa, 0, 0); self.assertEqual(raw.log[-2][2] >> 30, 1, "the next text job preloads the arena again")
    with self.assertRaises(ZhouyiError): chain(raw, [IM.text_only(K.k_gsram_dump(), sync_lint=False)], gm_pa=raw.gm_window()[0])

  def test_unknown_job_and_freed_tcbs(self):
    textgm.PROMOTE_QUIET = 1
    raw = FakeRaw(); tg = raw.text_gm
    lays, tcbs = chain(raw, [image(1)])
    raw.submit(tcbs.pa, 0, 0); self.assertTrue(tg.in_gm)
    other = raw.req_buf(dev.chain_tcbs(1) * dev.TCB_LEN)
    raw.submit(other.pa, 0, 0); self.assertFalse(tg.in_gm, "a chain built elsewhere may have used GM")
    j = tg.jobs[tcbs.pa]; raw.free_buf(tcbs); self.assertFalse(j.alive); self.assertNotIn(tcbs.pa, tg.jobs)

  def test_gemm_a_region(self):
    """The rows-mode GEMMs' A staging buffer lies above the arena, below the C partials, inside the window; the preload
    stops below it; once handed out, a GM-off (GSRAM user) or GM-data job is refused and text jobs run as before."""
    textgm.PROMOTE_QUIET = 1
    raw = FakeRaw(); tg = raw.text_gm; w = self._win(raw)
    lo, hi = textgm.gemm_a_region(raw)
    self.assertEqual((lo, hi), (w + textgm.GEMM_A_OFF, w + textgm.GEMM_A_OFF + textgm.GEMM_A_CAP))
    self.assertGreaterEqual(lo, w + textgm.TEXT_GM_CAP, "above the text arena (a preload covers [0, top) <= TEXT_GM_CAP)")
    self.assertLessEqual(hi, textgm.gemm_c_base(raw, 0) if textgm.TEXT_GM else w + textgm.GEMM_C_OFF, "below the C partials")
    self.assertGreaterEqual(textgm.GEMM_A_CAP, 17408 * 8 * 3, "rows 9-12 at K 17408 (rt 3)")
    lays, tcbs = chain(raw, [image(1)]); raw.submit(tcbs.pa, 0, 0)
    tg.arm_a(); self.assertTrue(tg.a_armed)
    lays, tcbs = chain(raw, [image(2)]); raw.submit(tcbs.pa, 0, 0)
    self.assertEqual(grid(tcbs), (1, raw.gm_window()[0]), "text jobs run GM on as before")
    self.assertLessEqual(tg.top, textgm.GEMM_A_OFF)
    with self.assertRaises(ZhouyiError): chain(raw, [image(3), IM.text_only(K.k_gsram_dump(), sync_lint=False)])
    with self.assertRaises(ZhouyiError): chain(raw, [image(4)], gm_pa=raw.gm_window()[0])

  def test_full_arena_ranks_by_launches(self):
    textgm.TEXT_GM_CAP, textgm.RANK_FIRST, textgm.PROMOTE_QUIET = 2 * 4096, 8, 1
    raw = FakeRaw(); tg = raw.text_gm
    imgs = [image(i, 120) for i in range(4)]            # ~2.4 KB each: 3 fit
    lays, tcbs = zip(*[chain(raw, [t]) for t in imgs])
    raw.submit(tcbs[0].pa, 0, 0)
    self.assertEqual([self._inarena(raw, l[0]) for l in lays], [True, True, True, False], "promoted in order (equal counts)")
    self.assertTrue(tg.full); self.assertEqual(tg.stats["ddr_images"], 1)
    for _ in range(8): raw.submit(tcbs[3].pa, 0, 0)       # the DDR image is the hot one
    raw.submit(tcbs[1].pa, 0, 0)
    self.assertGreaterEqual(tg.stats["ranks"], 1)
    self.assertTrue(self._inarena(raw, lays[3][0]), "the hottest image moved into the arena")
    self.assertEqual(sum(self._inarena(raw, l[0]) for l in lays), 3)
    for l, t, tc in zip(lays, imgs, tcbs):
      a = raw.dev_addr(l[0].text)
      self.assertEqual(spcs(tc, 1)[0], a, "every task TCB follows its image")
      self.assertEqual(raw.read(a, len(t)), t, "and finds its bytes there")
    self.assertEqual(tg.stats["preloads"], 2, "the re-packed arena preloaded again")


class TestGsramScan(unittest.TestCase):
  PROBES = {  # every GSRAM probe kernel: its image must be flagged by the scan
    "k_lsram_port": lambda: K.k_lsram_port("v1", space="gsram"), "k_lsram_latency": lambda: K.k_lsram_latency(4, space="gsram"),
    "k_lsram_dma_contend": lambda: K.k_lsram_dma_contend("dmaload", space="gsram"), "k_lsram_port2": lambda: K.k_lsram_port2("st1", space="gsram"),
    "k_sram_reach": lambda: K.k_sram_reach("dma", "gsram"), "k_sram_share": lambda: K.k_sram_share("gsram"),
    "k_sram_to_sram": lambda: K.k_sram_to_sram("dma"), "k_sram_dma": lambda: K.k_sram_dma("g2l_ig_ol"), "k_gm_remap": K.k_gm_remap,
    "k_gsram_dump": K.k_gsram_dump, "k_gsram_stamp": lambda: K.k_gsram_stamp([0, 64]), "k_gsram_probe_all": K.k_gsram_probe_all,
    "k_gsram_read": lambda: K.k_gsram_read([0, 64]), "k_gsram_rw": lambda: K.k_gsram_rw(vector=True), "k_gsram_roundtrip": K.k_gsram_roundtrip,
    "k_gsram_reach_off": lambda: K.k_gsram_reach_off(0),
  }
  # an image that gets a GSRAM address at run time: the scan cannot see it, so its launcher marks it or keeps it out of GSRAM
  RUNTIME = {
    "k_gemm_gs": "its C-partials base is an argument: textgm.gemm_c_base (the GM window's C regions with text in GM)",
    "k_mma32_bench": "a timing probe given the GSRAM base in r0: Launch(..., gsram=True), or ZHOUYI_TEXT_GM=0",
  }
  # no kernel of their own: constants, encoders, cost models, docs, the policy itself
  OTHER = {"enc.py:dma", "gemm_plan.py:Config", "gemm_plan.py:task_cycles", "kern_tpc.py:_dma_fill_and_wait", "kern_tpc.py:_gsram_base",
           "kern_tpc.py:gsram_arm", "kern_tpc.py:k_gemm_gm", "ops.py:GemmGSRunner", "tec_res.py:GsramWindow", "dev.py:gm_fields",
           "dev.py:build_group_tcbs", "hangid.py:_Where", "launch.py:Launch", "textgm.py:gsram_refs", "textgm.py:gemm_c_base",
           "textgm.py:_Img", "textgm.py:TextGM"}

  def test_probes_flagged(self):
    for name, mk in self.PROBES.items(): self.assertTrue(textgm.gsram_refs(IM.text_only(mk(), sync_lint=False)), name)

  def test_production_images_clean(self):
    G = K.k_gemm_gs
    kw = dict(drain="dma", b8=True, kw=4, bscale=True)
    for name, mk in {
        "lin48": lambda: G(48, 3, 28, c_stride=28 * 6 * 768, drain="dma", a_rb_stride=24 * 96, c_rb_stride=6 * 768, b8=False),
        "lin48 b8": lambda: G(48, 3, 28, c_stride=28 * 6 * 768, drain="dma", a_rb_stride=24 * 96, c_rb_stride=6 * 768, b8=True),
        "fp16 tec": lambda: G(24, 6, 8, c_stride=8 * 6 * 768, drain="tec"), "fp16": lambda: G(24, 6, 8, c_stride=8 * 6 * 768, drain="dma"),
        "rowmax": lambda: G(24, 6, 8, c_stride=8 * (6 * 768 + 192), drain="dma", rowmax=True),
        "e4m3": lambda: G(32, 3, 2, c_stride=2 * 3 * 768, **kw), "e4m3 rows4 single": lambda: G(32, 3, 2, c_stride=2 * 3 * 768, rows=4, scales="single", **kw),
        "q8 rows8": lambda: G(8, 3, 2, c_stride=2 * 3 * 768, rows=8, q8=True, **kw), "q8f rows4": lambda: G(32, 3, 2, c_stride=2 * 3 * 768, rows=4, q8f=True, **kw),
        "tern rows4": lambda: G(32, 3, 2, c_stride=2 * 3 * 768, rows=4, scales="single", tern=True, tscale="f16", **kw),
        "tern rows8": lambda: G(32, 3, 2, c_stride=2 * 3 * 768, rows=8, scales="single", tern=True, tscale="f16", **kw),
        "tern rows12": lambda: G(32, 3, 2, c_stride=2 * 3 * 768, rows=12, scales="single", tern=True, **kw),
        "tern prefill": lambda: G(32, 3, 4, c_stride=4 * 3 * 768, scales="single", tern=True, tscale="f16", **kw),
        "gemm_gm": lambda: K.k_gemm_gm(28, c_stride=28 * 6 * 768, c_rb_stride=4608, b8=True), "gm_stage": K.k_gm_stage, "e4m3_stream": K.k_e4m3_stream}.items():
      self.assertEqual(textgm.gsram_refs(IM.text_only(mk())), [], name)
    self.assertEqual(textgm.gsram_refs(ops_zhouyi._BARRIER_TEXT), [])

  def test_gemm_c_base_not_gsram(self):
    raw = FakeRaw(); old = textgm.TEXT_GM
    try:
      textgm.TEXT_GM = True
      for t in range(12):
        c = textgm.gemm_c_base(raw, t)
        self.assertFalse(textgm.GSRAM_LO <= c < textgm.GSRAM_HI)
        self.assertEqual(c, raw.gm_window()[1] + textgm.GEMM_C_OFF + 65536 * t)
        self.assertGreaterEqual(c, raw.gm_window()[1] + textgm.TEXT_GM_CAP, "above the text arena")
      self.assertLessEqual(textgm.gemm_c_base(raw, 11) + 65536, raw.gm_window()[1] + dev.GM_WINDOW)
    finally: textgm.TEXT_GM = old

  def test_sources_closed_list(self):
    """Every function / class in the Zhouyi sources that names GSRAM (or a GSRAM DMA space, or a 0xF800.. constant) is listed above.
    A new GSRAM user fails here until it is either flagged by the scan (a probe), marked at its launch, or moved out of GSRAM."""
    pat = re.compile(r"gsram|dma_shared|0xf80[0-3]", re.I)
    files = set(ROOT.glob("extra/zhouyi/*.py")) | set(ROOT.glob("tinygrad/runtime/support/zhouyi/*.py")) | set(ROOT.glob("tinygrad/runtime/**/*zhouyi*.py"))
    found = set()
    for f in sorted(files):
      src = f.read_text()
      for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and pat.search(ast.get_source_segment(src, n) or ""): found.add(f"{f.name}:{n.name}")
    want = {f"kern_tpc.py:{n}" for n in self.PROBES} | {f"kern_tpc.py:{n}" for n in self.RUNTIME} | self.OTHER
    self.assertEqual(sorted(found - want), [], "unlisted GSRAM users: flag, mark or move them (textgm.py), then list them here")
    self.assertEqual(sorted(want - found), [], "listed but gone: drop them from the list")


class TestVectorStub(unittest.TestCase):
  def test_shared_handler(self):
    from tinygrad.runtime.support import compiler_zhouyi as C
    self.assertEqual(C.VECTOR_BUNDLES, 46)
    slots = [l for l in C.VECTORS if "b .Lexc" in l]
    self.assertEqual(slots, [f"\t{{ mov r3, {k}; b .Lexc; }}" for k in range(1, 16)])
    self.assertIn("\t{ st r3, [r2+0]; }", C.HANDLER)

if __name__ == "__main__":
  unittest.main()
