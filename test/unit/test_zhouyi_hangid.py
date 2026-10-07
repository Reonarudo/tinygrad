"""ZHOUYI_HANG_ID (runtime/support/zhouyi/hangid.py) without hardware: a fake RawDevice over host memory, the real TCB builder,
the real `wait_jobs` with QUERY_STATUS faked to report one job in EXCEPTION."""
import ctypes, importlib.util, pathlib, subprocess, types, unittest
from tinygrad.runtime.autogen import aipu
from tinygrad.runtime.support.zhouyi import ZhouyiError, dev, hangid
from tinygrad.runtime.support.zhouyi.launch import CYCLE_STAMP_OFF

class FakeRaw:
  """`req_buf`/`dev_addr`/`cache_invalidate`/`submit` over ctypes memory; `wait_jobs` is the real one."""
  wait_jobs = dev.RawDevice.wait_jobs
  QUERY_SLOTS = dev.RawDevice.QUERY_SLOTS
  def __init__(self):
    self.asid0, self.fd, self.grid_id, self.bufs, self._banked, self._mem, self._next, self._job = 0x1_0000_0000, -1, 7, [], {}, [], 0x20_0000, 0x4660000
  def req_buf(self, n, data_type=0):
    m = -(-(n + 16) // 4096) * 4096
    mem = (ctypes.c_uint8 * m)(); self._mem.append(mem)
    b = dev.Buffer(self.asid0 + self._next, 0, n, ctypes.addressof(mem), map_bytes=m); self._next += m + 4096
    self.bufs.append(b); return b
  def dev_addr(self, b): return b.pa - self.asid0
  def cache_invalidate(self, b): pass
  def free_buf(self, b): self.bufs.remove(b)
  def submit(self, head, first, last, dbg_core=None):
    self._job += 1; return self._job

def fake_query(failing):
  def q(fd, __payload):
    __payload.poll_cnt = 1
    __payload.status[0].job_id, __payload.status[0].state = failing, 2
  return q

class TestHangId(unittest.TestCase):
  def setUp(self):
    self.level, self.log, self.q = hangid.LEVEL, hangid.LOG, aipu.AIPU_IOCTL_QUERY_STATUS
    hangid._TEXTS.clear(); hangid._JOBS.clear(); hangid._PP.clear(); hangid.CONTEXT.clear(); hangid.NAMER = None
  def tearDown(self): hangid.LEVEL, hangid.LOG, aipu.AIPU_IOCTL_QUERY_STATUS = self.level, self.log, self.q

  def test_parse(self):
    for v, want in [(None, (0, None)), ("", (0, None)), ("0", (0, None)), ("1", (1, None)), ("2", (2, None)),
                    ("1:/tmp/h.txt", (1, "/tmp/h.txt")), ("2:/a/b:c.txt", (2, "/a/b:c.txt")), ("2:", (2, None))]:
      self.assertEqual(hangid.parse(v), want, v)
    with self.assertRaises(ValueError): hangid.parse("/tmp/h.txt")

  def test_level1_log_path(self):
    """ZHOUYI_HANG_ID=1:<path>: the report is raised as at level 1 and also appended to the file, once per failure."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
      hangid.LEVEL, hangid.LOG = hangid.parse("1:" + os.path.join(d, "hang.txt"))
      raw = FakeRaw(); sub = self._job(raw, 1)
      for _ in range(2):
        jid = raw.submit(*sub); aipu.AIPU_IOCTL_QUERY_STATUS = fake_query(jid)
        with self.assertRaises(ZhouyiError) as cm: raw.wait_jobs([jid])
      with open(hangid.LOG) as f: text = f.read()
      self.assertEqual(text.count("[ZHOUYI_HANG_ID]"), 2)
      self.assertIn(str(cm.exception).split("\n", 1)[1], text)

  def _job(self, raw, level):
    """A fused job: g0 = kernel A (4 tasks), g1 = kernel B (4 tasks), g2 = a barrier; A's args name two buffers."""
    hangid.LEVEL = level
    if level: hangid.wrap_submit(raw)
    x, y = raw.req_buf(24 * 5120 * 4), raw.req_buf(8192)
    hangid.NAMER = lambda: [("x_in", x), ("q_rows", y)]
    texts = [raw.req_buf(256) for _ in range(3)]
    for t, name, frame in zip(texts, ["rows32", "r_24_16_256", "barrier"], [1056, 48, 0]):
      hangid.register_text(raw.dev_addr(t), name, frame=frame, lib=0, stamps=name != "barrier")
    groups, self.params = [], []
    for gi, w in enumerate([4, 4, 1]):
      tasks = []
      for _ in range(w):
        s, p, d = raw.req_buf(8192), raw.req_buf(4096), raw.req_buf(4096); self.params.append(p)
        (ctypes.c_uint32 * 4).from_address(p.va)[:] = [raw.dev_addr(x), raw.dev_addr(y) + 0x40, 21, 0]
        tasks.append(dev.TaskSlot(sp=raw.dev_addr(s) + s.nbytes, pp=raw.dev_addr(p), dp=raw.dev_addr(d)))
      groups.append(dev.GroupSlot(spc=raw.dev_addr(texts[gi]), cp=0, tasks=tasks))
    if level >= 2: hangid.note_params(raw, self.params)
    tcbs = raw.req_buf(dev.chain_tcbs(9) * dev.TCB_LEN)
    dev.build_group_tcbs(raw, tcbs, groups, dep=[0, dev.DEP_PRE_ALL, dev.DEP_PRE_ALL])
    return (tcbs.pa, tcbs.pa + dev.TASK_INDEX * dev.TCB_LEN, tcbs.pa + (dev.TASK_INDEX + 8) * dev.TCB_LEN)

  def test_off_message_unchanged(self):
    raw = FakeRaw(); sub = self._job(raw, 0)
    jid = raw.submit(*sub); aipu.AIPU_IOCTL_QUERY_STATUS = fake_query(jid)
    with self.assertRaises(ZhouyiError) as cm: raw.wait_jobs([jid])
    self.assertEqual(str(cm.exception), f"ZHOUYI job {jid:#x} state=2 (EXCEPTION)")
    self.assertNotIn("submit", vars(raw))                      # level 0: submit is not wrapped

  def test_level1_report(self):
    raw = FakeRaw(); sub = self._job(raw, 1); hangid.CONTEXT.update(phase="prefill", layer=11)
    jid = raw.submit(*sub); aipu.AIPU_IOCTL_QUERY_STATUS = fake_query(jid)
    with self.assertRaises(ZhouyiError) as cm: raw.wait_jobs([jid])
    msg = str(cm.exception); print(msg)
    self.assertTrue(msg.startswith(f"ZHOUYI job {jid:#x} state=2 (EXCEPTION)\n[ZHOUYI_HANG_ID]"))
    for want in ["'layer': 11", "9 task(s) in 3 group(s)", "g0 (TCB group 0): rows32 -- 4 task(s), dep none", "g1 (TCB group 1): r_24_16_256",
                 "dep PRE_ALL", "barrier", "'x_in'", "'q_rows+0x40' [8192 B all sets]", "a2=21", "frame 1056 B", "exception vector 0", "all sets"]:
      self.assertIn(want, msg)
    self.assertNotIn("stamps", msg)

  def test_level2_first_unfinished_group(self):
    raw = FakeRaw(); sub = self._job(raw, 2)
    for p in self.params: (ctypes.c_uint32 * 2).from_address(p.va + CYCLE_STAMP_OFF)[:] = [5, 9]   # stale values from a "previous" run
    jid = raw.submit(*sub)                                                                           # poisons every stamping task
    for p in self.params: self.assertEqual(list((ctypes.c_uint32 * 2).from_address(p.va + CYCLE_STAMP_OFF)), [hangid.POISON] * 2)
    for p in self.params[:4] + self.params[5:6]: (ctypes.c_uint32 * 2).from_address(p.va + CYCLE_STAMP_OFF)[:] = [100, 350]   # g0 done, g1 1/4
    aipu.AIPU_IOCTL_QUERY_STATUS = fake_query(jid)
    with self.assertRaises(ZhouyiError) as cm: raw.wait_jobs([jid])
    msg = str(cm.exception); print(msg)
    self.assertIn("(250 cycles)", msg)
    self.assertIn("3/4 task(s) of g1 never reached the epilogue", msg)
    self.assertIn("=> first group with unfinished tasks: g1 r_24_16_256", msg)

  def test_graph_report(self):
    from tinygrad.runtime.graph.zhouyi import ZhouyiGraph
    raw = FakeRaw(); hangid.LEVEL = 1
    bx, by = raw.req_buf(4096), raw.req_buf(100)
    def ji(name, gs, member=False):
      prg = types.SimpleNamespace(display_name=name, chain_member=True) if member else types.SimpleNamespace(p=types.SimpleNamespace(name=name, global_size=[gs, 1, 1]), lib=b"x")
      return types.SimpleNamespace(prg=prg, bufs=[bx, by])
    jc = [ji("E_a", 1), ji("rows32", 12), ji("\x1b[33mTEC gemm_gs\x1b[0m", 13, True), ji("r_b", 4), ji("E_c", 1)]
    chain = types.SimpleNamespace(items=[(jc[1], jc[1].prg, 12), (jc[2], jc[2].prg, 13), (jc[3], jc[3].prg, 4)], layout=[4, 4, 1], closing_barrier=False)
    g = ZhouyiGraph.__new__(ZhouyiGraph)
    g.jit_cache, g.plan, g._raw, g.n_chained = jc, [("solo", jc[0]), ("chain", chain), ("solo", jc[4])], raw, 3
    g._j_of = {id(x): j for j, x in enumerate(jc)}
    msg = g._hang_report(1, "frozen"); print(msg)
    for want in ["step 1 [chain] raised; steps 0..0 completed", "jit[1] rows32 gs=12 (job groups g0-g3: a barrier, then one per core)",
                 "jit[2] TEC gemm_gs (job groups g4-g7)", "jit[3] r_b gs=4 (job group g8)", "previous launch: jit[0] E_a gs=1",
                 "next launch:     jit[4] E_c gs=1", ">> step 1", "step 0 [solo] done"]:
      self.assertIn(want, msg)

  def test_compiler_unchanged_when_off(self):
    """Level 0: the image text (VECTORS, epilogue) and so the compile cache key are the parent commit's."""
    here = pathlib.Path(__file__).resolve().parents[2]
    old = subprocess.run(["git", "show", "HEAD~0:tinygrad/runtime/support/compiler_zhouyi.py"], cwd=here, capture_output=True, text=True).stdout
    if "_stamp" in old:   # once committed, compare with the commit before the switch
      old = subprocess.run(["git", "show", "7a238a908:tinygrad/runtime/support/compiler_zhouyi.py"], cwd=here, capture_output=True, text=True).stdout
      if not old: self.skipTest("the reference commit 7a238a908 is not in this checkout's history")
    spec = importlib.util.spec_from_loader("old_compiler", loader=None); m = importlib.util.module_from_spec(spec); exec(old, m.__dict__)
    from tinygrad.runtime.support import compiler_zhouyi as new
    if hangid.LEVEL < 2 and new.STAMP_START == "":
      # the vector table itself changed since (one shared handler, textgm.py's arena): level 0 adds no stamp to it
      self.assertFalse(any("209" in l for l in new.VECTORS))
      self.assertEqual(new.VECTORS[-1], new.FP_ENABLE)
      self.assertEqual(new.epilogue(3), m.epilogue(3))

if __name__ == "__main__": unittest.main()
