import ctypes, functools, hashlib, os, pathlib, re, struct, tempfile
from tinygrad.device import Compiler, CompileError
from tinygrad.runtime.support.zhouyi import ZhouyiError, elf as _elf, hangid as _hid
from tinygrad.runtime.support.zhouyi.launch import BLOB_HEADER, BLOB_VERSION, CYCLE_STAMP_OFF

# libaiputoolchain.so exports btaipucc (clang), btaipuas (llvm-mc) and btaipuld as int(argc, argv) entry points, driven in-process.
# It is the vendor's AIPU compiler toolchain as installed on the device (not bundled): ZHOUYI_TOOLCHAIN_DIR, else TOOLCHAIN_DIR.
# Its DT_NEEDED siblings are loaded first by absolute path (an LD_LIBRARY_PATH would shadow system libraries). It statically
# carries LLVM 11, so it must be loaded RTLD_LOCAL or its symbols clash with a later system libLLVM (the CPU backend).
TOOLCHAIN_DIR = "/usr/share/cix/lib/onnxruntime"
TOOLCHAIN_LIBS = ("libaipu_buildtool.so", "libaipu_toolchain_core.so", "libaiputoolchain.so")
CC1_ARGS = ["-cc1", "-triple", "aipux2--", "-S", "-disable-free", "-mrelocation-model", "static", "-mthread-model", "single",
            "-fno-jump-tables", "-mframe-pointer=all", "-cl-std=CL1.2", "-target-cpu", "X2_1204", "-target-feature", "-long-calls",
            "-target-feature", "-short-bundle", "-mfloat-abi", "hard", "-O2"]
AS_ARGS = ["-triple", "aipux2--", "-mcpu=X2_1204", "-filetype=obj"]

@functools.cache
def toolchain_lib() -> ctypes.CDLL:
  d = os.getenv("ZHOUYI_TOOLCHAIN_DIR", TOOLCHAIN_DIR)
  if not os.path.isfile(f"{d}/libaiputoolchain.so"):
    raise CompileError(f"ZHOUYI: the AIPU compiler toolchain (libaiputoolchain.so) is not in {d}; install the vendor's NPU toolchain "
                       "package, or set ZHOUYI_TOOLCHAIN_DIR to the directory holding libaiputoolchain.so and its sibling libraries")
  for dep in TOOLCHAIN_LIBS: lib = ctypes.CDLL(f"{d}/{dep}", mode=ctypes.RTLD_LOCAL)
  for sym in ("btaipucc", "btaipuas", "btaipuld"):
    getattr(lib, sym).argtypes, getattr(lib, sym).restype = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)], ctypes.c_int
  return lib

def _run(tool:str, args:list[str]):
  argv = [tool] + args
  if (rc := getattr(toolchain_lib(), tool)(len(argv), (ctypes.c_char_p * (len(argv)+1))(*[a.encode() for a in argv], None))) != 0:
    raise CompileError(f"ZHOUYI: {tool} returned {rc}")

# The image: slot 0 of a 16-entry vector table branches to main; every other slot k (an exception) is ONE bundle -- `mov r3, k` beside
# a branch to the shared handler, which records the vector number in the task's private buffer (word 0; the TCB pointer comes from
# ctrl0[20] as in the kernel prologue, the loaded register is used only PAD bundles later since the load latency is exposed), writes
# the data cache back and spins -- a timed-out job can then say which exception each task took (`ZhouyiProgram.exception_vectors`).
# 46 bundles (736 B) before main, where one handler per vector took 452 (7.2 KB, most of a small kernel's image: textgm.py's GM
# text arena holds the images). Note: a memory access that never completes is NOT an exception: it presents as a hang with no
# vector recorded. Main enables the fp units (without it every vector instruction traps), runs the kernel, and each `br ra` becomes
# the vendor's task epilogue: flush the data cache, wait for it, exit.
FP_ENABLE = ("\t{ mfctrl1 r9, 2; }\n\t{ bfs r9, r9, 31, 0; }\n\t{ mtctrl1 r9, 2; }\n"
             "\t{ mfctrl1 r9, 0x10; }\n\t{ bfs r9, r9, 31, 0; }\n\t{ mtctrl1 r9, 0x10; }")
PAD = 8
FLUSH = ["\t{ mfctrl0 r2, 0x82; }", "\t{ orl r2, r2, 4; }", "\t{ mtctrl0 r2, 0x82; }", "\t{ mfctrl0 r3, 0x90; }", "\t{ movh r3, 381; }", "\t{ mtctrl0 r3, 0x90; }"]
# the shared handler (r3 = the vector number, set by the slot): r3 is not touched until its store
HANDLER = [".Lexc:", "\t{ mfctrl0 r2, 20; }", *["\t{ nop; }"] * PAD, "\t{ ld r2, [r2+60]; }", *["\t{ nop; }"] * PAD,
           "\t{ st r3, [r2+0]; }", *FLUSH, ".Lmf:", "\t{ mfctrl0 r2, 0x82; }", "\t{ andl r2, r2, 4; }", "\t{ cbnz r2, .Lmf; }", "\t{ b .Lhang; }"]
# ZHOUYI_HANG_ID=2 (hangid.py): the TEC cycle counter (ctrl0[0xd1]) into word `slot` of the task's param tail (pp + CYCLE_STAMP_OFF):
# slot 0 after the fp enable, slot 1 before each task epilogue (whose writeback lands it in DDR). r2/r3 as the epilogue and the
# vectors use them, r9 (the fp enable's scratch) as padding: like the vectors, every loaded or control-read value is used only PAD
# bundles later. Empty (and the image unchanged) unless the switch is at 2.
def _stamp(slot:int) -> str:
  pad = [f"\t{{ mov r9, {slot}; }}"] * PAD
  return "\n".join(["\t{ mfctrl0 r2, 20; }", *pad, "\t{ ld r2, [r2+56]; }", *pad, f"\t{{ mov r3, {CYCLE_STAMP_OFF}; }}", "\t{ add r2, r2, r3; }",
                    "\t{ mfctrl0 r3, 209; }", *pad, f"\t{{ st r3, [r2+{4 * slot}]; }}"])
STAMP_START, STAMP_END = (_stamp(0), _stamp(1) + "\n\t") if _hid.LEVEL >= 2 else ("", "")
def epilogue(n:int) -> str:
  return STAMP_END + (f"{{ mfctrl0 r2, 0x82; }}\n\t{{ orl r2, r2, 4; }}\n\t{{ mtctrl0 r2, 0x82; }}\n\t{{ mfctrl0 r3, 0x90; }}\n\t{{ movh r3, 381; }}\n\t"
          f"{{ mtctrl0 r3, 0x90; }}\n\t.Lflush{n}:\n\t{{ mfctrl0 r2, 0x82; }}\n\t{{ andl r2, r2, 4; }}\n\t{{ cbnz r2, .Lflush{n}; }}\n\t"
          "{ wfe r0, 0; }\n\t{ exit; }")
VECTORS = ["\t.text", "\t.globl _entry", "\t.p2align 4", "_entry:", "\t{ b .Lmain; }"] + [f"\t{{ mov r3, {k}; b .Lexc; }}" for k in range(1, 16)] + \
          [".Lhang:", "\t{ b .Lhang; }"] + HANDLER + [".Lmain:", FP_ENABLE] + ([STAMP_START] if STAMP_START else [])
VECTOR_BUNDLES = 16 + 1 + len([l for l in HANDLER if l.startswith("\t{")])   # bundles before main (46): never executed but on a fault
# directives irrelevant to a bare-metal image; `.section` must stay, or the constant pool would be assembled into `.text`
DROP = (".file", ".ident", ".globl", ".type", ".size")

def build_object(src:str) -> bytes:
  """OpenCL C -> the assembled ELF object of the whole image."""
  with tempfile.TemporaryDirectory() as td:
    pathlib.Path(f"{td}/k.cl").write_text(src)
    _run("btaipucc", [*CC1_ARGS, "-main-file-name", f"{td}/k.cl", "-o", f"{td}/k.s", "-x", "cl", f"{td}/k.cl"])
    body = pathlib.Path(f"{td}/k.s").read_text()
    # every `br ra` becomes the task epilogue below, so the image must be ONE function: a helper the compiler did not inline
    # would end the task at its first return (silently: no fault, no stores after the call). Mark such helpers always_inline.
    if (calls := sorted(set(re.findall(r"\bbl\s+([A-Za-z_]\w*)", body)))):
      raise CompileError(f"ZHOUYI: the kernel calls {calls}; mark them __attribute__((always_inline)) (each `br ra` is a task exit)")
    if not (blks := list(re.finditer(r"\{[^{}]*\bbr\s+ra\b[^{}]*\}", body))): raise CompileError("ZHOUYI: no 'br ra' in the compiled kernel")
    out, prev = [], 0
    for i, m in enumerate(blks):   # only the `br ra` is dropped: its bundle may also hold a real store
      keep = [l for l in m.group(0)[1:-1].splitlines() if l.strip() and not re.fullmatch(r"\s*br\s+ra\s*", l)]
      out += [body[prev:m.start()], ("{\n" + "\n".join(keep) + "\n\t}\n") if keep else "", epilogue(i)]
      prev = m.end()
    body = "\n".join(l for l in ("".join(out) + body[prev:]).splitlines() if not l.strip().startswith(DROP) and ".note.GNU-stack" not in l)
    pathlib.Path(f"{td}/img.s").write_text("\n".join(VECTORS) + "\n" + body + "\n")
    _run("btaipuas", [*AS_ARGS, "-o", f"{td}/img.o", f"{td}/img.s"])
    return pathlib.Path(f"{td}/img.o").read_bytes()

class ZhouyiCompiler(Compiler):
  def __init__(self):
    key = "\0".join([str(BLOB_VERSION), BLOB_HEADER, *CC1_ARGS, *AS_ARGS, *VECTORS, epilogue(0), *DROP])
    super().__init__("zhouyi_" + hashlib.sha256(key.encode()).hexdigest()[:16])
  def compile(self, src:str) -> bytes:
    try: text, rodata = _elf.lift(build_object(src))
    except ZhouyiError as e: raise CompileError(str(e)) from e
    return struct.pack(BLOB_HEADER, len(text), len(rodata)) + text + rodata
  def disassemble(self, lib:bytes):
    for i in range(0, len(lib), 16): print(f"{i:04x}  " + " ".join(f"{x:08x}" for x in struct.unpack_from("<4I", lib, i)))
