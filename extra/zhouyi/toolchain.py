"""The installed AIPU compiler toolchain as an oracle for our encoder (never at runtime).

`libaiputoolchain.so` (located and loaded by `compiler_zhouyi.toolchain_lib`: `ZHOUYI_TOOLCHAIN_DIR` or the vendor's install
directory) exports `btaipucc` (clang), `btaipuas` (llvm-mc) and `btaipuld` (lld) as `int(argc, argv)` entry points, driven
in-process. Codegen (`ZhouyiCompiler`) uses the same library; here `asm()` assembles text we also encode ourselves, so
`image.build_encoded` can byte-diff whole images.
"""
import ctypes, pathlib, struct, tempfile

from tinygrad.runtime.support.zhouyi import ZhouyiError
from tinygrad.runtime.support.compiler_zhouyi import AS_ARGS, CC1_ARGS, toolchain_lib   # noqa: F401  (CC1_ARGS / AS_ARGS re-exported)


class Toolchain:
  """btaipucc / btaipuas / btaipuld, in-process."""
  def __init__(self): self.lib, self.ncalls = toolchain_lib(), 0
  def __call__(self, tool: str, args: list[str]):
    self.ncalls += 1
    argv = [tool] + args
    arr = (ctypes.c_char_p * (len(argv)+1))(*[a.encode() for a in argv], None)
    if (rc := getattr(self.lib, tool)(len(argv), arr)) != 0: raise ZhouyiError(f"{tool} returned {rc}")


_TC: Toolchain|None = None


def toolchain() -> Toolchain:
  """Process-wide singleton."""
  global _TC
  if _TC is None: _TC = Toolchain()
  return _TC


# ***************** the oracle *****************
def text_of(obj: bytes) -> bytes:
  """`.text` of an ELF32 object. Independent of `elf.py` so the oracle does not depend on the code it checks."""
  shoff = struct.unpack_from("<I", obj, 0x20)[0]
  ent, num, sx = struct.unpack_from("<HHH", obj, 0x2E)
  sh = lambda i: struct.unpack_from("<IIIIII", obj, shoff + i*ent)
  stroff = sh(sx)[4]
  for i in range(num):
    nm, _typ, _fl, _addr, off, size = sh(i)
    if obj[stroff+nm:obj.index(b"\0", stroff+nm)] == b".text": return obj[off:off+size]
  return b""


def asm(body: str, cpu: str = "X2_1204") -> bytes:
  """ORACLE. Assembly text -> the vendor assembler's `.text` bytes."""
  with tempfile.TemporaryDirectory() as td:
    s, o = f"{td}/a.s", f"{td}/a.o"
    pathlib.Path(s).write_text(body)
    toolchain()("btaipuas", ["-triple", "aipux2--", f"-mcpu={cpu}", "-filetype=obj", "-o", o, s])
    return text_of(pathlib.Path(o).read_bytes())


def asm_each(instrs: list[str], cpu: str = "X2_1204", pre: str = "") -> list[tuple[int, ...]]:
  """ORACLE. One instruction per bundle -> one 4-word tuple per instruction."""
  body = ".text\n.globl f\nf:\n" + pre + "".join("\t{\n\t\t%s\n\t}\n" % i for i in instrs)
  raw = asm(body, cpu)
  return [struct.unpack_from("<4I", raw, i) for i in range(0, len(raw), 16)]
