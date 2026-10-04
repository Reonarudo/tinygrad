"""ELF32 lift: extract `.text`, build the `cp`-relative constant pool and apply its relocations.

`objcopy` does not support arch 0x29a, so section headers are parsed here. clang places constants it
cannot encode inline (e.g. bool->float casts) in a pool addressed off `cp`, so the pool must be loaded.
"""
import struct
from typing import NamedTuple

from . import ZhouyiError, round_up

SHT_PROGBITS, SHF_WRITE, SHF_ALLOC, SHF_EXECINSTR = 1, 0x1, 0x2, 0x4
# The only constant-pool relocations the zhouyi clang target emits. Each writes a symbol's byte offset
# from `cp` (section-relative, not absolute) into the low/high half of a `mov`/`movh` pair.
R_CP_REL_HI, R_CP_REL_LO = 6, 7
OPC_MOV, OPC_MOVH = 6, 7                 # bits [23:21] of the encoded instruction word
OPC_SHIFT, IMM_SHIFT, IMM_MASK = 21, 5, 0xffff   # immediate is a 16-bit field at bit 5


class Sec(NamedTuple):
  """One ELF32 section header, with its name resolved."""
  name:str
  typ:int
  flags:int
  addr:int
  off:int
  size:int
  link:int
  info:int
  align:int
  entsize:int


def sections(obj:bytes) -> list[Sec]:
  """Section headers in index order (`st_shndx` indexes this list)."""
  shoff, ent, num, sx = (struct.unpack_from("<I", obj, 0x20)[0], *struct.unpack_from("<HHH", obj, 0x2E))
  def sh(i): return struct.unpack_from("<IIIIIIIIII", obj, shoff + i*ent)
  stroff = sh(sx)[4]
  return [Sec(obj[stroff+f[0]:obj.index(b"\0", stroff+f[0])].decode(), *f[1:]) for f in (sh(i) for i in range(num))]


def find(secs:list[Sec], name:str) -> Sec|None: return next((s for s in secs if s.name == name), None)


def symbols(obj:bytes, secs:list[Sec]) -> list[tuple[str, int, int]]:
  """(name, st_value, st_shndx) per symbol. st_value is relative to the symbol's section."""
  st, strt = find(secs, ".symtab"), find(secs, ".strtab")
  if st is None or strt is None: return []
  out = []
  for k in range(st.size//16):
    nmo, val, _sz, _info, _other, shndx = struct.unpack_from("<IIIBBH", obj, st.off+k*16)
    out.append((obj[strt.off+nmo:obj.index(b"\0", strt.off+nmo)].decode(), val, shndx))
  return out


def constant_pool(obj:bytes, secs:list[Sec]) -> tuple[bytes, dict[int, int]]:
  """Concatenate every read-only allocatable data section into the pool `cp` points at.

  Returns (pool, {section index: offset}). Layout follows section-header order at each section's
  alignment, matching the vendor linker. SHF_WRITE sections are excluded (tasks would share them);
  `relocate` rejects cp-relative references to them."""
  pool, base = bytearray(), {}
  for i, s in enumerate(secs):
    if s.typ != SHT_PROGBITS or not (s.flags & SHF_ALLOC) or (s.flags & (SHF_EXECINSTR|SHF_WRITE)) or s.size == 0: continue
    pool += b"\0" * (round_up(len(pool), max(s.align, 1)) - len(pool))
    base[i] = len(pool)
    pool += obj[s.off:s.off+s.size]
  return bytes(pool), base


def relocate(obj:bytes, secs:list[Sec], text:bytes, base:dict[int, int], pool_size:int) -> bytes:
  """Apply the cp-relative relocations into `.text`.

  Values are pool offsets, so the patched text is load-address independent and cacheable."""
  syms, out = symbols(obj, secs), bytearray(text)
  if (r := find(secs, ".rela.text")) is None: return bytes(out)
  for j in range(r.size//12):
    # The addend is Elf32_Sword (signed).
    r_off, r_info, r_add = struct.unpack_from("<IIi", obj, r.off+j*12)
    typ, si = r_info & 0xff, r_info >> 8
    if typ not in (R_CP_REL_HI, R_CP_REL_LO):
      raise ZhouyiError(f"unsupported relocation type={typ} at .text+{r_off:#x} (only cp_rel_hi/lo are understood)")
    sname, sval, shndx = syms[si]
    if shndx not in base:
      raise ZhouyiError(f"cp-relative relocation against {sname!r} in section index {shndx}, which is not in the "
                      "constant pool (not read-only allocatable data) — the constant would be read from the "
                      "wrong address")
    val = base[shndx] + sval + r_add
    if not 0 <= val <= pool_size:
      raise ZhouyiError(f"cp-relative value {val} for {sname!r} falls outside the {pool_size}-byte constant pool")
    w = struct.unpack_from("<I", out, r_off)[0]
    want = OPC_MOV if typ == R_CP_REL_LO else OPC_MOVH
    if (opc := (w >> OPC_SHIFT) & 0x7) != want:
      raise ZhouyiError(f"relocation type={typ} at .text+{r_off:#x} targets opcode {opc}, expected {want}")
    if (w >> IMM_SHIFT) & IMM_MASK:
      raise ZhouyiError(f"relocation site .text+{r_off:#x} is not a zero placeholder — the assembler already "
                      "resolved it, so patching would OR two values together")
    imm = (val if typ == R_CP_REL_LO else val >> 16) & IMM_MASK
    struct.pack_into("<I", out, r_off, w | (imm << IMM_SHIFT))
  return bytes(out)


def lift(obj:bytes) -> tuple[bytes, bytes]:
  """Object bytes -> (relocated .text, constant pool). The pool is loaded at `cp+0`."""
  secs = sections(obj)
  if (t := find(secs, ".text")) is None: raise ZhouyiError("no .text in object")
  pool, base = constant_pool(obj, secs)
  return relocate(obj, secs, obj[t.off:t.off+t.size], base, len(pool)), pool
