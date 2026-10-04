"""Zhouyi V3 NPU support layer: everything that talks to the NPU without tinygrad's scheduler.

`dev.py` (the raw `/dev/aipu` ioctl submit path and the TCB chain), `launch.py` (the buffers and chains of one launch, the
compiled-blob format), `elf.py` (the ELF lift of the compiler's output), `hangid.py` (opt-in hung-job reports).

Invariant: nothing under this package imports tinygrad's scheduler or device classes (only `tinygrad.runtime.autogen` for
the KMD bindings). Errors are raised as `ZhouyiError`; `ops_zhouyi.py` translates them where needed.
"""


class ZhouyiError(Exception):
  """Anything this layer refuses to do. `ops_zhouyi.py` re-raises it as `CompileError` where needed."""


def round_up(x: int, a: int) -> int:
  """Round `x` up to a multiple of `a`."""
  return (x + a - 1) // a * a


PAGE = 4096


def mapped_extent(nbytes: int) -> int:
  """Bytes `REQ_BUF` allocates and `mmap` maps for a request of `nbytes` (page-rounded).

  `munmap`/`FREE_BUF` must be given this value, not the logical size."""
  return round_up(nbytes, PAGE)
