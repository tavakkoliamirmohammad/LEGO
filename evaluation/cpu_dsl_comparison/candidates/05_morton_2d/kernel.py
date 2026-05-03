"""05_morton_2d: read from a Z-Morton encoded 1-D buffer using inline bit ops.

Expected verdict: PARITY or WIN vs gcc on the gather path, since both must
perform irregular gather reads. GCC can't auto-vectorize Morton decode; LEGO
emits vector.gather via the NonAffine path.

The kernel computes a simplified 1-D Morton-style index transform:
  morton(i) = (spread even bits of i) | (spread odd bits of i << 1)

using the standard bit-interleave trick (two-level bit shuffle). This is the
simplest gather pattern that exercises real bitwise ops in the DSL.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 65536        # 256 * 256 flat array (matches the c_baseline's expected N)
TILE = 16


def kernel_scalar(A, B):
    """NumPy reference: 1-D Morton-style index scramble.

    Computes a simplified Morton mapping:
      i → mask low bits to stay in-bounds
    The bit-interleave approach:
      ti  = i & 0x5555  (extract even bits)
      tj  = (i >> 1) & 0x5555  (extract odd bits)
      m   = ti | (tj << 1)   (reinterleave → produces a gather index)
      idx = m & (N - 1)       (clamp to buffer size)
    """
    indices = np.arange(N, dtype=np.int32)
    ti = indices & 0x5555
    tj = (indices >> 1) & 0x5555
    morton = ti | (tj << 1)
    morton = morton & (N - 1)  # mask to in-bounds
    B[:] = A[morton]


@cpu_kernel(grid=(N,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N], B: Buffer[N]):
    for i in tile_range:
        # Real Morton decode using bitwise ops now supported in the DSL.
        ti = i & 0x5555
        tj = (i >> 1) & 0x5555
        morton = ti | (tj << 1)
        morton = morton & (N - 1)
        B[i] = A[morton]
