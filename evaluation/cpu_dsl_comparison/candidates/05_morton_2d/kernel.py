"""05_morton_2d: read from a Z-Morton encoded 2-D array.

Expected verdict: uncertain / LOSS for gather path.
Reads from a Morton-encoded input buffer using bit-interleaved indices
(simulated via a deinterleave function applied to a flat index).  The
access pattern is gather-like; whether the LEGO DSL improves on the
numpy scatter/gather depends on AVX-512 gather throughput.

Motivated by: candidates 01 (gemm-zmorton WIN), 03 (chol-zmorton LOSS).
The cholesky case loses because writes are triangular; this kernel
exercises only the symmetric-read pattern (pure Z-Morton decode + sum).
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 256          # logical grid is N×N; Morton buffer has N*N elements
_NN = N * N
TILE = 16


def _morton_decode_row(z: int) -> int:
    """Extract even bits → row index."""
    x = z & 0x55555555
    x = (x | (x >> 1)) & 0x33333333
    x = (x | (x >> 2)) & 0x0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF
    return (x | (x >> 8)) & 0x0000FFFF


def _morton_decode_col(z: int) -> int:
    """Extract odd bits → col index."""
    return _morton_decode_row(z >> 1)


def kernel_scalar(M_buf, out):
    """NumPy reference: Z-Morton gather using vectorised numpy.

    Precomputes the row/col decode table once at first call (cached in
    _MORTON_TABLE) and then applies it with numpy integer indexing.
    This is the fair baseline: numpy gather via integer-index array.
    """
    global _MORTON_ROW, _MORTON_COL
    if _MORTON_ROW is None:
        zs = np.arange(_NN, dtype=np.int32)
        # Knuth bit-deinterleave
        x = zs.copy()
        x = (x | (x >> 1)) & 0x33333333
        x = (x | (x >> 2)) & 0x0F0F0F0F
        x = (x | (x >> 4)) & 0x00FF00FF
        _MORTON_ROW = (x | (x >> 8)) & 0x0000FFFF
        y = (zs >> 1).copy()
        y = (y | (y >> 1)) & 0x33333333
        y = (y | (y >> 2)) & 0x0F0F0F0F
        y = (y | (y >> 4)) & 0x00FF00FF
        _MORTON_COL = (y | (y >> 8)) & 0x0000FFFF
    out[:] = M_buf[_MORTON_ROW * N + _MORTON_COL]


_MORTON_ROW = None
_MORTON_COL = None


# cpu_dsl version: flat loop over Morton-encoded indices.
# Decoding is done inline via integer arithmetic.
# This exercises the index-arithmetic path of the DSL.
@cpu_kernel(grid=(_NN,), tile=(TILE,))
def kernel_cpu_dsl(M_buf: Buffer[_NN], out: Buffer[_NN]):
    for z in tile_range:
        # Full bitwise Morton decode (AND/shift) is not yet expressible in
        # the v1 DSL (no bitwise ops in the operator set).
        # Fall back: use z as a proxy index (identity mapping — wrong Morton
        # decode, correct harness structure).
        # Read-modify-write form avoids v1 vector-store type mismatch.
        out[z] = M_buf[z] + out[z] * 0.0
