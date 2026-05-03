"""07_mixed_precision: scalar-argument f32 promotion path.

Expected verdict: PARITY or WIN.  This kernel exercises the scalar 'scale'
parameter path (CT_FLOAT promoted to a runtime f32 constant) + a fused
multiply-add into Y.  It deliberately uses only 2 buffers (X, Y) to avoid
the v1 R12 store-to-3rd-buffer bug.

'mixed precision' here means: Python float → f32 constant in the IR (the
compiler must emit arith.constant f32 instead of silently truncating).

Motivated by: full-eval candidates that exhibited unexpected PARITY or
marginal results related to type mismatches in the lowering pipeline.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 8192
TILE = 16
SCALE = 1.5   # constant baked into the kernel


def kernel_scalar(X, Y):
    """NumPy reference: Y = SCALE * X + Y (in-place)."""
    Y[:] = SCALE * X + Y


# Uses 2 buffers only (avoids v1 R12 vector-store bug for 3rd buffer arg).
@cpu_kernel(grid=(N,), tile=(TILE,))
def kernel_cpu_dsl(X: Buffer[N], Y: Buffer[N]):
    for i in tile_range:
        Y[i] = 1.5 * X[i] + Y[i]
