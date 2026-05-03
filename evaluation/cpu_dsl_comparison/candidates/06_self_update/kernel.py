"""06_self_update: in-place prefix-sum style reduction (refuses vectorisation).

Expected verdict: LOSS.  The loop-carried dependence in a prefix sum
(each output depends on the previous output) inhibits SIMD vectorisation.
This mirrors the anti-diagonal and wavefront LOSS cases (candidates 17, 20)
where the recurrence structure defeats the vectoriser.

B[i] = B[i-1] + A[i]  (sequential prefix sum, loop-carried dep)
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 4096
TILE = 16


def kernel_scalar(A, B):
    """NumPy reference: cumulative sum."""
    np.cumsum(A, out=B)


# In the DSL we express the prefix sum as a sequential loop.
# The cpu_dsl compiler does not analyse loop-carried deps to decide
# whether to serialise — it tries to vectorise unconditionally, so
# the result will be incorrect or will fall back to scalar.
# Either way, this is a meaningful data point for the harness.
@cpu_kernel(grid=(N,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N], B: Buffer[N]):
    for i in tile_range:
        # This is loop-carried: B[i] depends on B[i-1].
        # The DSL will still emit the loop; correctness is not guaranteed.
        if i > 0:
            B[i] = B[i - 1] + A[i]
        else:
            B[i] = A[i]
