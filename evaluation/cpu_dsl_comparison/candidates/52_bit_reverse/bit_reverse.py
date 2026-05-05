"""52_bit_reverse: scatter-write to bit-reversed index permutation.

Tier-3 verification.  Reorders ``A[i]`` into ``B[bitrev(i)]`` for
20-bit indices (N = 1<<20).  The bit-reversal is a sequence of
mask+shift operations (delta-swap form).

This is fundamentally a *scatter* with a non-affine index, so the
existing vector pipeline correctly leaves it scalar.  clang -O3 also
emits scalar code (the data dependency through bit shuffles defeats
auto-vectorisation).  We benchmark for *parity verification* — both
sides should run at roughly the same speed since neither vectorises.

Bit-reverse is the canonical pattern for FFT-style stride permutations
and serves as a stress test for the ``LegoVectorize`` Morton path,
which uses the same delta-swap recogniser to detect the *forward*
transform.  Here we exercise the inverse direction to confirm we don't
accidentally vectorise an unsafe scatter.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1 << 20  # 20-bit indices


def _ref(A, B):
    idxs = np.arange(N, dtype=np.uint32)
    j = ((idxs & 0x55555) << 1) | ((idxs >> 1) & 0x55555)
    j = ((j & 0x33333) << 2) | ((j >> 2) & 0x33333)
    j = ((j & 0x0F0F0) << 4) | ((j >> 4) & 0x0F0F0)
    j = ((j & 0x00FF0) << 8) | ((j >> 8) & 0x000FF)
    B[j] = A


@benchmark(
    reference=_ref, n_iters=100, warmup=10, rtol=0.0,
    meta={"N": N, "layout_class": "BitReverse", "prior_verdict": "NEW"},
)
@cpu_kernel
def bit_reverse(A: Buffer[N], B: Buffer[N]):
    for i in range(N):
        # 20-bit reversal via 4 delta-swap stages.
        j = ((i & 0x55555) << 1) | ((i >> 1) & 0x55555)
        j = ((j & 0x33333) << 2) | ((j >> 2) & 0x33333)
        j = ((j & 0x0F0F0) << 4) | ((j >> 4) & 0x0F0F0)
        j = ((j & 0x00FF0) << 8) | ((j >> 8) & 0x000FF)
        B[j] = A[i]


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)
    B = np.zeros(N, dtype=np.float32)
    return A, B


if __name__ == "__main__":
    A, B = _make_args()
    rec = bit_reverse.measure(A, B)
    rec["verified"] = bit_reverse.verify(A, np.zeros(N, dtype=np.float32))
    print(json.dumps(rec, default=str))
