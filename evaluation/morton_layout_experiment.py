#!/usr/bin/env python3
"""Morton-vs-Row layout experiment on 2D Jacobi 5-point stencil.

Tests the hypothesis: changing one line at the top (Row → ZCurve) gives
LEGO a substantial speedup on a 2D-spatial-locality kernel because Morton
makes neighbor accesses share cache lines.

Usage:  python morton_layout_experiment.py [N]
        N must be a power of 2; default 1024.
"""

import json
import math
import sys
import numpy as np

from lego.backend.cpu_dsl import cpu_kernel, Buffer
from lego.core import Row
# Use the SLOW per-bit-loop ZCurve to test the bit-spread recognizer pass.
# If the recognizer is firing, this should produce code as fast as the
# hand-rolled Morton2DFast (bit-magic) variant.
from lego.frontends.python_mlir import ZCurve
def Morton2D(N): return ZCurve((N, N))._layout


# ----------------------------------------------------------------------
# Grid size — must be power of 2 for clean Morton encoding.
# ----------------------------------------------------------------------
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
NN = N * N

LAYOUT_ROW    = Row(N, N)
LAYOUT_MORTON = Morton2D(N)


# ----------------------------------------------------------------------
# Two kernels — same body, different layouts at the top.
# ----------------------------------------------------------------------
@cpu_kernel
def jacobi_row(A: Buffer(LAYOUT_ROW, NN), B: Buffer(LAYOUT_ROW, NN)):
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            B[i, j] = 0.25 * (A[i - 1, j] + A[i + 1, j]
                              + A[i, j - 1] + A[i, j + 1])


@cpu_kernel
def jacobi_morton(A: Buffer(LAYOUT_MORTON, NN), B: Buffer(LAYOUT_MORTON, NN)):
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            B[i, j] = 0.25 * (A[i - 1, j] + A[i + 1, j]
                              + A[i, j - 1] + A[i, j + 1])


# ----------------------------------------------------------------------
# Morton encoding (Python — for data permutation only)
# ----------------------------------------------------------------------
def morton_encode(i, j, nbits):
    z = 0
    for k in range(nbits):
        z |= ((i >> k) & 1) << (2 * k)
        z |= ((j >> k) & 1) << (2 * k + 1)
    return z


def make_morton_perm(N):
    nbits = int(math.log2(N))
    perm = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            perm[i, j] = morton_encode(i, j, nbits)
    return perm


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print(f"\n{'='*70}")
    print(f"  N = {N}  ({N}×{N} grid, {NN} cells, {NN*4/1e6:.1f} MB per array)")
    print(f"  3-row working set: {3*N*4/1024:.0f} KB  (L1=32KB, L2=1MB on Xeon 6330)")
    print(f"{'='*70}")

    rng = np.random.default_rng(42)
    A_logical = rng.standard_normal((N, N)).astype(np.float32)

    # Row-major: flat = A_logical.reshape(NN)
    A_row_flat = A_logical.reshape(NN).copy()
    B_row_flat = np.zeros(NN, dtype=np.float32)

    # Morton: A_morton[morton(i,j)] = A_logical[i,j]
    perm = make_morton_perm(N)
    A_morton_flat = np.zeros(NN, dtype=np.float32)
    A_morton_flat[perm.reshape(NN)] = A_logical.reshape(NN)
    B_morton_flat = np.zeros(NN, dtype=np.float32)

    # Sanity check on the permutation
    for (i, j) in [(0, 0), (3, 5), (N-1, N-1), (N//2, N//2)]:
        assert math.isclose(
            float(A_morton_flat[perm[i, j]]),
            float(A_logical[i, j]),
            rel_tol=0,
        ), f"perm sanity failed at ({i},{j})"

    # Scale measurement reps with N so each LEGO kernel call burns ≈ <30s
    # of wall time at large N (where each iter already takes 100s of ms).
    if N <= 2048:
        nit_row, nwu_row, nit_mor, nwu_mor = 50, 5, 50, 5
    elif N <= 4096:
        nit_row, nwu_row, nit_mor, nwu_mor = 50, 5, 20, 3
    elif N <= 8192:
        nit_row, nwu_row, nit_mor, nwu_mor = 20, 3, 5, 2
    else:  # 16384 and above
        nit_row, nwu_row, nit_mor, nwu_mor = 5, 2, 3, 1

    print()
    print(f"  (using row=({nit_row}, {nwu_row}), morton=({nit_mor}, {nwu_mor}))")
    print("  LEGO Row, vector ...")
    t_row_vec = jacobi_row.bench_self_timed(
        A_row_flat, B_row_flat, n_iters=nit_row, n_warmup=nwu_row, target="x86")

    print("  LEGO Row, scalar ...")
    t_row_sca = jacobi_row.bench_self_timed(
        A_row_flat, B_row_flat, n_iters=nit_row, n_warmup=nwu_row, target="scalar")

    t_morton_vec = float("nan")
    try:
        print("  LEGO Morton, vector ...")
        t_morton_vec = jacobi_morton.bench_self_timed(
            A_morton_flat, B_morton_flat, n_iters=nit_mor, n_warmup=nwu_mor, target="x86")
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {str(e).splitlines()[0]}")

    t_morton_sca = float("nan")
    try:
        print("  LEGO Morton, scalar ...")
        t_morton_sca = jacobi_morton.bench_self_timed(
            A_morton_flat, B_morton_flat, n_iters=nit_mor, n_warmup=nwu_mor, target="scalar")
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {str(e).splitlines()[0]}")

    print()
    print(f"  Row    vec : {t_row_vec:>8.4f} ms")
    print(f"  Row    scl : {t_row_sca:>8.4f} ms   (vec_iso = {t_row_sca/t_row_vec:.2f}x)")
    print(f"  Morton vec : {t_morton_vec:>8.4f} ms"
          + (f"   (Row/Morton = {t_row_vec/t_morton_vec:.2f}x)" if not math.isnan(t_morton_vec) else "   FAILED"))
    print(f"  Morton scl : {t_morton_sca:>8.4f} ms"
          + (f"   (Row scl / Morton scl = {t_row_sca/t_morton_sca:.2f}x)" if not math.isnan(t_morton_sca) else "   FAILED"))

    # Backwards-compat aliases for the verification block:
    t_row, t_morton = t_row_vec, (t_morton_vec if not math.isnan(t_morton_vec) else t_morton_sca)

    # Correctness: build numpy reference, compare both outputs
    B_ref = np.zeros((N, N), dtype=np.float32)
    A_ref_2d = A_logical
    B_ref[1:-1, 1:-1] = 0.25 * (
        A_ref_2d[:-2, 1:-1] + A_ref_2d[2:, 1:-1]
        + A_ref_2d[1:-1, :-2] + A_ref_2d[1:-1, 2:]
    )

    B_row_2d = B_row_flat.reshape(N, N)
    row_ok = np.allclose(B_row_2d, B_ref, rtol=1e-4, atol=1e-5)

    # Un-permute Morton output before comparing
    B_morton_logical = B_morton_flat[perm.reshape(NN)].reshape(N, N)
    morton_ok = np.allclose(B_morton_logical, B_ref, rtol=1e-4, atol=1e-5)

    print(f"  Row correct?    {row_ok}")
    print(f"  Morton correct? {morton_ok}")

    print()
    if t_morton < t_row:
        print(f"  Morton wins by {t_row/t_morton:.2f}x ({(1 - t_morton/t_row)*100:.1f}% faster)")
    else:
        print(f"  Row wins by {t_morton/t_row:.2f}x ({(1 - t_row/t_morton)*100:.1f}% faster)")

    return {
        "N": N,
        "lego_row_vec_ms":    round(t_row_vec, 6),
        "lego_row_scl_ms":    round(t_row_sca, 6),
        "lego_morton_vec_ms": round(t_morton_vec, 6) if not math.isnan(t_morton_vec) else None,
        "lego_morton_scl_ms": round(t_morton_sca, 6) if not math.isnan(t_morton_sca) else None,
        "row_correct":        bool(row_ok),
        "morton_correct":     bool(morton_ok),
    }


if __name__ == "__main__":
    result = main()
    print()
    print("  RESULT JSON:", json.dumps(result))
