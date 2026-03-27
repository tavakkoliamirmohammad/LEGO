"""cuTile matrix multiplication with LEGO-computed block swizzling.

Demonstrates using LEGO's layout algebra to replace the manual 2D swizzle
scheduling logic.  The ``L_pid.inv(bid)`` call computes (bidx, bidy)
tile coordinates from a 1D block ID, with L2-cache-friendly group ordering.

Without LEGO (manual cuTile):
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m

With LEGO:
    L_pid = OrderBy(Col(Max(num_pid_m // GM, 1), 1),
                    Col(Min(num_pid_m, GM), num_pid_n)).TileBy(
                        [num_pid_m, num_pid_n])
    bidx, bidy = L_pid.inv(bid)

Usage:
    python matmul.py                  # source verification + GPU test
    LEGO_SAVE_KERNEL=1 python matmul.py  # also saves generated kernel
"""

import math
import torch
import cuda.tile as ct
from sympy import Max, Min

from lego.core import OrderBy, Col
from lego.frontends.cutile_jit import cutile_jit, get_cutile_kernel_source

ConstInt = ct.Constant[int]


# ---------------------------------------------------------------------------
# Reference cuTile kernel (no LEGO) — manual swizzle
# ---------------------------------------------------------------------------

@ct.kernel
def matmul_cutile(A, B, C,
                  tm: ConstInt, tn: ConstInt, tk: ConstInt):
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bid = ct.bid(0)

    # Manual swizzle
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bidx = first_bid_m + (bid % group_size_m)
    bidy = (bid % num_bid_in_group) // group_size_m

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    for k in range(num_tiles_k):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk),
                    padding_mode=zero_pad).astype(dtype)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn),
                    padding_mode=zero_pad).astype(dtype)
        accumulator = ct.mma(a, b, accumulator)
    accumulator = ct.astype(accumulator, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=accumulator)


# ---------------------------------------------------------------------------
# LEGO cuTile kernel — layout algebra replaces manual swizzle
# ---------------------------------------------------------------------------

@cutile_jit
@ct.kernel
def matmul_lego(A, B, C,
                tm: ConstInt, tn: ConstInt, tk: ConstInt):
    GM = 8
    M = A.shape[0]
    N = B.shape[1]
    bid = ct.bid(0)

    # LEGO swizzle: replaces ~8 lines of manual index math
    num_pid_m = ct.cdiv(M, tm)
    num_pid_n = ct.cdiv(N, tn)
    L_pid = OrderBy(Col(Max(num_pid_m // GM, 1), 1),
                    Col(Min(num_pid_m, GM), num_pid_n)).TileBy(
                        [num_pid_m, num_pid_n])
    bidx, bidy = L_pid.inv(bid)

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    for k in range(num_tiles_k):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk),
                    padding_mode=zero_pad).astype(dtype)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn),
                    padding_mode=zero_pad).astype(dtype)
        accumulator = ct.mma(a, b, accumulator)
    accumulator = ct.astype(accumulator, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=accumulator)


# ---------------------------------------------------------------------------
# Verification & source inspection
# ---------------------------------------------------------------------------

def verify_source():
    """Check that LEGO eliminates layout expressions."""
    def matmul_raw(A, B, C, tm, tn, tk):
        GM = 8
        M = A.shape[0]
        N = B.shape[1]
        bid = ct.bid(0)
        num_pid_m = ct.cdiv(M, tm)
        num_pid_n = ct.cdiv(N, tn)
        L_pid = OrderBy(Col(Max(num_pid_m // GM, 1), 1),
                        Col(Min(num_pid_m, GM), num_pid_n)).TileBy(
                            [num_pid_m, num_pid_n])
        bidx, bidy = L_pid.inv(bid)

    src = get_cutile_kernel_source(matmul_raw)
    print("=== LEGO-generated cuTile matmul swizzle ===")
    print(src)
    print()

    assert 'L_pid' not in src, "Layout variable not eliminated"
    assert 'OrderBy' not in src, "OrderBy not eliminated"
    assert 'Col(' not in src, "Col not eliminated"
    assert 'bidx =' in src, "bidx not computed"
    assert 'bidy =' in src, "bidy not computed"
    print("Source verification passed.\n")


def verify_gpu():
    """Run both kernels on GPU and compare against PyTorch."""
    M, N, K = 512, 512, 512
    tm, tn, tk = 64, 64, 32
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    expected = torch.matmul(A, B)

    stream = torch.cuda.current_stream()
    grid = (math.ceil(M / tm) * math.ceil(N / tn), 1, 1)

    # Reference cuTile kernel
    C_ref = torch.empty(M, N, device='cuda', dtype=torch.float16)
    ct.launch(stream, grid, matmul_cutile, (A, B, C_ref, tm, tn, tk))
    assert torch.allclose(C_ref, expected, atol=1e-1, rtol=1e-1), \
        "Reference cuTile matmul mismatch!"
    print("Reference cuTile matmul: PASS")

    # LEGO cuTile kernel
    C_lego = torch.empty(M, N, device='cuda', dtype=torch.float16)
    ct.launch(stream, grid, matmul_lego, (A, B, C_lego, tm, tn, tk))
    assert torch.allclose(C_lego, expected, atol=1e-1, rtol=1e-1), \
        "LEGO cuTile matmul mismatch!"
    print("LEGO cuTile matmul:      PASS")

    # Both kernels produce the same result
    assert torch.allclose(C_ref, C_lego, atol=1e-5), \
        "Reference and LEGO matmul results differ!"
    print("Ref vs LEGO match:       PASS\n")


if __name__ == "__main__":
    verify_source()
    if torch.cuda.is_available():
        verify_gpu()
    else:
        print("No GPU available, skipping GPU verification.")
