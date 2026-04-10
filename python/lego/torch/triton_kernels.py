"""
LEGO Triton Kernel Codegen (Layer 1 -- GPU Path)

Uses ``@lego.jit`` to rewrite standard Triton kernels with LEGO layout
algebra index expressions. The layout is expressed as::

    L_A = OrderBy(Row(M, K)).TileBy([M/BM, K/BK], [BM, BK])

and indexing via ``L_A[block_m, block_k, :, :]`` is compiled by LEGO's
rewriter into pure arithmetic Triton code -- no permutation tables, no
extra memory indirection, just O(1) index math per element.

This is the same mechanism used by all Triton examples in
``python/examples/triton/``.
"""

import torch

try:
    import triton
    import triton.language as tl
    from lego.core import Row, Col, OrderBy
    from lego.frontends.triton_jit import jit as lego_jit
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    @lego_jit
    @triton.jit
    def _lego_mm_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # LEGO layout algebra: row-major matrices tiled into blocks.
        # The rewriter compiles L_A[pid_m, k, :, :] into arithmetic
        # Triton index expressions -- no tables, no indirection.
        L_A = OrderBy(Row(M, K)).TileBy([M / BM, K / BK], [BM, BK])
        L_B = OrderBy(Row(K, N)).TileBy([K / BK, N / BN], [BK, BN])
        L_C = OrderBy(Row(M, N)).TileBy([M / BM, N / BN], [BM, BN])

        acc = tl.zeros((BM, BN), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BK)):
            a = tl.load(a_ptr + L_A[pid_m, k, :, :])
            b = tl.load(b_ptr + L_B[k, pid_n, :, :])
            acc = tl.dot(a, b, acc)

        tl.store(c_ptr + L_C[pid_m, pid_n, :, :], acc)


    def triton_lego_mm(a, b, a_layout=None, b_layout=None):
        """Layout-aware matrix multiply using LEGO-generated Triton kernel.

        The kernel uses LEGO layout algebra for index computation.
        No permutation tables -- pure arithmetic codegen.
        """
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"Inner dimensions don't match: {K} vs {K2}"

        c = torch.empty(M, N, device=a.device, dtype=torch.float32)

        BM, BN, BK = 64, 64, 32
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

        _lego_mm_kernel[grid](
            a, b, c,
            M, N, K,
            BM=BM, BN=BN, BK=BK,
        )
        return c.to(a.dtype)


    def triton_lego_bmm(a, b, a_layout=None, b_layout=None):
        """Layout-aware batched matrix multiply using LEGO Triton kernel."""
        B_dim, M, K = a.shape
        _, K2, N = b.shape
        assert K == K2

        c = torch.empty(B_dim, M, N, device=a.device, dtype=a.dtype)
        for i in range(B_dim):
            c[i] = triton_lego_mm(a[i], b[i], a_layout=a_layout, b_layout=b_layout)
        return c
