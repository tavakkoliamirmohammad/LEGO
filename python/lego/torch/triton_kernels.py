"""
LEGO Triton Kernel Codegen (Layer 1 -- GPU Path)

Generates Triton matmul kernels that use LEGO layout algebra for index
computation. When a tensor has a LEGO layout, the kernel reads elements
using the layout's inverse permutation table (physical->logical remapping).

Falls back to standard Triton matmul when no layout is attached.
"""

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _mm_kernel(
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        A_perm,
        has_a_perm: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k

            # Load A tile
            a_ptrs_flat = offs_m[:, None] * K + k_offs[None, :]
            a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
            if has_a_perm:
                # Layout-aware: remap flat indices through inverse perm table
                safe_ptrs = tl.where(a_mask, a_ptrs_flat, 0)
                a_phys_idx = tl.load(A_perm + safe_ptrs, mask=a_mask, other=0)
                a = tl.load(A + a_phys_idx, mask=a_mask, other=0.0)
            else:
                a = tl.load(A + a_ptrs_flat, mask=a_mask, other=0.0)

            # Load B tile (standard indexing)
            b_ptrs = k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
            b = tl.load(B + b_ptrs, mask=b_mask, other=0.0)

            acc += tl.dot(a, b)

        # Store C
        c_ptrs = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(C + c_ptrs, acc, mask=c_mask)


    def triton_lego_mm(a, b, a_layout=None, b_layout=None):
        """Layout-aware matrix multiply using Triton.

        If a_layout is provided, the kernel reads A using the layout's
        inverse permutation table (physical->logical remapping).
        """
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"Inner dimensions don't match: {K} vs {K2}"

        c = torch.empty(M, N, device=a.device, dtype=a.dtype)

        has_a_perm = a_layout is not None
        if has_a_perm:
            import numpy as np
            from lego.backend.compiler import LayoutCompiler
            base = a_layout._base if hasattr(a_layout, "_base") else a_layout
            compiler = LayoutCompiler(base._layout, base._shape, "i64")
            _, inv = compiler.get_permutation_table()
            a_perm = torch.from_numpy(np.ascontiguousarray(inv)).to(a.device)
        else:
            a_perm = torch.empty(0, dtype=torch.long, device=a.device)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _mm_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            a_perm, has_a_perm,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return c


    def triton_lego_bmm(a, b, a_layout=None, b_layout=None):
        """Layout-aware batched matrix multiply using Triton."""
        B_dim, M, K = a.shape
        _, K2, N = b.shape
        assert K == K2

        c = torch.empty(B_dim, M, N, device=a.device, dtype=a.dtype)
        for i in range(B_dim):
            c[i] = triton_lego_mm(a[i], b[i], a_layout=a_layout, b_layout=b_layout)
        return c
