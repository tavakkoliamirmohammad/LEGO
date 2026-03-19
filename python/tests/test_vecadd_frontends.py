"""End-to-end vecadd tests for all frontends, verified against PyTorch.

Triton: full GPU execution + comparison
Numba CUDA: source generation test (GPU execution requires numba+CUDA)
JAX: source generation test (GPU execution requires jax+jaxlib)
"""

import pytest
import torch
from lego.core import OrderBy, Row
from lego.rewriter import rewrite
from lego.frontends.triton_jit import TritonAdapter


# ── Triton vecadd (full GPU test) ────────────────────────────────────────

class TestTritonVecadd:
    """Full GPU test: LEGO-rewritten Triton vecadd vs PyTorch."""

    @pytest.fixture
    def vecadd_fn(self):
        import triton
        import triton.language as tl
        from lego.frontends.triton_jit import jit as lego_jit

        @lego_jit
        @triton.jit
        def vecadd_kernel(x_ptr, y_ptr, z_ptr, n_elements,
                          BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            L = OrderBy(Row(n_elements)).TileBy(
                [n_elements / BLOCK_SIZE], [BLOCK_SIZE])
            offsets = L[pid, :]
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            z = x + y
            tl.store(z_ptr + offsets, z, mask=mask)

        return vecadd_kernel

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_vecadd_matches_pytorch(self, vecadd_fn):
        import triton
        N = 2**16
        x = torch.randn(N, device='cuda')
        y = torch.randn(N, device='cuda')
        z = torch.empty_like(x)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        vecadd_fn[grid](x, y, z, N, BLOCK_SIZE=BLOCK_SIZE)

        expected = x + y
        assert torch.allclose(z, expected, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_vecadd_multiple_sizes(self, vecadd_fn):
        import triton
        for N in [128, 1024, 8192, 2**16]:
            x = torch.randn(N, device='cuda')
            y = torch.randn(N, device='cuda')
            z = torch.empty_like(x)

            BLOCK_SIZE = 1024
            grid = (triton.cdiv(N, BLOCK_SIZE),)
            vecadd_fn[grid](x, y, z, N, BLOCK_SIZE=BLOCK_SIZE)

            expected = x + y
            assert torch.allclose(z, expected, atol=1e-5), f"Failed for N={N}"


# ── Triton source verification ───────────────────────────────────────────

class TestTritonVecaddSource:
    """Verify the generated Triton source is correct."""

    def test_generated_source(self):
        import triton.language as tl
        from lego.frontends.triton_jit import get_kernel_source

        def vecadd_kernel(x_ptr, y_ptr, z_ptr, n_elements,
                          BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            L = OrderBy(Row(n_elements)).TileBy(
                [n_elements / BLOCK_SIZE], [BLOCK_SIZE])
            offsets = L[pid, :]
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            z = x + y
            tl.store(z_ptr + offsets, z, mask=mask)

        src = get_kernel_source(vecadd_kernel)

        # Must contain Triton-specific tokens
        assert 'tl.arange(0, BLOCK_SIZE)' in src
        assert 'tl.load(' in src
        assert 'tl.store(' in src
        # Layout DSL must be fully eliminated
        assert 'OrderBy' not in src
        assert 'TileBy' not in src


# ── Numba CUDA source verification ──────────────────────────────────────

class TestNumbaCUDAVecaddSource:
    """Verify the generated Numba CUDA source is correct."""

    def test_generated_source(self):
        from lego.frontends.numba_jit import NumbaCUDAAdapter

        def vecadd(x, y, z, N):
            i = 0  # placeholder for cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
            L = OrderBy(Row(N)).TileBy([N])
            idx = L[i]
            if idx < N:
                z[idx] = x[idx] + y[idx]

        src = rewrite(vecadd, NumbaCUDAAdapter(), return_source=True)

        # Must NOT contain Triton tokens
        assert 'tl.' not in src
        # Layout DSL must be fully eliminated
        assert 'OrderBy' not in src
        assert 'TileBy' not in src
        # idx should simplify to i
        assert 'idx = i' in src

    def test_generated_source_parseable(self):
        """The generated source must be valid Python."""
        from lego.frontends.numba_jit import NumbaCUDAAdapter
        import ast

        def vecadd(x, y, z, N):
            i = 0
            L = OrderBy(Row(N)).TileBy([N])
            idx = L[i]
            if idx < N:
                z[idx] = x[idx] + y[idx]

        src = rewrite(vecadd, NumbaCUDAAdapter(), return_source=True)
        # Should parse without errors
        ast.parse(src)


# ── JAX source verification ─────────────────────────────────────────────

class TestJAXVecaddSource:
    """Verify the generated JAX source is correct."""

    def test_generated_source(self):
        from lego.frontends.jax_jit import JAXAdapter

        def vecadd(x, y, N):
            L = OrderBy(Row(N)).TileBy([N])
            offs = L[:]
            return x + y

        src = rewrite(vecadd, JAXAdapter(), return_source=True)

        # Must contain JAX-specific tokens
        assert 'jnp.arange' in src
        # Must NOT contain Triton tokens
        assert 'tl.' not in src
        # Layout DSL must be fully eliminated
        assert 'OrderBy' not in src
        assert 'TileBy' not in src

    def test_generated_source_parseable(self):
        """The generated source must be valid Python."""
        from lego.frontends.jax_jit import JAXAdapter
        import ast

        def vecadd(x, y, N):
            L = OrderBy(Row(N)).TileBy([N])
            offs = L[:]
            return x + y

        src = rewrite(vecadd, JAXAdapter(), return_source=True)
        ast.parse(src)
