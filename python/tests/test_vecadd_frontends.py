"""End-to-end vecadd tests for all frontends, verified against PyTorch.

Skipped entirely when GPU dependencies (triton, numba, jax) are not installed.
"""

import functools
import math

import numpy as np
import pytest
import torch

try:
    import triton
    import triton.language as tl
    import jax
    import jax.numpy as jnp
    from numba import cuda
except ImportError:
    pytest.skip("GPU dependencies (triton/jax/numba) not available", allow_module_level=True)

from lego.core import OrderBy, Row
from lego.rewriter import rewrite
from lego.frontends.triton_jit import TritonAdapter, jit as triton_lego_jit
from lego.frontends.numba_jit import jit as numba_lego_jit
from lego.frontends.jax_jit import jit as jax_lego_jit


# ── Triton vecadd (full GPU test) ────────────────────────────────────────

@triton_lego_jit
@triton.jit
def _vecadd_kernel(x_ptr, y_ptr, z_ptr, n_elements,
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


class TestTritonVecadd:
    """Full GPU test: LEGO-rewritten Triton vecadd vs PyTorch."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_vecadd_matches_pytorch(self):
        N = 2**16
        x = torch.randn(N, device='cuda')
        y = torch.randn(N, device='cuda')
        z = torch.empty_like(x)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        _vecadd_kernel[grid](x, y, z, N, BLOCK_SIZE=BLOCK_SIZE)

        expected = x + y
        assert torch.allclose(z, expected, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_vecadd_multiple_sizes(self):
        for N in [128, 1024, 8192, 2**16]:
            x = torch.randn(N, device='cuda')
            y = torch.randn(N, device='cuda')
            z = torch.empty_like(x)

            BLOCK_SIZE = 1024
            grid = (triton.cdiv(N, BLOCK_SIZE),)
            _vecadd_kernel[grid](x, y, z, N, BLOCK_SIZE=BLOCK_SIZE)

            expected = x + y
            assert torch.allclose(z, expected, atol=1e-5), f"Failed for N={N}"


# ── Triton source verification ───────────────────────────────────────────

class TestTritonVecaddSource:
    """Verify the generated Triton source is correct."""

    def test_generated_source(self):
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


# ── Numba CUDA vecadd (full GPU test) ────────────────────────────────────

@numba_lego_jit
@cuda.jit
def _numba_vecadd_kernel(x, y, z, N):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    L = OrderBy(Row(N)).TileBy([N])
    idx = L[i]
    if idx < N:
        z[idx] = x[idx] + y[idx]


class TestNumbaCUDAVecadd:
    """Full GPU test: LEGO-rewritten Numba CUDA vecadd vs PyTorch."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_vecadd_matches_pytorch(self):
        N = 2**16
        x_torch = torch.randn(N)
        y_torch = torch.randn(N)
        expected = (x_torch + y_torch).numpy()

        x_np = x_torch.numpy()
        y_np = y_torch.numpy()
        d_x = cuda.to_device(x_np)
        d_y = cuda.to_device(y_np)
        d_z = cuda.device_array(N, dtype=np.float32)

        threads = 256
        blocks = math.ceil(N / threads)
        _numba_vecadd_kernel[blocks, threads](d_x, d_y, d_z, N)
        z_np = d_z.copy_to_host()

        assert np.allclose(z_np, expected, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_vecadd_multiple_sizes(self):
        for N in [128, 1024, 8192, 2**16]:
            x_np = np.random.randn(N).astype(np.float32)
            y_np = np.random.randn(N).astype(np.float32)
            expected = x_np + y_np

            d_x = cuda.to_device(x_np)
            d_y = cuda.to_device(y_np)
            d_z = cuda.device_array(N, dtype=np.float32)

            threads = 256
            blocks = math.ceil(N / threads)
            _numba_vecadd_kernel[blocks, threads](d_x, d_y, d_z, N)
            z_np = d_z.copy_to_host()

            assert np.allclose(z_np, expected, atol=1e-5), f"Failed for N={N}"


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


# ── JAX vecadd (execution test) ──────────────────────────────────────────

@jax_lego_jit
@functools.partial(jax.jit, static_argnums=(2,))
def _jax_vecadd(x, y, N):
    L = OrderBy(Row(N)).TileBy([N])
    offs = L[:]
    return x[offs] + y[offs]


class TestJAXVecadd:
    """Execution test: LEGO-rewritten JAX vecadd vs PyTorch."""

    def test_vecadd_matches_pytorch(self):
        N = 2**16
        x_torch = torch.randn(N)
        y_torch = torch.randn(N)
        expected = (x_torch + y_torch).numpy()

        x_jax = jnp.array(x_torch.numpy())
        y_jax = jnp.array(y_torch.numpy())
        z_jax = _jax_vecadd(x_jax, y_jax, N)

        assert np.allclose(np.asarray(z_jax), expected, atol=1e-5)

    def test_vecadd_multiple_sizes(self):
        for N in [128, 1024, 8192, 2**16]:
            x_np = np.random.randn(N).astype(np.float32)
            y_np = np.random.randn(N).astype(np.float32)
            expected = x_np + y_np

            z_jax = _jax_vecadd(jnp.array(x_np), jnp.array(y_np), N)

            assert np.allclose(np.asarray(z_jax), expected, atol=1e-5), \
                f"Failed for N={N}"


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
