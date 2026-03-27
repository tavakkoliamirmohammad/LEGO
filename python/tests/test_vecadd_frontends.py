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

from sympy import Max, Min
from lego.core import OrderBy, Row, Col
from lego.rewriter import rewrite
from lego.frontends.triton_jit import TritonAdapter, jit as triton_lego_jit
from lego.frontends.numba_jit import jit as numba_lego_jit
from lego.frontends.jax_jit import jit as jax_lego_jit
from lego.frontends.cutile_jit import CutileAdapter, cutile_jit, get_cutile_kernel_source

try:
    import cuda.tile as ct
    _has_cutile = True
except ImportError:
    _has_cutile = False


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


# ── cuTile source verification ─────────────────────────────────────────

class TestCutileVecaddSource:
    """Verify the generated cuTile source is correct."""

    def test_generated_source(self):
        def vecadd(A, B, C, N, TILE):
            bid = 0  # placeholder for ct.bid(0)
            L = OrderBy(Row(N)).TileBy([N / TILE], [TILE])
            offsets = L[bid, :]

        src = rewrite(vecadd, CutileAdapter(), return_source=True)

        # Must contain cuTile-specific tokens
        assert 'ct.arange' in src
        # Must NOT contain Triton or JAX tokens
        assert 'tl.' not in src
        assert 'jnp.' not in src
        # Layout DSL must be fully eliminated
        assert 'OrderBy' not in src
        assert 'TileBy' not in src

    def test_generated_source_parseable(self):
        """The generated source must be valid Python."""
        import ast

        def vecadd(A, B, C, N, TILE):
            bid = 0
            L = OrderBy(Row(N)).TileBy([N / TILE], [TILE])
            offsets = L[bid, :]

        src = rewrite(vecadd, CutileAdapter(), return_source=True)
        ast.parse(src)

    def test_offsets_simplified(self):
        def vecadd(A, B, C, N, TILE):
            bid = 0
            L = OrderBy(Row(N)).TileBy([N / TILE], [TILE])
            offsets = L[bid, :]

        src = rewrite(vecadd, CutileAdapter(), return_source=True)
        assert 'ct.arange(TILE, dtype=ct.int32)' in src
        assert 'TILE * bid' in src


class TestCutileMatmulSwizzleSource:
    """Verify swizzled matmul source generation for cuTile."""

    def test_swizzle_inv_simplified(self):
        def matmul(A, B, C, M, N, K, tm, tn, tk, GM):
            bid = 0
            num_pid_m = M // tm
            num_pid_n = N // tn
            L_pid = OrderBy(Col(Max(num_pid_m // GM, 1), 1),
                            Col(Min(num_pid_m, GM), num_pid_n)).TileBy(
                                [num_pid_m, num_pid_n])
            bidx, bidy = L_pid.inv(bid)

        src = rewrite(matmul, CutileAdapter(), return_source=True)

        # Layout DSL fully eliminated
        assert 'L_pid' not in src
        assert 'OrderBy' not in src
        assert 'Col(' not in src
        # bidx and bidy are computed
        assert 'bidx =' in src
        assert 'bidy =' in src

    def test_generated_source_parseable(self):
        import ast

        def matmul(A, B, C, M, N, K, tm, tn, tk, GM):
            bid = 0
            num_pid_m = M // tm
            num_pid_n = N // tn
            L_pid = OrderBy(Col(Max(num_pid_m // GM, 1), 1),
                            Col(Min(num_pid_m, GM), num_pid_n)).TileBy(
                                [num_pid_m, num_pid_n])
            bidx, bidy = L_pid.inv(bid)

        src = rewrite(matmul, CutileAdapter(), return_source=True)
        ast.parse(src)


# ── cuTile GPU execution tests ─────────────────────────────────────────

_skip_cutile_gpu = pytest.mark.skipif(
    not (_has_cutile and torch.cuda.is_available()),
    reason="cuda.tile and CUDA GPU required",
)


@_skip_cutile_gpu
class TestCutileVecadd:
    """Full GPU test: LEGO-rewritten cuTile vecadd vs PyTorch."""

    @staticmethod
    def _make_kernels():
        @ct.kernel
        def ref_kernel(a, b, c, TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = bid * TILE + ct.arange(TILE, dtype=ct.int32)
            a_tile = ct.gather(a, indices)
            b_tile = ct.gather(b, indices)
            ct.scatter(c, indices, a_tile + b_tile)

        @cutile_jit
        @ct.kernel
        def lego_kernel(a, b, c, N: int, TILE: ct.Constant[int]):
            bid = ct.bid(0)
            L = OrderBy(Row(N)).TileBy([N / TILE], [TILE])
            offsets = L[bid, :]
            a_tile = ct.gather(a, offsets)
            b_tile = ct.gather(b, offsets)
            ct.scatter(c, offsets, a_tile + b_tile)

        return ref_kernel, lego_kernel

    def test_vecadd_matches_pytorch(self):
        ref_kernel, lego_kernel = self._make_kernels()
        N = 2 ** 16
        TILE = 1024
        a = torch.randn(N, device='cuda')
        b = torch.randn(N, device='cuda')
        expected = a + b
        stream = torch.cuda.current_stream()
        grid = (math.ceil(N / TILE), 1, 1)

        c_lego = torch.empty_like(a)
        ct.launch(stream, grid, lego_kernel, (a, b, c_lego, N, TILE))
        assert torch.allclose(c_lego, expected, atol=1e-5)

    def test_vecadd_multiple_sizes(self):
        ref_kernel, lego_kernel = self._make_kernels()
        TILE = 1024
        stream = torch.cuda.current_stream()
        for N in [1024, 8192, 2 ** 16]:
            a = torch.randn(N, device='cuda')
            b = torch.randn(N, device='cuda')
            c = torch.empty_like(a)
            grid = (math.ceil(N / TILE), 1, 1)
            ct.launch(stream, grid, lego_kernel, (a, b, c, N, TILE))
            assert torch.allclose(c, a + b, atol=1e-5), f"Failed for N={N}"

    def test_matches_reference_kernel(self):
        ref_kernel, lego_kernel = self._make_kernels()
        N = 2 ** 16
        TILE = 1024
        a = torch.randn(N, device='cuda')
        b = torch.randn(N, device='cuda')
        stream = torch.cuda.current_stream()
        grid = (math.ceil(N / TILE), 1, 1)

        c_ref = torch.empty_like(a)
        ct.launch(stream, grid, ref_kernel, (a, b, c_ref, TILE))
        c_lego = torch.empty_like(a)
        ct.launch(stream, grid, lego_kernel, (a, b, c_lego, N, TILE))
        assert torch.allclose(c_ref, c_lego, atol=1e-5)


@_skip_cutile_gpu
class TestCutileMatmul:
    """Full GPU test: LEGO-rewritten cuTile matmul vs PyTorch."""

    @staticmethod
    def _make_kernels():
        @cutile_jit
        @ct.kernel
        def lego_kernel(A, B, C,
                        tm: ct.Constant[int], tn: ct.Constant[int],
                        tk: ct.Constant[int]):
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

        return lego_kernel

    def test_matmul_matches_pytorch(self):
        lego_kernel = self._make_kernels()
        M, N, K = 512, 512, 512
        tm, tn, tk = 64, 64, 32
        A = torch.randn(M, K, device='cuda', dtype=torch.float16)
        B = torch.randn(K, N, device='cuda', dtype=torch.float16)
        C = torch.empty(M, N, device='cuda', dtype=torch.float16)
        expected = torch.matmul(A, B)

        stream = torch.cuda.current_stream()
        grid = (math.ceil(M / tm) * math.ceil(N / tn), 1, 1)
        ct.launch(stream, grid, lego_kernel, (A, B, C, tm, tn, tk))
        assert torch.allclose(C, expected, atol=1e-1, rtol=1e-1)
