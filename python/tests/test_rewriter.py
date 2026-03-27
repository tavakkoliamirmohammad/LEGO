"""Tests for the DSL-agnostic rewriter and per-backend code generation.

Each test defines a plain function (no actual DSL import needed), runs the
rewriter with ``return_source=True``, and checks that the generated source
contains the expected DSL-specific tokens.
"""

import ast
import pytest
from sympy import Max, Min
from lego.core import OrderBy, Row, Col, BroadcastRange, TritonRange, get_arange
from lego.rewriter import rewrite, _parse_and_normalize, _build_eval_env
from lego.frontends._adapter import DSLAdapter
from lego.frontends.triton_jit import TritonAdapter, TritonCodePrinter
from lego.frontends.numba_jit import NumbaCUDAAdapter, NumbaCUDACodePrinter
from lego.frontends.jax_jit import JAXAdapter, JAXCodePrinter
from lego.frontends.cutile_jit import CutileAdapter, CutileCodePrinter


# ── helpers ──────────────────────────────────────────────────────────────

def _rewrite_source(fn, adapter):
    """Run the rewriter and return generated source as a string."""
    return rewrite(fn, adapter, return_source=True)


# ── core / shared ────────────────────────────────────────────────────────

class TestBroadcastRangeAlias:
    def test_triton_range_is_broadcast_range(self):
        assert TritonRange is BroadcastRange


class TestLegoArange:
    def test_get_arange_type(self):
        from lego.core import lego_arange
        a = get_arange(0, 128)
        assert isinstance(a, lego_arange)

    def test_get_arange_args(self):
        a = get_arange(0, 64)
        assert int(a.args[0]) == 0
        assert int(a.args[1]) == 64


# ── Triton backend ───────────────────────────────────────────────────────

# ── Triton availability check (used to skip triton-dependent classes) ────
try:
    import triton  # noqa: F401
    _has_triton = True
except ImportError:
    _has_triton = False


@pytest.mark.skipif(not _has_triton, reason="triton not installed")
class TestTritonParsing:
    """Verify that the Triton adapter produces valid Triton code."""

    def _make_vecadd(self):
        """Return a plain function that mimics a Triton vecadd kernel."""
        # We import tl here so the function's globals include it;
        # the rewriter evaluates code in original_fn.__globals__.
        import triton.language as tl
        from lego.core import OrderBy, Row

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

    def test_output_contains_tl_arange(self):
        fn = self._make_vecadd()
        src = _rewrite_source(fn, TritonAdapter())
        assert 'tl.arange(0, BLOCK_SIZE)' in src

    def test_layout_removed(self):
        fn = self._make_vecadd()
        src = _rewrite_source(fn, TritonAdapter())
        assert 'OrderBy' not in src
        assert 'TileBy' not in src
        assert 'Row(' not in src

    def test_pointer_vars_kept(self):
        """Variables used in tl.load/store must remain as runtime assignments."""
        fn = self._make_vecadd()
        src = _rewrite_source(fn, TritonAdapter())
        assert 'tl.load(' in src
        assert 'tl.store(' in src

    def test_pid_kept_as_runtime(self):
        fn = self._make_vecadd()
        src = _rewrite_source(fn, TritonAdapter())
        assert 'pid = tl.program_id(0)' in src

    def test_offsets_expression(self):
        fn = self._make_vecadd()
        src = _rewrite_source(fn, TritonAdapter())
        assert 'offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)' in src


class TestTritonPrinter:
    def test_arange_rendering(self):
        p = TritonCodePrinter()
        a = get_arange(0, 128)
        assert p.doprint(a) == 'tl.arange(0, 128)'

    def test_broadcast_range_1d(self):
        p = TritonCodePrinter()
        a = get_arange(0, 64)
        br = BroadcastRange(a, 0, 1)
        rendered = p.doprint(br)
        # 1D → no slicing
        assert 'tl.arange(0, 64)' in rendered
        assert 'None' not in rendered

    def test_broadcast_range_2d(self):
        p = TritonCodePrinter()
        a = get_arange(0, 64)
        br = BroadcastRange(a, 0, 2)
        rendered = p.doprint(br)
        assert '[:, None]' in rendered


# ── Numba CUDA backend ──────────────────────────────────────────────────

class TestNumbaCUDAParsing:
    """Verify that the Numba CUDA adapter produces valid Numba code."""

    def _make_vecadd(self):
        from lego.core import OrderBy, Row

        def vecadd(x, y, z, N):
            i = 0  # placeholder for cuda.threadIdx.x + ...
            L = OrderBy(Row(N)).TileBy([N])
            idx = L[i]
            z[idx] = x[idx] + y[idx]

        return vecadd

    def test_layout_removed(self):
        fn = self._make_vecadd()
        src = _rewrite_source(fn, NumbaCUDAAdapter())
        assert 'OrderBy' not in src
        assert 'TileBy' not in src
        assert 'Row(' not in src

    def test_no_tl_references(self):
        fn = self._make_vecadd()
        src = _rewrite_source(fn, NumbaCUDAAdapter())
        assert 'tl.' not in src

    def test_idx_simplified(self):
        fn = self._make_vecadd()
        src = _rewrite_source(fn, NumbaCUDAAdapter())
        # With TileBy([N]), L[i] should simplify to just i
        assert 'idx = i' in src


class TestNumbaCUDAPrinter:
    def test_arange_rendering(self):
        p = NumbaCUDACodePrinter()
        a = get_arange(0, 128)
        assert p.doprint(a) == 'range(0, 128)'

    def test_broadcast_range_scalar(self):
        p = NumbaCUDACodePrinter()
        a = get_arange(0, 64)
        br = BroadcastRange(a, 0, 1)
        rendered = p.doprint(br)
        # Numba uses scalar indexing, no broadcast
        assert 'None' not in rendered


# ── JAX backend ──────────────────────────────────────────────────────────

class TestJAXParsing:
    """Verify that the JAX adapter produces valid JAX code."""

    def _make_vecadd(self):
        from lego.core import OrderBy, Row

        def vecadd(x, y, N):
            L = OrderBy(Row(N)).TileBy([N])
            offs = L[:]
            return x + y  # simplified for parsing test

        return vecadd

    def test_layout_removed(self):
        fn = self._make_vecadd()
        src = _rewrite_source(fn, JAXAdapter())
        assert 'OrderBy' not in src
        assert 'TileBy' not in src
        assert 'Row(' not in src

    def test_no_tl_references(self):
        fn = self._make_vecadd()
        src = _rewrite_source(fn, JAXAdapter())
        assert 'tl.' not in src

    def test_output_contains_jnp_arange(self):
        fn = self._make_vecadd()
        src = _rewrite_source(fn, JAXAdapter())
        assert 'jnp.arange' in src


class TestJAXPrinter:
    def test_arange_rendering(self):
        p = JAXCodePrinter()
        a = get_arange(0, 128)
        assert p.doprint(a) == 'jnp.arange(0, 128)'

    def test_broadcast_range_1d(self):
        p = JAXCodePrinter()
        a = get_arange(0, 64)
        br = BroadcastRange(a, 0, 1)
        rendered = p.doprint(br)
        assert 'jnp.arange(0, 64)' in rendered
        assert 'None' not in rendered

    def test_broadcast_range_2d(self):
        p = JAXCodePrinter()
        a = get_arange(0, 64)
        br = BroadcastRange(a, 0, 2)
        rendered = p.doprint(br)
        assert '[:, None]' in rendered


# ── cuTile backend ──────────────────────────────────────────────────────

class TestCutileParsing:
    """Verify that the cuTile adapter produces valid cuTile code."""

    def _make_vecadd(self):
        from lego.core import OrderBy, Row

        def vecadd(A, B, C, N, TILE):
            bid = 0  # placeholder for ct.bid(0)
            L = OrderBy(Row(N)).TileBy([N / TILE], [TILE])
            offsets = L[bid, :]

        return vecadd

    def test_layout_removed(self):
        fn = self._make_vecadd()
        src = _rewrite_source(fn, CutileAdapter())
        assert 'OrderBy' not in src
        assert 'TileBy' not in src
        assert 'Row(' not in src

    def test_no_tl_references(self):
        fn = self._make_vecadd()
        src = _rewrite_source(fn, CutileAdapter())
        assert 'tl.' not in src

    def test_output_contains_ct_arange(self):
        fn = self._make_vecadd()
        src = _rewrite_source(fn, CutileAdapter())
        assert 'ct.arange' in src

    def test_offsets_expression(self):
        fn = self._make_vecadd()
        src = _rewrite_source(fn, CutileAdapter())
        assert 'ct.arange(TILE, dtype=ct.int32)' in src
        assert 'TILE * bid' in src


class TestCutileSwizzleParsing:
    """Verify that cuTile adapter handles .inv() for block scheduling."""

    def _make_swizzle(self):
        # Max, Min, OrderBy, Col must be in the module globals (not local
        # imports) so the rewriter can eval them via fn.__globals__.
        def matmul(A, B, C, M, N, K, tm, tn, GM):
            bid = 0  # placeholder for ct.bid(0)
            num_pid_m = M // tm
            num_pid_n = N // tn
            L_pid = OrderBy(Col(Max(num_pid_m // GM, 1), 1),
                            Col(Min(num_pid_m, GM), num_pid_n)).TileBy(
                                [num_pid_m, num_pid_n])
            bidx, bidy = L_pid.inv(bid)

        return matmul

    def test_layout_removed(self):
        fn = self._make_swizzle()
        src = _rewrite_source(fn, CutileAdapter())
        assert 'L_pid' not in src
        assert 'OrderBy' not in src
        assert 'Col(' not in src

    def test_inv_produces_assignments(self):
        fn = self._make_swizzle()
        src = _rewrite_source(fn, CutileAdapter())
        assert 'bidx =' in src
        assert 'bidy =' in src

    def test_generated_source_parseable(self):
        fn = self._make_swizzle()
        src = _rewrite_source(fn, CutileAdapter())
        ast.parse(src)


class TestCutilePrinter:
    def test_arange_rendering_zero_start(self):
        p = CutileCodePrinter()
        a = get_arange(0, 128)
        assert p.doprint(a) == 'ct.arange(128, dtype=ct.int32)'

    def test_arange_rendering_nonzero_start(self):
        p = CutileCodePrinter()
        a = get_arange(4, 68)
        rendered = p.doprint(a)
        assert 'ct.arange(64, dtype=ct.int32)' in rendered
        assert '+ 4' in rendered

    def test_broadcast_range_1d(self):
        p = CutileCodePrinter()
        a = get_arange(0, 64)
        br = BroadcastRange(a, 0, 1)
        rendered = p.doprint(br)
        assert 'ct.arange(64, dtype=ct.int32)' in rendered
        assert 'None' not in rendered

    def test_broadcast_range_2d(self):
        p = CutileCodePrinter()
        a = get_arange(0, 64)
        br = BroadcastRange(a, 0, 2)
        rendered = p.doprint(br)
        assert '[:, None]' in rendered


# ── Adapter ABC ──────────────────────────────────────────────────────────

class TestDSLAdapterABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            DSLAdapter()

    def test_triton_adapter_is_dsl_adapter(self):
        assert isinstance(TritonAdapter(), DSLAdapter)

    def test_numba_adapter_is_dsl_adapter(self):
        assert isinstance(NumbaCUDAAdapter(), DSLAdapter)

    def test_jax_adapter_is_dsl_adapter(self):
        assert isinstance(JAXAdapter(), DSLAdapter)

    def test_cutile_adapter_is_dsl_adapter(self):
        assert isinstance(CutileAdapter(), DSLAdapter)


# ── Rewriter internals ──────────────────────────────────────────────────

class TestRewriterInternals:
    def test_parse_and_normalize(self):
        def dummy(x, y):
            return x + y

        tree, func_def, source = _parse_and_normalize(dummy, dummy)
        assert func_def.name == 'dummy'
        assert func_def.lineno == 1
        assert len(func_def.args.args) == 2

    def test_build_eval_env(self):
        def dummy(M, N):
            pass

        tree, func_def, _ = _parse_and_normalize(dummy, dummy)
        env = _build_eval_env(dummy, func_def)
        import sympy as sp
        assert isinstance(env['M'], sp.Symbol)
        assert isinstance(env['N'], sp.Symbol)
        assert env['M'].is_integer
        assert env['M'].is_positive
