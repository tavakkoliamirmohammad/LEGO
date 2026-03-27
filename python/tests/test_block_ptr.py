"""Tests for block_ptr (TMA) code generation support."""

import ast
import pytest
import sympy as sp

from lego.core import OrderBy, Row, Col, RegP, TileByLayout
from lego.rewriter import (
    rewrite, _format_tuple, _format_make_block_ptr,
    _try_block_ptr_pattern, _extract_subscript_indices,
)
from lego.frontends.triton_jit import (
    TritonAdapter, TritonCodePrinter, BlockPtrInfo, extract_block_ptr_metadata,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _rewrite_source(fn, adapter):
    return rewrite(fn, adapter, return_source=True)


def _sym(name):
    return sp.Symbol(name, integer=True, positive=True)


# ── BlockPtrInfo metadata extraction ─────────────────────────────────────

class TestBlockPtrMetadata:
    def test_row_2d(self):
        M, K, BM, BK = sp.symbols('M K BM BK', integer=True, positive=True)
        pid_m, k = _sym('pid_m'), _sym('k')
        L = OrderBy(Row(M, K)).TileBy([M / BM, K / BK], [BM, BK])

        info = extract_block_ptr_metadata(L,[pid_m, k, slice(None), slice(None)])
        assert info is not None
        assert info.shape == (M, K)
        assert info.strides == (K, sp.Integer(1))
        assert info.offsets == (BM * pid_m, BK * k)
        assert info.block_shape == (BM, BK)
        assert info.order == (1, 0)

    def test_col_2d(self):
        M, K, BM, BK = sp.symbols('M K BM BK', integer=True, positive=True)
        pid_m, k = _sym('pid_m'), _sym('k')
        L = OrderBy(Col(M, K)).TileBy([M / BM, K / BK], [BM, BK])

        info = extract_block_ptr_metadata(L,[pid_m, k, slice(None), slice(None)])
        assert info is not None
        assert info.strides == (sp.Integer(1), M)
        assert info.order == (0, 1)

    def test_1d_layout(self):
        N, BS = sp.symbols('N BS', integer=True, positive=True)
        pid = _sym('pid')
        L = OrderBy(Row(N)).TileBy([N / BS], [BS])

        info = extract_block_ptr_metadata(L,[pid, slice(None)])
        assert info is not None
        assert info.shape == (N,)
        assert info.strides == (sp.Integer(1),)
        assert info.block_shape == (BS,)
        assert info.order == (0,)

    def test_boundary_dims_empty_when_divisible(self):
        M, K, BM, BK = sp.symbols('M K BM BK', integer=True, positive=True)
        L = OrderBy(Row(M, K)).TileBy([M / BM, K / BK], [BM, BK])
        info = extract_block_ptr_metadata(L,[_sym('i'), _sym('j'), slice(None), slice(None)])
        assert info.boundary_dims == ()

    def test_incompatible_multi_perm(self):
        M, K, BM, BK = sp.symbols('M K BM BK', integer=True, positive=True)
        L = OrderBy(Row(M, K), Row(K, M)).TileBy([M / BM, K / BK], [BM, BK])
        info = extract_block_ptr_metadata(L,[_sym('i'), _sym('j'), slice(None), slice(None)])
        assert info is None

    def test_incompatible_regp(self):
        M, K, BM, BK = sp.symbols('M K BM BK', integer=True, positive=True)
        L = OrderBy(RegP((M, K), (1, 0))).TileBy([M / BM, K / BK], [BM, BK])
        info = extract_block_ptr_metadata(L,[_sym('i'), _sym('j'), slice(None), slice(None)])
        assert info is None

    def test_wrong_subscript_shape(self):
        M, K, BM, BK = sp.symbols('M K BM BK', integer=True, positive=True)
        L = OrderBy(Row(M, K)).TileBy([M / BM, K / BK], [BM, BK])
        # Too few indices
        info = extract_block_ptr_metadata(L,[_sym('i'), slice(None)])
        assert info is None


# ── Code formatting helpers ──────────────────────────────────────────────

class TestFormatHelpers:
    def test_format_tuple_ints(self):
        printer = TritonCodePrinter()
        result = _format_tuple([1, 0], printer)
        assert result == '(1, 0)'

    def test_format_tuple_single(self):
        printer = TritonCodePrinter()
        result = _format_tuple([sp.Integer(1)], printer)
        assert result == '(1,)'

    def test_format_tuple_sympy(self):
        printer = TritonCodePrinter()
        M, K = sp.symbols('M K')
        result = _format_tuple([K, sp.Integer(1)], printer)
        assert 'K' in result
        assert '1' in result

    def test_format_make_block_ptr(self):
        printer = TritonCodePrinter()
        M, K, BM, BK = sp.symbols('M K BM BK')
        info = BlockPtrInfo(
            shape=(M, K), strides=(K, sp.Integer(1)),
            offsets=(sp.Integer(0), sp.Integer(0)),
            block_shape=(BM, BK), order=(1, 0),
            boundary_dims=())
        code = _format_make_block_ptr('a_ptr', info, printer)
        assert 'tl.make_block_ptr' in code
        assert 'base=a_ptr' in code
        assert 'shape=' in code
        assert 'strides=' in code
        assert 'order=' in code


# ── Triton source generation with block_ptr ──────────────────────────────

try:
    import triton  # noqa: F401
    _has_triton = True
except ImportError:
    _has_triton = False


@pytest.mark.skipif(not _has_triton, reason="triton not installed")
class TestTritonBlockPtrGeneration:

    def _make_simple_kernel(self):
        import triton.language as tl
        from lego.core import OrderBy, Row

        def kernel(a_ptr, c_ptr, M, N,
                   BM: tl.constexpr, BN: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            L = OrderBy(Row(M, N)).TileBy([M / BM, N / BN], [BM, BN])
            a_ptrs = a_ptr + L[pid_m, pid_n, :, :]
            a = tl.load(a_ptrs)
            c_ptrs = c_ptr + L[pid_m, pid_n, :, :]
            tl.store(c_ptrs, a)

        return kernel

    def test_make_block_ptr_generated(self):
        fn = self._make_simple_kernel()
        src = _rewrite_source(fn, TritonAdapter(use_block_ptr=True))
        assert 'tl.make_block_ptr' in src
        assert 'base=a_ptr' in src

    def test_layout_removed(self):
        fn = self._make_simple_kernel()
        src = _rewrite_source(fn, TritonAdapter(use_block_ptr=True))
        assert 'OrderBy' not in src
        assert 'TileBy' not in src
        assert 'Row(' not in src

    def test_no_tl_arange(self):
        """block_ptr mode should not generate tl.arange offset expressions."""
        fn = self._make_simple_kernel()
        src = _rewrite_source(fn, TritonAdapter(use_block_ptr=True))
        assert 'tl.arange' not in src

    def test_shape_and_strides(self):
        fn = self._make_simple_kernel()
        src = _rewrite_source(fn, TritonAdapter(use_block_ptr=True))
        assert 'shape=' in src
        assert 'strides=' in src
        assert 'block_shape=' in src
        assert 'order=' in src

    def test_default_mode_no_block_ptr(self):
        """Default (use_block_ptr=False) should NOT generate make_block_ptr."""
        fn = self._make_simple_kernel()
        src = _rewrite_source(fn, TritonAdapter())
        assert 'tl.make_block_ptr' not in src
        assert 'tl.advance' not in src


@pytest.mark.skipif(not _has_triton, reason="triton not installed")
class TestTritonBlockPtrLoop:
    """Test loop hoisting and tl.advance generation."""

    def _make_matmul_kernel(self):
        import triton.language as tl
        from lego.core import OrderBy, Row

        def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                          BM: tl.constexpr, BN: tl.constexpr,
                          BK: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)

            L_A = OrderBy(Row(M, K)).TileBy([M / BM, K / BK], [BM, BK])
            L_B = OrderBy(Row(K, N)).TileBy([K / BK, N / BN], [BK, BN])
            L_C = OrderBy(Row(M, N)).TileBy([M / BM, N / BN], [BM, BN])

            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BK)):
                a_ptrs = a_ptr + L_A[pid_m, k, :, :]
                b_ptrs = b_ptr + L_B[k, pid_n, :, :]
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                acc = tl.dot(a, b, acc)

            c_ptrs = c_ptr + L_C[pid_m, pid_n, :, :]
            tl.store(c_ptrs, acc)

        return matmul_kernel

    def test_advance_generated(self):
        fn = self._make_matmul_kernel()
        src = _rewrite_source(fn, TritonAdapter(use_block_ptr=True))
        assert 'tl.advance(' in src

    def test_block_ptr_hoisted_before_loop(self):
        """make_block_ptr for loop-dependent vars should appear before the for."""
        fn = self._make_matmul_kernel()
        src = _rewrite_source(fn, TritonAdapter(use_block_ptr=True))
        lines = src.split('\n')
        # Find the for-loop line
        for_idx = None
        for i, line in enumerate(lines):
            if 'for k in' in line:
                for_idx = i
                break
        assert for_idx is not None
        # make_block_ptr for a_ptrs/b_ptrs should be before the for loop
        before_loop = '\n'.join(lines[:for_idx])
        assert 'a_ptrs = tl.make_block_ptr' in before_loop
        assert 'b_ptrs = tl.make_block_ptr' in before_loop

    def test_no_make_block_ptr_inside_loop(self):
        """The loop body should NOT contain make_block_ptr (hoisted out)."""
        fn = self._make_matmul_kernel()
        src = _rewrite_source(fn, TritonAdapter(use_block_ptr=True))
        lines = src.split('\n')
        # Find the for-loop line and its indentation
        for_idx = next(i for i, l in enumerate(lines) if 'for k in' in l)
        for_indent = len(lines[for_idx]) - len(lines[for_idx].lstrip())
        # Loop body: lines with deeper indentation than the for statement
        loop_body_lines = []
        for line in lines[for_idx + 1:]:
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= for_indent:
                break
            loop_body_lines.append(line)
        body_src = '\n'.join(loop_body_lines)
        assert 'tl.make_block_ptr' not in body_src

    def test_non_loop_block_ptr_stays(self):
        """c_ptrs (not in a loop) should still be a make_block_ptr."""
        fn = self._make_matmul_kernel()
        src = _rewrite_source(fn, TritonAdapter(use_block_ptr=True))
        assert 'c_ptrs = tl.make_block_ptr' in src

    def test_advance_deltas(self):
        """Check that advance uses correct delta values."""
        fn = self._make_matmul_kernel()
        src = _rewrite_source(fn, TritonAdapter(use_block_ptr=True))
        # a_ptrs advance should have 0 in first dim, BK in second
        assert 'tl.advance(a_ptrs, (0, BK))' in src
        # b_ptrs advance should have BK in first dim, 0 in second
        assert 'tl.advance(b_ptrs, (BK, 0))' in src


@pytest.mark.skipif(not _has_triton, reason="triton not installed")
class TestTritonBlockPtrAutoMode:
    """Test use_block_ptr='auto' mode."""

    def _make_kernel(self):
        import triton.language as tl
        from lego.core import OrderBy, Row

        def kernel(a_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            L = OrderBy(Row(M, N)).TileBy([M / BM, N / BN], [BM, BN])
            a_ptrs = a_ptr + L[pid_m, pid_n, :, :]
            a = tl.load(a_ptrs)

        return kernel

    def test_auto_generates_block_ptr(self):
        fn = self._make_kernel()
        src = _rewrite_source(fn, TritonAdapter(use_block_ptr='auto'))
        assert 'tl.make_block_ptr' in src


@pytest.mark.skipif(not _has_triton, reason="triton not installed")
class TestTritonBlockPtrLoadRewriting:
    """Test mask -> boundary_check rewriting."""

    def _make_kernel_with_mask(self):
        import triton.language as tl
        from lego.core import OrderBy, Row

        def kernel(a_ptr, c_ptr, M, N,
                   BM: tl.constexpr, BN: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            L = OrderBy(Row(M, N)).TileBy([M / BM, N / BN], [BM, BN])
            a_ptrs = a_ptr + L[pid_m, pid_n, :, :]
            a = tl.load(a_ptrs, mask=pid_m < M)
            c_ptrs = c_ptr + L[pid_m, pid_n, :, :]
            tl.store(c_ptrs, a, mask=pid_m < M)

        return kernel

    def test_mask_removed(self):
        fn = self._make_kernel_with_mask()
        src = _rewrite_source(fn, TritonAdapter(use_block_ptr=True))
        assert 'mask=' not in src

    def test_load_without_mask_preserved(self):
        """tl.load(block_ptr) without mask should work."""
        import triton.language as tl
        from lego.core import OrderBy, Row

        def kernel(a_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            L = OrderBy(Row(M, N)).TileBy([M / BM, N / BN], [BM, BN])
            a_ptrs = a_ptr + L[pid_m, pid_n, :, :]
            a = tl.load(a_ptrs)

        src = _rewrite_source(kernel, TritonAdapter(use_block_ptr=True))
        assert 'tl.load(a_ptrs)' in src


# ── Adapter options ──────────────────────────────────────────────────────

class TestAdapterOptions:
    def test_default_no_block_ptr(self):
        adapter = TritonAdapter()
        assert adapter.get_rewriter_options() == {'use_block_ptr': False}

    def test_block_ptr_enabled(self):
        adapter = TritonAdapter(use_block_ptr=True)
        assert adapter.get_rewriter_options() == {'use_block_ptr': True}

    def test_auto_mode(self):
        adapter = TritonAdapter(use_block_ptr='auto')
        assert adapter.get_rewriter_options() == {'use_block_ptr': 'auto'}

    def test_base_adapter_default_options(self):
        from lego.frontends._adapter import DSLAdapter
        from lego.frontends.numba_jit import NumbaCUDAAdapter
        assert NumbaCUDAAdapter().get_rewriter_options() == {}
