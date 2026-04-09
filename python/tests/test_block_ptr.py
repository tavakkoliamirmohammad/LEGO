"""Tests for block_ptr (TMA) code generation support."""

import ast
import pytest
import sympy as sp

from lego.core import OrderBy, Row, Col, RegP, TileByLayout
from lego.rewriter import rewrite
from lego.frontends.triton_jit import (
    TritonAdapter, TritonCodePrinter, BlockPtrInfo, extract_block_ptr_metadata,
    _format_tuple, _format_make_block_ptr, _extract_subscript_indices,
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
        assert info.shape == (M, K)
        assert info.strides == (sp.Integer(1), M)
        assert info.offsets == (BM * pid_m, BK * k)
        assert info.block_shape == (BM, BK)
        assert info.order == (0, 1)
        assert info.boundary_dims == ()

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

    def test_regp_2d(self):
        """RegP with transposed perm is strided — should produce col-major strides."""
        M, K, BM, BK = sp.symbols('M K BM BK', integer=True, positive=True)
        pid_m, k = _sym('pid_m'), _sym('k')
        L = OrderBy(RegP((M, K), (1, 0))).TileBy([M / BM, K / BK], [BM, BK])
        info = extract_block_ptr_metadata(L, [pid_m, k, slice(None), slice(None)])
        assert info is not None
        assert info.shape == (M, K)
        assert info.strides == (sp.Integer(1), M)
        assert info.order == (0, 1)
        assert info.offsets == (BM * pid_m, BK * k)
        assert info.block_shape == (BM, BK)

    def test_nonlinear_genp(self):
        """GenP with a non-linear apply should return None."""
        from lego.core import GenP
        M, K, BM, BK = sp.symbols('M K BM BK', integer=True, positive=True)
        with pytest.warns(UserWarning, match="could not derive inverse"):
            genp = GenP((M, K), lambda idx: idx[0]**2 + idx[1])
        L = OrderBy(genp).TileBy([M / BM, K / BK], [BM, BK])
        info = extract_block_ptr_metadata(L, [_sym('i'), _sym('j'), slice(None), slice(None)])
        assert info is None

    def test_row_3d(self):
        """3D row-major layout should produce correct strides and order."""
        M, K, N = sp.symbols('M K N', integer=True, positive=True)
        BM, BK, BN = sp.symbols('BM BK BN', integer=True, positive=True)
        i, j, k = _sym('i'), _sym('j'), _sym('k')
        L = OrderBy(Row(M, K, N)).TileBy([M / BM, K / BK, N / BN], [BM, BK, BN])
        info = extract_block_ptr_metadata(
            L, [i, j, k, slice(None), slice(None), slice(None)])
        assert info is not None
        assert info.shape == (M, K, N)
        assert info.strides == (K * N, N, sp.Integer(1))
        assert info.order == (2, 1, 0)
        assert info.block_shape == (BM, BK, BN)

    def test_col_3d(self):
        """3D col-major layout should produce correct strides and order."""
        M, K, N = sp.symbols('M K N', integer=True, positive=True)
        BM, BK, BN = sp.symbols('BM BK BN', integer=True, positive=True)
        i, j, k = _sym('i'), _sym('j'), _sym('k')
        L = OrderBy(Col(M, K, N)).TileBy([M / BM, K / BK, N / BN], [BM, BK, BN])
        info = extract_block_ptr_metadata(
            L, [i, j, k, slice(None), slice(None), slice(None)])
        assert info is not None
        assert info.shape == (M, K, N)
        assert info.strides == (sp.Integer(1), M, M * K)
        assert info.order == (0, 1, 2)

    def test_col_1d(self):
        """1D Col should be identical to 1D Row (single stride of 1)."""
        N, BS = sp.symbols('N BS', integer=True, positive=True)
        pid = _sym('pid')
        L = OrderBy(Col(N)).TileBy([N / BS], [BS])
        info = extract_block_ptr_metadata(L, [pid, slice(None)])
        assert info is not None
        assert info.shape == (N,)
        assert info.strides == (sp.Integer(1),)
        assert info.order == (0,)

    def test_row_4d(self):
        """4D row-major: strides are cumulative products from the right."""
        A, B, C, D = sp.symbols('A B C D', integer=True, positive=True)
        BA, BB, BC, BD = sp.symbols('BA BB BC BD', integer=True, positive=True)
        i, j, k, l = _sym('i'), _sym('j'), _sym('k'), _sym('l')
        L = OrderBy(Row(A, B, C, D)).TileBy(
            [A / BA, B / BB, C / BC, D / BD], [BA, BB, BC, BD])
        subs = [i, j, k, l, slice(None), slice(None), slice(None), slice(None)]
        info = extract_block_ptr_metadata(L, subs)
        assert info is not None
        assert info.shape == (A, B, C, D)
        # Row-major strides: [B*C*D, C*D, D, 1]
        assert info.strides == (B * C * D, C * D, D, sp.Integer(1))
        assert info.order == (3, 2, 1, 0)
        assert info.block_shape == (BA, BB, BC, BD)

    def test_col_4d(self):
        """4D col-major: strides are cumulative products from the left."""
        A, B, C, D = sp.symbols('A B C D', integer=True, positive=True)
        BA, BB, BC, BD = sp.symbols('BA BB BC BD', integer=True, positive=True)
        i, j, k, l = _sym('i'), _sym('j'), _sym('k'), _sym('l')
        L = OrderBy(Col(A, B, C, D)).TileBy(
            [A / BA, B / BB, C / BC, D / BD], [BA, BB, BC, BD])
        subs = [i, j, k, l, slice(None), slice(None), slice(None), slice(None)]
        info = extract_block_ptr_metadata(L, subs)
        assert info is not None
        assert info.shape == (A, B, C, D)
        # Col-major strides: [1, A, A*B, A*B*C]
        assert info.strides == (sp.Integer(1), A, A * B, A * B * C)
        assert info.order == (0, 1, 2, 3)
        assert info.block_shape == (BA, BB, BC, BD)

    def test_row_col_2d_are_transpose(self):
        """Row(M, K) and Col(K, M) should produce the same linearization."""
        M, K, BM, BK = sp.symbols('M K BM BK', integer=True, positive=True)
        i, j = _sym('i'), _sym('j')
        # Row(M, K): flat = i*K + j  → strides (K, 1), order (1, 0)
        L_row = OrderBy(Row(M, K)).TileBy([M / BM, K / BK], [BM, BK])
        info_row = extract_block_ptr_metadata(
            L_row, [i, j, slice(None), slice(None)])
        # Col(M, K): flat = i + j*M  → strides (1, M), order (0, 1)
        L_col = OrderBy(Col(M, K)).TileBy([M / BM, K / BK], [BM, BK])
        info_col = extract_block_ptr_metadata(
            L_col, [i, j, slice(None), slice(None)])
        assert info_row is not None and info_col is not None
        # Row innermost dim is last, Col innermost dim is first
        assert info_row.order == (1, 0)
        assert info_col.order == (0, 1)
        # Strides are inverses of each other's structure
        assert info_row.strides[1] == sp.Integer(1)  # Row: last stride is 1
        assert info_col.strides[0] == sp.Integer(1)  # Col: first stride is 1

    def test_regp_identity(self):
        """RegP with identity perm should match Row exactly."""
        M, K, BM, BK = sp.symbols('M K BM BK', integer=True, positive=True)
        i, j = _sym('i'), _sym('j')
        L_regp = OrderBy(RegP((M, K), (0, 1))).TileBy([M / BM, K / BK], [BM, BK])
        L_row = OrderBy(Row(M, K)).TileBy([M / BM, K / BK], [BM, BK])
        subs = [i, j, slice(None), slice(None)]
        info_regp = extract_block_ptr_metadata(L_regp, subs)
        info_row = extract_block_ptr_metadata(L_row, subs)
        assert info_regp is not None
        assert info_regp.strides == info_row.strides
        assert info_regp.order == info_row.order

    def test_regp_3d(self):
        """3D RegP with perm (2,0,1) — strides reflect permuted mixed-radix."""
        M, K, N = sp.symbols('M K N', integer=True, positive=True)
        BM, BK, BN = sp.symbols('BM BK BN', integer=True, positive=True)
        i, j, k = _sym('i'), _sym('j'), _sym('k')
        L = OrderBy(RegP((M, K, N), (2, 0, 1))).TileBy(
            [M / BM, K / BK, N / BN], [BM, BK, BN])
        info = extract_block_ptr_metadata(
            L, [i, j, k, slice(None), slice(None), slice(None)])
        assert info is not None
        assert info.shape == (M, K, N)
        # perm (2,0,1): permuted order is (k,i,j) with dims (N,M,K)
        # flat = k*M*K + i*K + j  →  strides = (K, 1, M*K)
        assert info.strides == (K, sp.Integer(1), M * K)
        # order ascending by stride: dim1(1) < dim0(K) < dim2(M*K)
        assert info.order == (1, 0, 2)

    def test_linear_genp(self):
        """GenP with a linear apply equivalent to Row should succeed."""
        from lego.core import GenP
        M, K, BM, BK = sp.symbols('M K BM BK', integer=True, positive=True)
        i, j = _sym('i'), _sym('j')
        genp = GenP((M, K), lambda idx: idx[0] * K + idx[1])
        L = OrderBy(genp).TileBy([M / BM, K / BK], [BM, BK])
        info = extract_block_ptr_metadata(L, [i, j, slice(None), slice(None)])
        assert info is not None
        assert info.shape == (M, K)
        assert info.strides == (K, sp.Integer(1))
        assert info.order == (1, 0)

    def test_three_tile_groups_rejected(self):
        """Layouts with != 2 tile groups should return None."""
        M, K, BM, BK = sp.symbols('M K BM BK', integer=True, positive=True)
        inner = OrderBy(Row(M, K))
        layout = TileByLayout(
            input_chain=[inner],
            tile_groups=[[M / BM], [K / BK], [BM, BK]],
            group_dims=[(M / BM, K / BK, BM, BK)],
        )
        info = extract_block_ptr_metadata(
            layout, [_sym('i'), _sym('j'), _sym('k'),
                     slice(None), slice(None), slice(None)])
        assert info is None

    def test_boundary_dims_with_regp(self):
        """Boundary detection should work for RegP layouts too."""
        M, N, BM, BN = sp.symbols('M N BM BN', integer=True, positive=True)
        i, j = _sym('i'), _sym('j')
        # Use ceiling div so grid*block != shape (non-divisible)
        import sympy
        L = OrderBy(RegP((M, N), (1, 0))).TileBy(
            [sympy.ceiling(M / BM), sympy.ceiling(N / BN)], [BM, BN])
        info = extract_block_ptr_metadata(L, [i, j, slice(None), slice(None)])
        assert info is not None
        # ceil(M/BM)*BM != M in general → both dims need boundary check
        assert len(info.boundary_dims) == 2

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


# ── Matmul example source verification ────────────────────────────────────

@pytest.mark.skipif(not _has_triton, reason="triton not installed")
class TestMatmulBlockPtrSource:
    """Verify the matmul example still produces correct tl.make_block_ptr output."""

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

    def test_a_block_ptr_shape_strides(self):
        """L_A = Row(M,K): shape=(M,K), strides=(K,1), order=(1,0)."""
        src = _rewrite_source(self._make_matmul_kernel(),
                              TritonAdapter(use_block_ptr=True))
        assert 'shape=(M, K)' in src
        assert 'strides=(K, 1)' in src
        assert 'order=(1, 0)' in src

    def test_b_block_ptr_shape_strides(self):
        """L_B = Row(K,N): shape=(K,N), strides=(N,1), order=(1,0)."""
        src = _rewrite_source(self._make_matmul_kernel(),
                              TritonAdapter(use_block_ptr=True))
        assert 'shape=(K, N)' in src
        assert 'strides=(N, 1)' in src

    def test_c_block_ptr_shape_strides(self):
        """L_C = Row(M,N): shape=(M,N), strides=(N,1), order=(1,0)."""
        src = _rewrite_source(self._make_matmul_kernel(),
                              TritonAdapter(use_block_ptr=True))
        assert 'c_ptrs = tl.make_block_ptr' in src
        assert 'shape=(M, N)' in src

    def test_advance_deltas_correct(self):
        """a advances by (0,BK), b advances by (BK,0)."""
        src = _rewrite_source(self._make_matmul_kernel(),
                              TritonAdapter(use_block_ptr=True))
        assert 'tl.advance(a_ptrs, (0, BK))' in src
        assert 'tl.advance(b_ptrs, (BK, 0))' in src

    def test_no_layout_artifacts(self):
        """No OrderBy/Row/TileBy/tl.arange in generated source."""
        src = _rewrite_source(self._make_matmul_kernel(),
                              TritonAdapter(use_block_ptr=True))
        assert 'OrderBy' not in src
        assert 'TileBy' not in src
        assert 'Row(' not in src
        assert 'tl.arange' not in src


# ── RegP kernel with block_ptr codegen ───────────────────────────────────

@pytest.mark.skipif(not _has_triton, reason="triton not installed")
class TestRegPBlockPtrCodegen:
    """Smoke-test: RegP-based layout with use_block_ptr=True generates valid code."""

    def _make_regp_kernel(self):
        import triton.language as tl
        from lego.core import OrderBy, RegP

        def kernel(a_ptr, c_ptr, M, N,
                   BM: tl.constexpr, BN: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            # RegP with perm (1,0) = column-major
            L = OrderBy(RegP((M, N), (1, 0))).TileBy([M / BM, N / BN], [BM, BN])
            a_ptrs = a_ptr + L[pid_m, pid_n, :, :]
            a = tl.load(a_ptrs)
            c_ptrs = c_ptr + L[pid_m, pid_n, :, :]
            tl.store(c_ptrs, a)

        return kernel

    def test_regp_generates_make_block_ptr(self):
        src = _rewrite_source(self._make_regp_kernel(),
                              TritonAdapter(use_block_ptr=True))
        assert 'tl.make_block_ptr' in src

    def test_regp_col_major_strides(self):
        """RegP(perm=(1,0)) should emit strides=(1, M) and order=(0, 1)."""
        src = _rewrite_source(self._make_regp_kernel(),
                              TritonAdapter(use_block_ptr=True))
        assert 'strides=(1, M)' in src
        assert 'order=(0, 1)' in src

    def test_regp_shape_preserved(self):
        src = _rewrite_source(self._make_regp_kernel(),
                              TritonAdapter(use_block_ptr=True))
        assert 'shape=(M, N)' in src

    def test_regp_no_layout_artifacts(self):
        src = _rewrite_source(self._make_regp_kernel(),
                              TritonAdapter(use_block_ptr=True))
        assert 'RegP' not in src
        assert 'OrderBy' not in src
        assert 'tl.arange' not in src

    def _make_regp_matmul_kernel(self):
        """Matmul where B is column-major via RegP — tests loop hoisting too."""
        import triton.language as tl
        from lego.core import OrderBy, Row, RegP

        def kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)

            L_A = OrderBy(Row(M, K)).TileBy([M / BM, K / BK], [BM, BK])
            # B stored column-major
            L_B = OrderBy(RegP((K, N), (1, 0))).TileBy([K / BK, N / BN], [BK, BN])
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

        return kernel

    def test_regp_matmul_advance_generated(self):
        """Loop hoisting and tl.advance should work with RegP layouts."""
        src = _rewrite_source(self._make_regp_matmul_kernel(),
                              TritonAdapter(use_block_ptr=True))
        assert 'tl.advance(a_ptrs' in src
        assert 'tl.advance(b_ptrs' in src

    def test_regp_matmul_b_col_major(self):
        """B's block_ptr should have col-major strides=(1, K) and order=(0, 1)."""
        src = _rewrite_source(self._make_regp_matmul_kernel(),
                              TritonAdapter(use_block_ptr=True))
        assert 'strides=(1, K)' in src
        assert 'order=(0, 1)' in src


# ── Adapter options ──────────────────────────────────────────────────────

class TestAdapterOptions:
    def test_default_no_block_ptr(self):
        adapter = TritonAdapter()
        assert adapter.use_block_ptr is False

    def test_block_ptr_enabled(self):
        adapter = TritonAdapter(use_block_ptr=True)
        assert adapter.use_block_ptr is True

    def test_auto_mode(self):
        adapter = TritonAdapter(use_block_ptr='auto')
        assert adapter.use_block_ptr == 'auto'
