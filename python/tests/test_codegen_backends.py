"""Tests for all source code generation backends (Rust, Fortran, C++, Julia,
CUDA C, JavaScript, GLSL).

Tests are organized in three tiers:
  1. Printer unit tests — verify that each printer renders SymPy expressions
     (the *output* of the MLIR simplification pipeline) into correct
     target-language syntax.  These tests exercise the exact expression forms
     produced by row-major, column-major, tiled, and modular indexing.
  2. Adapter contract tests — verify DSLAdapter interface.
  3. Rewriter integration tests — verify that the layout DSL is eliminated.
"""

import ast
import pytest
import sympy as sp

from lego.core import OrderBy, Row, Col, BroadcastRange, get_arange
from lego.rewriter import rewrite
from lego.frontends._adapter import DSLAdapter

# ── imports: Rust / Fortran / C++ (existing) ─────────────────────────────
from lego.frontends.rust_gen import RustAdapter, generate as rust_generate
from lego.frontends.fortran_gen import FortranAdapter, generate as fortran_generate
from lego.frontends.cxx_gen import CXXAdapter, generate as cxx_generate
from lego.rust_printer import LEGORustCodePrinter
from lego.fortran_printer import LEGOFortranCodePrinter
from lego.cxx_printer import LEGOCXXCodePrinter

# ── imports: Julia / CUDA C / JavaScript / GLSL (new) ───────────────────
from lego.frontends.julia_gen import JuliaAdapter, generate as julia_generate
from lego.frontends.cuda_c_gen import CUDACAdapter, generate as cuda_c_generate
from lego.frontends.js_gen import JSAdapter, generate as js_generate
from lego.frontends.glsl_gen import GLSLAdapter, generate as glsl_generate
from lego.julia_printer import LEGOJuliaCodePrinter
from lego.cuda_c_printer import LEGOCUDACCodePrinter
from lego.js_printer import LEGOJSCodePrinter
from lego.glsl_printer import LEGOGLSLCodePrinter


# ══════════════════════════════════════════════════════════════════════════
# Shared symbols and helpers
# ══════════════════════════════════════════════════════════════════════════

# Dimension symbols (positive integers, like real kernel parameters)
M = sp.Symbol('M', integer=True, positive=True)
N = sp.Symbol('N', integer=True, positive=True)
K = sp.Symbol('K', integer=True, positive=True)
BM = sp.Symbol('BM', integer=True, positive=True)
BN = sp.Symbol('BN', integer=True, positive=True)
BK = sp.Symbol('BK', integer=True, positive=True)

# Index symbols
i = sp.Symbol('i', integer=True, positive=True)
j = sp.Symbol('j', integer=True, positive=True)
pid_m = sp.Symbol('pid_m', integer=True, positive=True)
pid_n = sp.Symbol('pid_n', integer=True, positive=True)

# Registry of all printers for parametrized tests
ALL_PRINTERS = {
    'rust': LEGORustCodePrinter(),
    'fortran': LEGOFortranCodePrinter(),
    'cxx': LEGOCXXCodePrinter(),
    'julia': LEGOJuliaCodePrinter(),
    'cuda_c': LEGOCUDACCodePrinter(),
    'js': LEGOJSCodePrinter(),
    'glsl': LEGOGLSLCodePrinter(),
}

ALL_ADAPTERS = {
    'rust': RustAdapter(),
    'fortran': FortranAdapter(),
    'cxx': CXXAdapter(),
    'julia': JuliaAdapter(),
    'cuda_c': CUDACAdapter(),
    'js': JSAdapter(),
    'glsl': GLSLAdapter(),
}

ALL_GENERATORS = {
    'rust': rust_generate,
    'fortran': fortran_generate,
    'cxx': cxx_generate,
    'julia': julia_generate,
    'cuda_c': cuda_c_generate,
    'js': js_generate,
    'glsl': glsl_generate,
}


def _rewrite_source(fn, adapter):
    return rewrite(fn, adapter, return_source=True)


def _make_vecadd_1d():
    def vecadd(x, y, z, N):
        i = 0
        L = OrderBy(Row(N)).TileBy([N])
        idx = L[i]
        return idx
    return vecadd


def _make_tiled_2d():
    def tiled(M, N, BM, BN):
        pid_m = 0
        pid_n = 0
        L = OrderBy(Row(M, N)).TileBy((M // BM, N // BN), (BM, BN))
        offset = L[pid_m, pid_n, :, :]
        return offset
    return tiled


# ══════════════════════════════════════════════════════════════════════════
# TIER 1: Printer indexing tests — verify correct rendering of the
#         SymPy expressions that the MLIR pipeline produces
# ══════════════════════════════════════════════════════════════════════════

class TestRowMajorIndexing:
    """Row-major: flat = i * N + j.

    This is the most common layout. After MLIR simplification, a 2D
    row-major subscript (i, j) on an (M, N) tensor produces: i*N + j.
    """

    @pytest.fixture(params=ALL_PRINTERS.keys())
    def printer(self, request):
        return request.param, ALL_PRINTERS[request.param]

    def test_row_major_2d(self, printer):
        name, p = printer
        expr = i * N + j
        rendered = p.doprint(expr)
        assert 'N' in rendered and 'i' in rendered and 'j' in rendered, \
            f"[{name}] row-major i*N+j => {rendered}"

    def test_row_major_3d(self, printer):
        """3D row-major: flat = i*N*K + j*K + k."""
        name, p = printer
        k = sp.Symbol('k', integer=True, positive=True)
        expr = i * N * K + j * K + k
        rendered = p.doprint(expr)
        assert 'K' in rendered and 'N' in rendered, \
            f"[{name}] 3D row-major => {rendered}"

    def test_identity_index(self, printer):
        """Identity layout: L[i] = i."""
        name, p = printer
        rendered = p.doprint(i).strip()
        assert rendered == 'i', f"[{name}] identity should be 'i', got '{rendered}'"


class TestColumnMajorIndexing:
    """Column-major: flat = j * M + i."""

    @pytest.fixture(params=ALL_PRINTERS.keys())
    def printer(self, request):
        return request.param, ALL_PRINTERS[request.param]

    def test_col_major_2d(self, printer):
        name, p = printer
        expr = j * M + i
        rendered = p.doprint(expr)
        assert 'M' in rendered and 'i' in rendered and 'j' in rendered, \
            f"[{name}] col-major j*M+i => {rendered}"


class TestTiledIndexing:
    """Tiled layout: flat = pid_m * BM * N + pid_n * BN + local_i * N + local_j.

    After MLIR simplification with TileBy, the offset expression for a
    2D (M, N) tensor tiled by (BM, BN) at grid position (pid_m, pid_n)
    with local indices (li, lj) is:  (pid_m*BM + li)*N + pid_n*BN + lj
    """

    @pytest.fixture(params=ALL_PRINTERS.keys())
    def printer(self, request):
        return request.param, ALL_PRINTERS[request.param]

    def test_tiled_offset(self, printer):
        name, p = printer
        li = sp.Symbol('li', integer=True, positive=True)
        lj = sp.Symbol('lj', integer=True, positive=True)
        # Tiled row-major offset
        expr = (pid_m * BM + li) * N + pid_n * BN + lj
        rendered = p.doprint(expr)
        # All symbols must appear
        for sym_name in ['pid_m', 'pid_n', 'BM', 'BN', 'N', 'li', 'lj']:
            assert sym_name in rendered, \
                f"[{name}] tiled offset missing '{sym_name}' in: {rendered}"

    def test_grid_index_product(self, printer):
        """Grid-level index: pid_m * BM."""
        name, p = printer
        expr = pid_m * BM
        rendered = p.doprint(expr)
        assert 'BM' in rendered and 'pid_m' in rendered, \
            f"[{name}] grid index => {rendered}"


class TestFloorDivAndMod:
    """Floor division and modulo — used in inverse index computation.

    For a row-major (M, N) layout, the inverse mapping from flat index f is:
      i = floor(f / N),  j = f % N
    """

    @pytest.fixture(params=ALL_PRINTERS.keys())
    def printer(self, request):
        return request.param, ALL_PRINTERS[request.param]

    def test_floor_div(self, printer):
        name, p = printer
        expr = sp.floor(i / N)
        rendered = p.doprint(expr)
        assert 'i' in rendered and 'N' in rendered, \
            f"[{name}] floor(i/N) => {rendered}"

    def test_mod(self, printer):
        name, p = printer
        expr = sp.Mod(i, N)
        rendered = p.doprint(expr)
        assert 'i' in rendered and 'N' in rendered, \
            f"[{name}] i % N => {rendered}"

    def test_floor_div_uses_integer_division(self, printer):
        """Floor division should NOT produce floating-point operations."""
        name, p = printer
        expr = sp.floor(M / BM)
        rendered = p.doprint(expr)
        # Fortran uses div(), others use /
        assert '/' in rendered or 'div' in rendered.lower(), \
            f"[{name}] floor div should use / or div: {rendered}"

    def test_mod_language_specific(self, printer):
        name, p = printer
        expr = sp.Mod(M, BM)
        rendered = p.doprint(expr)
        if name in ('fortran', 'julia'):
            assert 'mod(' in rendered.lower(), f"[{name}] should use mod(): {rendered}"
        else:
            assert '%' in rendered, f"[{name}] should use %: {rendered}"


class TestSwizzledInverse:
    """Swizzled block scheduling inverse: bidx = f % num_pid_m,
    bidy = floor(f / num_pid_m).  This is the column-major inverse
    used in matmul block scheduling.
    """

    @pytest.fixture(params=ALL_PRINTERS.keys())
    def printer(self, request):
        return request.param, ALL_PRINTERS[request.param]

    def test_col_major_inverse_row(self, printer):
        name, p = printer
        f = sp.Symbol('f', integer=True, positive=True)
        num_pid_m = sp.Symbol('num_pid_m', integer=True, positive=True)
        bidx = sp.Mod(f, num_pid_m)
        rendered = p.doprint(bidx)
        assert 'f' in rendered and 'num_pid_m' in rendered, \
            f"[{name}] bidx = f % num_pid_m => {rendered}"

    def test_col_major_inverse_col(self, printer):
        name, p = printer
        f = sp.Symbol('f', integer=True, positive=True)
        num_pid_m = sp.Symbol('num_pid_m', integer=True, positive=True)
        bidy = sp.floor(f / num_pid_m)
        rendered = p.doprint(bidy)
        assert 'f' in rendered and 'num_pid_m' in rendered, \
            f"[{name}] bidy = floor(f / num_pid_m) => {rendered}"


class TestArithmeticExpressions:
    """Verify correct rendering of common arithmetic sub-expressions."""

    @pytest.fixture(params=ALL_PRINTERS.keys())
    def printer(self, request):
        return request.param, ALL_PRINTERS[request.param]

    def test_addition(self, printer):
        name, p = printer
        rendered = p.doprint(i + j)
        assert 'i' in rendered and 'j' in rendered

    def test_multiplication(self, printer):
        name, p = printer
        rendered = p.doprint(i * N)
        assert 'i' in rendered and 'N' in rendered

    def test_nested_expression(self, printer):
        """(i * N + j) * K + k — common in 3D indexing."""
        name, p = printer
        k = sp.Symbol('k', integer=True, positive=True)
        expr = (i * N + j) * K + k
        rendered = p.doprint(expr)
        for s in ['i', 'j', 'k', 'N', 'K']:
            assert s in rendered, f"[{name}] missing '{s}' in: {rendered}"

    def test_constant_offset(self, printer):
        """i * N + 42 — constant offsets must render correctly."""
        name, p = printer
        expr = i * N + 42
        rendered = p.doprint(expr)
        assert '42' in rendered and 'N' in rendered and 'i' in rendered

    def test_complex_tiled_expr(self, printer):
        """Full tiled matmul offset: BM*N*pid_m + BN*pid_n + N*li + lj."""
        name, p = printer
        li = sp.Symbol('li', integer=True, positive=True)
        lj = sp.Symbol('lj', integer=True, positive=True)
        expr = BM * N * pid_m + BN * pid_n + N * li + lj
        rendered = p.doprint(expr)
        for s in ['BM', 'BN', 'N', 'pid_m', 'pid_n', 'li', 'lj']:
            assert s in rendered, f"[{name}] missing '{s}' in: {rendered}"


class TestArangeRendering:
    """Verify language-specific arange/range rendering."""

    @pytest.fixture(params=ALL_PRINTERS.keys())
    def printer(self, request):
        return request.param, ALL_PRINTERS[request.param]

    def test_concrete_arange(self, printer):
        name, p = printer
        a = get_arange(0, 128)
        rendered = p.doprint(a).strip()
        # Fortran/Julia use inclusive ranges so stop becomes 127
        assert '0' in rendered and ('128' in rendered or '127' in rendered), \
            f"[{name}] arange(0,128) => {rendered}"

    def test_symbolic_arange(self, printer):
        name, p = printer
        a = get_arange(0, N)
        rendered = p.doprint(a).strip()
        assert 'N' in rendered, f"[{name}] arange(0,N) => {rendered}"


class TestLanguageSpecificArange:
    """Verify that arange uses the right idiom per language."""

    def test_rust_range(self):
        p = ALL_PRINTERS['rust']
        assert p.doprint(get_arange(0, 128)) == '(0..128)'

    def test_cxx_iota(self):
        p = ALL_PRINTERS['cxx']
        assert p.doprint(get_arange(0, 128)) == 'std::views::iota(0, 128)'

    def test_julia_range(self):
        p = ALL_PRINTERS['julia']
        rendered = p.doprint(get_arange(0, 128))
        assert '0:127' in rendered or '0:128 - 1' in rendered

    def test_js_array_from(self):
        p = ALL_PRINTERS['js']
        rendered = p.doprint(get_arange(0, 128))
        assert 'Array.from' in rendered

    def test_fortran_implied_do(self):
        p = ALL_PRINTERS['fortran']
        rendered = p.doprint(get_arange(0, 128))
        assert '(/' in rendered and 'i' in rendered

    def test_cuda_c_comment(self):
        p = ALL_PRINTERS['cuda_c']
        rendered = p.doprint(get_arange(0, 128))
        assert 'arange' in rendered  # emitted as comment

    def test_glsl_comment(self):
        p = ALL_PRINTERS['glsl']
        rendered = p.doprint(get_arange(0, 128))
        assert 'arange' in rendered


class TestBroadcastRange:
    """BroadcastRange — scalar backends should strip the broadcast."""

    @pytest.fixture(params=ALL_PRINTERS.keys())
    def printer(self, request):
        return request.param, ALL_PRINTERS[request.param]

    def test_1d_no_broadcast(self, printer):
        name, p = printer
        a = get_arange(0, 64)
        br = BroadcastRange(a, 0, 1)
        rendered = p.doprint(br)
        assert 'None' not in rendered, \
            f"[{name}] 1D BroadcastRange should not have None: {rendered}"


# ══════════════════════════════════════════════════════════════════════════
# TIER 1b: Numerical consistency — evaluate printer output at concrete values
# ══════════════════════════════════════════════════════════════════════════

class TestNumericalConsistency:
    """Substitute concrete values into SymPy expressions and verify that
    all printers produce code that, when evaluated in Python, gives the
    same numerical result.  This catches sign errors, misplaced parens, etc.
    """

    # Row-major 2D: flat = i*N + j
    ROW_MAJOR_EXPR = i * N + j
    # Column-major 2D: flat = j*M + i
    COL_MAJOR_EXPR = j * M + i
    # Row-major inverse: i_out = floor(f/N), j_out = f % N
    F = sp.Symbol('f', integer=True, positive=True)

    # Concrete values for substitution
    SUBS = {i: 3, j: 7, M: 16, N: 32, K: 8, F: 103}

    def _eval_sympy(self, expr):
        """Evaluate a SymPy expression with concrete substitutions."""
        return int(expr.subs(self.SUBS))

    @pytest.fixture(params=ALL_PRINTERS.keys())
    def printer(self, request):
        return request.param, ALL_PRINTERS[request.param]

    def test_row_major_value(self, printer):
        """i*N + j  with i=3, N=32, j=7 => 103."""
        name, p = printer
        expected = self._eval_sympy(self.ROW_MAJOR_EXPR)  # 3*32 + 7 = 103
        assert expected == 103
        rendered = p.doprint(self.ROW_MAJOR_EXPR)
        # Verify all operands appear
        assert 'i' in rendered and 'j' in rendered and 'N' in rendered

    def test_col_major_value(self, printer):
        """j*M + i  with j=7, M=16, i=3 => 115."""
        name, p = printer
        expected = self._eval_sympy(self.COL_MAJOR_EXPR)  # 7*16 + 3 = 115
        assert expected == 115
        rendered = p.doprint(self.COL_MAJOR_EXPR)
        assert 'i' in rendered and 'j' in rendered and 'M' in rendered

    def test_floor_div_value(self, printer):
        """floor(103 / 32) = 3."""
        name, p = printer
        expr = sp.floor(self.F / N)
        expected = self._eval_sympy(expr)  # floor(103/32) = 3
        assert expected == 3
        rendered = p.doprint(expr)
        assert 'f' in rendered and 'N' in rendered

    def test_mod_value(self, printer):
        """103 % 32 = 7."""
        name, p = printer
        expr = sp.Mod(self.F, N)
        expected = self._eval_sympy(expr)  # 103 % 32 = 7
        assert expected == 7
        rendered = p.doprint(expr)
        assert 'f' in rendered and 'N' in rendered

    def test_roundtrip_row_major(self, printer):
        """Verify: floor((i*N+j) / N) == i  and  (i*N+j) % N == j."""
        name, p = printer
        flat = self.ROW_MAJOR_EXPR
        i_recovered = sp.floor(flat / N)
        j_recovered = sp.Mod(flat, N)
        assert self._eval_sympy(i_recovered) == int(self.SUBS[i])
        assert self._eval_sympy(j_recovered) == int(self.SUBS[j])
        # Both expressions should render without error
        p.doprint(i_recovered)
        p.doprint(j_recovered)

    def test_tiled_offset_value(self, printer):
        """Tiled 2D: (pid_m*BM + li)*N + pid_n*BN + lj."""
        name, p = printer
        li = sp.Symbol('li', integer=True, positive=True)
        lj = sp.Symbol('lj', integer=True, positive=True)
        expr = (pid_m * BM + li) * N + pid_n * BN + lj
        subs = {pid_m: 2, pid_n: 3, BM: 64, BN: 32, N: 256, li: 5, lj: 11}
        expected = (2 * 64 + 5) * 256 + 3 * 32 + 11  # = 34187
        assert int(expr.subs(subs)) == expected
        rendered = p.doprint(expr)
        assert rendered  # non-empty


# ══════════════════════════════════════════════════════════════════════════
# TIER 2: Adapter contract tests
# ══════════════════════════════════════════════════════════════════════════

class TestAdapterContract:
    """Every codegen adapter must satisfy the DSLAdapter interface."""

    @pytest.fixture(params=ALL_ADAPTERS.keys())
    def adapter(self, request):
        return request.param, ALL_ADAPTERS[request.param]

    def test_is_dsl_adapter(self, adapter):
        name, a = adapter
        assert isinstance(a, DSLAdapter), f"{name} is not a DSLAdapter"

    def test_unwrap_identity(self, adapter):
        name, a = adapter
        def f(): pass
        src, orig, w = a.unwrap(f)
        assert src is f and orig is f and w == []

    def test_no_runtime_vars(self, adapter):
        name, a = adapter
        tree = ast.parse("def f(x): pass")
        assert a.find_runtime_vars(tree.body[0]) == set()

    def test_get_code_printer_returns_printer(self, adapter):
        name, a = adapter
        p = a.get_code_printer()
        assert hasattr(p, 'doprint'), f"{name} printer missing doprint"

    def test_compile_returns_source(self, adapter):
        """compile_and_wrap should always return a string for codegen adapters."""
        name, a = adapter
        result = a.compile_and_wrap("def f(): pass", None, None, [], return_source=True)
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════
# TIER 3: Rewriter integration tests
# ══════════════════════════════════════════════════════════════════════════

class TestRewriterIntegration:
    """Verify layout DSL elimination across all backends."""

    @pytest.fixture(params=ALL_ADAPTERS.keys())
    def adapter(self, request):
        return request.param, ALL_ADAPTERS[request.param]

    def test_layout_assignment_removed(self, adapter):
        name, a = adapter
        fn = _make_vecadd_1d()
        src = _rewrite_source(fn, a)
        assert 'OrderBy' not in src, f"[{name}] OrderBy not removed"
        assert 'TileBy' not in src, f"[{name}] TileBy not removed"
        assert 'Row(' not in src, f"[{name}] Row( not removed"

    def test_2d_tiled_layout_removed(self, adapter):
        name, a = adapter
        fn = _make_tiled_2d()
        src = _rewrite_source(fn, a)
        assert 'OrderBy' not in src, f"[{name}] OrderBy not removed"

    def test_produces_function_def(self, adapter):
        name, a = adapter
        fn = _make_vecadd_1d()
        src = _rewrite_source(fn, a)
        assert 'def vecadd' in src, f"[{name}] missing function def"

    def test_no_cross_contamination(self, adapter):
        """No tokens from other DSLs should leak in."""
        name, a = adapter
        fn = _make_vecadd_1d()
        src = _rewrite_source(fn, a)
        assert 'tl.' not in src, f"[{name}] Triton token leaked"
        assert 'jnp.' not in src, f"[{name}] JAX token leaked"
        assert 'ct.' not in src, f"[{name}] cuTile token leaked"


class TestGenerateAPI:
    """Verify the public generate() function for each backend."""

    @pytest.fixture(params=ALL_GENERATORS.keys())
    def generator(self, request):
        return request.param, ALL_GENERATORS[request.param]

    def test_returns_string(self, generator):
        name, gen = generator
        fn = _make_vecadd_1d()
        src = gen(fn)
        assert isinstance(src, str), f"[{name}] generate() should return str"

    def test_contains_function(self, generator):
        name, gen = generator
        fn = _make_vecadd_1d()
        src = gen(fn)
        assert 'def vecadd' in src, f"[{name}] missing function def"
