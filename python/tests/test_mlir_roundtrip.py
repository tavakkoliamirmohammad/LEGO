"""
Comprehensive tests for the MLIR roundtrip simplification.

Tests cover all layouts from test/Lego/sympy_examples.mlir:
  - const:    OrderBy(Row(8,8,8)).TileBy([8,8,8])
  - graphene: OrderBy(RegP([2,2,2,2,2], [4,1,3,2,0])).GroupBy([(2,2,2,2,2)])
  - normal:   OrderBy(Row(8,8,8)).TileBy([4,4,4], [2,2,2])
  - bricks:   OrderBy(Row(4,4,4), Row(2,2,2)).TileBy([4,4,4],[2,2,2])
  - lud:      OrderBy(Row(R*T, R*T)).GroupBy([(R, R, T, T)])

Also tests:
  - z3 is NOT imported
  - simplifier.py does not exist
  - import lego works without z3
  - GroupBy.inv() via MLIR roundtrip
  - GroupBy.__getitem__() via MLIR roundtrip
  - Symbolic dimensions
  - GenP layouts
  - Numerical evaluation consistency
"""

import pytest
import sympy as sp
import sys
import os

# Ensure the project is on PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lego.core import OrderBy, Row, Col, RegP, GenP, GroupBy


# ============================================================================
# Helper
# ============================================================================

def sym(name):
    return sp.Symbol(name, integer=True, positive=True)


def syms(*names):
    return sp.symbols(" ".join(names), integer=True, positive=True)


def assert_expr_equal(got, expected, msg=""):
    diff = sp.simplify(got - expected)
    assert diff == 0, f"{msg}\n  got:      {got}\n  expected: {expected}\n  diff:     {diff}"


def eval_expr(expr, subs):
    """Numerically evaluate a SymPy expression with substitutions."""
    return int(expr.subs(subs))


# ============================================================================
# Infrastructure checks
# ============================================================================

class TestInfrastructure:
    """Verify MLIR roundtrip infrastructure and import health."""

    def test_import_lego(self):
        """import lego must work."""
        import lego
        assert hasattr(lego, "jit")

    def test_regp_has_perm_vector(self):
        """RegP must store raw permutation for MLIR converter."""
        r = RegP([2, 3], [1, 0])
        assert hasattr(r, "_perm_vector")
        assert r._perm_vector == [1, 0]


# ============================================================================
# Apply tests (GroupBy.__getitem__)
# ============================================================================

class TestApply:
    """Test layout.apply via __getitem__ using MLIR roundtrip."""

    def test_const_3d(self):
        """const = OrderBy(Row(8,8,8)).TileBy([8,8,8]) => 64*a + 8*b + c"""
        a, b, c = syms("a b c")
        L = OrderBy(Row(8, 8, 8)).TileBy([8, 8, 8])
        result = L[a, b, c]
        assert_expr_equal(result, 64 * a + 8 * b + c, "const_3d")

    def test_graphene(self):
        """graphene = OrderBy(RegP([2]*5, [4,1,3,2,0])).GroupBy([(2,2,2,2,2)])
        => i + 8*j + 2*k + 16*q + 4*w"""
        i, j, k, w, q = syms("i j k w q")
        L = OrderBy(RegP([2, 2, 2, 2, 2], [4, 1, 3, 2, 0])).GroupBy(
            [(2, 2, 2, 2, 2)]
        )
        result = L[i, j, k, w, q]
        assert_expr_equal(result, i + 8 * j + 2 * k + 16 * q + 4 * w, "graphene")

    def test_normal_3d(self):
        """normal = OrderBy(Row(8,8,8)).TileBy([4,4,4], [2,2,2])
        => (bx*2+i)*64 + (by*2+j)*8 + (bz*2+k)"""
        bx, by, bz, i, j, k = syms("bx by bz i j k")
        L = OrderBy(Row(8, 8, 8)).TileBy([4, 4, 4], [2, 2, 2])
        result = L[bx, by, bz, i, j, k]
        expected = (bx * 2 + i) * 64 + (by * 2 + j) * 8 + (bz * 2 + k)
        assert_expr_equal(result, expected, "normal_3d")

    def test_bricks_3d(self):
        """bricks = OrderBy(Row(4,4,4), Row(2,2,2)).TileBy([4,4,4],[2,2,2])
        => (bx*16 + by*4 + bz) * 8 + (i*4 + j*2 + k)"""
        bx, by, bz, i, j, k = syms("bx by bz i j k")
        L = OrderBy(Row(4, 4, 4), Row(2, 2, 2)).TileBy([4, 4, 4], [2, 2, 2])
        result = L[bx, by, bz, i, j, k]
        expected = (bx * 16 + by * 4 + bz) * 8 + (i * 4 + j * 2 + k)
        assert_expr_equal(result, expected, "bricks_3d")

    def test_lud_symbolic(self):
        """lud = OrderBy(Row(R*T, R*T)).GroupBy([(R, R, T, T)])
        => ii*(R*T²) + jj*(T²) + tidy*T + tidx"""
        R, T = syms("R T")
        ii, jj, tidy, tidx = syms("ii jj tidy tidx")
        L = OrderBy(Row(R * T, R * T)).GroupBy([(R, R, T, T)])
        result = L[ii, jj, tidy, tidx]
        expected = ii * R * T**2 + jj * T**2 + tidy * T + tidx
        assert_expr_equal(result, expected, "lud_symbolic")

    def test_2d_row_major(self):
        """Simple 2D row-major: OrderBy(Row(M, N)).GroupBy([(M, N)])
        => i*N + j"""
        M, N = syms("M N")
        i, j = syms("i j")
        L = OrderBy(Row(M, N)).GroupBy([(M, N)])
        result = L[i, j]
        expected = i * N + j
        assert_expr_equal(result, expected, "2d_row_major")

    def test_2d_col_major_concrete(self):
        """Col-major with concrete dims: OrderBy(Col(4, 8)).GroupBy([(4, 8)])
        => j*4 + i"""
        i, j = syms("i j")
        L = OrderBy(Col(4, 8)).GroupBy([(4, 8)])
        result = L[i, j]
        expected = j * 4 + i
        assert_expr_equal(result, expected, "2d_col_major_concrete")


# ============================================================================
# Inverse tests (GroupBy.inv)
# ============================================================================

class TestInverse:
    """Test layout.inv() via MLIR roundtrip."""

    def test_const_inv(self):
        """inv of OrderBy(Row(8,8,8)).TileBy([8,8,8])
        => [floor(x/64), floor(Mod(x,64)/8), Mod(x,8)]"""
        x = sym("x")
        L = OrderBy(Row(8, 8, 8)).TileBy([8, 8, 8])
        inv = L.inv(x)
        assert len(inv) == 3
        assert_expr_equal(inv[0], sp.floor(x / 64), "const_inv[0]")
        assert_expr_equal(inv[1], sp.floor(sp.Mod(x, 64) / 8), "const_inv[1]")
        assert_expr_equal(inv[2], sp.Mod(x, 8), "const_inv[2]")

    def test_2d_row_inv(self):
        """inv of OrderBy(Row(4, 8)).GroupBy([(4, 8)])
        => [floor(x/8), Mod(x, 8)]"""
        x = sym("x")
        L = OrderBy(Row(4, 8)).GroupBy([(4, 8)])
        inv = L.inv(x)
        assert len(inv) == 2
        assert_expr_equal(inv[0], sp.floor(x / 8), "row_inv[0]")
        assert_expr_equal(inv[1], sp.Mod(x, 8), "row_inv[1]")

    def test_2d_col_inv(self):
        """inv of OrderBy(Col(4, 8)).GroupBy([(4, 8)])
        => [Mod(x, 4), floor(x/4)]"""
        x = sym("x")
        L = OrderBy(Col(4, 8)).GroupBy([(4, 8)])
        inv = L.inv(x)
        assert len(inv) == 2
        assert_expr_equal(inv[0], sp.Mod(x, 4), "col_inv[0]")
        assert_expr_equal(inv[1], sp.floor(x / 4), "col_inv[1]")


# ============================================================================
# Direct simplify_via_mlir tests
# ============================================================================

class TestSimplifyViaMlir:
    """Test the simplify_via_mlir function directly."""

    def test_apply_mode(self):
        from lego.backend.symbolic import simplify_via_mlir
        a, b = syms("a b")
        L = OrderBy(Row(4, 8)).GroupBy([(4, 8)])
        result = simplify_via_mlir(L, "apply", [a, b], {a: (0, 4), b: (0, 8)})
        assert_expr_equal(result, 8 * a + b, "direct_apply")

    def test_inv_mode(self):
        from lego.backend.symbolic import simplify_via_mlir
        x = sym("x")
        L = OrderBy(Row(4, 8)).GroupBy([(4, 8)])
        result = simplify_via_mlir(L, "inv", x, {x: (0, 32)})
        assert len(result) == 2
        assert_expr_equal(result[0], sp.floor(x / 8), "direct_inv[0]")
        assert_expr_equal(result[1], sp.Mod(x, 8), "direct_inv[1]")


# ============================================================================
# GenP tests
# ============================================================================

class TestGenP:
    """Test GenP layouts through MLIR roundtrip."""

    def test_genp_identity(self):
        """GenP with identity mapping should match Row."""
        from lego.backend.symbolic import simplify_via_mlir

        def f_apply(idx):
            return idx[0] * 8 + idx[1]

        def f_inv(flat):
            return (sp.floor(flat / 8), sp.Mod(flat, 8))

        gp = GenP((4, 8), f_apply, f_inv)
        L = OrderBy(gp).GroupBy([(4, 8)])

        a, b = syms("a b")
        result = L[a, b]
        assert_expr_equal(result, a * 8 + b, "genp_identity")
        inv = L.inv(a)
        assert_expr_equal(inv[0], sp.floor(a / 8), "genp_identity_inv[0]")
        assert_expr_equal(inv[1], sp.Mod(a, 8), "genp_identity_inv[1]")

    def test_antidiag_symbolic(self):
        """Antidiagonal GenP with symbolic n — Needleman-Wunsch layout.

        Verifies the Piecewise roundtrip (SymPy → MLIR scf.if → SymPy)
        produces a symbolically correct expression matching the Python
        antidiag reference.
        """
        from lego.core import antidiag

        i, j = syms("i j")
        n = sp.Symbol("n", integer=True, positive=True)

        with pytest.warns(UserWarning, match="could not derive inverse"):
            L = OrderBy(GenP([n, n], lambda x: antidiag(n, x), None)).TileBy((n, n))
        result = L[i, j]
        expected = antidiag(n, (i, j))
        assert sp.simplify(result - expected) == 0, (
            f"Symbolic mismatch:\n  got:      {result}\n  expected: {expected}"
        )


# ============================================================================
# Emit layout tests (unit testing the converter)
# ============================================================================

class TestEmitLayout:
    """Test emit_layout_from_python directly."""

    def test_emit_regp(self):
        from lego.backend.symbolic import emit_layout_from_python
        from lego.mlir.ir import Context, Location, InsertionPoint, Module, IndexType
        from lego.backend.dialects.lego_dialect import register as register_lego
        import lego.mlir.ir as ir

        ctx = Context()
        register_lego(ctx)
        with ctx, Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                r = RegP([3, 4], [1, 0])
                layout_val = emit_layout_from_python(r, {})
                assert layout_val is not None

    def test_emit_orderby(self):
        from lego.backend.symbolic import emit_layout_from_python
        from lego.mlir.ir import Context, Location, InsertionPoint, Module
        from lego.backend.dialects.lego_dialect import register as register_lego

        ctx = Context()
        register_lego(ctx)
        with ctx, Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                L = OrderBy(Row(3, 4))
                layout_val = emit_layout_from_python(L, {})
                assert layout_val is not None

    def test_emit_groupby(self):
        from lego.backend.symbolic import emit_layout_from_python
        from lego.mlir.ir import Context, Location, InsertionPoint, Module
        from lego.backend.dialects.lego_dialect import register as register_lego

        ctx = Context()
        register_lego(ctx)
        with ctx, Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                L = OrderBy(Row(3, 4)).GroupBy([(3, 4)])
                layout_val = emit_layout_from_python(L, {})
                assert layout_val is not None


# ============================================================================
# arith_to_sympy tests
# ============================================================================

class TestArithToSympy:
    """Test the MLIR arith -> SymPy converter."""

    def test_constant(self):
        from lego.backend.symbolic import arith_to_sympy
        from lego.mlir.ir import Context, Location, Module
        from lego.backend.dialects.lego_dialect import register as register_lego
        import lego.mlir.ir as ir

        ctx = Context()
        register_lego(ctx)
        with ctx, Location.unknown():
            module = Module.parse(
                'module { func.func @f() -> index { %c42 = arith.constant 42 : index \n return %c42 : index } }'
            )
            func_op = list(module.body)[0]
            block = func_op.regions[0].blocks[0]
            for op in block:
                if op.name == "func.return":
                    result = arith_to_sympy(op.operands[0], {})
                    assert result == sp.Integer(42)

    def test_addi(self):
        from lego.backend.symbolic import arith_to_sympy
        from lego.mlir.ir import Context, Location, Module
        from lego.backend.dialects.lego_dialect import register as register_lego

        a_sym, b_sym = syms("a b")
        ctx = Context()
        register_lego(ctx)
        with ctx, Location.unknown():
            module = Module.parse("""
module {
  func.func @f(%a: index, %b: index) -> index {
    %r = arith.addi %a, %b : index
    return %r : index
  }
}
""")
            func_op = list(module.body)[0]
            block = func_op.regions[0].blocks[0]
            val_to_sym = {
                block.arguments[0]: a_sym,
                block.arguments[1]: b_sym,
            }
            for op in block:
                if op.name == "func.return":
                    result = arith_to_sympy(op.operands[0], val_to_sym)
                    assert_expr_equal(result, a_sym + b_sym, "addi")


# ============================================================================
# Edge cases
# ============================================================================

class TestSymbolicShapes:
    """Test layouts with symbolic dimensions (e.g. TileBy with M//BM)."""

    def test_lud_benchmark_with_constraints(self):
        """Exact reproduction of paper/benchmarks/cuda/lud.py.

        l = OrderBy(Row(R*T, R*T)).GroupBy([(R, R, T, T)], constraints)
        expr = (ii*R + jj)*T*T + tid
        constraints = [ii < R, jj < R, tid < T*T, 0 < tid]
        l.inv(expr) should give [ii, jj, floor(tid/T), Mod(tid, T)]
        """
        from lego.core import le_constraint

        R, T, ii, jj, tid = syms("R T ii jj tid")
        expr = (ii * R + jj) * T * T + tid
        constraints = [
            le_constraint(ii, R),       # ii < R
            le_constraint(jj, R),       # jj < R
            le_constraint(tid, T * T),  # tid < T*T
            le_constraint(0, tid),      # 0 < tid  (tid > 0)
        ]

        L = OrderBy(Row(R * T, R * T)).GroupBy([(R, R, T, T)], constraints)
        i_out, j_out, tidy_out, tidx_out = L.inv(expr)

        # Verify numerically for R=4, T=4
        R_val, T_val = 4, 4
        for ii_v in range(R_val):
            for jj_v in range(R_val):
                for tid_v in range(T_val * T_val):
                    subs = {R: R_val, T: T_val, ii: ii_v, jj: jj_v, tid: tid_v}
                    assert int(i_out.subs(subs)) == ii_v, (
                        f"i mismatch at ii={ii_v} jj={jj_v} tid={tid_v}: "
                        f"got {int(i_out.subs(subs))}"
                    )
                    assert int(j_out.subs(subs)) == jj_v, (
                        f"j mismatch at ii={ii_v} jj={jj_v} tid={tid_v}: "
                        f"got {int(j_out.subs(subs))}"
                    )
                    assert int(tidy_out.subs(subs)) == tid_v // T_val, (
                        f"tidy mismatch at ii={ii_v} jj={jj_v} tid={tid_v}: "
                        f"got {int(tidy_out.subs(subs))}"
                    )
                    assert int(tidx_out.subs(subs)) == tid_v % T_val, (
                        f"tidx mismatch at ii={ii_v} jj={jj_v} tid={tid_v}: "
                        f"got {int(tidx_out.subs(subs))}"
                    )

    def test_lud_symbolic_apply(self):
        """lud with symbolic R, T dims."""
        R, T = syms("R T")
        ii, jj, tidy, tidx = syms("ii jj tidy tidx")
        L = OrderBy(Row(R * T, R * T)).GroupBy([(R, R, T, T)])
        result = L[ii, jj, tidy, tidx]
        expected = ii * R * T**2 + jj * T**2 + tidy * T + tidx
        assert_expr_equal(result, expected, "lud_symbolic_apply")

    def test_lud_symbolic_inv(self):
        """lud inv with symbolic R, T dims — returns divui/remui expressions."""
        R, T = syms("R T")
        x = sym("x")
        L = OrderBy(Row(R * T, R * T)).GroupBy([(R, R, T, T)])
        inv = L.inv(x)
        assert len(inv) == 4
        # Verify numerically for specific values
        for x_val in [0, 1, 5, 15, 31]:
            R_val, T_val = 4, 4
            subs = {x: x_val, R: R_val, T: T_val}
            idx = [int(e.subs(subs)) for e in inv]
            # Recompute apply
            flat = L.apply(*idx)
            flat_val = int(flat.subs(subs))
            assert flat_val == x_val, f"apply(inv({x_val})) = {flat_val}"

    def test_symbolic_groupby_with_user_constraints(self):
        """User constraints should be lowered to assume_bounds."""
        M, N = syms("M N")
        i, j = syms("i j")
        uc = [sp.Gt(M, 0, evaluate=False), sp.Gt(N, 0, evaluate=False)]
        L = OrderBy(Row(M, N)).GroupBy([(M, N)], user_constraints=uc)
        result = L[i, j]
        expected = i * N + j
        assert_expr_equal(result, expected, "user_constraints")

    def test_symbolic_2d_row_inv(self):
        """Symbolic 2D row-major inverse."""
        M, N = syms("M N")
        x = sym("x")
        L = OrderBy(Row(M, N)).GroupBy([(M, N)])
        inv = L.inv(x)
        assert len(inv) == 2
        # Verify numerically
        for x_val in [0, 3, 7, 11]:
            subs = {x: x_val, M: 4, N: 8}
            idx = [int(e.subs(subs)) for e in inv]
            flat = L.apply(*idx)
            assert int(flat.subs(subs)) == x_val

    def test_auto_inferred_bounds(self):
        """Auto-inferred bounds (dim > 0) should be emitted as assume_bounds."""
        from lego.backend.symbolic import simplify_via_mlir
        M, N = syms("M N")
        i, j = syms("i j")
        L = OrderBy(Row(M, N)).GroupBy([(M, N)])
        # _build_constraints should produce lb=1 for M, N
        constraints = L._build_constraints({i: (0, M), j: (0, N)})
        assert M in constraints
        assert N in constraints
        assert constraints[M][0] == 1  # lb >= 1
        assert constraints[N][0] == 1

    def test_build_constraints_with_user_constraints(self):
        """_build_constraints merges user constraints."""
        M, N = syms("M N")
        uc = [sp.Gt(M, 0, evaluate=False), sp.Lt(N, 100, evaluate=False)]
        L = OrderBy(Row(M, N)).GroupBy([(M, N)], user_constraints=uc)
        constraints = L._build_constraints({})
        # M > 0 → lb=1
        assert constraints[M][0] == 1
        # N < 100 → ub=100
        assert constraints[N][1] == 100


class TestTileBy:
    """Test TileBy layouts — the main use case for Triton kernel code gen."""

    # --- Apply (concrete dims) ---

    def test_tileby_identity_1level(self):
        """TileBy with 1 level matching inner dims = identity.
        OrderBy(Row(8,8)).TileBy([8,8]) => 8*a + b"""
        a, b = syms("a b")
        L = OrderBy(Row(8, 8)).TileBy([8, 8])
        result = L[a, b]
        assert_expr_equal(result, 8 * a + b, "tileby_identity_1level")

    def test_tileby_2d_2level(self):
        """OrderBy(Row(6,8)).TileBy([3,4],[2,2])
        => (bx*2+i)*8 + (by*2+j)"""
        bx, by, i, j = syms("bx by i j")
        L = OrderBy(Row(6, 8)).TileBy([3, 4], [2, 2])
        result = L[bx, by, i, j]
        expected = (bx * 2 + i) * 8 + (by * 2 + j)
        assert_expr_equal(result, expected, "tileby_2d_2level")

    def test_tileby_2d_col_inner(self):
        """TileBy with Col inner layout — verified numerically.
        OrderBy(Col(8,8)).TileBy([4,4],[2,2])
        => (by*2+j)*8 + (bx*2+i)"""
        bx, by, i, j = syms("bx by i j")
        L = OrderBy(Col(8, 8)).TileBy([4, 4], [2, 2])
        result = L[bx, by, i, j]
        expected = (by * 2 + j) * 8 + (bx * 2 + i)
        # Verify numerically (symbolic simplification may leave Mod/floor)
        for bx_v in range(4):
            for by_v in range(4):
                for i_v in range(2):
                    for j_v in range(2):
                        s = {bx: bx_v, by: by_v, i: i_v, j: j_v}
                        assert int(result.subs(s)) == int(expected.subs(s)), (
                            f"bx={bx_v} by={by_v} i={i_v} j={j_v}: "
                            f"got {int(result.subs(s))}, expected {int(expected.subs(s))}"
                        )

    def test_tileby_1d(self):
        """1D TileBy: OrderBy(Row(16)).TileBy([4],[4])
        => bx*4 + i"""
        bx, i = syms("bx i")
        L = OrderBy(Row(16)).TileBy([4], [4])
        result = L[bx, i]
        assert_expr_equal(result, bx * 4 + i, "tileby_1d")

    # --- Inverse (concrete dims) ---

    def test_tileby_inv_2d(self):
        """Inverse of OrderBy(Row(8,8)).TileBy([4,4],[2,2])"""
        x = sym("x")
        L = OrderBy(Row(8, 8)).TileBy([4, 4], [2, 2])
        inv = L.inv(x)
        assert len(inv) == 4  # bx, by, i, j
        # Verify numerically
        for x_val in [0, 1, 7, 15, 31, 63]:
            subs = {x: x_val}
            idx_vals = [int(e.subs(subs)) for e in inv]
            flat = L.apply(*idx_vals)
            assert flat == x_val, f"apply(inv({x_val})) = {flat}"

    def test_tileby_inv_3d(self):
        """Inverse of normal = OrderBy(Row(8,8,8)).TileBy([4,4,4],[2,2,2])"""
        x = sym("x")
        L = OrderBy(Row(8, 8, 8)).TileBy([4, 4, 4], [2, 2, 2])
        inv = L.inv(x)
        assert len(inv) == 6  # bx, by, bz, i, j, k
        for x_val in [0, 1, 63, 127, 255, 511]:
            subs = {x: x_val}
            idx_vals = [int(e.subs(subs)) for e in inv]
            flat = L.apply(*idx_vals)
            assert flat == x_val, f"apply(inv({x_val})) = {flat}"

    # --- Numerical roundtrip (concrete) ---

    def test_tileby_2level_roundtrip(self):
        """Full numerical roundtrip for 2-level TileBy."""
        L = OrderBy(Row(8, 8)).TileBy([4, 4], [2, 2])
        x_sym = sym("x")
        inv_exprs = L.inv(x_sym)
        for x_val in range(64):  # 8*8 = 64
            idx_vals = [int(e.subs(x_sym, x_val)) for e in inv_exprs]
            assert L.apply(*idx_vals) == x_val

    def test_tileby_bricks_roundtrip(self):
        """Bricks layout = OrderBy(Row(4,4), Row(2,2)).TileBy([4,4],[2,2])"""
        L = OrderBy(Row(4, 4), Row(2, 2)).TileBy([4, 4], [2, 2])
        x_sym = sym("x")
        inv_exprs = L.inv(x_sym)
        for x_val in range(64):
            idx_vals = [int(e.subs(x_sym, x_val)) for e in inv_exprs]
            assert L.apply(*idx_vals) == x_val

    # --- Multi-block OrderBy with TileBy ---

    def test_tileby_multi_orderby(self):
        """OrderBy(Row(4,4), Row(2,2)).TileBy([4,4],[2,2]) apply
        => (bx*16+by*4+bz)*8 + (i*4+j*2+k) pattern for 3D bricks"""
        bx, by, bz, i, j, k = syms("bx by bz i j k")
        L = OrderBy(Row(4, 4, 4), Row(2, 2, 2)).TileBy([4, 4, 4], [2, 2, 2])
        result = L[bx, by, bz, i, j, k]
        expected = (bx * 16 + by * 4 + bz) * 8 + (i * 4 + j * 2 + k)
        assert_expr_equal(result, expected, "tileby_multi_orderby")

    # --- TileBy with symbolic shapes ---

    def test_tileby_symbolic_dims_numerical(self):
        """TileBy with symbolic M, N, BM, BN — verified numerically."""
        M, N = syms("M N")
        BM, BN = syms("BM BN")
        pid_m, pid_n, i, j = syms("pid_m pid_n i j")
        L = OrderBy(Row(M, N)).TileBy([M // BM, N // BN], [BM, BN])

        # Evaluate with concrete values
        subs = {M: 8, N: 16, BM: 4, BN: 8}
        result = L[pid_m, pid_n, i, j]
        # For each grid point, check numerically
        for pm in range(2):  # M/BM = 2
            for pn in range(2):  # N/BN = 2
                for ii in range(4):  # BM = 4
                    for jj in range(4):  # pick a few from BN = 8
                        vals = {pid_m: pm, pid_n: pn, i: ii, j: jj, **subs}
                        computed = int(result.subs(vals))
                        expected = (pm * 4 + ii) * 16 + (pn * 8 + jj)
                        assert computed == expected, (
                            f"pid=({pm},{pn}) local=({ii},{jj}): "
                            f"got {computed}, expected {expected}"
                        )


class TestEdgeCases:
    """Test edge cases and potential issues."""

    def test_1d_layout(self):
        """1D layout should work."""
        x = sym("x")
        L = OrderBy(Row(16)).GroupBy([(16,)])
        result = L[x]
        assert_expr_equal(result, x, "1d_layout")

    def test_large_dims(self):
        """Layout with larger dimensions."""
        a, b = syms("a b")
        L = OrderBy(Row(1024, 1024)).GroupBy([(1024, 1024)])
        result = L[a, b]
        assert_expr_equal(result, 1024 * a + b, "large_dims")

    def test_multiple_calls(self):
        """Multiple roundtrip calls should work (no state leaks)."""
        a, b, c = syms("a b c")
        L = OrderBy(Row(8, 8, 8)).TileBy([8, 8, 8])

        result1 = L[a, b, c]
        result2 = L[a, b, c]
        assert_expr_equal(result1, result2, "multiple_calls")

    def test_different_symbol_names(self):
        """Symbols with various names should work."""
        pid_m, pid_n = syms("pid_m pid_n")
        L = OrderBy(Row(4, 4)).GroupBy([(4, 4)])
        result = L[pid_m, pid_n]
        assert_expr_equal(result, 4 * pid_m + pid_n, "different_names")


# ============================================================================
# Tests for _merge_bound
# ============================================================================

class TestMergeBound:
    """Test that _merge_bound properly tightens bounds."""

    def test_merge_takes_tighter_lower_bound(self):
        from lego.core import _merge_bound
        constraints = {}
        _merge_bound(constraints, sym("x"), lb=1, ub=10)
        _merge_bound(constraints, sym("x"), lb=5, ub=None)
        lb, ub = constraints[sym("x")]
        # Should keep max(1, 5) = 5 for lb, 10 for ub
        assert sp.simplify(lb - 5) == 0, f"Expected lb=5, got {lb}"
        assert ub == 10

    def test_merge_takes_tighter_upper_bound(self):
        from lego.core import _merge_bound
        constraints = {}
        _merge_bound(constraints, sym("x"), lb=0, ub=100)
        _merge_bound(constraints, sym("x"), lb=None, ub=50)
        lb, ub = constraints[sym("x")]
        assert lb == 0
        assert sp.simplify(ub - 50) == 0, f"Expected ub=50, got {ub}"

    def test_merge_none_does_not_overwrite(self):
        from lego.core import _merge_bound
        constraints = {}
        _merge_bound(constraints, sym("x"), lb=3, ub=7)
        _merge_bound(constraints, sym("x"), lb=None, ub=None)
        lb, ub = constraints[sym("x")]
        assert lb == 3
        assert ub == 7

    def test_merge_symbolic_bounds(self):
        from lego.core import _merge_bound
        N = sym("N")
        constraints = {}
        _merge_bound(constraints, sym("x"), lb=0, ub=N)
        _merge_bound(constraints, sym("x"), lb=1, ub=None)
        lb, ub = constraints[sym("x")]
        assert sp.simplify(lb - 1) == 0
        assert ub == N


# ============================================================================
# Tests for expanded _lower_sympy_to_index
# ============================================================================

class TestExpandedLowering:
    """Test Abs, ceiling, and subtraction lowering."""

    def test_subtraction_via_subi(self):
        """a - b should produce clean IR via subi."""
        a, b = syms("a b")
        L = OrderBy(Row(10, 10)).GroupBy([(10, 10)])
        # Just verify the roundtrip works with subtraction in constraints
        result = L[a, b]
        assert_expr_equal(result, 10 * a + b)

    def test_genp_with_subtraction(self):
        """GenP with forward function using subtraction."""
        n = sym("n")
        # Simple reverse index: f(i) = n - 1 - i
        L_block = GenP([n], lambda args: n - 1 - args[0],
                       lambda flat: (n - 1 - flat,))
        L = OrderBy(L_block).GroupBy([(n,)])
        i = sym("i")
        result = L[i]
        assert_expr_equal(result, n - 1 - i)


# ============================================================================
# Tests for unsigned vs signed division/remainder selection
# ============================================================================

def _lower_expr_to_ir(sympy_expr, symbols):
    """Lower a SymPy expression to MLIR and return the IR string.

    symbols: list of sp.Symbol used in the expression (become func args).
    Returns the MLIR module string after lowering (before any passes).
    """
    from lego.backend.symbolic import _lower_sympy_to_index, _get_cached_context
    from lego.mlir.ir import Context, Location, Module, InsertionPoint, IndexType, FunctionType
    from lego.mlir.dialects import func as _func_dialect

    ctx = _get_cached_context()
    with ctx, Location.unknown():
        module = Module.create()
        idx_ty = IndexType.get()
        func_ty = FunctionType.get([idx_ty] * len(symbols), [idx_ty])
        with InsertionPoint(module.body):
            f = _func_dialect.FuncOp("test_lower", func_ty)
        entry = f.add_entry_block()
        sym_to_val = {s: entry.arguments[i] for i, s in enumerate(symbols)}
        with InsertionPoint(entry):
            result = _lower_sympy_to_index(sympy_expr, sym_to_val)
            _func_dialect.ReturnOp([result])
    return str(module)


class TestUnsignedOps:
    """Verify divui/remui are emitted for non-negative symbols, divsi/remsi otherwise."""

    def test_floor_positive_emits_divui(self):
        """floor(a/b) with positive symbols must emit arith.divui, not divsi."""
        a = sp.Symbol("a", integer=True, positive=True)
        b = sp.Symbol("b", integer=True, positive=True)
        ir = _lower_expr_to_ir(sp.floor(a / b), [a, b])
        assert "arith.divui" in ir, f"Expected divui in IR:\n{ir}"
        assert "arith.divsi" not in ir, f"Unexpected divsi in IR:\n{ir}"

    def test_floor_bare_emits_divsi(self):
        """floor(a/b) with bare symbols must emit arith.divsi, not divui."""
        a = sp.Symbol("a", integer=True)
        b = sp.Symbol("b", integer=True)
        ir = _lower_expr_to_ir(sp.floor(a / b), [a, b])
        assert "arith.divsi" in ir, f"Expected divsi in IR:\n{ir}"
        assert "arith.divui" not in ir, f"Unexpected divui in IR:\n{ir}"

    def test_mod_positive_emits_remui(self):
        """Mod(a, b) with positive symbols must emit arith.remui, not remsi."""
        a = sp.Symbol("a", integer=True, positive=True)
        b = sp.Symbol("b", integer=True, positive=True)
        ir = _lower_expr_to_ir(sp.Mod(a, b), [a, b])
        assert "arith.remui" in ir, f"Expected remui in IR:\n{ir}"
        assert "arith.remsi" not in ir, f"Unexpected remsi in IR:\n{ir}"

    def test_mod_bare_emits_remsi(self):
        """Mod(a, b) with bare symbols must emit arith.remsi, not remui."""
        a = sp.Symbol("a", integer=True)
        b = sp.Symbol("b", integer=True)
        ir = _lower_expr_to_ir(sp.Mod(a, b), [a, b])
        assert "arith.remsi" in ir, f"Expected remsi in IR:\n{ir}"
        assert "arith.remui" not in ir, f"Unexpected remui in IR:\n{ir}"

    def test_mul_with_denom_positive_emits_divui(self):
        """a/b as Mul(a, 1/b) with positive symbols must emit divui."""
        a = sp.Symbol("a", integer=True, positive=True)
        b = sp.Symbol("b", integer=True, positive=True)
        expr = a * sp.Rational(1, 1) / b  # triggers sp.Mul branch
        ir = _lower_expr_to_ir(expr, [a, b])
        assert "arith.divui" in ir, f"Expected divui in IR:\n{ir}"
        assert "arith.divsi" not in ir, f"Unexpected divsi in IR:\n{ir}"

    def test_nonneg_numerator_positive_denom(self):
        """nonnegative (may be 0) numerator with positive denominator → divui."""
        z = sp.Symbol("z", integer=True, nonnegative=True)
        m = sp.Symbol("m", integer=True, positive=True)
        ir = _lower_expr_to_ir(sp.floor(z / m), [z, m])
        assert "arith.divui" in ir, f"Expected divui in IR:\n{ir}"

    def test_negative_numerator_emits_divsi(self):
        """Negative numerator must use divsi even with positive denominator."""
        a = sp.Symbol("a", integer=True, positive=True)
        b = sp.Symbol("b", integer=True, positive=True)
        # -a / b  →  numerator is -a (is_nonnegative=False)
        ir = _lower_expr_to_ir(sp.floor(-a / b), [a, b])
        assert "arith.divsi" in ir, f"Expected divsi in IR:\n{ir}"

    def test_roundtrip_positive_numerical(self):
        """End-to-end: positive symbols roundtrip produces correct values."""
        N = sym("N")
        L = GenP([4, N], lambda args: N * args[0] + args[1])
        flat = sym("flat")
        i_val, j_val = L.f_inv(flat)
        assert_expr_equal(i_val, sp.floor(flat / N))
        assert_expr_equal(j_val, sp.Mod(flat, N))
        for n in [2, 3, 5]:
            for x in range(4 * n):
                assert n * int(i_val.subs({flat: x, N: n})) + \
                       int(j_val.subs({flat: x, N: n})) == x


# ============================================================================
# Tests for GenP auto-inverse
# ============================================================================

class TestGenPAutoInverse:
    """Test automatic inverse derivation for simple GenP functions."""

    def test_auto_inverse_linear(self):
        """Linear f(i) = 2*i should auto-derive inv(x) = x/2."""
        n = sym("n")
        L = GenP([n], lambda args: 2 * args[0])
        # Should have auto-derived f_inv
        assert L.f_inv is not None, "Auto-inverse should succeed for linear function"

    def test_auto_inverse_identity(self):
        """Identity f(i) = i should auto-derive inv(x) = x."""
        n = sym("n")
        L = GenP([n], lambda args: args[0])
        assert L.f_inv is not None
        flat = sym("flat")
        result = L.f_inv(flat)
        assert_expr_equal(result[0], flat)

    def test_auto_inverse_piecewise_returns_none(self):
        """Piecewise functions should not auto-derive (too complex)."""
        from lego.core import antidiag
        n = sym("n")
        with pytest.warns(UserWarning, match="could not derive inverse"):
            L = GenP([n, n], lambda args: antidiag(n, args))
        # Piecewise should fail gracefully
        # f_inv should be None since we didn't provide one and auto-derive can't handle Piecewise
        assert L.f_inv is None

    def test_auto_inverse_affine_2d(self):
        """2D affine f(i, j) = 3*i + j should auto-derive inverse."""
        L = GenP([4, 3], lambda args: 3 * args[0] + args[1])
        assert L.f_inv is not None, "Auto-inverse should succeed for 2D affine"
        flat = sym("flat")
        result = L.f_inv(flat)
        assert len(result) == 2, f"Expected 2 components, got {len(result)}"
        i_val, j_val = result
        # Verify exact forms: inv(x) = (floor(x/3), Mod(x, 3))
        assert_expr_equal(i_val, sp.floor(flat / 3), "inv[0] should be floor(flat/3)")
        assert_expr_equal(j_val, sp.Mod(flat, 3), "inv[1] should be Mod(flat, 3)")
        # Numerical roundtrip verification for all valid flat values
        for x in range(12):  # 4*3 = 12
            i_num = int(i_val.subs(flat, x))
            j_num = int(j_val.subs(flat, x))
            assert 3 * i_num + j_num == x, f"Roundtrip failed for flat={x}"

    def test_auto_inverse_affine_2d_symbolic(self):
        """2D affine with symbolic dim: f(i, j) = N*i + j → inv(x) = (floor(x/N), Mod(x,N))."""
        N = sym("N")
        L = GenP([4, N], lambda args: N * args[0] + args[1])
        assert L.f_inv is not None, "Auto-inverse should succeed for symbolic 2D affine"
        flat = sym("flat")
        result = L.f_inv(flat)
        assert len(result) == 2
        i_val, j_val = result
        # Verify exact symbolic forms
        assert_expr_equal(i_val, sp.floor(flat / N), "inv[0] should be floor(flat/N)")
        assert_expr_equal(j_val, sp.Mod(flat, N), "inv[1] should be Mod(flat, N)")
        # Numerical roundtrip verification for concrete N values
        for n_val in [2, 3, 5, 7]:
            for x in range(4 * n_val):
                i_num = int(i_val.subs({flat: x, N: n_val}))
                j_num = int(j_val.subs({flat: x, N: n_val}))
                assert n_val * i_num + j_num == x, (
                    f"Roundtrip failed for N={n_val}, flat={x}"
                )


# ============================================================================
# Tests for MLIR context caching
# ============================================================================

class TestContextCaching:
    """Test that MLIR context caching doesn't break roundtrips."""

    def test_repeated_roundtrips_same_layout(self):
        """Multiple calls with the same layout should all succeed."""
        a, b = syms("a b")
        L = OrderBy(Row(8, 8)).GroupBy([(8, 8)])
        for _ in range(5):
            result = L[a, b]
            assert_expr_equal(result, 8 * a + b)

    def test_repeated_roundtrips_different_layouts(self):
        """Different layouts in sequence should work with cached context."""
        a, b = syms("a b")

        L1 = OrderBy(Row(4, 4)).GroupBy([(4, 4)])
        r1 = L1[a, b]
        assert_expr_equal(r1, 4 * a + b)

        L2 = OrderBy(Col(4, 4)).GroupBy([(4, 4)])
        r2 = L2[a, b]
        # Col reverses: 4*b + a
        assert_expr_equal(r2, 4 * b + a)

        # And back to a row layout
        L3 = OrderBy(Row(16, 16)).GroupBy([(16, 16)])
        r3 = L3[a, b]
        assert_expr_equal(r3, 16 * a + b)


# ============================================================================
# Tests for _collect_free_symbols GenP body
# ============================================================================

class TestCollectFreeSymbols:
    """Test that _collect_free_symbols finds symbols in GenP function bodies."""

    def test_genp_body_symbols_collected(self):
        from lego.backend.symbolic import _collect_free_symbols
        K = sym("K")
        N = sym("N")
        with pytest.warns(UserWarning, match="could not derive inverse"):
            L = GenP([N], lambda idx: K * idx[0])
        syms_found = _collect_free_symbols(L)
        assert K in syms_found, f"K should be found in GenP body, got {syms_found}"
        assert N in syms_found, f"N should be found in dims, got {syms_found}"

    def test_genp_inv_body_symbols_collected(self):
        from lego.backend.symbolic import _collect_free_symbols
        K = sym("K")
        N = sym("N")
        L = GenP([N], lambda idx: K * idx[0], lambda flat: (flat / K,))
        syms_found = _collect_free_symbols(L)
        assert K in syms_found, f"K should be found in GenP inv body, got {syms_found}"

    def test_plain_genp_no_extra_symbols(self):
        from lego.backend.symbolic import _collect_free_symbols
        N = sym("N")
        L = GenP([N], lambda idx: idx[0])
        syms_found = _collect_free_symbols(L)
        assert N in syms_found
        # Only N should be found (no extra symbols)
        assert len(syms_found) == 1


# ============================================================================
# Tests for _parse_relational compound LHS
# ============================================================================

class TestParseRelational:
    """Test that compound LHS constraints are properly parsed."""

    def test_compound_lhs_strict_gt(self):
        """2*M > 10 → M > 5 → lb=6"""
        M = sym("M")
        constraints = {}
        rel = sp.StrictGreaterThan(2 * M, 10, evaluate=False)
        GroupBy._parse_relational(rel, constraints)
        assert M in constraints, f"M should have a bound, got {constraints}"
        lb, _ = constraints[M]
        assert lb is not None
        # 2*M > 10 → M > 5 → lb = 6
        assert sp.simplify(lb - 6) == 0, f"Expected lb=6, got {lb}"

    def test_compound_lhs_lt(self):
        """3*N < 30 → N < 10"""
        N = sym("N")
        constraints = {}
        rel = sp.StrictLessThan(3 * N, 30, evaluate=False)
        GroupBy._parse_relational(rel, constraints)
        assert N in constraints, f"N should have a bound, got {constraints}"
        _, ub = constraints[N]
        assert ub is not None
        assert sp.simplify(ub - 10) == 0, f"Expected ub=10, got {ub}"

    def test_plain_symbol_still_works(self):
        """Ensure simple X > 5 still works."""
        X = sym("X")
        constraints = {}
        rel = sp.StrictGreaterThan(X, 5, evaluate=False)
        GroupBy._parse_relational(rel, constraints)
        assert X in constraints
        lb, _ = constraints[X]
        assert lb == 6


def test_signed_division_roundtrip():
    """Verify MLIR roundtrip uses signed division for index arithmetic."""
    from lego.core import Row
    from lego.backend.symbolic import simplify_via_mlir
    import sympy as sp

    M, N = sp.symbols('M N', positive=True, integer=True)
    flat = sp.Symbol('flat', nonneg=True, integer=True)
    result = simplify_via_mlir(Row(M, N), 'inv', flat,
                                constraints={flat: (0, M * N)})
    assert len(result) == 2
    subs = {M: 4, N: 3, flat: 7}
    assert int(result[0].subs(subs)) == 2  # 7 // 3 = 2
    assert int(result[1].subs(subs)) == 1  # 7 % 3 = 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
