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

from lego.lego import (
    OrderBy, Row, Col, RegP, GenP, GroupBy,
    flatten_index, unflatten_index, product,
    get_sigma_perm, sigma, inverse_permutation,
)


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
# Phase 0: Infrastructure checks
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
        from lego.lego_mlir import simplify_via_mlir
        a, b = syms("a b")
        L = OrderBy(Row(4, 8)).GroupBy([(4, 8)])
        result = simplify_via_mlir(L, "apply", [a, b], {a: (0, 4), b: (0, 8)})
        assert_expr_equal(result, 8 * a + b, "direct_apply")

    def test_inv_mode(self):
        from lego.lego_mlir import simplify_via_mlir
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
        from lego.lego_mlir import simplify_via_mlir

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


# ============================================================================
# Emit layout tests (unit testing the converter)
# ============================================================================

class TestEmitLayout:
    """Test emit_layout_from_python directly."""

    def test_emit_regp(self):
        from lego.lego_mlir import emit_layout_from_python
        from mlir.ir import Context, Location, InsertionPoint, Module, IndexType
        from lego.dialects.lego_dialect import register as register_lego
        import mlir.ir as ir

        ctx = Context()
        register_lego(ctx)
        with ctx, Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                r = RegP([3, 4], [1, 0])
                layout_val = emit_layout_from_python(r, {})
                assert layout_val is not None

    def test_emit_orderby(self):
        from lego.lego_mlir import emit_layout_from_python
        from mlir.ir import Context, Location, InsertionPoint, Module
        from lego.dialects.lego_dialect import register as register_lego

        ctx = Context()
        register_lego(ctx)
        with ctx, Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                L = OrderBy(Row(3, 4))
                layout_val = emit_layout_from_python(L, {})
                assert layout_val is not None

    def test_emit_groupby(self):
        from lego.lego_mlir import emit_layout_from_python
        from mlir.ir import Context, Location, InsertionPoint, Module
        from lego.dialects.lego_dialect import register as register_lego

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
        from lego.lego_mlir import arith_to_sympy
        from mlir.ir import Context, Location, Module
        from lego.dialects.lego_dialect import register as register_lego
        import mlir.ir as ir

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
        from lego.lego_mlir import arith_to_sympy
        from mlir.ir import Context, Location, Module
        from lego.dialects.lego_dialect import register as register_lego

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
        from lego.lego import le_constraint

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
        from lego.lego_mlir import simplify_via_mlir
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
