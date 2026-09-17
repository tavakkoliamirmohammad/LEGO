"""Delta addressing for tile layouts.

A warp-level instruction touches a tile at coordinates (lane_row + dr, lane_col + dc) where the lane
parts vary per thread and (dr, dc) are compile-time constants (the m-tile, n-tile and k-step of the
access).  Evaluating the layout expression for every access hoists one 64-bit address per access out
of the main loop (dozens of live registers); instead we compute one base address per tile and lane,
    base = layout(lane_row, lane_col),
and express every access as
    layout(lane_row + dr, lane_col + dc) == (base ^ K) + C
with constants K (an XOR on the swizzled chunk bits) and C (a plain offset).  The constants are found
and verified exhaustively over the lane set against the layout's own evaluation, so the rewrite is
either exact or not used (the access falls back to the full expression).  Row-major, padded and
XOR-swizzled layouts admit it for the ldmatrix / mma / cp.async lane sets; a rotation layout does not.
"""
from lego.mlir.ir import IndexType, IntegerType, IntegerAttr, StringAttr, UnitAttr, Operation
from lego.mlir.dialects import arith as arith_d


def _idx(v: int):
    return arith_d.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), int(v))).result


def delta_constants(concrete, rows, cols, dr: int, dc: int):
    """(K, C) with concrete.flat(r + dr, c + dc) == (concrete.flat(r, c) ^ K) + C for all lane
    coordinates (r, c) in zip(rows, cols); None if no such pair exists."""
    src = [int(concrete.flat(r, c)) for r, c in zip(rows, cols)]
    tgt = [int(concrete.flat(r + dr, c + dc)) for r, c in zip(rows, cols)]
    unit = max(1, int(getattr(concrete, "unit", 1)))
    width = int(getattr(concrete, "cols", 0)) + int(getattr(concrete, "pad", 0))
    candidates = [0] + [k * unit for k in range(1, max(1, width // unit))]
    for K in candidates:
        offs = {t - (s ^ K) for s, t in zip(src, tgt)}
        if len(offs) == 1:
            return K, offs.pop()
    return None


def opaque(v):
    """Copy an index through a volatile PTX mov so LLVM cannot hoist the addresses derived from it
    out of the loop (they are recomputed from the base each iteration: one xor + an immediate)."""
    i64 = IntegerType.get_signless(64)
    x = arith_d.IndexCastOp(i64, v).result
    y = Operation.create("llvm.inline_asm", results=[i64], operands=[x],
                         attributes={"asm_string": StringAttr.get("mov.b64 $0, $1;"),
                                     "constraints": StringAttr.get("=l,l"),
                                     "has_side_effects": UnitAttr.get()}).result
    return arith_d.IndexCastOp(IndexType.get(), y).result


class LaneTile:
    """Addresses into one tile layout for a fixed lane -> (row, col) mapping.

    row_val / col_val are the MLIR index values of the lane coordinates; rows_fn / cols_fn give the
    same coordinates numerically for thread ids 0..threads-1 (used to verify the rewrite).
    """

    def __init__(self, ctx, layout, row_val, col_val, rows_fn, cols_fn, threads: int, enabled: bool = True):
        self.ctx, self.layout, self.enabled = ctx, layout, enabled
        self.concrete = getattr(layout, "concrete", None)
        self.row_val, self.col_val = row_val, col_val
        self.rows = [int(rows_fn(t)) for t in range(threads)]
        self.cols = [int(cols_fn(t)) for t in range(threads)]
        self.base = ctx.apply(layout, row_val, col_val) if enabled else None
        self.stats = {"rewritten": 0, "fallback": 0}

    def in_loop(self) -> "LaneTile":
        """A view whose base is an opaque per-iteration copy (call inside the loop body)."""
        t = LaneTile.__new__(LaneTile)
        t.__dict__.update(self.__dict__)
        t.base = opaque(self.base) if self.enabled else None
        return t

    def at(self, dr: int, dc: int):
        """Index value of layout(row + dr, col + dc)."""
        kc = delta_constants(self.concrete, self.rows, self.cols, dr, dc) if (self.enabled and self.concrete is not None) else None
        if kc is None:
            self.stats["fallback"] += 1
            r = arith_d.AddIOp(self.row_val, _idx(dr)).result if dr else self.row_val
            c = arith_d.AddIOp(self.col_val, _idx(dc)).result if dc else self.col_val
            return self.ctx.apply(self.layout, r, c)
        self.stats["rewritten"] += 1
        K, C = kc
        v = self.base
        if K:
            v = arith_d.XOrIOp(v, _idx(K)).result
        if C:
            v = arith_d.AddIOp(v, _idx(C)).result
        return v
