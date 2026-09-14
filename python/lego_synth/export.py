"""Export synthesized concrete layouts as LEGO algebra objects (GenP with explicit inverse).

Bitwise operations are expressed arithmetically (SymPy has no integer XOR), the same
way `lego.frontends.python_mlir.Swizzle` does, so the expressions lower through the
existing LEGO pipeline.
"""
import math
from sympy import Integer, Mod, floor

from .layouts import RowMajor, XorSwizzle, Rotation
from .lego_adapter import load_core


def _bitwise():
    core = load_core()
    return getattr(core, "BitXor", None), getattr(core, "BitAnd", None)


def _bit(v, k):
    return Mod(floor(v / Integer(2 ** k)), Integer(2))


def _and_const(v, mask: int, nbits: int):
    return sum((_bit(v, k) * Integer(2 ** k) for k in range(nbits) if (mask >> k) & 1), Integer(0))


def _xor(a, b, nbits: int):
    return sum((Mod(_bit(a, k) + _bit(b, k), Integer(2)) * Integer(2 ** k) for k in range(nbits)), Integer(0))


def to_lego(layout):
    core = load_core()
    rows, cols = layout.rows, layout.cols

    if isinstance(layout, RowMajor):
        stride = Integer(cols + layout.pad)

        def f_apply(args):
            i, j = args
            return i * stride + j

        def f_inv(f):
            return (floor(f / stride), Mod(f, stride))

        return core.GenP((rows, cols), f_apply, f_inv)

    if isinstance(layout, XorSwizzle):
        unit, s, m = layout.unit, layout.shift, layout.mask
        nchunks = math.ceil(cols / unit)
        cbits = max(1, math.ceil(math.log2(max(nchunks, 2))))
        rbits = max(1, math.ceil(math.log2(max(rows, 2))))

        BitXor, BitAnd = _bitwise()

        def swz(i):  # chunk-index XOR operand: (i >> s) & m
            shifted = floor(i / Integer(2 ** s))
            return BitAnd(shifted, Integer(m)) if BitAnd else _and_const(shifted, m, rbits)

        def xor(a, b):
            return BitXor(a, b) if BitXor else _xor(a, b, cbits)

        def f_apply(args):
            i, j = args
            chunk, low = floor(j / Integer(unit)), Mod(j, Integer(unit))
            return i * Integer(cols) + xor(chunk, swz(i)) * Integer(unit) + low

        def f_inv(f):
            i = floor(f / Integer(cols))
            jp = Mod(f, Integer(cols))
            chunk, low = floor(jp / Integer(unit)), Mod(jp, Integer(unit))
            return (i, xor(chunk, swz(i)) * Integer(unit) + low)

        return core.GenP((rows, cols), f_apply, f_inv)

    if isinstance(layout, Rotation):
        unit, r = layout.unit, layout.r
        nchunks = cols // unit

        def f_apply(args):
            i, j = args
            chunk, low = floor(j / Integer(unit)), Mod(j, Integer(unit))
            return i * Integer(cols) + Mod(chunk + Integer(r) * i, Integer(nchunks)) * Integer(unit) + low

        def f_inv(f):
            i = floor(f / Integer(cols))
            jp = Mod(f, Integer(cols))
            chunk, low = floor(jp / Integer(unit)), Mod(jp, Integer(unit))
            return (i, Mod(chunk - Integer(r) * i, Integer(nchunks)) * Integer(unit) + low)

        return core.GenP((rows, cols), f_apply, f_inv)

    raise TypeError(f"no LEGO export for {type(layout).__name__}")


def to_lego_tile(layout):
    """The GenP wrapped as a tile-level LEGO layout usable as a `Shared`/`Buffer` layout."""
    core = load_core()
    return core.GroupBy([(layout.rows, layout.cols)], [core.OrderBy(to_lego(layout))])
