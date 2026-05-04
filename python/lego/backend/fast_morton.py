"""Custom Morton (Z-order) layout that emits MLIR bit-magic ops directly.

The standard ``ZCurve`` / per-bit-loop ``GenP`` lowers each Morton encode to
~4*nbits arith ops via the SymPy → arith pipeline (``floor(x/2^k) % 2``
patterns), even after ``lego-strength-reduction`` rewrites them to shifts.

This module bypasses SymPy entirely and emits the standard
**log-time bit-spread** in MLIR directly:

    spread(x):
        x = (x | (x << 8)) & 0x00FF00FF    # nbits=16 only
        x = (x | (x << 4)) & 0x0F0F0F0F
        x = (x | (x << 2)) & 0x33333333
        x = (x | (x << 1)) & 0x55555555

    morton(i, j) = spread(i) | (spread(j) << 1)

Per Morton encode this is ~12 ops/coord vs ~40 in the per-bit form — a ~3x
reduction in arithmetic. Pure ``arith`` ops; portable across x86, ARM, GPU.

This is a "duck-typed" layout — it's a normal :class:`GenP` for type checks,
but carries an ``mlir_apply`` callable that is preferred over the SymPy
``f_apply`` when present. The ``emit_layout_from_python`` GenP branch in
``symbolic.py`` checks for this attribute.
"""

from typing import List, Tuple
import math

from lego.core import GenP
from lego.mlir.ir import IndexType, IntegerAttr
from lego.mlir.dialects import arith


def _const(v: int):
    """Build an index-typed arith constant."""
    return arith.constant(IndexType.get(),
                          IntegerAttr.get(IndexType.get(), int(v)))


def _spread_bits(x, nbits: int):
    """Spread the low ``nbits`` of ``x`` into every other bit position.

    For nbits ≤ 16, uses the standard 4-stage log-time bit-magic. The masks
    are tailored to the bit count (we only spread `nbits`, not all 32). The
    intermediate results never exceed 2*nbits bits.
    """
    if nbits > 16:
        raise ValueError(
            f"_spread_bits requires nbits ≤ 16 (got {nbits}); the masks "
            "below assume bit-spread fits in a 32-bit lane.")

    def shl(a, k):
        return arith.shli(a, _const(k))
    def shr(a, k):
        return arith.shrui(a, _const(k))
    def band(a, mask):
        return arith.andi(a, _const(mask))
    def bor(a, b):
        return arith.ori(a, b)

    # Mask the input to exactly nbits.
    mask_in = (1 << nbits) - 1
    x = band(x, mask_in)

    # Stage 1: spread by 8 bits if nbits > 8.
    if nbits > 8:
        x = bor(x, shl(x, 8))
        x = band(x, 0x00FF00FF)
    # Stage 2: spread by 4.
    if nbits > 4:
        x = bor(x, shl(x, 4))
        x = band(x, 0x0F0F0F0F)
    # Stage 3: spread by 2.
    if nbits > 2:
        x = bor(x, shl(x, 2))
        x = band(x, 0x33333333)
    # Stage 4: spread by 1.
    x = bor(x, shl(x, 1))
    x = band(x, 0x55555555)
    return x


def _spread_bits_sympy_fallback(x_sym, nbits: int):
    """SymPy expression for ``_spread_bits`` — used by f_apply fallback."""
    import sympy as sp
    result = sp.Integer(0)
    for k in range(nbits):
        bit = sp.Mod(sp.floor(x_sym / sp.Integer(1 << k)), sp.Integer(2))
        result = result + bit * sp.Integer(1 << (2 * k))
    return result


def Morton2DFast(N: int):
    """Build a 2-D Morton (Z-order) layout with fast bit-magic emission.

    Returns a :class:`GenP` over ``(N, N)``. The forward function:

      * If the caller goes through ``emit_layout_from_python``'s normal SymPy
        path, ``f_apply`` produces the per-bit form (correct, slow).
      * If the caller honours the ``mlir_apply`` attribute, the body is
        populated by direct MLIR emission of the bit-magic chain.
    """
    assert N > 0 and (N & (N - 1)) == 0, f"Morton2DFast: N must be power of 2, got {N}"
    nbits = (N - 1).bit_length()

    def f_apply(args):
        import sympy as sp
        i, j = args
        return (_spread_bits_sympy_fallback(i, nbits)
                + _spread_bits_sympy_fallback(j, nbits) * sp.Integer(2))

    def f_inv(z):
        import sympy as sp
        i = sp.Integer(0)
        j = sp.Integer(0)
        for k in range(nbits):
            i_bit = sp.Mod(sp.floor(z / sp.Integer(1 << (2 * k))), sp.Integer(2))
            j_bit = sp.Mod(sp.floor(z / sp.Integer(1 << (2 * k + 1))), sp.Integer(2))
            i = i + i_bit * sp.Integer(1 << k)
            j = j + j_bit * sp.Integer(1 << k)
        return (i, j)

    layout = GenP((N, N), f_apply, f_inv)

    def mlir_apply(block_args: List):
        """Emit the Morton 2-D apply body from raw MLIR Values.

        ``block_args`` are the gen_p apply region's index-typed arguments
        (i, j). Returns the flat index Value to yield from the region.
        """
        i, j = block_args
        i_spread = _spread_bits(i, nbits)
        j_spread = _spread_bits(j, nbits)
        # j goes into odd positions: (j_spread << 1)
        j_shifted = arith.shli(j_spread, _const(1))
        return arith.ori(i_spread, j_shifted)

    layout.mlir_apply = mlir_apply
    return layout
