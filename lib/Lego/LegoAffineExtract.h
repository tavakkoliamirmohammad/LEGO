//===- LegoAffineExtract.h - Extract AffineExpr from arith address chains ===//
//
// Walks the SSA chain producing a memref index and tries to express it as an
// MLIR ``AffineExpr`` in the standard (dim, symbol) form. Returns std::nullopt
// when any op in the chain is not affine-representable (e.g. bitwise interleave
// for Z-Morton, ``i * j`` of two IVs, data-dependent ``memref.load`` of an
// index buffer).
//
// Why this exists
// ---------------
// LegoVectorize.cpp's ``evalLinearInIV`` solves the same problem with custom
// (coeff, constant, invariants) tuples. By emitting an ``AffineExpr`` instead
// we can reuse upstream MLIR machinery — ``affine::makeComposedFoldedAffineApply``,
// ``affine::DependenceAnalysis``, ``linalg::vectorize`` — without
// re-implementing the analysis layer for each new use site.
//
// Coordinate system
// -----------------
// The returned ``AffineExpr`` is in coordinates where:
//   - dim 0    = the inner-loop induction variable (``iv`` argument)
//   - sym 0..N = loop-invariant Values referenced by the address, in the order
//                they were first encountered during the walk
//
// The caller can convert the result to an ``AffineMap`` and use any upstream
// affine machinery (composition, simplification, dependence tests).
//
//===----------------------------------------------------------------------===//

#ifndef LEGO_LIB_LEGOAFFINEEXTRACT_H
#define LEGO_LIB_LEGOAFFINEEXTRACT_H

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::lego {

struct AffineExtractResult {
  /// Expression in ``(d0, s0..sN-1)`` form. ``d0`` is the IV; symbols are
  /// loop-invariant Values listed in ``symbols``.
  AffineExpr expr;
  /// Loop-invariant Values bound to symbols 0..N-1 (positional).
  llvm::SmallVector<Value, 4> symbols;
};

/// Attempt to express ``v`` as an ``AffineExpr`` over the IV ``iv`` plus
/// loop-invariant symbols.
///
/// Returns ``std::nullopt`` if any op in the SSA cone of ``v`` is not
/// affine-representable. Loop-invariance is decided structurally: a Value is
/// invariant iff it is defined outside the loop containing ``iv`` (i.e. its
/// SSA cone never reaches ``iv``). Constants are folded into the expression.
///
/// Supported ops:
///   - ``arith.constant`` (integer)
///   - The IV itself (becomes ``d0``)
///   - Block / function args defined outside the loop (become symbols)
///   - ``arith.addi``, ``arith.subi``
///   - ``arith.muli`` where at least one operand is invariant
///   - ``arith.divsi``, ``arith.divui``, ``arith.remsi``, ``arith.remui`` with
///     constant RHS (becomes ``floordiv`` / ``mod``)
///   - ``arith.shli``, ``arith.shrsi``, ``arith.shrui`` with constant RHS
///     (canonicalised to ``mul``/``floordiv`` by 2^k)
///   - ``arith.andi`` with mask ``2^k - 1`` (canonicalised to ``mod 2^k``)
///   - ``arith.index_cast``, ``arith.index_castui`` (transparent)
///
/// Unsupported (returns ``nullopt``):
///   - ``arith.muli`` of two non-invariant operands (quadratic)
///   - ``arith.andi``/``arith.ori``/``arith.xori`` with non-power-of-2 masks
///     (e.g. Z-Morton ``0x5555``)
///   - ``arith.shli``/``arith.shrsi``/``arith.shrui`` with non-constant RHS
///   - ``memref.load`` (data-dependent gather)
///   - Any op whose operand cone reaches a non-affine op
std::optional<AffineExtractResult>
tryBuildAffineExpr(Value v, Value iv, MLIRContext *ctx);

}  // namespace mlir::lego

#endif  // LEGO_LIB_LEGOAFFINEEXTRACT_H
