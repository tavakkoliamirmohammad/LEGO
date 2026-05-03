//===- LegoAffineExtract.cpp - Walk arith chain into AffineExpr ============//
//
// Implementation of tryBuildAffineExpr. See header for design notes.
//
//===----------------------------------------------------------------------===//

#include "LegoAffineExtract.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace mlir::lego {

namespace {

/// Working state for the recursive walk. Keeps the symbol table and a memo of
/// already-visited Values so we don't re-walk shared sub-DAGs.
struct ExtractState {
  MLIRContext *ctx;
  Value iv;
  llvm::SmallVector<Value, 4> symbols;
  /// Cache: Value -> AffineExpr (if affine) or nullopt (if non-affine).
  /// We use a separate set for nullopt entries since DenseMap can't store
  /// std::optional<AffineExpr> cleanly.
  llvm::DenseMap<Value, AffineExpr> cache;
  llvm::DenseSet<Value> rejected;

  /// Bind a loop-invariant Value to a symbol index, reusing an existing
  /// symbol if we've seen this Value before.
  AffineExpr getOrCreateSymbol(Value v) {
    for (size_t i = 0; i < symbols.size(); ++i) {
      if (symbols[i] == v)
        return getAffineSymbolExpr(i, ctx);
    }
    unsigned pos = symbols.size();
    symbols.push_back(v);
    return getAffineSymbolExpr(pos, ctx);
  }
};

/// Match ``v`` as a compile-time integer constant. Returns nullopt for
/// non-constants.
static std::optional<int64_t> matchConstantInt(Value v) {
  IntegerAttr attr;
  if (matchPattern(v, m_Constant(&attr)))
    return attr.getInt();
  return std::nullopt;
}

/// Is ``mask`` of the form ``2^k - 1`` for some k >= 0? Used to canonicalise
/// ``arith.andi %x, mask`` into ``%x mod 2^k``.
static std::optional<int64_t> matchPowerOfTwoMinusOne(int64_t mask) {
  if (mask < 0) return std::nullopt;
  if (mask == 0) return 0;
  // mask = 2^k - 1 iff mask + 1 is a power of two.
  uint64_t plus1 = uint64_t(mask) + 1;
  if ((plus1 & (plus1 - 1)) != 0) return std::nullopt;
  // log2(plus1).
  int64_t k = 0;
  while ((uint64_t(1) << k) < plus1) ++k;
  return k;
}

/// Is ``v`` loop-invariant w.r.t. the IV? Decided structurally: walk the SSA
/// cone of ``v``; if it never reaches ``iv``, it's invariant.
static bool isLoopInvariant(Value v, Value iv,
                            llvm::DenseMap<Value, bool> &cache) {
  if (auto it = cache.find(v); it != cache.end())
    return it->second;
  if (v == iv) {
    cache[v] = false;
    return false;
  }
  Operation *def = v.getDefiningOp();
  if (!def) {
    // Block or function argument: invariant unless it IS the iv (handled above).
    cache[v] = true;
    return true;
  }
  // Recursively check operands.
  bool inv = true;
  for (Value op : def->getOperands()) {
    if (!isLoopInvariant(op, iv, cache)) {
      inv = false;
      break;
    }
  }
  cache[v] = inv;
  return inv;
}

/// Recursive walker. Returns the AffineExpr or nullptr (rejected).
static AffineExpr build(Value v, ExtractState &S) {
  // Cache hits.
  if (auto it = S.cache.find(v); it != S.cache.end())
    return it->second;
  if (S.rejected.count(v))
    return nullptr;

  auto reject = [&]() -> AffineExpr {
    S.rejected.insert(v);
    return nullptr;
  };
  auto accept = [&](AffineExpr e) -> AffineExpr {
    S.cache[v] = e;
    return e;
  };

  // The IV itself becomes dim 0.
  if (v == S.iv)
    return accept(getAffineDimExpr(0, S.ctx));

  // Compile-time constants.
  if (auto c = matchConstantInt(v))
    return accept(getAffineConstantExpr(*c, S.ctx));

  Operation *def = v.getDefiningOp();
  if (!def) {
    // Block / function argument that isn't the IV → loop-invariant symbol.
    return accept(S.getOrCreateSymbol(v));
  }

  // Identity casts are transparent.
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(def)) {
    AffineExpr inner = build(def->getOperand(0), S);
    if (!inner) return reject();
    return accept(inner);
  }

  // ----- Binary arith ops -----
  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    AffineExpr a = build(add.getLhs(), S);
    AffineExpr b = build(add.getRhs(), S);
    if (!a || !b) return reject();
    return accept(a + b);
  }
  if (auto sub = dyn_cast<arith::SubIOp>(def)) {
    AffineExpr a = build(sub.getLhs(), S);
    AffineExpr b = build(sub.getRhs(), S);
    if (!a || !b) return reject();
    return accept(a - b);
  }

  // muli: at least one side must be invariant (constant or pure-symbol).
  // AffineExpr's operator* with a non-constant non-symbolic side throws.
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    AffineExpr a = build(mul.getLhs(), S);
    AffineExpr b = build(mul.getRhs(), S);
    if (!a || !b) return reject();
    // AffineExpr multiplication is only valid if at least one side is purely
    // symbolic-or-constant (no dim references).
    auto isSymOrConst = [](AffineExpr e) {
      return e.isSymbolicOrConstant();
    };
    if (isSymOrConst(a) || isSymOrConst(b))
      return accept(a * b);
    return reject();
  }

  // divsi/divui: RHS must be a positive constant for affine floordiv.
  if (auto div = dyn_cast<arith::DivSIOp>(def)) {
    AffineExpr a = build(div.getLhs(), S);
    auto rhs = matchConstantInt(div.getRhs());
    if (!a || !rhs || *rhs <= 0) return reject();
    return accept(a.floorDiv(*rhs));
  }
  if (auto div = dyn_cast<arith::DivUIOp>(def)) {
    AffineExpr a = build(div.getLhs(), S);
    auto rhs = matchConstantInt(div.getRhs());
    if (!a || !rhs || *rhs <= 0) return reject();
    return accept(a.floorDiv(*rhs));
  }
  if (auto rem = dyn_cast<arith::RemSIOp>(def)) {
    AffineExpr a = build(rem.getLhs(), S);
    auto rhs = matchConstantInt(rem.getRhs());
    if (!a || !rhs || *rhs <= 0) return reject();
    return accept(a % *rhs);
  }
  if (auto rem = dyn_cast<arith::RemUIOp>(def)) {
    AffineExpr a = build(rem.getLhs(), S);
    auto rhs = matchConstantInt(rem.getRhs());
    if (!a || !rhs || *rhs <= 0) return reject();
    return accept(a % *rhs);
  }

  // shli with constant RHS = mul by 2^k.
  if (auto shl = dyn_cast<arith::ShLIOp>(def)) {
    auto rhs = matchConstantInt(shl.getRhs());
    if (!rhs || *rhs < 0 || *rhs >= 63) return reject();
    AffineExpr a = build(shl.getLhs(), S);
    if (!a) return reject();
    int64_t scale = int64_t(1) << *rhs;
    return accept(a * scale);
  }
  // shrsi / shrui with constant RHS = floordiv by 2^k.
  if (auto shr = dyn_cast<arith::ShRSIOp>(def)) {
    auto rhs = matchConstantInt(shr.getRhs());
    if (!rhs || *rhs < 0 || *rhs >= 63) return reject();
    AffineExpr a = build(shr.getLhs(), S);
    if (!a) return reject();
    int64_t div = int64_t(1) << *rhs;
    return accept(a.floorDiv(div));
  }
  if (auto shr = dyn_cast<arith::ShRUIOp>(def)) {
    auto rhs = matchConstantInt(shr.getRhs());
    if (!rhs || *rhs < 0 || *rhs >= 63) return reject();
    AffineExpr a = build(shr.getLhs(), S);
    if (!a) return reject();
    int64_t div = int64_t(1) << *rhs;
    return accept(a.floorDiv(div));
  }
  // andi with mask 2^k - 1 = mod 2^k.
  if (auto andOp = dyn_cast<arith::AndIOp>(def)) {
    auto rhs = matchConstantInt(andOp.getRhs());
    if (!rhs) {
      // Try the other side.
      auto lhs = matchConstantInt(andOp.getLhs());
      if (!lhs) return reject();
      auto k = matchPowerOfTwoMinusOne(*lhs);
      if (!k) return reject();
      AffineExpr a = build(andOp.getRhs(), S);
      if (!a) return reject();
      int64_t mod = int64_t(1) << *k;
      return accept(a % mod);
    }
    auto k = matchPowerOfTwoMinusOne(*rhs);
    if (!k) return reject();
    AffineExpr a = build(andOp.getLhs(), S);
    if (!a) return reject();
    int64_t mod = int64_t(1) << *k;
    return accept(a % mod);
  }

  // ori, xori with no power-of-2 trick: not affine.
  // Anything else: not affine — but if every operand is loop-invariant, the
  // whole result is invariant and we can bind it as a symbol.
  llvm::DenseMap<Value, bool> invCache;
  if (isLoopInvariant(v, S.iv, invCache))
    return accept(S.getOrCreateSymbol(v));

  return reject();
}

}  // namespace

std::optional<AffineExtractResult>
tryBuildAffineExpr(Value v, Value iv, MLIRContext *ctx) {
  ExtractState S;
  S.ctx = ctx;
  S.iv = iv;
  AffineExpr e = build(v, S);
  if (!e) return std::nullopt;
  return AffineExtractResult{e, std::move(S.symbols)};
}

}  // namespace mlir::lego
