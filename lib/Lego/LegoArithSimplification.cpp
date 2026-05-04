#define GEN_PASS_DEF_LEGOARITHSIMPLIFICATIONPASS
#define GEN_PASS_DEF_LEGOSTRENGTHREDUCTIONPASS
#include "Lego/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

// ============================================================================
// Shared matchers — extract the common structure from div/rem pattern pairs.
// ============================================================================

/// Match `addi(muli(q, d), r)` where one mul operand equals `divisor`.
/// Returns (q, r) on success.  Handles commutativity of both addi and muli.
static std::optional<std::pair<Value, Value>>
matchQTimesDPlusR(Value numerator, Value divisor) {
  auto addOp = numerator.getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return std::nullopt;

  Value terms[2] = {addOp.getLhs(), addOp.getRhs()};
  for (int i = 0; i < 2; ++i) {
    auto mulOp = terms[i].getDefiningOp<arith::MulIOp>();
    if (!mulOp)
      continue;
    if (mulOp.getLhs() == divisor)
      return std::make_pair(mulOp.getRhs(), terms[1 - i]);
    if (mulOp.getRhs() == divisor)
      return std::make_pair(mulOp.getLhs(), terms[1 - i]);
  }
  return std::nullopt;
}

/// In `addi(muli(q, s), r)` with divisor `muli(k, s)`, find the shared
/// factor `s` between the numerator term and divisor.
/// Returns (q, s, k, r) on success.
struct SharedFactorMatch {
  Value q, s, k, r;
};

static std::optional<SharedFactorMatch>
matchSharedFactor(Value numerator, Value divisor) {
  auto addOp = numerator.getDefiningOp<arith::AddIOp>();
  auto divMul = divisor.getDefiningOp<arith::MulIOp>();
  if (!addOp || !divMul)
    return std::nullopt;

  Value addTerms[2] = {addOp.getLhs(), addOp.getRhs()};
  Value dm[2] = {divMul.getLhs(), divMul.getRhs()};

  for (int i = 0; i < 2; ++i) {
    auto termMul = addTerms[i].getDefiningOp<arith::MulIOp>();
    if (!termMul)
      continue;

    Value tm[2] = {termMul.getLhs(), termMul.getRhs()};
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b)
        if (tm[a] == dm[b] && tm[a] != divisor) {
          Value q = tm[1 - a], s = tm[a], k = dm[1 - b];
          // Guard: don't fire if q is already remui(_, k) (prevents loops).
          if (auto rem = q.getDefiningOp<arith::RemUIOp>())
            if (rem.getRhs() == k)
              continue;
          return SharedFactorMatch{q, s, k, addTerms[1 - i]};
        }
  }
  return std::nullopt;
}

/// Match the mixed-radix structure:
///   numerator = addi(muli(remui(a, n), m), remui(b, m))
///   divisor   = muli(n, m)
/// Returns true on match (the sum is provably < divisor).
static bool matchMixedRadixBound(Value numerator, Value divisor) {
  auto divMul = divisor.getDefiningOp<arith::MulIOp>();
  auto addOp = numerator.getDefiningOp<arith::AddIOp>();
  if (!divMul || !addOp)
    return false;

  Value nc[2] = {divMul.getLhs(), divMul.getRhs()};
  Value at[2] = {addOp.getLhs(), addOp.getRhs()};

  for (int d = 0; d < 2; ++d) {
    Value n = nc[d], m = nc[1 - d];
    for (int t = 0; t < 2; ++t) {
      auto hiMul = at[t].getDefiningOp<arith::MulIOp>();
      if (!hiMul)
        continue;
      Value mo[2] = {hiMul.getLhs(), hiMul.getRhs()};
      for (int mi = 0; mi < 2; ++mi) {
        if (mo[mi] != m)
          continue;
        auto remA = mo[1 - mi].getDefiningOp<arith::RemUIOp>();
        if (!remA || remA.getRhs() != n)
          continue;
        auto remB = at[1 - t].getDefiningOp<arith::RemUIOp>();
        if (!remB || remB.getRhs() != m)
          continue;
        return true;
      }
    }
  }
  return false;
}

// ============================================================================
// Patterns — thin wrappers over the shared matchers.
// ============================================================================

// (q * d + r) % d  →  r % d
struct SimplifyRemId : public OpRewritePattern<arith::RemUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::RemUIOp op,
                                PatternRewriter &rewriter) const override {
    if (auto m = matchQTimesDPlusR(op.getLhs(), op.getRhs())) {
      rewriter.replaceOpWithNewOp<arith::RemUIOp>(op, m->second, op.getRhs());
      return success();
    }
    return failure();
  }
};

// (q * d + r) / d  →  q + r / d
struct SimplifyDivId : public OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    if (auto m = matchQTimesDPlusR(op.getLhs(), op.getRhs())) {
      auto [q, r] = *m;
      Value rDivD = arith::DivUIOp::create(rewriter, op.getLoc(), r, op.getRhs());
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, q, rDivD);
      return success();
    }
    return failure();
  }
};

// (x + c) / d  →  x/d + c/d   when c % d == 0
struct SimplifyDivConst : public OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    APInt dVal;
    if (!matchPattern(op.getRhs(), m_ConstantInt(&dVal)) || dVal.isZero())
      return failure();
    auto addOp = op.getLhs().getDefiningOp<arith::AddIOp>();
    if (!addOp)
      return failure();

    Value x;
    APInt cVal;
    if (matchPattern(addOp.getRhs(), m_ConstantInt(&cVal)))
      x = addOp.getLhs();
    else if (matchPattern(addOp.getLhs(), m_ConstantInt(&cVal)))
      x = addOp.getRhs();
    else
      return failure();
    if (cVal.urem(dVal) != 0)
      return failure();

    Value newDiv = arith::DivUIOp::create(rewriter, op.getLoc(), x, op.getRhs());
    Value newConst = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIndexAttr(cVal.udiv(dVal).getZExtValue()));
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, newDiv, newConst);
    return success();
  }
};

// (x / d) * d + (x % d)  →  x
struct ReconstructId : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    auto check = [](Value maybeMul, Value maybeRem) -> Value {
      auto mulOp = maybeMul.getDefiningOp<arith::MulIOp>();
      auto remOp = maybeRem.getDefiningOp<arith::RemUIOp>();
      if (!mulOp || !remOp)
        return nullptr;
      Value x = remOp.getLhs(), d = remOp.getRhs();
      auto tryDiv = [&](Value v, Value other) -> bool {
        if (other != d) return false;
        auto div = v.getDefiningOp<arith::DivUIOp>();
        return div && div.getLhs() == x && div.getRhs() == d;
      };
      if (tryDiv(mulOp.getLhs(), mulOp.getRhs()) ||
          tryDiv(mulOp.getRhs(), mulOp.getLhs()))
        return x;
      return nullptr;
    };
    if (Value r = check(op.getLhs(), op.getRhs())) {
      rewriter.replaceOp(op, r); return success();
    }
    if (Value r = check(op.getRhs(), op.getLhs())) {
      rewriter.replaceOp(op, r); return success();
    }
    return failure();
  }
};

// (x % d) / d  →  0
struct SimplifyDivOfRem : public OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    auto rem = op.getLhs().getDefiningOp<arith::RemUIOp>();
    if (!rem || rem.getRhs() != op.getRhs())
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
    return success();
  }
};

// (x % d) % d  →  x % d
struct SimplifyRemOfRem : public OpRewritePattern<arith::RemUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::RemUIOp op,
                                PatternRewriter &rewriter) const override {
    auto inner = op.getLhs().getDefiningOp<arith::RemUIOp>();
    if (!inner || inner.getRhs() != op.getRhs())
      return failure();
    rewriter.replaceOp(op, op.getLhs());
    return success();
  }
};

// ---- Extended: shared-factor decomposition (div/rem pair) -----------------
//
//  (q*s + r) / (k*s)  →  q/k + ((q%k)*s + r) / (k*s)
//  (q*s + r) % (k*s)  →  ((q%k)*s + r) % (k*s)

static Value buildLowPart(PatternRewriter &rewriter, Location loc,
                           const SharedFactorMatch &m, Value divisor) {
  Value qRemK = arith::RemUIOp::create(rewriter, loc, m.q, m.k);
  Value low = arith::MulIOp::create(rewriter, loc, qRemK, m.s);
  return arith::AddIOp::create(rewriter, loc, low, m.r);
}

struct ExtendedSimplifyDivId : public OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    auto m = matchSharedFactor(op.getLhs(), op.getRhs());
    if (!m) return failure();
    Location loc = op.getLoc();
    Value qDivK = arith::DivUIOp::create(rewriter, loc, m->q, m->k);
    Value lowDiv = arith::DivUIOp::create(
        rewriter, loc, buildLowPart(rewriter, loc, *m, op.getRhs()), op.getRhs());
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, qDivK, lowDiv);
    return success();
  }
};

struct ExtendedSimplifyRemId : public OpRewritePattern<arith::RemUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::RemUIOp op,
                                PatternRewriter &rewriter) const override {
    auto m = matchSharedFactor(op.getLhs(), op.getRhs());
    if (!m) return failure();
    rewriter.replaceOpWithNewOp<arith::RemUIOp>(
        op, buildLowPart(rewriter, op.getLoc(), *m, op.getRhs()), op.getRhs());
    return success();
  }
};

// ---- Mixed-radix bound (div/rem pair) ------------------------------------
//
//  divui(remui(a,n)*m + remui(b,m), n*m)  →  0
//  remui(remui(a,n)*m + remui(b,m), n*m)  →  remui(a,n)*m + remui(b,m)

struct SimplifyMixedRadixDiv : public OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    if (!matchMixedRadixBound(op.getLhs(), op.getRhs()))
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
    return success();
  }
};

struct SimplifyMixedRadixRem : public OpRewritePattern<arith::RemUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::RemUIOp op,
                                PatternRewriter &rewriter) const override {
    if (!matchMixedRadixBound(op.getLhs(), op.getRhs()))
      return failure();
    rewriter.replaceOp(op, op.getLhs());
    return success();
  }
};

// ---- Distributive law: muli(a,c) + muli(b,c) → muli(addi(a,b), c) --------
//
// Factors out a shared multiplicand from two addends.  Handles commutativity
// of both addi and muli.

struct DistributiveFactor : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    auto mulL = op.getLhs().getDefiningOp<arith::MulIOp>();
    auto mulR = op.getRhs().getDefiningOp<arith::MulIOp>();
    if (!mulL || !mulR)
      return failure();

    // Try all combinations: mulL = a*c, mulR = b*c (c shared)
    Value ml[2] = {mulL.getLhs(), mulL.getRhs()};
    Value mr[2] = {mulR.getLhs(), mulR.getRhs()};
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        if (ml[i] == mr[j]) {
          Value c = ml[i], a = ml[1 - i], b = mr[1 - j];
          Value sum = arith::AddIOp::create(rewriter, op.getLoc(), a, b);
          rewriter.replaceOpWithNewOp<arith::MulIOp>(op, sum, c);
          return success();
        }
    return failure();
  }
};

// ============================================================================
// Pass
// ============================================================================

struct LegoArithSimplificationPass
    : public mlir::lego::impl::LegoArithSimplificationPassBase<
          LegoArithSimplificationPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SimplifyRemId, SimplifyDivId, SimplifyDivConst,
                 ReconstructId, SimplifyDivOfRem, SimplifyRemOfRem,
                 ExtendedSimplifyDivId, ExtendedSimplifyRemId,
                 SimplifyMixedRadixDiv, SimplifyMixedRadixRem,
                 DistributiveFactor>(
                     &getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// ============================================================================
// Strength Reduction Pass (runs after main simplification)
//
// Converts power-of-2 divui/remui to shift/mask operations.
// Kept separate from the algebraic simplification pass because it
// would interfere with div/rem pattern matchers in the fixed-point loop.
// ============================================================================

/// If `value` is a constant equal to a power of 2, return log2.
static std::optional<unsigned> matchPowerOfTwo(Value value) {
  APInt val;
  if (!matchPattern(value, m_ConstantInt(&val)))
    return std::nullopt;
  if (val.isPowerOf2())
    return val.exactLogBase2();
  return std::nullopt;
}

/// Recognize a value as provably-nonnegative without a full value-range
/// analysis.  Used to gate strength-reduction of signed div/rem by powers of
/// two: ``divsi(x, 2^k)`` equals ``shrsi(x, k)`` only when ``x >= 0``.
///
/// We accept block arguments of `lego.gen_p` apply/inv regions (these are
/// layout indices by construction), `scf.for` IVs whose lower bound is a
/// nonneg constant, results of unsigned ops (``divui``/``remui``/``shrui``),
/// constants ``>= 0``, and recursively closed-form combinations through
/// addi/muli/andi (preserving nonneg).
static bool isProvablyNonNegative(Value v, unsigned depth = 0) {
  if (depth > 8) return false;             // avoid pathological chains
  if (!v) return false;
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    APInt iv;
    if (matchPattern(cst.getOperation(), m_ConstantInt(&iv)))
      return !iv.isNegative();
  }
  // Block-argument cases: lego.gen_p apply/inv region or scf.for IV.
  if (auto bArg = dyn_cast<BlockArgument>(v)) {
    Operation *parent = bArg.getOwner()->getParentOp();
    if (!parent) return false;
    StringRef name = parent->getName().getStringRef();
    if (name == "lego.gen_p" || name == "lego.reg_p")
      return true;                         // layout index args
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (bArg == forOp.getInductionVar())
        return isProvablyNonNegative(forOp.getLowerBound(), depth + 1);
    }
    return false;
  }
  // Unsigned ops always produce nonneg results.
  if (v.getDefiningOp<arith::DivUIOp>() ||
      v.getDefiningOp<arith::RemUIOp>() ||
      v.getDefiningOp<arith::ShRUIOp>())
    return true;
  // Bitwise AND with a nonneg operand is bounded above by it ⇒ nonneg.
  if (auto andOp = v.getDefiningOp<arith::AndIOp>())
    return isProvablyNonNegative(andOp.getLhs(), depth + 1) ||
           isProvablyNonNegative(andOp.getRhs(), depth + 1);
  // Closed-form: addi/muli/shli over nonneg operands stay nonneg.
  if (auto addOp = v.getDefiningOp<arith::AddIOp>())
    return isProvablyNonNegative(addOp.getLhs(), depth + 1) &&
           isProvablyNonNegative(addOp.getRhs(), depth + 1);
  if (auto mulOp = v.getDefiningOp<arith::MulIOp>())
    return isProvablyNonNegative(mulOp.getLhs(), depth + 1) &&
           isProvablyNonNegative(mulOp.getRhs(), depth + 1);
  if (auto shlOp = v.getDefiningOp<arith::ShLIOp>())
    return isProvablyNonNegative(shlOp.getLhs(), depth + 1);
  return false;
}

struct StrengthReduceDiv : public OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    auto log2 = matchPowerOfTwo(op.getRhs());
    if (!log2 || *log2 == 0) // skip divide-by-1
      return failure();
    Value shift = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIndexAttr(*log2));
    rewriter.replaceOpWithNewOp<arith::ShRUIOp>(op, op.getLhs(), shift);
    return success();
  }
};

struct StrengthReduceRem : public OpRewritePattern<arith::RemUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::RemUIOp op,
                                PatternRewriter &rewriter) const override {
    auto log2 = matchPowerOfTwo(op.getRhs());
    if (!log2 || *log2 == 0) // skip mod-by-1
      return failure();
    uint64_t maskValue = (1ULL << *log2) - 1;
    Value mask = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIndexAttr(maskValue));
    rewriter.replaceOpWithNewOp<arith::AndIOp>(op, op.getLhs(), mask);
    return success();
  }
};

// divsi(x, 2^k) → shrui(x, k)   when x is provably nonneg.
struct StrengthReduceDivSI : public OpRewritePattern<arith::DivSIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivSIOp op,
                                PatternRewriter &rewriter) const override {
    auto log2 = matchPowerOfTwo(op.getRhs());
    if (!log2 || *log2 == 0)
      return failure();
    if (!isProvablyNonNegative(op.getLhs()))
      return failure();
    Value shift = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIndexAttr(*log2));
    rewriter.replaceOpWithNewOp<arith::ShRUIOp>(op, op.getLhs(), shift);
    return success();
  }
};

// remsi(x, 2^k) → andi(x, mask)   when x is provably nonneg.
struct StrengthReduceRemSI : public OpRewritePattern<arith::RemSIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::RemSIOp op,
                                PatternRewriter &rewriter) const override {
    auto log2 = matchPowerOfTwo(op.getRhs());
    if (!log2 || *log2 == 0)
      return failure();
    if (!isProvablyNonNegative(op.getLhs()))
      return failure();
    uint64_t maskValue = (1ULL << *log2) - 1;
    Value mask = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIndexAttr(maskValue));
    rewriter.replaceOpWithNewOp<arith::AndIOp>(op, op.getLhs(), mask);
    return success();
  }
};

// muli(x, 2^k) → shli(x, k)   (handles commutativity)
struct StrengthReduceMul : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    // Try both operands (muli is commutative)
    for (int side = 0; side < 2; ++side) {
      Value constOp = side == 0 ? op.getRhs() : op.getLhs();
      Value other   = side == 0 ? op.getLhs() : op.getRhs();
      auto log2 = matchPowerOfTwo(constOp);
      if (!log2 || *log2 == 0) // skip multiply-by-1
        continue;
      Value shift = arith::ConstantOp::create(
          rewriter, op.getLoc(),
          rewriter.getIndexAttr(*log2));
      rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, other, shift);
      return success();
    }
    return failure();
  }
};

struct LegoStrengthReductionPass
    : public mlir::lego::impl::LegoStrengthReductionPassBase<
          LegoStrengthReductionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<StrengthReduceDiv, StrengthReduceRem,
                 StrengthReduceDivSI, StrengthReduceRemSI,
                 StrengthReduceMul>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoArithSimplificationPass() {
  return std::make_unique<LegoArithSimplificationPass>();
}
std::unique_ptr<Pass> createLegoStrengthReductionPass() {
  return std::make_unique<LegoStrengthReductionPass>();
}
} // namespace lego
} // namespace mlir
