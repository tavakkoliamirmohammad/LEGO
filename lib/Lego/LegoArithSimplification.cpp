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

// ============================================================================
// Bit-spread recognition → portable log-time bit-magic
//
// Recognizes the per-bit form of a bit-spread (e.g., the apply body of a
// Morton/Z-order GenP after strength reduction):
//
//   sum_k  shli( andi( shrui(x_g, K_g_k), 1 ), S_g_k )
//
// across one or more source values (groups). For each group g:
//   * source bits K_g_k are 0, 1, 2, ..., n_g - 1 (contiguous, ascending)
//   * destination positions S_g_k are offset_g + 2 * k (stride-2 spread,
//     i.e., 2-D Morton — the most common case)
//   * offset_g ∈ {0, 1}
//
// When matched the addi tree is replaced by a portable 4-stage bit-magic
// spread per source, OR'd together at the end:
//
//   y = x & ((1 << n) - 1)
//   y = (y | (y << 8)) & 0x00FF00FF        (when n > 8)
//   y = (y | (y << 4)) & 0x0F0F0F0F        (when n > 4)
//   y = (y | (y << 2)) & 0x33333333        (when n > 2)
//   y = (y | (y << 1)) & 0x55555555
//   if offset == 1:  y = y << 1
//
// vs the per-bit form: 4*n_g ops per source becomes ≈ 12 ops per source
// (independent of n_g for n_g ≤ 16). For 2-D Morton with n=10 this is a
// ~3-4× reduction in per-access arithmetic. Pure ``arith`` ops — portable
// across x86, ARM, GPU.
//
// Soundness: the rewrite is bit-exact for non-negative x with x < (1<<n).
// Strength reduction (above) gates on ``isProvablyNonNegative`` for the
// upstream div/rem→shift conversions, so by the time this pattern fires
// every shrui/andi has the right semantics. The ``& ((1<<n)-1)`` head of
// the bit-magic chain truncates any high garbage explicitly.
// ============================================================================

struct BitTerm { Value src; unsigned K; unsigned S; };

static std::optional<uint64_t> matchConstUInt(Value v) {
  APInt a;
  if (matchPattern(v, m_ConstantInt(&a))) {
    if (a.isNonNegative()) return a.getZExtValue();
  }
  return std::nullopt;
}

/// If a constant is a power of 2, return log2; else nullopt.
static std::optional<unsigned> constLog2(Value v) {
  auto c = matchConstUInt(v);
  if (!c || *c == 0) return std::nullopt;
  if ((*c & (*c - 1)) != 0) return std::nullopt;     // not power of 2
  return static_cast<unsigned>(llvm::Log2_64(*c));
}

/// Match a value as "bit 0 extraction": ``andi(x, 1)`` OR ``remsi(x, 2)`` OR
/// ``remui(x, 2)``.  Returns the inner ``x`` on success.
static std::optional<Value> matchBitZero(Value v) {
  if (auto andOp = v.getDefiningOp<arith::AndIOp>()) {
    if (auto m = matchConstUInt(andOp.getRhs()); m && *m == 1ULL)
      return andOp.getLhs();
    if (auto m = matchConstUInt(andOp.getLhs()); m && *m == 1ULL)
      return andOp.getRhs();
    return std::nullopt;
  }
  if (auto rem = v.getDefiningOp<arith::RemUIOp>()) {
    if (auto m = matchConstUInt(rem.getRhs()); m && *m == 2ULL)
      return rem.getLhs();
    return std::nullopt;
  }
  if (auto rem = v.getDefiningOp<arith::RemSIOp>()) {
    if (auto m = matchConstUInt(rem.getRhs()); m && *m == 2ULL)
      return rem.getLhs();
    return std::nullopt;
  }
  return std::nullopt;
}

/// Match a single "shift right by k" step: ``shrui(x, k)`` OR
/// ``divui(x, 2^k)`` OR ``divsi(x, 2^k)``. Returns (x, k) on success.
static std::optional<std::pair<Value, unsigned>> matchShiftRightOne(Value v) {
  if (auto shr = v.getDefiningOp<arith::ShRUIOp>()) {
    auto k = matchConstUInt(shr.getRhs());
    if (!k) return std::nullopt;
    return std::make_pair(shr.getLhs(), static_cast<unsigned>(*k));
  }
  if (auto dv = v.getDefiningOp<arith::DivUIOp>()) {
    auto k = constLog2(dv.getRhs());
    if (!k) return std::nullopt;
    return std::make_pair(dv.getLhs(), *k);
  }
  if (auto dv = v.getDefiningOp<arith::DivSIOp>()) {
    auto k = constLog2(dv.getRhs());
    if (!k) return std::nullopt;
    return std::make_pair(dv.getLhs(), *k);
  }
  return std::nullopt;
}

/// Walk a chain of shift-right (shrui / divui / divsi) operations,
/// accumulating the total shift amount.  Returns (deepest_source, total_k).
/// If ``v`` itself is not a shift, returns (v, 0) — distinguishable from
/// a no-op only by the K value.
static std::pair<Value, unsigned> matchShiftRightChain(Value v) {
  unsigned K = 0;
  while (true) {
    auto step = matchShiftRightOne(v);
    if (!step) break;
    K += step->second;
    v = step->first;
  }
  return std::make_pair(v, K);
}

/// Match a "shift left by k" operation: ``shli(x, k)`` OR ``muli(x, 2^k)``.
/// Returns (x, k) on success.
static std::optional<std::pair<Value, unsigned>> matchShiftLeft(Value v) {
  if (auto shl = v.getDefiningOp<arith::ShLIOp>()) {
    auto k = matchConstUInt(shl.getRhs());
    if (!k) return std::nullopt;
    return std::make_pair(shl.getLhs(), static_cast<unsigned>(*k));
  }
  if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
    if (auto k = constLog2(mul.getRhs()))
      return std::make_pair(mul.getLhs(), *k);
    if (auto k = constLog2(mul.getLhs()))
      return std::make_pair(mul.getRhs(), *k);
  }
  return std::nullopt;
}

static std::optional<BitTerm> matchBitTerm(Value v) {
  unsigned S = 0;
  // Outer placement: shli(x, S) or muli(x, 2^S).  Optional (S=0).
  if (auto sl = matchShiftLeft(v)) {
    S = sl->second;
    v = sl->first;
  }
  // Bit-zero extraction: andi(_, 1), remsi(_, 2), or remui(_, 2).
  auto z = matchBitZero(v);
  if (!z) return std::nullopt;
  Value rest = *z;
  // Walk the full shift-right chain.  Canonicalize/CSE may compose multiple
  // shifts into nested ops (e.g. ``shrui(shrui(flat, 6), k)`` for the i-bits
  // of a flat-merged Morton index after lego-to-arith); we want the deepest
  // non-shift source so that all bit-extracts on the same logical coordinate
  // group together.
  auto [src, K] = matchShiftRightChain(rest);
  return BitTerm{src, K, S};
}

static void flattenAddTree(Value v, SmallVectorImpl<Value> &leaves) {
  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    flattenAddTree(add.getLhs(), leaves);
    flattenAddTree(add.getRhs(), leaves);
  } else {
    leaves.push_back(v);
  }
}

/// Emit one stage of the bit-magic spread:
///   y = (y | (y << shift)) & mask
static Value emitSpreadStage(OpBuilder &b, Location loc, Value y, Type ty,
                             unsigned shift, uint64_t mask) {
  Value sft = arith::ConstantOp::create(b, loc, ty, b.getIntegerAttr(ty, shift));
  Value shifted = arith::ShLIOp::create(b, loc, y, sft);
  Value ored = arith::OrIOp::create(b, loc, y, shifted);
  Value mskC = arith::ConstantOp::create(b, loc, ty, b.getIntegerAttr(ty, mask));
  return arith::AndIOp::create(b, loc, ored, mskC);
}

/// Emit the Hacker's Delight bit-spread for ``stride``-d Morton, truncated
/// to ``nbits`` input bits.  Returns a Value of type ``ty`` whose low
/// stride*nbits bits hold the spread result with input bit ``i`` placed
/// at output position ``stride * i``.
///
/// Stride 2 (2-D Morton) and stride 3 (3-D Morton) are supported.  The
/// stage masks for each stride are derived from Hacker's Delight 4.1 §7-2
/// (delta-swap form for stride-2) and §7.1.3 (general perfect shuffle for
/// stride > 2) — see comments below.
static Value emitMortonSpread(OpBuilder &b, Location loc, Value src, Type ty,
                              unsigned nbits, unsigned stride = 2) {
  // Truncate input to nbits.
  uint64_t headMask = (nbits >= 64) ? ~0ULL : ((1ULL << nbits) - 1ULL);
  Value y = src;
  if (y.getType() != ty) {
    if (isa<IndexType>(y.getType()) && isa<IntegerType>(ty))
      y = arith::IndexCastOp::create(b, loc, ty, y);
    else if (isa<IntegerType>(y.getType()) && isa<IndexType>(ty))
      y = arith::IndexCastOp::create(b, loc, ty, y);
  }
  Value mskHead = arith::ConstantOp::create(
      b, loc, ty, b.getIntegerAttr(ty, headMask));
  y = arith::AndIOp::create(b, loc, y, mskHead);

  if (stride == 2) {
    // 2-D Morton.  Spread input bit i → output position 2i.
    if (nbits > 8) y = emitSpreadStage(b, loc, y, ty, 8, 0x00FF00FFULL);
    if (nbits > 4) y = emitSpreadStage(b, loc, y, ty, 4, 0x0F0F0F0FULL);
    if (nbits > 2) y = emitSpreadStage(b, loc, y, ty, 2, 0x33333333ULL);
    y = emitSpreadStage(b, loc, y, ty, 1, 0x55555555ULL);
  } else {
    assert(stride == 3 && "emitMortonSpread: only stride 2 / 3 supported");
    // 3-D Morton.  Spread input bit i → output position 3i.
    // Constants are the standard libmorton-style 32-bit stage masks for
    // up to 10-bit input → 30-bit output (covers integer indices < 2^10).
    if (nbits > 8) y = emitSpreadStage(b, loc, y, ty, 16, 0xFF0000FFULL);
    if (nbits > 4) y = emitSpreadStage(b, loc, y, ty,  8, 0x0F00F00FULL);
    if (nbits > 2) y = emitSpreadStage(b, loc, y, ty,  4, 0xC30C30C3ULL);
    y = emitSpreadStage(b, loc, y, ty,  2, 0x49249249ULL);
  }
  return y;
}

struct RecognizeBitSpread : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    // Only fire on the *root* of an addi tree (no addi user).
    for (Operation *user : op->getUsers())
      if (isa<arith::AddIOp>(user))
        return failure();

    SmallVector<Value> leaves;
    flattenAddTree(op.getResult(), leaves);
    if (leaves.size() < 4) return failure();      // tiny trees aren't worth it.

    // Decompose every leaf into a bit-extract-and-place term.
    SmallVector<BitTerm, 32> terms;
    terms.reserve(leaves.size());
    for (Value leaf : leaves) {
      auto bt = matchBitTerm(leaf);
      if (!bt) return failure();
      terms.push_back(*bt);
    }

    // Group by source.
    llvm::SmallDenseMap<Value, SmallVector<BitTerm, 16>, 4> groups;
    for (auto &t : terms) groups[t.src].push_back(t);
    if (groups.empty()) return failure();

    struct GroupInfo {
      Value src;
      unsigned nbits;        // number of contiguous bits being spread
      unsigned offset;       // S_min: which Morton lane (0..stride-1)
      unsigned stride;       // 2 for 2-D Morton, 3 for 3-D Morton
      unsigned k_min;        // pre-shift amount applied to src before spread
    };
    SmallVector<GroupInfo, 4> infos;
    infos.reserve(groups.size());
    for (auto &kv : groups) {
      auto &grp = kv.second;
      std::sort(grp.begin(), grp.end(),
                [](const BitTerm &a, const BitTerm &b) { return a.K < b.K; });
      if (grp.empty()) return failure();
      // K = k_min, k_min+1, ..., k_min+n-1 (contiguous from some k_min ≥ 0).
      unsigned k_min = grp[0].K;
      for (size_t i = 0; i < grp.size(); ++i)
        if (grp[i].K != k_min + static_cast<unsigned>(i))
          return failure();
      // Detect stride from the S sequence.  S = offset + stride*i with
      // stride ∈ {2, 3} (2-D / 3-D Morton).  Larger strides are rejected
      // so the matcher is not over-eager on unrelated bit-spread patterns.
      unsigned offset = grp[0].S;
      unsigned stride = 2;     // default if grp.size() == 1
      if (grp.size() >= 2) {
        if (grp[1].S <= grp[0].S) return failure();
        stride = grp[1].S - grp[0].S;
      }
      if (stride < 2 || stride > 3) return failure();
      if (offset >= stride) return failure();
      for (size_t i = 0; i < grp.size(); ++i)
        if (grp[i].S != offset + stride * static_cast<unsigned>(i))
          return failure();
      unsigned nbits = static_cast<unsigned>(grp.size());
      if (nbits == 0 || nbits > 16) return failure();
      // Stride-3 emit currently covers up to 10-bit input (32-bit output).
      if (stride == 3 && nbits > 10) return failure();
      infos.push_back({kv.first, nbits, offset, stride, k_min});
    }

    // All checks passed — emit the bit-magic for each group, OR together.
    Location loc = op.getLoc();
    Type resTy = op.getType();
    Value acc;
    for (auto &g : infos) {
      // Pre-shift source by k_min if needed (e.g. if the per-bit form was
      // composed with an outer shrui by `k_min` during canonicalize).
      Value src = g.src;
      if (g.k_min > 0) {
        Value kc = arith::ConstantOp::create(
            rewriter, loc, resTy, rewriter.getIntegerAttr(resTy, g.k_min));
        if (src.getType() != resTy) {
          if (isa<IndexType>(src.getType()) && isa<IntegerType>(resTy))
            src = arith::IndexCastOp::create(rewriter, loc, resTy, src);
          else if (isa<IntegerType>(src.getType()) && isa<IndexType>(resTy))
            src = arith::IndexCastOp::create(rewriter, loc, resTy, src);
        }
        src = arith::ShRUIOp::create(rewriter, loc, src, kc);
      }
      Value spread =
          emitMortonSpread(rewriter, loc, src, resTy, g.nbits, g.stride);
      // Shift the lane (lane 0 → no shift, lane k → << k).
      if (g.offset > 0) {
        Value sh = arith::ConstantOp::create(
            rewriter, loc, resTy, rewriter.getIntegerAttr(resTy, g.offset));
        spread = arith::ShLIOp::create(rewriter, loc, spread, sh);
      }
      acc = acc ? arith::OrIOp::create(rewriter, loc, acc, spread).getResult()
                : spread;
    }
    rewriter.replaceOp(op, acc);
    return success();
  }
};

struct LegoStrengthReductionPass
    : public mlir::lego::impl::LegoStrengthReductionPassBase<
          LegoStrengthReductionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<StrengthReduceDiv, StrengthReduceRem,
                 StrengthReduceDivSI, StrengthReduceRemSI,
                 StrengthReduceMul, RecognizeBitSpread>(&getContext());
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
