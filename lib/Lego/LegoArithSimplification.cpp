#define GEN_PASS_DEF_LEGOARITHSIMPLIFICATIONPASS
#include "Lego/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

// Pattern A1: (q * d + r) % d -> r % d
struct SimplifyRemId : public OpRewritePattern<arith::RemUIOp> {
  using OpRewritePattern<arith::RemUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::RemUIOp op,
                                PatternRewriter &rewriter) const override {
    Value numerator = op.getLhs();
    Value divisor = op.getRhs();

    // Match numerator = addi(..., ...)
    auto addOp = numerator.getDefiningOp<arith::AddIOp>();
    if (!addOp)
      return failure();

    Value terms[2] = {addOp.getLhs(), addOp.getRhs()};

    // Check if either term is (q * d)
    for (int i = 0; i < 2; ++i) {
      if (auto mulOp = terms[i].getDefiningOp<arith::MulIOp>()) {
        Value mulLhs = mulOp.getLhs();
        Value mulRhs = mulOp.getRhs();

        // Check if mul operand matches divisor
        if (mulLhs == divisor || mulRhs == divisor) {
          // Found (q * d + r). Rewrite to r % d.
          Value r = terms[1 - i];
          rewriter.replaceOpWithNewOp<arith::RemUIOp>(op, r, divisor);
          return success();
        }
      }
    }

    return failure();
  }
};

// Pattern A2: (q * d + r) / d -> q + (r / d)
// Since we are in unsigned arithmetic, if r < d, (r/d) -> 0, result -> q.
struct SimplifyDivId : public OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern<arith::DivUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    Value numerator = op.getLhs();
    Value divisor = op.getRhs();

    // Match numerator = addi(..., ...)
    auto addOp = numerator.getDefiningOp<arith::AddIOp>();
    if (!addOp)
      return failure();

    Value terms[2] = {addOp.getLhs(), addOp.getRhs()};

    for (int i = 0; i < 2; ++i) {
      Value term = terms[i];
      Value other = terms[1 - i];

      if (auto mulOp = term.getDefiningOp<arith::MulIOp>()) {
        Value mulLhs = mulOp.getLhs();
        Value mulRhs = mulOp.getRhs();

        if (mulLhs == divisor || mulRhs == divisor) {
          // Found (q * d + r) / d
          Value q = (mulLhs == divisor) ? mulRhs : mulLhs;
          
          // Result = q + (r / d)
          // We let subsequent canonicalizations handle (r / d) -> 0 if simplification is possible.
          Value rDivD = arith::DivUIOp::create(rewriter, op.getLoc(), other, divisor);
          rewriter.replaceOpWithNewOp<arith::AddIOp>(op, q, rDivD);
          return success();
        }
      }
    }
    return failure();
  }
};

// Pattern A6: (x + c) / d -> x/d + c/d  (if c % d == 0)
struct SimplifyDivConst : public OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern<arith::DivUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    Value numerator = op.getLhs();
    Value divisor = op.getRhs();

    // Match divisor being a constant
    APInt dVal;
    if (!matchPattern(divisor, m_ConstantInt(&dVal)) || dVal.isZero())
      return failure();

    // Match numerator = addi(x, c)
    auto addOp = numerator.getDefiningOp<arith::AddIOp>();
    if (!addOp)
      return failure();

    Value x;
    APInt cVal;
    

    // Check commutativity: x + c or c + x
    if (matchPattern(addOp.getRhs(), m_ConstantInt(&cVal))) {
      x = addOp.getLhs();
    } else if (matchPattern(addOp.getLhs(), m_ConstantInt(&cVal))) {
      x = addOp.getRhs();
    } else {
      return failure();
    }

    // Check if c is a multiple of d
    if (cVal.urem(dVal) != 0)
      return failure();

    // Rewrite to (x / d) + (c / d)
    Value newDiv = arith::DivUIOp::create(rewriter, op.getLoc(), x, divisor);
    Value newConst = arith::ConstantOp::create(rewriter, op.getLoc(), rewriter.getIndexAttr(cVal.udiv(dVal).getZExtValue()));
    
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, newDiv, newConst);
    return success();
  }
};

// Pattern B/C: (x / d) * d + (x % d) -> x
struct ReconstructId : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // We look for addi( mul(div(x, d), d), rem(x, d) )
    // Order doesn't matter.

    auto checkMatch = [&](Value maybeMul, Value maybeRem) -> Value {
      auto mulOp = maybeMul.getDefiningOp<arith::MulIOp>();
      auto remOp = maybeRem.getDefiningOp<arith::RemUIOp>();
      if (!mulOp || !remOp)
        return nullptr;

      Value x = remOp.getLhs();
      Value d = remOp.getRhs();

      // Check mul = div(x, d) * d
      Value mulLhs = mulOp.getLhs();
      Value mulRhs = mulOp.getRhs();

      auto checkDiv = [&](Value val, Value other) {
        if (other != d) return false;
        if (auto divOp = val.getDefiningOp<arith::DivUIOp>()) {
           return divOp.getLhs() == x && divOp.getRhs() == d;
        }
        return false;
      };

      if (checkDiv(mulLhs, mulRhs) || checkDiv(mulRhs, mulLhs)) {
        return x;
      }
      return nullptr;
    };

    if (Value res = checkMatch(lhs, rhs)) {
      rewriter.replaceOp(op, res);
      return success();
    }
    if (Value res = checkMatch(rhs, lhs)) {
       rewriter.replaceOp(op, res);
       return success();
    }

    return failure();
  }
};


// Pattern: divui(remui(x, d), d) -> 0
// Always true: x % d is in [0, d-1], so (x % d) / d = 0.
struct SimplifyDivOfRem : public OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern<arith::DivUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    Value numerator = op.getLhs();
    Value divisor = op.getRhs();

    auto remOp = numerator.getDefiningOp<arith::RemUIOp>();
    if (!remOp)
      return failure();

    if (remOp.getRhs() != divisor)
      return failure();

    // (x % d) / d -> 0
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
    return success();
  }
};

// Pattern: remui(remui(x, d), d) -> remui(x, d)
// Always true: x % d is in [0, d-1], so (x % d) % d = x % d.
struct SimplifyRemOfRem : public OpRewritePattern<arith::RemUIOp> {
  using OpRewritePattern<arith::RemUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::RemUIOp op,
                                PatternRewriter &rewriter) const override {
    Value numerator = op.getLhs();
    Value divisor = op.getRhs();

    auto innerRem = numerator.getDefiningOp<arith::RemUIOp>();
    if (!innerRem)
      return failure();

    if (innerRem.getRhs() != divisor)
      return failure();

    // (x % d) % d -> x % d
    rewriter.replaceOp(op, numerator);
    return success();
  }
};

// ============================================================================
// Extended SimplifyDivId: shared factor between numerator term and divisor.
//
//   divui(addi(muli(q, s), r), muli(k, s))
//     → addi(divui(q, k), divui(addi(muli(remui(q, k), s), r), muli(k, s)))
//
// Algebraic identity: (q*s + r) / (k*s) = q/k + ((q%k)*s + r) / (k*s)
//
// The second term is structurally simpler (q replaced by remui(q,k))
// and typically folds to 0 via SimplifyMixedRadixDiv below.
// ============================================================================
struct ExtendedSimplifyDivId : public OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern<arith::DivUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    Value numerator = op.getLhs();
    Value divisor = op.getRhs();

    auto addOp = numerator.getDefiningOp<arith::AddIOp>();
    if (!addOp)
      return failure();

    auto divMul = divisor.getDefiningOp<arith::MulIOp>();
    if (!divMul)
      return failure();

    // Try both sides of the addi as the muli(q, s) term.
    Value addTerms[2] = {addOp.getLhs(), addOp.getRhs()};
    for (int i = 0; i < 2; ++i) {
      auto termMul = addTerms[i].getDefiningOp<arith::MulIOp>();
      if (!termMul)
        continue;

      // Find shared factor s between termMul and divMul.
      // termMul = muli(q, s), divMul = muli(k, s)  (commutative)
      Value q, s, k;
      Value tm[2] = {termMul.getLhs(), termMul.getRhs()};
      Value dm[2] = {divMul.getLhs(), divMul.getRhs()};

      bool found = false;
      for (int a = 0; a < 2 && !found; ++a)
        for (int b = 0; b < 2 && !found; ++b)
          if (tm[a] == dm[b]) {
            s = tm[a];
            q = tm[1 - a];
            k = dm[1 - b];
            found = true;
          }

      if (!found)
        continue;

      // Don't fire if s == divisor (that's the basic SimplifyDivId case).
      if (s == divisor)
        continue;

      // Don't fire if q is already a remui by k — that means we already
      // decomposed and this is the "low part".  Prevents infinite loops.
      if (auto remOp = q.getDefiningOp<arith::RemUIOp>())
        if (remOp.getRhs() == k)
          continue;

      Location loc = op.getLoc();
      Value r = addTerms[1 - i];

      // q / k
      Value qDivK = arith::DivUIOp::create(rewriter, loc, q, k);
      // (q % k) * s + r
      Value qRemK = arith::RemUIOp::create(rewriter, loc, q, k);
      Value lowTerm = arith::MulIOp::create(rewriter, loc, qRemK, s);
      Value lowSum = arith::AddIOp::create(rewriter, loc, lowTerm, r);
      // divui(lowSum, divisor)  — typically folds to 0
      Value lowDiv = arith::DivUIOp::create(rewriter, loc, lowSum, divisor);

      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, qDivK, lowDiv);
      return success();
    }
    return failure();
  }
};

// Similarly for remui:
//   remui(addi(muli(q, s), r), muli(k, s))
//     → addi(muli(remui(q, k), s), remui(r, s)) when applicable
// But the simpler form: just strip the high term.
//   remui(addi(muli(q, s), r), muli(k, s))
//     → remui(addi(muli(remui(q, k), s), r), muli(k, s))
struct ExtendedSimplifyRemId : public OpRewritePattern<arith::RemUIOp> {
  using OpRewritePattern<arith::RemUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::RemUIOp op,
                                PatternRewriter &rewriter) const override {
    Value numerator = op.getLhs();
    Value divisor = op.getRhs();

    auto addOp = numerator.getDefiningOp<arith::AddIOp>();
    if (!addOp)
      return failure();

    auto divMul = divisor.getDefiningOp<arith::MulIOp>();
    if (!divMul)
      return failure();

    Value addTerms[2] = {addOp.getLhs(), addOp.getRhs()};
    for (int i = 0; i < 2; ++i) {
      auto termMul = addTerms[i].getDefiningOp<arith::MulIOp>();
      if (!termMul)
        continue;

      Value q, s, k;
      Value tm[2] = {termMul.getLhs(), termMul.getRhs()};
      Value dm[2] = {divMul.getLhs(), divMul.getRhs()};

      bool found = false;
      for (int a = 0; a < 2 && !found; ++a)
        for (int b = 0; b < 2 && !found; ++b)
          if (tm[a] == dm[b]) {
            s = tm[a];
            q = tm[1 - a];
            k = dm[1 - b];
            found = true;
          }

      if (!found || s == divisor)
        continue;

      if (auto remOp = q.getDefiningOp<arith::RemUIOp>())
        if (remOp.getRhs() == k)
          continue;

      Location loc = op.getLoc();
      Value r = addTerms[1 - i];

      // (q % k) * s + r
      Value qRemK = arith::RemUIOp::create(rewriter, loc, q, k);
      Value lowTerm = arith::MulIOp::create(rewriter, loc, qRemK, s);
      Value lowSum = arith::AddIOp::create(rewriter, loc, lowTerm, r);

      rewriter.replaceOpWithNewOp<arith::RemUIOp>(op, lowSum, divisor);
      return success();
    }
    return failure();
  }
};

// ============================================================================
// Mixed-radix bound:
//   divui(addi(muli(remui(a, n), m), remui(b, m)), muli(n, m)) → 0
//
// Proof: remui(a,n) ∈ [0,n-1], so muli(remui(a,n), m) ∈ [0, (n-1)*m].
//        remui(b,m) ∈ [0,m-1].
//        Sum ∈ [0, n*m - 1] < n*m = divisor.
// ============================================================================
struct SimplifyMixedRadixDiv : public OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern<arith::DivUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    Value numerator = op.getLhs();
    Value divisor = op.getRhs();

    auto divMul = divisor.getDefiningOp<arith::MulIOp>();
    if (!divMul)
      return failure();

    auto addOp = numerator.getDefiningOp<arith::AddIOp>();
    if (!addOp)
      return failure();

    Value n_candidates[2] = {divMul.getLhs(), divMul.getRhs()};
    Value addTerms[2] = {addOp.getLhs(), addOp.getRhs()};

    // Try each (n, m) assignment from the divisor = muli(n, m).
    for (int d = 0; d < 2; ++d) {
      Value n = n_candidates[d];
      Value m = n_candidates[1 - d];

      // Try each addi operand as the muli(remui(a,n), m) term.
      for (int t = 0; t < 2; ++t) {
        Value hiTerm = addTerms[t];
        Value loTerm = addTerms[1 - t];

        // Check hiTerm = muli(remui(a, n), m)
        auto hiMul = hiTerm.getDefiningOp<arith::MulIOp>();
        if (!hiMul)
          continue;

        // Find which mul operand is m and which is remui(a, n).
        Value mulOps[2] = {hiMul.getLhs(), hiMul.getRhs()};
        for (int mi = 0; mi < 2; ++mi) {
          if (mulOps[mi] != m)
            continue;
          auto remA = mulOps[1 - mi].getDefiningOp<arith::RemUIOp>();
          if (!remA || remA.getRhs() != n)
            continue;

          // Check loTerm = remui(b, m)
          auto remB = loTerm.getDefiningOp<arith::RemUIOp>();
          if (!remB || remB.getRhs() != m)
            continue;

          // Match! The sum is < n*m, so divui is 0.
          rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
          return success();
        }
      }
    }
    return failure();
  }
};

// Mixed-radix bound for remui (dual of SimplifyMixedRadixDiv):
//   remui(addi(muli(remui(a,n), m), remui(b,m)), muli(n,m))
//     → addi(muli(remui(a,n), m), remui(b,m))
// The sum is already < n*m, so the remui is identity.
struct SimplifyMixedRadixRem : public OpRewritePattern<arith::RemUIOp> {
  using OpRewritePattern<arith::RemUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::RemUIOp op,
                                PatternRewriter &rewriter) const override {
    Value numerator = op.getLhs();
    Value divisor = op.getRhs();

    auto divMul = divisor.getDefiningOp<arith::MulIOp>();
    if (!divMul)
      return failure();

    auto addOp = numerator.getDefiningOp<arith::AddIOp>();
    if (!addOp)
      return failure();

    Value n_candidates[2] = {divMul.getLhs(), divMul.getRhs()};
    Value addTerms[2] = {addOp.getLhs(), addOp.getRhs()};

    for (int d = 0; d < 2; ++d) {
      Value n = n_candidates[d];
      Value m = n_candidates[1 - d];

      for (int t = 0; t < 2; ++t) {
        Value hiTerm = addTerms[t];
        Value loTerm = addTerms[1 - t];

        auto hiMul = hiTerm.getDefiningOp<arith::MulIOp>();
        if (!hiMul)
          continue;

        Value mulOps[2] = {hiMul.getLhs(), hiMul.getRhs()};
        for (int mi = 0; mi < 2; ++mi) {
          if (mulOps[mi] != m)
            continue;
          auto remA = mulOps[1 - mi].getDefiningOp<arith::RemUIOp>();
          if (!remA || remA.getRhs() != n)
            continue;

          auto remB = loTerm.getDefiningOp<arith::RemUIOp>();
          if (!remB || remB.getRhs() != m)
            continue;

          // Sum < n*m, so remui is identity.
          rewriter.replaceOp(op, numerator);
          return success();
        }
      }
    }
    return failure();
  }
};

struct LegoArithSimplificationPass
    : public mlir::lego::impl::LegoArithSimplificationPassBase<
          LegoArithSimplificationPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SimplifyRemId, SimplifyDivId, SimplifyDivConst,
                 ReconstructId, SimplifyDivOfRem, SimplifyRemOfRem,
                 ExtendedSimplifyDivId, ExtendedSimplifyRemId,
                 SimplifyMixedRadixDiv, SimplifyMixedRadixRem>(
                     &getContext());

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
} // namespace lego
} // namespace mlir
