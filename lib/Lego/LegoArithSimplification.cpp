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


struct LegoArithSimplificationPass
    : public mlir::lego::impl::LegoArithSimplificationPassBase<
          LegoArithSimplificationPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SimplifyRemId, SimplifyDivId, SimplifyDivConst,
                 ReconstructId>(&getContext());

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
