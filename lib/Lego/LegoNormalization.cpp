#define GEN_PASS_DEF_LEGONORMALIZATIONPASS
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Lego/LegoOps.h"
#include "Lego/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <numeric>
#include "Lego/LegoUtils.h"

using namespace mlir;
using namespace mlir::lego;

namespace {


// ============================================================================
// RowOp Rewrite Pattern
// ============================================================================

struct RowOpRewrite : public OpRewritePattern<RowOp> {
  using OpRewritePattern<RowOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RowOp op,
                                PatternRewriter &rewriter) const override {
    // Row(*dims) = RegP(dims, identity_perm)
    Location loc = op.getLoc();
    auto dims = op.getDims();
    int d = dims.size();
    
    // Identity permutation: [0, 1, ..., d-1]
    SmallVector<int64_t> perm(d);
    std::iota(perm.begin(), perm.end(), 0);

    // Create RegPOp
    auto regPOp = RegPOp::create(rewriter, loc, op.getType(),
                                 rewriter.getI64ArrayAttr(perm), op.getDims());

    rewriter.replaceOp(op, regPOp.getResult());
    return success();
  }
};

// ============================================================================
// ColOp Rewrite Pattern
// ============================================================================

struct ColOpRewrite : public OpRewritePattern<ColOp> {
  using OpRewritePattern<ColOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ColOp op,
                                PatternRewriter &rewriter) const override {
    // Col(*dims) = RegP(dims, reversed_identity)
    Location loc = op.getLoc();
    auto dims = op.getDims();
    int d = dims.size();

    // Reversed identity: [d-1, ..., 0]
    SmallVector<int64_t> perm(d);
    for (int i = 0; i < d; ++i) {
      perm[i] = d - 1 - i;
    }

    // Create RegPOp
    auto regPOp = RegPOp::create(rewriter, loc, op.getType(),
                                 rewriter.getI64ArrayAttr(perm), op.getDims());

    rewriter.replaceOp(op, regPOp.getResult());
    return success();
  }
};

// ============================================================================
// TileByOp Rewrite Pattern
// ============================================================================

struct TileByOpRewrite : public OpRewritePattern<TileByOp> {
  using OpRewritePattern<TileByOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TileByOp op,
                                PatternRewriter &rewriter) const override {

    auto tileShapeAttr = op.getTileShape();
    auto tileShape = extractI64Array(tileShapeAttr);
    if (tileShape.empty())
      return rewriter.notifyMatchFailure(op, "Invalid tile shape");

    int64_t d_tile = tileShape[0];
    int64_t q_tile = tileShape.size();

    auto tileDims = op.getTileDims();
    Location loc = op.getLoc();

    // ---- Identity 1 ----
    // TileBy(OrderBy(P0(d0), P1(d1), ...), [[d0], [d1], ...])
    //   → OrderBy(P0(d0), P1(d1), ...)
    // When each tile level's dims match the corresponding inner block's
    // dims, the TileBy is identity — just use the inner layout directly.
    {
      Value inner = op.getInput();
      if (auto obOp = inner.getDefiningOp<OrderByOp>()) {
        auto perms = obOp.getPerms();
        if (q_tile == (int64_t)perms.size()) {
          bool allMatch = true;
          for (int64_t level = 0; level < q_tile && allMatch; ++level) {
            auto blockDims = getLayoutInputDims(perms[level]);
            if ((int64_t)blockDims.size() != d_tile) {
              allMatch = false;
              break;
            }
            for (int64_t k = 0; k < d_tile; ++k) {
              if (tileDims[level * d_tile + k] != blockDims[k]) {
                allMatch = false;
                break;
              }
            }
          }
          if (allMatch) {
            rewriter.replaceOp(op, inner);
            return success();
          }
        }
      }
    }

    // ---- General case: full GroupBy normalization ----

    // 2. Identify the chain of OrderBy objects from input.
    //    Python: for o in self.chain: ...
    //    Each chain element must be a FULL OrderBy (not individual perms),
    //    so that every GroupBy object has product-of-dims == total elements.
    SmallVector<Value> chain;
    chain.push_back(op.getInput());

    SmallVector<Value> groupByObjects;
    for (size_t i = 0; i < chain.size(); ++i) {
        Value obj = chain[i];

        // Compute shuffle params using the full OrderBy's {d, q}
        auto [d_obj, q_obj] = getLayoutDQ(obj);
        auto objDims = getLayoutInputDims(obj);
        auto sigma_o = getSigmaPerm(d_obj, q_obj);
        auto sigma_o_inv = inversePermutation(sigma_o);
        auto reshuffleDims = sigmaValues(ValueRange(objDims), sigma_o);

        // Create RegP(σ(o_dims), σ⁻¹)
        auto regPOp = RegPOp::create(
            rewriter, loc, op.getType(), rewriter.getI64ArrayAttr(sigma_o_inv),
            reshuffleDims);

        // Wrap in OrderBy
        auto orderByOp = OrderByOp::create(rewriter, loc, regPOp.getType(),
                                           ValueRange{regPOp.getResult()});

        groupByObjects.push_back(obj);
        groupByObjects.push_back(orderByOp.getResult());
    }

    // Tile reshuffle: RegP(tileDims, σ(d,q))
    {
        auto sigma_dq = getSigmaPerm(d_tile, q_tile);
        auto regPOp = RegPOp::create(rewriter, loc, op.getType(),
                                     rewriter.getI64ArrayAttr(sigma_dq),
                                     tileDims);

        auto orderByOp = OrderByOp::create(rewriter, loc, regPOp.getType(),
                                           ValueRange{regPOp.getResult()});

        groupByObjects.push_back(orderByOp.getResult());
    }

    // 6. Create GroupByOp
    //    GroupBy([dims], new_order_by + ...)
    auto groupByOp = GroupByOp::create(rewriter, loc, op.getType(),
                                       tileDims,
                                       groupByObjects);

    rewriter.replaceOp(op, groupByOp.getResult());

    return success();
  }
};

// ============================================================================
// AssumeBoundsOp Rewrite Pattern
// ============================================================================

struct AssumeBoundsOpRewrite : public OpRewritePattern<AssumeBoundsOp> {
  using OpRewritePattern<AssumeBoundsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AssumeBoundsOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value val = op.getValue();

    if (Value lb = op.getLb()) {
      auto cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge, val, lb);
      AssumeOp::create(rewriter, loc, cmp.getResult());
    }

    if (Value ub = op.getUb()) {
      auto cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt, val, ub);
      AssumeOp::create(rewriter, loc, cmp.getResult());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// ============================================================================
// Pass Definition
// ============================================================================

struct LegoNormalizationPassImpl
    : public mlir::lego::impl::LegoNormalizationPassBase<LegoNormalizationPassImpl> {
  using mlir::lego::impl::LegoNormalizationPassBase<
      LegoNormalizationPassImpl>::LegoNormalizationPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<TileByOpRewrite, RowOpRewrite, ColOpRewrite,
                 AssumeBoundsOpRewrite>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoNormalizationPass() {
  return std::make_unique<LegoNormalizationPassImpl>();
}
} // namespace lego
} // namespace mlir
