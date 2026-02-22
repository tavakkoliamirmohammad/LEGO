#define GEN_PASS_DEF_LEGODESUGARPASS
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

    // 2. Identify the chain of objects from input OrderBy
    //    Python: for o in self.chain: ...
    Operation *inputDefOp = op.getInput().getDefiningOp();
    SmallVector<Value> chain;
    if (auto orderByOp = dyn_cast<OrderByOp>(inputDefOp)) {
      auto range = orderByOp.getPerms();
      chain.append(range.begin(), range.end());
    } else {
      chain.push_back(op.getInput());
    }


    
    SmallVector<Value> groupByObjects;
    for (size_t i = 0; i < chain.size(); ++i) {
        Value obj = chain[i];
        
        // Compute shuffle params
        auto [d_obj, q_obj] = getLayoutDQ(obj);
        auto objDims = getLayoutInputDims(obj);
        auto sigma_o = getSigmaPerm(d_obj, q_obj);
        auto sigma_o_inv = inversePermutation(sigma_o);
        auto reshuffleDims = sigmaValues(ValueRange(objDims), sigma_o);
        
        // Create RegP
        auto regPOp = RegPOp::create(
            rewriter, loc, op.getType(), rewriter.getI64ArrayAttr(sigma_o_inv),
            reshuffleDims);

        // Wrap in OrderBy
        auto orderByOp = OrderByOp::create(rewriter, loc, regPOp.getType(),
                                           ValueRange{regPOp.getResult()});

        groupByObjects.push_back(obj);
        groupByObjects.push_back(orderByOp.getResult());
    }
    
    // Final reshuffle
    {
        auto sigma_dq = getSigmaPerm(d_tile, q_tile);
        auto reshuffledTileDims = sigmaValues(tileDims, sigma_dq);
        auto regPOp = RegPOp::create(rewriter, loc, op.getType(),
                                     rewriter.getI64ArrayAttr(sigma_dq),
                                     reshuffledTileDims);

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
// Pass Definition
// ============================================================================

struct LegoDesugarPassImpl
    : public mlir::lego::impl::LegoDesugarPassBase<LegoDesugarPassImpl> {
  using mlir::lego::impl::LegoDesugarPassBase<
      LegoDesugarPassImpl>::LegoDesugarPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<TileByOpRewrite, RowOpRewrite, ColOpRewrite>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoDesugarPass() {
  return std::make_unique<LegoDesugarPassImpl>();
}
} // namespace lego
} // namespace mlir
