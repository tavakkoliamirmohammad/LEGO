#define GEN_PASS_DEF_LEGOMATERIALIZEASSUMEBOUNDSPASS
#include "Lego/Passes.h"
#include "Lego/LegoOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace {

/// Converts `lego.assume_bounds %x ub: %d` directly into
/// `%x_bounded = arith.remui %x, %d` and replaces downstream uses.
/// Also handles the already-lowered form: lego.assume(cmpi slt, %x, %d).
/// All assume/assume_bounds ops are erased afterward.
struct LegoMaterializeAssumeBoundsPass
    : public mlir::lego::impl::LegoMaterializeAssumeBoundsPassBase<
          LegoMaterializeAssumeBoundsPass> {

  LegoMaterializeAssumeBoundsPass() = default;
  LegoMaterializeAssumeBoundsPass(bool cleanupMode) { cleanup = cleanupMode; }

  void runOnOperation() override {
    if (cleanup) {
      runCleanup();
      return;
    }

    OpBuilder builder(&getContext());

    // Phase 1: Collect (value, upper_bound) pairs from assume_bounds ops.
    SmallVector<std::tuple<Operation *, Value, Value>> workList;

    getOperation()->walk([&](lego::AssumeBoundsOp op) {
      Value x = op.getValue();
      Value ub = op.getUb();
      if (!ub)
        return;
      // Only materialize bounds on block arguments (user-declared).
      // Skip computed values (e.g., from lego-generate-bounds-checks).
      if (!mlir::isa<BlockArgument>(x))
        return;
      if (x.getDefiningOp<arith::RemUIOp>())
        return;
      workList.emplace_back(op.getOperation(), x, ub);
    });

    // Also handle the already-lowered form: lego.assume(cmpi slt, %x, %d).
    // Only match when %x is a block argument (user-declared bounds),
    // not a computed value (bounds-checker assertions).
    getOperation()->walk([&](lego::AssumeOp op) {
      Value cond = op.getCondition();
      auto cmpOp = cond.getDefiningOp<arith::CmpIOp>();
      if (!cmpOp)
        return;
      if (cmpOp.getPredicate() != arith::CmpIPredicate::slt &&
          cmpOp.getPredicate() != arith::CmpIPredicate::ult)
        return;
      Value x = cmpOp.getLhs();
      if (!mlir::isa<BlockArgument>(x))
        return;
      if (x.getDefiningOp<arith::RemUIOp>())
        return;
      workList.emplace_back(op.getOperation(), x, cmpOp.getRhs());
    });

    // Phase 2: Insert remui and replace downstream uses.
    for (auto &[originOp, x, d] : workList) {
      builder.setInsertionPointAfter(originOp);
      Value bounded =
          arith::RemUIOp::create(builder, originOp->getLoc(), x, d);
      Operation *remOp = bounded.getDefiningOp();
      remOp->setAttr("lego.materialized", builder.getUnitAttr());

      x.replaceUsesWithIf(bounded, [&](OpOperand &operand) {
        Operation *user = operand.getOwner();
        if (user == remOp)
          return false;
        if (user->getBlock() == remOp->getBlock())
          return !user->isBeforeInBlock(remOp);
        return false;
      });
    }

    // Phase 3: Erase only the assume_bounds / assume ops we processed.
    // Do NOT erase unrelated lego.assume ops (e.g., from bounds checker).
    SmallPtrSet<Operation *, 8> processed;
    for (auto &[originOp, x, d] : workList)
      processed.insert(originOp);

    for (auto *op : processed) {
      SmallVector<Operation *> maybeDeadDefs;
      for (Value operand : op->getOperands())
        if (auto *defOp = operand.getDefiningOp())
          maybeDeadDefs.push_back(defOp);
      op->erase();
      for (auto *defOp : maybeDeadDefs)
        if (defOp->use_empty())
          defOp->erase();
    }
  }

  /// Cleanup mode: remove remui(block_arg, d) that we inserted.
  /// We tag inserted remui ops with a "lego.materialized" unit attribute
  /// so we can identify them later.
  void runCleanup() {
    SmallVector<arith::RemUIOp> toFold;
    getOperation()->walk([&](arith::RemUIOp op) {
      if (op->hasAttr("lego.materialized"))
        toFold.push_back(op);
    });
    for (auto op : toFold) {
      op.replaceAllUsesWith(op.getLhs());
      op.erase();
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoMaterializeAssumeBoundsPass(bool cleanupMode) {
  return std::make_unique<LegoMaterializeAssumeBoundsPass>(cleanupMode);
}
} // namespace lego
} // namespace mlir
