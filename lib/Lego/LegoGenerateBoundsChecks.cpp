#define GEN_PASS_DEF_LEGOGENERATEBOUNDSCHECKSPASS
#include "Lego/LegoOps.h"
#include "Lego/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::lego;

namespace {

struct LegoGenerateBoundsChecksPassImpl
    : public mlir::lego::impl::LegoGenerateBoundsChecksPassBase<
          LegoGenerateBoundsChecksPassImpl> {
  using mlir::lego::impl::LegoGenerateBoundsChecksPassBase<
      LegoGenerateBoundsChecksPassImpl>::LegoGenerateBoundsChecksPassBase;

  void runOnOperation() override {
    getOperation()->walk([](Operation *op) {
      if (auto applyOp = dyn_cast<ApplyOp>(op)) {
        OpBuilder builder(applyOp);
        AssertApplyBoundsOp::create(builder, applyOp.getLoc(),
                                    applyOp.getLayout(),
                                    applyOp.getIndices());
      } else if (auto invOp = dyn_cast<ApplyInverseOp>(op)) {
        OpBuilder builder(invOp);
        AssertInvBoundsOp::create(builder, invOp.getLoc(),
                                  invOp.getLayout(),
                                  invOp.getFlatIndex());
      }
    });
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoGenerateBoundsChecksPass() {
  return std::make_unique<LegoGenerateBoundsChecksPassImpl>();
}
} // namespace lego
} // namespace mlir
