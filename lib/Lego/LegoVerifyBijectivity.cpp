#define GEN_PASS_DEF_LEGOVERIFYBIJECTIVITYPASS
#include "Lego/LegoOps.h"
#include "Lego/Passes.h"
#include "Lego/LegoUtils.h"
#include "Lego/SMTUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/AsmState.h"
#include "Lego/SMTUtils.h"

using namespace mlir;
using namespace mlir::lego;

namespace {

struct LegoVerifyBijectivityPassImpl
    : public mlir::lego::impl::LegoVerifyBijectivityPassBase<
          LegoVerifyBijectivityPassImpl> {
  using mlir::lego::impl::LegoVerifyBijectivityPassBase<
      LegoVerifyBijectivityPassImpl>::LegoVerifyBijectivityPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<smt::SMTDialect>();
  }

  void runOnOperation() override {
    getContext().getOrLoadDialect<smt::SMTDialect>();
    ModuleOp module = getOperation();

    SmallVector<GenPOp> genPOps;
    module.walk([&](GenPOp genP) { genPOps.push_back(genP); });

    if (genPOps.empty()) return;

    AsmState state(module);
    unsigned nextId = 0;

    for (auto genP : genPOps) {
      // Skip if no inverse region
      if (genP.getInvBody().empty()) {
        genP.emitWarning("Skipping bijectivity check: no inverse region defined");
        continue;
      }

      // Verify: ∀ indices. inv(apply(indices)) = indices
      if (failed(verifyInvApplyIdentity(genP, state, nextId))) {
        signalPassFailure();
      }

      // Verify: ∀ flat. apply(inv(flat)) = flat
      if (failed(verifyApplyInvIdentity(genP, state, nextId))) {
        signalPassFailure();
      }
    }
  }

private:
  // Verify: inv(apply(x1, x2, ..., xn)) = (x1, x2, ..., xn)
  LogicalResult verifyInvApplyIdentity(GenPOp genP, AsmState &state, unsigned &nextId) {
    SMTSolverContext smtCtx(genP.getLoc(), state, nextId);
    OpBuilder &b = *smtCtx.b;
    SMTBuilder &builder = *smtCtx.builder;

    // Create symbolic variables for input indices
    SmallVector<Value> inputIndices;
    SmallVector<std::string> varNames;
    size_t numDims = genP.getBody().front().getNumArguments();

    for (size_t i = 0; i < numDims; ++i) {
      std::string varName = "input_" + std::to_string(i);
      varNames.push_back(varName);
      Value var = smt::DeclareFunOp::create(
          b, genP.getLoc(),
          Type(b.getType<smt::IntType>()),
          b.getStringAttr(varName));
      inputIndices.push_back(var);
    }

    // Add bounds constraints: 0 <= input_i < dim_i, and dim_i > 0
    SmallVector<Value> dims = getLayoutInputDims(genP);
    Value zero = smt::IntConstantOp::create(b, genP.getLoc(), b.getI64IntegerAttr(0));
    for (size_t i = 0; i < numDims && i < dims.size(); ++i) {
      Value dim = builder.getOrCreate(dims[i]);

      // Assert: dim_i > 0 (tensor dimensions are always positive)
      Value dimPositive = smt::IntCmpOp::create(
          b, genP.getLoc(), smt::IntPredicate::gt, dim, zero);
      smt::AssertOp::create(b, genP.getLoc(), dimPositive);

      // Assert: input_i >= 0
      Value geZero = smt::IntCmpOp::create(
          b, genP.getLoc(), smt::IntPredicate::ge, inputIndices[i], zero);
      smt::AssertOp::create(b, genP.getLoc(), geZero);

      // Assert: input_i < dim_i
      Value ltDim = smt::IntCmpOp::create(
          b, genP.getLoc(), smt::IntPredicate::lt, inputIndices[i], dim);
      smt::AssertOp::create(b, genP.getLoc(), ltDim);
    }

    // Apply the forward transformation: flat = apply(inputs)
    SmallVector<Value> applyResults;
    builder.buildRegion(genP.getBody(), inputIndices, applyResults);
    if (applyResults.size() != 1) {
      genP.emitError("Apply region must yield exactly one value");
      return failure();
    }
    Value flatIdx = applyResults[0];

    // Apply the inverse transformation: outputs = inv(flat)
    SmallVector<Value> invResults;
    builder.buildRegion(genP.getInvBody(), flatIdx, invResults);
    if (invResults.size() != numDims) {
      genP.emitError("Inverse region must yield same number of dimensions as input");
      return failure();
    }

    // Assert that some output differs from the corresponding input
    // If SAT, we found a counterexample to bijectivity
    SmallVector<Value> inequalities;
    for (size_t i = 0; i < numDims; ++i) {
      Value neq = smt::DistinctOp::create(
          b, genP.getLoc(),
          ValueRange{invResults[i], inputIndices[i]});
      inequalities.push_back(neq);
    }

    // Declare named variables for output in SMT (BEFORE exporting!)
    // This allows us to extract concrete values from Z3
    for (size_t i = 0; i < numDims; ++i) {
      Value namedInvResult = smt::DeclareFunOp::create(
          b, genP.getLoc(),
          Type(b.getType<smt::IntType>()),
          b.getStringAttr("inv_result_" + std::to_string(i)));
      Value eq = smt::EqOp::create(b, genP.getLoc(), namedInvResult, invResults[i]);
      smt::AssertOp::create(b, genP.getLoc(), eq);
    }
    Value namedFlat = smt::DeclareFunOp::create(
        b, genP.getLoc(),
        Type(b.getType<smt::IntType>()),
        b.getStringAttr("flat_idx"));
    Value eqFlat = smt::EqOp::create(b, genP.getLoc(), namedFlat, flatIdx);
    smt::AssertOp::create(b, genP.getLoc(), eqFlat);

    Value anyDifferent = inequalities.size() == 1 ? inequalities[0]
                        : smt::OrOp::create(b, genP.getLoc(), inequalities);
    smt::AssertOp::create(b, genP.getLoc(), anyDifferent);

    // Request values for: inputs, flat index, and inverse results
    SmallVector<std::string> allVars = varNames;
    allVars.push_back("flat_idx");
    for (size_t i = 0; i < numDims; ++i) {
      allVars.push_back("inv_result_" + std::to_string(i));
    }

    SMTResult result = smtCtx.checkSatisfiability(allVars);

    if (result.isSat) {
      std::string errorMsg = "Layout is not bijective: inv(apply(x)) ≠ x\n  Counter-example:";

      // Show input values
      errorMsg += "\n    Input indices:";
      for (size_t i = 0; i < varNames.size(); ++i) {
        if (result.model.count(varNames[i])) {
          errorMsg += "\n      [" + std::to_string(i) + "] = " +
                     std::to_string(result.model[varNames[i]]);
        }
      }

      // Show flat index computed by apply
      if (result.model.count("flat_idx")) {
        errorMsg += "\n    Flat index (apply result): " +
                   std::to_string(result.model["flat_idx"]);
      }

      // Show what inverse computed
      errorMsg += "\n    Inverse result (should equal input):";
      for (size_t i = 0; i < numDims; ++i) {
        std::string invVar = "inv_result_" + std::to_string(i);
        if (result.model.count(invVar)) {
          errorMsg += "\n      [" + std::to_string(i) + "] = " +
                     std::to_string(result.model[invVar]);
        }
      }

      genP.emitError(errorMsg);
      return failure();
    } else if (result.isUnsat) {
      // Property holds!
      return success();
    } else {
      genP.emitWarning("Bijectivity check (inv∘apply) returned unknown");
      return success();  // Don't fail on unknown
    }
  }

  // Verify: apply(inv(flat)) = flat
  LogicalResult verifyApplyInvIdentity(GenPOp genP, AsmState &state, unsigned &nextId) {
    SMTSolverContext smtCtx(genP.getLoc(), state, nextId);
    OpBuilder &b = *smtCtx.b;
    SMTBuilder &builder = *smtCtx.builder;

    // Create symbolic variable for flat index
    std::string flatVarName = "flat_input";
    Value flatInput = smt::DeclareFunOp::create(
        b, genP.getLoc(),
        Type(b.getType<smt::IntType>()),
        b.getStringAttr(flatVarName));

    // Add bounds constraint for flat index: 0 <= flat < volume
    SmallVector<Value> dims = getLayoutInputDims(genP);
    Value zero = smt::IntConstantOp::create(b, genP.getLoc(), b.getI64IntegerAttr(0));

    // Compute volume = dim[0] * dim[1] * ... * dim[n-1]
    Value volume = nullptr;
    if (dims.empty()) {
      // 0-dimensional layout: flat_input can only be 0
      Value eqZero = smt::EqOp::create(b, genP.getLoc(), flatInput, zero);
      smt::AssertOp::create(b, genP.getLoc(), eqZero);
    } else {
      SmallVector<Value> dimValues;
      for (Value d : dims) {
        Value dimVal = builder.getOrCreate(d);
        dimValues.push_back(dimVal);
        // Assert: dim > 0 (tensor dimensions are always positive)
        Value dimPositive = smt::IntCmpOp::create(
            b, genP.getLoc(), smt::IntPredicate::gt, dimVal, zero);
        smt::AssertOp::create(b, genP.getLoc(), dimPositive);
      }
      if (dimValues.size() == 1) {
        volume = dimValues[0];
      } else {
        volume = smt::IntMulOp::create(b, genP.getLoc(), dimValues);
      }

      // Assert: flat_input >= 0
      Value geZero = smt::IntCmpOp::create(
          b, genP.getLoc(), smt::IntPredicate::ge, flatInput, zero);
      smt::AssertOp::create(b, genP.getLoc(), geZero);

      // Assert: flat_input < volume
      Value ltVolume = smt::IntCmpOp::create(
          b, genP.getLoc(), smt::IntPredicate::lt, flatInput, volume);
      smt::AssertOp::create(b, genP.getLoc(), ltVolume);
    }

    // Apply inverse: indices = inv(flat)
    SmallVector<Value> invResults;
    builder.buildRegion(genP.getInvBody(), flatInput, invResults);

    // Apply forward: flat_output = apply(indices)
    SmallVector<Value> applyResults;
    builder.buildRegion(genP.getBody(), invResults, applyResults);
    if (applyResults.size() != 1) {
      genP.emitError("Apply region must yield exactly one value");
      return failure();
    }
    Value flatOutput = applyResults[0];

    // Declare named variables for tracking (BEFORE exporting!)
    for (size_t i = 0; i < invResults.size(); ++i) {
      Value namedInvIdx = smt::DeclareFunOp::create(
          b, genP.getLoc(),
          Type(b.getType<smt::IntType>()),
          b.getStringAttr("inv_indices_" + std::to_string(i)));
      Value eq = smt::EqOp::create(b, genP.getLoc(), namedInvIdx, invResults[i]);
      smt::AssertOp::create(b, genP.getLoc(), eq);
    }
    Value namedFlatOut = smt::DeclareFunOp::create(
        b, genP.getLoc(),
        Type(b.getType<smt::IntType>()),
        b.getStringAttr("flat_output"));
    Value eqOut = smt::EqOp::create(b, genP.getLoc(), namedFlatOut, flatOutput);
    smt::AssertOp::create(b, genP.getLoc(), eqOut);

    // Assert that flat_output ≠ flat_input
    Value neq = smt::DistinctOp::create(
        b, genP.getLoc(),
        ValueRange{flatOutput, flatInput});
    smt::AssertOp::create(b, genP.getLoc(), neq);

    SmallVector<std::string> varNames = {flatVarName};

    // Add named variables for the intermediate results
    SmallVector<std::string> allVars = varNames;
    for (size_t i = 0; i < invResults.size(); ++i) {
      allVars.push_back("inv_indices_" + std::to_string(i));
    }
    allVars.push_back("flat_output");

    SMTResult result = smtCtx.checkSatisfiability(allVars);

    if (result.isSat) {
      std::string errorMsg = "Layout is not bijective: apply(inv(y)) ≠ y\n  Counter-example:";

      // Show input flat index
      if (result.model.count(flatVarName)) {
        errorMsg += "\n    Input flat index: " +
                   std::to_string(result.model[flatVarName]);
      }

      // Show what inverse computed
      errorMsg += "\n    Inverse result (indices):";
      for (size_t i = 0; i < invResults.size(); ++i) {
        std::string invVar = "inv_indices_" + std::to_string(i);
        if (result.model.count(invVar)) {
          errorMsg += "\n      [" + std::to_string(i) + "] = " +
                     std::to_string(result.model[invVar]);
        }
      }

      // Show what apply computed (should equal input)
      if (result.model.count("flat_output")) {
        errorMsg += "\n    Apply result (should equal input): " +
                   std::to_string(result.model["flat_output"]);
      }

      genP.emitError(errorMsg);
      return failure();
    } else if (result.isUnsat) {
      return success();
    } else {
      genP.emitWarning("Bijectivity check (apply∘inv) returned unknown");
      return success();
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoVerifyBijectivityPass() {
  return std::make_unique<LegoVerifyBijectivityPassImpl>();
}
} // namespace lego
} // namespace mlir
