#define GEN_PASS_DEF_LEGOEXTERNALSMTVERIFIERPASS
#include "Lego/LegoOps.h"
#include "Lego/Passes.h"
#include "Lego/LegoUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include <fstream>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

using namespace mlir;
using namespace mlir::lego;

namespace {

struct SMTBuilder {
  std::string smtLib;
  DenseMap<Value, std::string> valNames;
  int varCounter = 0;

  std::string getOrCreateName(Value v) {
    if (valNames.count(v)) return valNames[v];
    std::string name = "v" + std::to_string(varCounter++);
    valNames[v] = name;
    return name;
  }

  void visit(Value v) {
    if (valNames.count(v)) return;
    
    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
      std::string name = getOrCreateName(v);
      smtLib += "(declare-const " + name + " Int)\n";
      return;
    }

    for (Value operand : defOp->getOperands()) {
      visit(operand);
    }

    std::string name = getOrCreateName(v);

    if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        std::string typeStr = v.getType().isInteger(1) ? "Bool" : "Int";
        smtLib += "(declare-const " + name + " " + typeStr + ")\n";
        if (typeStr == "Bool") {
          smtLib += "(assert (= " + name + " " + (intAttr.getInt() ? "true" : "false") + "))\n";
        } else {
          smtLib += "(assert (= " + name + " " + std::to_string(intAttr.getInt()) + "))\n";
        }
      } else {
        std::string typeStr = v.getType().isInteger(1) ? "Bool" : "Int";
        smtLib += "(declare-const " + name + " " + typeStr + ")\n";
      }
    } else if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
      smtLib += "(declare-const " + name + " Int)\n";
      smtLib += "(assert (= " + name + " (+ " + valNames[addOp.getLhs()] + " " + valNames[addOp.getRhs()] + ")))\n";
    } else if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
      smtLib += "(declare-const " + name + " Int)\n";
      smtLib += "(assert (= " + name + " (- " + valNames[subOp.getLhs()] + " " + valNames[subOp.getRhs()] + ")))\n";
    } else if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
      smtLib += "(declare-const " + name + " Int)\n";
      smtLib += "(assert (= " + name + " (* " + valNames[mulOp.getLhs()] + " " + valNames[mulOp.getRhs()] + ")))\n";
    } else if (auto divOp = dyn_cast<arith::DivUIOp>(defOp)) {
      smtLib += "(declare-const " + name + " Int)\n";
      smtLib += "(assert (= " + name + " (div " + valNames[divOp.getLhs()] + " " + valNames[divOp.getRhs()] + ")))\n";
    } else if (auto divSIOp = dyn_cast<arith::DivSIOp>(defOp)) {
      smtLib += "(declare-const " + name + " Int)\n";
      smtLib += "(assert (= " + name + " (div " + valNames[divSIOp.getLhs()] + " " + valNames[divSIOp.getRhs()] + ")))\n";
    } else if (auto remOp = dyn_cast<arith::RemUIOp>(defOp)) {
      smtLib += "(declare-const " + name + " Int)\n";
      smtLib += "(assert (= " + name + " (mod " + valNames[remOp.getLhs()] + " " + valNames[remOp.getRhs()] + ")))\n";
    } else if (auto remSIOp = dyn_cast<arith::RemSIOp>(defOp)) {
      smtLib += "(declare-const " + name + " Int)\n";
      smtLib += "(assert (= " + name + " (mod " + valNames[remSIOp.getLhs()] + " " + valNames[remSIOp.getRhs()] + ")))\n";
    } else if (auto cmpOp = dyn_cast<arith::CmpIOp>(defOp)) {
      smtLib += "(declare-const " + name + " Bool)\n";
      std::string pred = "";
      switch (cmpOp.getPredicate()) {
        case arith::CmpIPredicate::eq: pred = "="; break;
        case arith::CmpIPredicate::ne: pred = "distinct"; break;
        case arith::CmpIPredicate::slt:
        case arith::CmpIPredicate::ult: pred = "<"; break;
        case arith::CmpIPredicate::sle:
        case arith::CmpIPredicate::ule: pred = "<="; break;
        case arith::CmpIPredicate::sgt:
        case arith::CmpIPredicate::ugt: pred = ">"; break;
        case arith::CmpIPredicate::sge:
        case arith::CmpIPredicate::uge: pred = ">="; break;
      }
      smtLib += "(assert (= " + name + " (" + pred + " " + valNames[cmpOp.getLhs()] + " " + valNames[cmpOp.getRhs()] + ")))\n";
    } else if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp)) {
      valNames[v] = valNames[castOp.getIn()];
    } else if (auto extOp = dyn_cast<arith::ExtUIOp>(defOp)) {
      valNames[v] = valNames[extOp.getIn()];
    } else if (auto extSOp = dyn_cast<arith::ExtSIOp>(defOp)) {
      valNames[v] = valNames[extSOp.getIn()];
    } else if (auto truncOp = dyn_cast<arith::TruncIOp>(defOp)) {
      valNames[v] = valNames[truncOp.getIn()];
    } else {
      std::string typeStr = v.getType().isInteger(1) ? "Bool" : "Int";
      smtLib += "(declare-const " + name + " " + typeStr + ")\n";
    }
  }
};

struct LegoExternalSMTVerifierPassImpl
    : public mlir::lego::impl::LegoExternalSMTVerifierPassBase<
          LegoExternalSMTVerifierPassImpl> {
  using mlir::lego::impl::LegoExternalSMTVerifierPassBase<
      LegoExternalSMTVerifierPassImpl>::LegoExternalSMTVerifierPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<AssumeOp> assumes;
    SmallVector<AssertApplyBoundsOp> applies;
    SmallVector<AssertInvBoundsOp> invs;

    module.walk([&](Operation *op) {
      if (auto assume = dyn_cast<AssumeOp>(op)) assumes.push_back(assume);
      if (auto apply = dyn_cast<AssertApplyBoundsOp>(op)) applies.push_back(apply);
      if (auto inv = dyn_cast<AssertInvBoundsOp>(op)) invs.push_back(inv);
    });

    if (applies.empty() && invs.empty()) return;

    for (auto apply : applies) {
      SMTBuilder builder;
      builder.smtLib += "; SMT formulation for AssertApplyBoundsOp\n";
      builder.smtLib += "(set-logic QF_NIA)\n";

      SmallVector<Value> dims = getLayoutInputDims(apply.getLayout());
      for (Value d : dims) builder.visit(d);

      for (Value idx : apply.getIndices()) builder.visit(idx);

      for (auto assume : assumes) {
         builder.visit(assume.getCondition());
         builder.smtLib += "(assert " + builder.valNames[assume.getCondition()] + ")\n";
      }

      std::string outOfBounds = "(or ";
      for (size_t i = 0; i < apply.getIndices().size(); ++i) {
          std::string idxName = builder.valNames[apply.getIndices()[i]];
          std::string dimName = "1";
          if (i < dims.size()) {
             dimName = builder.valNames[dims[i]];
          }
          outOfBounds += "(< " + idxName + " 0) ";
          outOfBounds += "(>= " + idxName + " " + dimName + ") ";
      }
      outOfBounds += "false)"; // Trailing false to handle single-element `or` cleanly if needed
      builder.smtLib += "(assert " + outOfBounds + ")\n";
      builder.smtLib += "(check-sat)\n";
      
      char tempPattern[] = "/tmp/lego_smt_XXXXXX";
      int fd = mkstemp(tempPattern);
      if (fd == -1) {
          apply.emitError("Failed to create temporary SMT file");
          signalPassFailure();
          continue;
      }
      std::string tempFile(tempPattern);
      std::ofstream out(tempFile);
      out << builder.smtLib;
      out.close();
      close(fd);

      int ret = system(("PATH=/uufs/chpc.utah.edu/common/home/u1419116/projects/LEGO_transform/venv/bin:$PATH python3 /uufs/chpc.utah.edu/common/home/u1419116/projects/LEGO_transform/verify_bounds.py " + tempFile).c_str());
      remove(tempFile.c_str());
      if (WIFEXITED(ret) && WEXITSTATUS(ret) == 1) {
          apply.emitError("Out-of-bounds access is possible (proven by Z3)");
          signalPassFailure();
      }
    }

    for (auto inv : invs) {
      SMTBuilder builder;
      builder.smtLib += "; SMT formulation for AssertInvBoundsOp\n";
      builder.smtLib += "(set-logic QF_NIA)\n";

      SmallVector<Value> dims = getLayoutInputDims(inv.getLayout());
      for (Value d : dims) builder.visit(d);
      builder.visit(inv.getFlatIndex());

      for (auto assume : assumes) {
         builder.visit(assume.getCondition());
         builder.smtLib += "(assert " + builder.valNames[assume.getCondition()] + ")\n";
      }

      std::string volName = "vol_0";
      builder.smtLib += "(declare-const " + volName + " Int)\n";
      if (dims.empty()) {
         builder.smtLib += "(assert (= " + volName + " 1))\n";
      } else {
         std::string volExpr = builder.valNames[dims[0]];
         for (size_t i = 1; i < dims.size(); ++i) {
           volExpr = "(* " + volExpr + " " + builder.valNames[dims[i]] + ")";
         }
         builder.smtLib += "(assert (= " + volName + " " + volExpr + "))\n";
      }

      std::string flatIdx = builder.valNames[inv.getFlatIndex()];
      std::string outOfBounds = "(or (< " + flatIdx + " 0) (>= " + flatIdx + " " + volName + "))";
      builder.smtLib += "(assert " + outOfBounds + ")\n";
      builder.smtLib += "(check-sat)\n";

      char tempPattern[] = "/tmp/lego_smt_inv_XXXXXX";
      int fd = mkstemp(tempPattern);
      if (fd == -1) {
          inv.emitError("Failed to create temporary SMT file");
          signalPassFailure();
          continue;
      }
      std::string tempFile(tempPattern);
      std::ofstream out(tempFile);
      out << builder.smtLib;
      out.close();
      close(fd);

      int ret = system(("PATH=/uufs/chpc.utah.edu/common/home/u1419116/projects/LEGO_transform/venv/bin:$PATH python3 /uufs/chpc.utah.edu/common/home/u1419116/projects/LEGO_transform/verify_bounds.py " + tempFile).c_str());
      remove(tempFile.c_str());
      if (WIFEXITED(ret) && WEXITSTATUS(ret) == 1) {
          inv.emitError("Out-of-bounds flat index is possible (proven by Z3)");
          signalPassFailure();
      }
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoExternalSMTVerifierPass() {
  return std::make_unique<LegoExternalSMTVerifierPassImpl>();
}
} // namespace lego
} // namespace mlir
