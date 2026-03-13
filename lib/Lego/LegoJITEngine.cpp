//===- LegoJITEngine.cpp - JIT compilation engine for LEGO IR -------------===//
//
// Implements the JIT engine: parse MLIR text → run lego-to-llvm → JIT compile.
//
//===----------------------------------------------------------------------===//

#include "LegoJITEngine.h"
#include "Lego/Passes.h"
#include "Lego/LegoDialect.h"
#include "Lego/LegoOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/Support/TargetSelect.h"

using namespace mlir;
using namespace mlir::lego;

LegoJITEngine::~LegoJITEngine() = default;

std::unique_ptr<LegoJITEngine>
LegoJITEngine::create(const std::string &mlirText, std::string *errorMsg) {
  auto jit = std::unique_ptr<LegoJITEngine>(new LegoJITEngine());

  // Initialize LLVM targets for JIT compilation
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Set up context with all needed dialects
  jit->ctx = std::make_unique<MLIRContext>();
  jit->ctx->loadDialect<lego::LegoDialect, func::FuncDialect,
                         arith::ArithDialect, scf::SCFDialect,
                         memref::MemRefDialect, cf::ControlFlowDialect,
                         LLVM::LLVMDialect>();

  // Register LLVM IR translation interfaces
  registerBuiltinDialectTranslation(*jit->ctx);
  registerLLVMDialectTranslation(*jit->ctx);

  // Parse the MLIR text
  jit->module = parseSourceString<ModuleOp>(mlirText, jit->ctx.get());
  if (!jit->module) {
    if (errorMsg)
      *errorMsg = "Failed to parse MLIR module";
    return nullptr;
  }

  // Run the lego-to-llvm pipeline (defined in Passes.cpp)
  PassManager pm(jit->ctx.get());
  buildLegoToLLVMPipeline(pm);

  if (failed(pm.run(*jit->module))) {
    if (errorMsg)
      *errorMsg = "Failed to lower MLIR module to LLVM dialect";
    return nullptr;
  }

  // JIT compile
  auto engineExpected = ExecutionEngine::create(*jit->module);
  if (!engineExpected) {
    if (errorMsg) {
      llvm::raw_string_ostream os(*errorMsg);
      os << engineExpected.takeError();
    } else {
      llvm::consumeError(engineExpected.takeError());
    }
    return nullptr;
  }
  jit->engine = std::move(*engineExpected);
  return jit;
}

bool LegoJITEngine::invoke(const std::string &funcName, void *srcPtr,
                           void *dstPtr, int64_t numElements) {
  if (!engine)
    return false;

  llvm::SmallVector<void *, 3> args = {&srcPtr, &dstPtr, &numElements};
  auto result = engine->invokePacked(funcName, args);
  if (result) {
    llvm::consumeError(std::move(result));
    return false;
  }
  return true;
}
