/// LEGO WebAssembly pass pipeline.
///
/// Pipeline: LEGO ops -> Arith -> LLVM dialect -> WASM binary
///
/// Reuses the CPU lowering path (lego-lower + LLVM conversion), then
/// translates the LLVM dialect module to LLVM IR and emits a wasm32
/// object file via LLVM's WebAssembly backend.  The resulting binary
/// is stored as a base64-encoded string attribute `lego.wasm_binary`
/// on the top-level module, following the same pattern as SerializeSPIRVPass.

#ifdef LEGO_HAS_WASM

#include "Lego/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace mlir;

namespace {

// ============================================================================
// SerializeWasmPass — translate LLVM dialect to wasm32 object code
// ============================================================================

struct SerializeWasmPass
    : public PassWrapper<SerializeWasmPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerializeWasmPass)

  int optLevel;

  explicit SerializeWasmPass(int optLevel = 2) : optLevel(optLevel) {}

  StringRef getArgument() const override { return "lego-serialize-wasm"; }
  StringRef getDescription() const override {
    return "Translate LLVM dialect to WebAssembly object code";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // 1. Translate MLIR LLVM dialect -> LLVM IR
    llvm::LLVMContext llvmCtx;
    auto llvmModule = translateModuleToLLVMIR(moduleOp, llvmCtx);
    if (!llvmModule) {
      moduleOp.emitError("failed to translate LLVM dialect to LLVM IR");
      signalPassFailure();
      return;
    }

    // 2. Look up the wasm32 target
    llvm::Triple triple("wasm32-unknown-unknown");
    llvmModule->setTargetTriple(triple);

    std::string error;
    const auto *target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
      moduleOp.emitError("WebAssembly target not available: " + error);
      signalPassFailure();
      return;
    }

    // 3. Create target machine
    llvm::TargetOptions targetOpts;
    auto optLevelEnum = [&]() -> llvm::CodeGenOptLevel {
      switch (optLevel) {
      case 0: return llvm::CodeGenOptLevel::None;
      case 1: return llvm::CodeGenOptLevel::Less;
      case 3: return llvm::CodeGenOptLevel::Aggressive;
      default: return llvm::CodeGenOptLevel::Default;
      }
    }();

    std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
        triple, /*CPU=*/"generic", /*Features=*/"", targetOpts,
        /*RM=*/std::nullopt, /*CM=*/std::nullopt, optLevelEnum));

    if (!tm) {
      moduleOp.emitError("failed to create wasm32 TargetMachine");
      signalPassFailure();
      return;
    }

    llvmModule->setDataLayout(tm->createDataLayout());

    // Mark all functions as exported for wasm-ld visibility.
    for (auto &func : *llvmModule) {
      if (!func.isDeclaration())
        func.setVisibility(llvm::GlobalValue::DefaultVisibility);
    }

    // 4. Emit object code to buffer
    llvm::SmallString<0> objBuf;
    llvm::raw_svector_ostream objStream(objBuf);
    llvm::legacy::PassManager pm;

    if (tm->addPassesToEmitFile(pm, objStream, nullptr,
                                llvm::CodeGenFileType::ObjectFile)) {
      moduleOp.emitError("wasm32 target cannot emit object files");
      signalPassFailure();
      return;
    }
    pm.run(*llvmModule);

    // 5. Base64-encode the binary and store as module attribute
    auto encoded = llvm::encodeBase64(
        llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(objBuf.data()),
                                objBuf.size()));
    moduleOp->setAttr("lego.wasm_binary",
                       StringAttr::get(moduleOp.getContext(), encoded));
  }
};

} // namespace

// ============================================================================
// Pipeline definition
// ============================================================================

namespace mlir {
namespace lego {

void buildLegoToWasmPipelineImpl(OpPassManager &pm, int optLevel) {
  // Phase 1: Lower LEGO ops to arith (shared with all backends).
  buildLegoLowerPipeline(pm);

  // Phase 2: Lower to LLVM dialect (same as CPU path).
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Phase 3: Translate LLVM dialect to WASM binary.
  pm.addPass(std::make_unique<SerializeWasmPass>(optLevel));
}

void buildLegoToWasmPipeline(OpPassManager &pm,
                              const LegoToWasmPipelineOptions &options) {
  buildLegoToWasmPipelineImpl(pm, options.optLevel);
}

void buildLegoToWasmPipeline(OpPassManager &pm, int optLevel) {
  buildLegoToWasmPipelineImpl(pm, optLevel);
}

} // namespace lego
} // namespace mlir

#endif // LEGO_HAS_WASM
