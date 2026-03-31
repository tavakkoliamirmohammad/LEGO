/// LEGO WASM Driver — runs the full MLIR/LEGO compiler in the browser.
///
/// Exposes C functions callable from JavaScript:
///   - lego_compile(mlir_text, M, N) -> JSON with IR stages + WASM binary
///   - lego_free(ptr) -> free returned memory
///
/// Compiles the MLIR through LEGO -> Arith -> LLVM dialect, then
/// translates to LLVM IR and emits a standalone wasm32 module via
/// LLVM's WebAssembly backend.  JavaScript instantiates the module
/// and calls the exported apply(i,j) function natively.
///
/// Compiled with Emscripten to produce lego_driver.js + lego_driver.wasm.

#include "Lego/LegoOps.h"
#include "Lego/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include <string>
#include <cstring>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

using namespace mlir;

// ============================================================================
// Global MLIR context (initialized once)
// ============================================================================

static MLIRContext *getGlobalContext() {
  static MLIRContext *ctx = nullptr;
  if (!ctx) {
    ctx = new MLIRContext();
    ctx->loadDialect<lego::LegoDialect>();
    ctx->loadDialect<arith::ArithDialect>();
    ctx->loadDialect<func::FuncDialect>();
    ctx->loadDialect<scf::SCFDialect>();
    ctx->loadDialect<memref::MemRefDialect>();
    ctx->loadDialect<index::IndexDialect>();
    ctx->loadDialect<cf::ControlFlowDialect>();
    ctx->loadDialect<LLVM::LLVMDialect>();
    registerBuiltinDialectTranslation(*ctx);
    registerLLVMDialectTranslation(*ctx);
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllTargetInfos();
  }
  return ctx;
}

// ============================================================================
// Helpers
// ============================================================================

static std::string moduleToString(Operation *op) {
  std::string str;
  llvm::raw_string_ostream os(str);
  op->print(os);
  return str;
}

static std::string escapeJSON(const std::string &s) {
  std::string result;
  result.reserve(s.size() + 64);
  for (char c : s) {
    switch (c) {
    case '"':  result += "\\\""; break;
    case '\\': result += "\\\\"; break;
    case '\n': result += "\\n"; break;
    case '\r': result += "\\r"; break;
    case '\t': result += "\\t"; break;
    default:   result += c;
    }
  }
  return result;
}

static char *toCString(const std::string &s) {
  char *buf = (char *)malloc(s.size() + 1);
  memcpy(buf, s.data(), s.size());
  buf[s.size()] = '\0';
  return buf;
}

// ============================================================================
// WASM binary helpers — LEB128 encoding and object-to-module conversion
// ============================================================================

static void writeLEB128(llvm::SmallVectorImpl<uint8_t> &out, uint32_t val) {
  do {
    uint8_t byte = val & 0x7F;
    val >>= 7;
    if (val) byte |= 0x80;
    out.push_back(byte);
  } while (val);
}

static uint32_t readLEB128(const uint8_t *&ptr) {
  uint32_t result = 0;
  unsigned shift = 0;
  while (true) {
    uint8_t byte = *ptr++;
    result |= (uint32_t)(byte & 0x7F) << shift;
    shift += 7;
    if (!(byte & 0x80)) break;
  }
  return result;
}

/// Convert a wasm32 relocatable object into a standalone .wasm module.
///
/// For pure-arithmetic functions (no memory, no imports, no function calls),
/// the relocatable object's Code section has zero relocations, so we can
/// simply copy the standard sections and add an Export section.
static bool buildStandaloneWasm(llvm::ArrayRef<uint8_t> objBytes,
                                llvm::SmallVectorImpl<uint8_t> &out) {
  if (objBytes.size() < 8) return false;
  if (memcmp(objBytes.data(), "\0asm\1\0\0\0", 8) != 0) return false;

  // Parse all sections from the relocatable object.
  struct Section {
    uint8_t id;
    llvm::ArrayRef<uint8_t> data;
  };
  llvm::SmallVector<Section, 8> sections;

  const uint8_t *ptr = objBytes.data() + 8;
  const uint8_t *end = objBytes.data() + objBytes.size();
  while (ptr < end) {
    uint8_t id = *ptr++;
    uint32_t size = readLEB128(ptr);
    sections.push_back({id, llvm::ArrayRef<uint8_t>(ptr, size)});
    ptr += size;
  }

  // Build export section: export function 0 as "apply".
  llvm::SmallVector<uint8_t, 16> exportPayload;
  writeLEB128(exportPayload, 1);          // 1 export
  writeLEB128(exportPayload, 5);          // name length = 5
  exportPayload.append({'a','p','p','l','y'});
  exportPayload.push_back(0x00);          // kind: function
  writeLEB128(exportPayload, 0);          // function index 0

  // Helper to write a section.
  auto writeSection = [&](uint8_t id, llvm::ArrayRef<uint8_t> data) {
    out.push_back(id);
    writeLEB128(out, data.size());
    out.append(data.begin(), data.end());
  };

  // Write module header.
  out.append({'\0', 'a', 's', 'm', 1, 0, 0, 0});

  // Copy all standard sections (skip custom sections: id=0), inserting
  // the Export section (id=7) in the correct position.
  bool wroteExport = false;
  for (auto &sec : sections) {
    if (sec.id == 0) continue; // skip custom sections (linking, reloc.*)

    // Sections must appear in ascending id order.  Insert Export (7)
    // right before the first section with id > 7.
    if (!wroteExport && sec.id > 7) {
      writeSection(7, exportPayload);
      wroteExport = true;
    }
    writeSection(sec.id, sec.data);
  }
  if (!wroteExport) writeSection(7, exportPayload);

  return true;
}

// ============================================================================
// WASM binary generation — LLVM dialect -> LLVM IR -> wasm32 object -> module
// ============================================================================

static std::string emitWasmBinary(ModuleOp moduleOp) {
  // 1. Translate MLIR LLVM dialect to LLVM IR.
  llvm::LLVMContext llvmCtx;
  auto llvmModule = translateModuleToLLVMIR(moduleOp, llvmCtx);
  if (!llvmModule) return "";

  // 2. Set up the wasm32 target.
  llvm::Triple triple("wasm32-unknown-unknown");
  llvmModule->setTargetTriple(triple);

  std::string error;
  const auto *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) return "";

  // 3. Create target machine.
  llvm::TargetOptions targetOpts;
  std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
      triple, /*CPU=*/"generic", /*Features=*/"", targetOpts,
      /*RM=*/std::nullopt, /*CM=*/std::nullopt,
      llvm::CodeGenOptLevel::Default));
  if (!tm) return "";

  llvmModule->setDataLayout(tm->createDataLayout());

  // Mark all functions as exported so the linker/builder sees them.
  for (auto &func : *llvmModule)
    if (!func.isDeclaration())
      func.setVisibility(llvm::GlobalValue::DefaultVisibility);

  // 4. Emit wasm32 relocatable object code.
  llvm::SmallString<0> objBuf;
  llvm::raw_svector_ostream objStream(objBuf);
  llvm::legacy::PassManager pm;
  if (tm->addPassesToEmitFile(pm, objStream, nullptr,
                              llvm::CodeGenFileType::ObjectFile))
    return "";
  pm.run(*llvmModule);

  // 5. Convert relocatable object to standalone wasm module.
  llvm::SmallVector<uint8_t, 0> wasmBuf;
  if (!buildStandaloneWasm(
          llvm::ArrayRef<uint8_t>(
              reinterpret_cast<const uint8_t *>(objBuf.data()), objBuf.size()),
          wasmBuf))
    return "";

  // 6. Base64-encode the standalone module.
  return llvm::encodeBase64(wasmBuf);
}

// ============================================================================
// Exported: compile MLIR -> IR stages + WASM binary
// ============================================================================

extern "C" EXPORT
char *lego_compile(const char *mlir_text, int M, int N) {
  auto *ctx = getGlobalContext();

  auto moduleRef = parseSourceString<ModuleOp>(mlir_text, ctx);
  if (!moduleRef)
    return toCString("{\"error\": \"Failed to parse MLIR\"}");

  auto cloneArith = moduleRef->clone();
  auto cloneLLVM = moduleRef->clone();

  // Stage 1: LEGO dialect after canonicalize + CSE
  {
    PassManager pm(ctx);
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(pm.run(*moduleRef)))
      return toCString("{\"error\": \"canonicalize+cse failed\"}");
  }
  std::string legoIR = moduleToString(moduleRef->getOperation());

  // Stage 2: After lego-lower (-> arith)
  {
    PassManager pm(ctx);
    lego::buildLegoLowerPipeline(pm);
    if (failed(pm.run(cloneArith)))
      return toCString("{\"error\": \"lego-lower failed\"}");
  }
  std::string arithIR = moduleToString(cloneArith.getOperation());

  // Stage 3: After lego-to-llvm (-> LLVM dialect)
  {
    PassManager pm(ctx);
    lego::buildLegoToLLVMPipeline(pm);
    if (failed(pm.run(cloneLLVM)))
      return toCString("{\"error\": \"lego-to-llvm failed\"}");
  }
  std::string llvmIR = moduleToString(cloneLLVM.getOperation());

  // Stage 4: Generate standalone WASM binary from the LLVM dialect module.
  std::string wasmB64 =
      emitWasmBinary(cast<ModuleOp>(cloneLLVM.getOperation()));

  // Build JSON response.
  std::string json = "{";
  json += "\"lego\":\"" + escapeJSON(legoIR) + "\",";
  json += "\"arith\":\"" + escapeJSON(arithIR) + "\",";
  json += "\"llvm\":\"" + escapeJSON(llvmIR) + "\"";
  if (!wasmB64.empty()) {
    json += ",\"wasmBinary\":\"" + wasmB64 + "\"";
  }
  json += ",\"shape\":[" + std::to_string(M) + "," + std::to_string(N) + "]";
  json += "}";

  cloneArith.getOperation()->erase();
  cloneLLVM.getOperation()->erase();

  return toCString(json);
}

extern "C" EXPORT
void lego_free(char *ptr) {
  free(ptr);
}
