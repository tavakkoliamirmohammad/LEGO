/// LEGO WASM Driver — runs the full MLIR/LEGO compiler in the browser.
///
/// Exposes C functions callable from JavaScript:
///   - lego_compile(mlir_text, M, N) -> JSON with IR + mapping
///   - lego_free(ptr) -> free returned memory
///
/// After lowering to arith, evaluates @apply(i,j) for all (i,j) in
/// [0,M)×[0,N) and returns the complete mapping alongside the IR.
///
/// Compiled with Emscripten to produce lego_driver.js + lego_driver.wasm.

#include "Lego/LegoOps.h"
#include "Lego/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
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
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ADT/DenseMap.h"

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
// Arith evaluator — run the lowered @apply for concrete (i,j)
// ============================================================================

static int64_t evalArith(func::FuncOp func, int64_t i, int64_t j) {
  llvm::DenseMap<Value, int64_t> v;
  auto &entry = func.getBody().front();
  v[entry.getArgument(0)] = i;
  if (entry.getNumArguments() > 1)
    v[entry.getArgument(1)] = j;

  for (auto &op : entry.without_terminator()) {
    if (auto c = dyn_cast<arith::ConstantOp>(op))
      v[c.getResult()] = cast<IntegerAttr>(c.getValue()).getInt();
    else if (auto o = dyn_cast<arith::AddIOp>(op))
      v[o.getResult()] = v[o.getLhs()] + v[o.getRhs()];
    else if (auto o = dyn_cast<arith::SubIOp>(op))
      v[o.getResult()] = v[o.getLhs()] - v[o.getRhs()];
    else if (auto o = dyn_cast<arith::MulIOp>(op))
      v[o.getResult()] = v[o.getLhs()] * v[o.getRhs()];
    else if (auto o = dyn_cast<arith::DivUIOp>(op))
      v[o.getResult()] = (uint64_t)v[o.getLhs()] / (uint64_t)v[o.getRhs()];
    else if (auto o = dyn_cast<arith::RemUIOp>(op))
      v[o.getResult()] = (uint64_t)v[o.getLhs()] % (uint64_t)v[o.getRhs()];
    else if (auto o = dyn_cast<arith::ShLIOp>(op))
      v[o.getResult()] = v[o.getLhs()] << v[o.getRhs()];
    else if (auto o = dyn_cast<arith::ShRUIOp>(op))
      v[o.getResult()] = (uint64_t)v[o.getLhs()] >> v[o.getRhs()];
    else if (auto o = dyn_cast<arith::ShRSIOp>(op))
      v[o.getResult()] = v[o.getLhs()] >> v[o.getRhs()];
    else if (auto o = dyn_cast<arith::AndIOp>(op))
      v[o.getResult()] = v[o.getLhs()] & v[o.getRhs()];
    else if (auto o = dyn_cast<arith::OrIOp>(op))
      v[o.getResult()] = v[o.getLhs()] | v[o.getRhs()];
    else if (auto o = dyn_cast<arith::XOrIOp>(op))
      v[o.getResult()] = v[o.getLhs()] ^ v[o.getRhs()];
    else if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp,
                 arith::IndexCastOp, arith::IndexCastUIOp>(op))
      v[op.getResult(0)] = v[op.getOperand(0)];
  }

  auto ret = cast<func::ReturnOp>(entry.getTerminator());
  return v[ret.getOperand(0)];
}

// ============================================================================
// Exported: compile MLIR + compute mapping
// ============================================================================

extern "C" EXPORT
char *lego_compile(const char *mlir_text, int M, int N) {
  auto *ctx = getGlobalContext();

  auto moduleRef = parseSourceString<ModuleOp>(mlir_text, ctx);
  if (!moduleRef)
    return toCString("{\"error\": \"Failed to parse MLIR\"}");

  auto cloneLower = moduleRef->clone();
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

  // Stage 2: After lego-lower
  {
    PassManager pm(ctx);
    lego::buildLegoLowerPipeline(pm);
    if (failed(pm.run(cloneLower)))
      return toCString("{\"error\": \"lego-lower failed\"}");
  }
  std::string arithIR = moduleToString(cloneLower.getOperation());

  // Evaluate apply(i,j) for all (i,j) in [0,M)×[0,N)
  std::string mappingJSON;
  if (M > 0 && N > 0) {
    auto fn = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        cloneLower.getOperation(), StringAttr::get(ctx, "apply"));
    if (fn) {
      mappingJSON.reserve(M * N * 16);
      mappingJSON = "[";
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          if (i || j) mappingJSON += ',';
          mappingJSON += '[';
          mappingJSON += std::to_string(i);
          mappingJSON += ',';
          mappingJSON += std::to_string(j);
          mappingJSON += ',';
          mappingJSON += std::to_string(evalArith(fn, i, j));
          mappingJSON += ']';
        }
      }
      mappingJSON += ']';
    }
  }

  // Stage 3: After lego-to-llvm
  {
    PassManager pm(ctx);
    lego::buildLegoToLLVMPipeline(pm);
    if (failed(pm.run(cloneLLVM)))
      return toCString("{\"error\": \"lego-to-llvm failed\"}");
  }
  std::string llvmIR = moduleToString(cloneLLVM.getOperation());

  // JSON response
  std::string json = "{";
  json += "\"lego\":\"" + escapeJSON(legoIR) + "\",";
  json += "\"arith\":\"" + escapeJSON(arithIR) + "\",";
  json += "\"llvm\":\"" + escapeJSON(llvmIR) + "\"";
  if (!mappingJSON.empty()) {
    json += ",\"mapping\":" + mappingJSON;
    json += ",\"shape\":[" + std::to_string(M) + "," + std::to_string(N) + "]";
  }
  json += "}";

  cloneLower.getOperation()->erase();
  cloneLLVM.getOperation()->erase();

  return toCString(json);
}

extern "C" EXPORT
void lego_free(char *ptr) {
  free(ptr);
}
