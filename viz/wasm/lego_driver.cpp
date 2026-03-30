/// LEGO WASM Driver — runs the full MLIR/LEGO compiler in the browser.
///
/// Exposes C functions callable from JavaScript:
///   - lego_compile(mlir_text) -> JSON with IR at each stage + wasm blob
///   - lego_free(ptr) -> free returned memory
///
/// For each layout, returns:
///   1. LEGO dialect IR (after canonicalize + CSE)
///   2. Arith IR (after lego-lower)
///   3. LLVM IR (after lego-to-llvm)
///   4. WASM binary blob (after lego-to-wasm) — browser instantiates this
///      and calls apply(i,j) to compute the mapping
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

#include <string>
#include <cstring>
#include <vector>

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
    lego::registerLegoPipelines();
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

/// Base64 encode binary data for JSON transport.
static std::string base64Encode(const uint8_t *data, size_t len) {
  static const char table[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
  result.reserve(((len + 2) / 3) * 4);
  for (size_t i = 0; i < len; i += 3) {
    uint32_t n = ((uint32_t)data[i]) << 16;
    if (i + 1 < len) n |= ((uint32_t)data[i + 1]) << 8;
    if (i + 2 < len) n |= data[i + 2];
    result += table[(n >> 18) & 0x3F];
    result += table[(n >> 12) & 0x3F];
    result += (i + 1 < len) ? table[(n >> 6) & 0x3F] : '=';
    result += (i + 2 < len) ? table[n & 0x3F] : '=';
  }
  return result;
}

// ============================================================================
// Exported: compile MLIR text through all pipeline stages
// ============================================================================

/// Compile MLIR text and return IR at each pipeline stage + WASM blob.
///
/// Returns a JSON string:
/// {
///   "lego": "...",        // LEGO dialect IR
///   "arith": "...",       // After lego-lower
///   "llvm": "...",        // After lego-to-llvm
///   "wasm_b64": "...",    // Base64-encoded WASM binary (if lego-to-wasm succeeded)
///   "error": "..."        // Only present on failure
/// }
///
/// Caller must free() the returned pointer via lego_free().
extern "C" EXPORT
char *lego_compile(const char *mlir_text) {
  auto *ctx = getGlobalContext();

  auto moduleRef = parseSourceString<ModuleOp>(mlir_text, ctx);
  if (!moduleRef) {
    return toCString("{\"error\": \"Failed to parse MLIR\"}");
  }

  // Clone for each stage (passes mutate in place)
  auto cloneLower = moduleRef->clone();
  auto cloneLLVM = moduleRef->clone();
#ifdef LEGO_HAS_WASM
  auto cloneWasm = moduleRef->clone();
#endif

  // Stage 1: LEGO dialect after canonicalize + CSE
  {
    PassManager pm(ctx);
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(pm.run(*moduleRef))) {
      return toCString("{\"error\": \"canonicalize+cse failed\"}");
    }
  }
  std::string legoIR = moduleToString(moduleRef->getOperation());

  // Stage 2: After lego-lower
  {
    PassManager pm(ctx);
    lego::buildLegoLowerPipeline(pm);
    if (failed(pm.run(cloneLower))) {
      return toCString("{\"error\": \"lego-lower failed\"}");
    }
  }
  std::string arithIR = moduleToString(cloneLower.getOperation());

  // Stage 3: After lego-to-llvm
  {
    PassManager pm(ctx);
    lego::buildLegoToLLVMPipeline(pm);
    if (failed(pm.run(cloneLLVM))) {
      return toCString("{\"error\": \"lego-to-llvm failed\"}");
    }
  }
  std::string llvmIR = moduleToString(cloneLLVM.getOperation());

  // Stage 4: lego-to-wasm (produces binary blob)
  std::string wasmB64;
#ifdef LEGO_HAS_WASM
  {
    PassManager pm(ctx);
    lego::buildLegoToWasmPipeline(pm, lego::LegoToWasmPipelineOptions{});
    if (!failed(pm.run(cloneWasm))) {
      // Extract base64-encoded WASM binary from module attribute
      if (auto attr = cloneWasm.getOperation()->getAttrOfType<StringAttr>(
              "lego.wasm_binary")) {
        wasmB64 = attr.getValue().str();
      }
    }
  }
  cloneWasm.getOperation()->erase();
#endif

  // Build JSON response
  std::string json = "{";
  json += "\"lego\": \"" + escapeJSON(legoIR) + "\", ";
  json += "\"arith\": \"" + escapeJSON(arithIR) + "\", ";
  json += "\"llvm\": \"" + escapeJSON(llvmIR) + "\"";
  if (!wasmB64.empty()) {
    json += ", \"wasm_b64\": \"" + wasmB64 + "\"";
  }
  json += "}";

  cloneLower.getOperation()->erase();
  cloneLLVM.getOperation()->erase();

  return toCString(json);
}

/// Free a string returned by lego_compile.
extern "C" EXPORT
void lego_free(char *ptr) {
  free(ptr);
}
