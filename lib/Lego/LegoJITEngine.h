//===- LegoJITEngine.h - JIT compilation engine for LEGO IR -----*- C++ -*-===//
//
// Wraps MLIR's ExecutionEngine to JIT-compile LEGO IR modules that have been
// lowered to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LEGO_LEGOJITENGINE_H
#define LEGO_LEGOJITENGINE_H

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <memory>
#include <string>

namespace mlir {
namespace lego {

/// JIT engine that takes LEGO MLIR text, runs the lego-to-llvm pipeline,
/// and compiles it for execution.
class LegoJITEngine {
public:
  /// Construct from MLIR assembly text. Parses, lowers, and JIT-compiles.
  /// Returns nullptr on failure (check getError()).
  static std::unique_ptr<LegoJITEngine> create(const std::string &mlirText,
                                                std::string *errorMsg = nullptr);

  /// Invoke a function by name, passing raw pointers for src, dst, and count.
  /// The function must have signature: (memref<?xf32>, memref<?xf32>, index) -> ()
  /// or a bare-pointer equivalent: (f32*, f32*, i64) -> void
  bool invoke(const std::string &funcName, void *srcPtr, void *dstPtr,
              int64_t numElements);

  /// Get the underlying ExecutionEngine (for advanced use).
  ExecutionEngine *getEngine() { return engine.get(); }

  ~LegoJITEngine();

private:
  LegoJITEngine() = default;
  std::unique_ptr<MLIRContext> ctx;
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<ExecutionEngine> engine;
};

} // namespace lego
} // namespace mlir

#endif // LEGO_LEGOJITENGINE_H
