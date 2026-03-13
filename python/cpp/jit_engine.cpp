//===- jit_engine.cpp - nanobind wrapper for LegoJITEngine ----------------===//
//
// Exposes the LEGO JIT engine to Python as _lego_jit.CompiledLayout.
//
//===----------------------------------------------------------------------===//

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "../../lib/Lego/LegoJITEngine.h"

namespace nb = nanobind;

NB_MODULE(_lego_jit, m) {
  m.doc() = "LEGO MLIR JIT compilation engine";

  nb::class_<mlir::lego::LegoJITEngine>(m, "CompiledLayout")
      .def_static(
          "from_mlir",
          [](const std::string &mlirText) {
            std::string errorMsg;
            auto engine =
                mlir::lego::LegoJITEngine::create(mlirText, &errorMsg);
            if (!engine)
              throw std::runtime_error("JIT compilation failed: " + errorMsg);
            return engine.release();
          },
          nb::rv_policy::take_ownership,
          nb::arg("mlir_text"),
          "Parse MLIR text, lower through lego-to-llvm, and JIT compile.")
      .def(
          "invoke",
          [](mlir::lego::LegoJITEngine &self, uintptr_t srcPtr,
             uintptr_t dstPtr, int64_t numElements) {
            bool ok = self.invoke("transform", reinterpret_cast<void *>(srcPtr),
                                  reinterpret_cast<void *>(dstPtr),
                                  numElements);
            if (!ok)
              throw std::runtime_error("JIT invocation failed");
          },
          nb::arg("src_ptr"), nb::arg("dst_ptr"), nb::arg("num_elements"),
          "Invoke the 'transform' function with src/dst pointers and count.")
      .def(
          "invoke_named",
          [](mlir::lego::LegoJITEngine &self, const std::string &funcName,
             uintptr_t srcPtr, uintptr_t dstPtr, int64_t numElements) {
            bool ok = self.invoke(funcName, reinterpret_cast<void *>(srcPtr),
                                  reinterpret_cast<void *>(dstPtr),
                                  numElements);
            if (!ok)
              throw std::runtime_error("JIT invocation of '" + funcName +
                                       "' failed");
          },
          nb::arg("func_name"), nb::arg("src_ptr"), nb::arg("dst_ptr"),
          nb::arg("num_elements"),
          "Invoke a named function with src/dst pointers and count.");
}
