//===- LegoDialectPybind.cpp - nanobind LEGO dialect registration ---------===//
//
// Registers the LEGO dialect, required standard dialects, LEGO passes, and
// LLVM translation interfaces with MLIR's Python context using nanobind.
//
// Exposes PyConcreteType bindings for LayoutType and ViewType.
//
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"
#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/SCF.h"
#include "mlir-c/Dialect/MemRef.h"
#include "mlir-c/Dialect/GPU.h"
#include "mlir-c/Dialect/SPIRV.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "Lego/CAPI/Dialects.h"

namespace nb = nanobind;
using namespace nanobind::literals;
using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

//===----------------------------------------------------------------------===//
// PyConcreteType bindings for LEGO types
//===----------------------------------------------------------------------===//

struct PyLayoutType : PyConcreteType<PyLayoutType> {
  static constexpr IsAFunctionTy isaFunction = mlirLegoTypeIsALayout;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirLegoLayoutTypeGetTypeID;
  static constexpr const char *pyClassName = "LayoutType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return PyLayoutType(context->getRef(),
                              mlirLegoLayoutTypeGet(context.get()->get()));
        },
        "Get a !lego.layout type.", nb::arg("context").none() = nb::none());
  }
};

struct PyViewType : PyConcreteType<PyViewType> {
  static constexpr IsAFunctionTy isaFunction = mlirLegoTypeIsAView;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirLegoViewTypeGetTypeID;
  static constexpr const char *pyClassName = "ViewType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elementType, DefaultingPyMlirContext context) {
          return PyViewType(
              context->getRef(),
              mlirLegoViewTypeGet(context.get()->get(), elementType));
        },
        "Get a !lego.view<elementType> type.", "element_type"_a,
        nb::arg("context").none() = nb::none());
    c.def_prop_ro("element_type", [](const PyViewType &self) {
      return mlirLegoViewTypeGetElementType(self);
    });
  }
};

//===----------------------------------------------------------------------===//
// Module definition
//===----------------------------------------------------------------------===//

static void registerAndLoad(MlirDialectHandle handle, MlirContext ctx) {
  mlirDialectHandleRegisterDialect(handle, ctx);
  mlirDialectHandleLoadDialect(handle, ctx);
}

NB_MODULE(_legoDialects, m) {
  m.doc() = "LEGO MLIR dialect Python bindings";

  //===--------------------------------------------------------------------===//
  // LEGO type bindings (lego submodule)
  //===--------------------------------------------------------------------===//
  auto legoM = m.def_submodule("lego");
  PyLayoutType::bind(legoM);
  PyViewType::bind(legoM);

  //===--------------------------------------------------------------------===//
  // Dialect registration
  //===--------------------------------------------------------------------===//
  m.def(
      "register_lego_dialect",
      [](MlirContext ctx) {
        // Register LEGO passes and pipelines (idempotent)
        legoRegisterPasses();

        // Register ConvertToLLVMPatternInterface extensions BEFORE loading
        // dialects. This enables GPU→ROCDL/NVVM passes to discover LLVM
        // conversion patterns for memref/arith/cf/func/index/math.
        MlirDialectRegistry registry = mlirDialectRegistryCreate();
        legoRegisterConvertToLLVMExtensions(registry);
        mlirContextAppendDialectRegistry(ctx, registry);
        mlirDialectRegistryDestroy(registry);

        // Register LLVM IR translations (needed for ExecutionEngine)
        legoRegisterLLVMTranslations(ctx);

        // Register and load the LEGO dialect
        registerAndLoad(mlirGetDialectHandle__lego__(), ctx);

        // Register standard dialects used by the LEGO compiler
        registerAndLoad(mlirGetDialectHandle__arith__(), ctx);
        registerAndLoad(mlirGetDialectHandle__func__(), ctx);
        registerAndLoad(mlirGetDialectHandle__scf__(), ctx);
        registerAndLoad(mlirGetDialectHandle__memref__(), ctx);
        // GPU + SPIR-V dialects (always available, no GPU hardware needed)
        registerAndLoad(mlirGetDialectHandle__gpu__(), ctx);
        registerAndLoad(mlirGetDialectHandle__spirv__(), ctx);
        // Math dialect (needed for math.exp etc. in GPU kernels)
        registerAndLoad(mlirGetDialectHandle__math__(), ctx);
      },
      nb::arg("context"),
      "Register and load the LEGO dialect and required standard dialects.");
}
