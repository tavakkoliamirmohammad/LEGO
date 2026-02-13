set(LLVM_TARGET_DEFINITIONS LegoDialect.td)
mlir_tablegen(LegoDialect.h.inc -gen-dialect-decls)
mlir_tablegen(LegoDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(LegoOpsTypes.h.inc -gen-typedef-decls)
mlir_tablegen(LegoOpsTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRLegoDialectIncGen)

set(LLVM_TARGET_DEFINITIONS LegoOps.td)
mlir_tablegen(LegoOps.h.inc -gen-op-decls)
mlir_tablegen(LegoOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRLegoOpsIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
# mlir_tablegen(Passes.cpp.inc -gen-pass-defs) # Optional if we need definitions
add_public_tablegen_target(MLIRLegoPassesIncGen)
