add_mlir_public_c_api_library(MLIRLegoCAPI
  Dialects.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCAPIIR
  MLIRLego
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  # LLVM IR translation interfaces (for legoRegisterLLVMTranslations)
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
)
