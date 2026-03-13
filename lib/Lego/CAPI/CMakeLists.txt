# MLIRLego transitively provides dialect and LLVM translation deps.
add_mlir_public_c_api_library(MLIRLegoCAPI
  Dialects.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRLego
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRMemRefDialect
)
