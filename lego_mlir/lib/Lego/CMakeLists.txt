add_mlir_dialect_library(MLIRLego
  LegoDialect.cpp
  LegoOps.cpp

  DEPENDS
  MLIRLegoDialectIncGen
  MLIRLegoOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransformDialect
  MLIRTransformDialectInterfaces
  MLIRSideEffectInterfaces
)
