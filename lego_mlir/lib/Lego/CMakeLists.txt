include_directories(${CMAKE_BINARY_DIR}/lego_mlir/include)

add_mlir_dialect_library(MLIRLego
  LegoDialect.cpp
  LegoOps.cpp
  LegoDesugar.cpp
  LegoToArith.cpp
  LegoVerifyConsistency.cpp

  DEPENDS
  MLIRLegoDialectIncGen
  MLIRLegoOpsIncGen
  MLIRLegoPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransformDialect
  MLIRTransformDialectInterfaces
  MLIRSideEffectInterfaces
)
