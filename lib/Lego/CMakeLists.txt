include_directories(${CMAKE_BINARY_DIR}/include)

add_mlir_dialect_library(MLIRLego
  LegoDialect.cpp
  LegoOps.cpp
  LegoDesugar.cpp
  LegoToArith.cpp
  LegoVerifyConsistency.cpp
  LegoArithSimplification.cpp
  LegoGenerateBoundsChecks.cpp
  LegoExternalSMTVerifier.cpp
  Passes.cpp

  DEPENDS
  MLIRLegoDialectIncGen
  MLIRLegoOpsIncGen
  MLIRLegoPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSMT
  MLIRExportSMTLIB
)
