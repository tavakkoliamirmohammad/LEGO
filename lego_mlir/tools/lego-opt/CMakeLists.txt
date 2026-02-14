set(LIBS
  MLIRLego
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRDialect
  MLIRTransformDialect
  MLIROptLib
  # MLIRRegisterAllDialects
  # MLIRRegisterAllExtensions
  # MLIRRegisterAllPasses
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRMathDialect
)

add_llvm_executable(lego-opt lego-opt.cpp)
llvm_update_compile_flags(lego-opt)
mlir_target_link_libraries(lego-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(lego-opt)
