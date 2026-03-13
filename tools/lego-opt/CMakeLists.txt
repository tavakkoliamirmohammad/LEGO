set(LIBS
  MLIRLego
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRDialect
  MLIROptLib
  # MLIRRegisterAllDialects
  # MLIRRegisterAllExtensions
  # MLIRRegisterAllPasses
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRMathDialect
  MLIRArithTransforms
  # LLVM lowering pipeline deps
  MLIRControlFlowDialect
  MLIRLLVMDialect
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
)

add_llvm_executable(lego-opt lego-opt.cpp)
llvm_update_compile_flags(lego-opt)
mlir_target_link_libraries(lego-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(lego-opt)
