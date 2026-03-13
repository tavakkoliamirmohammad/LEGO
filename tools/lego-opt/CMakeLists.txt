# MLIRLego transitively provides LLVM lowering, dialect, and JIT deps.
# Only list libs that lego-opt needs directly beyond what MLIRLego provides.
set(LIBS
  MLIRLego
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIROptLib
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRMathDialect
  MLIRArithTransforms
)

add_llvm_executable(lego-opt lego-opt.cpp)
llvm_update_compile_flags(lego-opt)
mlir_target_link_libraries(lego-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(lego-opt)
