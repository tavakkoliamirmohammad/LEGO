add_subdirectory(include/Lego)
add_subdirectory(lib/Lego)
add_subdirectory(tools/lego-opt)
if(LLVM_INCLUDE_TESTS)
  add_subdirectory(test)
endif()

