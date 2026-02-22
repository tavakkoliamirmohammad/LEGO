
llvm_canonicalize_cmake_booleans(
  LLVM_BUILD_EXAMPLES
  )

set(LEGO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(LEGO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/..)
set(LEGO_TOOLS_DIR ${CMAKE_BINARY_DIR}/tools/lego-opt)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

# Check for lit in system
if(NOT LLVM_EXTERNAL_LIT)
  find_program(LLVM_LIT_PROGRAM lit)
  if(LLVM_LIT_PROGRAM)
    set(LLVM_EXTERNAL_LIT ${LLVM_LIT_PROGRAM} CACHE STRING "Command to run lit" FORCE)
  endif()
endif()

set(LEGO_TEST_DEPENDS lego-opt)
if(TARGET FileCheck)
  list(APPEND LEGO_TEST_DEPENDS FileCheck)
endif()
if(TARGET count)
  list(APPEND LEGO_TEST_DEPENDS count)
endif()
if(TARGET not)
  list(APPEND LEGO_TEST_DEPENDS not)
endif()

add_lit_testsuite(check-lego "Running the Lego regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  ${LEGO_TEST_DEPENDS}
  )
