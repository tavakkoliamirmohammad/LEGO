import os
import sys
import re
import platform
import subprocess

import lit.util
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# name: The name of this test suite.
config.name = 'LEGO'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.lego_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in the suite.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [config.lego_tools_dir, config.llvm_tools_dir]
tools = [
    'lego-opt'
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Feature flags for optional tests
# cuda-runner: available when mlir-runner + CUDA runtime libs exist
mlir_runner = os.path.join(config.llvm_tools_dir, 'mlir-runner')
if os.path.isfile(mlir_runner):
    config.available_features.add('cuda-runner')
    # Substitutions for shared libs used by GPU execution tests
    cuda_rt = os.path.join(config.llvm_tools_dir, '..', 'lib', 'libmlir_cuda_runtime.so')
    c_runner = os.path.join(config.llvm_tools_dir, '..', 'lib', 'libmlir_c_runner_utils.so')
    runner_utils = os.path.join(config.llvm_tools_dir, '..', 'lib', 'libmlir_runner_utils.so')
    config.substitutions.append(('%mlir_cuda_runtime', cuda_rt))
    config.substitutions.append(('%mlir_c_runner_utils', c_runner))
    config.substitutions.append(('%mlir_runner_utils', runner_utils))
    llvm_config.add_tool_substitutions(['mlir-runner'], tool_dirs)
