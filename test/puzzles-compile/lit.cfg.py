"""Lit config for compile-only puzzle tests.

Runs the same puzzles as test/puzzles/ but with LEGO_COMPILE_ONLY=1,
so only compilation to all GPU backends is tested — no GPU hardware needed.
"""
import os
import lit.formats

config.name = 'LEGO-Puzzles-Compile'
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.py']
config.excludes = ['__pycache__', 'bench_utils.py', 'lit.cfg.py', 'lit.site.cfg.py',
                   'puzzle_09_debugging.py']

# Tests live in the puzzles source directory (shared with the GPU suite).
config.test_source_root = config.puzzles_dir
config.test_exec_root = os.path.join(config.lego_obj_root, 'test', 'puzzles-compile')

# Substitutions — same as GPU suite but with LEGO_COMPILE_ONLY=1.
config.substitutions.append(('%{pythonpath}',
    config.pythonpath + ':' + config.puzzles_dir))
config.substitutions.append(('%{mlir_build_dir}', config.mlir_build_dir))
config.substitutions.append(('%{python}', config.python_executable))

# Enable the "gpu" feature so REQUIRES: gpu tests are NOT skipped.
# The actual GPU execution is skipped by LEGO_COMPILE_ONLY=1 in bench_utils.
config.available_features.add('gpu')

# Inject LEGO_COMPILE_ONLY into the environment for all test processes.
config.environment['LEGO_COMPILE_ONLY'] = '1'
