import os
import lit.formats

config.name = 'LEGO-Puzzles'
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.py']
config.excludes = ['__pycache__', 'bench_utils.py', 'lit.cfg.py', 'lit.site.cfg.py']

# Tests live in the puzzles source directory.
config.test_source_root = config.puzzles_dir
config.test_exec_root = os.path.join(config.lego_obj_root, 'test', 'puzzles')

# Substitutions used in puzzle RUN: lines.
config.substitutions.append(('%{pythonpath}',
    config.pythonpath + ':' + config.puzzles_dir))
config.substitutions.append(('%{mlir_build_dir}', config.mlir_build_dir))
config.substitutions.append(('%{python}', config.python_executable))

# Feature: nvidia-gpu — puzzle tests require a GPU to execute.
if config.host_has_nvidia_gpu:
    config.available_features.add('nvidia-gpu')
