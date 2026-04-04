import os
import subprocess
import lit.formats

config.name = 'LEGO-Puzzles'
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.py']
config.excludes = ['__pycache__', 'bench_utils.py', 'lit.cfg.py', 'lit.site.cfg.py',
                   'puzzle_09_debugging.py']

# Tests live in the puzzles source directory.
config.test_source_root = config.puzzles_dir
config.test_exec_root = os.path.join(config.lego_obj_root, 'test', 'puzzles')

# Substitutions used in puzzle RUN: lines.
config.substitutions.append(('%{pythonpath}',
    config.pythonpath + ':' + config.puzzles_dir))
config.substitutions.append(('%{mlir_build_dir}', config.mlir_build_dir))
config.substitutions.append(('%{python}', config.python_executable))

# GPU features — puzzles can run on NVIDIA (CUDA), AMD (ROCm), or any
# machine with wgpu (Vulkan/WebGPU/Metal).
if config.host_has_nvidia_gpu:
    config.available_features.add('nvidia-gpu')
if config.host_has_amd_gpu:
    config.available_features.add('amd-gpu')

# wgpu/Vulkan/Metal — probe at lit-time so puzzles can run on any GPU
try:
    subprocess.run([config.python_executable, '-c', 'import wgpu'],
                   check=True, capture_output=True)
    config.available_features.add('wgpu')
except (subprocess.CalledProcessError, FileNotFoundError):
    pass

# "gpu" feature: any GPU backend available
if config.available_features & {'nvidia-gpu', 'amd-gpu', 'wgpu'}:
    config.available_features.add('gpu')

# Each puzzle test uses ~1.7 GB RAM + GPU resources.
# Limit parallelism to avoid OOM kills when many tests run concurrently.
lit_config.parallelism_groups['gpu-puzzles'] = 4
config.parallelism_group = 'gpu-puzzles'
