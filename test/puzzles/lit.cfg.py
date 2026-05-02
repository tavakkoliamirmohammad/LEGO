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

# Per-backend GPU features — puzzles use REQUIRES: to select which tests run.
if config.enable_cuda_runner:
    config.available_features.add('nvidia-gpu')
    config.available_features.add('cuda')

if config.enable_rocm_runner:
    config.available_features.add('amd-gpu')
    config.available_features.add('rocm')

if config.enable_metal_runner:
    config.available_features.add('metal')

# wgpu/Vulkan/WebGPU — probe at lit-time (available on any GPU via wgpu)
try:
    subprocess.run([config.python_executable, '-c', 'import wgpu'],
                   check=True, capture_output=True)
    config.available_features.add('wgpu')
except (subprocess.CalledProcessError, FileNotFoundError):
    pass

# "gpu" feature: any runner enabled → puzzles with REQUIRES: gpu will run
if config.enable_cuda_runner or config.enable_rocm_runner or config.enable_metal_runner:
    config.available_features.add('gpu')

# SPIR-V execution: available if any GPU backend is enabled (SPIR-V runs on
# NVIDIA via Vulkan, AMD via Vulkan, Apple via Metal/MoltenVK)
if config.available_features & {'gpu', 'wgpu'}:
    config.available_features.add('spirv')

# Each puzzle test uses ~1.7 GB RAM + GPU resources.
# Limit parallelism to avoid OOM kills when many tests run concurrently.
# Reduced from 2 to 1 because puzzle_25/26 (warp shuffle) intermittently
# SIGBUS at >1 concurrent on shared CHPC nodes; serialized runs are stable.
lit_config.parallelism_groups['gpu-puzzles'] = 1
config.parallelism_group = 'gpu-puzzles'
