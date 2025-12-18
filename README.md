
# LEGO: A Layout Expression Language for Code Generation of Hierarchical Mapping

This repository contains the source code of the **LEGO** framework and the scripts used to execute and evaluate all benchmarks in the paper. LEGO provides an algebraic, compiler-agnostic framework for specifying and transforming memory layouts. Through integrations with Triton, CUDA, and MLIR, we compare LEGO-generated kernels with existing implementations and demonstrate that careful data layout reorganization can achieve state-of-the-art performance or significantly improve performance.

## Publication
LEGO is based on the following research work:

> **LEGO: A Layout Expression Language for Code Generation of Hierarchical Mapping**  
> Amir Mohammad Tavakkoli, Cosmin Oancea, and Mary Hall.  
> https://arxiv.org/pdf/2505.08091
> 
## Repository

- **URL:** https://github.com/tavakkoliamirmohammad/lego
- **Paper** https://arxiv.org/pdf/2505.08091



## Main Contributions

- **C1:** A general, simple, and easy-to-use abstraction for bijective layouts, expressing both computation and data.
- **C2:** A fully reproducible implementation of the abstraction.
- **C3:** Demonstration of efficient lowering to MLIR, Triton, and CUDA backends.
- **C4:** An evaluation showing performance competitive with state-of-the-art implementations.

## Requirements

### Hardware

- **GPU:** NVIDIA Ampere A100 80GB
- **Disk space:** ~40 GB free space recommended

### Software

| Package       | Version             | URL                                                        |
|--------------|---------------------|------------------------------------------------------------|
| LLVM/MLIR    | commit `48c8c45`    | https://github.com/llvm/llvm-project/commit/48c8c45       |
| Python       | 3.12.4              | https://www.python.org/                                   |
| Triton       | 3.2.0               | https://github.com/triton-lang/triton                     |
| PyTorch      | 2.5.1               | https://pytorch.org/                                      |
| CUDA Toolkit | 12.4                | https://developer.nvidia.com/cuda-toolkit-archive         |
| LEGO artifact| latest              | https://github.com/tavakkoliamirmohammad/lego             |

## Installation

### 1. Clone the LEGO repository

```bash
git clone https://github.com/tavakkoliamirmohammad/lego.git
cd lego
````

### 2. Clone LLVM and check out the required commit

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 48c8c45
```

### 3. Build and install LLVM/MLIR

From inside `llvm-project`:

```bash
mkdir build && cd build
cmake -G Ninja ../llvm \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
   -DMLIR_ENABLE_CUDA_RUNNER=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE="$(which python)" \
   -DLLVM_BUILD_EXAMPLES=OFF
ninja
```

Then set the build folder environment variable (still inside `build`):

```bash
export MLIR_BUILD_FOLDER="$(pwd)"
```

Return to the LEGO repository root:

```bash
cd /path/to/lego
```

### 4. Create the virtual environment and install Python packages

From the root of the LEGO repository:

```bash
./setup.sh
```

This script creates a virtual environment and installs all required Python packages (including Triton and PyTorch) using the versions listed above.

## Experiment Workflow

From the root of the artifact repository, the experiments can be reproduced with the following steps:

1. **Generate all kernel source code**

   ```bash
   ./benchmarks/gen_all_kernel.sh
   ```

   This script generates all required kernel source files (Triton, CUDA, and MLIR) used in the evaluation.

2. **Run all benchmarks and produce figures and tables**

   ```bash
   ./benchmarks/run_all_kernels.sh
   ```

   This script runs all benchmarks and generates the figures and tables reported in the paper.

## Evaluation and Expected Results

* Running `run_all_kernels.sh` will execute all benchmarks and produce the evaluation outputs.
* The generated figures corresponding to the evaluation section will be located in the `./figures` folder in the root of the artifact directory.
* Approximate time requirements:
  * **Workflow preparation (installation, builds, environment):** ~2 hours
  * **Experiment execution:** ~1.5 hours

## License

The MIT License (MIT)

Copyright (c) 2025 Amir Mohammad Tavakkoli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
