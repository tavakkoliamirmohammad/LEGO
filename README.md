# LEGO: Layout Expression Language for Code Generation

[![CI](https://github.com/tavakkoliamirmohammad/lego/actions/workflows/ci.yml/badge.svg)](https://github.com/tavakkoliamirmohammad/lego/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lego-layout)](https://pypi.org/project/lego-layout/)
[![Python](https://img.shields.io/pypi/pyversions/lego-layout)](https://pypi.org/project/lego-layout/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE.md)

LEGO is an algebraic, compiler-agnostic framework for specifying and transforming memory layouts. It provides composable layout primitives that lower through a custom MLIR dialect to generate optimized code for CPU and GPU targets.

[[LEGO: A Layout Expression Language for Code Generation of Hierarchical Mapping]](https://users.cs.utah.edu/~tavak/assets/pdf/LEGO-CGO26.pdf) [[CGO 2026 Artifact]](https://zenodo.org/records/17633994)

## Project Structure

```
LEGO/
├── python/                  # Python package (lego-layout)
│   ├── lego/
│   │   ├── core.py          # Layout primitives (Row, Col, RegP, GenP, OrderBy, GroupBy, TileByLayout)
│   │   ├── rewriter.py      # DSL-agnostic AST rewriting engine
│   │   ├── python_printer.py# SymPy code printer (base + DSL subclasses)
│   │   ├── rust_printer.py  # Rust code printer
│   │   ├── fortran_printer.py # Fortran code printer
│   │   ├── cxx_printer.py   # C++ code printer
│   │   ├── julia_printer.py # Julia code printer
│   │   ├── cuda_c_printer.py# CUDA C code printer
│   │   ├── js_printer.py    # JavaScript code printer
│   │   ├── glsl_printer.py  # GLSL code printer
│   │   ├── backend/         # MLIR compilation, JIT, SymPy lowering, PyTorch autograd
│   │   └── frontends/       # DSLAdapter ABC + adapters (Triton, cuTile, Numba, JAX, Rust, Fortran, C++, Julia, CUDA C, JS, GLSL, python_mlir)
│   ├── examples/            # Usage examples (triton, numba_cuda, jax, cutile, python_mlir, symbolic, rust, cxx, fortran, julia, cuda_c, js, glsl)
│   │   └── puzzles/         # GPU puzzles — multi-backend kernel tests (CUDA, ROCm, Vulkan, WebGPU, Metal)
│   └── tests/               # Python tests
│
├── include/Lego/           # MLIR dialect headers (ODS definitions, passes)
├── lib/Lego/               # MLIR dialect implementation (lowering, verification, simplification)
├── tools/lego-opt/         # MLIR optimizer CLI
├── test/                   # MLIR lit tests
│
├── viz/                    # LEGO Studio (browser-based visualizer)
│   ├── wasm/               # Emscripten-compiled LEGO compiler (lego_driver.wasm)
│   ├── js/                 # Frontend JavaScript
│   └── css/                # Styles
│
├── paper/                  # Paper benchmarks and evaluation scripts
├── docs/                   # Architecture and dialect documentation
├── scripts/                # Setup scripts
└── CMakeLists.txt          # Build system (monolithic and decoupled modes)
```

## Architecture

All paths flow through the MLIR `lego` dialect, which normalizes, simplifies,
and strength-reduces layout expressions before handing off to target-specific
backends. **JIT frontends** lower through the dialect, extract simplified
patterns, then return control to the original framework. **Source code
generators** lower through the dialect, extract SymPy expressions from the
optimized arith IR, then emit target-language source. **GPU/CPU backends**
lower the dialect all the way to machine code:

```
                                  User Code
                                      |
            +-------------------------+-------------------------+
            |                         |                         |
  +---------+---------+  +-----------+----------+  +------------+----------+
  |  JIT Frontends    |  |     GPU / CPU        |  |   Source CodeGen      |
  |  Triton, Numba,   |  |   KernelBuilder,     |  |  Rust, C++, Fortran, |
  |  JAX, cuTile      |  |   Tensor API         |  |  Julia, CUDA C,      |
  +---------+---------+  +-----------+----------+  |  JS, GLSL            |
            |                        |              +------------+----------+
            +------------+-----------+---------------------------+
                         |
            +------------+------------+
            |    lego MLIR dialect    |
            |  ...................    |
            |  normalization          |
            |  lowering               |
            |  simplification         |
            |  strength reduction     |
            |  verification (SMT)     |
            +-----+-------+--------+-+
                  |       |        |
                  v       v        v
  +---------------+  +----+----+  ++----------------+
  | extract       |  | extract |  | compile         |
  | patterns      |  | SymPy   |  | to target       |
  +-------+-------+  +----+----+  +---+-----+-----++
          |                |           |     |     |
  +-------+--------+  +---+------+    |     |     |
  | return to      |  | code     |    |     |     |
  | original       |  | printers |    |     |     |
  | framework      |  +---+------+    |     |     |
  |                |      |           |     |     |
  | Triton PTX,    |  +---+------+    |     |     |
  | Numba CUDA,    |  | target   |    |     |     |
  | JAX XLA,       |  | source   |    |     |     |
  | cuTile         |  | code     |    |     |     |
  +----------------+  +----------+    |     |     |
                                      |     |     |
    +-------------+-------------+-------------+-------------+
    |             |             |             |             |
  +-+----------+  |  +----------+-+  +--------+--+  +------+---------+
  | lego-to-   |  |  | lego-to-   |  | lego-to-  |  | lego-to-spirv  |
  | llvm       |  |  | nvvm       |  | rocdl     |  +------+---------+
  +-+----------+  |  +----------+-+  +--------+--+         |
    |             |             |             |      +------+------+
  +-+----------+  |  +----------+-+  +--------+--+  |   SPIR-V    |
  |    CPU     |  |  |   CUDA     |  |    AMD     |  |  (Vulkan)   |
  | X86, ARM   |  |  | PTX/cubin  |  |   HSACO    |  +------+------+
  +------------+  |  +------------+  +-----------+         |
                  |                              +---------+---------+
       +----------+----------+                   |         |         |
       | lego-to-llvmspirv   |              +----+---+ +---+----+ +-+------+
       +----------+----------+              |  naga  | |  naga  | |  naga  |
                  |                         +----+---+ +---+----+ +--+-----+
       +----------+----------+                   |         |         |
       |   LLVM SPIR-V      |              +----+---+ +---+----+ +--+-----+
       |   (OpenCL)         |              |  WGSL  | |  MSL   | |  GLSL  |
       +--------------------+              |(WebGPU)| |(Metal) | |(WebGL) |
                                           +--------+ +--------+ +--------+
```

### Frontends

| Frontend | Module | Decorator | Description |
|----------|--------|-----------|-------------|
| **Triton** | `lego.frontends.triton_jit` | `@lego.jit` | Transforms Triton GPU kernels via AST rewriting; supports `block_ptr` (TMA) code generation ([vecadd](python/examples/triton/hello_world.py), [matmul](python/examples/triton/matmul_lego.py), [vecadd block_ptr](python/examples/triton/vecadd_block_ptr_lego.py), [matmul block_ptr](python/examples/triton/matmul_block_ptr_lego.py)) |
| **cuTile** | `lego.frontends.cutile_jit` | `@lego.jit` | Transforms `cuda.tile` (cuTile) kernels via AST rewriting ([vecadd](python/examples/cutile/vecadd.py), [matmul](python/examples/cutile/matmul.py)) |
| **Numba CUDA** | `lego.frontends.numba_jit` | `@lego_jit` | Transforms Numba CUDA kernels, scalar thread indexing ([vecadd](python/examples/numba_cuda/hello_world.py), [matmul](python/examples/numba_cuda/matmul_lego.py)) |
| **JAX** | `lego.frontends.jax_jit` | `@lego_jit` | Transforms JAX functions, preserves `static_argnums` ([vecadd](python/examples/jax/hello_world.py), [matmul](python/examples/jax/matmul_lego.py)) |
| **Tensor API** | `lego.frontends.python_mlir` | -- | JIT-compiled layout transforms for NumPy/PyTorch with `torch.compile` support ([example](python/examples/python_mlir/hello_world.py)) |
| **Rust** | `lego.frontends.rust_gen` | `lego.rust_gen.generate()` | Generates Rust source code ([example](python/examples/rust/hello_world.py)) |
| **Fortran** | `lego.frontends.fortran_gen` | `lego.fortran_gen.generate()` | Generates Fortran source code ([example](python/examples/fortran/hello_world.py)) |
| **C++** | `lego.frontends.cxx_gen` | `lego.cxx_gen.generate()` | Generates C++ source code ([example](python/examples/cxx/hello_world.py)) |
| **Julia** | `lego.frontends.julia_gen` | `lego.julia_gen.generate()` | Generates Julia source code ([example](python/examples/julia/hello_world.py)) |
| **CUDA C** | `lego.frontends.cuda_c_gen` | `lego.cuda_c_gen.generate()` | Generates CUDA C kernel source code ([example](python/examples/cuda_c/hello_world.py)) |
| **JavaScript** | `lego.frontends.js_gen` | `lego.js_gen.generate()` | Generates JavaScript source for WebGPU/WASM ([example](python/examples/js/hello_world.py)) |
| **GLSL** | `lego.frontends.glsl_gen` | `lego.glsl_gen.generate()` | Generates GLSL shader source code ([example](python/examples/glsl/hello_world.py)) |
| **Symbolic** | `lego.core` | -- | SymPy-based algebraic layout expressions ([example](python/examples/symbolic/hello_world.py)) |

Each JIT frontend implements the `DSLAdapter` interface (`frontends/_adapter.py`), which defines four hooks: `unwrap`, `find_runtime_vars`, `get_code_printer`, and `compile_and_wrap`. The DSL-agnostic rewriter (`rewriter.py`) handles AST transformation and symbolic evaluation. The Triton adapter additionally supports `block_ptr` (TMA) code generation, emitting `tl.make_block_ptr` / `tl.advance` calls with automatic boundary checks.

### Source Code Generation Backends

Seven source-code generation backends take a Python function with LEGO layout expressions and emit equivalent index arithmetic in the target language. Each leverages SymPy's built-in code printers:

```python
import lego
from lego.core import OrderBy, Row

def index_kernel(M, N, BM, BN):
    L = OrderBy(Row(M, N)).TileBy((M // BM, N // BN), (BM, BN))
    offset = L[pid_m, pid_n, :, :]
    return offset

rust_src    = lego.rust_gen.generate(index_kernel)
cxx_src     = lego.cxx_gen.generate(index_kernel)
fortran_src = lego.fortran_gen.generate(index_kernel)
julia_src   = lego.julia_gen.generate(index_kernel)
cuda_src    = lego.cuda_c_gen.generate(index_kernel)
js_src      = lego.js_gen.generate(index_kernel)
glsl_src    = lego.glsl_gen.generate(index_kernel)
```

Key differences by language:
| Feature | Rust | C++ | Fortran | Julia | CUDA C | JavaScript | GLSL |
|---------|------|-----|---------|-------|--------|------------|------|
| Range | `(0..N)` | `std::views::iota(0, N)` | `(/ (i, i=0, N-1) /)` | `(0:N-1)` | comment | `Array.from(...)` | comment |
| Floor div | `a / b` | `a / b` | `a / b` | `div(a, b)` | `a / b` | `Math.floor(a/b)` | `a / b` |
| Modulo | `a % b` | `a % b` | `mod(a, b)` | `mod(a, b)` | `a % b` | `a % b` | `a % b` |
| Power | `.powi(n)` | `std::pow(a, n)` | `a**n` | `a^n` | `pow(a, n)` | `Math.pow(a, n)` | `pow(a, n)` |
| Sqrt | `(x as f64).sqrt()` | `std::sqrt(x)` | `sqrt(dble(x))` | `sqrt(x)` | `sqrt(x)` | `Math.sqrt(x)` | `sqrt(x)` |

### Tensor API

The Tensor API provides layout constructors and transforms for NumPy and PyTorch:

```python
from lego import Tiled, ColMajor, ZCurve, Swizzle, BlockCyclic, Batched

# Basic layouts
layout = Tiled((8, 8), tile_shape=(4, 4))
result = layout.transform(tensor)          # or layout(tensor)
back = layout.inverse_transform(result)

# GPU-oriented layouts
z = ZCurve((4, 4))          # Morton curve for 2D spatial locality
s = Swizzle((8, 8))         # XOR swizzle to avoid shared memory bank conflicts
bc = BlockCyclic((16,), 2, 2)  # ScaLAPACK-style distribution

# Batched transforms (vectorized, no Python loop)
batched = Batched(layout, batch_shape=(32,))
batched.transform(batch_tensor)  # (32, 8, 8)

# Composition and comparison
composed = layout_a.compose(layout_b)
assert RowMajor((4, 4)) == RowMajor((4, 4))
```

PyTorch integration compiles layout transforms to native PyTorch arithmetic via the MLIR lowering pipeline. Instead of materializing O(numel) permutation tables, layout index expressions are lowered through MLIR (`lego-lower` pass with simplification and strength reduction), extracted as SymPy expressions, and compiled to vectorized PyTorch functions. For example, `Col(4,8)` becomes `4*j + i` -- pure arithmetic, no lookup table.

```python
import lego
import torch

layout = lego.ColMajor((4, 8))
x = torch.randn(4, 8)

# Transform: uses compiled arithmetic (arange + mul + add + gather)
physical = layout.transform(x)       # autograd-compatible
logical = layout.inverse_transform(physical)  # round-trips exactly

# LegoTensor: layout-aware tensor subclass
lx = lego.as_lego_tensor(x, layout)
result = lx + lx          # operates on physical storage, no permutation
back = result.to_logical() # converts back to row-major

# torch.compile: traces through compiled index arithmetic
@torch.compile(backend="lego")
def fn(t):
    return layout.transform(t) * 2
```

`LegoTensor` is a `torch.Tensor` subclass that carries layout metadata. Elementwise ops between same-layout tensors operate directly on physical storage. `LegoArray` provides the same for NumPy.

### MLIR Backend

The `lego` MLIR dialect defines layout operations (`gen_p`, `reg_p`, `row`, `col`, `order_by`, `group_by`, `tile_by`, `apply`, `apply_inverse`) with types `!lego.layout` and `!lego.view<T>`. Layouts may contain symbolic (SymPy) dimensions, which are lowered to MLIR function parameters and resolved to concrete values at invocation time. The dialect includes passes for:

- **Normalization** -- desugar `row`/`col`/`tile_by` to primitive `reg_p`/`order_by`/`group_by`
- **Lowering** -- `lego` ops to `arith`/`scf`/`affine`
- **Simplification** -- optimize `divui`/`remui` patterns, distributive factoring (`muli(a,c) + muli(b,c) → muli(addi(a,b), c)`)
- **Strength Reduction** -- convert power-of-2 `muli`/`divui`/`remui` to shift/mask operations
- **Verification** -- bijectivity, GPU bank conflicts, memory coalescing

Six lowering pipelines target different backends:

| Pipeline | Target | Output | Shared memory |
|----------|--------|--------|---------------|
| `lego-to-llvm` | CPU | LLVM IR (X86, AArch64) | N/A |
| `lego-to-nvvm` | CUDA | PTX/cubin via NVPTX | Yes |
| `lego-to-rocdl` | AMD | HSACO via AMDGPU | Yes |
| `lego-to-spirv` | Vulkan/WebGPU/Metal/WebGL | SPIR-V binary → naga → WGSL/MSL/GLSL | Yes (workgroup) |
| `lego-to-llvmspirv` | SPIR-V (OpenCL) | LLVM dialect with SPIR-V calling conventions | Yes |
| WASM (Emscripten) | Browser | `lego_driver.wasm` — full compiler in the browser | N/A |

The NVVM, ROCDL, and LLVM SPIR-V backends share the same three-phase architecture:
1. `buildLegoGPUOutlinePipeline` -- LEGO lower + GPU kernel outlining
2. Backend-specific GPU-to-LLVM conversion (`GPUToNVVM` / `GPUToROCDL` / `GPUToLLVMSPV`)
3. `buildGPUHostLLVMPipeline` -- host-side LLVM lowering

Example: compile for sm_80 with max optimization:
```
lego-to-nvvm{chip=sm_80 opt-level=3}
```

## Requirements

| Dependency | Version          | Notes                         |
|-----------|------------------|-------------------------------|
| Python    | >= 3.12          | Tested with 3.12, 3.13, 3.14 |
| LLVM/MLIR | commit `7477045` | Included as a submodule       |
| CMake     | >= 3.20          |                               |
| Ninja     |                  | Recommended build generator   |
| NumPy     | 2.1.2            |                               |
| SymPy     | 1.14.0           |                               |

Optional:

| Dependency | Used by |
|-----------|---------|
| PyTorch   | Tensor API, `torch.compile` |
| Triton    | Triton JIT frontend |
| cuda.tile | cuTile JIT frontend |
| Numba     | Numba CUDA frontend |
| JAX       | JAX JIT frontend |
| wgpu      | Vulkan/WebGPU execution verification |
| naga-cli  | SPIR-V to WGSL/MSL conversion (`cargo install naga-cli`) |

## Installation

### Quick install (Python package only)

```bash
pip install lego-layout
```

This installs the core layout algebra and frontends. The MLIR dialect native extensions are included in the wheel when available.

**Platform support:**

| Platform | Wheel tag | GPU backends included |
|----------|-----------|----------------------|
| Linux x86_64 | `manylinux_2_28` (glibc 2.28+: RHEL 8+, Ubuntu 20.04+, Debian 11+) | CUDA (PTX), ROCm, Vulkan, WebGPU, Metal, LLVM SPIR-V |
| macOS ARM64 | `macosx_15_0_arm64` | CUDA (PTX), ROCm, Vulkan, WebGPU, Metal, LLVM SPIR-V |

All GPU backends are cross-compilers — no GPU hardware required at install time. The naga binary is bundled for SPIR-V to WGSL/MSL/GLSL conversion.

### Development install

#### 1. Clone and set up the environment

```bash
git clone https://github.com/tavakkoliamirmohammad/lego.git
cd lego
./scripts/setup.sh
source venv/bin/activate
pip install -e ./python
```

#### 2. Build the MLIR dialect

**Monolithic build** (builds LLVM/MLIR + LEGO together):

```bash
cmake -S . -B build -DLEGO_MONOLITHIC_LLVM=ON
cmake --build build -j$(nproc) --target check-lego
```

**Decoupled build** (uses a prebuilt MLIR for fast iteration):

```bash
cmake -S . -B build -DMLIR_DIR=<mlir_build>/lib/cmake/mlir -DLEGO_MONOLITHIC_LLVM=OFF
cmake --build build -j$(nproc) --target check-lego
```

The build system automatically detects and uses fast linkers (`mold`/`lld`) and `ccache`.

To customize LLVM targets (default `X86;NVPTX;AMDGPU;SPIRV`):

```bash
cmake -S . -B build -DLEGO_MONOLITHIC_LLVM=ON -DLEGO_LLVM_TARGETS="X86;NVPTX;AMDGPU;SPIRV;AArch64"
```

## Testing

```bash
# MLIR lit tests
cmake --build build --target check-lego

# Python tests
cmake --build build --target check-lego-python

# All tests
cmake --build build --target check-lego-all
```

## Citation

If you use LEGO in your research, please cite:

> Amir Mohammad Tavakkoli, Cosmin E. Oancea, and Mary Hall. "LEGO: A Layout Expression Language for Code Generation of Hierarchical Mapping." In *2026 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)*, pp. 228-241, 2026.

```bibtex
@INPROCEEDINGS{tavakkoli2026lego,
  author={Tavakkoli, Amir Mohammad and Oancea, Cosmin E. and Hall, Mary},
  booktitle={2026 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)},
  title={LEGO: A Layout Expression Language for Code Generation of Hierarchical Mapping},
  year={2026},
  pages={228-241},
  keywords={Codes;Algebra;Shape;Instruction sets;Layout;Graphics processing units;Organizations;Optimization;Indexing;Python;data layout;MLIR compiler;domain-specific optimization tools},
  doi={10.1109/CGO68049.2026.11394846}}
```

The paper artifact is available at: https://zenodo.org/records/17633994

## License

MIT License. See [LICENCE.md](LICENCE.md).
