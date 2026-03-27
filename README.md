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
│   │   ├── backend/         # MLIR compilation, JIT, SymPy lowering, PyTorch autograd
│   │   └── frontends/       # DSLAdapter ABC + adapters (Triton, cuTile, Numba CUDA, JAX, python_mlir)
│   ├── examples/            # Usage examples (triton, numba_cuda, jax, cutile, python_mlir, symbolic)
│   └── tests/               # Python tests
│
├── include/Lego/           # MLIR dialect headers (ODS definitions, passes)
├── lib/Lego/               # MLIR dialect implementation (lowering, verification, simplification)
├── tools/lego-opt/         # MLIR optimizer CLI
├── test/                   # MLIR lit tests
│
├── paper/                  # Paper benchmarks and evaluation scripts
├── docs/                   # Architecture and dialect documentation
├── scripts/                # Setup scripts
└── CMakeLists.txt          # Build system (monolithic and decoupled modes)
```

## Architecture

All frontends share a DSL-agnostic rewriter that evaluates layout algebra at
decoration time, then delegates code generation to a pluggable `DSLAdapter`.
The adapters lower through a unified MLIR-based backend:

```
                              User Code
                                 |
                        +--------+--------+
                        |                 |
                 @lego.jit          Tensor / Symbolic
                 decorator                API
                        |                 |
     +----------+-------+-------+----+    |
     |          |               |    |    |
 +---+----+ +--+----+ +------+ | +--+--+ |
 |Triton  | | Numba | | JAX  | | |cuTile| |
 |Adapter | | CUDA  | |Adapt.| | |Adapt.| |
 +---+----+ +--+----+ +--+---+ | +--+--+ |
     |          |          |    |    |    |
     +----------+----------+----+----+    |
                |                         |
       +--------+--------+               |
       |   rewriter.py   |               |
       |  AST transform  |               |
       |  + SymPy eval   |               |
       +--------+--------+               |
                |                         |
                +-----------+-------------+
                            |
                 +----------+----------+
                 |    lego dialect     |
                 |  .................. |
                 |  normalization      |
                 |  simplification     |
                 |  strength reduction |
                 |  verification       |
                 |  lowering           |
                 +----------+----------+
                            |
                 +----------+----------+
                 |     LLVM / MLIR     |
                 |  .................. |
                 |  X86 | AArch64 |   |
                 |       NVPTX        |
                 +---------------------+
```

### Frontends

| Frontend | Module | Decorator | Description |
|----------|--------|-----------|-------------|
| **Triton** | `lego.frontends.triton_jit` | `@lego.jit` | Transforms Triton GPU kernels via AST rewriting; supports `block_ptr` (TMA) code generation ([vecadd](python/examples/triton/hello_world.py), [matmul](python/examples/triton/matmul_lego.py), [vecadd block_ptr](python/examples/triton/vecadd_block_ptr_lego.py), [matmul block_ptr](python/examples/triton/matmul_block_ptr_lego.py)) |
| **cuTile** | `lego.frontends.cutile_jit` | `@lego.jit` | Transforms `cuda.tile` (cuTile) kernels via AST rewriting ([vecadd](python/examples/cutile/vecadd.py), [matmul](python/examples/cutile/matmul.py)) |
| **Numba CUDA** | `lego.frontends.numba_jit` | `@lego_jit` | Transforms Numba CUDA kernels, scalar thread indexing ([vecadd](python/examples/numba_cuda/hello_world.py), [matmul](python/examples/numba_cuda/matmul_lego.py)) |
| **JAX** | `lego.frontends.jax_jit` | `@lego_jit` | Transforms JAX functions, preserves `static_argnums` ([vecadd](python/examples/jax/hello_world.py), [matmul](python/examples/jax/matmul_lego.py)) |
| **Tensor API** | `lego.frontends.python_mlir` | -- | JIT-compiled layout transforms for NumPy/PyTorch with `torch.compile` support ([example](python/examples/python_mlir/hello_world.py)) |
| **Symbolic** | `lego.core` | -- | SymPy-based algebraic layout expressions ([example](python/examples/symbolic/hello_world.py)) |

Each JIT frontend implements the `DSLAdapter` interface (`frontends/_adapter.py`), which defines four hooks: `unwrap`, `find_runtime_vars`, `get_code_printer`, and `compile_and_wrap`. The DSL-agnostic rewriter (`rewriter.py`) handles AST transformation and symbolic evaluation. The Triton adapter additionally supports `block_ptr` (TMA) code generation, emitting `tl.make_block_ptr` / `tl.advance` calls with automatic boundary checks.

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

## Requirements

| Dependency | Version          | Notes                         |
|-----------|------------------|-------------------------------|
| Python    | >= 3.12          | Tested with 3.12, 3.13, 3.14 |
| LLVM/MLIR | commit `7477045` | Included as a submodule       |
| CMake     | >= 3.20          |                               |
| Ninja     |                  | Recommended build generator   |
| NumPy     | 2.1.2            |                               |
| SymPy     | 1.14.0           |                               |

Optional (for GPU frontends):

| Dependency | Frontend    |
|-----------|-------------|
| PyTorch   | Tensor API, `torch.compile` |
| Triton    | Triton JIT  |
| cuda.tile | cuTile JIT  |
| Numba     | Numba CUDA  |
| JAX       | JAX JIT     |

## Installation

### Quick install (Python package only)

```bash
pip install lego-layout
```

This installs the core layout algebra and frontends. The MLIR dialect native extensions are included in the wheel when available.

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

To customize LLVM targets (default `X86;NVPTX`):

```bash
cmake -S . -B build -DLEGO_MONOLITHIC_LLVM=ON -DLEGO_LLVM_TARGETS="X86;NVPTX;AArch64"
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
