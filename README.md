# LEGO: Layout Expression Language for Code Generation

[![CI](https://github.com/tavakkoliamirmohammad/lego/actions/workflows/ci.yml/badge.svg)](https://github.com/tavakkoliamirmohammad/lego/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lego-layout)](https://pypi.org/project/lego-layout/)
[![Python](https://img.shields.io/pypi/pyversions/lego-layout)](https://pypi.org/project/lego-layout/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE.md)

LEGO is an algebraic, compiler-agnostic framework for specifying and transforming memory layouts. It provides composable layout primitives that lower through a custom MLIR dialect to generate optimized code for CPU and GPU targets.

[[Paper]](https://users.cs.utah.edu/~tavak/assets/pdf/LEGO-CGO26.pdf) [[Artifact]](https://zenodo.org/records/17633994)

## Project Structure

```
LEGO/
├── python/                  # Python package (lego-layout)
│   ├── lego/
│   │   ├── core.py          # Layout primitives (Row, Col, RegP, GenP, OrderBy, GroupBy, TileByLayout)
│   │   ├── backend/         # MLIR compilation, JIT, SymPy lowering, PyTorch autograd
│   │   └── frontends/       # User-facing APIs (Triton @jit, NumPy/PyTorch tensor transforms)
│   ├── examples/            # Usage examples (triton, python_mlir, symbolic)
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

All frontends lower through a unified MLIR-based backend:

```
 ┌─────────────┐ ┌────────────┐ ┌───────────┐
 │ Triton @jit │ │ Tensor API │ │ Symbolic  │
 └──────┬──────┘ └─────┬──────┘ └─────┬─────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
                       v
        ┌──────────────────────────┐
        │       lego dialect       │
        │  ······················  │
        │  normalization           │
        │  simplification          │
        │  verification            │
        │  lowering                │
        └────────────┬─────────────┘
                     │
                     v
        ┌──────────────────────────┐
        │        LLVM / MLIR       │
        │  ······················  │
        │  X86 │ AArch64 │ NVPTX  │
        └──────────────────────────┘
```

### Frontends

- **Triton JIT** (`lego.jit`) -- `@jit` decorator that transforms Triton GPU kernels via AST rewriting ([example](python/examples/triton/hello_world.py))
- **Tensor API** (`lego.frontends.python_mlir`) -- JIT-compiled layout transforms for NumPy and PyTorch tensors ([example](python/examples/python_mlir/hello_world.py))
- **Symbolic** (`lego.core`) -- SymPy-based algebraic layout expressions ([example](python/examples/symbolic/hello_world.py))

### MLIR Backend

The `lego` MLIR dialect defines layout operations (`gen_p`, `reg_p`, `row`, `col`, `order_by`, `group_by`, `tile_by`, `apply`, `apply_inverse`) with types `!lego.layout` and `!lego.view<T>`. The dialect includes passes for:

- **Normalization** -- desugar `row`/`col`/`tile_by` to primitive `reg_p`/`order_by`/`group_by`
- **Lowering** -- `lego` ops to `arith`/`scf`/`affine`
- **Simplification** -- optimize `divui`/`remui` patterns
- **Verification** -- bijectivity, GPU bank conflicts, memory coalescing

## Requirements

| Dependency | Version          | Notes                         |
|-----------|------------------|-------------------------------|
| Python    | >= 3.12          | Tested with 3.12, 3.13, 3.14 |
| LLVM/MLIR | commit `a3d8e35` | Included as a submodule       |
| CMake     | >= 3.20          |                               |
| Ninja     |                  | Recommended build generator   |
| NumPy     | 2.1.2            |                               |
| SymPy     | 1.14.0           |                               |

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
