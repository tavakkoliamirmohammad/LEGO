# cpu_dsl_comparison — verification harness

A small set of `@cpu_kernel`-decorated Python kernels that compile through
LEGO's MLIR pipeline to native x86 code via `lego-to-x86-vector` →
`ExecutionEngine` (LLVM `opt_level=3` with host-CPU detection — equivalent
to `-O3 -march=native`).

Each candidate has a `<name>.py` with a numpy reference, runs the JIT'd
version, and emits a JSON line with timing + a `verified: bool` field.

The 10 candidates cover distinct vectorization-shape paths:

| Candidate              | Shape                          |
|------------------------|--------------------------------|
| `01_saxpy`             | unit-stride pointwise          |
| `02_gemm_row_major`    | reduction loop                 |
| `03_3pt_stencil_1d`    | constant-offset gather         |
| `05_morton_2d`         | bit-permuted index             |
| `07_mixed_precision`   | mixed dtypes                   |
| `08_brick_within_cell` | 3D brick                       |
| `43_spmv_indirect`     | data-dependent index (gather)  |
| `44_predicated_fma`    | predicated update              |
| `46_scatter_compute`   | scatter                        |
| `47_multi_reduce`      | multi-output reduction         |

## How to run

```bash
source /scratch/general/vast/u1419116/LEGO/venv/bin/activate
cd /scratch/general/vast/u1419116/LEGO/evaluation/cpu_dsl_comparison
make -C c_baselines all       # one-time: build C reference binaries
python run_all.py --quick --measure-repeats 5
```

`--quick` skips re-building the C baselines.

Outputs:
- `results.json` — one record per candidate
- `dashboard.md` — human-readable table

## C baseline variants

Three flavours, built per source via `c_baselines/Makefile`:

- `*_O3` — `gcc -O3` (conservative reference)
- `*_clang` — `clang-20 -O3 -march=native -mavx512f -ffast-math`
- `*_clang_const` — `clang-20 -O3 -march=native -mavx512f` plus
  `__builtin_assume(N == DEFAULT_N)` so clang sees N as a compile-time
  constant — the same visibility LEGO's frontend has when N is a Python
  literal (the apples-to-apples comparison column).
