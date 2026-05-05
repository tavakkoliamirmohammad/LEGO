# cpu_dsl_comparison — correctness + reference-perf harness

This is an engineering harness, not a research evidence artefact.

## What it does

A small set of `@cpu_kernel`-decorated Python kernels that compile through
LEGO's MLIR pipeline to native x86 code via the `lego-to-x86-vector`
pipeline → MLIR `ExecutionEngine` (LLVM `opt_level=3` with host-CPU
detection — equivalent to `-O3 -march=native`).

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

## What it is NOT

This dashboard does **not** support a "LEGO beats clang" claim.  Earlier
iterations of this harness reported 3–8× wins on Morton-style kernels;
those numbers came from a benchmarking artefact, not a codegen advantage:

- LEGO's `cpu_dsl` frontend bakes each candidate's `N` as a Python
  compile-time literal, so the loop trip count reaches LLVM as a constant.
- The C baselines by default take `N` as a runtime `argc/argv` parameter,
  so clang cannot fold the trip count into algebraic identities (e.g.
  proving the Morton bit-spread is the identity permutation for N <= 65536).

Once both sides see N as a compile-time constant (the `*_clang_const`
build, which adds `__builtin_assume(N == DEFAULT_N)`), the comparison
flattens to within ±10% and clang is occasionally faster.

The `vs_clang_const` column is the only one to read for an apples-to-apples
performance comparison.  `vs_c_O3` and `vs_clang` are kept as engineering
references for "how much room there is" against weak / aggressive C builds.

## How to run

```bash
source /scratch/general/vast/u1419116/LEGO/venv/bin/activate
cd /scratch/general/vast/u1419116/LEGO/evaluation/cpu_dsl_comparison
python run_all.py --quick --measure-repeats 5
```

`--quick` skips re-building the C baselines.  Run `make all` in
`c_baselines/` once after a change to the .c sources.

Outputs:
- `results.json` — one record per candidate
- `dashboard.md` — human-readable summary

## Vectorization mode

The cpu_dsl pipeline has a few `LegoVectorize*` matchers that try to
recognise patterns clang's auto-vectoriser doesn't (filtered reduce,
prefix scan, RLE, etc.).  Empirically these matchers are deadweight on
most kernels — LLVM's auto-vectoriser at `opt_level=3` produces equivalent
output.  The exceptions are `47_multi_reduce` and predicated-count shapes,
where the matcher emits ~10× faster code than LLVM's loop vectoriser.

For the LLVM-auto-vec path (clang-equivalent output for kernels without
fancy patterns):

```bash
LEGO_BYPASS_LEGO_VECTORIZE=1 python run_all.py --quick --measure-repeats 5
```

This bypasses every `LegoVectorize*` pass and lets LLVM's `LoopVectorize`
+ `SLP` handle the lowered scf.for + arith + memref directly.

## C baseline variants

Three flavours, built per source via `c_baselines/Makefile`:

- `*_O3` — `gcc -O3` (conservative reference)
- `*_clang` — `clang-20 -O3 -march=native -mavx512f -ffast-math` (max clang)
- `*_clang_const` — `clang-20 -O3 -march=native -mavx512f` + `BENCH_USE_DEFAULT_N`
   so the kernel calls `__builtin_assume(N == DEFAULT_N)` (apples-to-apples
   with LEGO's compile-time-N visibility — **the column to read**)
