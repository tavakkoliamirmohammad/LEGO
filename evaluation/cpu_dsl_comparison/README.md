# cpu_dsl_comparison — Baseline vs cpu_dsl Benchmark Harness

## Purpose

This folder holds a representative set of micro-benchmarks that cover the
full spectrum of verdict outcomes observed in the prior CASTLE CPU evaluation
rounds (AMD Round 1 and Intel audit).  For each candidate there is:

- `kernel.py` — a scalar/NumPy reference function (`kernel_scalar`) and a
  `@cpu_kernel`-decorated DSL version (`kernel_cpu_dsl`).
- `measure.py` — measures both versions with identical input data, checks
  correctness, and prints a single JSON record to stdout.

The top-level `run_all.py` script runs every candidate, collects results,
and prints a side-by-side table.

## How to run

```bash
source /scratch/general/vast/u1419116/LEGO/venv/bin/activate
cd /scratch/general/vast/u1419116/LEGO/evaluation/cpu_dsl_comparison
python run_all.py
```

`run_all.py` automatically sets `PYTHONPATH` to
`/scratch/general/vast/u1419116/LEGO/build/python_packages/lego` in the
subprocess environment, so the MLIR compiled pass pipelines (`lego-to-x86-vector`
etc.) are found correctly.  Do **not** export `PYTHONPATH=/scratch/.../LEGO/python`
in the shell — that directory lacks the compiled `.so` files and shadows the
installed package from the venv.

Full results (one JSON record per candidate) are written to `results.json`.

## Methodology caveat

The current comparison is **NumPy BLAS scalar reference vs cpu_dsl JIT**. This
is NOT apples-to-apples for the v1 prototype because:

1. **NumPy is BLAS-accelerated** for SAXPY/GEMM patterns (uses MKL/OpenBLAS).
   On AVX-512 hardware, this is already heavily vectorized.
2. **cpu_dsl JIT has ~100µs per-call overhead** (memref descriptor build +
   ctypes marshalling + JIT entry trampoline). At small N (≤8K elements), this
   dominates timing.

The result: at N=8192, cpu_dsl shows LOSS even though the inner-loop body is
correctly vectorized. The numbers measure call-overhead, not vectorization.

For a true apples-to-apples speedup measurement, see the **scalar-JIT vs
vectorized-JIT** harness invoked via `python run_all.py --isolate-vectorization`.
That harness compiles each kernel twice — once via `--lego-to-llvm` (scalar)
and once via `--lego-to-x86-vector` (vectorized) — and compares JIT-vs-JIT,
isolating the vectorization effect from JIT startup costs.

## Candidates

| Dir | Layout class | Prior verdict | Expected cpu_dsl verdict |
|-----|-------------|---------------|--------------------------|
| `01_saxpy` | Unit-stride FMA | WIN (trivial SIMD) | WIN |
| `02_gemm_row_major` | Row-major tiled GEMM | WIN (reg+L1 tile class) | WIN |
| `03_3pt_stencil_1d` | 1D 3-point stencil | WIN / R12 blocked | WIN once R12 lands |
| `04_col_major_inner` | Column-major inner loop | PARITY / LOSS | PARITY |
| `05_morton_2d` | Z-Morton encoded 2D | WIN (gemm), LOSS (chol) | LOSS / ERROR (bitwise ops needed) |
| `06_self_update` | In-place prefix-sum | LOSS (loop-carried dep) | LOSS |
| `07_mixed_precision` | Scalar f32 arg path | UNCERTAIN | PARITY / WIN |
| `08_brick_within_cell` | Within-brick vectorisation | PARITY vs NumPy (R1 proof-point) | PARITY |

## JSON schema

Each `measure.py` emits one JSON line:

```json
{
  "name":        "01_saxpy",
  "baseline_ms": 0.4,
  "cpu_dsl_ms":  0.18,
  "speedup":     2.22,
  "verdict":     "WIN",
  "notes":       ""
}
```

- `verdict = "WIN"` when `speedup > 1.05`.
- `verdict = "PARITY"` when `0.95 <= speedup <= 1.05`.
- `verdict = "LOSS"` when `speedup < 0.95`.
- `verdict = "ERROR"` when the DSL compilation or execution failed; the
  `notes` field contains the exception message.

## Limitations

Not all 34 prior eval candidates are reproduced here.  This is a
representative subset chosen to cover each verdict bucket and the main
layout classes.  As the cpu_dsl matures (R12 cross-brick shuffle, bitwise
ops, 2-D tiling support), new candidates can be added by creating a new
subdirectory under `candidates/` following the same `kernel.py` /
`measure.py` schema.

## Context

- Prior eval audit report: `evaluation/audit_report_intel.md`
- Infrastructure roadmap: `evaluation/roadmap.md`
- Prior proof-point benchmarks: `evaluation/cpu_vector_proof/`
- DSL implementation: `python/lego/backend/cpu_dsl.py`
- Pipeline builder: `python/lego/backend/cpu_builder.py`
