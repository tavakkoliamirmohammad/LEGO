# cpu_dsl_comparison

A small set of `@cpu_kernel`-decorated Python kernels that compile through
LEGO's MLIR pipeline to native x86 code via the `lego-to-x86-vector`
pipeline.  Each candidate has a `<name>.py` with a numpy reference, runs
the JIT'd version, and emits a JSON line with timing + a `verified: bool`
field.

The candidates cover distinct vectorization-shape paths:

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

## Run

```bash
source /scratch/general/vast/u1419116/LEGO/venv/bin/activate
cd /scratch/general/vast/u1419116/LEGO/evaluation/cpu_dsl_comparison
python run_all.py --measure-repeats 5
```

Outputs:
- `results.json`
- `dashboard.md`
