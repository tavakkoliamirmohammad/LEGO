# cpu_dsl_examples

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
| `04_morton_2d`         | bit-permuted index             |
| `05_mixed_precision`   | mixed dtypes                   |
| `06_brick_within_cell` | 3D brick                       |
| `07_spmv_indirect`     | data-dependent index (gather)  |
| `08_predicated_fma`    | predicated update              |
| `09_scatter_compute`   | scatter                        |
| `10_multi_reduce`      | multi-output reduction         |

## Run

```bash
source /scratch/general/vast/u1419116/LEGO/venv/bin/activate
cd /scratch/general/vast/u1419116/LEGO/evaluation/cpu_dsl_examples
python run_all.py --measure-repeats 5
```

Outputs:
- `results.json`
- `dashboard.md`
