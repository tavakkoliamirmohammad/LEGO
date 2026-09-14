# lego_synth: certified layout synthesis over the LEGO algebra (prototype)

Given the warp access patterns a kernel issues against a shared-memory tile (including
instruction-imposed ones such as `ldmatrix.x4` and `mma.sync` fragments) and a machine model
(banks, bank width, warp size), synthesize a layout that is bank-conflict-free for every
instruction, with the smallest footprint, and verify it exhaustively.

* `machine.py`     machine models (`ampere()` preset)
* `patterns.py`    warp access patterns: `row_vector` / `col_vector` (vector width, dense or masked
                   lane mapping), `ldmatrix_x4`, `mma_m16n8k16_a`
* `layouts.py`     concrete layout families: `RowMajor(pad)`, `XorSwizzle(shift, mask, unit)`, `Rotation(r, unit)`
* `eval_np.py`     exhaustive evaluator: wavefronts per instruction (broadcast-aware, 8 B halves, 16 B quarters)
* `templates.py`   the same families with holes, as Z3 bit-vector expressions
* `solve.py`       Z3 hole filling: one `Distinct` over bank ids per wavefront group, alignment constraints
* `search.py`      tiers `f2` (Triton-like), `affine_xor` (CuTe/Hexcute-like), `full` (LEGO); footprint minimization
* `problems.py`    named problems: `transpose`, `staging` (cp.async dense stores + ldmatrix loads)
* `ablation.py`    space-ablation tables; CLI in `scripts/ablation.py`
* `harness/`       CUDA harness through PyTorch `load_inline`: transpose kernels, smem-bound microbenchmark,
                   ldmatrix microbenchmark with fragment verification, Nsight runner (blocked on nodes with dcgm-exporter)
* `results/`       tables and Triton ground-truth evidence

Run tests (CPU): `python -m pytest tests/test_synth_*.py -q` from `python/` in `triton-venv`.
GPU tests need `source scripts/env-a6000.sh` on the A6000 node.

Reproduce the headline table:
`python -m lego_synth.scripts.ablation --kind staging --elem 2 --rows 64 --widths 32,48,64,80,96,112,128`
