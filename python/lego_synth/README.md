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
                   ldmatrix microbenchmark with fragment verification, Nsight runner (needs readable counters)
* `kernels/`       end-to-end kernels through the CASTLE MLIR pipeline: `gemm_mma.py` (cp.async + ldmatrix +
                   mma.sync, S-stage pipeline, grouped ordering, f16/f32 output), `attn_mma.py` (FlashAttention-2
                   structure; options `rows_per_warp`, `q_in_regs`, `out_dtype`, `sQ_layout`, `causal`),
                   `addr.py` (delta addressing: one per-lane base per tile, every access `(base ^ K) + C`,
                   constants verified exhaustively against the layout), `launch.py` (CuPy launcher, `max_regs`),
                   `target.py` (device detection, provenance)
* `bench/`         `gemm_sweep.py` (vs cuBLAS, Triton fixed + autotuned), `attn_sweep.py` (vs SDPA flash / cuDNN /
                   efficient, Triton tutorial fixed + autotuned), `triton_attention.py` (the tutorial kernel, padded to
                   the next power of two for D = 80/96/112), `hexcute_setup.sh` + `hexcute_attention.py` (the Hexcute
                   artifact's attention kernel tuned over its own space, in its own environment), `gemm_layout_ablation.py`,
                   `run_all.sh` (everything, device-tagged results)
* `scripts/`       `ablation.py`, `synth.py`, `calibrate_smem.py`, `bench_ldmatrix_w80.py`, `ncu_conflicts.py`
                   (Nsight bank-conflict counters vs the model)
* `results/`       tables and Triton ground-truth evidence; `results/<device-tag>/` per GPU

Run tests: `cd python && pytest --import-mode=importlib tests/test_synth_*.py tests/test_bitwise_ops.py -q`
after `source scripts/env.sh` (GPU tests skip without a CUDA device).

Reproduce the headline table:
`python -m lego_synth.scripts.ablation --kind staging --elem 2 --rows 64 --widths 32,48,64,80,96,112,128`

## Reproducing on another GPU

Nothing in the suite assumes a particular card: the PTX target, machine model, SM count, and
shared-memory opt-in limit are detected at run time (`lego_synth/kernels/target.py`), every result
file starts with a provenance header (device, driver, CUDA, torch, triton, cupy, git commit,
timestamp, seed), and configurations that exceed the device's shared-memory limit are skipped, not failed.

1. Build LEGO with the monolithic LLVM build (see the top-level README), then create the runtime shim and
   link the native bindings into the source tree: `bash scripts/setup-mlir-runtime-dir.sh`.
2. Python environment with `torch`, `triton`, `cupy-cuda12x`, `z3-solver`, `sympy`, `numpy`, `pytest`
   (the tested versions are in every result file's header).
3. `source scripts/env.sh` (derives paths from the repo; needs `CUDA_HOME` or `nvcc` on `PATH`;
   set `GCC_ROOT` if the CUDA host compiler must be a specific GCC). `scripts/env-a6000.sh` is the
   site-specific example this suite was developed with.
4. Tests: `cd python && pytest --import-mode=importlib tests/test_synth_*.py tests/test_bitwise_ops.py -q`
   (GPU tests skip without a CUDA device; the Nsight test skips when counters are locked).
5. Full suite into `python/lego_synth/results/<device-tag>/`: `bash python/lego_synth/bench/run_all.sh`
   (`run_all.sh quick` for a short version). Individual scripts accept `--seed`, `--out`, and
   configuration lists; see `python -m lego_synth.bench.gemm_sweep --help`. Long runs are best started
   detached (`nohup setsid bash python/lego_synth/bench/run_all.sh > run_all.log 2>&1 &`).
6. Optional Hexcute baseline: `bash python/lego_synth/bench/hexcute_setup.sh` clones and builds the
   artifact's hidet fork (branch `cgorebuttal`) into `../hexcute-bench/workspace` with its own venv;
   `run_all.sh` then runs `bench/hexcute_attention.py` automatically (attention only, non-causal only,
   as in the artifact; candidates that produce wrong results are excluded from its best time).
   Set `HEXCUTE_WS` to put it elsewhere. If the runtime library was built with a non-system GCC, that
   GCC's `lib64` must be on `LD_LIBRARY_PATH` (`scripts/env.sh` does this when `GCC_ROOT` is set).
7. Hardware counters: when `ncu` is on `PATH` and counters are readable, `run_all.sh` also records
   `ncu_conflicts.md` (measured vs predicted bank conflicts); otherwise that step is skipped.

Attention config grammar (`--configs`): `<BLOCK_M>x<BLOCK_N>s<stages>[r32][q0]` — `r32` gives each warp
two 16-row m-tiles (halves fragment loads per mma, costs registers), `q0` re-reads the Q fragments from
shared memory instead of pinning them in registers. GEMM: `<BM>x<BN>x<BK>w<warps_m>x<warps_n>`.

Notes for other architectures: the machine model covers NVIDIA from Volta on (32 banks x 4 B,
32-lane warps); `mma.sync.m16n8k16` needs sm_80 or newer; the bundled LLVM targets up to sm_90 and
newer GPUs run that PTX through forward compatibility; Hopper-specific paths (wgmma, TMA) are not
used, so Hopper numbers measure the Ampere-style kernel.
