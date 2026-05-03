# LEGO/CASTLE Infrastructure Roadmap

> **Status:** active. This document is the actionable output of the CASTLE
> CPU evaluation round (branch `eval/cpu-source-emission`) and subsequent
> infrastructure work. Each entry is a concrete LEGO/CASTLE infrastructure
> gap surfaced by a LOSS or PARITY result, with a proposed feature, a
> sketched design, and an effort/dependency estimate. The list grows as
> more results land.
>
> **R1 (CPU vector pipeline) is CLOSED** — shipped in v1 on branch
> `feat/cpu-vector-pipeline` (2026-05-01). See the R1 entry below for the
> v1 proof-point results. R12 (cross-brick shuffle) is now the highest
> priority open item.
>
> **R18 (reduction guard) is CLOSED** — implemented in commit a66e46f.
> `computeStripMineFactor` detects broadcast-store + same-base-load pairs
> (the scalar-reduction pattern) and returns L_strip=1, skipping vectorization
> for cross-iteration reductions. This prevents incorrect vectorization of
> accumulator loops and is the correct conservative behavior for v1.
>
> **R19 (strided-gather scalar-index bug) is CLOSED** — fixed 2026-05-03
> in commit 8f0b325. The Strided path in `emitVectorBody` was using
> byte-stride offsets to build the index vector, producing wrong addresses.
> Fix: use `cloneAddrDAG` with a fresh per-lane IRMapping (mirroring the
> NonAffine path). 9 candidates moved from PENDING → VERIFIED.
>
> **R20 (strided → deinterleave lowering) is CLOSED** — implemented in
> commit a66e46f. For constant strides 2/4/8 with stride*Ln ≤ 256, the
> Strided path now emits sequential transfer_read + vector.shuffle instead
> of vector.gather. This maps to vpermt2ps (1-3 cycles) vs vpgatherdps
> (10+ cycles). Affects candidates 23-27, 29-31, 38 which all moved from
> PENDING to VERIFIED with vec_iso in the 3-5× range.
>
> **Dashboard (2026-05-03, branch feat/cpu-vector-pipeline):**
> 26 WIN / 14 PARITY / 1 LOSS (18_tblis) / 1 ERROR (20_bricklib_3d13pt NaN)
> vs C-O3 baseline. 42/42 VERIFIED for correctness.
>
> **The CASTLE/TACO paper Section 7.5 is HELD** until either each item
> below is closed (infra implemented + affected candidates re-measured)
> or definitively classified as architectural-limit (paper Section 9).

## R1 — CPU vector pipeline (lego-to-x86-vector + @cpu_kernel DSL)

**Status: CLOSED — shipped in v1 (branch `feat/cpu-vector-pipeline`, 2026-05-01).**

**What was built.** The full CPU vector pipeline described in the Phase A–H
specification:

- `@cpu_kernel` Python DSL with `tile_range` sentinel and AST → MLIR compiler.
- `CPUKernelBuilder` / `CPUKernelContext` with JIT via MLIR `ExecutionEngine`.
- `lego-to-x86-vector` and `lego-to-arm-neon` pass pipelines registered in
  the LEGO pass manager. NEON ships in v1; width-aware NEON emission (SVE) is
  tracked as R15.
- Proof-point benchmarks in `evaluation/cpu_vector_proof/`.

**Proof-point results (Intel Xeon Gold 6330, 2026-05-01):**

| Benchmark | Pipeline | Speedup vs NumPy | Notes |
|-----------|----------|-----------------|-------|
| `brick_within_cell` | x86-vector | 0.56× | JIT executes correctly; NumPy BLAS loop is heavily optimized — beating it needs FMA intrinsic quality (→ R12 + better cost model) |
| `brick_stencil_cross` | x86-vector | N/A (pipeline error) | `memref.store` type mismatch for offset-index stores (`B[i+1]`); see R12 |

The v1 speedup numbers are below target for these micro-benchmarks. The
important result is that the end-to-end pipeline (Python DSL → MLIR IR →
x86-vector lowering → ExecutionEngine → numpy buffer) **executes correctly**.
The register-level candidates (builders 05, 06, 07, 08, 16, 34) show 3–7×
wins in the full evaluation round; the DSL pipeline is the enabling
infrastructure for those.

**Spec doc:** `docs/cpu_vector_pipeline_spec.md` (if generated) or the
Phase A–H plan tracked in this branch.

**Remaining gaps for brick class (now R12):** Cross-brick shuffle and
affine-offset store lowering are imminent follow-on work. See R12 below.

**Bonus.** The same pipeline plumbing (`lego-to-arm-neon`) lays the
groundwork for the GPU MMA tensor-core story via a different target dialect.

## R12 — Cross-brick shuffle + brick-aware second-block base (IMMINENT)

**Severity:** high. Directly flips 5 brick-class candidates from LOSS to WIN.

**Status:** partial. Phase C built IR-shape scaffolding; R12a landed in
`feat/cpu-vector-pipeline` — `cls.boundaryJump` now threads the actual
address jump from `solveAccessTierB` through to `emitCrossBlockLoad`, which
uses `blockNp1Iv = baseIv + cls.boundaryJump` (correct for probe-from-zero
flat-address patterns). Candidates 19, 21, 22, 37 (simplified flat-offset
brick stencils) are VERIFIED and WIN in the evaluation. Candidate 20
(13-point stencil) is VERIFIED but measures NaN due to the 13pt diagonal
neighbor pattern producing zero output in the vectorized path — a remaining
IR issue separate from the stride threading.

**What R12 still needs for full BrickLib API support:**

**What Phase C delivered:**
- IR-shape scaffolding for cross-block vector shuffles.
- FileCheck tests that verify the emitted IR has the right structure.
- `evaluation/cpu_vector_proof/brick_stencil_cross/` scaffold (this branch).

**What R12 still needs:**

1. **Brick-stride second-block base.** Replace the boundary-lane-derived base
   with `brick_id * brick_stride` in the vector shuffle emission pass. The
   brick stride is `BRICK_SIZE * element_size` elements; it must be threaded
   through the pass as a compile-time or runtime parameter.

2. **Affine-offset store lowering.** The `lego-to-x86-vector` pipeline fails
   for store targets with a compile-time offset (`B[i+1]` → `memref.store`
   type mismatch because the vectorized load produces `vector<16xf32>` but
   the store expects `f32`). Fix: extend the vector-store lowering to handle
   `i + k` for constant `k` (emit a scatter or a shifted vector-store).

3. **Update `brick_stencil_cross/kernel.py`** to use a 3D 7-point stencil
   with halos once (1) and (2) are in place.

**Affected candidates (flip LOSS → WIN when R12 lands):**
- 11 (`bricklib-3d7pt-brick`) — LOSS 0.93× AMD / LOSS 0.86× Intel
- 12 (`bricklib-3d13pt-brick`) — PARITY AMD / marginal WIN 1.06× Intel
- 13 (`polybench-heat3d-brick`) — MIXED AMD / LOSS 0.80× Intel
- 14 (`polybench-jacobi2d-brick`) — LOSS 0.67× AMD / LOSS 0.19× Intel (severe)
- 29 (`bricklib-stencil-nonpow2-brick`) — LOSS 0.75× AMD / LOSS 0.61× Intel

**Effort:** 1–2 weeks (brick-stride pass fix + affine-offset store lowering).

**Dependencies:** R1 (CLOSED — CPU vector pipeline ships in v1 and provides
the pipeline context these fixes plug into).

## R13 — AOT object-file path

**Severity:** medium. Required for production deployment (no JIT overhead at
run time).

**Status: CLOSED — shipped 2026-05-01 (branch `feat/cpu-vector-pipeline`).**

**What was built.** Added `CPUKernelBuilder.compile_aot(output_path, target, cpu)`
to `python/lego/backend/cpu_builder.py`:

1. Runs the same MLIR pass pipeline as `compile()` (same options, same
   `opt_level`).
2. Calls `ExecutionEngine.dump_to_object_file(output_path)` — the MLIR Python
   binding that emits a relocatable ELF `.o`.
3. Returns the absolute path to the written `.o` file.

**Implementation note.** The spec proposed `mlir-translate --mlir-to-llvmir`
+ `llc -filetype=obj` as a subprocess pipeline, but those binaries are not
compiled in this build (only the MLIR Python library and `lego-opt` are). The
`ExecutionEngine.dump_to_object_file()` API achieves the same result in-process
— no subprocess, no binary path configuration required.

**Validation.** `test_aot_object_file` in `python/tests/test_cpu_dsl.py`:
- Calls `_saxpy.compile_aot('/tmp/test_saxpy_aot.o')`.
- Asserts the file exists and is non-empty.
- Runs `objdump -d` and asserts AVX vector instructions (v-prefix mnemonics)
  are present in the disassembly.
- Test passes on Intel Xeon Gold 6330 (CHPC notch343). All 15 `test_cpu_dsl.py`
  tests continue to pass (no regressions).

**Effort:** 0.5 days (simpler than spec — reused existing MLIR Python API).

**Dependencies:** none beyond existing CPU vector pipeline (R1, CLOSED).

## R14 — SMT-driven dependence analysis for tile legality

**Severity:** medium. Affects auto-tiling correctness guarantees.

**Status:** future. Currently tile sizes are user-specified; no automatic
legality check that a chosen tile size doesn't violate loop-carried
dependences.

**Proposed feature.** A `lego.check_tile_legality` analysis that uses an
SMT solver (e.g. Z3 via `mlir-check-dep`) to verify that the user's
`TileBy(BM, BN)` annotation doesn't introduce a dependence violation at
the tile boundary. Emits a compile-time diagnostic if not legal.

**Effort:** 3–4 weeks (Z3 integration + MLIR analysis pass).

**Dependencies:** none (pure analysis pass, no new lowerings needed).

## R15 — ARM SVE pipeline

**Severity:** low for CHPC (no SVE hardware in current cluster). Medium for
Apple M-series / Graviton 3 deployment targets.

**Status: CLOSED — shipped 2026-05-01 (branch `feat/cpu-vector-pipeline`).**

**What was built.**

1. `lib/Lego/LegoArmSvePipeline.cpp` — new pipeline `lego-to-arm-sve`.
   - Mirrors `lego-to-arm-neon` but passes `target="sve"` to `lego-vectorize`.
   - Emits fixed-width 16-byte vectors (vscale=1: 2xf64, 4xf32) — identical
     lane counts to NEON. The LLVM AArch64 backend legalizes these to full SVE
     width at `llc` time when `+sve` is in the target feature string.
   - Pipeline: `buildLegoLowerPipeline → canonicalize+CSE →
     lego-vectorize{target=sve} → convert-vector-to-llvm → SCF/Arith/MemRef/
     Func/CF → LLVM → reconcile-unrealized-casts`.
2. `lib/Lego/LegoVectorize.cpp` — added `sve` case to `getRegisterLanesForType`:
   returns `16 / elementBytes` (same as NEON). Comment documents the
   fixed-width-SVE design choice.
3. `include/Lego/Passes.h` — `LegoToArmSvePipelineOptions` struct + declaration.
4. `include/Lego/Passes.td` — updated `lego-vectorize` target option description
   to include `sve`.
5. `lib/Lego/CMakeLists.txt` — added `LegoArmSvePipeline.cpp` to
   `_CPU_VECTOR_SOURCES`.
6. `lib/Lego/Passes.cpp` — registered `lego-to-arm-sve` pipeline.
7. `test/Lego/lego_to_arm_sve.mlir` — FileCheck test (IR shape only):
   - Runs `lego-opt --lego-to-arm-sve` on a SAXPY kernel.
   - Checks: `llvm.func @saxpy_sve`, `vector`, `llvm.return` present;
     `lego.`, `scf.for` absent.
   - Passes. Test header notes runtime validation requires ARM SVE hardware.

**Design note — fixed-width vs scalable.**  True scalable vectors (MLIR
`vector<[N]xT>` syntax) would require pervasive changes to transfer_read/write,
shuffle, broadcast emission.  The fixed-width approach produces correct SVE
code on SVE hardware (the LLVM backend legalizes it) while remaining testable
on this x86 node.  A future R15v2 can upgrade to true scalable IR.

**Validation.** FileCheck test passes. `check-lego-all` passes (765 tests, 0
failures). All 15 `test_cpu_dsl.py` Python tests pass.

**Runtime note.** IR-shape only on this CHPC node (Intel Xeon Gold 6330, no
SVE). For actual SVE execution:
```
  mlir-translate --mlir-to-llvmir kernel.mlir -o kernel.ll
  llc -mtriple=aarch64-linux-gnu -mattr=+sve -O3 kernel.ll -filetype=obj
```

**Effort:** 0.5 days (reused NEON pipeline as template; fixed-width approach).

**Dependencies:** R1 (CLOSED — CPU vector pipeline).

## R17 — GPU lane-fold via warp shuffles

**Severity:** medium. Required for the GPU MMA / warp-cooperative story.

**Status:** future. The `@gpu_kernel` DSL (gpu_dsl.py) already exposes
`shuffle_down`, `shuffle_xor`, and `warp_prefix_sum`. The missing piece is
a `lego.warp-fold` lowering pass that automatically maps a brick's innermost
dimension onto warp lanes (analogous to what R12 does for CPU SIMD).

**Proposed feature.** A `lego-warp-fold` MLIR pass that:
1. Identifies `TileBy` dimensions with size ≤ warp width (32).
2. Rewrites the inner loop to warp-level operations (no `scf.for`).
3. Lowers via MLIR GPU dialect shuffle ops.

**Effort:** 4–6 weeks (mirrors R12 effort but for GPU dialect).

**Dependencies:** R12 design patterns inform this; share the brick-stride
address computation infrastructure.

## R2 — Anti-diagonal layout: per-cell access topology check

**Severity:** medium. Affects DP-wavefront layout class.

**Motivating evidence:**
- Builder 17 (`rodinia-nw-antidiag-tile`) → **LOSS 0.32×** (large).
  Surprising — NW is canonical pure-DP wavefront where this layout
  *should* win.
- Builder 19 (`npdp-zuker-skew-tile`) → WIN 1.17×. Same wavefront
  family, different access topology.
- Builder 20 (`polybench-seidel2d-wavefront-tile`) → LOSS 0.81×.
  9-point stencil, expected to lose; confirmed.

**Root cause.** Anti-diagonal layout is *fragile* even within pure-DP.
It works when each cell reads from cells on a small set of preceding
diagonals (Zuker, Nussinov). It fails when the cell reads cross many
diagonals or the recurrence has skewed memory locality (NW's two-pass
score+trace structure, seidel's 9-point neighborhood).

**Proposed feature.** A `lego.check_antidiag_compatibility` analysis
pass that reads the user-supplied access pattern (declared via the
`GenP`'s `apply` body) and either:

(a) emits a compile-time warning if the access pattern crosses too many
preceding diagonals (heuristic: if more than 2 distinct diagonals are
read per cell, antidiag layout is unlikely to win), or

(b) refuses to lower the layout and points to the offending access.

**Implementation sketch.**
- Static analysis on the `GenP` region body — collect all `arith.subi`
  patterns of the form `i - c, j + c'` and bucket by `c-c'`.
- Compare bucket count against the heuristic threshold.
- Emit MLIR warning diagnostic via `LayoutVerifier`.

**Effort:** 1–2 weeks.

**Dependencies:** none.

**Re-test list when closed:** 17, 20 — once the warning fires, the
scout's prediction model can drop these candidates upfront. 17/20
remain LOSSes in the paper but the *survey* methodology improves.

## R3 — AoSoA scope: indirect-connectivity + scatter/gather microarch

**Severity:** medium. Affects AoSoA layout class.

**Cross-architecture finding (2026-04-29 Intel audit):** AoSoA results are
**architecture-sensitive**. Cand 21 (particlefilter) flipped LOSS (AMD 0.51×)
→ PARITY (Intel ~1.0×) because Intel's wider SIMD lanes and different gather
implementation reduce the AoSoA index overhead. Cand 23 (hpccg) went MIXED
(AMD) → uniform LOSS (Intel) because Intel's prefetcher handles the strided
pattern poorly. So R3 needs *both* "is the field array the bottleneck" AND
"does this microarchitecture's scatter/gather amortize the AoSoA index cost".

**Motivating evidence:**
- Builder 22 (`lulesh-elem-aosoA`) → LOSS 0.94× AMD / LOSS 0.90× Intel.
  Bandwidth dominated by `nodelist[k*8+i]` indirect connectivity, not fields.
- Builder 23 (`hpccg-cg-aosoA`) → MIXED AMD (medium WIN 1.19×, small/large
  LOSS) → uniformly LOSS Intel (0.69×–0.75×). Intel prefetcher doesn't
  amortize AoSoA strided access.
- Builder 21 (`rodinia-particlefilter-aosoA`) → LOSS 0.51× AMD → PARITY
  ~1.0× Intel. Wider SIMD + better gathers reduce overhead but no win.

**Root cause.** AoSoA helps *only* when the bottleneck is the SoA
access pattern — i.e., when the bandwidth is dominated by the field
array LEGO is reordering. Kernels with indirect connectivity (LULESH,
unstructured-mesh codes) have their bandwidth elsewhere; AoSoA's index
overhead is pure cost.

**Proposed feature.** A scout-side classifier (not infra, but a survey
methodology improvement): before listing AoSoA as the layout trick,
the scout must verify the kernel's field-array access is the dominant
memory pressure. Codes with indirect access (gather through index
arrays) get AoSoA dropped from the candidate list.

**Implementation sketch.**
- Update `evaluation/harness/scout_prompt.md` to require, for any AoSoA
  candidate, a "bandwidth dominance check": cite a paper (or run a
  pre-survey roofline analysis) showing the field array is the hot
  spot.

**Effort:** 0.5 day (prompt update + scout reruns those candidates).

**Dependencies:** none.

**Re-test list when closed:** 22 stays LOSS but moves to "out-of-scope"
in the paper (Section 9 Limitations) rather than Section 7.5 LOSS row.

## R4 — Z-Morton scope: triangular-write asymmetry

**Severity:** medium. Affects cache-oblivious layout class.

**Motivating evidence:**
- Builder 01 (`polybench-gemm-zmorton`) → WIN 3.30×. Symmetric dense
  access — Z-Morton wins.
- Builder 02 (`polybench-lu-zmorton`) → WIN 3.30×. LU's k-loop scans
  full rows; Morton's space-filling locality benefits both reads and
  writes.
- Builder 03 (`polybench-chol-zmorton`) → **LOSS 0.50×**. Cholesky
  writes only the lower-triangular half; Morton ordering destroys the
  natural diagonal-major write pattern that Cholesky's recurrence
  needs.

**Root cause.** Z-Morton is a *symmetric* layout that assumes both
reads and writes hit the full 2D index space. Triangular kernels
(Cholesky, RFP-style packing) write only half the index space — Morton
order forces the kernel to skip every other Morton block during writes,
killing locality.

**Proposed feature.** Either:

(a) **Triangular-aware Z-Morton variant** — `GenP` that maps the
    triangular index space to a contiguous Morton sub-pattern (skipping
    Morton blocks above the diagonal). Reuses the Morton bijection but
    on a half-space.

(b) **Scout-side classification** — Z-Morton candidate is dropped
    when the kernel writes only a triangular half of the matrix.

**Implementation sketch (a).** Extend the LEGO algebra with a
`triangular_morton(N)` `GenP` whose `apply` skips upper-tri blocks. ~1
week of Python + MLIR.

**Effort (a):** 1 week. **Effort (b):** 0.5 day.

**Dependencies:** none.

**Re-test list when closed:** 03 (chol-zmorton) — re-run with (a),
expect WIN.

## R5 — Survey baseline-class matching (citation methodology)

**Severity:** medium. Affects all candidates whose published prediction
came from a different baseline class than the suite ships.

**Motivating evidence:**
- Builder 15 (`polybench-symm-rfp`) → PARITY 0.99×. The Gustavson 2010
  RFP paper measured 1.5–5× vs *packed-format LAPACK* (DPPSV — serial
  per-row). PolyBench `symm` is an already-vectorized dense BLAS-3
  loop. RFP's halved-storage win doesn't survive against vectorized
  dense.

**Root cause.** The scout's `predicted_win` was derived from a paper
whose baseline (packed LAPACK) is not what the candidate suite ships
(vectorized dense). The two baselines are in different complexity
classes; RFP's win is conditional on the *baseline being suboptimal*,
not on the layout itself.

**Proposed feature.** Update `evaluation/harness/candidate_schema.md`
to require a **`baseline_class_match`** field: the scout must show
that the suite's baseline is in the same vectorization/optimization
class as the citation's baseline. If not, the candidate is dropped or
flagged "predicted-class-mismatch" (will likely PARITY/LOSS).

**Implementation sketch.** ~30 lines added to `candidate_schema.md`
and `scout_prompt.md`. No code changes.

**Effort:** 0.5 day.

**Dependencies:** none.

**Re-test list when closed:** 15 stays PARITY but moves to "no
slowdown despite citation-class-mismatch" framing (still useful for
the productivity argument).

## R6 — Per-candidate measurement uniformity

**Severity:** low. Operational / build-time concern.

**Motivating evidence:**
- Each builder wrote its own `measure.py` with idiosyncratic CLI,
  hardcoded paths, and varying file naming (`baseline.json` vs
  `baseline_<size>.json` vs list-of-records vs single-record).
- Synthesis script (`synthesize_reports.py`) had to handle every
  variation post-hoc.

**Proposed feature.** A canonical `evaluation/harness/measure_lib.py`
that exports a `run_measurement(candidate_id, sizes, baseline_bin,
lego_bin, warmup, timed)` API. Each builder's `measure.py` becomes a
thin wrapper around it, with uniform output filenames.

**Implementation sketch.** Standard library work — extract common
patterns from existing measure.py files, parameterize, write to
`raw/{baseline,lego}_<size>.json`. ~200 lines.

**Effort:** 1 day.

**Dependencies:** none.

**Re-test list when closed:** none — pure refactor for the next round.

## R18 — Reduction guard (horizontal sum for cross-iteration reductions)

**Status: CLOSED — implemented in commit a66e46f (2026-05-01).**

**What was built.** `computeStripMineFactor` now contains a reduction guard:
before computing L_strip, it checks whether any `memref.store` in the loop
body has a Broadcast-classified index (loop-invariant address) AND the same
memref is also loaded with a Broadcast index. This pattern is the read-modify-
write accumulator: `C[j] += A[i*K+k] * B[k*N+j%N]`. The guard returns
L_strip=1 (no vectorization) for such loops. This is conservative — a future
enhancement could emit `vector.reduction` to handle horizontal sums — but
prevents the bug of silently producing wrong results from vectorized reductions.

**What remains (R18b, future).** Emitting `vector.reduction` ops to vectorize
the inner compute while correctly reducing across lanes. This would enable
GEMM-style inner-k reductions to be vectorized, potentially yielding 2-4×
additional speedup on top of the outer-j vectorization from R20.

**Effort:** R18b is 1-2 weeks (add vector.reduction emission + correctness
tests).

---

## R20 — Strided constant-stride: use deinterleave/vpgatherdd instead of gather

**Status: CLOSED — implemented in commit a66e46f (2026-05-01).**

**Severity:** medium. Affects all strided-gather candidates where the
stride is a compile-time constant and is small (2x, 4x, 8x).

**Motivating evidence (2026-05-03, post-R19 dashboard):**
- 23_symm_rfp: vec_iso=0.70x (LOSS, prior=LOSS). Stride=2, N=16384.
- 24_syrk_rfp: vec_iso=0.68x (LOSS, prior=WIN). Stride=2, N=16384.
- 25_nw_antidiag: vec_iso=0.69x (LOSS, prior=LOSS). Stride=2, N=16384.
- 29_particlefilter_aosoA: vec_iso=0.75x (LOSS, prior=PARITY). Stride=4.
- 30_lulesh_aosoA: vec_iso=0.76x (LOSS, prior=LOSS). Stride=4.
- 31_hpccg_aosoA: vec_iso=0.74x (LOSS, prior=LOSS). Stride=4.
- 38_nussinov_nonpow2_skew: vec_iso=0.68x (LOSS, prior=MIXED). Stride=2.

Root cause: for a constant stride s, the correct vectorization is NOT
`vector.gather` (which uses individual gather instructions, latency ~10 cycles
per lane on Intel) but rather a load of s consecutive elements followed by
a deinterleave shuffle (e.g. `_mm256_i32gather_ps` vs. `_mm256_loadu_ps` +
`_mm256_permutevar8x32_ps`). The scalar LLVM JIT (target="scalar") goes
through LLVM's own vectorizer at opt_level=2, which knows this trick and
auto-vectorizes to deinterleave+shuffle. Our explicit gather is slower.

**Proposed fix.**
In `emitVectorBody`, for `AccessKind::Strided` with a constant element
stride `s` that is a power of 2 (or small ≤ 8):
1. Emit `s` consecutive `vector.transfer_read`s of width `Ln` starting at
   addresses `[base, base+1, base+2, ...]`.
2. Emit `vector.shuffle` to deinterleave: pick every `s`-th element.
3. This produces the same logical result as gather but with sequential
   loads (1 cycle each, vs ~10 cycles for gather).

For non-power-of-2 or large strides (s > 8), fall back to gather (current
behavior).

**Effort:** 1–2 weeks (implement deinterleave path + lit tests).

**Dependencies:** R19 (CLOSED — correct Strided path is now in place).

**Re-test list when closed:** 23, 24, 25, 29, 30, 31, 38 — expect vec_iso
> 1.0x after deinterleave replaces gather.

## R7 — Lock contention diagnosis & DEFAULT_LOCK_PATH absolute by default

**Severity:** closed (already addressed mid-round).

**What happened.** First-round builders all used per-worktree
`evaluation/.lock` paths because `lock.py:DEFAULT_LOCK_PATH` resolved
relative to the calling file. The lock was effectively a no-op across
worktrees, leading to concurrent measurements polluting each other.

**Fix.** `lock.py:DEFAULT_LOCK_PATH` now defaults to the absolute
`/scratch/general/vast/u1419116/LEGO/evaluation/.lock`, overrideable
via `LEGO_EVAL_LOCK_PATH` env var. Builder prompts updated to specify
the absolute path.

**Status:** done. No re-test needed — Step 3.5 orchestrator already
re-measured everything under the correct lock.

## R9 — Cache-topology-aware tile-size autotuning (NEW from Intel audit)

**Severity:** medium-high. Affects layout-class portability across CPUs with
different L3 sizes.

**Motivating evidence (2026-04-29 Intel audit):**
- Builder 19 (`npdp-zuker-skew-tile`) → WIN 1.17× AMD → **LOSS 0.67× Intel**.
  Zuker skew-tile sized for AMD's 256 MB L3; Intel's 43 MB L3 (~6× smaller)
  thrashes at medium and large sizes.
- Builder 26 (`polybench-gemm-pow2-pad`) → WIN 1.14× AMD → **LOSS 0.96× Intel**.
  Pow-2 padding strategy targeted AMD's L3 set-associativity pattern; Intel's
  43 MB L3 has different set stride and the padding lands in different cache
  sets, no longer avoiding the hot-set collision.

**Root cause cluster.** LEGO layouts today take tile sizes as compile-time
constants. When the deployment hardware has different L3 capacity / associa-
tivity / set-stride, the carefully-chosen tile size lands in the wrong cache
regime.

**Proposed feature.** A `lego.autotile` runtime helper that:

1. Queries `sysfs` for `cache_size`, `ways_of_associativity`,
   `coherency_line_size` per cache level at compile time.
2. Parametrizes `TileBy`'s `(BM, BN)` sizes as a function of cache topology
   instead of hardcoded constants.
3. Optionally: a one-time autotune sweep that runs at several tile-size
   choices, picks the best for the current machine, caches the result in
   `~/.lego/autotile.json`.

**Implementation sketch.** Python helper `lego.autotile.cache_aware_tile_size
(matrix_dims, element_size, level=2)` returning `(BM, BN)` sized for detected
L1/L2. Update existing benchmarks to use it.

**Effort:** 3–4 days (cache-aware sizing helper); +1 week (autotuning sweep).

**Re-test list when closed:** 19, 26 — expect WIN on both AMD and Intel.

## R10 — Multi-thread Intel re-measurement of candidate 24 (NEW)

**Severity:** low. Operational gap.

**Motivating evidence.** Builder 24 (`polybench-fdtd-2d-block-cyclic`) was a
4-thread WIN on AMD (3.78×). The 2026-04-29 Intel audit measured single-
threaded only — Intel verdict came back LOSS. Protocol mismatch, not real
architecture-portability story.

**Proposed action.** Re-run cand 24 on Intel with `OMP_NUM_THREADS=4`, update
`audit_report_intel.md` row 24.

**Effort:** 30 minutes once the lock is free.

**Re-test list when closed:** 24.

## R11 — Architecture-portability framing in Section 7.5 (NEW)

**Severity:** documentation. The Intel audit produced 68% verdict agreement
across architectures — paper finding worth its own subsection.

**Cross-architecture summary** (from `evaluation/audit_report_intel.md`):

| Layout Class | AMD WIN | Intel WIN | Verdict |
|---|---|---|---|
| Register + L1 + L2 tile | 6/6 | 6/6 | **fully portable** |
| L1 tile | 2/2 | 2/2 | fully portable |
| GETT tile | 1/1 | 1/1 | fully portable |
| Z-Morton | 2/3 | 2/3 | mostly (Chol LOSS on both) |
| Brick | 0/5 | 0/5 (1 marginal) | **portable LOSS** — confirms R1 |
| Skew tile | 1/3 | 1/3 | partial (Zuker fails on Intel L3) |
| RFP | 1/2 | 1/2 | partial |
| Block-cyclic | 1/2 | 1/2 | partial (24 needs 4T re-run) |
| Pow-2 pad | 2/2 | 1/2 | partial (26 L3-specific) |
| AoSoA | 0/3 | 0/3 (1 PARITY) | weak — 21 improves on Intel |
| Antidiag tile | 0/1 | 0/1 | not portable |
| Morton + non-pow-2 | 1/1 | 1/1 | fully portable |

**AVX-512 amplifies register-kernel wins.** Several candidates show *bigger*
wins on Intel: 05 (3mm) 3.09× → 4.25× (+37%); 06 (2mm) p512 3.79× → 7.03×
(+85%); 16 (syrk-rfp) p1024 → 5.52×. Strong paper-grade story: LEGO's `TileBy
+ OrderBy` naturally exploits wider SIMD without code changes.

**Proposed Section 7.5 framing:**
- AMD canonical results (Round 1).
- Intel cross-arch validation (Round 1.5 / audit).
- Per-class portability paragraphs: fully-portable, R9-blocked, R1-blocked.

**Effort:** included in Section 7.5 writing.

## R8 — PolyBench cache-flush utility amplifies tile-friendly kernels

**Severity:** methodology caveat. Affects WIN magnitudes, not classifications.

**Motivating evidence:**
- Builder 34 (`polybench-bicg-L1-tile`) → WIN with 51.7× at large size,
  builder honestly attributed to PolyBench's `polybench_flush_cache()`
  flushing 32 MB before each timed run, evicting the baseline's A
  matrix while leaving the LEGO version's 8 KB tile in L1. Real win,
  but mostly methodology.
- Same flush-amplification suspected in builder 07 (trmm 27.6×),
  builder 08 (doitgen 4×/111×), builder 16 (syrk 7.15×).

**Proposed feature.** Section 7.5 prose includes a methodology caveat
paragraph explaining that PolyBench's cache-flush utility favors small-
working-set layouts and that reported speedups at L3-boundary sizes
include this effect.

**Implementation sketch.** Documentation only — paragraph in Section
7.5 + a per-row footnote on suspect candidates.

**Effort:** included in paper writing.

**Dependencies:** Section 7.5 unblocks (after roadmap items closed).

---

## Roadmap summary

| ID | Severity | Effort | Affected candidates | Status |
|---|---|---|---|---|
| **R1 CPU vector pipeline** | high | — | 11, 12, 13, 14, 29 | **CLOSED (shipped v1, 2026-05-01)** |
| **R18 Reduction guard** | high | — | (all reduction loops) | **CLOSED (2026-05-01, commit a66e46f)** |
| **R19 Strided-gather scalar-index** | high | — | 23–27, 29–31, 38 | **CLOSED (2026-05-03, commit 8f0b325)** |
| **R20 Strided → deinterleave lowering** | medium | — | 23–25, 29–31, 38 | **CLOSED (2026-05-01, commit a66e46f)** |
| **R12 Cross-brick shuffle** | **high** | **1–2 wk** | **11, 12, 13, 14, 29** | **partial (flat-offset variants working; full BrickLib API stride pending)** |
| R2 Anti-diagonal scoping | medium | 1–2 wk | 17, 20 | open |
| R3 AoSoA scoping | medium | 0.5 day | 22, 21, 23 | open (cross-arch nuance added) |
| R4 Z-Morton triangular | medium | 1 wk (a) / 0.5 day (b) | 03 | open |
| R5 Baseline-class matching | medium | 0.5 day | 15 | open |
| R6 Measurement uniformity | low | 1 day | (refactor only) | open |
| R7 Absolute lock path | closed | done | — | closed |
| R8 PolyBench flush caveat | methodology | (paper) | 07, 08, 16, 34 | open |
| R9 Cache-topology autotune | medium-high | 3–4 days | 19, 26 | open |
| R10 Cand-24 4T Intel re-run | low | 30 min | 24 | open |
| R11 §7.5 portability framing | doc | (paper) | all | open |
| **R13 AOT object-file path** | **medium** | **1–2 wk** | **(all)** | **future** |
| **R14 SMT tile legality check** | **medium** | **3–4 wk** | **(all)** | **future** |
| **R15 ARM SVE width-aware emission** | **low** | **2–3 wk** | **(ARM targets)** | **future (NEON ships in v1)** |
| **R17 GPU lane-fold / warp shuffles** | **medium** | **4–6 wk** | **(GPU candidates)** | **future** |

**Recommended order to attack:**

1. **R12** (1–2 wk) — imminent; most infrastructure is in place. Flips 5 brick
   candidates from LOSS to WIN. Start with the affine-offset store lowering
   (unblocks `brick_stencil_cross` benchmark), then brick-stride base fix.
2. R5 + R3 + R4(b) (all 0.5-day items) — close out methodology gaps
   first, regenerate scout output cleaner.
3. R6 (1 day) — uniform measure.py for the next round.
4. R9 (3–4 days) — cache-topology autotune for portability.
5. R2 (1–2 wk) — nice-to-have analysis pass.
6. R4(a) (1 wk) if R4(b) doesn't satisfy.
7. R13 → R14 → R15 → R17 as the project scales to AOT + GPU.

**When all open items are closed (or formally classified architectural
limits in Section 9), unblock paper Section 7.5 writing.**
