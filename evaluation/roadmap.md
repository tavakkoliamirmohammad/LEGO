# LEGO/CASTLE Infrastructure Roadmap

> **Status:** open. This document is the actionable output of the CASTLE
> CPU evaluation round (branch `eval/cpu-source-emission`). Each entry
> below is a concrete LEGO/CASTLE infrastructure gap surfaced by a LOSS
> or PARITY result, with a proposed feature, a sketched design, and an
> effort/dependency estimate. The list grows as more results land.
>
> **The CASTLE/TACO paper Section 7.5 is HELD** until either each item
> below is closed (infra implemented + affected candidates re-measured)
> or definitively classified as architectural-limit (paper Section 9).

## R1 — SIMD intrinsic codegen via target-specific MLIR dialects

**Severity:** high. Affects the entire stencil/brick layout class.

**Cross-architecture confirmation (Intel Xeon Gold 6330 audit, 2026-04-29):**
brick LOSSes reproduce on **both** AMD AVX2 and Intel AVX-512 (11, 14, 29 LOSS
on both). Candidate 12 (3d13pt-brick) flips marginally to Intel WIN 1.06× via
AVX-512 gathers — wider vectors *partially* close the gap but the intrinsic-
codegen lift is still needed. **R1 is confirmed not arch-specific.**

**Motivating evidence:**
- Builder 11 (`bricklib-3d7pt-brick`) → LOSS 0.93× AMD / LOSS 0.86× Intel.
- Builder 12 (`bricklib-3d13pt-brick`) → PARITY AMD / **marginal WIN 1.06× Intel** (AVX-512 gathers help 13-point stencil).
- Builder 13 (`polybench-heat3d-brick`) → MIXED AMD / LOSS 0.80× Intel.
- Builder 14 (`polybench-jacobi2d-brick`) → LOSS 0.67× AMD / LOSS 0.19× Intel (severe).
- Builder 29 (`bricklib-stencil-nonpow2-brick`) → LOSS 0.75× AMD / LOSS 0.61× Intel.

**Root cause cluster.** BrickLib's published 1.9×–4.9× wins depend on
register-level dimension folding (the `brick()` macro folds the
innermost `lz` axis into AVX2 lanes). LEGO's source-emission today is
*layout-algebraic*: it computes index expressions but cannot emit
target-specific SIMD intrinsics. So we capture the cache-locality side
of bricking and lose the vectorization side, ending up below the naive
baseline (whose row-major auto-vectorization GCC handles well).

**Proposed feature.** A `lego.simd_lower` MLIR pass running *after*
`lego-strength-reduction` that:

1. Identifies inner loops over a "small enough" extent (≤ register
   width × 8 lanes, e.g. 4–8 doubles for AVX2).
2. Rewrites those loops to MLIR's `vector` dialect.
3. Lowers via `vector → x86vector` (or `arm_neon` / `arm_sve`) as
   selected by the target triple.

**Implementation sketch.**
- New CMake target `lego-simd-lower` in `lib/Lego/Conversion/`.
- Reuse MLIR's `vectorize-loops` pass family for the rewrite half;
  contribute a CASTLE-specific cost model that knows brick stride
  patterns are vectorizable.
- Plumb a `--target-cpu=zen3|skx|neoverse` flag through `lego-opt`.

**Effort:** 4–6 weeks (one engineer). Most of the work is the cost
model + target selection; the lowering itself reuses upstream MLIR.

**Dependencies:** needs an MLIR contribution path (or a vendored MLIR
fork) for the few patches that don't exist upstream.

**Re-test list when closed:** 11, 12, 14, 29 — re-run, expect WIN
1.5×–3× per the published BrickLib numbers.

**Bonus.** Closes the GPU MMA tensor-core intrinsic gap too — same
infrastructure plumbing, different target dialect.

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
| R1 SIMD intrinsics | high | 4–6 wk | 11, 12, 13, 14, 29 | open (confirmed cross-arch) |
| R2 Anti-diagonal scoping | medium | 1–2 wk | 17, 20 | open |
| R3 AoSoA scoping | medium | 0.5 day | 22, 21, 23 | open (cross-arch nuance added) |
| R4 Z-Morton triangular | medium | 1 wk (a) / 0.5 day (b) | 03 | open |
| R5 Baseline-class matching | medium | 0.5 day | 15 | open |
| R6 Measurement uniformity | low | 1 day | (refactor only) | open |
| R7 Absolute lock path | closed | done | — | closed |
| R8 PolyBench flush caveat | methodology | (paper) | 07, 08, 16, 34 | open |
| **R9 Cache-topology autotune** | **medium-high** | **3–4 days** | **19, 26** | **open (NEW)** |
| **R10 Cand-24 4T Intel re-run** | **low** | **30 min** | **24** | **open (NEW)** |
| **R11 §7.5 portability framing** | **doc** | **(paper)** | **all** | **open (NEW)** |

**Recommended order to attack:**

1. R5 + R3 + R4(b) (all 0.5-day items) — close out methodology gaps
   first, regenerate scout output cleaner.
2. R6 (1 day) — uniform measure.py for the next round.
3. R1 (4–6 wk) — biggest win, biggest effort. Unlocks the brick class
   AND the GPU MMA story for a future GPU evaluation round.
4. R2 (1–2 wk) — nice-to-have analysis pass.
5. R4(a) (1 wk) if R4(b) doesn't satisfy.

**When all open items are closed (or formally classified architectural
limits in Section 9), unblock paper Section 7.5 writing.**
