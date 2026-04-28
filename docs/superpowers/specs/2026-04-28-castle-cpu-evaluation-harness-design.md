# CASTLE CPU Evaluation Harness — Design

**Date:** 2026-04-28
**Author:** Amir Mohammad Tavakkoli (with Claude as design partner)
**Status:** Draft pending user sign-off
**Branch:** `eval/cpu-source-emission`
**Related paper:** `CASTLE-tex/draft.md` (TACO submission), Section 7.5 placeholder

---

## 1. Goal

Replace the `[PLACEHOLDER: runtime performance ...]` in CASTLE-tex/draft.md
Section 7.5 with real, defensible CPU-side speedup numbers obtained from a
reproducible, parallel research harness. The harness orchestrates one *scout*
subagent (literature + benchmark survey) and many *builder* subagents (one per
candidate kernel), all under explicit honesty rules:

- No fabricated numbers.
- Every cited claim is backed by a real BibTeX entry with DOI or arXiv ID.
- Every measured result has a corresponding raw-timing JSON file.
- LEGO version's output is byte-identical (integer) or within tolerance
  (floating-point) to the suite's reference output.
- Builders may not modify CASTLE source for this round (escape hatch closed).

## 2. Non-goals

- GPU DSL feature work (deferred to a future round).
- Tensor API / PyTorch / `torch.compile` integration (out of scope).
- New MLIR dialect ops, new lowering paths, or new MLIR passes
  (escape hatch closed).
- New SymPy code printers for languages CASTLE doesn't already emit.
- Running benchmarks on hardware other than the currently allocated CHPC node.
- Comparing against retuned baselines — baseline is the suite's as-shipped
  form.

## 3. Scope of layout-level optimization

The paper-grade win must come from a **layout-level** optimization that
naive `gcc -O3` (or equivalent for the target language) cannot recover from
the suite's as-shipped source. Eligible layout classes:

- Cache-oblivious recursive layouts (Z-Morton, Hilbert).
- Multi-level cache-conscious tiling (register × L1 × L2 × L3).
- Recursive bricking for stencils.
- Triangular / symmetric packing (RFP-style).
- Skewed / shifted layouts for diagonal sweeps (LU, NW, dynamic programming).
- AoSoA / interleaved struct packing for vectorization.
- Block-cyclic distribution for thread-level locality.
- Padding to break power-of-two stride associativity conflicts.
- **Power-of-two-restricted optimizations applied to non-power-of-two
  problem sizes.** Many published layout optimizations (swizzle masks,
  bit-trick index arithmetic, bank-count assumptions, fixed-tile-size
  CUTLASS-style GEMM kernels) are restricted to power-of-two dimensions
  in their original implementation. CASTLE's algebra has no such
  restriction: `RegP` / `GenP` / `OrderBy` / `TileBy` work over arbitrary
  dims and the strength-reduction pass simply does not fire when shifts
  cannot replace divisions. A candidate that reproduces a published
  pow-2-only optimization on a non-pow-2 problem size — and beats the
  best generally-applicable baseline at that size — is itself a paper-
  grade contribution.

Each must be expressible using only the LEGO primitives `Row`, `Col`,
`RegP`, `GenP`, `OrderBy`, `TileBy`. **`GroupBy` is forbidden in
user-facing layout code.** Tensor-API helpers (`lego.ZCurve`,
`lego.Swizzle`, `lego.BlockCyclic`, `lego.Tiled`, `Batched`) are not used —
if a layout requires Z-Morton, the builder writes the Morton bijection
inside a `GenP`.

## 4. Pipeline

```
user-written LEGO expression (Row/Col/RegP/GenP/OrderBy/TileBy)
    → SymPy
    → lego MLIR dialect
    → MLIR pass pipeline (as-shipped)
    → emitted source code (C / C++ / Fortran / Rust / Julia / CUDA-C / JS / GLSL)
    → suite's native compiler (gcc / gfortran / rustc / julia)
    → wall-clock measurement
```

## 5. Branch & directory layout

**Branch:** `eval/cpu-source-emission` off `main`.
**Builder branches:** one per candidate, named `eval/cpu-<id>`, off
`eval/cpu-source-emission`.
**Worktree per builder:** `../LEGO-eval-<id>` (sibling of the main repo
checkout), created with `git worktree add` so parallel builders cannot
collide on filesystem state.

```
evaluation/
├── README.md                   # how to reproduce end-to-end
├── harness/
│   ├── scout_prompt.md         # exact prompt the scout subagent receives
│   ├── builder_prompt.md       # exact prompt each builder subagent receives
│   ├── candidate_schema.md     # required structure for one survey row
│   ├── result_schema.json      # raw timing JSON + verification format
│   ├── machine.md              # the locked measurement node's specs
│   ├── verify.py               # output-hashing reference comparator
│   ├── lock.py                 # flock-based global mutex (N=1)
│   ├── dashboard.py            # regenerates evaluation/dashboard.md
│   └── stats.py                # median, IQR, speedup classification
├── references.bib              # BibTeX, scout-owned, builders append
├── survey.md                   # scout's deliverable
├── survey_summary.md           # scout's drop-list summary
├── dashboard.md                # auto-regenerated status table
└── candidates/
    └── 01-<suite>-<kernel>/    # one dir per candidate, one builder owns each
        ├── upstream/           # vendored suite source, untouched
        ├── kernel_baseline.<ext>
        ├── kernel_lego.py
        ├── run.sh
        ├── raw/
        │   ├── baseline.json
        │   └── lego.json
        ├── verify.log
        └── report.md
```

The directory is a sibling of `paper/` to keep `CASTLE-tex/` clean of
implementation churn; results distill into Section 7.5 prose only at the
end of the round.

## 6. Coding conventions enforced by the harness

- **Prefer `OrderBy` + `TileBy`** in user-facing layouts. `GroupBy` is
  allowed when no clean `TileBy` expression exists (e.g. asymmetric
  grouping that doesn't reduce to a tile-size pair, or grouping over a
  non-contiguous dim subset). Each `GroupBy` use must be accompanied by a
  one-sentence justification in `report.md` under the `groupby_usage`
  field. `verify.py` does not fail on `GroupBy`; it counts occurrences
  and surfaces them in the dashboard so we can see how often the
  exception was needed across the round.
- **Identical compiler flags** for baseline and LEGO version. The flag
  string per language is fixed in `harness/build_flags.json`:
  - C / C++: `-O3 -march=native -fopenmp`
  - Fortran: `-O3 -march=native -fopenmp`
  - Rust: `-C opt-level=3 -C target-cpu=native`
  - Julia: `--check-bounds=no -O3`
- **Compiler version recorded** in every `raw/*.json`.
- **No new CASTLE source modifications** — `git diff main -- lib/ include/
  python/lego/ tools/ test/` must return empty for any builder branch.

## 7. Scout subagent contract

**Mandate.** One scout, runs in foreground (so we approve its output before
launching any builder). Reads papers and benchmark suites. Writes
`evaluation/survey.md` and `evaluation/references.bib`. Never writes code,
never edits the LEGO repo, never runs benchmarks.

**Inputs handed to the scout via `scout_prompt.md`:**

- The eight layout-optimization classes from Section 3.
- The six LEGO primitives (`Row`, `Col`, `RegP`, `GenP`, `OrderBy`, `TileBy`).
- The eight source-emission targets (C, C++, Fortran, Rust, Julia, CUDA-C,
  JS, GLSL).
- The OrderBy + TileBy convention.
- The "no new dialect ops or lowering paths" rule.
- The cite-everything rule.

**Survey row schema (`candidate_schema.md`).** Every row must fill every
field. Drop rule: any missing field → drop the candidate.

```yaml
id: polybench-2mm-zcurve         # slug, also the directory name
suite: PolyBench/C 4.2.1         # name + version
kernel: 2mm                      # specific kernel within the suite
upstream_url: https://...        # exact location of source
license: MIT                     # accepted: MIT, BSD-2-Clause, BSD-3-Clause,
                                 # Apache-2.0, ISC, public domain, CC0;
                                 # copyleft (GPL, LGPL, AGPL) → drop
language: c++                    # one of the eight shipping printers
baseline:
  source_files: [linear-algebra/kernels/2mm/2mm.c]
  build: "gcc -O3 -march=native -fopenmp"
  threading: 24-thread OpenMP
layout_trick: "Z-curve traversal of the output tile"
layout_trick_citation: frigo1999cacheoblivious   # BibTeX key
why_compiler_cant: |
  PolyBench ships a row-major triple-nested loop. gcc -O3 will not
  reorder iteration to a Morton path; pluto would, but pluto is not
  invoked at -O3. Layout choice changes L2 miss rate, not just
  arithmetic.
lego_expressibility: |
  OrderBy(Row(M, N)).TileBy((M//BM, N//BN), (BM, BN)) composed with a
  GenP that maps tile coordinates to a Morton index.
predicted_win:
  value: "1.3x – 2.0x"
  source: frigo1999cacheoblivious
  type: published                 # one of: published | extrapolated | unknown
power_of_two_restriction:
  baseline_assumes_pow2: true     # original published win restricted to pow-2 dims?
  test_at_non_pow2_size: true     # if true, builder also runs at a deliberately
                                  # non-pow-2 size to demonstrate CASTLE's generality
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; turbo and
  governor as observed.
estimated_builder_effort: 1-2 days
risk_flags:
  - PolyBench dataset sizes may be too small to amortize tiling
```

**Granularity rule.** One row per **(kernel × layout-trick)** tuple, not
per kernel. Same kernel under different layouts becomes multiple
candidates so the paper's evaluation matrix can compare layouts head to
head.

**Honesty rules in the scout prompt:**

- No invented numbers. `predicted_win.type: unknown` is acceptable; a
  fabricated number is not.
- Every BibTeX key resolves to a real paper with DOI or arXiv ID.
- `why_compiler_cant` must name the specific compiler pass that would be
  needed (polyhedral, vectorization with non-affine indices, etc.) and
  why it doesn't fire on naive code at the suite's baseline build flags.

**Estimated count.** Realistic survivor count is **~30–50 candidates**
spanning at least six layout classes (see Section 3 sources estimate).
Scout returns every candidate that passes the drop rules; no upfront cap.

**Scout deliverables:**

1. `evaluation/survey.md` — ranked table; one yaml block per row plus a
   3–4 sentence "why this is interesting" intro paragraph per row.
2. `evaluation/references.bib` — every cited work as a full BibTeX entry
   with DOI or arXiv field populated.
3. `evaluation/survey_summary.md` — bullet list: which layout classes are
   represented, which are not, which kernels were dropped and why.

## 8. Builder subagent contract

**Mandate.** One builder per surviving survey row. Each runs in its own
worktree on its own branch (`eval/cpu-<id>`). Edits only files inside
`evaluation/candidates/<id>/`. May not touch `lib/`, `include/`,
`python/lego/`, `tools/`, or `test/`.

**Worktree lifecycle:**

```
git worktree add ../LEGO-eval-<id> eval/cpu-source-emission
cd ../LEGO-eval-<id>
git checkout -b eval/cpu-<id>
source venv/bin/activate
# … work …
git push origin eval/cpu-<id>
git worktree remove ../LEGO-eval-<id>
```

**Build–measure–iterate loop:**

1. **Vendor baseline.** Pull suite source under
   `candidates/<id>/upstream/`, untouched. License check; abort if
   non-permissive.
2. **Build baseline.** Suite's documented build incantation. Record
   exact command in `run.sh`. Verify it produces the suite's reference
   output.
3. **Author LEGO version.** Write `kernel_lego.py` using only `Row`,
   `Col`, `RegP`, `GenP`, `OrderBy`, `TileBy`. Generate source via the
   matching `lego.<lang>_gen.generate()` printer. Splice generated index
   arithmetic into the suite's kernel skeleton (data structures, I/O,
   timing harness unchanged).
4. **Verification gate (must pass before timing).** Run baseline + LEGO
   on the same input. Compare outputs:
   - Integer kernels → byte-identical, SHA-256 match.
   - Floating-point → `5e-6` relative tolerance for FP32, `1e-12` for
     FP64. Max-abs and max-rel error logged.
   - Fail → write `DROPPED-verification` to `report.md` and stop. No
     timing.
5. **Acquire the global lock** (`flock evaluation/.lock`). Hold for the
   full build + measurement cycle.
6. **Measure.**
   - Apply `taskset -c <core>` for thread pinning (always works).
   - Apply `numactl --membind`, governor=performance, turbo-disable
     **if available**; record the actual state in `repro_setup`.
   - **25 warmup → 100 timed iterations**, record per-iteration ns.
   - Sweep at minimum **3 problem sizes** (small / medium / large
     suite-defined).
   - Same protocol for baseline and LEGO under identical conditions.
   - Output: `raw/baseline.json`, `raw/lego.json`.
7. **Release the lock.**
8. **Classify.** Compute median and IQR for each timing distribution.
   Speedup = `baseline_median / lego_median`. Compare on the median
   ratio alone:
   - `speedup ≥ 1.02` → **WIN**.
   - `0.98 < speedup < 1.02` → **PARITY**.
   - `speedup ≤ 0.98` → **LOSS**.
   - Compile or runtime crash on LEGO side → **DROPPED-build**.

   IQR is recorded for sanity-checking but does not gate the verdict.
   Suspiciously high variance (IQR > 20% of median) is flagged in the
   dashboard for re-run review; it does not auto-disqualify.
9. **Iterate within scope.** If LOSS or unconvincing PARITY, builder may
   try other layouts (Z-Morton, RFP, deeper tiling) for the same kernel,
   sweep tile sizes, retry. Each attempt logged in `report.md`. **Hard
   cap: 8 layout attempts per candidate**, beyond which the candidate
   ships its best result and exits.
10. **Write `report.md`.** See Section 9.
11. **Commit + push branch `eval/cpu-<id>`.** No PR opened — orchestrator
    reviews after all builders finish.

**Best-effort reproducibility.** The harness records what it could and
could not apply on the current node. Honesty clause: any `report.md`
whose `repro_setup.turbo == "unknown"` says so explicitly. The paper's
Section 7.5 caveat sentence will state measurements were taken under
user-mode CHPC conditions, no frequency-lock claim.

**Iteration boundaries — what builders may NOT do:**

- Modify CASTLE source.
- Compare against a baseline they retuned themselves.
- Pick problem sizes that flatter LEGO; sizes are suite-defined or three-
  point sweep.
- Report a number without a corresponding `raw/*.json` it was computed
  from.
- Use any `lego.ZCurve` / `Tensor` / `PyTorch` API.

## 9. `report.md` schema

Every builder writes this file at the end of its run.

```yaml
candidate_id: polybench-2mm-zcurve
status: WIN | PARITY | LOSS | DROPPED-verification | DROPPED-build | DROPPED-needs-lowering
machine: <hostname>, <CPU model>, governor=<>, turbo=<off/on/unknown>
compiler: gcc 13.2.0
build_flags: "-O3 -march=native -fopenmp"
sizes_swept: [small, medium, large]
results:
  - size: medium
    baseline_median_ns: 1234567
    baseline_iqr_ns: 18000
    lego_median_ns: 987654
    lego_iqr_ns: 14000
    speedup: 1.25
    verification: PASS (sha256 match)
layouts_tried:
  - "OrderBy(Row(M,N)).TileBy(...)": LOSS at 0.93x
  - "OrderBy(Row).TileBy(...) + GenP(morton)": WIN at 1.25x
groupby_usage: []                    # list of {expression, justification}
                                     # if empty, layout used only OrderBy+TileBy
citations_used: [frigo1999cacheoblivious]
notes: |
  Free-form: anything the orchestrator should see — odd findings,
  potential improvements outside scope, suspected compiler-version
  sensitivity, etc.
```

## 10. Concurrency model

**Single global mutex via `flock(evaluation/.lock)`, N=1.**

Why N=1 and not the originally proposed two-lock (build N=4 + measure
N=1) design: with 30–50 builders on one shared node, parallel builds
during another builder's measurement contaminate timings via cache and
memory-bandwidth contention, regardless of taskset pinning. The only
honest answer is fully serial build+measure cycles. Reports, git
commits, and dashboard regeneration happen outside the lock.

**Wall-clock estimate:**

- Per candidate: build + verify + 25 warmup + 100 timed × 3 sizes × 2
  versions + ~3 layout retry attempts. Roughly 2× the iteration cost of
  the earlier 10/50 protocol, so per-candidate wall-clock is dominated
  by measurement and ranges ~5–25 min depending on kernel size.
- 50 candidates × ~15 min average = ~12 hours measurement-critical-path.
  Larger kernels (LULESH-class) push the upper bound to ~25 min each.
- Survey + distillation ~1.5 hours.
- **Total ~12–20 hours unattended** for a 30–50 candidate round, larger
  for full-stencil-suite kernels.

## 11. Hardware target

**Locked to the currently allocated CHPC node.** No `srun --exclusive`,
no cross-node measurements. The harness captures the actual node's specs
once at scaffolding time into `harness/machine.md`:

- `hostname`, `lscpu`, `numactl --hardware`, `uname -a`
- `gcc --version`, `gfortran --version`, `rustc --version`,
  `julia --version`
- Governor and turbo state as observed at scaffolding time
- GPU presence and model (recorded but unused unless a CUDA-C candidate
  surfaces opportunistically)

If the node changes between scaffolding and measurement, the harness
re-fingerprints and aborts if material specs differ.

## 12. Orchestration & cadence

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1 (orchestrator, foreground, ~30 min)                       │
│   - Branch eval/cpu-source-emission already exists.              │
│   - Scaffold evaluation/{harness/, README.md}.                   │
│   - Write scout_prompt.md, builder_prompt.md, candidate_schema   │
│   - Write verify.py, lock.py, dashboard.py, stats.py, machine.md │
│   - Commit. Show user the tree + the prompts.                    │
│   ──→ STOP. User reviews. Edits prompts if desired. Approves.    │
├─────────────────────────────────────────────────────────────────┤
│ Step 2 (scout subagent, foreground, ~30–60 min)                  │
│   - Orchestrator dispatches scout with scout_prompt.md.          │
│   - Scout writes survey.md + references.bib + survey_summary.md. │
│   ──→ STOP. User reviews survey table + bib. Drops/amends rows.  │
├─────────────────────────────────────────────────────────────────┤
│ Step 3 (builder subagents, parallel-spawn, serial-execute,        │
│         background, ~10 hours wall-clock)                         │
│   - Orchestrator dispatches one builder per surviving row,        │
│     run_in_background=true.                                       │
│   - Each runs its build-measure-iterate loop in its worktree.    │
│   - Global mutex serializes the build+measure cycles.             │
│   - dashboard.md auto-regenerated when each builder completes.    │
│   ──→ STOP. User reviews dashboard + per-candidate report.md.    │
├─────────────────────────────────────────────────────────────────┤
│ Step 4 (orchestrator, foreground, ~1 hour)                        │
│   - For survivors: distill report.md files into                   │
│     evaluation/section_7_5.md with speedup matrix table,          │
│     methodology paragraph, honest list of LOSSes & DROPs,         │
│     CHPC measurement caveat paragraph.                            │
│   ──→ STOP. User reviews prose, edits, decides if it goes into    │
│       CASTLE-tex/draft.md as the new Section 7.5.                 │
├─────────────────────────────────────────────────────────────────┤
│ Step 5 (user, optional)                                           │
│   - Open PR from eval/cpu-source-emission → main, or              │
│   - Cherry-pick just evaluation/ + section_7_5.md, or             │
│   - Discard branch and start a new round.                         │
└─────────────────────────────────────────────────────────────────┘
```

**Progress visibility.** `evaluation/dashboard.md` is the single source
of truth during Step 3. User can `git pull && cat evaluation/dashboard.md`
at any time. No /loop, no cron — work is finite and terminates.

**Interrupt model.** `harness/abort.sh <id>` signals a builder to stop;
`harness/restart.sh <id>` cleans the worktree and re-dispatches.

## 13. Result classification thresholds (recap)

| Class                  | Condition                                                       | Paper treatment                                          |
|------------------------|-----------------------------------------------------------------|----------------------------------------------------------|
| WIN                    | `speedup ≥ 1.02` (i.e. lego ≤ 0.98 × baseline)                  | Reported as speedup row in Section 7.5 matrix            |
| PARITY                 | `0.98 < speedup < 1.02`                                         | Reported as "no slowdown despite portability"            |
| LOSS                   | `speedup ≤ 0.98` (i.e. lego ≥ 1.02 × baseline)                  | Reported honestly with hypothesis about cause            |
| DROPPED-verification   | LEGO output fails reference comparison                          | Excluded from matrix; mentioned in caveats               |
| DROPPED-build          | LEGO source fails to compile or run on this node                | Excluded from matrix; mentioned in caveats               |
| DROPPED-needs-lowering | Candidate cannot be expressed without modifying CASTLE source   | Excluded from matrix; logged in dashboard for re-open    |

## 14. Risks & mitigations

| Risk                                                                                      | Mitigation                                                                                  |
|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `gcc -O3` already simplifies LEGO-emitted index arithmetic to be equivalent to baseline   | Scout's `why_compiler_cant` field forces specific articulation; layouts that are no-ops for the compiler get dropped before builder runs |
| Many candidates need new lowering features                                                | Single tally in dashboard; if >30% drop with `DROPPED-needs-lowering`, surface to user as a signal to re-open the escape hatch |
| Measurement noise on a non-frequency-locked node                                          | 25 warmup + 100 timed iterations per (size, version); IQR > 20% of median flags re-run; serial measurement under global mutex |
| 30–50 builder worktrees fill scratch                                                      | Worktree path under `/scratch/general/vast/u1419116/LEGO-eval-<id>`; 1 GB ceiling per worktree enforced by the builder prompt |
| Builders silently fabricate numbers                                                       | Every reported number must point to a `raw/*.json` it was computed from; verify.py grep-checks reports for orphaned numbers |
| Compiler version drift between scout's predictions and builder's measurements             | `report.md` records actual compiler version; predicted-vs-measured discrepancy logged in `notes` |

## 15. Out-of-scope, recorded for future rounds

- GPU DSL feature work (cp.async pipelines, MMA beyond puzzle 33, warp
  specialization, swizzled shared memory).
- Tensor API + PyTorch integration evaluation.
- Multi-node / distributed-memory layouts beyond what `BlockCyclic`-style
  expressions can stand in for on a single node.
- Autotuning over tile sizes (builders sweep manually within their 8-
  attempt cap; not the same as a search-based autotuner).
- New SymPy printer subclasses for languages CASTLE doesn't already emit.

## 16. Acceptance criteria for this design

The design is accepted when the user:

1. Confirms the directory layout matches their working preference.
2. Confirms the result classification thresholds (2% effect size on the median ratio; no CI/bootstrap gate).
3. Confirms the scout's `candidate_schema.md` covers what they need.
4. Confirms the wall-clock estimate (~12–20 hours unattended) is
   acceptable.
5. Confirms `OrderBy` + `TileBy` is sufficient and `GroupBy` should
   indeed be banned.

Once accepted, the orchestrator proceeds to Step 1 of Section 12 —
scaffolding the harness — and surfaces the scaffolded tree for review
before dispatching the scout.
