# Research-Loop Harness Pattern

> **Status:** project-agnostic methodology document. Distilled from the CASTLE
> CPU evaluation round (Apr 2026), but the pattern applies to any
> benchmark-driven research evaluation where you want **honest, defensible,
> reproducible performance numbers** out of an LLM-orchestrated subagent loop.
>
> Copy / adapt this doc into any new research project.

## When to use this pattern

Use it when you have:

- A claim you want to evaluate empirically (e.g. "transform X yields a
  speedup over baseline Y on benchmark suite Z").
- More candidate experiments than you can hand-run (≥ ~10).
- LLM agents available to dispatch as research workers.
- Shared compute (CHPC, cluster nodes, anything with serial access to a CPU/GPU)
  where measurement contention is a real risk.

**Don't use it for:**

- One-off ad hoc benchmarks (overhead not worth it).
- Pure correctness or formal verification work (this pattern measures *real*
  numbers, not model checks).
- Workloads where you already have all the data — this is for *generating*
  data, not analyzing it.

## Phase model

```
┌──────────────┐   ┌──────┐   ┌──────────────┐   ┌──────┐   ┌─────────────────┐   ┌──────┐   ┌──────┐
│ brainstorm + │ → │ spec │ → │ scaffold     │ → │ scout│ → │ builders        │ → │ audit│ → │ road │
│ tradeoff Q&A │   │ doc  │   │ harness/     │   │ sub- │   │ (one per cand,  │   │ pass │   │ map  │
│ with user    │   │      │   │ prompts/     │   │ agent│   │ parallel + lock-│   │      │   │ .md  │
│              │   │      │   │ schemas/     │   │      │   │ serialized      │   │      │   │      │
│              │   │      │   │ stats/lock   │   │      │   │ measurement)    │   │      │   │      │
└──────────────┘   └──────┘   └──────────────┘   └──────┘   └─────────────────┘   └──────┘   └──────┘
   ↑──────────────── iterate the loop after each round ────────────────↑
```

### 1. Brainstorm

Before scaffolding anything, surface the design tensions:

- What's the success metric? (Speedup? Coverage? Reproduction of published
  results?) Be specific — "we want to show X" not "we want to evaluate".
- What's the time horizon? (Days, weeks, months?) Different horizons justify
  different scaffolding investment.
- What are the *negative* outcomes you want to be able to claim? Honest "we
  tried but it didn't work" results are paper-strengthening; dishonest ones
  are paper-killing.
- Where do baselines come from? (Suite-shipped? Hand-tuned reference? Library
  call?) Match baseline class to citation class — easy to forget.

Get the user to say "yes" on the design before scaffolding. The brainstorming
skill (superpowers:brainstorming) is the right tool here.

### 2. Spec

Produce a written design document (~400–600 lines) covering:

- Goal & non-goals (be explicit about what you're NOT measuring).
- Branch + directory layout: separate evaluation work from main source.
- Conventions enforced by the harness (style, build flags, banned imports).
- Scout subagent contract — exactly what it produces and what its drop rules
  are.
- Builder subagent contract — exactly what each one does, in what order, with
  what failure modes.
- Result classification: WIN / PARITY / LOSS thresholds, IQR flagging.
- Single global mutex for serial measurement (critical — see methodology).
- Hardware target: lock to one specific node; record fingerprint.
- Orchestration cadence: when the user reviews, when subagents run, what
  blocks on what.

Commit the spec. Don't start scaffolding until the user signs off.

### 3. Scaffold (the harness itself)

A small, well-tested Python package + a few prompt files. Keep it minimal:

```
evaluation/
├── README.md                   # how to reproduce
├── harness/
│   ├── lock.py                 # global flock — ABSOLUTE path, env-overridable
│   ├── stats.py                # median, IQR, classify(speedup) → verdict
│   ├── verify.py               # SHA-256 + FP-tolerance comparator
│   ├── dashboard.py            # render dashboard.md from candidates/*/report.md
│   ├── synthesize_reports.py   # build report.md from raw/*.json (when builders died)
│   ├── build_flags.json        # pinned compiler flags per language
│   ├── result_schema.json      # JSON Schema for raw timing records
│   ├── candidate_schema.md     # YAML schema for survey rows
│   ├── machine.md              # node fingerprint (hostname, lscpu, etc.)
│   ├── scout_prompt.md         # full scout subagent prompt
│   ├── builder_prompt.md       # full builder subagent prompt
│   └── run_all_measurements.sh # orchestrator-driven serial loop
├── references.bib              # BibTeX, scout-owned, builders append
├── survey.md                   # scout output
├── candidates/<id>/            # one dir per candidate
│   ├── kernel_baseline.<ext>
│   ├── kernel_lego.py          # (or whatever the "transformed" version is)
│   ├── run.sh
│   ├── raw/{baseline,lego}*.json
│   ├── verify.log
│   └── report.md
├── dashboard.md                # auto-generated status table
└── audit_report*.md            # cross-arch / re-measurement audit reports
```

**TDD every Python module.** lock.py has subtle race-condition pitfalls;
stats.py has off-by-one boundaries; verify.py has FP-tolerance edge cases.
Tests catch them.

### 4. Scout (research-only subagent)

One subagent. Reads papers and benchmark suites. Produces:

- `survey.md` — one entry per candidate (kernel × transform tuple), with
  a YAML block per the schema.
- `references.bib` — BibTeX with DOI/arXiv for every cited claim.
- `survey_summary.md` — class breakdown, drop list with reasons.

**Scout hard rules:**

1. **No invented numbers.** Predicted gains cite a paper or are marked
   `unknown`.
2. **No invented papers.** Every BibTeX key resolves to a real paper.
3. **Cite everything.** Claims point to BibTeX keys.
4. **No code, no benchmark runs.** Scout is research-only.

**Drop rules** (apply before any candidate is dispatched):

- Any required field empty → drop.
- License not in accepted list → drop. (For research evaluation, "no LICENSE
  file" is OK; copyleft is borderline; ask the user.)
- Predicted-baseline-class mismatches measured-baseline-class → drop or
  flag. (E.g., paper compared against scalar code; suite ships vectorized.)
- Required transform isn't expressible in the available primitives → drop.
- "Why the existing tool can't do this" is hand-wavy → drop or flag.

**Granularity:** one row per (kernel × transform) tuple, not per kernel.
Same kernel under different transforms is multiple rows.

### 5. Builders (one per candidate, parallel, serial-measure)

Each builder agent owns one candidate. Works in its own git worktree on its
own branch (`eval/<id>`). Edits only that candidate's directory. Other repo
state is read-only.

**Builder loop per candidate** (per layout attempt, max ~8 attempts):

1. **Vendor baseline source.** From upstream URL, untouched, license-checked.
2. **Build baseline.** Suite's documented command (or pinned flags from
   `build_flags.json`). Verify against suite reference output.
3. **Author the "transformed" version.** Apply the layout / optimization
   you're testing.
4. **Generate / compile transformed code.** Same flags as baseline.
5. **Verification gate.** Output must match baseline (byte-identical for
   integers, FP-tolerance for floats). On fail → `DROPPED-verification`,
   stop.
6. **Acquire the global lock.** `flock -x <ABSOLUTE_PATH>/evaluation/.lock`.
7. **Measure.** Standard protocol: 25 warmup + 100 timed iterations
   per (size, version), at ≥ 3 sizes. Pin with `taskset` / `numactl` where
   available; record what was actually applied.
8. **Release the lock.**
9. **Classify** with `stats.classify(baseline_median, version_median)`.
   Best class across sizes is the candidate verdict (WIN > PARITY > LOSS).
10. **Iterate** if not satisfied (try other variants); cap layout attempts.
11. **Write report.md.** YAML block with the schema, raw file references.
12. **Commit + push branch.** Do NOT open a PR.

**Builder hard rules:**

- No source-base modifications. `git diff main -- <protected paths>` must be
  empty.
- Identical compiler flags for baseline and version-under-test. Varying
  flags is the #1 way fake speedups creep in.
- Every reported number must reference a `raw/*.json` it was computed from.
  No invented numbers.
- Methodological discipline: avoid pow-2 sizes that hit baseline cache
  pathologies (use a non-pow-2 size sweep that spans the same compute volume).
- If you can't reach a clean WIN after the cap → ship the best honest result
  (LOSS / PARITY). Negative results are paper-strengthening when honest.

**Builder verification checklist:**

- [ ] No source modifications outside candidate dir.
- [ ] `raw/*.json` validates against `result_schema.json`.
- [ ] Verification gate passed; logged in `verify.log`.
- [ ] Speedup numbers in `report.md` reference raw files.
- [ ] Branch is pushed.

### 6. Audit (cross-architecture or post-hoc IQR)

After all builders finish, an audit pass re-measures every candidate at a
relaxed protocol (10 warmup + 30 timed) on either:

- The same node, just to spot-check IQR / contention pollution.
- A different architecture (different vendor / vector width / cache size)
  for portability validation.

**Audit goals:**

- Verdict-bucket agreement (WIN/PARITY/LOSS) across runs, NOT magnitude
  agreement.
- Flag any candidate where (a) audit verdict ≠ original verdict, or (b)
  speedup magnitude differs by > 20% relative.

**Audit output:** `audit_report.md` with:

- Verdict-agreement matrix.
- Disagreements analyzed (architecture-dependent? L3-capacity-sensitive?
  protocol mismatch?).
- Flagged candidates for full re-measurement.

### 7. Roadmap (the actionable output)

Every LOSS / PARITY / disagreement is a candidate roadmap entry. Cluster by
root cause. For each entry, document:

- **Severity** (high / medium / low).
- **Motivating evidence** — list the candidates this entry covers.
- **Root cause cluster** — what infrastructure gap caused it?
- **Proposed feature** — concrete design.
- **Implementation sketch** — 1 paragraph.
- **Effort** — days/weeks (be honest).
- **Dependencies** — what blocks it.
- **Re-test list when closed** — which candidates flip to WIN if this is
  fixed.

Sort by impact × tractability. Save to `roadmap.md`.

**Iteration model.** Don't write the paper section yet. Either:

1. Close roadmap entries (implement the feature, re-measure affected
   candidates), OR
2. Formally classify them as architectural-limit (paper Section 9
   "Limitations").

Then write the paper section against the post-iteration data.

## Honesty rules (apply at every layer)

These are the rules that separate paper-grade research from theater:

1. **No invented numbers.** Every speedup in a report references a raw
   timing record.
2. **No fabricated papers.** Every BibTeX entry resolves to a real DOI or
   arXiv ID.
3. **Verification before timing.** Output correctness check passes before
   any measurement gets reported.
4. **Methodological transparency.** Builders self-report when they used a
   reduced protocol (e.g. 10/30 instead of 25/100), what `repro_setup`
   options were unavailable, what biases their measurement may carry.
5. **Negative results published.** LOSSes go in the report alongside WINs.
   Architectural limits go in the paper's Limitations section.
6. **Match baseline class to citation class.** If your survey cites a paper
   that beat scalar LAPACK and the suite ships vectorized BLAS-3, the
   prediction class is wrong — flag it.

## Measurement methodology (the boring-but-critical bits)

### Serial measurement under a global mutex

On any shared compute node, concurrent measurements pollute each other via
shared L3 cache, memory bandwidth, and scheduler jitter. **Use a single global
file lock** to serialize the measurement-critical phase across all
parallel-dispatched workers.

```python
# evaluation/harness/lock.py — minimal correct version
import fcntl, os
from contextlib import contextmanager
from pathlib import Path

DEFAULT_LOCK_PATH = Path(os.environ.get(
    "EVAL_LOCK_PATH",
    "/scratch/.../evaluation/.lock",  # ABSOLUTE path
))

@contextmanager
def acquire(lock_path=None):
    path = Path(lock_path) if lock_path else DEFAULT_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = open(path, "a")  # 'a' so we don't truncate
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    finally:
        if fd is not None:
            fd.close()
```

**Critical pitfalls:**

- Lock path must be **absolute** so per-worktree builders converge on the same
  lock. A path like `Path(__file__).parent.parent / ".lock"` resolves
  per-worktree and is useless for cross-builder serialization.
- `open("a")` not `open("w")` — `"w"` truncates the file every acquire and
  introduces a small TOCTOU window.
- Guard `fd.close()` with `if fd is not None:` for the case where `open()`
  itself raises.
- Use `flock(2)` (BSD-style, per-FD), not `lockf(2)` (POSIX, per-process).

### Sample size and verdict thresholds

- **25 warmup + 100 timed** for canonical numbers. Median is robust to
  outliers; IQR captures variance.
- **10 warmup + 30 timed** for audit / spot-check passes. Statistically
  meaningful, 3× faster.
- **WIN threshold: speedup ≥ 1.02** (2%). Below 2% is in the noise floor on
  shared CPU nodes.
- **LOSS threshold: speedup ≤ 0.98**.
- **PARITY:** between 0.98 and 1.02. PARITY rows still get reported — "no
  slowdown despite portability" is a productivity argument.
- **IQR / median > 20%:** flag for re-measurement, not auto-disqualify.

### Hardware fingerprinting

Capture once and lock to it:

```bash
hostname; uname -a; lscpu; numactl --hardware; free -h
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
gcc --version | head -1
nvidia-smi -L 2>/dev/null || echo "no GPU"
```

Embed the SHA-256 of `machine.md` in every `raw/*.json` so a reviewer can
detect node drift.

When you migrate machines, **preserve the old fingerprint** as
`machine_<old_machine>.md` so the original raw JSONs' fingerprint references
still resolve.

### Cross-architecture validation

Run the audit on a different vendor / microarchitecture:

- AMD AVX2 (Zen 2/3/4) vs. Intel AVX-512 (Skylake-X / Ice Lake / Sapphire
  Rapids).
- Intel x86 vs. ARM Neoverse / Apple M-series.

You're testing whether the *verdict bucket* is robust across architectures,
not whether magnitudes match. Verdict-flips reveal real architecture
dependencies; magnitude-changes reveal cache-size / vector-width effects.

## Subagent dispatch tactics

- **One scout** (research-only).
- **One builder per candidate.** Dispatch in parallel; they queue at the
  global lock during measurement.
- **One audit orchestrator** (sequential by design).
- **Use the cheapest model that handles each role.** Most builder tasks are
  mechanical when the plan is well-specified; reserve more capable models for
  scout / audit which need judgment.
- **Provide the full task text in the prompt.** Don't ask subagents to
  "read the plan" — they'll lose context. Paste the relevant section.
- **Allow questions before work.** Each subagent prompt should explicitly
  invite clarification questions before starting.

## Failure modes you will hit

These are inevitable; design the harness to absorb them:

- **Rate-limited subagents.** Their work persists on disk (worktree + git
  branch). Re-dispatch with "your build state is intact" hint to resume.
- **Lock contention.** First-pass builders may use a per-worktree lock by
  mistake (path-resolution bug). Patch the lock path to absolute, re-run
  affected measurements.
- **Cache pathologies at pow-2 sizes.** Baseline kernels hit set-associativity
  conflicts at strides like 1024 doubles, inflating speedups dramatically.
  Audit at non-pow-2 sizes to separate "real layout win" from "artifact".
- **`-march=native` AMD binaries fail to run on Intel.** Cross-arch audits
  must rebuild from source, not just reuse binaries.
- **Multi-threaded baselines incorrectly compared at single-thread.**
  Block-cyclic and similar layouts only win with threading.
- **Hardware module missing.** `module load gcc/<version>` may need to run
  before any compile on cluster nodes.
- **Concurrent measurements look like wins.** A measurement under contention
  shows higher baseline (because the *baseline* version was contended by some
  other builder), making the layout look artificially good. Always serialize.

## Iteration cadence

The loop is:

1. **Round 1:** scaffold → scout → 30–50 builders → audit → roadmap.md.
2. **User picks a roadmap item to attack.** (E.g., R1 SIMD intrinsics.)
3. **Implement the feature.**
4. **Round 2:** re-run only the affected candidates (the roadmap entry's
   "re-test list when closed").
5. **Update roadmap.md** with new evidence; close or re-classify the entry.
6. **Repeat** until roadmap is exhausted.
7. **Then write the paper section.**

This is the difference between a one-shot evaluation and a research loop.
The one-shot gives you "here's some numbers". The loop gives you "here's
where the system genuinely wins, here's where it has architectural limits,
and here's what we built to close the gaps in between".

## Anti-patterns

Things to avoid that look reasonable but corrupt the methodology:

- **Spawning many measurement workers in parallel for "speed".** They
  contaminate each other's timings.
- **Using a relative lock path.** Per-worktree locks don't serialize across
  workers.
- **Measuring under whatever load the cluster happens to have.** Either
  exclusive-allocate the node or accept that timings are noisy and audit them.
- **Reporting the largest size's speedup without checking for L3 boundary
  pathologies.** Pow-2 sizes near cache capacity cause the *baseline* to
  thrash and inflate the reported "speedup".
- **Letting the scout invent BibTeX entries.** Always verify DOIs.
- **Writing the paper section before iterating on the roadmap.** Locks in
  weak numbers; gives reviewers easy LOSS rows to nitpick.
- **Running on the same machine forever.** Cross-arch audit is a free
  reproducibility / portability story for the paper.
- **Skipping verification gates.** A WIN that produced wrong output is a bug,
  not a result.

## File layout cheat-sheet

```
docs/
├── superpowers/
│   ├── specs/<date>-<topic>-design.md     # the spec
│   └── plans/<date>-<topic>-impl.md       # the implementation plan
└── research-harness-pattern.md            # this document

evaluation/                                 # owned by the harness
├── README.md
├── harness/                                # tested Python + prompts + schemas
├── references.bib
├── survey*.md                              # scout output
├── candidates/<id>/                        # builder output
├── dashboard.md                            # auto-generated
├── audit_report*.md                        # one per audit run
└── roadmap.md                              # actionable infra items

# Each builder works in its own worktree:
../<repo>-eval-<id>/                        # git worktree per candidate
```

## Checklist for adopting this pattern in a new project

- [ ] Brainstorm with user; produce signed-off design spec.
- [ ] Implement scaffold (lock, stats, verify, dashboard, schemas, prompts).
- [ ] TDD on each Python module.
- [ ] Verify lock path is absolute.
- [ ] Capture machine fingerprint to `machine.md`.
- [ ] Dispatch scout; review survey; drop bad candidates.
- [ ] Dispatch one builder per surviving candidate.
- [ ] Watch the dashboard; let serial measurement run.
- [ ] Audit pass when round complete.
- [ ] Cluster LOSSes/PARITY into `roadmap.md`.
- [ ] **Hold the paper section** until roadmap entries are closed or
      classified architectural-limit.
- [ ] Iterate the loop on each closed roadmap entry.
- [ ] Write the paper.

---

*This document distills the methodology used in the CASTLE CPU evaluation
round (Apr 2026). The pattern is research-stack-agnostic — it applies to any
benchmark-driven evaluation that wants honest, defensible, reproducible
numbers out of an LLM-orchestrated subagent loop.*
