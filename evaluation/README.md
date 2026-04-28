# CASTLE CPU Evaluation

Research harness for the CASTLE/TACO paper's Section 7.5 CPU evaluation.
Detailed design lives in
`docs/superpowers/specs/2026-04-28-castle-cpu-evaluation-harness-design.md`.

## What is here

- `harness/` — prompts, schemas, and Python utilities used by the scout
  and builder subagents.
- `references.bib` — BibTeX, owned by the scout, appended to by builders.
- `survey.md` — scout output (one entry per candidate). Created by the
  scout subagent; not present until Step 2 of the orchestration.
- `survey_summary.md` — scout drop-list summary.
- `dashboard.md` — auto-regenerated status table during builder runs.
- `candidates/` — one directory per candidate, owned by one builder.

## Workflow at a glance

1. Orchestrator scaffolds `harness/` (this state).
2. Scout subagent surveys benchmarks, writes `survey.md` and
   `references.bib`.
3. One builder subagent per surviving survey row, in its own git
   worktree on branch `eval/cpu-<id>`, runs build → verify → measure
   → classify under the global mutex `evaluation/.lock`.
4. Orchestrator distills surviving builder reports into the paper's
   Section 7.5 prose.

## How to reproduce one round

```bash
PYTHONPATH=. pytest evaluation/harness/tests/ -v

# Then the scout subagent (orchestrator dispatches with
# evaluation/harness/scout_prompt.md as context).

# Then one builder subagent per survivor row, with
# evaluation/harness/builder_prompt.md plus the row's YAML.

python -m evaluation.harness.dashboard       # regenerate status

# Distill survivor reports into CASTLE-tex Section 7.5.
```

## Conventions

- LEGO layouts in user-facing code prefer `OrderBy` + `TileBy`.
  `GroupBy` is allowed when needed but each occurrence must carry a
  one-sentence justification in the candidate's `report.md`.
- Compiler flags are pinned per-language in
  `harness/build_flags.json` and applied identically to baseline and
  LEGO versions.
- Verdict thresholds: speedup ≥ 1.02× → WIN, ≤ 0.98× → LOSS, else
  PARITY. Tunable via `harness/stats.py:EFFECT_THRESHOLD`.
- Single global mutex via `flock(evaluation/.lock)` serializes all
  build + measurement cycles to keep timings honest on the shared node.
