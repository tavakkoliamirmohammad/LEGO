#!/usr/bin/env bash
# Step 3.5: orchestrator-driven serial measurement.
#
# For every candidate worktree under /scratch/general/vast/u1419116/LEGO-eval-*
# that has measure.py or run.sh and does NOT yet have a committed report.md,
# acquire the global lock and run the candidate's measurement script.
#
# Reports are NOT synthesized here — the run.sh / measure.py does that itself.
# A separate synthesize step picks up any candidates whose measure.py wrote
# raw/*.json but did not commit a report.md.
#
# Run from anywhere; logs to evaluation/measurement_orchestrator.log.

set -uo pipefail

LEGO_ROOT=/scratch/general/vast/u1419116/LEGO
GLOBAL_LOCK="$LEGO_ROOT/evaluation/.lock"
LOG="$LEGO_ROOT/evaluation/measurement_orchestrator.log"

# shellcheck disable=SC1091
source "$LEGO_ROOT/venv/bin/activate"

mkdir -p "$LEGO_ROOT/evaluation"
{
  echo "=== orchestrator starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
} >> "$LOG"

cd "$LEGO_ROOT"

for d in /scratch/general/vast/u1419116/LEGO-eval-*; do
  cid=$(basename "$d" | sed 's|^LEGO-eval-||')
  cand_dir="$d/evaluation/candidates/$cid"
  branch="eval/cpu-$cid"

  if [[ ! -d "$cand_dir" ]]; then
    echo "$(date -u +%H:%M:%S) skip $cid: candidate dir missing" >> "$LOG"
    continue
  fi

  # Skip if branch already has a committed report.md on the canonical branch
  if git -C "$LEGO_ROOT" show "$branch:evaluation/candidates/$cid/report.md" >/dev/null 2>&1; then
    echo "$(date -u +%H:%M:%S) skip $cid: report already committed on $branch" >> "$LOG"
    continue
  fi

  # Skip if measure.py / run.sh is missing
  if [[ ! -f "$cand_dir/measure.py" && ! -f "$cand_dir/run.sh" ]]; then
    echo "$(date -u +%H:%M:%S) skip $cid: no measure.py / run.sh" >> "$LOG"
    continue
  fi

  echo "$(date -u +%H:%M:%S) measuring $cid" >> "$LOG"
  start=$(date +%s)

  if [[ -f "$cand_dir/measure.py" ]]; then
    flock -x "$GLOBAL_LOCK" bash -c "cd '$cand_dir' && python3 measure.py" \
        >> "$LOG" 2>&1
    rc=$?
  else
    flock -x "$GLOBAL_LOCK" bash -c "cd '$cand_dir' && bash run.sh" \
        >> "$LOG" 2>&1
    rc=$?
  fi

  elapsed=$(( $(date +%s) - start ))
  echo "$(date -u +%H:%M:%S) $cid exit=$rc elapsed=${elapsed}s" >> "$LOG"
done

echo "=== orchestrator finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
