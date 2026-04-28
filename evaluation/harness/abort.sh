#!/usr/bin/env bash
# abort.sh <candidate_id>
#
# Best-effort abort of a builder agent for the given candidate.
# Cleanly removes the worktree and the candidate's raw/ directory so
# restart.sh can re-dispatch from a clean slate. The orchestrator
# (Claude) is responsible for actually stopping the running subagent
# via TaskStop; this script just cleans the filesystem state.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <candidate_id>" >&2
  exit 1
fi

cid="$1"
repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
worktree="${repo_root%/*}/LEGO-eval-${cid}"

if git -C "${repo_root}" worktree list | grep -q "${worktree}"; then
  echo "removing worktree ${worktree}"
  git -C "${repo_root}" worktree remove --force "${worktree}" || true
fi

cand_dir="${repo_root}/evaluation/candidates/${cid}"
if [[ -d "${cand_dir}/raw" ]]; then
  echo "clearing raw/ for ${cid}"
  rm -rf "${cand_dir}/raw"
fi

echo "abort complete for ${cid}"
