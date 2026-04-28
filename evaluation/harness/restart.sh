#!/usr/bin/env bash
# restart.sh <candidate_id>
#
# Calls abort.sh, then prints a one-liner the orchestrator can use to
# re-dispatch the builder. Does NOT itself dispatch a subagent (that's
# the orchestrator's job).

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <candidate_id>" >&2
  exit 1
fi

cid="$1"
here="$(cd "$(dirname "$0")" && pwd)"

bash "${here}/abort.sh" "${cid}"

cat <<MSG
Re-dispatch instructions:
  - Read evaluation/candidates/${cid}/ for the candidate's YAML block.
  - Dispatch a fresh builder subagent with the prompt from
    evaluation/harness/builder_prompt.md, parameterized on
    candidate_id=${cid}.
MSG
