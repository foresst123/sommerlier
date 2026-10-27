#!/usr/bin/env bash
# Stop the background processes a job script started:  bash tools/job_stop.sh front|asr|llm <OUT>
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HERE/tools/job_common.sh"
[ $# -eq 2 ] || { echo "usage: bash tools/job_stop.sh front|asr|llm <OUT dir>" >&2; exit 2; }
file="$2/_resident/logs/$1.pids"
[ -f "$file" ] || { echo "no pids file: $file" >&2; exit 1; }
job_stop_pids "$file"
