#!/usr/bin/env bash
# Stop EVERY pipeline process of your user on THIS machine (runners, model workers, vLLM engines), whatever job
# started them and even when the pids files are gone, then wait until they are really gone.
#
#   bash tools/job_kill_all.sh        # lists them and asks
#   YES=1 bash tools/job_kill_all.sh  # no question
#
# Only this machine: run it on each machine that runs a job. It never touches other users' processes.
set -uo pipefail
PATTERN='run_variants_resident\.py|_worker\.py|VLLM::EngineCore|multiprocessing\.resource_tracker'
list() {
    ps -u "$(id -un)" -o pid=,ppid=,etime=,cmd= | awk -v me="$$" -v pat="$PATTERN" \
        '$1 != me && $2 != me && $0 ~ pat && $0 !~ /job_kill_all/ {print}'
}
procs=$(list)
if [ -z "$procs" ]; then echo "no pipeline processes of $(id -un) on $(hostname)"; exit 0; fi
echo "$procs" | cut -c1-150
if [ "${YES:-}" != 1 ]; then
    read -r -p "stop these $(echo "$procs" | wc -l) processes on $(hostname)? [y/N] " a
    [ "$a" = y ] || exit 1
fi
for round in TERM TERM KILL; do
    for pid in $(list | awk '{print $1}'); do kill -"$round" "$pid" 2>/dev/null; done
    for _ in 1 2 3 4 5 6; do
        sleep 2
        [ -z "$(list)" ] && break 2
    done
done
left=$(list)
if [ -n "$left" ]; then echo "still running:"; echo "$left" | cut -c1-150; exit 1; fi
echo "all gone. GPU memory:"
nvidia-smi --query-gpu=index,memory.used --format=csv 2>/dev/null || true
