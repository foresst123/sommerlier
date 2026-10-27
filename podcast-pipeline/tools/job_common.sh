#!/usr/bin/env bash
# Shared by tools/job_front.sh, tools/job_asr.sh and tools/job_llm.sh. Source it; it runs nothing by itself.
#
#   job_init NAME                       front | asr | llm: checks AUDIO, OUT, GPUS; sets the common run_variants args
#   job_copies STAGE "ARGS" [denoise]   one copy of STAGE per GPU position in GPU_POS (x REPLICAS_PER_GPU), --claim when >1
#   job_plan STAGE "ARGS"               PLAN_REPLICAS copies of a CPU-only stage (CUDA_VISIBLE_DEVICES empty)
#   job_proc NAME GPUS "ARGS" [denoise] one process script; GPUS is "none" or GPU positions ("0", "0 1")
#   job_start                           prints the process -> GPU -> stage table; DRY=1 stops there, else starts them in the background with nohup
#   job_set_gpus POS...                 used by the process scripts: CUDA_VISIBLE_DEVICES of the job at those positions
#
# Without JOBID (the usual way) GPUS are the PHYSICAL GPU ids you pick: GPUS="2 3" runs on cards 2 and 3, no srun.
# With JOBID=<n> the script runs inside that Slurm job (srun --jobid) and GPUS are positions in the job's own card
# list: Slurm sets CUDA_VISIBLE_DEVICES to the job's cards (maybe "4,5"), so position 1 is the second one.
#
# Knobs (environment): AUDIO, OUT (required), GPUS (physical ids), JOBID (optional, see above), CACHE, ENV_NAME=a100, CONDA_ENV=som,
#   VARIANTS=tools/variants_production.json, DRY=1 (write scripts and print the table only),
#   FORCE=1 (stop this job's earlier background processes first). NO_SRUN is accepted and ignored: srun is only used when JOBID is given.
#   RECLAIM_AFTER=0     minutes; a --claim older than this with no ready/failed marker is taken over (0 = never). Passed as
#                       --reclaim-after to every --claim process, only when not 0.
#   WAIT_TIMEOUT=0      minutes; a process waiting for a --wait-signal gives up after this (0 = waits forever). Passed as
#                       --wait-timeout to every process that waits, only when not 0.

job_gpu_list() {
    # prints the CUDA_VISIBLE_DEVICES value of the job's cards at the given positions
    local -a job=() out=()
    local p
    if [ "${JOB_GPU_MODE:-}" = physical ]; then
        local IFS=,
        echo "$*"
        return 0
    fi
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=, read -r -a job <<< "$CUDA_VISIBLE_DEVICES"
    fi
    for p in "$@"; do
        if [ "${#job[@]}" -gt 0 ]; then
            if [ "$p" -ge "${#job[@]}" ]; then
                echo "GPU position $p is outside the job's devices (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >&2
                return 1
            fi
            out+=("${job[$p]}")
        else
            out+=("$p")
        fi
    done
    local IFS=,
    echo "${out[*]}"
}

job_set_gpus() {
    local v
    v=$(job_gpu_list "$@") || return 1
    export CUDA_VISIBLE_DEVICES="$v"
    echo "[gpu] CUDA_VISIBLE_DEVICES=$v"
}

job_init() {
    JOB_NAME=$1
    if [ -z "${AUDIO:-}" ] || [ -z "${OUT:-}" ]; then
        echo "usage: AUDIO=<audio dir> OUT=<out dir> [GPUS=\"0 1\"] [JOBID=<slurm job id>] bash tools/job_${JOB_NAME}.sh" >&2
        exit 2
    fi
    AUDIO=$(realpath -m "$AUDIO")
    OUT=$(realpath -m "$OUT")
    CACHE=$(realpath -m "${CACHE:-$OUT/_resident/cache}")
    GPUS=${GPUS:-0 1}
    read -r -a GPU_POS <<< "$GPUS"
    if [ "${#GPU_POS[@]}" -ne 2 ]; then
        echo "GPUS needs two GPU ids (e.g. GPUS=\"0 1\"), got: $GPUS" >&2
        exit 2
    fi
    local p
    for p in "${GPU_POS[@]}"; do
        case "$p" in ''|*[!0-9]*) echo "GPUS must be numbers, got: $GPUS" >&2; exit 2 ;; esac
    done
    REPLICAS_PER_GPU=${REPLICAS_PER_GPU:-1}
    PLAN_REPLICAS=${PLAN_REPLICAS:-1}
    for p in "$REPLICAS_PER_GPU" "$PLAN_REPLICAS"; do
        case "$p" in ''|*[!0-9]*|0) echo "REPLICAS_PER_GPU and PLAN_REPLICAS must be whole numbers >= 1" >&2; exit 2 ;; esac
    done
    RECLAIM_AFTER=${RECLAIM_AFTER:-0}
    WAIT_TIMEOUT=${WAIT_TIMEOUT:-0}
    for p in "$RECLAIM_AFTER" "$WAIT_TIMEOUT"; do
        case "$p" in ''|*[!0-9.]*|*.*.*|.) echo "RECLAIM_AFTER and WAIT_TIMEOUT must be minutes (a number, 0 = off), got: $p" >&2; exit 2 ;; esac
    done
    ENV_NAME=${ENV_NAME:-a100}
    CONDA_ENV=${CONDA_ENV:-som}
    VARIANTS=${VARIANTS:-tools/variants_production.json}
    DRY=${DRY:-0}
    FORCE=${FORCE:-0}
    JOBID=${JOBID:-}      # only an explicit JOBID turns srun on; an inherited SLURM_JOB_ID does not
    case "$JOBID" in ''|*[!0-9]*) [ -z "$JOBID" ] || { echo "JOBID must be a number, got: $JOBID" >&2; exit 2; } ;; esac
    LAUNCH_DIR="$OUT/_resident/launch_$JOB_NAME"
    LOG_DIR="$OUT/_resident/logs"
    mkdir -p "$LAUNCH_DIR" "$LOG_DIR"
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")
    COMMON=(--audio-dir "$AUDIO" --variants "$VARIANTS" --out-root "$OUT" --env "$ENV_NAME"
            --layout default --skip-bootstrap --resume --per-file --cache-dir "$CACHE")
    # The stage folders keep their JSON (the line reads denoise.json, selection.json) but no WAVs: STAGE_AUDIO=1 brings them back.
    [ "${STAGE_AUDIO:-0}" = 1 ] || COMMON+=(--no-stage-audio)
    PROC_NAMES=()
    PROC_GPU=()
    PROC_STAGE=()
    PROC_FILES=()
}

job_nonzero() {
    awk -v v="$1" 'BEGIN { exit !(v + 0 > 0) }'
}

job_proc() {
    # NAME GPUS "ARGS" [denoise]: writes <launch>/<n>_<NAME>.sh and records it for the table
    local name=$1 gpus=$2 args=$3 denoise=${4:-0}
    local index=$(( ${#PROC_NAMES[@]} + 1 ))
    # --reclaim-after / --wait-timeout only when the knob is set (the default command line stays the same)
    case " $args " in *" --claim "*) if job_nonzero "$RECLAIM_AFTER"; then args="$args --reclaim-after $RECLAIM_AFTER"; fi ;; esac
    case " $args " in *" --wait-signal "*) if job_nonzero "$WAIT_TIMEOUT"; then args="$args --wait-timeout $WAIT_TIMEOUT"; fi ;; esac
    local script="$LAUNCH_DIR/${index}_${name}.sh"
    local log="$LOG_DIR/${JOB_NAME}_${index}_${name}.log"
    local gpu_line shown gpu_args=""
    if [ "$gpus" = none ]; then
        gpu_line='export CUDA_VISIBLE_DEVICES=""'
        shown="none (CPU)"
    else
        local -a pos
        read -r -a pos <<< "$gpus"
        local second=1
        if [ "${#pos[@]}" -eq 1 ]; then second=0; fi
        gpu_line="job_set_gpus ${pos[*]} || exit 7"
        gpu_args="--gpu-1 0 --gpu-2 $second"
        shown="GPU ${pos[*]}"
    fi
    {
        echo '#!/usr/bin/env bash'
        echo "# written by tools/job_${JOB_NAME}.sh (process ${name}); run it by hand to restart this one process"
        echo "exec > >(tee -a $(printf '%q' "$log")) 2>&1"
        echo "[ -f $(printf '%q' "$CONDA_BASE/etc/profile.d/conda.sh") ] && source $(printf '%q' "$CONDA_BASE/etc/profile.d/conda.sh") && conda activate $(printf '%q' "$CONDA_ENV")"
        echo "cd $(printf '%q' "$HERE")"
        echo "source tools/job_common.sh"
        if [ "$denoise" = denoise ] || [ "$denoise" = 1 ]; then
            echo "source tools/denoise_env.sh"
        fi
        if [ -z "$JOBID" ]; then echo 'export JOB_GPU_MODE=physical'; fi
        echo "$gpu_line"
        printf 'echo "[gpu] %s: %s, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"\n' "$name" "$shown"
        if [ "$gpus" != none ]; then
            echo 'nvidia-smi -L 2>&1 | head -3 || true'
        fi
        printf 'python run_variants_resident.py'
        printf ' %q' "${COMMON[@]}"
        if [ -n "$gpu_args" ]; then
            printf ' %s' "$gpu_args"
        fi
        printf ' %s\n' "$args"
        echo 'echo "[exit $?]"'
    } > "$script"
    chmod +x "$script"
    PROC_NAMES+=("$name")
    PROC_GPU+=("$shown")
    PROC_STAGE+=("$args")
    PROC_FILES+=("$script")
}

job_copies() {
    # one copy per GPU position, REPLICAS_PER_GPU copies on each card; every copy claims its own files
    local stage=$1 args=$2 denoise=${3:-0}
    local total=$(( ${#GPU_POS[@]} * REPLICAS_PER_GPU ))
    local claim=""
    if [ "$total" -gt 1 ]; then claim=" --claim"; fi
    local gpu copy name
    for gpu in "${GPU_POS[@]}"; do
        for ((copy = 1; copy <= REPLICAS_PER_GPU; copy++)); do
            name="${stage}_gpu${gpu}"
            if [ "$REPLICAS_PER_GPU" -gt 1 ]; then name="${name}_r${copy}"; fi
            job_proc "$name" "$gpu" "$args$claim" "$denoise"
        done
    done
}

job_plan() {
    # CPU-only stage: PLAN_REPLICAS copies, no CUDA device
    local stage=$1 args=$2
    local claim=""
    if [ "$PLAN_REPLICAS" -gt 1 ]; then claim=" --claim"; fi
    local copy
    for ((copy = 1; copy <= PLAN_REPLICAS; copy++)); do
        job_proc "${stage}_cpu${copy}" none "$args$claim" 0
    done
}

job_run_bg() {
    # starts one process script in the background, in its own process group, detached from this terminal
    local script=$1
    if [ -n "$JOBID" ]; then
        setsid nohup srun --jobid="$JOBID" --overlap bash "$script" >/dev/null 2>&1 &
    else
        setsid nohup bash "$script" >/dev/null 2>&1 &
    fi
    echo $!
}

job_stop_pids() {
    # stops every process group recorded in a pids file; leaves the file in place
    local file=$1 pid name
    [ -f "$file" ] || return 0
    while read -r pid name; do
        [ -n "$pid" ] || continue
        kill -TERM -- "-$pid" 2>/dev/null && echo "stopped $name (process group $pid)"
    done < "$file"
}

job_sweep_orphans() {
    # the model workers run in their own process group, so stopping the job leaves them behind holding VRAM:
    # stop this user's worker processes whose parent is gone (ppid 1)
    local pid n=0
    for pid in $(ps -u "$(id -un)" -o pid=,ppid=,cmd= | awk '$2==1 && /_worker\.py|VLLM|EngineCore/ {print $1}'); do
        kill -TERM "$pid" 2>/dev/null && n=$((n + 1))
    done
    if [ "$n" -gt 0 ]; then
        echo "stopped $n orphaned worker process(es) that still held GPU memory"
    fi
}

job_start() {
    echo "== job ${JOB_NAME}: JOBID=${JOBID:-none}  GPUS=\"$GPUS\" ($( [ -n "$JOBID" ] && echo "positions in job $JOBID" || echo "physical GPU ids"))  REPLICAS_PER_GPU=$REPLICAS_PER_GPU  OUT=$OUT"
    printf '  %-34s %-12s %s\n' process GPU stage
    local i
    for i in "${!PROC_NAMES[@]}"; do
        printf '  %-34s %-12s %s\n' "${PROC_NAMES[i]}" "${PROC_GPU[i]}" "${PROC_STAGE[i]}"
    done
    if [ "$DRY" = 1 ]; then
        echo "DRY=1: process scripts are in $LAUNCH_DIR; nothing started"
        return 0
    fi
    local pids="$LOG_DIR/${JOB_NAME}.pids" pid name alive=0
    if [ -n "${ONLY:-}" ]; then
        # restart only the processes whose name matches the regex ONLY (e.g. ONLY='^all_'); the others keep running
        local kept="" started=0
        if [ -f "$pids" ]; then
            while read -r pid name; do
                [ -n "$pid" ] || continue
                if [[ $name =~ $ONLY ]]; then
                    kill -TERM -- "-$pid" 2>/dev/null && echo "stopped $name (process group $pid)"
                else
                    kept+="$pid $name"$'\n'
                fi
            done < "$pids"
            sleep 2
        fi
        printf '%s' "$kept" > "$pids"
        for i in "${!PROC_NAMES[@]}"; do
            if [[ ${PROC_NAMES[i]} =~ $ONLY ]]; then
                pid=$(job_run_bg "${PROC_FILES[i]}")
                echo "$pid ${PROC_NAMES[i]}" >> "$pids"
                started=$((started + 1))
            fi
        done
        echo "restarted $started process(es) matching ONLY=$ONLY; the others were left running"
        return 0
    fi
    if [ -f "$pids" ]; then
        while read -r pid name; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then alive=$((alive + 1)); fi
        done < "$pids"
        if [ "$alive" -gt 0 ]; then
            if [ "$FORCE" = 1 ]; then
                job_stop_pids "$pids"
                sleep 2
            else
                echo "$alive process(es) of job ${JOB_NAME} are still running (see $pids). Stop them first (bash tools/job_stop.sh $JOB_NAME $OUT) or FORCE=1" >&2
                exit 4
            fi
        fi
    fi
    : > "$pids"
    for i in "${!PROC_NAMES[@]}"; do
        pid=$(job_run_bg "${PROC_FILES[i]}")
        echo "$pid ${PROC_NAMES[i]}" >> "$pids"
    done
    echo "started ${#PROC_NAMES[@]} background processes (nohup, no tmux); pids in $pids"
    echo "  logs:  $LOG_DIR/${JOB_NAME}_*.log   (tail -f one of them)"
    echo "  state: python tools/line_status.py $OUT"
    echo "  stop:  bash tools/job_stop.sh $JOB_NAME $OUT   (start again later: --resume continues)"
}
