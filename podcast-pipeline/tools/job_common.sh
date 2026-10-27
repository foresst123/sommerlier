#!/usr/bin/env bash
# Shared by tools/job_front.sh, tools/job_asr.sh and tools/job_llm.sh. Source it; it runs nothing by itself.
#
#   job_init NAME                       front | asr | llm: checks AUDIO, OUT, GPUS; sets the common run_variants args
#   job_copies STAGE "ARGS" [denoise]   one copy of STAGE per GPU position in GPU_POS (x REPLICAS_PER_GPU), --claim when >1
#   job_plan STAGE "ARGS"               PLAN_REPLICAS copies of a CPU-only stage (CUDA_VISIBLE_DEVICES empty)
#   job_proc NAME GPUS "ARGS" [denoise] one process script; GPUS is "none" or GPU positions ("0", "0 1")
#   job_start                           prints the process -> GPU -> stage table; DRY=1 stops there, else starts tmux
#   job_set_gpus POS...                 used by the process scripts: CUDA_VISIBLE_DEVICES of the job at those positions
#
# Positions index the job's own GPU list (GPUS="0 1" by default). Inside srun, Slurm sets CUDA_VISIBLE_DEVICES
# to the job's cards (maybe "4,5"), so position 1 is the job's second card whatever its number. Outside a job
# there is no CUDA_VISIBLE_DEVICES and the positions are the physical ids.
#
# Knobs (environment): AUDIO, OUT (required), JOBID, GPUS, CACHE, ENV_NAME=a100, CONDA_ENV=som,
#   VARIANTS=tools/variants_production.json, SESSION, DRY=1 (write scripts and print the table only),
#   NO_SRUN=1 (run without srun, e.g. on a machine with no Slurm), FORCE=1 (reuse a tmux session name).

job_gpu_list() {
    # prints the CUDA_VISIBLE_DEVICES value of the job's cards at the given positions
    local -a job=() out=()
    local p
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
        echo "usage: AUDIO=<audio dir> OUT=<out dir> [JOBID=<slurm job id>] [GPUS=\"0 1\"] bash tools/job_${JOB_NAME}.sh" >&2
        exit 2
    fi
    AUDIO=$(realpath -m "$AUDIO")
    OUT=$(realpath -m "$OUT")
    CACHE=$(realpath -m "${CACHE:-$OUT/_resident/cache}")
    GPUS=${GPUS:-0 1}
    read -r -a GPU_POS <<< "$GPUS"
    if [ "${#GPU_POS[@]}" -ne 2 ]; then
        echo "GPUS needs two positions in the job (e.g. GPUS=\"0 1\"), got: $GPUS" >&2
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
    ENV_NAME=${ENV_NAME:-a100}
    CONDA_ENV=${CONDA_ENV:-som}
    VARIANTS=${VARIANTS:-tools/variants_production.json}
    SESSION=${SESSION:-job_$JOB_NAME}
    DRY=${DRY:-0}
    FORCE=${FORCE:-0}
    JOBID=${JOBID:-${SLURM_JOB_ID:-}}
    case "$JOBID" in ''|*[!0-9]*) [ -z "$JOBID" ] || { echo "JOBID must be a number, got: $JOBID" >&2; exit 2; } ;; esac
    if [ "$DRY" != 1 ] && [ -z "$JOBID" ] && [ "${NO_SRUN:-0}" != 1 ] && command -v srun >/dev/null 2>&1; then
        # a shell outside the job may see other jobs' cards: name the job instead of guessing
        echo "Slurm is here: say which job to run in.   JOBID=<slurm job id> bash tools/job_${JOB_NAME}.sh   (NO_SRUN=1 to skip)" >&2
        exit 6
    fi
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

job_proc() {
    # NAME GPUS "ARGS" [denoise]: writes <launch>/<n>_<NAME>.sh and records it for the table
    local name=$1 gpus=$2 args=$3 denoise=${4:-0}
    local index=$(( ${#PROC_NAMES[@]} + 1 ))
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

job_window_cmd() {
    if [ -n "$JOBID" ]; then
        printf 'srun --jobid=%s --overlap --pty bash %q; exec bash' "$JOBID" "$1"
    else
        printf 'bash %q; exec bash' "$1"
    fi
}

job_start() {
    echo "== job ${JOB_NAME}: JOBID=${JOBID:-none}  GPUS=\"$GPUS\" (positions in the job)  REPLICAS_PER_GPU=$REPLICAS_PER_GPU  OUT=$OUT"
    printf '  %-34s %-12s %s\n' process GPU stage
    local i
    for i in "${!PROC_NAMES[@]}"; do
        printf '  %-34s %-12s %s\n' "${PROC_NAMES[i]}" "${PROC_GPU[i]}" "${PROC_STAGE[i]}"
    done
    if [ "$DRY" = 1 ]; then
        echo "DRY=1: process scripts are in $LAUNCH_DIR; nothing started"
        return 0
    fi
    command -v tmux >/dev/null || { echo "tmux not found; run the scripts in $LAUNCH_DIR by hand" >&2; exit 5; }
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        if [ "$FORCE" = 1 ]; then
            tmux kill-session -t "$SESSION"
        else
            echo "tmux session '$SESSION' already exists: tmux attach -t $SESSION   (or SESSION=other, or FORCE=1)" >&2
            exit 4
        fi
    fi
    local cmd
    for i in "${!PROC_NAMES[@]}"; do
        cmd=$(job_window_cmd "${PROC_FILES[i]}")
        if [ "$i" -eq 0 ]; then
            tmux new-session -d -s "$SESSION" -n "${PROC_NAMES[i]}" "$cmd"
        else
            tmux new-window -t "$SESSION" -n "${PROC_NAMES[i]}" "$cmd"
        fi
    done
    echo "started ${#PROC_NAMES[@]} processes in tmux session '$SESSION'"
    echo "  watch: tmux attach -t $SESSION   (Ctrl+B then n / p switches windows, Ctrl+B then D leaves)"
    echo "  logs:  $LOG_DIR/"
    echo "  stop:  tmux kill-session -t $SESSION   (start the job script again later: --resume continues)"
}
