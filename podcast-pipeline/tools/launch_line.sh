#!/usr/bin/env bash
# One command that starts the whole 6-process line in tmux (one window per process).
#
#   bash tools/launch_line.sh AUDIO_DIR OUT_DIR [extra run_variants_resident.py args for every process]
#
# Process 1 denoise point 1 -> 2 diarization -> 3 denoise point 2 -> 4 separation -> 5 denoise point 3 -> 6 ASR.
# Each one waits for the .ready_<stage> file of the one before it, see doc/chay_nhieu_variant_mot_process.md.
#
# Knobs (environment variables):
#   GPUS="0 0 1 1 0 1"   position (in the job's GPU list) of process 1..6; default balances the cards by rough memory use
#   SIDON_WORKERS=2      Sidon workers on the separation GPU (the profile says 3; fewer = less VRAM)
#   SESSION=line         tmux session name           CONDA_ENV=som   conda env to activate
#   VARIANTS=tools/variants_placement.json           ENVNAME=a100   profile
#   JOBID=18790          Slurm job whose GPUs the processes use (default: $SLURM_JOB_ID). Required on a Slurm machine: each window runs
#                        `srun --jobid=18790 --overlap --pty bash <script>`. Needed when this shell is outside the job
#                        (nvidia-smi says "No devices were found"); leave unset inside an salloc/srun shell.
#   DRY=1                write the six scripts and print them, start nothing
#   FORCE=1              reuse a tmux session name / an OUT dir written under other settings
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
AUDIO=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
EXTRA=("$@")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SESSION=${SESSION:-line}
CONDA_ENV=${CONDA_ENV:-som}
VARIANTS=${VARIANTS:-tools/variants_placement.json}
ENVNAME=${ENVNAME:-a100}
SIDON_WORKERS=${SIDON_WORKERS:-2}
DRY=${DRY:-0}
FORCE=${FORCE:-0}
# Inside a job shell Slurm already exports SLURM_JOB_ID; a tmux window does NOT inherit this shell's environment
# when the tmux server was started elsewhere, so every window is sent into the job with srun regardless.
JOBID=${JOBID:-${SLURM_JOB_ID:-}}
read -r -a GPU <<< "${GPUS:-0 0 1 1 0 1}"

NAMES=(p1_denoise1 p2_diarization p3_denoise2 p4_separation p5_denoise3 p6_asr)
STAGE_ARGS=(
  "--stop-after denoise --signal denoise"
  "--stop-after diarization --wait-signal denoise --signal diarization"
  "--stop-after denoise_pre_separation --wait-signal diarization --signal denoise_pre_separation"
  "--stop-after separation --wait-signal denoise_pre_separation --signal separation --set performance.stages.separation.workers_per_gpu=${SIDON_WORKERS}"
  "--stop-after denoise_pre_asr --wait-signal separation --signal denoise_pre_asr"
  "--stop-after asr --wait-signal denoise_pre_asr"
)
[ "${#GPU[@]}" -eq 6 ] || { echo "GPUS needs 6 numbers (one per process), got: ${GPUS:-}" >&2; exit 2; }

# The music switch decides which cache and results a run may touch: never mix them in one OUT dir.
MUSIC=off
for ((i = 0; i < ${#EXTRA[@]}; i++)); do
  [ "${EXTRA[i]}" = "--music" ] && MUSIC=${EXTRA[i + 1]:-off}
done
LAUNCH_DIR="$OUT/_resident/launch"
mkdir -p "$LAUNCH_DIR" "$OUT/_resident/logs"
if [ -f "$OUT/variants_manifest.json" ] && [ ! -f "$OUT/_resident/music_$MUSIC" ] && [ "$FORCE" != 1 ]; then
  echo "$OUT already holds results that were not made with --music $MUSIC." >&2
  echo "Use a new OUT dir (their diarization/separation caches would otherwise be reused), or FORCE=1." >&2
  exit 3
fi
touch "$OUT/_resident/music_$MUSIC"

if [ "$DRY" != 1 ] && tmux has-session -t "$SESSION" 2>/dev/null && [ "$FORCE" != 1 ]; then
  echo "tmux session '$SESSION' already exists: tmux attach -t $SESSION   (or SESSION=other, or FORCE=1)" >&2
  exit 4
fi

# Without a job the windows inherit this shell, and a shell outside the Slurm job sees no GPU: the denoise
# backends then fall back to the unfiltered audio and that result is cached. Refuse instead of starting.
if [ "$DRY" != 1 ] && [ -z "$JOBID" ] && [ "${NO_SRUN:-0}" != 1 ] && command -v srun >/dev/null 2>&1; then
  # A shell on a compute node can see every GPU of the machine, including other jobs' cards, so "sees a GPU"
  # proves nothing: without srun into the job the windows run on cards that are not yours.
  echo "Slurm is here: say which job to run in.   JOBID=<slurm job id> bash tools/launch_line.sh ...   (NO_SRUN=1 to skip)" >&2
  exit 6
fi
if [ "$DRY" != 1 ] && [ -z "$JOBID" ] && [ "${NO_GPU_CHECK:-0}" != 1 ]; then
  if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "This shell sees no GPU. Run it as:  JOBID=<slurm job id> bash tools/launch_line.sh ...   (NO_GPU_CHECK=1 to skip)" >&2
    exit 6
  fi
fi
case "$JOBID" in ''|*[!0-9]*) [ -z "$JOBID" ] || { echo "JOBID must be a number, got: $JOBID" >&2; exit 2; };; esac

# What a tmux window runs for process i.
window_cmd() {
  local script=$1
  if [ -n "$JOBID" ]; then
    printf 'srun --jobid=%s --overlap --pty bash %q; exec bash' "$JOBID" "$script"
  else
    printf 'bash %q; exec bash' "$script"
  fi
}

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")
COMMON=(--audio-dir "$AUDIO" --variants "$VARIANTS" --out-root "$OUT" --env "$ENVNAME" --layout default
        --skip-bootstrap --resume "${EXTRA[@]}")

for i in 0 1 2 3 4 5; do
  script="$LAUNCH_DIR/$((i + 1))_${NAMES[i]}.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "# written by tools/launch_line.sh; run it by hand to restart this one process"
    echo "exec > >(tee -a $(printf '%q' "$OUT/_resident/logs/$((i + 1))_${NAMES[i]}.log")) 2>&1"
    echo "[ -f $(printf '%q' "$CONDA_BASE/etc/profile.d/conda.sh") ] && source $(printf '%q' "$CONDA_BASE/etc/profile.d/conda.sh") && conda activate $(printf '%q' "$CONDA_ENV")"
    echo "cd $(printf '%q' "$HERE")"
    echo "source tools/denoise_env.sh"
    # GPUS holds positions in the job's own GPU list: inside srun Slurm sets CUDA_VISIBLE_DEVICES to the
    # devices of the job (maybe physical ids such as 4,5), and a hard-coded 0 or 1 would be somebody else's card.
    echo "_slot=${GPU[i]}"
    echo '_job=(${CUDA_VISIBLE_DEVICES//,/ })'
    echo 'if [ "${#_job[@]}" -gt 0 ]; then export CUDA_VISIBLE_DEVICES=${_job[$((_slot % ${#_job[@]}))]}; else export CUDA_VISIBLE_DEVICES=$_slot; fi'
    echo 'echo "[gpu] job devices: ${_job[*]:-none set} -> this process uses CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"'
    echo 'nvidia-smi -L 2>&1 | head -3 || true'
    printf 'python run_variants_resident.py'
    printf ' %q' "${COMMON[@]}"
    echo " ${STAGE_ARGS[i]}"
    echo 'echo "[exit $?]"'
  } > "$script"
  chmod +x "$script"
done

if [ "$DRY" = 1 ]; then
  for i in 0 1 2 3 4 5; do
    f="$LAUNCH_DIR/$((i + 1))_${NAMES[i]}.sh"
    echo "== window ${NAMES[i]}: $(window_cmd "$f")"
    sed -n '4,$p' "$f"
  done
  exit 0
fi

command -v tmux >/dev/null || { echo "tmux not found; run the scripts in $LAUNCH_DIR by hand" >&2; exit 5; }
[ "$FORCE" = 1 ] && tmux kill-session -t "$SESSION" 2>/dev/null || true
for i in 0 1 2 3 4 5; do
  script="$LAUNCH_DIR/$((i + 1))_${NAMES[i]}.sh"
  if [ "$i" -eq 0 ]; then
    tmux new-session -d -s "$SESSION" -n "${NAMES[i]}" "$(window_cmd "$script")"
  else
    tmux new-window -t "$SESSION" -n "${NAMES[i]}" "$(window_cmd "$script")"
  fi
done
echo "started 6 processes in tmux session '$SESSION' (GPUs: ${GPU[*]}${JOBID:+, Slurm job $JOBID})"
echo "  watch:   tmux attach -t $SESSION        (Ctrl+B then n / p to switch windows, Ctrl+B then D to leave)"
echo "  logs:    $OUT/_resident/logs/"
echo "  stop:    tmux kill-session -t $SESSION  (re-run this command later: --resume continues)"
