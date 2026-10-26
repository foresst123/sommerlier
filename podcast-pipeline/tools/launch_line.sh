#!/usr/bin/env bash
# One command that starts the whole 6-process line in tmux (one window per process).
#
#   bash tools/launch_line.sh AUDIO_DIR OUT_DIR [extra run_variants_resident.py args for every process]
#
# Process 1 denoise point 1 -> 2 diarization -> 3 denoise point 2 -> 4 separation -> 5 denoise point 3 -> 6 ASR.
# Each one waits for the .ready_<stage> file of the one before it, see doc/chay_nhieu_variant_mot_process.md.
#
# Knobs (environment variables):
#   GPUS="0 0 1 1 0 1"   physical GPU of process 1..6 (default balances the cards by rough memory use)
#   SIDON_WORKERS=2      Sidon workers on the separation GPU (the profile says 3; fewer = less VRAM)
#   SESSION=line         tmux session name           CONDA_ENV=som   conda env to activate
#   VARIANTS=tools/variants_placement.json           ENVNAME=a100   profile
#   DRY=1                write the six scripts and print them, start nothing
#   FORCE=1              reuse a tmux session name / an OUT dir written under other settings
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
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
    echo "export CUDA_VISIBLE_DEVICES=${GPU[i]}"
    printf 'python run_variants_resident.py'
    printf ' %q' "${COMMON[@]}"
    echo " ${STAGE_ARGS[i]}"
    echo 'echo "[exit $?]"'
  } > "$script"
  chmod +x "$script"
done

if [ "$DRY" = 1 ]; then
  for f in "$LAUNCH_DIR"/*.sh; do echo "== $f"; sed -n '4,$p' "$f"; done
  exit 0
fi

command -v tmux >/dev/null || { echo "tmux not found; run the scripts in $LAUNCH_DIR by hand" >&2; exit 5; }
[ "$FORCE" = 1 ] && tmux kill-session -t "$SESSION" 2>/dev/null || true
for i in 0 1 2 3 4 5; do
  script="$LAUNCH_DIR/$((i + 1))_${NAMES[i]}.sh"
  if [ "$i" -eq 0 ]; then
    tmux new-session -d -s "$SESSION" -n "${NAMES[i]}" "bash $(printf '%q' "$script"); exec bash"
  else
    tmux new-window -t "$SESSION" -n "${NAMES[i]}" "bash $(printf '%q' "$script"); exec bash"
  fi
done
echo "started 6 processes in tmux session '$SESSION' (GPUs: ${GPU[*]})"
echo "  watch:   tmux attach -t $SESSION        (Ctrl+B then n / p to switch windows, Ctrl+B then D to leave)"
echo "  logs:    $OUT/_resident/logs/"
echo "  stop:    tmux kill-session -t $SESSION  (re-run this command later: --resume continues)"
