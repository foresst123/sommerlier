#!/usr/bin/env bash
# JOB 1 "front": the GPUs of one Slurm job run denoise, diarization, separation and all.
# Each GPU stage has one copy per GPU of the job (copy i sees only GPU i); the copies share the files with --claim.
# separation_plan is CPU only and builds the windows between diarization and separation.
#
#   AUDIO=/data/vi_podcasts_20 OUT=/data/run1 GPUS="0 1" bash tools/job_front.sh
#
# Chain:  denoise -> diarization -> separation_plan -> separation -> all
#         all waits conversation_selection (tools/job_llm.sh); tools/job_asr.sh waits separation.
#
# Knobs:  GPUS="0 1"          PHYSICAL GPU ids to use (no srun, no JOBID), one copy per GPU; JOBID=<n> optional: then positions in that Slurm job
#         REPLICAS_PER_GPU=1  copies per GPU per stage (raise only when VRAM is spare)
#         PLAN_REPLICAS=1     copies of separation_plan (CPU); more than 1 uses --claim
#         SIDON_WORKERS=2     Sidon workers per copy (performance.stages.separation.workers_per_gpu)
#         diarization          workers=1 per copy (each copy is on its own card)
#         FINAL=$OUT/final    where the all processes write the final dataset: <FINAL>/lhotse/manifest.jsonl.gz (one
#                             Lhotse manifest of the whole run) and the conversations of each recording in <FINAL>/
#         ONLY='^all_'        restart only the processes whose name matches (the others keep running)
#         VARIANTS, ENV_NAME=a100, CONDA_ENV=som, CACHE, DRY=1, FORCE=1 (stop the earlier processes first)
#         AUDIO_SEPARATOR_MODEL_DIR: rf_crowd checkpoints; found by tools/denoise_env.sh when unset
#         RECLAIM_AFTER, WAIT_TIMEOUT (minutes, 0 = off): see tools/job_common.sh
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HERE/tools/job_common.sh"
SIDON_WORKERS=${SIDON_WORKERS:-2}
PLAN_REPLICAS=${PLAN_REPLICAS:-1}
job_init front
# rf_crowd needs its checkpoint folder: without AUDIO_SEPARATOR_MODEL_DIR every file falls back and is marked failed
if [ -z "${AUDIO_SEPARATOR_MODEL_DIR:-}" ]; then
  set +u; pushd "$HERE" >/dev/null; source tools/denoise_env.sh || true; popd >/dev/null; set -u
fi
if [ -z "${AUDIO_SEPARATOR_MODEL_DIR:-}" ] && [ -z "${DRY:-}" ]; then
  echo "AUDIO_SEPARATOR_MODEL_DIR is not set and tools/denoise_env.sh found no rf_crowd checkpoint; export it, or run: source tools/denoise_env.sh --download" >&2
  exit 2
fi
FINAL=$(realpath -m "${FINAL:-$OUT/final}")
case "$FINAL" in *\"*|*\\*) echo "FINAL must not contain a double quote or a backslash, got: $FINAL" >&2; exit 2 ;; esac
# all writes the final root: --set outputs.clean_two_channel_dir (JSON string, shell-quoted) comes after the
# run's own "" (run_variants_resident.py build_argv), so it wins. Only all gets it: a comparison run keeps "".
FINAL_SET=$(printf '%q' "outputs.clean_two_channel_dir=\"$FINAL\"")
job_copies denoise "--stop-after denoise --signal denoise" denoise
job_copies diarization "--stop-after diarization --wait-signal denoise --signal diarization --set performance.stages.diarization.workers=1"
job_plan separation_plan "--stop-after separation_plan --wait-signal diarization --signal separation_plan"
job_copies separation "--stop-after separation --wait-signal separation_plan --signal separation --set performance.stages.separation.workers_per_gpu=$SIDON_WORKERS"
job_copies all "--stop-after all --wait-signal conversation_selection --set $FINAL_SET" denoise
job_start
