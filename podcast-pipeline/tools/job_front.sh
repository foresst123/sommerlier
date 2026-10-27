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
#         VARIANTS, ENV_NAME=a100, CONDA_ENV=som, CACHE, SESSION, DRY=1, NO_SRUN=1, FORCE=1
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HERE/tools/job_common.sh"
SIDON_WORKERS=${SIDON_WORKERS:-2}
PLAN_REPLICAS=${PLAN_REPLICAS:-1}
job_init front
job_copies denoise "--stop-after denoise --signal denoise" denoise
job_copies diarization "--stop-after diarization --wait-signal denoise --signal diarization --set performance.stages.diarization.workers=1"
job_plan separation_plan "--stop-after separation_plan --wait-signal diarization --signal separation_plan"
job_copies separation "--stop-after separation --wait-signal separation_plan --signal separation --set performance.stages.separation.workers_per_gpu=$SIDON_WORKERS"
job_copies all "--stop-after all --wait-signal conversation_selection" denoise
job_start
