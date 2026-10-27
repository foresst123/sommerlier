#!/usr/bin/env bash
# JOB 2 "asr": one process uses both GPUs of the job (Qwen3 on the first, Whisper and PhoWhisper on the second).
# It waits for the separation signal of tools/job_front.sh and writes the asr signal for tools/job_llm.sh.
#
#   AUDIO=/data/vi_podcasts_20 OUT=/data/run1 GPUS="2 3" bash tools/job_asr.sh
#
# Knobs:  GPUS="2 3"  physical GPU ids (JOBID optional: positions in that job); the first is gpu_1 (Qwen3), the second gpu_2 (Whisper)
#         VARIANTS, ENV_NAME=a100, CONDA_ENV=som, CACHE, SESSION, DRY=1, NO_SRUN=1, FORCE=1
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HERE/tools/job_common.sh"
job_init asr
job_proc asr "$GPUS" "--stop-after asr --wait-signal separation --signal asr"
job_start
