#!/usr/bin/env bash
# JOB 3 "llm": refinement, speaker_relabel and conversation_selection with Qwen3.5-9B (vLLM) on both GPUs of the job.
# The refinement placement of the profile ("pipelined", split 0.5) puts the model over gpu_1 and gpu_2.
# It waits for the asr signal of tools/job_asr.sh and writes the conversation_selection signal for tools/job_front.sh.
#
#   AUDIO=/data/vi_podcasts_20 OUT=/data/run1 GPUS="4 5" bash tools/job_llm.sh
#
# Knobs:  GPUS="4 5"  physical GPU ids (JOBID optional: positions in that job) (gpu_1 = first, gpu_2 = second)
#         VARIANTS, ENV_NAME=a100, CONDA_ENV=som, CACHE, SESSION, DRY=1, NO_SRUN=1, FORCE=1
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HERE/tools/job_common.sh"
job_init llm
job_proc conversation_selection "$GPUS" "--stop-after conversation_selection --wait-signal asr --signal conversation_selection"
job_start
