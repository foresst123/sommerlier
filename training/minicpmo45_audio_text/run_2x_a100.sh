#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/a100_2x.yaml}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  accelerate launch --num_processes 2 \
  -m minicpmo_train.train \
  --config "${CONFIG_PATH}"
