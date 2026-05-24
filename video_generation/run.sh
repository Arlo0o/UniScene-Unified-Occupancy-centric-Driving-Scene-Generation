#!/usr/bin/env bash
set -euo pipefail

# Inference entry for UniScenev2 video generation.
# Example:
#   GPU_NUM=4 CKPT=checkpoint/video_generation bash run.sh

CONFIG=${CONFIG:-config/inference.py}
CKPT=${CKPT:-checkpoint/video_generation}
OUTPUT=${OUTPUT:-outputs/video_generation}
GPU_NUM=${GPU_NUM:-1}
PORT=${PORT:-29507}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

torchrun \
  --master_port "${PORT}" \
  --nproc_per_node "${GPU_NUM}" \
  tools/inference.py "${CONFIG}" \
  --sample-every 1 \
  --load "${CKPT}" \
  --output "${OUTPUT}"
