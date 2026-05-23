#!/usr/bin/env bash
set -euo pipefail

VAE_CONFIG=${VAE_CONFIG:-config/train_3dvae_nuplan_400_mini.py}
VAE_CKPT=${VAE_CKPT:-checkpoint/occ_generation/3dvae.pth}
RESULTS_DIR=${RESULTS_DIR:-outputs/train_occdit_debug}
BATCH_SIZE=${BATCH_SIZE:-1}
LOG_EVERY=${LOG_EVERY:-100}
CKPT_EVERY=${CKPT_EVERY:-100}
IMAGESET=${IMAGESET:-data/nuplan_mini_train_clip_infos_dit.pkl}
OCC_ROOT=${OCC_ROOT:-data/occ_quan/nuplan_quantized_400_400_32}
BEV_ROOT=${BEV_ROOT:-data/nuplan_bev_400/mini}

if [ ! -f "$VAE_CONFIG" ]; then
    echo "Missing VAE config: $VAE_CONFIG"
    exit 1
fi

if [ ! -f "$VAE_CKPT" ]; then
    echo "Missing VAE checkpoint: $VAE_CKPT"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=26342 tools/train_OccDiT_nuplan_400_mini.py \
    --vae_config "$VAE_CONFIG" \
    --vae_ckpt "$VAE_CKPT" \
    --results-dir "$RESULTS_DIR" \
    --imageset "$IMAGESET" \
    --occ-root "$OCC_ROOT" \
    --bev-root "$BEV_ROOT" \
    --dit-batch-size "$BATCH_SIZE" \
    --log-every "$LOG_EVERY" \
    --ckpt-every "$CKPT_EVERY" \
    --epochs 10000 \
    --global-seed 42 \
    --lambda_noise_prior 0.15 \
    --debug
