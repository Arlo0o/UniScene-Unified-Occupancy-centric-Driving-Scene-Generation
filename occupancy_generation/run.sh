#!/usr/bin/env bash
set -euo pipefail

CKPT=${CKPT:-checkpoint/occ_generation/dit.pt}
VAE_CKPT=${VAE_CKPT:-checkpoint/occ_generation/3dvae.pth}
VAE_CONFIG=${VAE_CONFIG:-config/train_3dvae_nuplan_400_full.py}
RESULT_DIR=${RESULT_DIR:-outputs/occ_generation}
IMAGESET=${IMAGESET:-data/nuplan_mini_val_clip_infos_dit.pkl}
OCC_ROOT=${OCC_ROOT:-data/occ_quan/nuplan_quantized_400_400_32}
BEV_ROOT=${BEV_ROOT:-data/nuplan_bev_400/mini}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29502}

torchrun --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" \
  tools/eval_OccDiT_nuplan_demo_hr_mini.py \
  --ckpt "$CKPT" \
  --vae_ckpt "$VAE_CKPT" \
  --vae_config "$VAE_CONFIG" \
  --result_dir "$RESULT_DIR" \
  --imageset "$IMAGESET" \
  --occ-root "$OCC_ROOT" \
  --bev-root "$BEV_ROOT" \
  --lambda_noise_prior "${LAMBDA_NOISE_PRIOR:-0.3}" \
  --cfg-scale "${CFG_SCALE:-7}" \
  --num-sampling-steps "${NUM_SAMPLING_STEPS:-200}" \
  --save_occ \
  --inversion \
  "$@"
