#!/usr/bin/env bash
set -euo pipefail

CFG=${CFG:-nuplan_sample72_r400_intenw10_pluckeremb_histemb_smlosscos_lidarcond_flim_allloc_fixrange.yaml}
CKPT=${CKPT:-checkpoint/lidar_generation/checkpoint.pth}
WORK_DIR=${WORK_DIR:-outputs/lidar_generation}
WORKERS=${WORKERS:-4}

python tools/test.py \
  --cfg_file "$CFG" \
  --ckpt "$CKPT" \
  --work_dir "$WORK_DIR" \
  --workers "$WORKERS" \
  --save_to_file \
  "$@"
