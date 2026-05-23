#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
START_IDX=${1:-0}
END_IDX=${2:-500}
CHUNK=${3:-0}
GPU_NUM=${GPU_NUM:-4}

cd "${SCRIPT_DIR}/diff-gaussian-rasterization"
pip install -e . -v

cd "${SCRIPT_DIR}/.."
torchrun --nproc_per_node="${GPU_NUM}" "${SCRIPT_DIR}/render_train_condition_nuplan_fast.py" \
  --pkl_path "${PKL_PATH:-data/nuplan_pkls/trainval/nuplan_trainval_10hz_train_chunk_${CHUNK}_.pkl}" \
  --occ_path "${OCC_PATH:-data/nuplan-occ/GT_occ_fast3_10hztrain_r400/dense_voxels_with_semantic}" \
  --layout_path "${LAYOUT_PATH:-data/nuplan_bev}" \
  --dataset_path "${DATASET_PATH:-data/nuplan-all/sensor_blobs/trainval}" \
  --version "${VERSION:-trainval}" \
  --render_path "${RENDER_PATH:-data/nuplan-occ-render-trainval_train_chunk_${CHUNK}/}" \
  --start_idx "${START_IDX}" \
  --end_idx "${END_IDX}"
