#!/usr/bin/env bash
set -euo pipefail

# Build symlinks for NuPlan camera shards, then launch multi-GPU training.
# Set NUPLAN_CAMERA_ROOTS as a colon-separated list of directories whose
# immediate children are NuPlan camera scene folders.

: "${NUPLAN_CAMERA_ROOTS:?Set NUPLAN_CAMERA_ROOTS before running this script.}"

SENSOR_LINK_DIR=${SENSOR_LINK_DIR:-dataset1/nuplan/sensor_blobs_train}
OCC_RENDER_ROOT=${OCC_RENDER_ROOT:-}
CONFIG=${CONFIG:-config/train.py}
GPU_NUM=${GPU_NUM:-8}
PORT=${PORT:-27518}

mkdir -p "${SENSOR_LINK_DIR}"

IFS=':' read -r -a SRC_DIRS <<< "${NUPLAN_CAMERA_ROOTS}"
for src_dir in "${SRC_DIRS[@]}"; do
  find "${src_dir}" -mindepth 1 -maxdepth 1 -type d | while read -r subdir; do
    dir_name=$(basename "${subdir}")
    ln -sfn "${subdir}" "${SENSOR_LINK_DIR}/${dir_name}"
  done
done

if [[ -n "${OCC_RENDER_ROOT}" ]]; then
  mkdir -p dataset1
  ln -sfn "${OCC_RENDER_ROOT}" dataset1/nuplan-occ-render-mini
fi

torchrun \
  --master_port "${PORT}" \
  --nproc_per_node "${GPU_NUM}" \
  tools/train.py "${CONFIG}"
