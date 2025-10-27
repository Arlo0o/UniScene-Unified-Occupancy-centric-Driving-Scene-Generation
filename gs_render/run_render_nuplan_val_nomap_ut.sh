#!/bin/bash
GPU_NUM=1
apt install libgl1 -y

cd gs_render/diff-gaussian-rasterization/
pip install -e . -v
cd ../../

torchrun --nproc_per_node=$GPU_NUM gs_render/render_train_condition_nuplan_ut.py \
--pkl_path data/nuplan_pkls/trainval/nuplan_trainval_10hz_val.pkl \
--occ_path data/GT_occ_fast3_10hzval_r400/dense_voxels_with_semantic \
--dataset_path data/nuplan-all/sensor_blobs/trainval \
--version trainval \
--render_path data/nuplan-occ-render-trainval_val/ \
--processed_path data/nuplan-occ/nuplan-occ-render-trainval_val_ut \
--occ_ext .npz \
--start_idx $1 \
--end_idx $2 \
--with_ut
