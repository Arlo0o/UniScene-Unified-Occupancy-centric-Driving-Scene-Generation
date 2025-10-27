#!/bin/bash
GPU_NUM=1
conda activate nksr
apt install libgl1 -y

torchrun --nproc_per_node=$GPU_NUM gs_render/render_train_condition_nuplan_ut.py \
--pkl_path data/nuplan_pkls/mini/nuplan_mini_10hz_val.pkl \
--occ_path data/occ_quan/nuplan_quantized_400_400_32 \
--dataset_path data/nuplan_all/sensor_blobs/mini \
--version mini \
--render_path data/nuplan-occ-render-mini_val_ut/ \
--config_file gs_render/data_process/config_nuplan_r400.yaml \
--occ_ext .npy \
--start_idx $1 \
--end_idx $2 \
--with_ut