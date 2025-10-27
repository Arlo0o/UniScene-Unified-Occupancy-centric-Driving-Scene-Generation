#!/bin/bash 

conda activate nksr
torchrun --nproc_per_node 8 gs_render/data_process/generate_occ_nuplan.py \
--save_path data/nuplan/GT_occ_nuplan_mini_r400 \
--config_path gs_render/data_process/config_nuplan_r400.yaml \
--start_idx $1 \
--end_idx $2 \
--dataroot data/nuplan_all/sensor_blobs/mini \
--pkl_path data/nuplan_pkls/mini/nuplan_mini_10hz_train.pkl \
--num_workers 4