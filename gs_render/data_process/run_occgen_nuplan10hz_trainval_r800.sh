#!/bin/bash 

conda activate nksr
torchrun --nproc_per_node 8 gs_render/data_process/generate_occ_nuplan.py \
--save_path data/nuplan/GT_occ_fast4_trainval_r800 \
--config_path gs_render/data_process/config_nuplan_r800.yaml \
--start_idx $1 \
--end_idx $2 \
--dataroot data/nuplan_all/sensor_blobs/trainval \
--pkl_path data/nuplan_pkls/trainval/nuplan_trainval_10hz_train_chunk_$3_.pkl \
--num_workers 4