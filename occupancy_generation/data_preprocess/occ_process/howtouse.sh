python3 occ_process.py \
--quantize_size  400 400 32  \
--data_base_path "/mnt/dataset/nuplan-occ/1-1-01/nuplan/GT_occ_fast/dense_voxels_with_semantic" \
--save_base_path "occ_quan" \
--config_path "config/nuplan.yaml" \
--method "max"

python3 occ_process.py \
--quantize_size  200 200 16  \
--data_base_path "/mnt/dataset/nuplan-occ/1-1-01/nuplan/GT_occ_fast/dense_voxels_with_semantic" \
--save_base_path "occ_quan" \
--config_path "config/nuplan.yaml" \
--method "max"


python3 occ_process_sample.py \
--quantize_size  200 200 16  \
--data_base_path "/mnt/dataset/nuplan-occ/1-1-01/nuplan/GT_occ_fast/dense_voxels_with_semantic" \
--save_base_path "occ_quan_sample" \
--config_path "config/nuplan.yaml" \
--method "max"

python3 occ_process_sample.py \
--quantize_size  400 400 32  \
--data_base_path "/mnt/dataset/nuplan-occ/1-1-01/nuplan/GT_occ_fast/dense_voxels_with_semantic" \
--save_base_path "occ_quan_sample" \
--config_path "config/nuplan.yaml" \
--method "max"

python3 occ_process_parallels.py --quantize_size  400 400 32  --data_base_path "/mnt/dataset/nuplan-occ/1-1-01/nuplan/GT_occ_fast/dense_voxels_with_semantic" --save_base_path "occ_quan" --config_path "config/nuplan.yaml" --method "max" --workers 16

# mini val
python3 occ_process_parallels.py --quantize_size  400 400 32  --data_base_path "/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/GT_occ_fast_val/dense_voxels_with_semantic" --save_base_path "occ_quan" --config_path "config/nuplan.yaml" --method "max" --workers 16

python3 occ_process_parallels.py --quantize_size  200 200 16  --data_base_path "/lpai/dataset/nuplan-occ/1-1-01/GT_occ_fast_val/dense_voxels_with_semantic" --save_base_path "occ_quan" --config_path "config/nuplan.yaml" --method "max" --workers 32


# trainval train
python3 occ_process_sample_trainval_400.py \
--quantize_size  200 200 16  \
--data_base_path "/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_val/GT_occ_fast3_10hzval_r400/dense_voxels_with_semantic" \
--save_base_path "occ_quan_sample" \
--config_path "config/nuplan.yaml" \
--method "max"

# trainval
# val
python3 occ_process_parallels_trainval_400.py --quantize_size  200 200 16  --data_base_path "/lpai/dataset/nuplan-occ/1-1-01/GT_occ_fast3_10hzval_r400/dense_voxels_with_semantic" --save_base_path "occ_quan_trainval/val" --config_path "config/nuplan.yaml" --method "max" --workers 64 --pkl_path /mnt/dataset/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/trainval/nuplan_trainval_10hz_val.pkl