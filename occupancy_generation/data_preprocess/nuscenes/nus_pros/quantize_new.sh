#!/bin/bash
start_s_p=0
end_s_p=210000
step=10000

for (( s_p=start_s_p; s_p<=end_s_p; s_p+=step )); do
    e_p=$((s_p + step))
    python /gpfs/public-shared/fileset-groups/crosshair/guojiazhe/code/occworld-dev_12hz/nus_pros/nuscene_process.py \
        --quantize_size  400 400 32  \
        --occ_base_path "s3://sdagent-shard-bj-baiducloud/crosshairs/zouyingshuang-share/occ/8-30-nksr/dense_voxels_with_semantic/" \
        --save_base_path "/gpfs/public-shared/fileset-groups/crosshair/guojiazhe/occ_12hz" \
        --config_path "/gpfs/public-shared/fileset-groups/crosshair/guojiazhe/code/occworld-dev_12hz/nus_pros/nuscene.yaml" \
        --method "max" \
        --s_e $s_p $e_p &
done


# # quantize occ
# python3 occ_preprocess/nusc_process.py \
# --quantize_size  400 400 16  \
# --data_base_path "/path/to/occ_orig" \
# --save_base_path "/path/to/occ_quant" \
# --config_path "config/nuplan.yaml" \
# --method "max"
