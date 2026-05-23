import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import numpy as np
from tqdm import tqdm
from nuplan_dataset import nuplan_bev_dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm



# val_dataset = nuplan_bev_dataset(
#     data_path = "/data/longhun/3D/nuplan/maps",
#     return_len = 1,
#     offset = 0,
#     imageset = "/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_val.pkl",
#     nusc = None,
#     occ_dataroot = None,
#     quantize_size=(400,400,32),
#     # quantize_size=(200,200,16),
# )


# val_dataloader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=12)

val_save_dir = "/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_400"
# os.makedirs(val_save_dir, exist_ok=True)

# for token, bevmap in tqdm(val_dataloader, desc="Saving val data"):
#     for i in range(len(token)):
#         print(f"save token: {token[i]}, shape is {bevmap['gt_masks_bev'][i].shape}")
#         if os.path.exists(f"{val_save_dir}/{token[i]}.npz"):
#             print(f"File {val_save_dir}/{token[i]}.npz already exists, skipping.")
#             continue
#         # Save the ground truth BEV masks and auxiliary BEV data
#         np.savez_compressed(f"{val_save_dir}/{token[i]}.npz", gt_bev_masks=bevmap['gt_masks_bev'][i].numpy().astype(np.int8), gt_aux_bev=bevmap['gt_aux_bev'][i].numpy().astype(np.int32))


train_dataset = nuplan_bev_dataset(
    data_path = "/mnt/datasets/nuplan-all/2-0-0/dataset",
    return_len = 1,
    offset = 0,
    imageset = "/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_train.pkl",
    nusc = None,
    occ_dataroot = None,
    quantize_size=(400,400,32),
)

# token不同, 所以可以放到一起. 
train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=False, num_workers=12)

for token, bevmap in tqdm(train_dataloader, desc="Saving train data"):
    for i in range(len(token)):
        print(f"save token: {token[i]}")
        np.savez_compressed(f"{val_save_dir}/{token[i]}.npz", gt_bev_masks=bevmap['gt_masks_bev'][i].numpy().astype(np.int8), gt_aux_bev=bevmap['gt_aux_bev'][i].numpy().astype(np.int32))

