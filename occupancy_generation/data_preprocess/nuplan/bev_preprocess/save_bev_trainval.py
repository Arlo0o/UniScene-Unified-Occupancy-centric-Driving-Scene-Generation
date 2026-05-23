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



val_dataset = nuplan_bev_dataset(
    data_path = "/mnt/datasets/nuplan-all/2-0-0/dataset",
    return_len = 1,
    offset = 0,
    imageset = "/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/trainval/nuplan_trainval_10hz_val.pkl",
    nusc = None,
    occ_dataroot = None,
    quantize_size=(400,400,32),
)


val_dataloader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=12)

save_dir = "/data/zhuhu/3DVision_datasets/Occ/nuplan/trainval_bev"
os.makedirs(save_dir, exist_ok=True)

# for iter_i, batch in enumerate(train_dataloader):


for iter_i, (token, bevmap) in enumerate(tqdm(val_dataloader, desc="Saving val data")):
    for i in range(len(token)):
        print(f"save token: {token[i]}=={iter_i}, bev shape {bevmap['gt_masks_bev'][i].shape}")
        np.savez_compressed(f"{save_dir}/{token[i]}.npz", gt_bev_masks=bevmap['gt_masks_bev'][i].numpy().astype(np.int8), gt_aux_bev=bevmap['gt_aux_bev'][i].numpy().astype(np.int32))


# train_dataset = nuplan_bev_dataset(
#     data_path = "/mnt/datasets/nuplan-all/2-0-0/dataset",
#     return_len = 1,
#     offset = 0,
#     imageset = "/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/trainval/nuplan_trainval_10hz_train_chunk_4_.pkl",
#     nusc = None,
#     occ_dataroot = None,
# )

# train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=False, num_workers=24)

# for token, bevmap in tqdm(train_dataloader, desc="Saving train data"):
#     for i in range(len(token)):
#         print(f"save token: {token[i]}")
#         np.savez_compressed(f"{train_save_dir}/{token[i]}.npz", gt_bev_masks=bevmap['gt_masks_bev'][i].numpy().astype(np.int8), gt_aux_bev=bevmap['gt_aux_bev'][i].numpy().astype(np.int32))


# train_dataset = nuplan_bev_dataset(
#     data_path = "/mnt/datasets/nuplan-all/2-0-0/dataset",
#     return_len = 1,
#     offset = 0,
#     imageset = "/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/trainval/nuplan_trainval_10hz_train_chunk_5_.pkl",
#     nusc = None,
#     occ_dataroot = None,
# )

# train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=False, num_workers=24)

# for token, bevmap in tqdm(train_dataloader, desc="Saving train data"):
#     for i in range(len(token)):
#         print(f"save token: {token[i]}")
#         np.savez_compressed(f"{train_save_dir}/{token[i]}.npz", gt_bev_masks=bevmap['gt_masks_bev'][i].numpy().astype(np.int8), gt_aux_bev=bevmap['gt_aux_bev'][i].numpy().astype(np.int32))

