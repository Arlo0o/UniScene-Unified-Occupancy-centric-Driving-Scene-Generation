import sys
# sys.path.append('/lpai/volumes/lmm-data-proc/liuhongsi/code/UniScene-V2/gs_render/diff-gaussian-rasterization')
import copy
import os
import shutil
import time
import random
import gc
from filelock import FileLock
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cv2
import numpy as np
import numba as nb
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import yaml
from pyquaternion import Quaternion

import torch
from torch.utils.data import Dataset, DataLoader
import pickle



occ_colors_map = np.array(
    [   
        [255, 158, 0, 255],  #  1 vehicle  orange
        [255, 99, 71, 255],  #  2 [place_holder]  Tomato
        [255, 140, 0, 255],  #  3 [place_holder]  Darkorange
        [255, 69, 0, 255],  #  4 [place_holder]  Orangered
        [233, 150, 70, 255],  #  5 czone_sign  Darksalmon
        [220, 20, 60, 255],  #  6 bicycle  Crimson
        [255, 61, 99, 255],  #  7 generic_object  Red
        [0, 0, 230, 255],  #  8 pedestrian  Blue
        [47, 79, 79, 255],  #  9 traffic_cone  Darkslategrey
        [112, 128, 144, 255],  #  10 barrier  Slategrey
        [0, 207, 191, 255],  # 11  background_surface  nuTonomy green  
        [255,   0, 255, 255],  #  12 drive surface  
        [75, 0, 75, 255],  #  13  no drive surface
        [0, 175,   0, 255],  # 14 road bound line  
        [255,   0,   0, 255], # 15 road line Burlywood 
        [0, 175, 0, 255],  # 16 None  Green
        [0, 0, 0, 255],  # unknown
    ]
).astype(np.uint8)


def replace_occ_grid_with_bev_nuplan(input_occ, bevlayout, driva_area_idx=1, bev_replace_idx=[1,2,15, 17],
                                     occ_replace_new_idx=[12, 13, 14, 15]):
    # self.classes= ['drivable_area','ped_crossing','walkway','stop_line','carpark_area','road_divider','lane_divider','road_block']
    # nuplan
    # self.classes= ['intersections','generic_drivable_areas','walkways','carpark_areas','crosswalks','lane_group_connectors','lane_groups_polygons','road_segments']
    # 需要把 walkways 换成另外一种颜色,10
    # 需要把 drivable_area 换成另外一种颜色,11
    # lane_divider 换成另外一种颜色,12
    # stop_line del
    # road_block 13
    # occ road [11] drivable area

    # default ped_crossing->18; stop_line->19 (del); roal_divider->20; lane_divider->21
    # default shape: input_occ: [200,200,16]; bevlayout: [18,200,200]

    roal_divider_mask = bevlayout[15, :, :].astype(np.uint8)
    lane_divider_mask = bevlayout[17, :, :].astype(np.uint8)

    roal_divider_mask = cv2.dilate(
        roal_divider_mask, np.ones((2, 2), np.uint8))
    lane_divider_mask = cv2.dilate(
        lane_divider_mask, np.ones((2, 2), np.uint8))

    bevlayout[15, :, :] = roal_divider_mask.astype(bool)
    bevlayout[17, :, :] = lane_divider_mask.astype(bool)

    n = len(bev_replace_idx)
    x_max, y_max = input_occ.shape[0], input_occ.shape[1]
    output_occ = input_occ.copy()  # numpy copy() ; tensor clone()
    bev_replace_mask = []
    for i in range(n):
        bev_replace_mask.append(bevlayout[bev_replace_idx[i]] == 1)

    for x in range(x_max):
        for y in range(y_max):
            for i in range(n):
                if bev_replace_mask[i][x, y]:
                    occupancy_data = input_occ[x, y, :]

                    if driva_area_idx in occupancy_data:
                        max_11_index = np.where(
                            occupancy_data == driva_area_idx)
                        output_occ[x, y, max_11_index] = occ_replace_new_idx[i]
    return output_occ


def load_occ_layout(layout_path):
    # load layout data
    layout = np.load(open(layout_path, 'rb'), encoding='bytes', allow_pickle=True)
    #layout = layout['bev_map']
    layout = layout['arr_0']
    return layout



class OccDataset_w_bev(Dataset):
    def __init__(self, pkl_path, bev_dir, occ_dir):
        super().__init__()
        self.pkl_path = pkl_path
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
        self.infos = pkl_data["infos"]
        self.scene_tokens=pkl_data["scene_tokens"]
        self.bev_dir = bev_dir
        self.occ_dir = occ_dir

    def __len__(self):
        return len(self.scene_tokens)
    
    @staticmethod
    def collect_fn(batch):
        assert len(batch) == 1
        return batch[0]

    def __getitem__(self, idx):
        scene_list = self.scene_tokens[idx]

        bev_datas = []
        occ_datas = []
        token_list = []
        for token in scene_list:
            bev_data = np.load(f"{self.bev_dir}/{token}.npz")['gt_bev_masks']
            bev_datas.append(bev_data)
            occ_data = np.load(f"{self.occ_dir}/{token}.npy")
            occ_datas.append(occ_data)

            token_list.append(token)
        
        return dict(
            bev_data=bev_datas,
            occ_data=occ_datas,
            token_list=token_list
        )



if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default="data/nuplan-all/sensor_blobs/trainval")
    parser.add_argument('--pkl_path', type=str, default='/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_val.pkl')
    parser.add_argument('--version', type=str, default='trainval')
    parser.add_argument('--occ_dir', type=str, default="/mnt/datasets/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_200_200_16")
    parser.add_argument('--processed_path', type=str, default=None)
    parser.add_argument('--bev_dir', type=str, default="/mnt/datasets/nuplan-bev/0-1-0/nuplan_bev_new/mini/val")
    parser.add_argument('--render_path', type=str, default="data/nuplan-occ-render-trainval_val/")
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_interval', type=int, default=200)
    parser.add_argument('--gs_scale', type=float, default=0.01)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=500)
    parser.add_argument('--save_depth_as_uint16', action='store_true', default=False)
    
    args = parser.parse_args()

    pkl_path = args.pkl_path
    bev_dir = args.bev_dir
    occ_dir = args.occ_dir
    bev_layer_to_merge = [0,1,3,4,5,6,7]


    process_dataset = OccDataset_w_bev(args.pkl_path, args.bev_dir, args.occ_dir)
    process_dataloader = DataLoader(process_dataset, batch_size=1, num_workers=0)
    for iter_i, data in tqdm(enumerate(process_dataloader),total=len(process_dataloader)):
        for i in range(len(data['bev_data'])):
            replace_occ_token = data['token_list'][i][0]
            bev_data = data["bev_data"][i][0].numpy()
            bev_data_comb=bev_data[bev_layer_to_merge, :,:]
            bev_data_comb=np.any(bev_data_comb, axis=0).astype(int)
            bev_data[1,:,:]=bev_data_comb

            print(f"=> {replace_occ_token} orig data elemtns: {np.unique(data['occ_data'][i])}")

            replaced_occ = replace_occ_grid_with_bev_nuplan(data["occ_data"][i][0].numpy(), bev_data)
            print(f"=> {replace_occ_token} replaced occ elments: {np.unique(replaced_occ)}")
        print(f"=> {iter_i}")

