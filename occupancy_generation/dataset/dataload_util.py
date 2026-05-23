from datetime import time
import pstats
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
from torchvision.transforms.functional import rotate
import random
import torch.nn.functional as F
import pickle
from pyquaternion import Quaternion
import glob
import numba
import cv2
import uuid

@numba.njit
def one_hot_encode(data: np.ndarray):
    data = data.transpose(1, 2, 0)
    assert data.ndim == 3
    n = data.shape[2]

    assert n <= 30  # ensure int32 does not overflow
    # assert data.dtype == np.uint8
    for x in np.unique(data):
        assert x in [0, 1]

    # shift = np.arange(n, np.int32)[None, None]
    shift = np.zeros((1, 1, n), np.int32)
    shift[0, 0, :] = np.arange(0, n, 1, np.int32)

    binary = (data > 0)  # bool
    # after shift, numpy keeps int32, numba change dtype to int64
    binary = (binary << shift).sum(-1)  # move bit to left and combine to one
    binary = binary.astype(np.int32)

    return binary

@numba.njit
def one_hot_decode(data: np.ndarray, n: int):
    """
    returns (h, w, n) np.int64 {0, 1}
    """
    # shift = np.arange(n, dtype=np.int32)[None, None]
    shift = np.zeros((1, 1, n), np.int32)
    shift[0, 0, :] = np.arange(0, n, 1, np.int32)

    # x = np.array(data)[..., None]
    x = np.zeros((*data.shape, 1), data.dtype)
    x[..., 0] = data
    # after shift, numpy keeps int32, numba changes dtype to int64
    x = (x >> shift) & 1  # only keep the lowest bit, for each n

    x = x.transpose(2, 0, 1)
    return x


def replace_occ_grid_with_bev_nuplan(input_occ, bevlayout, driva_area_idx=1, bev_replace_idx=[1,15, 17],
                                     occ_replace_new_idx=[12, 14, 15]):
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

    # roal_divider_mask = cv2.dilate(
    #     roal_divider_mask, np.ones((3, 3), np.uint8))
    # lane_divider_mask = cv2.dilate(
    #     lane_divider_mask, np.ones((3, 3), np.uint8))

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

def pBEV(input): # 18,200,200 -> 1,50,50
    w_bev=[1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19]
    w_bev=torch.tensor(w_bev).reshape(-1,1).float()
    down_input = F.adaptive_avg_pool2d(input, output_size=50).clone().detach()
    # print(down_input.shape)
    # down_input = down_input.permute(0,2, 3, 1)
    down_input = down_input.permute(1,2,0)
    down_input = down_input.reshape(-1,18)
    down_input = torch.mm(down_input,w_bev)
    down_input = down_input.reshape(-1,50,50)
    # rotate 90
    down_input = torch.rot90(down_input, k=1, dims=(1, 2))
    # flip for the same xy oridinate like Zmid
    down_input = torch.flip(down_input, dims=[1])

    return down_input

def nBEV1(data_b,ch_use): # 18,200,200 -> 1,200,200
    data_b = data_b[ch_use]
    mask = data_b>0.01
    cumulative_mask = np.cumsum(mask, axis=0)
    max_index_map = np.argmax(cumulative_mask, axis=0)
    max_index_map = max_index_map / (len(ch_use) -1)
    all_zero_mask = np.all(mask == 0, axis=0)
    max_index_map[all_zero_mask] = -1
    data_b= np.array([max_index_map])
    return data_b

def Int2Bit_BEV(data_b,ch_use): # 18,200,200 -> 4,200,200
    data_b = data_b[ch_use]
    mask = data_b>0.01
    cumulative_mask = np.cumsum(mask, axis=0)
    max_index_map = np.argmax(cumulative_mask, axis=0)+1
    # max_index_map = max_index_map / (len(ch_use) -1)
    all_zero_mask = np.all(mask == 0, axis=0)
    max_index_map[all_zero_mask] = 0

    bev_4c= np.zeros((4,200,200))
    for i in range(4):
        bev_4c[i] = (max_index_map >> i) & 1
    # data_b= np.array([max_index_map])
    return bev_4c

def rot_flip(input): # (C,H,W)
    # flip for the same xy oridinate like Zmid
    input = torch.rot90(input, k=1, dims=(1, 2))
    input = torch.flip(input, dims=[1])
    return input

class CustomDataset_pBEV(Dataset):
    def __init__(self, folder_a, folder_b):
        self.files_a = sorted([os.path.join(folder_a, f) for f in os.listdir(folder_a) if f.endswith('.npz')])
        self.files_b = sorted([os.path.join(folder_b, f) for f in os.listdir(folder_b) if f.endswith('.npz')])
          
        assert len(self.files_a) == len(self.files_b), "Number of files in folder_a and folder_b should be the same."
    
    def __getitem__(self, index):
        data_a = np.load(self.files_a[index])
        data_b = np.load(self.files_b[index])

        data_b = torch.tensor(data_b['arr_0'],dtype=torch.float)
        data_b = pBEV(data_b)
        return torch.tensor(data_a['arr_0']), data_b
    
    def __len__(self):
        return len(self.files_a)

class CustomDataset_nBEV1(Dataset):
    def __init__(self, folder_a, folder_b,bev_ch_use=None,aug=False,meta_num=1):
        self.files_a = sorted([os.path.join(folder_a, f) for f in os.listdir(folder_a) if f.endswith('.npz')])
        self.files_b = sorted([os.path.join(folder_b, f) for f in os.listdir(folder_b) if f.endswith('.npz')])
        self.ch_use = bev_ch_use
        self.aug = aug
        self.meta_num = meta_num
        with open("occ_token.pkl", 'rb') as f:
            self.token_dict=pickle.load(f)

        assert len(self.files_a) == len(self.files_b), "Number of files in folder_a and folder_b should be the same."
    
    def __getitem__(self, index):
        data_a = np.load(self.files_a[index])
        data_b = np.load(self.files_b[index])
        
        data_b = data_b['arr_0']

        if self.ch_use:
            data_b = nBEV1(data_b,self.ch_use)

        data_b = torch.tensor(data_b,dtype=torch.float)
        data_b = rot_flip(data_b)

        data_a = torch.tensor(data_a['arr_0'])

        if self.aug:
            # print("rotate aug!")
            if random.random()<0.2:
                random_angle = random.randint(-30, 30)
                data_a=rotate(data_a,random_angle)
                data_b=rotate(data_b,random_angle)
            if random.random()<0.3:
                data_a=torch.fliplr(data_a)
                data_b=torch.fliplr(data_b)

        file_path = self.files_b[index]
        token=file_path.split('/')[-1].replace('.npz','')
        occ_path=self.token_dict[token]
        label = np.load(occ_path)
        data_occ = label['semantics']

        c_score = cal_occ_meta(data_occ,self.meta_num)
        c_score= torch.tensor(c_score,dtype=torch.float)

        return data_a, data_b, c_score
    
    def __len__(self):
        return len(self.files_a)


class CustomDataset_Int2Bit(Dataset):
    def __init__(self, folder_a, folder_b,bev_ch_use=None):
        self.files_a = sorted([os.path.join(folder_a, f) for f in os.listdir(folder_a) if f.endswith('.npz')])
        self.files_b = sorted([os.path.join(folder_b, f) for f in os.listdir(folder_b) if f.endswith('.npz')])
        self.ch_use = bev_ch_use
        assert len(self.files_a) == len(self.files_b), "Number of files in folder_a and folder_b should be the same."
    
    def __getitem__(self, index):
        data_a = np.load(self.files_a[index])
        data_b = np.load(self.files_b[index])
        
        data_b = data_b['arr_0']

        if self.ch_use:
            data_b = Int2Bit_BEV(data_b,self.ch_use)

        data_b = torch.tensor(data_b,dtype=torch.float)
        data_b = rot_flip(data_b)
        # data_b = torch.rot90(data_b, k=1, dims=(1, 2))
        # data_b = torch.flip(data_b, dims=[1])

        return torch.tensor(data_a['arr_0']), data_b
    
    def __len__(self):
        return len(self.files_a)

def cal_occ_meta(data_occ,meta_num):
    # all
    valid_pts_num=np.sum(data_occ != 17)
    c_score = min(valid_pts_num/50000,1)
    
    terrain_pts=np.sum(data_occ == 14)
    c_terrain = min(terrain_pts/7000,1)

    manmade_pts=np.sum(data_occ == 15)
    c_manmade = min(manmade_pts/16000,1)

    vegetation_pts=np.sum(data_occ == 16)
    c_vegetation = min(vegetation_pts/20000,1)

    pts_other = valid_pts_num - terrain_pts - manmade_pts - vegetation_pts
    c_other = min(pts_other/25000,1)

    if meta_num==1:
        # c_score= torch.tensor([c_score],dtype=torch.float)
        c_score = [c_score]
    elif meta_num==4:
        # c_score= torch.tensor([c_score,c_terrain,c_manmade,c_vegetation],dtype=torch.float)
        # c_score=[c_score,c_terrain,c_manmade,c_vegetation]
        c_score=[c_other,c_terrain,c_manmade,c_vegetation]
    
    return c_score

class CustomDataset_nBEV1_eval(Dataset):
    def __init__(self, folder_a, folder_b,bev_ch_use=None,meta_num=1):
        self.files_a = sorted([os.path.join(folder_a, f) for f in os.listdir(folder_a) if f.endswith('.npz')])
        self.files_b = sorted([os.path.join(folder_b, f) for f in os.listdir(folder_b) if f.endswith('.npz')])
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        with open("occ_token.pkl", 'rb') as f:
            self.token_dict=pickle.load(f)
        
    def __getitem__(self, index):

        data_a = np.load(self.files_a[index])
        data_b = np.load(self.files_b[index])
        file_path = self.files_b[index]
        token=file_path.split('/')[-1].replace('.npz','')

        data_b = data_b['arr_0']

        if self.ch_use:
            data_b = nBEV1(data_b,self.ch_use)

        data_b = torch.tensor(data_b,dtype=torch.float)
        data_b = rot_flip(data_b)
        data_a = torch.tensor(data_a['arr_0'])

        occ_path=self.token_dict[token]
        label = np.load(occ_path)
        data_occ = label['semantics']

        c_score = cal_occ_meta(data_occ,self.meta_num)
        c_score= torch.tensor(c_score,dtype=torch.float)

        data_occ = torch.from_numpy(data_occ)

        return data_a, data_b, data_occ, c_score
    
    def __len__(self):
        return len(self.files_b)


class CustomDataset_nBEV1_time(Dataset):
    def __init__(self, imageset, gts_path, folder_a, folder_b,bev_ch_use=None,meta_num=1):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        self.nusc_infos = data['infos']
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]

        self.gts_path =gts_path
        self.Zmid_root = folder_a
        self.Bev_root = folder_b
        # self.files_a = sorted([os.path.join(folder_a, f) for f in os.listdir(folder_a) if f.endswith('.npz')])
        # self.files_b = sorted([os.path.join(folder_b, f) for f in os.listdir(folder_b) if f.endswith('.npz')])
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        
    def __getitem__(self, index):

        index = index % len(self.nusc_infos)
        scene_name = self.scene_names[index]
        scene_len = self.scene_lens[index]
        idx=5
        self.return_len=10#scene_len

        occs = []
        tokens=[]

        Zmids =[]
        Bevs = []
        occ_metas = [] 
        for i in range(self.return_len):
            token = self.nusc_infos[scene_name][idx + i]['token']
            tokens.append(token)
            label_file = os.path.join(self.gts_path, f'{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)

            Zmid_file =  os.path.join(self.Zmid_root, f'{token}.npz')
            Zmid = np.load(Zmid_file)
            Zmid = Zmid['arr_0']
            Zmids.append(Zmid)

            Bev_file =  os.path.join(self.Bev_root, f'{token}.npz')
            Bev = np.load(Bev_file)
            Bev = Bev['arr_0']
            if self.ch_use:
                Bev = nBEV1(Bev,self.ch_use)
            Bevs.append(Bev)

            # valid_pts_num=np.sum(occ != 17)
            # c_score = min(valid_pts_num/50000,1)
            c_score = cal_occ_meta(occ,self.meta_num)
            occ_metas.append(c_score)
            # occ_metas.append([c_score])

            # c_score= torch.tensor([c_score],dtype=torch.float)

            
        data_occ = np.stack(occs).astype(np.int64)
        data_occ = torch.from_numpy(data_occ)

        data_Z = np.stack(Zmids)
        data_Z = torch.from_numpy(data_Z)

        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        data_Bev = torch.flip(data_Bev, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)
        # data_metas = torch.tensor(occ_metas)

        return data_Z, data_Bev, data_occ,data_metas
    
    def __len__(self):
        return len(self.scene_names)


class CustomDataset_2frame_time(Dataset):
    def __init__(self, imageset, gts_path, folder_a, folder_b,bev_ch_use=None,meta_num=1):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        self.nusc_infos = data['infos']
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]

        self.gts_path =gts_path
        self.Zmid_root = folder_a
        self.Bev_root = folder_b
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        # with open("occ_token.pkl", 'rb') as f:
        #     self.token_dict=pickle.load(f)
        
    def __getitem__(self, index):

        index = index % len(self.nusc_infos)
        scene_name = self.scene_names[index]
        scene_len = self.scene_lens[index]

        max_return_len=6#scene_len

        idx1 = np.random.randint(0, scene_len - max_return_len)
        idx2 = idx1 + np.random.randint(1, max_return_len)
        
        idx_s = [idx1,idx2]
        occs = []
        tokens=[]

        Zmids =[]
        Bevs = []
        occ_metas = [] 
        Pose = []
        for idx in idx_s:
            token = self.nusc_infos[scene_name][idx]['token']
            tokens.append(token)
            label_file = os.path.join(self.gts_path, f'{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)

            Zmid_file =  os.path.join(self.Zmid_root, f'{token}.npz')
            Zmid = np.load(Zmid_file)
            Zmid = Zmid['arr_0']
            Zmids.append(Zmid)

            Bev_file =  os.path.join(self.Bev_root, f'{token}.npz')
            Bev = np.load(Bev_file)
            Bev = Bev['arr_0']
            if self.ch_use:
                Bev = nBEV1(Bev,self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta(occ,self.meta_num)
            occ_metas.append(c_score)

            info = self.nusc_infos[scene_name][idx]
            ego2global_translation = info['ego2global_translation']
            ego2global_rotation = info['ego2global_rotation']
            ego2global= np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T

            lidar2global=ego2global @ lidar2ego
            Pose.append(lidar2global)

        
        data_occ = np.stack(occs).astype(np.int64)
        data_occ = torch.from_numpy(data_occ)

        data_Z = np.stack(Zmids)
        data_Z = torch.from_numpy(data_Z)

        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        data_Bev = torch.flip(data_Bev, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)

        Pose_1to2 = np.linalg.inv(Pose[1]) @ Pose[0]

        Pose_t = Pose_1to2[:3,3]
        Pose_r = Pose_1to2[:3, :3].flatten()

        Pose_meta = np.concatenate((Pose_t,Pose_r))
        Pose_meta = torch.tensor(Pose_meta,dtype=torch.float)

        return data_Z, data_Bev, data_occ, data_metas, Pose_meta
    
    def __len__(self):
        return len(self.scene_names) * 40


class CustomDataset_2frame_continuous(Dataset):
    def __init__(self, imageset, gts_path, folder_a, folder_b,bev_ch_use=None,meta_num=1,Tframe=6):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        self.nusc_infos = data['infos']
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]

        self.gts_path =gts_path
        self.Zmid_root = folder_a
        self.Bev_root = folder_b
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe
        # with open("occ_token.pkl", 'rb') as f:
        #     self.token_dict=pickle.load(f)
        
    def __getitem__(self, index):

        scene_index = index % len(self.nusc_infos)
        scene_name = self.scene_names[scene_index]
        scene_len = self.scene_lens[scene_index]

        max_return_len=self.Tframe#scene_len

        idx1 = np.random.randint(0, scene_len - max_return_len)
        # idx2 = idx1 + np.random.randint(1, max_return_len)
        
        # idx_s = [idx1,idx2]
        idx_s = list(range(idx1,idx1+max_return_len))
        occs = []
        tokens=[]

        Zmids =[]
        Bevs = []
        occ_metas = [] 
        Pose = []
        for idx in idx_s:
            token = self.nusc_infos[scene_name][idx]['token']
            tokens.append(token)
            label_file = os.path.join(self.gts_path, f'{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)

            Zmid_file =  os.path.join(self.Zmid_root, f'{token}.npz')
            Zmid = np.load(Zmid_file)
            Zmid = Zmid['arr_0']
            Zmids.append(Zmid)

            Bev_file =  os.path.join(self.Bev_root, f'{token}.npz')
            Bev = np.load(Bev_file)
            Bev = Bev['arr_0']
            if self.ch_use:
                Bev = nBEV1(Bev,self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta(occ,self.meta_num)
            occ_metas.append(c_score)

            info = self.nusc_infos[scene_name][idx]
            ego2global_translation = info['ego2global_translation']
            ego2global_rotation = info['ego2global_rotation']
            ego2global= np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T

            lidar2global=ego2global @ lidar2ego
            Pose.append(lidar2global)

        
        data_occ = np.stack(occs).astype(np.int64)
        data_occ = torch.from_numpy(data_occ)

        data_Z = np.stack(Zmids)
        data_Z = torch.from_numpy(data_Z)

        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        data_Bev = torch.flip(data_Bev, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)

        Pose_metas=[]
        for i in range(max_return_len-1):
            Pose_1to2 = np.linalg.inv(Pose[i+1]) @ Pose[i]
            Pose_t = Pose_1to2[:3,3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t,Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas,dtype=torch.float)

        return data_Z, data_Bev, data_occ, data_metas, Pose_metas
    
    def __len__(self):
        return len(self.scene_names) * 32
        # return len(self.scene_names) * 5



class CustomDataset_Tframe_continuous(Dataset):
    def __init__(self, imageset, gts_path, folder_b,bev_ch_use=None,meta_num=1,Tframe=6,training=False):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        self.nusc_infos = data['infos']
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]

        self.gts_path =gts_path
        self.Bev_root = folder_b
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe
        self.training = training
        # with open("occ_token.pkl", 'rb') as f:
        #     self.token_dict=pickle.load(f)
        
    def __getitem__(self, index):

        scene_index = index % len(self.nusc_infos)
        scene_name = self.scene_names[scene_index]
        scene_len = self.scene_lens[scene_index]

        max_return_len=self.Tframe#scene_len

        idx1 = np.random.randint(0, scene_len - max_return_len)
        # idx2 = idx1 + np.random.randint(1, max_return_len)
        
        # idx_s = [idx1,idx2]
        idx_s = list(range(idx1,idx1+max_return_len))
        occs = []
        tokens=[]

        Zmids =[]
        Bevs = []
        occ_metas = [] 
        Pose = []
        for idx in idx_s:
            token = self.nusc_infos[scene_name][idx]['token']
            tokens.append(token)
            label_file = os.path.join(self.gts_path, f'{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)


            Bev_file =  os.path.join(self.Bev_root, f'{token}.npz')
            Bev = np.load(Bev_file)
            Bev = Bev['arr_0']
            if self.ch_use:
                Bev = nBEV1(Bev,self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta(occ,self.meta_num)
            occ_metas.append(c_score)

            info = self.nusc_infos[scene_name][idx]
            ego2global_translation = info['ego2global_translation']
            ego2global_rotation = info['ego2global_rotation']
            ego2global= np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T

            lidar2global=ego2global @ lidar2ego
            Pose.append(lidar2global)

        
        data_occ = np.stack(occs).astype(np.int64)
        data_occ = torch.from_numpy(data_occ)

        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        data_Bev = torch.flip(data_Bev, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)

        Pose_metas=[]
        for i in range(max_return_len-1):
            Pose_1to2 = np.linalg.inv(Pose[i+1]) @ Pose[i]
            Pose_t = Pose_1to2[:3,3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t,Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas,dtype=torch.float)

        return data_occ, data_Bev, data_metas, Pose_metas
    
    def __len__(self):
        if self.training:
            return len(self.scene_names) * 32 # train 32 test 5
        else:
            return len(self.scene_names) * 5
        

def cal_occ_meta_nuplan(data_occ,meta_num):
    # all
    valid_pts_num=np.sum(data_occ != 0)
    c_score = min(valid_pts_num/50000,1)

    terrain_pts=np.sum(data_occ == 1)
    c_terrain = min(terrain_pts/7000,1)

    manmade_pts=np.sum(data_occ == 2)
    c_manmade = min(manmade_pts/16000,1)

    vegetation_pts=np.sum(data_occ == 3)
    c_vegetation = min(vegetation_pts/20000,1)

    pts_other = valid_pts_num - terrain_pts - manmade_pts - vegetation_pts
    c_other = min(pts_other/25000,1)

    if meta_num==1:
        # c_score= torch.tensor([c_score],dtype=torch.float)
        c_score = [c_score]
    elif meta_num==4:
        # c_score= torch.tensor([c_score,c_terrain,c_manmade,c_vegetation],dtype=torch.float)
        # c_score=[c_score,c_terrain,c_manmade,c_vegetation]
        c_score=[c_other,c_terrain,c_manmade,c_vegetation]
    
    return c_score


class Nuplan_Occ_bev_Dataset(Dataset):
    def __init__(self, imageset, gts_path, bev_path,bev_ch_use=None,meta_num=1,Tframe=5,training=False):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        self.nuplan_infos = data['infos']
        # self.scene_names = data['scene_tokens']
        # self.scene_names = []
        self.scene_lens = []
        # print(new_occ_pkl_data_train["scene_tokens"][0])
        for i in range(len(data['scene_tokens'])):
            # scene_name = data['scene_tokens'][i][0]
            # scene_name = str(list(scene_name)[0])
            if len(data['scene_tokens'][i])>100:
                self.scene_lens.append(len(data['scene_tokens'][i]))
                # self.scene_names.append(data["scene_tokens"][i])

        self.gts_path =gts_path
        self.bev_path = bev_path
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe
        self.training = training
        # self.poses = 
        

    def occ_arguement_mask(self, occ):
        # occ 是 200x200x16 的数据
        h, w, d = occ.shape
        assert h == 200 and w == 200, "occ 的尺寸必须是 200x200x16"

        masks = {}

        # 创建一个全为 0 的数组
        zero_filled = np.zeros_like(occ)

        # 定义所有可能的区域
        regions = {
            'top_half': (slice(0, h//2), slice(None), slice(None)),
            'bottom_half': (slice(h//2, h), slice(None), slice(None)),
            'left_half': (slice(None), slice(0, w//2), slice(None)),
            'right_half': (slice(None), slice(w//2, w), slice(None)),
            'top_3_4': (slice(0, 3*h//4), slice(None), slice(None)),
            'bottom_3_4': (slice(h//4, h), slice(None), slice(None)),
            'left_3_4': (slice(None), slice(0, 3*w//4), slice(None)),
            'right_3_4': (slice(None), slice(w//4, w), slice(None))
        }

        # 随机选择一个区域
        import random
        selected_region_name = random.choice(list(regions.keys()))
        selected_region = regions[selected_region_name]

        # 创建 mask
        mask = zero_filled.copy()
        mask[selected_region] = occ[selected_region]

        return mask, selected_region_name
    
    def __len__(self):
        if self.training:
            return len(self.scene_lens) * 32 # train 32 test 5
        else:
            return len(self.scene_lens) * 5
        
    def __getitem__(self, index):

        scene_index = index % len(self.scene_lens)
        # scene_name = str(list(self.scene_names[scene_index].keys())[0])
        scene_len = self.scene_lens[scene_index]

        max_return_len=self.Tframe#scene_len

        idx1 = np.random.randint(0, scene_len - max_return_len)
        # idx2 = idx1 + np.random.randint(1, max_return_len)
        
        # idx_s = [idx1,idx2]
        idx_s = list(range(idx1,idx1+max_return_len))
        occs_output = []
        occs_input = []
        tokens=[]

        Zmids =[]
        Bevs = []
        occ_metas = [] 
        Pose = []
        layer_to_merge=[0,1,3,4,5,6,7]
        scene_metas = {}
        for idx in idx_s:
            # token = self.nusc_infos[scene_name][idx]['token']
            token = self.nuplan_infos[idx]['token']
            tokens.append(token)
            occ_file = os.path.join(self.gts_path, f'{token}.npy')
            occ = np.load(occ_file)
            # print(f"occ shape: {occ.shape}")
            # argutment occ
            occs_output.append(occ)
            # if self.training:
            #     occ_argued, _ = self.occ_arguement_mask(occ)
            #     occs_input.append(occ_argued)


            bev_data =  np.load(os.path.join(self.bev_path, f'{token}.npz'))['gt_bev_masks']
            # print(f"bev shape{bev_data.shape}")
            bev_data[1,:,:] = np.any(bev_data[layer_to_merge, :, :], axis=0).astype(int)

            if self.ch_use:
                Bev = nBEV1(bev_data,self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta_nuplan(occ,self.meta_num)
            occ_metas.append(c_score)

            info = self.nuplan_infos[idx]
            ego2global_translation = info['ego2global_translation']
            ego2global_rotation = info['ego2global_rotation']
            ego2global= np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T

            lidar2global=ego2global @ lidar2ego
            Pose.append(lidar2global)

        scene_metas.update({"scene_tokens": tokens})
        out_data_occ = np.stack(occs_output).astype(np.int64)
        out_data_occ = torch.from_numpy(out_data_occ)
        if self.training:
            in_data_occ = np.stack(occs_input).astype(np.int64)
            in_data_occ = torch.from_numpy(in_data_occ)
        else: 
            in_data_occ = out_data_occ


        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        # data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        # data_Bev = torch.flip(data_Bev, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)
        

        Pose_metas=[]
        for i in range(max_return_len-1):
            Pose_1to2 = np.linalg.inv(Pose[i+1]) @ Pose[i]
            Pose_t = Pose_1to2[:3,3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t,Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas,dtype=torch.float)

        # return in_data_occ, out_data_occ, data_Bev, data_metas, Pose_metas
        return in_data_occ,  data_Bev, data_metas, Pose_metas, scene_metas
    

class Nuplan_Occ_bev_HR_full(Dataset):
    def __init__(self, imageset, gts_path, bev_path,bev_ch_use=None,meta_num=1,Tframe=5,training=False, debug=False):
        with open(imageset, 'rb') as f:
            pkl_data = pickle.load(f)
        self.nuplan_infos = pkl_data['infos']
        self.scene_lens = []
        for i in range(len(pkl_data['scene_tokens'])):
            if len(pkl_data['scene_tokens'][i])>100:
                self.scene_lens.append(len(pkl_data['scene_tokens'][i]))

        self.gts_path =gts_path
        self.bev_path = bev_path
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe
        self.training = training
        self.debug = debug
        if "clip_infos" in pkl_data:
            self.clip_infos = pkl_data['clip_infos']
        if self.debug:
            self.clip_infos = self.clip_infos[:1000]


    def occ_arguement_mask(self, occ):
        # occ 是 200x200x16 的数据
        h, w, d = occ.shape
        assert h == 200 and w == 200, "occ 的尺寸必须是 200x200x16"

        masks = {}

        # 创建一个全为 0 的数组
        zero_filled = np.zeros_like(occ)

        # 定义所有可能的区域
        regions = {
            'top_half': (slice(0, h//2), slice(None), slice(None)),
            'bottom_half': (slice(h//2, h), slice(None), slice(None)),
            'left_half': (slice(None), slice(0, w//2), slice(None)),
            'right_half': (slice(None), slice(w//2, w), slice(None)),
            'top_3_4': (slice(0, 3*h//4), slice(None), slice(None)),
            'bottom_3_4': (slice(h//4, h), slice(None), slice(None)),
            'left_3_4': (slice(None), slice(0, 3*w//4), slice(None)),
            'right_3_4': (slice(None), slice(w//4, w), slice(None))
        }

        # 随机选择一个区域
        import random
        selected_region_name = random.choice(list(regions.keys()))
        selected_region = regions[selected_region_name]

        # 创建 mask
        mask = zero_filled.copy()
        mask[selected_region] = occ[selected_region]

        return mask, selected_region_name
    
    def __len__(self):
        return len(self.clip_infos)
        # if self.training:
        #     return len(self.scene_lens) * 32 # train 32 test 5
        # else:
        #     return len(self.scene_lens) * 5

    def convert_to_voxel_grid_int(self, voxels_):
        voxel = np.zeros((400,400,32), dtype=np.int32)

        x = voxels_[:,0].astype(np.int32)
        y = voxels_[:,1].astype(np.int32)
        z = voxels_[:,2].astype(np.int32)
        val = voxels_[:,3].astype(np.int32)+1

        mask = (
            (x >= 0) & (x < 400) &
            (y >= 0) & (y < 400) &
            (z >= 0) & (z < 32)
        )
        x = x[mask]
        y = y[mask]
        z = z[mask]
        val = val[mask]

        voxel[x, y, z] = val
        return voxel
    
    def __getitem__(self, index):
        # 最大重试次数，避免无限循环
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                clip = self.clip_infos[index]
                occ_datas = []
                tokens = []
                Bevs = []
                occ_metas = [] 
                Pose = []
                layer_to_merge = [0,1,3,4,5,6,7]
                scene_metas = {}
                
                # 检查当前clip的所有数据是否存在
                all_files_exist = True
                for idx in clip:
                    token = self.nuplan_infos[idx]['token']
                    occ_file = os.path.join(self.gts_path, f'{token}/{token}.npz')
                    bev_file = os.path.join(self.bev_path, f'{token}.npz')
                    if not (os.path.exists(occ_file) and os.path.exists(bev_file)):
                        all_files_exist = False
                        break
                
                # 如果文件不存在，随机选择新的index
                if not all_files_exist:
                    index = np.random.randint(0, len(self.clip_infos))
                    continue
                
                # 所有文件都存在，开始加载数据
                for idx in clip:
                    token = self.nuplan_infos[idx]['token']
                    tokens.append(token)
                    
                    # 加载occ数据
                    occ_file = os.path.join(self.gts_path, f'{token}/{token}.npz')
                    occ = np.load(occ_file)['occ']
                    occ_voxel_grid = self.convert_to_voxel_grid_int(occ)
                    occ_datas.append(occ_voxel_grid)

                    # 加载bev数据
                    bev_data = np.load(os.path.join(self.bev_path, f'{token}.npz'))['gt_bev_masks']
                    bev_data[1,:,:] = np.any(bev_data[layer_to_merge, :, :], axis=0).astype(int)

                    if self.ch_use:
                        Bev = nBEV1(bev_data, self.ch_use)
                    else:
                        Bev = bev_data
                    Bevs.append(Bev)

                    # 计算occ meta
                    c_score = cal_occ_meta_nuplan(occ, self.meta_num)
                    occ_metas.append(c_score)

                    # 处理pose信息
                    info = self.nuplan_infos[idx]
                    ego2global_translation = info['ego2global_translation']
                    ego2global_rotation = info['ego2global_rotation']
                    ego2global = np.eye(4)
                    ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
                    ego2global[:3, 3] = np.array(ego2global_translation).T

                    lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
                    lidar2ego = np.eye(4)
                    lidar2ego[:3, :3] = lidar2ego_r
                    lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T

                    lidar2global = ego2global @ lidar2ego
                    Pose.append(lidar2global)
                
                # 数据加载成功，跳出重试循环
                break
                
            except Exception as e:
                print(f"数据加载失败，重试中... (尝试 {attempt + 1}/{max_attempts}): {str(e)}")
                index = np.random.randint(0, len(self.clip_infos))
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"数据加载失败，已达到最大重试次数: {str(e)}")

        scene_metas.update({"scene_tokens": tokens})
        data_occs = np.stack(occ_datas).astype(np.int64)
        data_occs = torch.from_numpy(data_occs)

        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        # data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        # data_Bev = torch.flip(data_Bev, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)
        
        Pose_metas = []
        max_return_len = self.Tframe
        for i in range(max_return_len-1):
            Pose_1to2 = np.linalg.inv(Pose[i+1]) @ Pose[i]
            Pose_t = Pose_1to2[:3,3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t,Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas, dtype=torch.float)

        # return in_data_occ, out_data_occ, data_Bev, data_metas, Pose_metas
        return data_occs,  data_Bev, data_metas, Pose_metas, scene_metas

class Nuplan_Occ_bev_HR_mini(Dataset):
    def __init__(self, imageset, gts_path, bev_path,bev_ch_use=None,meta_num=1,Tframe=5,training=False, debug=False):
        with open(imageset, 'rb') as f:
            pkl_data = pickle.load(f)
        self.nuplan_infos = pkl_data['infos']
        self.scene_lens = []
        for i in range(len(pkl_data['scene_tokens'])):
            if len(pkl_data['scene_tokens'][i])>100:
                self.scene_lens.append(len(pkl_data['scene_tokens'][i]))

        self.gts_path =gts_path
        self.bev_path = bev_path
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe
        self.training = training
        self.debug = debug
        if "clip_infos" in pkl_data:
            self.clip_infos = pkl_data['clip_infos']
        if self.debug:
            self.clip_infos = self.clip_infos[:1000]


    def occ_arguement_mask(self, occ):
        # occ 是 200x200x16 的数据
        h, w, d = occ.shape
        assert h == 200 and w == 200, "occ 的尺寸必须是 200x200x16"

        masks = {}

        # 创建一个全为 0 的数组
        zero_filled = np.zeros_like(occ)

        # 定义所有可能的区域
        regions = {
            'top_half': (slice(0, h//2), slice(None), slice(None)),
            'bottom_half': (slice(h//2, h), slice(None), slice(None)),
            'left_half': (slice(None), slice(0, w//2), slice(None)),
            'right_half': (slice(None), slice(w//2, w), slice(None)),
            'top_3_4': (slice(0, 3*h//4), slice(None), slice(None)),
            'bottom_3_4': (slice(h//4, h), slice(None), slice(None)),
            'left_3_4': (slice(None), slice(0, 3*w//4), slice(None)),
            'right_3_4': (slice(None), slice(w//4, w), slice(None))
        }

        # 随机选择一个区域
        import random
        selected_region_name = random.choice(list(regions.keys()))
        selected_region = regions[selected_region_name]

        # 创建 mask
        mask = zero_filled.copy()
        mask[selected_region] = occ[selected_region]

        return mask, selected_region_name
    
    def __len__(self):
        return len(self.clip_infos)
        # if self.training:
        #     return len(self.scene_lens) * 32 # train 32 test 5
        # else:
        #     return len(self.scene_lens) * 5
        
    def __getitem__(self, index):

        # scene_index = index % len(self.scene_lens)
        # # scene_name = str(list(self.scene_names[scene_index].keys())[0])
        # scene_len = self.scene_lens[scene_index]

        max_return_len=self.Tframe#scene_len

        # idx1 = np.random.randint(0, scene_len - max_return_len)
        # # idx2 = idx1 + np.random.randint(1, max_return_len)
        
        # # idx_s = [idx1,idx2]
        # idx_s = list(range(idx1,idx1+max_return_len))
        clip = self.clip_infos[index]
        occ_datas = []
        tokens=[]

        Zmids =[]
        Bevs = []
        occ_metas = [] 
        Pose = []
        layer_to_merge=[0,1,2,3,4,5,6,7]
        scene_metas = {}
        for idx in clip:
            # token = self.nusc_infos[scene_name][idx]['token']
            token = self.nuplan_infos[idx]['token']
            tokens.append(token)
            occ_file = os.path.join(self.gts_path, f'{token}.npy')
            occ = np.load(occ_file)
            # print(f"occ shape: {occ.shape}")
            # argutment occ
            occ_datas.append(occ)
            # if self.training:
            #     occ_argued, _ = self.occ_arguement_mask(occ)
            #     occs_input.append(occ_argued)


            bev_data =  np.load(os.path.join(self.bev_path, f'{token}.npz'))['gt_bev_masks']
            # print(f"bev shape{bev_data.shape}")
            bev_data[1,:,:] = np.any(bev_data[layer_to_merge, :, :], axis=0).astype(int)

            if self.ch_use:
                Bev = nBEV1(bev_data,self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta_nuplan(occ,self.meta_num)
            occ_metas.append(c_score)

            info = self.nuplan_infos[idx]
            ego2global_translation = info['ego2global_translation']
            ego2global_rotation = info['ego2global_rotation']
            ego2global= np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T

            lidar2global=ego2global @ lidar2ego
            Pose.append(lidar2global)

        scene_metas.update({"scene_tokens": tokens})
        data_occs = np.stack(occ_datas).astype(np.int64)
        data_occs = torch.from_numpy(data_occs)


        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        # data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        # data_Bev = torch.flip(data_Bev, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)
        

        Pose_metas=[]
        for i in range(max_return_len-1):
            Pose_1to2 = np.linalg.inv(Pose[i+1]) @ Pose[i]
            Pose_t = Pose_1to2[:3,3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t,Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas,dtype=torch.float)

        # return in_data_occ, out_data_occ, data_Bev, data_metas, Pose_metas
        return data_occs,  data_Bev, data_metas, Pose_metas, scene_metas



class Nuplan_Occ_bev_LR_mini(Dataset):
    def __init__(self, imageset, gts_path, bev_path,bev_ch_use=None,meta_num=1,Tframe=5,training=False):
        with open(imageset, 'rb') as f:
            pkl_data = pickle.load(f)
        self.nuplan_infos = pkl_data['infos']
        self.scene_lens = []
        for i in range(len(pkl_data['scene_tokens'])):
            if len(pkl_data['scene_tokens'][i])>100:
                self.scene_lens.append(len(pkl_data['scene_tokens'][i]))

        self.gts_path =gts_path
        self.bev_path = bev_path
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe
        self.training = training
        if "clip_infos" in pkl_data:
            self.clip_infos = pkl_data['clip_infos']


    def occ_arguement_mask(self, occ):
        # occ 是 200x200x16 的数据
        h, w, d = occ.shape
        assert h == 200 and w == 200, "occ 的尺寸必须是 200x200x16"

        masks = {}

        # 创建一个全为 0 的数组
        zero_filled = np.zeros_like(occ)

        # 定义所有可能的区域
        regions = {
            'top_half': (slice(0, h//2), slice(None), slice(None)),
            'bottom_half': (slice(h//2, h), slice(None), slice(None)),
            'left_half': (slice(None), slice(0, w//2), slice(None)),
            'right_half': (slice(None), slice(w//2, w), slice(None)),
            'top_3_4': (slice(0, 3*h//4), slice(None), slice(None)),
            'bottom_3_4': (slice(h//4, h), slice(None), slice(None)),
            'left_3_4': (slice(None), slice(0, 3*w//4), slice(None)),
            'right_3_4': (slice(None), slice(w//4, w), slice(None))
        }

        # 随机选择一个区域
        import random
        selected_region_name = random.choice(list(regions.keys()))
        selected_region = regions[selected_region_name]

        # 创建 mask
        mask = zero_filled.copy()
        mask[selected_region] = occ[selected_region]

        return mask, selected_region_name
    
    def __len__(self):
        return len(self.clip_infos)
        # if self.training:
        #     return len(self.scene_lens) * 32 # train 32 test 5
        # else:
        #     return len(self.scene_lens) * 5
        
    def __getitem__(self, index):

        # scene_index = index % len(self.scene_lens)
        # # scene_name = str(list(self.scene_names[scene_index].keys())[0])
        # scene_len = self.scene_lens[scene_index]

        max_return_len=self.Tframe#scene_len

        # idx1 = np.random.randint(0, scene_len - max_return_len)
        # # idx2 = idx1 + np.random.randint(1, max_return_len)
        
        # # idx_s = [idx1,idx2]
        # idx_s = list(range(idx1,idx1+max_return_len))
        clip = self.clip_infos[index]
        occ_datas = []
        tokens=[]

        Zmids =[]
        Bevs = []
        occ_metas = [] 
        Pose = []
        layer_to_merge=[0,1,2,3,4,5,6,7]
        scene_metas = {}
        for idx in clip:
            # token = self.nusc_infos[scene_name][idx]['token']
            token = self.nuplan_infos[idx]['token']
            tokens.append(token)
            occ_file = os.path.join(self.gts_path, f'{token}.npy')
            occ = np.load(occ_file)
            # print(f"occ shape: {occ.shape}")
            # argutment occ
            occ_datas.append(occ)
            # if self.training:
            #     occ_argued, _ = self.occ_arguement_mask(occ)
            #     occs_input.append(occ_argued)


            bev_data =  np.load(os.path.join(self.bev_path, f'{token}.npz'))['gt_bev_masks']
            # print(f"bev shape{bev_data.shape}")
            bev_data[1,:,:] = np.any(bev_data[layer_to_merge, :, :], axis=0).astype(int)

            if self.ch_use:
                Bev = nBEV1(bev_data,self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta_nuplan(occ,self.meta_num)
            occ_metas.append(c_score)

            info = self.nuplan_infos[idx]
            ego2global_translation = info['ego2global_translation']
            ego2global_rotation = info['ego2global_rotation']
            ego2global= np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T

            lidar2global=ego2global @ lidar2ego
            Pose.append(lidar2global)


        scene_metas.update({"scene_tokens": tokens})
        data_occs = np.stack(occ_datas).astype(np.int64)
        data_occs = torch.from_numpy(data_occs)


        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        # data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        # data_Bev = torch.flip(data_Bev, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)
        
        Pose_metas=[]
        for i in range(max_return_len-1):
            Pose_1to2 = np.linalg.inv(Pose[i+1]) @ Pose[i]
            Pose_t = Pose_1to2[:3,3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t,Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas,dtype=torch.float)

        return data_occs,  data_Bev, data_metas, Pose_metas, scene_metas
        # return data_occs,  data_Bev, data_metas,  scene_metas


class Nuplan_Occbev_bev_Dataset(Dataset):
    def __init__(self, imageset, gts_path, bev_path,bev_ch_use=None,meta_num=1,Tframe=6,training=False):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        self.nuplan_infos = data['infos']
        self.scene_lens = []
        for i in range(len(data['scene_tokens'])):
            if len(data['scene_tokens'][i])>50:
                self.scene_lens.append(len(data['scene_tokens'][i]))

        self.gts_path =gts_path
        self.bev_path = bev_path
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe
        self.training = training

    def occ_arguement_mask(self, occ):
        # occ 是 200x200x16 的数据
        h, w, d = occ.shape
        assert h == 200 and w == 200, "occ 的尺寸必须是 200x200x16"

        masks = {}

        # 创建一个全为 0 的数组
        zero_filled = np.zeros_like(occ)

        # 定义所有可能的区域
        regions = {
            'top_half': (slice(0, h//2), slice(None), slice(None)),
            'bottom_half': (slice(h//2, h), slice(None), slice(None)),
            'left_half': (slice(None), slice(0, w//2), slice(None)),
            'right_half': (slice(None), slice(w//2, w), slice(None)),
            'top_3_4': (slice(0, 3*h//4), slice(None), slice(None)),
            'bottom_3_4': (slice(h//4, h), slice(None), slice(None)),
            'left_3_4': (slice(None), slice(0, 3*w//4), slice(None)),
            'right_3_4': (slice(None), slice(w//4, w), slice(None))
        }

        # 随机选择一个区域
        import random
        selected_region_name = random.choice(list(regions.keys()))
        selected_region = regions[selected_region_name]

        # 创建 mask
        mask = zero_filled.copy()
        mask[selected_region] = occ[selected_region]

        return mask, selected_region_name
    
    def __len__(self):
        if self.training:
            return len(self.scene_lens) * 32 # train 32 test 5
        else:
            return len(self.scene_lens) * 5
        
    def __getitem__(self, index):

        scene_index = index % len(self.scene_lens)
        # scene_name = str(list(self.scene_names[scene_index].keys())[0])
        scene_len = self.scene_lens[scene_index]

        max_return_len=self.Tframe#scene_len

        idx1 = np.random.randint(0, scene_len - max_return_len)
        # idx2 = idx1 + np.random.randint(1, max_return_len)
        
        # idx_s = [idx1,idx2]
        idx_s = list(range(idx1,idx1+max_return_len))
        occs_output = []
        occs_input = []
        tokens=[]

        Zmids =[]
        Bevs = []
        occ_metas = [] 
        Pose = [] 
        layer_to_merge=[0,1,3,4,5,6,7]
        for idx in idx_s:
            # token = self.nusc_infos[scene_name][idx]['token']
            token = self.nuplan_infos[idx]['token']
            tokens.append(token)
            occ_file = os.path.join(self.gts_path, f'{token}.npy')
            occ_data = np.load(occ_file)

            bev_data =  np.load(os.path.join(self.bev_path, f'{token}.npz'))['gt_bev_masks']
            # print(f"bev shape{bev_data.shape}")
            bev_data[1,:,:] = np.any(bev_data[layer_to_merge, :, :], axis=0).astype(int)

            occ_with_bev = replace_occ_grid_with_bev_nuplan(input_occ=occ_data, bevlayout=bev_data)
            # print(f"occ shape: {occ_with_bev.shape}")
            # argutment occ
            occs_output.append(occ_with_bev)
            if self.training:
                occ_argued, _ = self.occ_arguement_mask(occ_with_bev)
                occs_input.append(occ_argued)


            if self.ch_use:
                Bev = nBEV1(bev_data,self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta_nuplan(occ_with_bev,self.meta_num)
            occ_metas.append(c_score)

            info = self.nuplan_infos[idx]
            ego2global_translation = info['ego2global_translation']
            ego2global_rotation = info['ego2global_rotation']
            ego2global= np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T

            lidar2global=ego2global @ lidar2ego
            Pose.append(lidar2global)


        out_data_occ = np.stack(occs_output).astype(np.int64)
        out_data_occ = torch.from_numpy(out_data_occ)
        if self.training:
            in_data_occ = np.stack(occs_input).astype(np.int64)
            in_data_occ = torch.from_numpy(in_data_occ)
        else: 
            in_data_occ = out_data_occ


        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        data_Bev = torch.flip(data_Bev, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)

        Pose_metas=[]
        for i in range(max_return_len-1):
            Pose_1to2 = np.linalg.inv(Pose[i+1]) @ Pose[i]
            Pose_t = Pose_1to2[:3,3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t,Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas,dtype=torch.float)
        
        in_data_occ = out_data_occ.clone()

        return in_data_occ, out_data_occ, data_Bev, data_metas, Pose_metas


class Nuplan_Occbev_bev_Dataset_demo(Dataset):
    def __init__(self, imageset, gts_path, bev_path,bev_ch_use=None,meta_num=1,Tframe=6,training=False):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        self.nuplan_infos = data['infos']
        self.scene_lens = []
        for i in range(len(data['scene_tokens'])):
            if len(data['scene_tokens'][i])>50:
                self.scene_lens.append(len(data['scene_tokens'][i]))

        self.gts_path =gts_path
        self.bev_path = bev_path
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe
        self.training = training

    def occ_arguement_mask(self, occ):
        # occ 是 200x200x16 的数据
        h, w, d = occ.shape
        assert h == 200 and w == 200, "occ 的尺寸必须是 200x200x16"

        masks = {}

        # 创建一个全为 0 的数组
        zero_filled = np.zeros_like(occ)

        # 定义所有可能的区域
        regions = {
            'top_half': (slice(0, h//2), slice(None), slice(None)),
            'bottom_half': (slice(h//2, h), slice(None), slice(None)),
            'left_half': (slice(None), slice(0, w//2), slice(None)),
            'right_half': (slice(None), slice(w//2, w), slice(None)),
            'top_3_4': (slice(0, 3*h//4), slice(None), slice(None)),
            'bottom_3_4': (slice(h//4, h), slice(None), slice(None)),
            'left_3_4': (slice(None), slice(0, 3*w//4), slice(None)),
            'right_3_4': (slice(None), slice(w//4, w), slice(None))
        }

        # 随机选择一个区域
        import random
        selected_region_name = random.choice(list(regions.keys()))
        selected_region = regions[selected_region_name]

        # 创建 mask
        mask = zero_filled.copy()
        mask[selected_region] = occ[selected_region]

        return mask, selected_region_name
    
    def __len__(self):
        if self.training:
            return len(self.scene_lens) * 32 # train 32 test 5
        else:
            return len(self.scene_lens) * 5
        
    def __getitem__(self, index):

        scene_index = index % len(self.scene_lens)
        # scene_name = str(list(self.scene_names[scene_index].keys())[0])
        scene_len = self.scene_lens[scene_index]

        max_return_len=self.Tframe#scene_len

        idx1 = np.random.randint(0, scene_len - max_return_len)
        # idx2 = idx1 + np.random.randint(1, max_return_len)
        
        # idx_s = [idx1,idx2]
        idx_s = list(range(idx1,idx1+max_return_len))
        occs_output = []
        occs_input = []
        tokens=[]

        Zmids =[]
        Bevs = []
        occ_metas = [] 
        Pose = [] 
        layer_to_merge=[0,1,3,4,5,6,7]
        for idx in idx_s:
            # token = self.nusc_infos[scene_name][idx]['token']
            token = self.nuplan_infos[idx]['token']
            tokens.append(token)
            occ_file = os.path.join(self.gts_path, f'{token}.npy')
            occ_data = np.load(occ_file)

            bev_data =  np.load(os.path.join(self.bev_path, f'{token}.npz'))['gt_bev_masks']
            # print(f"bev shape{bev_data.shape}")
            bev_data[1,:,:] = np.any(bev_data[layer_to_merge, :, :], axis=0).astype(int)

            occ_with_bev = replace_occ_grid_with_bev_nuplan(input_occ=occ_data, bevlayout=bev_data)
            # print(f"occ shape: {occ_with_bev.shape}")
            # argutment occ
            occs_output.append(occ_with_bev)
            if self.training:
                occ_argued, _ = self.occ_arguement_mask(occ_with_bev)
                occs_input.append(occ_argued)


            if self.ch_use:
                Bev = nBEV1(bev_data,self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta_nuplan(occ_with_bev,self.meta_num)
            occ_metas.append(c_score)

            info = self.nuplan_infos[idx]
            ego2global_translation = info['ego2global_translation']
            ego2global_rotation = info['ego2global_rotation']
            ego2global= np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T

            lidar2global=ego2global @ lidar2ego
            Pose.append(lidar2global)


        out_data_occ = np.stack(occs_output).astype(np.int64)
        out_data_occ = torch.from_numpy(out_data_occ)
        if self.training:
            in_data_occ = np.stack(occs_input).astype(np.int64)
            in_data_occ = torch.from_numpy(in_data_occ)
        else: 
            in_data_occ = out_data_occ


        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        data_Bev = torch.flip(data_Bev, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)

        Pose_metas=[]
        for i in range(max_return_len-1):
            Pose_1to2 = np.linalg.inv(Pose[i+1]) @ Pose[i]
            Pose_t = Pose_1to2[:3,3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t,Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas,dtype=torch.float)
        
        in_data_occ = out_data_occ.clone()

        return in_data_occ, out_data_occ, data_Bev, data_metas, Pose_metas


class Nuplan_Occbev_bev_Dataset_clip(Dataset):
    def __init__(self, imageset, gts_path, bev_path,bev_ch_use=None,meta_num=1,Tframe=6,training=False):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        self.nuplan_infos = data['infos']
        self.scene_lens = []
        for i in range(len(data['scene_tokens'])):
            if len(data['scene_tokens'][i])>50:
                self.scene_lens.append(len(data['scene_tokens'][i]))

        self.gts_path =gts_path
        self.bev_path = bev_path
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe
        self.training = training

    def occ_arguement_mask(self, occ):
        # occ 是 200x200x16 的数据
        h, w, d = occ.shape
        assert h == 200 and w == 200, "occ 的尺寸必须是 200x200x16"

        masks = {}

        # 创建一个全为 0 的数组
        zero_filled = np.zeros_like(occ)

        # 定义所有可能的区域
        regions = {
            'top_half': (slice(0, h//2), slice(None), slice(None)),
            'bottom_half': (slice(h//2, h), slice(None), slice(None)),
            'left_half': (slice(None), slice(0, w//2), slice(None)),
            'right_half': (slice(None), slice(w//2, w), slice(None)),
            'top_3_4': (slice(0, 3*h//4), slice(None), slice(None)),
            'bottom_3_4': (slice(h//4, h), slice(None), slice(None)),
            'left_3_4': (slice(None), slice(0, 3*w//4), slice(None)),
            'right_3_4': (slice(None), slice(w//4, w), slice(None))
        }

        # 随机选择一个区域
        import random
        selected_region_name = random.choice(list(regions.keys()))
        selected_region = regions[selected_region_name]

        # 创建 mask
        mask = zero_filled.copy()
        mask[selected_region] = occ[selected_region]

        return mask, selected_region_name
    
    def __len__(self):
        if self.training:
            return len(self.scene_lens) * 32 # train 32 test 5
        else:
            return len(self.scene_lens) * 5
        
    def __getitem__(self, index):

        scene_index = index % len(self.scene_lens)
        # scene_name = str(list(self.scene_names[scene_index].keys())[0])
        scene_len = self.scene_lens[scene_index]

        max_return_len=self.Tframe#scene_len

        idx1 = np.random.randint(0, scene_len - max_return_len)
        # idx2 = idx1 + np.random.randint(1, max_return_len)
        
        # idx_s = [idx1,idx2]
        idx_s = list(range(idx1,idx1+max_return_len))
        occs_output = []
        occs_input = []
        tokens=[]

        Zmids =[]
        Bevs = []
        occ_metas = [] 
        Pose = [] 
        layer_to_merge=[0,1,3,4,5,6,7]
        for idx in idx_s:
            # token = self.nusc_infos[scene_name][idx]['token']
            token = self.nuplan_infos[idx]['token']
            tokens.append(token)
            occ_file = os.path.join(self.gts_path, f'{token}.npy')
            occ_data = np.load(occ_file)

            bev_data =  np.load(os.path.join(self.bev_path, f'{token}.npz'))['gt_bev_masks']
            # print(f"bev shape{bev_data.shape}")
            bev_data[1,:,:] = np.any(bev_data[layer_to_merge, :, :], axis=0).astype(int)

            occ_with_bev = replace_occ_grid_with_bev_nuplan(input_occ=occ_data, bevlayout=bev_data)
            # print(f"occ shape: {occ_with_bev.shape}")
            # argutment occ
            occs_output.append(occ_with_bev)
            if self.training:
                occ_argued, _ = self.occ_arguement_mask(occ_with_bev)
                occs_input.append(occ_argued)


            if self.ch_use:
                Bev = nBEV1(bev_data,self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta_nuplan(occ_with_bev,self.meta_num)
            occ_metas.append(c_score)

            info = self.nuplan_infos[idx]
            ego2global_translation = info['ego2global_translation']
            ego2global_rotation = info['ego2global_rotation']
            ego2global= np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T

            lidar2global=ego2global @ lidar2ego
            Pose.append(lidar2global)


        out_data_occ = np.stack(occs_output).astype(np.int64)
        out_data_occ = torch.from_numpy(out_data_occ)
        if self.training:
            in_data_occ = np.stack(occs_input).astype(np.int64)
            in_data_occ = torch.from_numpy(in_data_occ)
        else: 
            in_data_occ = out_data_occ


        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        data_Bev = torch.flip(data_Bev, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)

        Pose_metas=[]
        for i in range(max_return_len-1):
            Pose_1to2 = np.linalg.inv(Pose[i+1]) @ Pose[i]
            Pose_t = Pose_1to2[:3,3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t,Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas,dtype=torch.float)
        
        in_data_occ = out_data_occ.clone()

        return in_data_occ, out_data_occ, data_Bev, data_metas, Pose_metas


class Nuplan_Occ_bev_Dataset_pro(Dataset):
    # only nes
    def __init__(self, imageset, gts_path, bev_path,bev_ch_use=None,meta_num=1,Tframe=6,training=False):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        self.nuplan_infos = data['infos']
        # self.scene_tokens = data['scene_tokens']
        self.scene_names = []
        self.scene_lens = []
        # print(new_occ_pkl_data_train["scene_tokens"][0])
        for i in range(len(data['scene_tokens'])):
            scene_name = list(data['scene_tokens'][i].keys())[0]
            # scene_name = str(list(scene_name)[0])
            if len(data['scene_tokens'][i][scene_name])>100:
                self.scene_lens.append(data['scene_tokens'][i])
                # self.scene_names.append()

        self.gts_path =gts_path
        self.bev_path = bev_path
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe
        self.training = training
        # bev_file_temp = "/mnt/volumes/ad-lmm-data-proc-bd-ga/hzhu/data/nuplan_bev/mini/train/aceca657ec465591.npy"
        # bev_temp = one_hot_decode(np.load(bev_file_temp), 18)
        # print(bev_temp.shape)

    def __len__(self):
        if self.training:
            return len(self.scene_lens) * 32 # train 32 test 5
        else:
            return len(self.scene_lens) * 5
        
    def __getitem__(self, index):

        scene_index = index % len(self.scene_lens)
        
        # scene_name = str(list(self.scene_names[scene_index].keys())[0])
        # scene_name = self.scene_names[scene_index]
        scene_len = len(list(self.scene_lens[scene_index].values())[0])
        scene_name = list(self.scene_lens[scene_index].keys())[0]

        max_return_len=self.Tframe#scene_len
        # print(f"==>>{scene_len}, and {max_return_len}")
        idx1 = np.random.randint(0, scene_len - max_return_len)
        # idx2 = idx1 + np.random.randint(1, max_return_len)
        
        # idx_s = [idx1,idx2]
        idx_s = list(range(idx1,idx1+max_return_len))
        occs = []
        tokens=[]

        Zmids =[]
        Bevs = []
        occ_metas = [] 
        Pose = []
        for idx in idx_s:
            # token = self.nusc_infos[scene_name][idx]['token']
            # token = self.nuplan_infos[scene_name][idx]['token']
            token = self.scene_lens[scene_index][scene_name][idx]
            tokens.append(token)
            occ_file = os.path.join(self.gts_path, f'{token}.npy')
            occ = np.load(occ_file)
            occs.append(occ)


            Bev_file =  os.path.join(self.bev_path, f'{token}.npy')
            Bev = one_hot_decode(np.load(Bev_file), 18)

            if self.ch_use:
                Bev = nBEV1(Bev,self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta_nuplan(occ,self.meta_num)
            occ_metas.append(c_score)

            info = self.nuplan_infos[idx]
            ego2global_translation = info['ego2global_translation']
            ego2global_rotation = info['ego2global_rotation']
            ego2global= np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T

            lidar2global=ego2global @ lidar2ego
            Pose.append(lidar2global)

        
        data_occ = np.stack(occs).astype(np.int64)
        data_occ = torch.from_numpy(data_occ)

        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        data_Bev = torch.flip(data_Bev, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)

        Pose_metas=[]
        for i in range(max_return_len-1):
            Pose_1to2 = np.linalg.inv(Pose[i+1]) @ Pose[i]
            Pose_t = Pose_1to2[:3,3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t,Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas,dtype=torch.float)

        return data_occ, data_Bev, data_metas, Pose_metas
    


class CustomDataset_Tframe_12hz(Dataset):
    def __init__(self, imageset, occ_base_path, folder_b,bev_ch_use=None,meta_num=1,Tframe=8,training=False,use_clip=False,return_token=False,use_occ3d=False):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        self.nusc_infos = data['infos']
        # self.scene_names = list(self.nusc_infos.keys())
        # self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]

        self.occ_base_path =occ_base_path
        self.Bev_root = folder_b
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe
        self.return_len = Tframe
        self.return_token = return_token
        # self.training = training
        # with open("occ_token.pkl", 'rb') as f:
        #     self.token_dict=pickle.load(f)
        self.start_on_keyframe = True
        self.start_on_firstframe = False
        
        self.use_clip = use_clip
        self.occ3d = use_occ3d
        if use_clip:
            self.clip_infos=self.build_clips(self.nusc_infos,data['scene_tokens'])
        else:
            self.get_scene_len(self.nusc_infos,data['scene_tokens'])

    def fliter_clips(self,clip):
        for frame in clip:
            token = self.nusc_infos[frame]['token']
            folder_path = F"{self.occ_base_path}/{token}"
            npy_files = glob.glob(f"{folder_path}/*.npy")
            if not npy_files:
                return 0
            file_path = npy_files[0]
            if not os.path.exists(file_path):
                return 0
            # if os.path.exists(f"{self.occ_base_path}/{token}.npy")==False:
                # return 0
        return 1
    
    def fliter_useful_clips(self,clip):
        useful_tokens = ["2f5de0aeca704127925cf8490ff5a21d4",
                    "fd8420396768425eabec9bdddf7e64b6",
                    "b2329c1fe1714b9bb2642b257d38755d",
                    "119fb420f7574100b8407f21475cfff3",
                    "13350a2c46de430f9b20264cfb1bd3d1",
                    "0c4faeb94da6430f8c1c581ac5adbb79"] 
        for frame in clip:
            token = self.nusc_infos[frame]['token']
            if token in useful_tokens:
                return True
        return False
    
    def get_scene_len(self,data_infos,scene_tokens):
        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)}
        scene_lens=[]
        scene_infos=[]
        for scene in scene_tokens[:100]:
            scene_lens.append(len(scene))
            scene_token_idx = [self.token_data_dict[token] for token in scene]
            scene_infos.append(scene_token_idx)
        self.scene_lens = scene_lens
        self.scene_infos = scene_infos
  
    def build_clips(self, data_infos, scene_tokens):
        """Since the order in self.data_infos may change on loading, we
        calculate the index for clips after loading.

        Args:
            data_infos (list of dict): loaded data_infos
            scene_tokens (2-dim list of str): 2-dim list for tokens to each
            scene 

        Returns:
            2-dim list of int: int is the index in self.data_infos
        """
        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)}
        all_clips = []
        for scene in scene_tokens:
            for start in range(len(scene) - self.return_len + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + self.return_len]]
                # if self.fliter_clips(clip)==0:
                #     continue
                
                # if self.fliter_useful_clips(clip)==0:
                #     continue
                all_clips.append(clip)
                if self.start_on_firstframe:
                    break
                
            clip = [self.token_data_dict[token]  for token in scene[-self.return_len:]]
            all_clips.append(clip)
            
        # logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
        #              f"continuous scenes. Cut into {self.video_length}-clip, "
        #              f"which has {len(all_clips)} in total.")
        return all_clips
    
    def __getitem__(self, index):

        if self.use_clip:
            clip = self.clip_infos[index]
        else:
            scene_index = index % len(self.scene_lens)
            scene_clip = self.scene_infos[scene_index]
            scene_len = self.scene_lens[scene_index]
            # scene_name = self.scene_names[scene_index]
            # scene_len = self.scene_lens[scene_index]
            # max_return_len=self.Tframe#scene_len

            # idx1 = np.random.randint(0, scene_len - self.Tframe)
            idx1 = 8
            # Using sequential idx
            # idx1 = index    
            clip = scene_clip[idx1:idx1+self.Tframe]
        
        # print(clip)
        occs = []
        tokens=[]

        Zmids =[]
        Bevs = []
        occ_metas = [] 
        Pose = []
        Bevs_ori = [] 
        for frame in clip:
            token = self.nusc_infos[frame]['token']
            tokens.append(token)
            # label_file = os.path.join(self.gts_path, f'{scene_name}/{token}/labels.npz')
            # label = np.load(label_file)
            # occ = label['semantics']
            # occ = np.load(f"{self.occ_base_path}/{token}.npy")
            occ = np.load(glob.glob(f"{self.occ_base_path}/{token}/*.npy")[0])
            
            if self.occ3d:
                occ[occ==0] = 17
            occs.append(occ)


            Bev_file =  os.path.join(self.Bev_root, f'{token}.npz')
            Bev = np.load(Bev_file)
            # npz_data['arr_0'].shape
            # print(f"=> Bev shape: {Bev['arr_0'].shape}")
            # Bev = Bev['arr_0'][:,::4,::4] # 18,200,200
            Bev = Bev['arr_0'] # 18,200,200
            Bevs_ori.append(Bev)
            if self.ch_use:
                Bev = nBEV1(Bev,self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta(occ,self.meta_num)
            occ_metas.append(c_score)

            info = self.nusc_infos[frame]
            ego2global_translation = info['ego2global_translation']
            ego2global_rotation = info['ego2global_rotation']
            ego2global= np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T

            lidar2global=ego2global @ lidar2ego
            Pose.append(lidar2global)

        
        data_occ = np.stack(occs).astype(np.int64)
        data_occ = torch.from_numpy(data_occ)

        data_Bev = np.stack(Bevs).astype(np.float32) 
        data_Bev = torch.from_numpy(data_Bev)
        # data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        # data_Bev = torch.flip(data_Bev, dims=[2])

        data_Bev_ori = np.stack(Bevs_ori)
        data_Bev_ori = torch.from_numpy(data_Bev_ori)
        
        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)

        Pose_metas=[]
        for i in range(self.Tframe-1):
            Pose_1to2 = np.linalg.inv(Pose[i+1]) @ Pose[i]
            Pose_t = Pose_1to2[:3,3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t,Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas,dtype=torch.float)

        if self.return_token==False:
            return data_occ, data_Bev, data_metas, Pose_metas
        return data_occ, data_Bev, data_metas, Pose_metas,tokens,data_Bev_ori
    
    def __len__(self):
        if self.use_clip:
            return len(self.clip_infos)
        else:
            return len(self.scene_lens) *40 #training
        

class CustomDataset_save_bevlayout_png(Dataset):
    def __init__(self, imageset, folder_b,bev_ch_use=None,return_token=True):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        self.nusc_infos = data['infos']
        self.Bev_root = folder_b
        self.ch_use = bev_ch_use
        self.return_token = return_token
    
    def __getitem__(self, index):

        token = self.nusc_infos[index]['token']
        Bev_file =  os.path.join(self.Bev_root, f'{token}.npz')
        Bev = np.load(Bev_file)
        Bev = Bev['arr_0'] # 18,800,800
        
        return Bev, token
    
    def __len__(self):
        
        return len(self.nusc_infos)
        
class CustomDataset_wm_continuous(Dataset):
    def __init__(self, imageset, gts_path, folder_a, folder_b, bev_ch_use=None, meta_num=1, Tframe=6):
        with open(imageset, "rb") as f:
            data = pickle.load(f)
        self.nusc_infos = data["infos"]
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]

        self.gts_path = gts_path
        self.Zmid_root = folder_a
        self.Bev_root = folder_b
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe
        # with open("occ_token.pkl", 'rb') as f:
        #     self.token_dict=pickle.load(f)

    def __getitem__(self, index):

        scene_index = index % len(self.nusc_infos)
        scene_name = self.scene_names[scene_index]
        scene_len = self.scene_lens[scene_index]

        max_return_len = self.Tframe  # scene_len

        idx1 = np.random.randint(0, scene_len - max_return_len)
        # idx2 = idx1 + np.random.randint(1, max_return_len)

        # idx_s = [idx1,idx2]
        idx_s = list(range(idx1, idx1 + max_return_len))
        occs = []
        tokens = []

        Zmids = []
        Bevs = []
        occ_metas = []
        Pose = []
        for idx in idx_s:
            token = self.nusc_infos[scene_name][idx]["token"]
            tokens.append(token)
            label_file = os.path.join(self.gts_path, f"{scene_name}/{token}/labels.npz")
            label = np.load(label_file)
            occ = label["semantics"]
            occs.append(occ)

            # Zmid_file = os.path.join(self.Zmid_root, f"{token}.npz")
            # Zmid = np.load(Zmid_file)
            # Zmid = Zmid["arr_0"]
            # Zmids.append(Zmid)

            Bev_file = os.path.join(self.Bev_root, f"{token}.npz")
            Bev = np.load(Bev_file)
            Bev = Bev["arr_0"]
            if self.ch_use:
                Bev = nBEV1(Bev, self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta(occ, self.meta_num)
            occ_metas.append(c_score)

            info = self.nusc_infos[scene_name][idx]
            ego2global_translation = info["ego2global_translation"]
            ego2global_rotation = info["ego2global_rotation"]
            ego2global = np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"]).T

            lidar2global = ego2global @ lidar2ego
            Pose.append(lidar2global)

        data_occ = np.stack(occs).astype(np.int64)
        data_occ = torch.from_numpy(data_occ)

        data_Z = None
        # data_Z = np.stack(Zmids)
        # data_Z = torch.from_numpy(data_Z)

        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        data_Bev = torch.flip(data_Bev, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)

        Pose_metas = []
        for i in range(max_return_len - 1):
            Pose_1to2 = np.linalg.inv(Pose[i + 1]) @ Pose[i]
            Pose_t = Pose_1to2[:3, 3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t, Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas, dtype=torch.float)

        return data_Z, data_Bev, data_occ, data_metas, Pose_metas

    def __len__(self):
        return len(self.scene_names) * 32
        # return len(self.scene_names) * 5

def preprocess_infos_to_dict(data):
  """
  将 data['infos'] 列表转换为以 token 为键的字典。

  Args:
    data: 包含 'infos' 列表的字典。

  Returns:
    一个以 token 为键，原始信息字典为值的字典。
    如果输入无效或处理中出错，则返回一个空字典或 None。
  """
  if 'infos' not in data or not isinstance(data['infos'], list):
    print("错误：'data' 中没有 'infos' 键或其值不是列表。")
    return {} # 返回空字典

  infos_dict = {}
  for item in data['infos']:
    # 确保元素是字典并且包含 'token' 键
    if isinstance(item, dict) and 'token' in item:
      token_value = item['token']
      # 处理可能的 token 重复：这里简单地以后出现的为准，
      # 您也可以根据需求决定如何处理（例如，报错、合并等）
      infos_dict[token_value] = item
    # else:
      # 可以选择记录或忽略格式不正确的项
      # print(f"警告：跳过格式不正确的项：{item}")

  return infos_dict

class CustomDataset_wm_continuous_pro(Dataset):
    # only static
    def __init__(self, imageset, gts_path, folder_a, bev_path, bev_ch_use=None, meta_num=1, Tframe=6):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        self.nuplan_infos = preprocess_infos_to_dict(data=data)

        self.gts_path = gts_path
        self.Zmid_root = folder_a
        self.bev_path = bev_path
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe

        self.scene_lens = []
        # self.scene_names = data['scene_tokens_with_stationary_segments']
        self.scene_names = []
        for i in range(len(data['scene_tokens_with_stationary_segments'])):
            scene_name = data['scene_tokens_with_stationary_segments'][i]
            # scene_name = str(list(scene_name)[0])
            if len(data['scene_tokens_with_stationary_segments'][i])>10:
                self.scene_lens.append(len(data['scene_tokens_with_stationary_segments'][i]))
                self.scene_names.append(scene_name)


    def __getitem__(self, index):

        scene_index = index % len(self.scene_lens)
        scene_name = self.scene_names[scene_index]
        scene_len = self.scene_lens[scene_index]

        max_return_len = self.Tframe  # scene_len

        idx1 = np.random.randint(0, scene_len - max_return_len)
        # idx2 = idx1 + np.random.randint(1, max_return_len)

        # idx_s = [idx1,idx2]
        idx_s = list(range(idx1, idx1 + max_return_len))
        occs = []
        tokens = []

        # Zmids = []
        Bevs = []
        occ_metas = []
        Pose = []
        for idx in idx_s:
            token = self.scene_names[scene_index][idx]
            tokens.append(token)
            occ_file = os.path.join(self.gts_path, f'{token}.npy')
            occ = np.load(occ_file)
            occs.append(occ)

            # Zmid_file = os.path.join(self.Zmid_root, f"{token}.npz")
            # Zmid = np.load(Zmid_file)
            # Zmid = Zmid["arr_0"]
            # Zmids.append(Zmid)

            Bev_file =  os.path.join(self.bev_path, f'{token}.npz')
            # Bev = one_hot_decode(np.load(Bev_file), 18)
            Bev = np.load(Bev_file)['gt_bev_masks']

            if self.ch_use:
                Bev = nBEV1(Bev, self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta(occ, self.meta_num)
            occ_metas.append(c_score)

            info = self.nuplan_infos[token]
            ego2global_translation = info["ego2global_translation"]
            ego2global_rotation = info["ego2global_rotation"]
            ego2global = np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"]).T

            lidar2global = ego2global @ lidar2ego
            Pose.append(lidar2global)

        data_occ = np.stack(occs).astype(np.int64)
        data_occ = torch.from_numpy(data_occ)

        data_Z = None
        # data_Z = np.stack(Zmids)
        # data_Z = torch.from_numpy(data_Z)

        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        data_Bev = torch.flip(data_Bev, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)

        Pose_metas = []
        for i in range(max_return_len - 1):
            Pose_1to2 = np.linalg.inv(Pose[i + 1]) @ Pose[i]
            Pose_t = Pose_1to2[:3, 3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t, Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas, dtype=torch.float)

        # return data_Z, data_Bev, data_occ, data_metas, Pose_metas
        return data_Bev, data_occ, data_metas, Pose_metas

    def __len__(self):
        return len(self.scene_names) * 32
        # return len(self.scene_names) * 5

