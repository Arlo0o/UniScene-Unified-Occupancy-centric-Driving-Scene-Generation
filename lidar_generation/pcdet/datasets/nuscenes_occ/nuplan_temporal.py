import os
from typing import Tuple
import random
from collections import defaultdict
import pickle
from pypcd import pypcd
import numpy as np
from pyquaternion import Quaternion
import torch
from ..dataset import DatasetTemplate
from .nuplan_constants import *
from ..augmentor.data_augmentor_occ2lidar import DataAugmentorOcc2LiDAR
from .nuplan import NuPlanOccDataset

def cartesian_to_spherical(coords):
    # coords 是大小为 (N, 3) 的 ndarray，表示 N 个点的 (x, y, z) 坐标
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    
    # 计算 r
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # 计算 theta (xy 平面的角度)
    theta = np.arctan2(y, x)
    
    # 计算 phi (与 z 轴的夹角)
    phi = np.arctan2(np.sqrt(x**2 + y**2), z)
    
    # 返回大小为 (N, 3) 的球坐标 (theta, phi, r)
    return np.stack((theta, phi, r), axis=-1)

class NuPlanOccDatasetTemporal(NuPlanOccDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        self.data_root = dataset_cfg.lidar_path
        self.occ_path = dataset_cfg.occ_path
        self.pkl_path = dataset_cfg.pkl_path
        self.top_lidar_only = dataset_cfg.get('top_lidar_only', False)
        self.occ_size = self.grid_size
        if not self.training:
            self.pkl_path = dataset_cfg.val_pkl_path
        self.load_infos(self.pkl_path)
        # self.full_list = os.listdir(self.occ_path)
        self.compute_missing_points = False
        self.data_augmentor = DataAugmentorOcc2LiDAR(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        self.lidar_height = 1.7
        self.return_len = dataset_cfg.get('return_len', 8)
        self.offset =  dataset_cfg.get('offset', 0)
        self.times = 200

    def __len__(self):
        return len(self.full_list) * self.times

    def load_infos(self, info_file):
        # TODO: 先生成好
        with open(info_file, 'rb') as f:
            infos = pickle.load(f)

        self.token_to_seq_id_map = {}
        for scene_id, sample_tokens in enumerate(infos['scene_tokens']):
            self.token_to_seq_id_map.update(dict(zip(sample_tokens, [scene_id]*len(sample_tokens))))

        token_to_info_id = dict(zip([info['token'] for info in infos['infos']], range(len(infos['infos']))))
        self.token_to_lidar_path = dict(zip([info['token'] for info in infos['infos']], [info['lidar_path'] for info in infos['infos']]))

        self.full_list = infos['scene_tokens']

        # for raydrop labeling
        # self.seq_id_to_info_map = {}
        # self.seq_id_to_times = {}
        # self.seq_id_to_poses = {}
        # for scene_id, sample_tokens in enumerate(infos['scene_tokens']):
        #     self.seq_id_to_info_map[scene_id] = [infos['infos'][token_to_info_id[sample_token]] for sample_token in sample_tokens]

        #     times = [infos['infos'][token_to_info_id[sample_token]]['timestamp']/1e6 for sample_token in sample_tokens]
        #     times = torch.tensor(times, dtype=torch.float64) 
        #     self.seq_id_to_times[scene_id] = times

        #     poses = []
        #     for info in self.seq_id_to_info_map[scene_id]:
        #         l2e_r = info['lidar2ego_rotation']
        #         l2e_t = info['lidar2ego_translation']
        #         e2g_r = info['ego2global_rotation']
        #         e2g_t = info['ego2global_translation']
        #         l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        #         e2g_r_mat = Quaternion(e2g_r).rotation_matrix
        #         l2e_mat = np.zeros((4, 4), dtype='float')
        #         l2e_mat[-1, -1] = 1.0
        #         l2e_mat[:3, :3] = l2e_r_mat
        #         l2e_mat[:3, -1] = np.array(l2e_t)

        #         e2g_mat = np.zeros((4, 4), dtype='float')
        #         e2g_mat[-1, -1] = 1.0
        #         e2g_mat[:3, :3] = e2g_r_mat
        #         e2g_mat[:3, -1] = np.array(e2g_t)

        #         pose = e2g_mat @ l2e_mat # lidar -> world
        #         poses.append(pose)
        #     poses = torch.tensor(np.array(poses), dtype=torch.float64)
        #     self.seq_id_to_poses[scene_id] = poses
    
    def __getitem__(self, idx):
        idx = idx % self.times
        sample_tokens = self.full_list[idx]
        scene_len = len(sample_tokens)
        start_idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        data_dict_all = []
        for idx in range(start_idx, start_idx + self.return_len + self.offset):
            input_dict = {}
            occ_filename = sample_tokens[idx]
            input_dict['frame_id'] = occ_filename
            lidar_filename = os.path.join(self.data_root, self.token_to_lidar_path[occ_filename])

            if (not os.path.exists(os.path.join(self.occ_path, occ_filename+'.npy'))) or (not os.path.exists(lidar_filename)):
                print(f'occ not found !!! {occ_filename}')
                return self.__getitem__(random.randint(0, len(self)-1))


            try:
                occ = np.load(os.path.join(self.occ_path, occ_filename+'.npy'))#['occ']
            except FileNotFoundError:
                return self.__getitem__(random.randint(0, len(self)-1))
            occ_loc = np.stack(occ.nonzero(), axis=-1)[:, [2, 1, 0]]
            occ = np.concatenate([occ_loc, occ[occ_loc[:, 2], occ_loc[:, 1], occ_loc[:, 0]][:, None]], axis=-1)
            # to zyx
            #occ = occ[:, [2,1,0,3]]
            
            try:
                if self.compute_missing_points:
                    cur_token = occ_filename.split('/')[0]
                    seq_id = self.token_to_seq_id_map[occ_filename.split('/')[0]]
                    seq_infos = self.seq_id_to_info_map[seq_id]
                    tokens = [info['token'] for info in seq_infos]
                    times = self.seq_id_to_times[seq_id]
                    poses = self.seq_id_to_poses[seq_id]
                    cur_time = times[tokens.index(cur_token)]
                    cur_pose = poses[tokens.index(cur_token)]
                    lidar, did_return, lidar_idxs = self.load_nuscenes_laserscan(lidar_filename, lidar_range = self.point_cloud_range, poses=poses, times=times, cur_time=cur_time, cur_pose=cur_pose)
                else:
                    lidar, did_return, lidar_idxs = self.load_nuscenes_laserscan(lidar_filename, lidar_range = self.point_cloud_range)
            except FileNotFoundError:
                return self.__getitem__(random.randint(0, len(self)-1))

            # occ_path = self.sample_dict.get(  os.path.join(*lidar_filename.split("/")[-3:]), "None"  )
            # #print( occ_path )
            # occ_path_out = self.occ_root + "scene_"+ occ_path.split("/")[0] +"/occupancy/" + occ_path.split("/")[1] + ".npy"
            # #print( occ_path_out )
            # occ = np.load(occ_path_out, encoding='bytes', allow_pickle=True)

            # lidar origin
            # 根据nuplan雷达配置，得到各个雷达射出位置
            lidar_origin = np.zeros_like(lidar[:, :3])
            for lidar_idx in range(self.lidar_num):
                lidar_origin[lidar[:, -1].astype('int')==lidar_idx] = np.array(self.lidar_origins[lidar_idx])
            # lidar = np.concatenate([lidar, lidar_origin], axis=-1)
            input_dict['sensor_loc'] = lidar_origin
            
            lidar[:, 2] -= self.lidar_height
            input_dict['points'] = lidar
            input_dict['did_return'] = did_return
            # to xyz(absolute coords) for data augmentor
            input_dict['occ'] = occ[:, [2, 1, 0, 3]].astype(lidar.dtype)
            voxel_size = np.array(self.voxel_size).reshape((-1, 3))
            pc_range = np.array(self.point_cloud_range[:3]).reshape((-1, 3))
            input_dict['occ'][:, :3] = (input_dict['occ'][:, :3] + 0.5) * voxel_size + pc_range
            input_dict['occ'][:, 2] -= self.lidar_height

            vis = False
            if vis:
                rad = np.zeros((input_dict['points'].shape[0], 3))
                rad[:, 0] = 255
                white = 255 * np.ones((input_dict['occ'].shape[0], 3))
                for_vis = np.concatenate([np.concatenate([input_dict['points'][:, :3], rad], axis=-1), np.concatenate([input_dict['occ'][:, :3], white], axis=-1)], axis=0)
                for_vis.astype('float32').tofile('z.bin')

            if idx > start_idx:
                input_dict['flip_x'] = data_dict_all[0]['flip_x']
                input_dict['flip_y'] = data_dict_all[0]['flip_y']
                input_dict['noise_rot'] = data_dict_all[0]['noise_rot']
                input_dict['noise_scale'] = data_dict_all[0]['noise_scale']
                input_dict['noise_translate'] = data_dict_all[0]['noise_translate']

            data_dict = self.prepare_data(data_dict=input_dict)

            vis = False
            if vis:
                rad = np.zeros((data_dict['points'].shape[0], 3))
                rad[:, 0] = 255
                white = 255 * np.ones((data_dict['occ'].shape[0], 3))
                for_vis = np.concatenate([np.concatenate([data_dict['points'][:, :3], rad], axis=-1), np.concatenate([data_dict['occ'][:, :3], white], axis=-1)], axis=0)
                for_vis.astype('float32').tofile('z.bin')

            # occ feature (x, y, z, theta, phi, r, cls)
            # to zyx for voxelization
            data_dict['occ'][:, :3] = ((data_dict['occ'][:, :3] - pc_range) / voxel_size)
            data_dict['occ'] = data_dict['occ'].astype(occ.dtype)
            occ_range_mask = (data_dict['occ'][:, 0] >= 0) & (data_dict['occ'][:, 0] < self.grid_size[0]) & \
                            (data_dict['occ'][:, 1] >= 0) & (data_dict['occ'][:, 1] < self.grid_size[1]) & \
                            (data_dict['occ'][:, 2] >= 0) & (data_dict['occ'][:, 2] < self.grid_size[2])
            data_dict['occ'] = data_dict['occ'][occ_range_mask]
            data_dict['occ'] = data_dict['occ'][:, [2, 1, 0, 3]]
            
            xyz = data_dict['occ'][:, [2, 1, 0]]
            occ_labels = data_dict['occ'][:, -1]
            xyz = (xyz + 0.5) * voxel_size + pc_range
            tpr = cartesian_to_spherical(xyz)
            cls_encoded = np.eye(len(self.class_names))[occ_labels]
            occ_feature = np.concatenate([xyz, tpr, cls_encoded], axis=-1)
            data_dict['occ'] = np.concatenate([data_dict['occ'], occ_feature], axis=-1)

            vis = False
            if vis:
                rad = np.zeros((data_dict['points'].shape[0], 3))
                rad[:, 0] = 255
                white = 255 * np.ones((data_dict['occ'].shape[0], 3))
                for_vis = np.concatenate([np.concatenate([data_dict['points'][:, :3], rad], axis=-1), np.concatenate([data_dict['occ'][:, 4:7], white], axis=-1)], axis=0)
                for_vis.astype('float32').tofile('z.bin')

            
            xyz = data_dict['occ'][:, [2, 1, 0]].astype(np.int32)
            data_dict['grid'] = np.zeros(self.occ_size, dtype=bool)
            data_dict['grid'][xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
            data_dict['grid'] = torch.from_numpy(data_dict['grid'])

            data_dict_all.append(data_dict)
        return self.collate_temporal(data_dict_all)

    @staticmethod
    def collate_temporal(batch_list):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        ret = {}
        ret['temporal_len'] = len(batch_list)
        for key, val in data_dict.items():
            if key in ['points', 'occ', 'did_return', 'sensor_loc']:
                coors = []
                if isinstance(val[0], list):
                    val = [i for item in val for i in item]
                for i, coor in enumerate(val):
                    if key == 'did_return':
                        coor = coor[:, np.newaxis].astype(np.int32)
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key in ['grid']:
                ret[key] = np.stack(val)
            elif key in ['frame_id', 'end_flag', 'tra']:
                ret[key] = val
        return ret

    @staticmethod
    def collate_batch(batch_list):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        ret = {}
        ret['batch_size'] = len(batch_list) * batch_list[0]['temporal_len']
        ret['real_batch_size'] = len(batch_list)
        for key, val in data_dict.items():
            if key in ['points', 'occ', 'did_return', 'sensor_loc']:
                coors = []
                if isinstance(val[0], list):
                    val = [i for item in val for i in item]
                for i, coor in enumerate(val):
                    coor[:, 0] += i * batch_list[0]['temporal_len']
                    coors.append(coor)
                ret[key] = np.concatenate(coors, axis=0)
            elif key in ['grid']:
                ret[key] = np.concatenate(val)
            elif key in ['frame_id', 'end_flag', 'tra']:
                ret[key] = val
        return ret
