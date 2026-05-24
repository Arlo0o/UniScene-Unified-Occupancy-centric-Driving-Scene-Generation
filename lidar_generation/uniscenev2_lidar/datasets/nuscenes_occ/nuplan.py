import os
from typing import Tuple
from collections import defaultdict
import pickle
import random
import copy
from glob import glob
import json
import shutil
from pypcd import pypcd
import numpy as np
from chamferdist import ChamferDistance
from pyquaternion import Quaternion
import torch
import torch.distributed as dist
from ..dataset import DatasetTemplate
from .nuplan_constants import *
from ..augmentor.data_augmentor_occ2lidar import DataAugmentorOcc2LiDAR

def get_missing_points(origins, points, lidar_ids, max_rings, tol=0.5, min_points_per_missing=5):
    # points: [N, 6 (x, y, z, intensity, ring, lidar_info)]
    missing_points_all = []
    missing_polars_all = []
    for lidar_id in lidar_ids:
        origin = origins[lidar_id]
        missing_points = []
        missing_polars = []
        for i in range(max_rings[lidar_id]):
            ring = points[:, 4]
            lidar_info = points[:, 5]
            points_ring = points[np.logical_and(ring==i, lidar_info==lidar_id)]

            thetas = (origin[2] - points_ring[:, 2]) / np.linalg.norm(points_ring[:, :3] - origin[:3], axis=1)
            thetas = -np.rad2deg(np.arcsin(thetas))
            phis = np.rad2deg(np.arctan2((points_ring[:, 1] - origin[1]), (points_ring[:, 0] - origin[0])))
            points_polar = np.column_stack((phis, thetas))

            # 去角度跳变
            if lidar_id == 3:
                points_polar[:, 0] = ((points_polar[:, 0] + 180) + 180) % 360
            elif lidar_id == 1:
                m = points_polar[:, 0] < -175
                points_polar = points_polar[~m]
            elif lidar_id == 2:
                m = points_polar[:, 0] > 175
                points_polar = points_polar[~m]

            sort_idx = np.argsort(points_polar[:, 0])
            points_polar = points_polar[sort_idx]

            dx = np.diff(points_polar[:, 0])
            dy = np.diff(points_polar[:, 1])
            dt = np.sqrt(dx**2 + dy**2)
            t = np.concatenate([[0], np.cumsum(dt)])
            spacing = np.median(np.diff(t))
            missing_p = np.where(dt > (spacing + spacing * tol))[0]
            missing_t = []
            for i in range(len(missing_p)):
                missing_seg = np.arange(t[missing_p[i]]+spacing, t[missing_p[i] + 1] - spacing * tol, spacing)
                if len(missing_seg) < min_points_per_missing:
                    continue
                missing_t.append(missing_seg)
            if len(missing_t) == 0:
                continue
            missing_t = np.concatenate(missing_t)

            coef_x = np.polyfit(t, points_polar[:, 0], deg=3)
            coef_y = np.polyfit(t, points_polar[:, 1], deg=3)
            poly_x = np.poly1d(coef_x)
            poly_y = np.poly1d(coef_y)

            interp_x = poly_x(missing_t)
            interp_y = poly_y(missing_t)
            if lidar_id == 3:
                interp_x = (interp_x - 180) % 360 - 180

            missing_polar = np.column_stack((interp_x, interp_y))

            missing_distance = 2e3
            x = (
                np.cos(np.deg2rad(missing_polar[:, 0]))
                * np.cos(np.deg2rad(missing_polar[:, 1]))
                * missing_distance
            )
            y = (
                np.sin(np.deg2rad(missing_polar[:, 0]))
                * np.cos(np.deg2rad(missing_polar[:, 1]))
                * missing_distance
            )
            z = np.sin(np.deg2rad(missing_polar[:, 1])) * missing_distance
            missing_points.append(np.column_stack([x, y, z]))
            missing_polars.append(missing_polar)
        if missing_points.__len__() == 0:
            missing_points = np.zeros((0, 3), dtype=origins.dtype)
            missing_polars = np.zeros((0, 2), dtype=origins.dtype)
        else:
            missing_points = np.concatenate(missing_points, axis=0)
            missing_polars = np.concatenate(missing_polars, axis=0)
        missing_points_all.append(missing_points)
        missing_polars_all.append(missing_polars)
    #if len(missing_points) == 0:
    #    return np.empty((0, 3)), np.empty((0, 2))
    # missing_points = np.concatenate(missing_points, axis=0)
    # missing_polars = np.concatenate(missing_polars, axis=0)
    return missing_points_all, missing_polars_all


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

class NuPlanOccDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        self.data_root = dataset_cfg.lidar_path
        self.occ_path = dataset_cfg.occ_path
        self.pkl_path = dataset_cfg.pkl_path
        self.top_lidar_only = dataset_cfg.get('top_lidar_only', False)
        self.occ_size = self.grid_size
        self.selected_seq = self.dataset_cfg.get('selected_seq', None)
        self.compute_missing_points = dataset_cfg.get('COMPUTE_MISSING_POINTS')
        if not self.training:
            self.pkl_path = dataset_cfg.val_pkl_path
        self.load_infos(self.pkl_path)
        # self.full_list = os.listdir(self.occ_path)
        self.full_list = list(self.token_to_lidar_path.keys())
        self.data_augmentor = DataAugmentorOcc2LiDAR(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        self.lidar_height = 1.7
        self.lidar_num = 5
        if dataset_cfg.get('lidar_origin_type', 'fixed') == 'fixed':
            self.lidar_origins = copy.deepcopy(NUPLAN_LIDAR_LOCS)
        else:
            self.lidar_origins = copy.deepcopy(NUPLAN_LIDAR_LOCS_NAIVE)

        for k, v in self.lidar_origins.items():
            self.lidar_origins[k][-1] -= self.lidar_height
        self.random_choice_lidar = dataset_cfg.get('random_choice_lidar', False)

    def __len__(self):
        return self.full_list.__len__()

    def load_infos(self, info_file):
        if isinstance(info_file, list):
            import gc
            infos = {'scene_tokens': [], 'infos': []}
            for chunk_file in info_file:
                with open(chunk_file, 'rb') as f:
                    chunk_infos = pickle.load(f)
                    infos['scene_tokens'].extend(chunk_infos['scene_tokens'])
                    infos['infos'].extend({'token': info['token'], 'lidar_path': info['lidar_path']} for info in chunk_infos['infos'])
                del chunk_infos
                gc.collect()
        else:
            # TODO: 先生成好
            with open(info_file, 'rb') as f:
                infos = pickle.load(f)

        if self.selected_seq is not None:
            # 取一个序列
            infos['scene_tokens'] = [infos['scene_tokens'][self.selected_seq]]
            infos['infos'] = [info for info in infos['infos'] if info['token'] in infos['scene_tokens'][0] ]

        token_to_info_id = dict(zip([info['token'] for info in infos['infos']], range(len(infos['infos']))))
        self.token_to_lidar_path = dict(zip([info['token'] for info in infos['infos']], [info['lidar_path'] for info in infos['infos']]))

        if self.compute_missing_points:
            self.token_to_seq_id_map = {}
            for scene_id, sample_tokens in enumerate(infos['scene_tokens']):
                self.token_to_seq_id_map.update(dict(zip(sample_tokens, [scene_id]*len(sample_tokens))))

            self.seq_id_to_info_map = {}
            self.seq_id_to_times = {}
            self.seq_id_to_poses = {}
            for scene_id, sample_tokens in enumerate(infos['scene_tokens']):
                self.seq_id_to_info_map[scene_id] = [infos['infos'][token_to_info_id[sample_token]] for sample_token in sample_tokens]

                times = [infos['infos'][token_to_info_id[sample_token]]['timestamp']/1e6 for sample_token in sample_tokens]
                times = torch.tensor(times, dtype=torch.float64) 
                self.seq_id_to_times[scene_id] = times

                poses = []
                for info in self.seq_id_to_info_map[scene_id]:
                    l2e_r = info['lidar2ego_rotation']
                    l2e_t = info['lidar2ego_translation']
                    e2g_r = info['ego2global_rotation']
                    e2g_t = info['ego2global_translation']
                    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                    e2g_r_mat = Quaternion(e2g_r).rotation_matrix
                    l2e_mat = np.zeros((4, 4), dtype='float')
                    l2e_mat[-1, -1] = 1.0
                    l2e_mat[:3, :3] = l2e_r_mat
                    l2e_mat[:3, -1] = np.array(l2e_t)

                    e2g_mat = np.zeros((4, 4), dtype='float')
                    e2g_mat[-1, -1] = 1.0
                    e2g_mat[:3, :3] = e2g_r_mat
                    e2g_mat[:3, -1] = np.array(e2g_t)

                    pose = e2g_mat @ l2e_mat # lidar -> world
                    poses.append(pose)
                poses = torch.tensor(np.array(poses), dtype=torch.float64)
                self.seq_id_to_poses[scene_id] = poses
    
    def __getitem__(self, idx):
        input_dict = {}
        occ_filename = self.full_list[idx] # self.full_list[idx].split('.')[0]
        input_dict['frame_id'] = occ_filename
        lidar_filename = os.path.join(self.data_root, self.token_to_lidar_path[occ_filename])
        if (not os.path.exists(os.path.join(self.occ_path, occ_filename+'.npy'))) or (not os.path.exists(lidar_filename)):
            if (not os.path.exists(os.path.join(self.occ_path, occ_filename+'.npy'))):
                print(f'occ not found !!! {occ_filename}')
            elif (not os.path.exists(lidar_filename)):
                print(f'lidar not found !!! {lidar_filename}')
            else:
                print(f'???')
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

        if self.random_choice_lidar and self.training:
            choice_n_lidar = random.randint(1, NUPLAN_LIDAR_NUM)
            lidar_chosen = random.sample(list(range(NUPLAN_LIDAR_NUM)), choice_n_lidar)
            lidar_chosen_mask = np.zeros((NUPLAN_LIDAR_NUM,), dtype=bool)
            lidar_chosen_mask[lidar_chosen] = True
            data_dict['lidar_chosen_mask'] = lidar_chosen_mask
        else:
            lidar_chosen = self.dataset_cfg.get('lidar_chosen', '0,1,2,3,4')
            lidar_chosen = lidar_chosen.split(',')
            lidar_chosen = list(map(int, lidar_chosen))
            lidar_chosen_mask = np.zeros((NUPLAN_LIDAR_NUM,), dtype=bool)
            lidar_chosen_mask[lidar_chosen] = True
            data_dict['lidar_chosen_mask'] = lidar_chosen_mask
        return data_dict

    def load_nuscenes_laserscan(self, file, lidar_range = None, poses=None, times=None, cur_time=None, cur_pose=None):
        #import open3d as o3d
        #raw = np.fromfile(file, dtype=np.float32)
        #points = raw.reshape((-1, 5))


        points_pcd = pypcd.PointCloud.from_path(file)
        x = points_pcd.pc_data['x']
        y = points_pcd.pc_data['y']
        z = points_pcd.pc_data['z']
        intensity = points_pcd.pc_data['intensity']
        ring = points_pcd.pc_data['ring']
        lidar_info = points_pcd.pc_data['lidar_info']
        points = np.stack([x, y, z, intensity, ring, lidar_info], axis=1)
        
        if self.top_lidar_only:
            top_mask = (lidar_info == 0)
            points = points[top_mask]

        points[..., 3] = points[..., 3] / MAX_RELECTANCE_VALUE
        #pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:,0:3]))
        #o3d.visualization.draw_geometries([pc])
        #points = points[:, :3]
        # # todo: 归一化
        # result[:, 0] = (result[:, 0] - lidar_range[0])/(lidar_range[3]- lidar_range[0])
        # result[:, 1] = (result[:, 1] - lidar_range[1])/(lidar_range[4]- lidar_range[1])
        # result[:, 2] = (result[:, 2] - lidar_range[2])/(lidar_range[5]- lidar_range[2])

        did_return = np.ones((points.shape[0],), dtype=bool)
        if self.compute_missing_points:
            point_cloud = torch.from_numpy(points)

            pc = point_cloud.clone()

            vis=False
            if vis:
                pc_ori = point_cloud.clone().numpy()[:, :3]
                pc_ori_color = np.zeros_like(pc_ori)
                pc_ori_color[:, 0] = 255.0
                pc_ori = np.concatenate([pc_ori, pc_ori_color], axis=1)

                pc_after = pc_without_ego_motion_comp.clone().numpy()[:, :3]
                pc_after_color = np.zeros_like(pc_after)
                pc_after_color[:, 1] = 255.0
                pc_after = np.concatenate([pc_after, pc_after_color], axis=1)
                np.concatenate([pc_ori, pc_after]).astype('float32').tofile('z.bin')
            
            # add missing points
            missing_points_all = self._get_missing_points(pc)
            for lidar_idx, missing_points in enumerate(missing_points_all):
                missing_points_all[lidar_idx] = torch.from_numpy(missing_points_all[lidar_idx]).to(torch.float32)
                _n_points = missing_points_all[lidar_idx].shape[0]
                missing_points_all[lidar_idx] = torch.cat([missing_points_all[lidar_idx], torch.zeros((_n_points, 2),  dtype=torch.float32), torch.ones((_n_points, 1), dtype=torch.float32)*lidar_idx], dim=-1)
            missing_points = torch.cat(missing_points_all)

            # add missing points to point clouds
            points = torch.cat([point_cloud, missing_points], dim=0)
            did_return = torch.linalg.norm(points[:, :3], dim=1) < 1e3
            points, did_return = points.numpy(), did_return.numpy()

        return points, did_return, lidar_info

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            if 'calib' in data_dict:
                calib = data_dict['calib']
            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict
                }
            )
            if 'calib' in data_dict:
                data_dict['calib'] = calib
        data_dict = self.set_lidar_aug_matrix(data_dict)

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        return data_dict

    @staticmethod
    def collate_batch(batch_list):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        ret = {}
        ret['batch_size'] = len(batch_list)
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
            elif key in ['points_in_occ', 'grid']:
                ret[key] = val
            elif key in ['lidar_chosen_mask']:
                ret[key] = np.stack(val)
            elif key in ['frame_id', 'end_flag', 'tra']:
                ret[key] = val
        return ret
    
    def evaluation(self, dist_test=False, world_size=1, rank=0, tmpdir=None):
        if dist_test == False:
            # lidar-wise
            if isinstance(self.avg_chamfer, (defaultdict, dict)):
                ret_str = '\n'
                ret_dict = {}
                for k, v in self.avg_chamfer.items():
                    avg_chamfer = np.mean(v)
                    ret_str += f'avg chamfer dist ({k}): {avg_chamfer}\n' 
                    ret_dict[f'avg_chamfer_dist_{k}'] = avg_chamfer
                return ret_str, ret_dict
            else:
                avg_chamfer = np.mean(self.avg_chamfer)
                return f'avg chamfer dist: {avg_chamfer}', {'avg_chamfer_dist': {avg_chamfer}}
        else:
            os.makedirs(tmpdir, exist_ok=True)

            dist.barrier()
            pickle.dump(self.avg_chamfer, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
            dist.barrier()

            if rank != 0:
                return None, None

            part_list = []
            for i in range(world_size):
                part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
                part_list.append(pickle.load(open(part_file, 'rb')))

            # print(part_list)
            if isinstance(self.avg_chamfer, (defaultdict, dict)):
                gathered_res = defaultdict(list)
                for item in part_list:
                    for k, v in item.items():
                        gathered_res[k].extend(v)
                ret_str = '\n'
                ret_dict = {}
                for k, v in gathered_res.items():
                    avg_chamfer = np.mean(v)
                    ret_str += f'avg chamfer dist ({k}): {avg_chamfer}\n' 
                    ret_dict[f'avg_chamfer_dist_{k}'] = avg_chamfer
                shutil.rmtree(tmpdir)
                return ret_str, ret_dict
                
            else:
                ordered_results = []
                # for res in zip(*part_list):
                #     ordered_results.extend(list(res))
                for res in part_list:
                    ordered_results.extend(res)
                shutil.rmtree(tmpdir)
                avg_chamfer = np.mean(ordered_results)
            return f'avg chamfer dist: {avg_chamfer}', {'avg_chamfer_dist': {avg_chamfer}}

    def update_chamfer_distance(self, pred_pcd_list, gt_pcd_list, frame_ids=None, save_path=None, save_type='npy', lidar_wise=False):
        assert len(pred_pcd_list) == len(gt_pcd_list)
        chamfer_distance = ChamferDistance()
        chamfer_dist_all = 0
        if not hasattr(self, 'avg_chamfer'):
            if lidar_wise:
                self.avg_chamfer = defaultdict(list)
            else:
                self.avg_chamfer = []
        for bs, (pred_pcd, gt_pcd) in enumerate(zip(pred_pcd_list, gt_pcd_list)):
            if lidar_wise:
                lidar_idx = gt_pcd[:, -1].int()
                unique_lidar_ids = torch.unique(lidar_idx)
                for lidar_id in unique_lidar_ids:
                    lidar_mask = (lidar_idx == lidar_id)
                    pred_pcd_single = pred_pcd[lidar_mask, :3]
                    gt_pcd_single = gt_pcd[lidar_mask, :3]
                    with torch.no_grad():
                        cd = chamfer_distance(
                            pred_pcd_single[None, ...].detach(),
                            gt_pcd_single[None, ...],
                            bidirectional=True,
                            point_reduction='mean'
                            ) 
                    chamfer_dist_value = cd.item()
                    chamfer_dist_value = chamfer_dist_value / 2.0
                    self.avg_chamfer[f'lidar_{lidar_id}'].append(chamfer_dist_value)
            else:
                pred_pcd = pred_pcd[:, :3]
                gt_pcd = gt_pcd[:, :3]
                with torch.no_grad():
                    cd = chamfer_distance(
                        pred_pcd[None, ...].detach(),
                        gt_pcd[None, ...],
                        bidirectional=True,
                        point_reduction='mean'
                        )

                #chamfer_dist_value = (cd_forward / pred_pcd.shape[0]) + (cd_backward / gt_pcd.shape[0])
                chamfer_dist_value = cd.item()
                chamfer_dist_value = chamfer_dist_value / 2.0
                self.avg_chamfer.append(chamfer_dist_value)
                #chamfer_dist_all += chamfer_dist_value
            #return chamfer_dist_all / len(pred_pcd_list)
                #print(chamfer_dist_value)

            vis=False
            if vis:
                pred_pcd.detach().cpu().numpy().astype('float32').tofile(f'examples/unet_renderv2_ep16/{bs}.bin')
                
        if save_path is not None:
            pred_save_path = os.path.join(save_path, 'pred')
            gt_save_path = os.path.join(save_path, 'gt')
            if not os.path.exists(pred_save_path):
                os.makedirs(pred_save_path, exist_ok=True)
                os.makedirs(gt_save_path, exist_ok=True)
            for pred_pcd, frame_id in zip(pred_pcd_list, frame_ids):
                if save_type == 'npy':
                    np.save(os.path.join(pred_save_path, f'{frame_id}.npy'), pred_pcd.detach().cpu().numpy())
                elif save_type == 'bin':
                    pred_pcd.detach().cpu().numpy().astype('float32').tofile(os.path.join(pred_save_path, f'{frame_id}.bin'))
                else:
                    raise NotImplementedError
            # for gt_pcd, frame_id in zip(gt_pcd_list, frame_ids):
            #     if save_type == 'npy':
            #         np.save(os.path.join(gt_save_path, f'{frame_id}.npy'), gt_pcd.cpu().numpy())
            #     elif save_type == 'bin':
            #         gt_pcd.detach().cpu().numpy().astype('float32').tofile(os.path.join(gt_save_path, f'{frame_id}.bin'))
            #     else:
            #         raise NotImplementedError


    def _get_missing_points(
        self,
        points,
    ) -> torch.Tensor:
        n_lidar = len(NUPLAN_LIDAR_LOCS)
        max_rings = [len(NUPLAN_THETA_MAP[i]) for i in range(n_lidar)]
        

        missing_points_all, _ = get_missing_points(np.array(list(NUPLAN_LIDAR_LOCS.values())), points, list(range(n_lidar)), max_rings, tol=0.5)
        
        vis=False
        if vis:
            for i in range(n_lidar):
                missing_points = missing_points_all[i]
                o = NUPLAN_LIDAR_LOCS[i]
                thetas = (o[2] - missing_points[:, 2]) / np.linalg.norm(missing_points[:, :3] - o[:3], axis=1)
                thetas = -np.rad2deg(np.arcsin(thetas))
                phis = np.rad2deg(np.arctan2((missing_points[:, 1] - o[1]), (missing_points[:, 0] - o[0])))
                missing_polars = np.column_stack((phis, thetas))

                lidar_mask = (points[:,-1].to(torch.int64)==i)
                lidar_points = points[lidar_mask]

                xyz = (lidar_points)[:,:3]
                theta = (o[-1] - xyz[:, -1]) / torch.sqrt((xyz[:, 0]-o[0])**2 + (xyz[:, 1]-o[1])**2 + (xyz[:, -1]-o[-1])**2)
                theta = -torch.rad2deg(torch.asin(theta))

                phi = torch.rad2deg(torch.atan2((xyz[:, 1] - o[1]), (xyz[:, 0] - o[0])))

                plt.figure()
                plt.scatter(phi, theta, s=1)
                plt.scatter(missing_polars[:, 0], missing_polars[:, 1], s=1, c='r')
                plt.savefig(f'z_{i}.png')
        return missing_points_all
    

    @staticmethod
    def _remove_ego_motion_compensation(
        point_cloud: torch.Tensor, l2ws: torch.Tensor, times: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Removes ego motion compensation from point cloud.

        Args:
            point_cloud: Point cloud to remove ego motion compensation from (in world frame). Shape: [num_points, 5+N] x,y,z,intensity,timestamp,(channel_id)
            l2ws: Poses of the lidar. Shape: [num_poses, 4, 4]
            times: Timestamps of the lidar poses. Shape: [num_poses]

        Returns:
            Point cloud without ego motion compensation in sensor frame. Shape: [num_points, 5+N] x,y,z,intensity,timestamp,(channel_id)
            Lidar pose for each point in the point cloud. Shape: [num_points, 4, 4]
        """

        interpolated_l2ws, _, _ = interpolate_trajectories(
            l2ws.unsqueeze(1), times - times.min(), point_cloud[:, 4] - times.min(), clamp_frac=False
        )
        interpolated_l2ws = interpolated_l2ws[:, :3, :4]
        interpolated_w2ls = pose_utils.inverse(interpolated_l2ws)
        homogen_points = torch.cat([point_cloud[:, :3], torch.ones_like(point_cloud[:, -1:])], dim=-1)
        points = torch.matmul(interpolated_w2ls, homogen_points.unsqueeze(-1))[:, :, 0]
        return torch.cat([points, point_cloud[:, 3:]], dim=-1), interpolated_l2ws