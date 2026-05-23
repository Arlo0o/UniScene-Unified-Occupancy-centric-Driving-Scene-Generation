from typing import Tuple
from collections import defaultdict
import numpy as np
import os
from glob import glob
import json
import pickle
import shutil
from pyquaternion import Quaternion
import torch
import torch.distributed as dist
from chamferdist import ChamferDistance
from ..dataset import DatasetTemplate
from ..augmentor.data_augmentor_occ2lidar import DataAugmentorOcc2LiDAR
from ...utils.occ_lidar_match_utils import compute_pos_in_voxel
from .nuscenes_constants import (NUSCENES_ELEVATION_MAPPING, NUSCENES_SKIP_ELEVATION_CHANNELS, NUSCENES_AZIMUTH_RESOLUTION, 
                                 DUMMY_DISTANCE_VALUE, MAX_RELECTANCE_VALUE, LIDAR_CHANNELS, LIDAR_FREQUENCY)
from .utils.poses import interpolate_trajectories
from .utils import poses as pose_utils

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

class Occ2LiDARDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        lidar_root = dataset_cfg.lidar_root
        occ_root = dataset_cfg.occ_root
        occ_anno_file = dataset_cfg.occ_anno_file
        split_root = dataset_cfg.split_root
        self.n_points_in_voxel = dataset_cfg.N_POINTS_IN_VOXEL
        self.class_names = class_names
        self.point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.occ_size = [512, 512, 40]
        self.occ_voxel_size = [0.2, 0.2, 0.2]
        self.occ_root = occ_root
        full_list = []
        ## 选择lidar索引occ或者occ索引lidar
        #if self.training:
        full_list = glob( lidar_root+"*" )
            # /generator_tos/data/nuscenes/samples/LIDAR_TOP/n008-2018-09-18-14-54-39-0400__LIDAR_TOP__1537297544449565.pcd.bin
        
        self.full_list = full_list

        # train list or val list
        if self.mode == 'train':
            split_file = 'train.txt'
            info_file = 'data/openocc/nuscenes_infos_temporal_train_converted.pkl'
        elif self.mode == 'val' or self.mode == 'test':
            split_file = 'val.txt'
            info_file = 'data/openocc/nuscenes_infos_temporal_val_converted.pkl'
        else:
            raise NotImplementedError
        with open(os.path.join(split_root, split_file), 'r') as f:
            lines = f.readlines()
            lines = [s.strip() for s in lines]
        self.full_list = lines#[:100]

        self.length = len(self.full_list)
        with open(occ_anno_file, "r") as anno_json:
            self.sample_dict = json.load(anno_json)

        self.scale_xyz = self.occ_size[0] * self.occ_size[1] * self.occ_size[2]
        self.scale_yz = self.occ_size[1] * self.occ_size[2]
        self.scale_z = self.occ_size[2]

        del self.data_augmentor
        self.data_augmentor = DataAugmentorOcc2LiDAR(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None

        self.compute_missing_points = self.dataset_cfg.get('COMPUTE_MISSING_POINTS', False)# and self.training
        if self.compute_missing_points:
            self.load_infos(info_file)
    
    def __len__(self):
        return self.length
    
    def load_infos(self, info_file):
        with open(info_file, 'rb') as f:
            infos = pickle.load(f)

        self.token_to_seq_id_map = {}
        self.seq_id_to_info_map = {}
        self.seq_id_to_times = {}
        self.seq_id_to_poses = {}
        for i, item in enumerate(infos):
            tokens, seq_infos = list(zip(*item))
            self.token_to_seq_id_map.update(dict(zip(tokens, (i+np.zeros((len(tokens)), dtype=int)).tolist())))
            self.seq_id_to_info_map[i] = [{'token': item['token'], 'lidar_top_data_token': item['lidar_top_data_token'] if 'lidar_top_data_token' in item else None} for item in seq_infos]#seq_infos
            times = [info['timestamp']/1e6 for info in seq_infos]
            times = torch.tensor(times, dtype=torch.float64) 
            self.seq_id_to_times[i] = times

            poses = []
            for info in seq_infos:
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
            self.seq_id_to_poses[i] = poses

    def load_nuscenes_laserscan(self, file, lidar_range = None, poses=None, times=None, cur_time=None, cur_pose=None):
        #import open3d as o3d
        raw = np.fromfile(file, dtype=np.float32)
        points = raw.reshape((-1, 5))

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
            # nuscenes lidar time is the time of the end of the sweep. Add estimated times per point
            offsets = np.repeat(np.linspace(-1 / LIDAR_FREQUENCY, 0, points.shape[0] // LIDAR_CHANNELS), LIDAR_CHANNELS)
            # warning: here we overwrite the beam index channel with time offsets, since we assume x,y,z,r,t format
            channel_id = points[..., 4].astype(np.int32)
            points[..., 4] = offsets
            # concatenate pc with channel id
            points = np.concatenate([points, channel_id[:, None]], axis=-1)
            point_cloud = torch.from_numpy(points)

        
            # remove ego motion compensation
            
            #for point_cloud, l2w, time in zip(point_clouds, poses, times):
            pc = point_cloud.clone()
            # absolute time
            pc[:, 4] = pc[:, 4] + cur_time
            # project to world frame
            pc[..., :3] = pose_utils.transform_points(pc[..., :3], cur_pose.unsqueeze(0).to(pc))
            # remove ego motion compensation
            pc_without_ego_motion_comp, interpolated_poses = self._remove_ego_motion_compensation(pc, poses, times)
            # reset time
            pc_without_ego_motion_comp[:, 4] = point_cloud[:, 4].clone()
            # transform to common lidar frame again
            interpolated_poses = torch.matmul(
                pose_utils.inverse(cur_pose.unsqueeze(0)).float(), pose_utils.to4x4(interpolated_poses).float()
            )
            # move channel from index 5 to 3
            pc_without_ego_motion_comp = pc_without_ego_motion_comp[..., [0, 1, 2, 5, 3, 4]]

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
            missing_points = self._get_missing_points(pc_without_ego_motion_comp, interpolated_poses, "LIDAR_TOP", dist_cutoff=0.05)
            missing_points = missing_points[:, [0, 1, 2, 4, 5]]

            # add missing points to point clouds
            points = torch.cat([point_cloud[:, :5], missing_points], dim=0)
            did_return = torch.linalg.norm(points[:, :3], dim=1) < 1e3
            points, did_return = points.numpy(), did_return.numpy()

        return points, did_return

    def __getitem__(self, idx):
        input_dict = {}
        lidar_filename = self.full_list[idx]
        #lidar_filename = self.full_list[0]
        input_dict['frame_id'] = lidar_filename.split('/')[-1].split('.')[0]

        occ_path = self.sample_dict.get(  os.path.join(*lidar_filename.split("/")[-3:]), "None"  )
        #print( occ_path )
        occ_path_out = self.occ_root + "scene_"+ occ_path.split("/")[0] +"/occupancy/" + occ_path.split("/")[1] + ".npy"
        #print( occ_path_out )
        occ = np.load(occ_path_out, encoding='bytes', allow_pickle=True)

        if self.compute_missing_points:
            cur_token = occ_path.split('/')[1]
            seq_id = self.token_to_seq_id_map[occ_path.split('/')[1]]
            seq_infos = self.seq_id_to_info_map[seq_id]
            #tokens = [info['token'] for info in seq_infos]
            tokens = [info['lidar_top_data_token'] for info in seq_infos]
            times = self.seq_id_to_times[seq_id]
            poses = self.seq_id_to_poses[seq_id]
            cur_time = times[tokens.index(cur_token)]
            cur_pose = poses[tokens.index(cur_token)]
            lidar, did_return = self.load_nuscenes_laserscan(lidar_filename, lidar_range = self.point_cloud_range, poses=poses, times=times, cur_time=cur_time, cur_pose=cur_pose)
        else:
            lidar, did_return = self.load_nuscenes_laserscan(lidar_filename, lidar_range = self.point_cloud_range)
        
        input_dict['points'] = lidar
        input_dict['did_return'] = did_return
        # to xyz(absolute coords) for data augmentor
        input_dict['occ'] = occ[:, [2, 1, 0, 3]].astype(lidar.dtype)
        voxel_size = np.array(self.voxel_size).reshape((-1, 3))
        pc_range = np.array(self.point_cloud_range[:3]).reshape((-1, 3))
        input_dict['occ'][:, :3] = (input_dict['occ'][:, :3] + 0.5) * voxel_size + pc_range

        vis = False
        if vis:
            rad = np.zeros((input_dict['points'].shape[0], 3))
            rad[:, 0] = 255
            white = 255 * np.ones((input_dict['occ'].shape[0], 3))
            for_vis = np.concatenate([np.concatenate([input_dict['points'][:, :3], rad], axis=-1), np.concatenate([input_dict['occ'][:, :3], white], axis=-1)], axis=0)
            for_vis.astype('float32').tofile('z.bin')

        # 数据增强会有潜在的bug，occ本质上是离散的，数据增强中的旋转缩放平移等操作会破坏其原本的规则分布（因为后面要进行体素化）
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

        # occ-lidar match
        if self.dataset_cfg.get('COMPUTE_OCC_LIDAR_MATCH', True):
            points = data_dict['points'][:, :3]
            #occ_loc = data_dict['occ'][:, [2, 1, 0]]
            #data_dict['points_in_occ'] = self.assign_points_to_voxels(torch.from_numpy(points), torch.from_numpy(occ_loc).int(), torch.tensor(self.voxel_size), torch.tensor(self.point_cloud_range), self.n_points_in_voxel)
            
            occ_loc = data_dict['occ'][:, [7, 8, 9]]
            data_dict['points_in_occ'] = self.assign_points_to_voxels(torch.from_numpy(points), torch.from_numpy(occ_loc), torch.tensor(self.voxel_size), torch.tensor(self.point_cloud_range), self.n_points_in_voxel)


        vis = False
        if vis:
            C = data_dict['points_in_occ']
            choices = np.random.randint(0, C.shape[0], (1000,))
            choiced_pts = C[choices]
            choiced_pts = choiced_pts[(choiced_pts!=0).any(-1)][:, :3]
            choiced_occ = data_dict['occ'][choices][:, 4:7]
            rad = np.zeros((choiced_pts.shape[0], 3))
            rad[:, 0] = 255
            white = 255 * np.ones((choiced_occ.shape[0], 3))
            for_vis = np.concatenate([np.concatenate([choiced_pts, rad], axis=-1), np.concatenate([choiced_occ, white], axis=-1)], axis=0)
            for_vis.astype('float32').tofile('z.bin')
            
        xyz = data_dict['occ'][:, [2, 1, 0]].astype(np.int32)
        data_dict['grid'] = np.zeros(self.occ_size, dtype=bool)
        data_dict['grid'][xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
        data_dict['grid'] = torch.from_numpy(data_dict['grid'])

        return data_dict

    def assign_points_to_voxels(self, A, B, voxel_size, point_cloud_range, K):
        """
        :param A: 点云张量，大小为 [N, 3]
        :param B: 体素坐标张量，大小为 [M, 3]
        :param voxel_size: 体素大小，标量或向量 [3]
        :param point_cloud_range: 点云的范围，向量 [6] (min_x, min_y, min_z, max_x, max_y, max_z)
        :param K: 每个体素中的最大点数
        :return: 大小为 [M, K, 3] 的张量，表示每个体素中的点
        """
        N = A.size(0)  # 点云中点的数量
        M = B.size(0)  # 体素的数量
        
        # 计算点云坐标的体素索引 (global voxel coordinates)
        voxel_indices = ((A - point_cloud_range[:3]) / voxel_size).int()  # [N, 3]
        
        # 创建一个tensor来存储每个体素中的点，初始化为0
        C = torch.zeros((M, K, 3))

        num_vertical = 32     # Number of vertical steps
        vertical_angles = np.linspace(np.deg2rad(-30), np.deg2rad(10), num_vertical)

        mask = (B[:, 1] - (np.pi/2))[:, None] - vertical_angles[None, :]
        mask = mask.abs().min(-1)[0] < np.deg2rad(0.1)

        C[mask] = True
        return C

    # def assign_points_to_voxels(self, A, B, voxel_size, point_cloud_range, K):
    #     """
    #     :param A: 点云张量，大小为 [N, 3]
    #     :param B: 体素坐标张量，大小为 [M, 3]
    #     :param voxel_size: 体素大小，标量或向量 [3]
    #     :param point_cloud_range: 点云的范围，向量 [6] (min_x, min_y, min_z, max_x, max_y, max_z)
    #     :param K: 每个体素中的最大点数
    #     :return: 大小为 [M, K, 3] 的张量，表示每个体素中的点
    #     """
    #     N = A.size(0)  # 点云中点的数量
    #     M = B.size(0)  # 体素的数量
        
    #     # 计算点云坐标的体素索引 (global voxel coordinates)
    #     voxel_indices = ((A - point_cloud_range[:3]) / voxel_size).int()  # [N, 3]
        
    #     # 创建一个tensor来存储每个体素中的点，初始化为0
    #     C = torch.zeros((M, K, 3))

    #     voxel_indices_flatten = voxel_indices[:, 0] * self.scale_yz + \
    #                             voxel_indices[:, 1] * self.scale_z + \
    #                             voxel_indices[:, 2]
    #     B_flatten = B[:, 0] * self.scale_yz + \
    #                 B[:, 1] * self.scale_z + \
    #                 B[:, 2]

    #     # 将体素索引和体素坐标B比较，找出点属于哪个体素
    #     #voxel_match = (voxel_indices_flatten.unsqueeze(1) == B_flatten.unsqueeze(0))  # [N, M]

    #     # Sort B and get the sorted indices
    #     sorted_B, sorted_indices_B = torch.sort(B_flatten)

    #     # Find positions where elements of A would be inserted in sorted_B
    #     positions = torch.searchsorted(sorted_B, voxel_indices_flatten)

    #     # Check if the elements at these positions actually match A
    #     matches = torch.zeros((positions.shape[0],), dtype=bool)
    #     pos_mask = (positions < len(B))
    #     matches[pos_mask] = (sorted_B[positions[pos_mask]] == voxel_indices_flatten[pos_mask])

    #     # Get the indices in B corresponding to matches
    #     indices_in_B = sorted_indices_B[positions[matches]]

    #     # Get the indices in A corresponding to matches
    #     indices_in_A = torch.nonzero(matches, as_tuple=True)[0]

    #     # Stack the indices to form the index pair array C
    #     matched_idx = torch.stack([indices_in_A, indices_in_B], dim=1)

    #     point_idx = matched_idx[:, 0]
    #     voxel_idx = matched_idx[:, 1]
        
    #     pos_in_voxel = compute_pos_in_voxel(voxel_idx)
    #     index_mask = pos_in_voxel < self.n_points_in_voxel
    #     voxel_idx = voxel_idx[index_mask]
    #     pos_in_voxel = pos_in_voxel[index_mask]
    #     point_idx = point_idx[index_mask]
        
    #     C[voxel_idx, pos_in_voxel, :] = A[point_idx, :]
    #     return C

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
            if key in ['points', 'occ', 'did_return']:
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
            elif key in ['frame_id', 'end_flag', 'tra']:
                ret[key] = val
        return ret
    
    def evaluation(self, dist_test=False, world_size=1, rank=0, tmpdir=None):
        if dist_test == False:
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

            ordered_results = []
            for res in zip(*part_list):
                ordered_results.extend(list(res))
            shutil.rmtree(tmpdir)
            avg_chamfer = np.mean(ordered_results)
            return f'avg chamfer dist: {avg_chamfer}', {'avg_chamfer_dist': {avg_chamfer}}

    def update_chamfer_distance(self, pred_pcd_list, gt_pcd_list, frame_ids=None, save_path=None, save_type='npy'):
        assert len(pred_pcd_list) == len(gt_pcd_list)
        chamfer_distance = ChamferDistance()
        chamfer_dist_all = 0
        if not hasattr(self, 'avg_chamfer'):
            self.avg_chamfer = []
        for bs, (pred_pcd, gt_pcd) in enumerate(zip(pred_pcd_list, gt_pcd_list)):
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
            for gt_pcd, frame_id in zip(gt_pcd_list, frame_ids):
                if save_type == 'npy':
                    np.save(os.path.join(gt_save_path, f'{frame_id}.npy'), gt_pcd.cpu().numpy())
                elif save_type == 'bin':
                    gt_pcd.detach().cpu().numpy().astype('float32').tofile(os.path.join(gt_save_path, f'{frame_id}.bin'))
                else:
                    raise NotImplementedError


    def _get_missing_points(
        self,
        point_cloud: torch.Tensor,
        l2ws: torch.Tensor,
        lidar_name: str,
        dist_cutoff: float = 1.0,
        ignore_regions: list = [],
        outlier_thresh: float = 0.2,
        lidar_elevation_mapping = NUSCENES_ELEVATION_MAPPING,
        skip_elevation_channels = NUSCENES_SKIP_ELEVATION_CHANNELS,
        lidar_azimuth_resolution = NUSCENES_AZIMUTH_RESOLUTION
    ) -> torch.Tensor:
        """Finds missing points in the point cloud according to sensor spec (self.config.lidar_elevation_mapping)

        Args:
            point_cloud: Point cloud to find missing points in (in sensor frame). Shape: [num_points, 4+x] x,y,z,channel_id(timestamp, intensity, etc.)
            l2ws: Poses of the lidar. Shape: [num_points, 4, 4]
            lidar_name: Name of the lidar
            dist_cutoff: Distance cutoff for points to consider. Points closer than this will be ignored.
            ignore_regions: List of regions to ignore. Each region is a list of [min_azimuth, max_azimuth, min_elevation, max_elevation]
            outlier_thresh: Threshold for outlier elevation values. If the median elevation of the missing points is more than this value away from the median elevation of the points in the channel, we ignore the missing points.

        Returns:
            Missing points in the point cloud, in world_frame. Shape: [num_points, 3+x] x,y,z,(timestamp, intensity, etc.)
        """
        dist = torch.norm(point_cloud[:, :3], dim=-1)
        dist_mask = dist > dist_cutoff
        dist = dist[dist_mask]
        point_cloud = point_cloud[dist_mask]
        l2ws = l2ws[dist_mask]
        elevation = torch.arcsin(point_cloud[:, 2] / dist)
        elevation = torch.rad2deg(elevation)
        azimuth = torch.atan2(point_cloud[:, 1], point_cloud[:, 0])
        azimuth = torch.rad2deg(azimuth)

        # find missing points
        missing_points = []
        missing_points_sensor_frame = []
        assert lidar_elevation_mapping is not None, "Must specify lidar elevation mapping"
        assert lidar_azimuth_resolution is not None, "Must specify lidar azimuth resolution"
        assert skip_elevation_channels is not None, "Must specify skip elevation channels"
        for channel_id, expected_elevation in lidar_elevation_mapping[lidar_name].items():
            if channel_id in skip_elevation_channels[lidar_name]:
                continue
            channel_mask = (point_cloud[:, 3] - channel_id).abs() < 0.1  # handle floats
            curr_azimuth = azimuth[channel_mask]
            curr_l2ws = l2ws[channel_mask]
            curr_elev = elevation[channel_mask]
            if not channel_mask.any():
                continue
            curr_azimuth, sort_idx = curr_azimuth.sort()
            curr_l2ws = curr_l2ws[sort_idx]
            curr_elev = curr_elev[sort_idx]

            # find missing azimuths, we should have 360 / lidar_azimuth_resolution azimuths
            num_expected_azimuths = int(360 / lidar_azimuth_resolution[lidar_name]) + 1
            expected_idx = torch.arange(num_expected_azimuths, device=curr_azimuth.device)
            # find offset
            offset = curr_azimuth[0] % lidar_azimuth_resolution[lidar_name]
            current_idx = (
                ((curr_azimuth - offset + 180) / lidar_azimuth_resolution[lidar_name]).round().int()
            )
            missing_idx = expected_idx[torch.isin(expected_idx, current_idx, invert=True)]
            # interpolate missing azimuths
            missing_azimuth = (
                torch.from_numpy(
                    np.interp(
                        missing_idx,
                        torch.cat([torch.tensor(-1).view(1), current_idx, torch.tensor(num_expected_azimuths).view(1)]),
                        torch.cat(
                            [
                                (-180 + offset - lidar_azimuth_resolution[lidar_name]).view(1),
                                curr_azimuth,
                                (180 + offset + lidar_azimuth_resolution[lidar_name]).view(1),
                            ]
                        ),
                    )
                )
                .float()
                .view(-1, 1)
            )
            missing_elevation = (
                torch.from_numpy(
                    np.interp(
                        missing_idx,
                        torch.cat([torch.tensor(-1).view(1), current_idx, torch.tensor(num_expected_azimuths).view(1)]),
                        torch.cat(
                            [
                                torch.tensor(expected_elevation).view(1),
                                curr_elev.view(-1),
                                torch.tensor(expected_elevation).view(1),
                            ]
                        ),
                    )
                )
                .float()
                .view(-1, 1)
            )
            elevation_outlier_mask = (missing_elevation - curr_elev.median()).abs() > outlier_thresh
            missing_elevation[elevation_outlier_mask] = curr_elev.median()

            ignore_mask = torch.zeros_like(missing_azimuth.squeeze(-1)).bool()
            for ignore_region in ignore_regions:
                ignore_mask = (
                    (missing_azimuth > ignore_region[0])
                    & (missing_azimuth < ignore_region[1])
                    & (missing_elevation > ignore_region[2])
                    & (missing_elevation < ignore_region[3])
                ).squeeze(-1) | ignore_mask
            missing_azimuth = missing_azimuth[~ignore_mask]
            missing_elevation = missing_elevation[~ignore_mask]

            # missing_elevation = torch.ones_like(missing_azimuth) * current_elevation
            missing_distance = torch.ones_like(missing_azimuth) * DUMMY_DISTANCE_VALUE
            x = (
                torch.cos(torch.deg2rad(missing_azimuth))
                * torch.cos(torch.deg2rad(missing_elevation))
                * missing_distance
            )
            y = (
                torch.sin(torch.deg2rad(missing_azimuth))
                * torch.cos(torch.deg2rad(missing_elevation))
                * missing_distance
            )
            z = torch.sin(torch.deg2rad(missing_elevation)) * missing_distance

            # for debug
            # missing_distance = torch.arange(0, 20, 0.1).to(missing_azimuth.dtype).unsqueeze(0).expand(missing_azimuth.shape[0], -1).unsqueeze(-1)
            # n_dist = missing_distance.shape[1]
            # x = (
            #     torch.cos(torch.deg2rad(missing_azimuth)).unsqueeze(1)
            #     * torch.cos(torch.deg2rad(missing_elevation)).unsqueeze(1)
            #     * missing_distance
            # ).view(-1, 1)
            # y = (
            #     torch.sin(torch.deg2rad(missing_azimuth)).unsqueeze(1)
            #     * torch.cos(torch.deg2rad(missing_elevation)).unsqueeze(1)
            #     * missing_distance
            # ).view(-1, 1)
            # z = (torch.sin(torch.deg2rad(missing_elevation)).unsqueeze(1) * missing_distance).view(-1, 1)
            # missing_azimuth = missing_azimuth.repeat_interleave(n_dist, dim=0)

            points = torch.cat([x, y, z], dim=-1)
            missing_points_sensor_frame.append(points)
            # transform points from sensor space to world space
            points = torch.cat([points, torch.ones_like(points[:, -1:])], dim=-1)
            # find closest pose idx
            closest_pose_idx = torch.searchsorted(curr_azimuth, missing_azimuth.squeeze())
            closest_pose_idx = closest_pose_idx.clamp(max=len(curr_azimuth) - 1)
            if closest_pose_idx.shape == ():
                closest_pose_idx = closest_pose_idx.unsqueeze(0)
            closest_l2w = curr_l2ws[closest_pose_idx]
            points = torch.matmul(closest_l2w.float(), points.unsqueeze(-1))[:, :, 0]
            closest_pc_values = point_cloud[channel_mask][sort_idx][closest_pose_idx, 3:]
            points = torch.cat([points, closest_pc_values], dim=-1)
            missing_points.append(points)

        missing_points = torch.cat(missing_points, dim=0)
        missing_points_sensor_frame = torch.cat(missing_points_sensor_frame, dim=0)
        return missing_points
    

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