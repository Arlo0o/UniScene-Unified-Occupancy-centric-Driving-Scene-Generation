from collections import defaultdict
import numpy as np
import numba as nb
import copy
import os
from glob import glob
import json
import pickle
from pyquaternion import Quaternion
import torch
from chamferdist import ChamferDistance
from ..dataset import DatasetTemplate
from ..augmentor.data_augmentor_occ2lidar import DataAugmentorOcc2LiDAR
from ...utils.occ_lidar_match_utils import compute_pos_in_voxel
from .occ_lidar import Occ2LiDARDataset
from .nuscenes_constants import (NUSCENES_ELEVATION_MAPPING, NUSCENES_SKIP_ELEVATION_CHANNELS, NUSCENES_AZIMUTH_RESOLUTION, 
                                 DUMMY_DISTANCE_VALUE, MAX_RELECTANCE_VALUE, LIDAR_CHANNELS, LIDAR_FREQUENCY)
from .utils import poses as pose_utils

def voxel2world(voxel,
                voxel_size=np.array([0.125, 0.125, 0.125]),
                pc_range=np.array([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0])):
    """
    voxel: [N, 3]
    """
    return voxel * voxel_size[None, :] + pc_range[:3][None, :]


def world2voxel(wolrd,
                voxel_size=np.array([0.2, 0.2, 0.2]),
                pc_range=np.array([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0])):
    """
    wolrd: [N, 3]
    """
    return (wolrd - pc_range[:3][None, :]) / voxel_size[None, :]

# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)

    return processed_label

def load_occ_gt(occ_path,  ## 原始分辨率800*800*64, 调整grid_size进行下采样
                grid_size=np.array([500, 500, 40 ]), # 200, 200, 16
                unoccupied=0,
                resample=False):
    #  [z y x cls] or [z y x vx vy vz cls]
    pcd = np.load( occ_path, encoding='bytes', allow_pickle=True)
    if not resample:
        return pcd
    pcd_label = pcd[..., -1:]
    pcd_label[pcd_label == 0] = 255
    pcd_np_cor = voxel2world(pcd[..., [0, 1, 2]] + 0.5)  # x y z
    untransformed_occ = copy.deepcopy(pcd_np_cor)  # N 4

    # bevdet augmentation
    # pcd_np_cor = (results['bda_mat'] @ torch.from_numpy(pcd_np_cor).unsqueeze(-1).float()).squeeze(-1).numpy()
    pcd_np_cor = world2voxel(pcd_np_cor)

    # make sure the point is in the grid
    pcd_np_cor = np.clip(pcd_np_cor, np.array([0, 0, 0]), grid_size - 1)
    transformed_occ = copy.deepcopy(pcd_np_cor)
    pcd_np = np.concatenate([pcd_np_cor, pcd_label], axis=-1)

    # 255: noise, 1-16 normal classes, 0 unoccupied
    pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
    pcd_np = pcd_np.astype(np.int64)
    processed_label = np.ones(grid_size, dtype=np.uint8) * unoccupied
    processed_label = nb_process_label(processed_label, pcd_np)

    noise_mask = processed_label == 255
    processed_label[noise_mask] = 0
    return processed_label

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

class Occ2LiDARDatasetNKSRCustomVis(Occ2LiDARDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        DatasetTemplate.__init__(self,
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.lidar_root = dataset_cfg.lidar_root
        occ_root = dataset_cfg.occ_root
        occ_anno_file = dataset_cfg.occ_anno_file
        split_root = dataset_cfg.split_root
        self.n_points_in_voxel = dataset_cfg.N_POINTS_IN_VOXEL
        self.class_names = class_names
        self.point_cloud_range = dataset_cfg.POINT_CLOUD_RANGE
        self.occ_size = dataset_cfg.GRID_SIZE
        self.occ_voxel_size = dataset_cfg.VOXEL_SIZE
        self.occ_root = occ_root
        #full_list = []
        ## 选择lidar索引occ或者occ索引lidar
        #if self.training:
        #full_list = glob( lidar_root+"*" )
            # /generator_tos/data/nuscenes/samples/LIDAR_TOP/n008-2018-09-18-14-54-39-0400__LIDAR_TOP__1537297544449565.pcd.bin
        
        # self.full_list = full_list
        self.full_list = glob('vis_1119/1119_demo/*/*/*.npz')

        # train list or val list
        if self.mode == 'train':
            split_file = 'nuScenes_nksr_occ_train.json'
            info_file = 'data/openocc/nuscenes_advanced_12Hz_infos_training_converted.pkl'
        elif self.mode == 'val' or self.mode == 'test':
            split_file = 'nuScenes_nksr_occ_val.json'
            info_file = 'data/openocc/nuscenes_advanced_12Hz_infos_validation_converted.pkl'
        else:
            raise NotImplementedError
        # with open(os.path.join(split_root, split_file), 'r') as split_json:
        #     self.full_list = json.load(split_json)

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

    def __getitem__(self, idx):
        input_dict = {}
        occ_filename = self.full_list[idx]
        self.cur_occ_file_name = occ_filename
        #occ_filename = self.full_list[str(0)]
        input_dict['frame_id'] = '-'.join(occ_filename.split('/')[-3:])
        occ = np.load(occ_filename)['occ']
        # with open(occ_filename, 'rb') as f:
        #     occ = load_occ_gt(f, grid_size=np.array(self.occ_size) )
        #     occ[occ==255] = 0
        occ_loc = np.stack(occ.nonzero(), axis=-1)[:, [2, 1, 0]]
        occ = np.concatenate([occ_loc, occ[occ_loc[:, 2], occ_loc[:, 1], occ_loc[:, 0]][:, None]], axis=-1)
        # !!! 没有用到 随便弄一个
        lidar_filename = os.path.join(self.lidar_root ,'samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin')

        if self.compute_missing_points:
            cur_token = occ_filename.split('/')[0]
            seq_id = self.token_to_seq_id_map[occ_filename.split('/')[0]]
            seq_infos = self.seq_id_to_info_map[seq_id]
            tokens = [info['token'] for info in seq_infos]
            times = self.seq_id_to_times[seq_id]
            poses = self.seq_id_to_poses[seq_id]
            cur_time = times[tokens.index(cur_token)]
            cur_pose = poses[tokens.index(cur_token)]
            lidar, did_return = self.load_nuscenes_laserscan(lidar_filename, lidar_range = self.point_cloud_range, poses=poses, times=times, cur_time=cur_time, cur_pose=cur_pose)
        else:
            lidar, did_return = self.load_nuscenes_laserscan(lidar_filename, lidar_range = self.point_cloud_range)
        
        # occ_path = self.sample_dict.get(  os.path.join(*lidar_filename.split("/")[-3:]), "None"  )
        # #print( occ_path )
        # occ_path_out = self.occ_root + "scene_"+ occ_path.split("/")[0] +"/occupancy/" + occ_path.split("/")[1] + ".npy"
        # #print( occ_path_out )
        # occ = np.load(occ_path_out, encoding='bytes', allow_pickle=True)
        
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
        if self.dataset_cfg.get('COMPUTE_OCC_LIDAR_MATCH', False):
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

    def update_chamfer_distance(self, pred_pcd_list, gt_pcd_list, frame_ids=None, save_path=None, save_type='bin'):
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

        save_path = os.path.join('/'.join(self.cur_occ_file_name.split('/')[:2]))
        if save_path is not None:
            #pred_save_path = os.path.join(save_path, 'pred')
            pred_save_path = save_path
            #gt_save_path = os.path.join(save_path, 'gt')
            
            if not os.path.exists(pred_save_path):
                os.makedirs(pred_save_path, exist_ok=True)
                #os.makedirs(gt_save_path, exist_ok=True)
            for pred_pcd, frame_id in zip(pred_pcd_list, frame_ids):
                #subfolder, frame_id = frame_id.split('-')
                subfolder, token = '/'.join(frame_id.split('-')[:-1]), frame_id.split('-')[-1].split('.')[0]+'_lidar'
                cur_pred_save_path = os.path.join(pred_save_path, subfolder)
                os.makedirs(cur_pred_save_path, exist_ok=True)
                if save_type == 'npy':
                    np.save(os.path.join(cur_pred_save_path, f'{token}.npy'), pred_pcd.detach().cpu().numpy())
                elif save_type == 'bin':
                    pred_pcd.detach().cpu().numpy().astype('float32').tofile(os.path.join(cur_pred_save_path, f'{token}.bin'))
                else:
                    raise NotImplementedError
            # for gt_pcd, frame_id in zip(gt_pcd_list, frame_ids):
            #     if save_type == 'npy':
            #         np.save(os.path.join(gt_save_path, f'{frame_id}.npy'), gt_pcd.cpu().numpy())
            #     elif save_type == 'bin':
            #         gt_pcd.detach().cpu().numpy().astype('float32').tofile(os.path.join(gt_save_path, f'{frame_id}.bin'))
            #     else:
            #         raise NotImplementedError