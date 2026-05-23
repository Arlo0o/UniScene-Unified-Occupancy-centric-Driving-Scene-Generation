from collections import defaultdict
import numpy as np
import numba as nb
import copy
import os
from glob import glob
from natsort import natsorted
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


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid

def draw_return(
    voxels,          # semantic occupancy predictions
    pred_pts,        # lidarseg predictions
    vox_origin,
    voxel_size=0.2,  # voxel size in the real world
    grid=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    save_dir=None,
    cam_positions=None,
    focal_positions=None,
    timestamp=None,
    mode=0,
    sem=False,
):
    w, h, z = voxels.shape

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    if mode == 0:
        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    elif mode == 1:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        pred_pts = pred_pts[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, pred_pts.reshape(-1)]).T
    elif mode == 2:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        gt_label = pt_label[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, gt_label.reshape(-1)]).T
    else:
        raise NotImplementedError

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        # (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 17)
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 17)
    ]
    print(len(fov_voxels))
    
    # import pdb; pdb.set_trace()
    # fig = plt.figure(figsize=(6,6))
    # ax = fig.add_subplot(111,projection='3d')

    voxel_size = sum(voxel_size) / 3
    colors = np.array(
        [
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 150, 245, 255],       # car                  blue
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255,   0,   0, 255],       # pedestrian           red
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red  # add
            [139, 137, 137, 255],
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
            # [  0, 255, 127, 255],       # ego car              dark cyan
            # [255,  99,  71, 255],       # ego car
            # [  0, 191, 255, 255]        # ego car
            [0, 0,0,0], #17 empty
            [  0, 128, 0, 255],       # 18             
            [128,  0,  0, 255],       # 19
            [  0, 128, 128, 255],       # 20
            [0, 255, 0, 255],        # 21
        ]
    ).astype(np.uint8)

    # colors = np.array( # occ waymo vis
    #     [
    #         [0,   0,   0, 255],  # 0 undefined
    #         [255, 158, 0, 255],  # 1 car  orange
    #         [0, 0, 230, 255],    # 2 pedestrian  Blue
    #         [47, 79, 79, 255],   # 3 sign  Darkslategrey
    #         [220, 20, 60, 255],  # 4 CYCLIST  Crimson
    #         [255, 69, 0, 255],   # 5 traiffic_light  Orangered
    #         [255, 140, 0, 255],  # 6 pole  Darkorange
    #         [233, 150, 70, 255], # 7 construction_cone  Darksalmon
    #         [255, 61, 99, 255],  # 8 bycycle  Red
    #         [112, 128, 144, 255],# 9 motorcycle  Slategrey
    #         [222, 184, 135, 255],# 10 building Burlywood
    #         [0, 175, 0, 255],    # 11 vegetation  Green
    #         [165, 42, 42, 255],  # 12 trunk  nuTonomy green
    #         [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
    #         [75, 0, 75, 255], # 14 walkable, sidewalk
    #         [255, 0, 0, 255], # 15 unobsrvd
    #         [128, 128, 128, 255], # 16 for vis
    #     ]
    # ).astype(np.uint8)

    # print(fov_voxels[:, 3])
    p_colors=colors[fov_voxels[:, 3].astype(np.uint8)-1]/255
    # p_colors=colors[fov_voxels[:, 3].astype(np.uint8)]/255

    return fov_voxels,p_colors

def trans_semantic_asNus(voxel_label,mapping_dict):
    voxel_label_nus = np.zeros_like(voxel_label,dtype=int)
    for key in mapping_dict.keys():
        voxel_label_nus[voxel_label==key] = mapping_dict[key]
    return voxel_label_nus

class WaymoOccDataset(Occ2LiDARDataset):
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

        # train list or val list
        if self.mode == 'train':
            split_file = 'nuScenes_nksr_occ_train.json'
            info_file = 'data/openocc/nuscenes_advanced_12Hz_infos_training_converted.pkl'
        elif self.mode == 'val' or self.mode == 'test':
            split_file = 'nuScenes_nksr_occ_val.json'
            info_file = 'data/openocc/nuscenes_advanced_12Hz_infos_validation_converted.pkl'
        else:
            raise NotImplementedError
        with open(os.path.join(split_root, split_file), 'r') as split_json:
            self.full_list = json.load(split_json)

        if 'SAMPLE_GAP' in dataset_cfg:
            sample_gap = dataset_cfg.SAMPLE_GAP
            temp = list(zip(self.full_list.keys(), self.full_list.values()))
            downsampled = temp[::sample_gap]
            ks, vs = zip(*downsampled)
            ks = list(map(str, list(range(len(ks)))))
            self.full_list = dict(zip(ks, vs))


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

            
        self.lidar_height = self.dataset_cfg['LIDAR_HEIGHT']
        self.full_list = natsorted(os.listdir('data/occ_waymo/026'))
        self.length = len(self.full_list)
        self.data_root = 'data/occ_waymo/026'

    def __getitem__(self, idx):
        input_dict = {}
        occ_filename = self.full_list[idx]
        #occ_filename = '100_04.npz'
        #occ_filename = self.full_list[str(0)]
        # input_dict['frame_id'] = '-'.join(occ_filename.split('/'))
        # with open(self.occ_root + "/" + occ_filename+ ".npy", 'rb') as f:
        #     occ = load_occ_gt(f, grid_size=np.array(self.occ_size) )
        #     occ[occ==255] = 0

        input_dict['frame_id'] = occ_filename.split('.')[0]
        file = os.path.join(self.data_root, occ_filename)
        data  = np.load(file)
        voxel_label = data['voxel_label']
        lidar_mask = data['origin_voxel_state']
        camera_mask = data['final_voxel_state']
        infov = data['infov']
        ego2global = data['ego2global']
        mapping_dict ={0:0,1:4,2:7,3:15,4:2,5:15,6:15,7:8,8:2,9:6,10:15,11:14,12:16,13:11,14:13,15:0,23:0}
        voxel_label_asNus = trans_semantic_asNus(voxel_label,mapping_dict)

        occ,p_colors =draw_return(voxel_label_asNus, 
                                    None, # predict_pts,
                                    [-40, -40, -1], 
                                    [0.4] * 3, 
                                    )
        lidar_height = self.lidar_height
        occ[:, 2] -= lidar_height

        #occ_loc = np.stack(occ.nonzero(), axis=-1)[:, [2, 1, 0]]
        #occ = np.concatenate([occ_loc, occ[occ_loc[:, 2], occ_loc[:, 1], occ_loc[:, 0]][:, None]], axis=-1)
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
        input_dict['occ'] = occ.astype(lidar.dtype)
        voxel_size = np.array(self.voxel_size).reshape((-1, 3))
        pc_range = np.array(self.point_cloud_range[:3]).reshape((-1, 3))
        # input_dict['occ'][:, :3] = (input_dict['occ'][:, :3] + 0.5) * voxel_size + pc_range

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
        data_dict['occ'] = data_dict['occ'].astype(int)
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
