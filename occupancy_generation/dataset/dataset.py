import os, numpy as np, pickle, time, uuid
from pyquaternion import Quaternion
from copy import deepcopy
from . import OPENOCC_DATASET
from .map_utils import VectorizedLocalMap, visualize_bev_hdmap, LiDARInstanceLines, DataContainer as DC

import torch
import glob
from typing import Any, Dict, Tuple
import cv2

from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB

from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes, Box3DMode
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuplan.database.maps_db.map_api import NuPlanMapWrapper
from nuplan.database.maps_db.map_explorer import NuPlanMapExplorer

from collections.abc import Sequence


from PIL import Image,ImageDraw
from torch.utils.data.dataloader import DataLoader

def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


# from check_brainpp_occ import read_occ_HR
@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar_ori:
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            nusc_dataroot=None,
            times=5,
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts'
        ):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        self.nusc_infos = data['infos']
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        # self.nusc = nusc
        
        self.times = times
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR

        xbound=[-40,40,0.4]
        ybound=[-40,40,0.4]
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2]) # 调节分辨率， 200或者是800
        canvas_w = int(patch_w / xbound[2]) 
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        # self.use_valid_flag=True
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)*self.times

    def __getitem__(self, index):
        index = index % len(self.nusc_infos)
        scene_name = self.scene_names[index]
        scene_len = self.scene_lens[index]
        idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        # idx=0
        # self.return_len=scene_len

        occs = []
        tokens=[]
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            tokens.append(token)
            label_file = os.path.join(self.data_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        # input_occs = np.stack(occs, dtype=np.int64)
        input_occs = np.stack(occs).astype(np.int64)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        # output_occs = np.stack(occs, dtype=np.int64)
        output_occs = np.stack(occs).astype(np.int64)
        metas = {}
        metas.update(scene_token=tokens)
        metas.update(scene_name=scene_name)
        metas.update(self.get_meta_data(scene_name, idx))
        metas.update(self.get_image_info(scene_name,idx))
        # metas.update(self.get_meta_data(scene_name, idx))
        # metas.update(self.get_meta_info(scene_name, idx))

        if self.test_mode:
            metas.update(self.get_meta_info(scene_name, idx))


        return input_occs[:self.return_len], output_occs[self.offset:], metas

    def get_meta_data(self, scene_name, idx):
        gt_modes = []
        xys = []
        for i in range(self.return_len + self.offset):
            xys.append(self.nusc_infos[scene_name][idx+i]['gt_ego_fut_trajs'][0]) #1*2
            gt_modes.append(self.nusc_infos[scene_name][idx+i]['pose_mode'])
        xys = np.asarray(xys)
        gt_modes = np.asarray(gt_modes)
        return {'rel_poses': xys, 'gt_mode': gt_modes}


    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        # import pdb; pdb.set_trace()
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []
        
        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            # import pdb; pdb.set_trace()
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t
            
            
            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            #cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            #focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
        
        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))
        
        return input_dict


@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar:
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            nusc_dataroot=None,
            times=5,
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts'
        ):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        self.nusc_infos = data['infos']
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.nusc = nusc

        self.nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_dataroot, verbose=True)
        self.maps = {}
        LOCATIONS = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(nusc_dataroot, location)

        self.classes= ['drivable_area','ped_crossing','walkway','stop_line','carpark_area','road_divider','lane_divider','road_block']
        self.object_classes =['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']
        self.times = times
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR

        xbound=[-40,40,0.4]
        ybound=[-40,40,0.4]
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.use_valid_flag=True

        self.lidar2canvas = np.array([
            [canvas_h / patch_h, 0, canvas_h / 2],
            [0, canvas_w / patch_w, canvas_w / 2],
            [0, 0, 1]
        ])


        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)#*self.times

    def __getitem__(self, index):
        index = index % len(self.nusc_infos)
        scene_name = self.scene_names[index]
        scene_len = self.scene_lens[index]
        # idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        idx=0
        self.return_len=scene_len

        occs = []
        tokens=[]
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            tokens.append(token)
            label_file = os.path.join(self.data_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        # input_occs = np.stack(occs, dtype=np.int64)
        input_occs = np.stack(occs).astype(np.int64)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        # output_occs = np.stack(occs, dtype=np.int64)
        output_occs = np.stack(occs).astype(np.int64)
        metas = {}
        metas.update(scene_token=tokens)
        metas.update(scene_name=scene_name)
        metas.update(self.get_meta_data(scene_name, idx))
        metas.update(self.get_image_info(scene_name,idx))
        # metas.update(self.get_meta_data(scene_name, idx))
        # metas.update(self.get_meta_info(scene_name, idx))

        if self.test_mode:
            metas.update(self.get_meta_info(scene_name, idx))

        # # train vqvae_4  for bev layout
        bevmaps=[]

        for i in range(self.return_len + self.offset):
            # token = self.nusc_infos[scene_name][idx + i]['token']
            # bevmap.update(scene_token=token)
            # bevmap.update(self.get_map_info(metas,scene_name,idx+i) )
            metas.update(self.get_meta_info(scene_name, idx+i))
            bevmap=self.get_map_info(metas,scene_name,idx+i)
            bevmaps.append(bevmap)
        bevmaps= np.stack(bevmaps).astype(bool)

        return input_occs[:self.return_len], output_occs[self.offset:], metas, bevmaps

    def get_meta_data(self, scene_name, idx):
        gt_modes = []
        xys = []
        for i in range(self.return_len + self.offset):
            xys.append(self.nusc_infos[scene_name][idx+i]['gt_ego_fut_trajs'][0]) #1*2
            gt_modes.append(self.nusc_infos[scene_name][idx+i]['pose_mode'])
        xys = np.asarray(xys)
        gt_modes = np.asarray(gt_modes)
        return {'rel_poses': xys, 'gt_mode': gt_modes}

    def get_meta_info(self, scene_name, idx):
        """Get annotation info according to the given index.

        """
        # T = 6
        # idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        fut_valid_flag = info['valid_flag']
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
                # print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        
        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        # if self.with_attr:
        #     gt_fut_trajs = info['gt_agent_fut_trajs'][mask]
        #     gt_fut_masks = info['gt_agent_fut_masks'][mask]
        #     gt_fut_goal = info['gt_agent_fut_goal'][mask]
        #     gt_lcf_feat = info['gt_agent_lcf_feat'][mask]
        #     gt_fut_yaw = info['gt_agent_fut_yaw'][mask]
        #     attr_labels = np.concatenate(
        #         [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1
        #     ).astype(np.float32)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
       

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            # attr_labels=attr_labels,
            fut_valid_flag=fut_valid_flag,)
        
        return anns_results
    def _project_dynamic_bbox(self, dynamic_mask, data):
        '''We use PIL for projection, while CVT use cv2. The results are
        slightly different due to anti-alias of line, but should be similar.
        '''
        for cls_id, cls_name in enumerate(self.object_classes):
            # pick boxes
            cls_mask = data['gt_labels_3d'] == cls_id
            boxes = data['gt_bboxes_3d'][cls_mask]
            if len(boxes) < 1:
                continue
            # get coordinates on canvas. the order of points matters.
            bottom_corners_lidar = boxes.corners[:, [0, 3, 7, 4], :2]

            bottom_corners_canvas = np.dot(
                # np.pad(flipped_points, ((0, 0), (0, 0), (0, 1)),
                np.pad(bottom_corners_lidar.numpy(), ((0, 0), (0, 0), (0, 1)),
                       constant_values=1.0),
                self.lidar2canvas.T)[..., :2]  # N, 4, xy
            # draw
            # Mod !!!
            points=bottom_corners_canvas
            centers = np.mean(points, axis=1, keepdims=True)
            points[:,:,1]= 2*centers[:,:,1]-points[:,:,1]
            bottom_corners_canvas=points

            render = Image.fromarray(dynamic_mask[cls_id])
            draw = ImageDraw.Draw(render)
            for box in bottom_corners_canvas:
                draw.polygon(
                    box.round().astype(np.int32).flatten().tolist(), fill=1)
            # save
            dynamic_mask[cls_id, :] = np.array(render)[:]
        return dynamic_mask

    def _project_dynamic(self, static_label, data):
        """for dynamic mask, one class per channel
        case 1: data is None, set all values to zeros
        """
        # setup
        ch = len(self.object_classes)
        dynamic_mask = np.zeros((ch, *self.canvas_size), dtype=np.uint8)

        # if int, set ch=object_classes with all zeros; otherwise, project
        if data is not None:
            dynamic_mask = self._project_dynamic_bbox(dynamic_mask, data)

        # combine with static_label
        dynamic_mask = dynamic_mask.transpose(0, 2, 1)
        combined_label = np.concatenate([static_label, dynamic_mask], axis=0)
        return combined_label

    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        # import pdb; pdb.set_trace()
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []
        
        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            # import pdb; pdb.set_trace()
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t
            
            
            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            #cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            #focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
        
        
        
        
        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))
        
        return input_dict

    def get_map_info(self, data,scene_name,idx):

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
        ego2lidar = np.linalg.inv(lidar2ego)

        lidar2global=ego2global @ lidar2ego

        map_pose = lidar2global[:2, 3]
        patch_box = (
            map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])  # angle between v and x-axis
        patch_angle = yaw / np.pi * 180

        

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        # cut semantics from nuscenesMap
        # sample_token=info["token"]
        scene_token=self.nusc.field2token('scene', 'name', scene_name)
        log_token= self.nusc.get('scene', scene_token[0])['log_token']
        location = self.nusc.get('log', log_token)['location']

        # location = info["location"]
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)  # TODO why need transpose here?
        masks = masks.astype(np.bool_)

        # here we handle possible combinations of semantics
        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.int64)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        
        bevmap={}
        if self.object_classes is not None:
            bevmap["gt_masks_bev_static"] = labels
            final_labels = self._project_dynamic(labels, data)
            # aux_labels = self._get_dynamic_aux(data)
            bevmap["gt_masks_bev"] = final_labels
            # data["gt_aux_bev"] = aux_labels
        else:
            bevmap["gt_masks_bev_static"] = labels
            bevmap["gt_masks_bev"] = labels

        bevmap["gt_masks_bev"] = bevmap["gt_masks_bev"][:, ::-1, :].copy()
        
        return bevmap["gt_masks_bev"]

        
@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidarTraverse(nuScenesSceneDatasetLidar_ori):
    def __init__(
        self,
        data_path,
        return_len,
        offset,
        imageset='train',
        nusc=None,
        times=1,
        test_mode=False,
        use_valid_flag=True,
        input_dataset='gts',
        output_dataset='gts',
    ):
        super().__init__(data_path, return_len, offset, imageset, nusc, times, test_mode, input_dataset, output_dataset)
        self.scene_lens = [l - self.return_len - self.offset for l in self.scene_lens]
        self.use_valid_flag = use_valid_flag
        self.CLASSES = [
            'noise', 'animal' ,'human.pedestrian.adult', 'human.pedestrian.child',
            'human.pedestrian.construction_worker',
            'human.pedestrian.personal_mobility',
            'human.pedestrian.police_officer',
            'human.pedestrian.stroller', 'human.pedestrian.wheelchair',
            'movable_object.barrier', 'movable_object.debris',
            'movable_object.pushable_pullable', 'movable_object.trafficcone',
            'static_object.bicycle_rack', 'vehicle.bicycle',
            'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car',
            'vehicle.construction', 'vehicle.emergency.ambulance',
            'vehicle.emergency.police', 'vehicle.motorcycle',
            'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
            'flat.other', 'flat.sidewalk', 'flat.terrain', 'flat.traffic_marking',
            'static.manmade', 'static.other', 'static.vegetation',
            'vehicle.ego'
        ]
        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR
        
    def __len__(self):
        'Denotes the total number of samples'
        # return sum(self.scene_lens)
        return len(self.nusc_infos)*5
    
    def __getitem__(self, index):
        for i, scene_len in enumerate(self.scene_lens):
            if index < scene_len:
                scene_name = self.scene_names[i]
                idx = index
                break
            else:
                index -= scene_len
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        # input_occs = np.stack(occs, dtype=np.int64)
        input_occs = np.stack(occs).astype(np.int64)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        # output_occs = np.stack(occs, dtype=np.int64)
        output_occs = np.stack(occs).astype(np.int64)
        metas = {}
        metas.update(scene_name=scene_name)
        metas.update(scene_token=self.nusc_infos[scene_name][4]['token'])
        metas.update(self.get_meta_data(scene_name, idx))
        if self.test_mode:
            metas.update(self.get_meta_info(scene_name, idx))
        metas.update(self.get_image_info(scene_name,idx))
        # import pdb; pdb.set_trace()
        return input_occs[:self.return_len], output_occs[self.offset:], metas
    
    def get_meta_info(self, scene_name, idx):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        fut_valid_flag = info['valid_flag']
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        '''gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
                print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        '''
        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        if self.with_attr:
            gt_fut_trajs = info['gt_agent_fut_trajs'][mask]
            gt_fut_masks = info['gt_agent_fut_masks'][mask]
            gt_fut_goal = info['gt_agent_fut_goal'][mask]
            gt_lcf_feat = info['gt_agent_lcf_feat'][mask]
            gt_fut_yaw = info['gt_agent_fut_yaw'][mask]
            attr_labels = np.concatenate(
                [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1
            ).astype(np.float32)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            #gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            attr_labels=attr_labels,
            fut_valid_flag=fut_valid_flag,)
        
        return anns_results
        
        
        
    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        # import pdb; pdb.set_trace()
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []
        
        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            # import pdb; pdb.set_trace()
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t
            
            
            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            #cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            #focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
        
        
        
        
        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))
        
        return input_dict
        

@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar_HR:
    def __init__(
            self, 
            occ_base_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            nusc_dataroot=None,
            times=5,
            quantize_size=(200,200,16),
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts'
        ):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        self.nusc_infos = data['infos']
        # self.scene_names = list(self.nusc_infos.keys())
        # self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.occ_base_path = occ_base_path
        self.return_len = return_len
        self.offset = offset
        self.quantize_size=quantize_size
        # self.nusc = nusc

        # self.nusc = NuScenes(version='advanced_12Hz_trainval', dataroot=nusc_dataroot, verbose=True)
        self.maps = {}
        LOCATIONS = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(nusc_dataroot, location)

        self.classes= ['drivable_area','ped_crossing','walkway','stop_line','carpark_area','road_divider','lane_divider','road_block']
        self.object_classes =['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']
        self.times = times
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR

        # for 200 resolution
        xbound=[-50,50,0.5]
        ybound=[-50,50,0.5]

        # for 800 resolution
        # xbound=[-50,50,0.125]
        # ybound=[-50,50,0.125]

        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.use_valid_flag=True

        self.lidar2canvas = np.array([
            [canvas_h / patch_h, 0, canvas_h / 2],
            [0, canvas_w / patch_w, canvas_w / 2],
            [0, 0, 1]
        ])

        self.start_on_keyframe = True
        self.start_on_firstframe = False

        self.clip_infos=self.build_clips(self.nusc_infos,data['scene_tokens'])

    def fliter_clips(self,clip):
        for frame in clip:
            token = self.nusc_infos[frame]['token']
            folder_path = f"{self.occ_base_path}/{token}"  # 获取文件夹路径
            # 使用 glob 模式查找文件夹下的 .npy 文件
            npy_files = glob.glob(f"{folder_path}/*.npy")
            if not npy_files:  # 如果没有找到 .npy 文件
                return 0
            file_path = npy_files[0]  # 取第一个匹配的 .npy 文件路径
            if not os.path.exists(file_path):  # 检查文件是否存在
                return 0
            # if os.path.exists(f"{self.occ_base_path}/{token}.npy")==False:
                # return 0
        return 1
            
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
                if self.fliter_clips(clip)==0:
                    continue
                all_clips.append(clip)
                if self.start_on_firstframe:
                    break
        # logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
        #              f"continuous scenes. Cut into {self.video_length}-clip, "
        #              f"which has {len(all_clips)} in total.")
        return all_clips

        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.clip_infos)

    def __getitem__(self, index):
        # index = index % len(self.nusc_infos)
        # scene_name = self.scene_names[index]
        # scene_len = self.scene_lens[index]
        # # idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        # idx=0
        # self.return_len=scene_len
        clip = self.clip_infos[index]
        occs = []
        tokens=[]
        bevmaps=[]
        metas = {}
        

        for frame in clip:
            token = self.nusc_infos[frame]['token']
            tokens.append(token)
            # occ = read_occ_HR(token,self.occ_base_path,self.quantize_size)
            occ = np.load(glob.glob(f"{self.occ_base_path}/{token}/*.npy")[0])
            # occ = np.load(f"{self.occ_base_path}/{token}.npy")
            occs.append(occ)

            metas.update(self.get_meta_info(frame))

            bevmap=self.get_map_info(metas,frame) # 获得 bevmap
            bevmaps.append(bevmap)
        
        metas.update(scene_token=tokens)
        input_occs = np.stack(occs).astype(np.int64)
        bevmaps= np.stack(bevmaps).astype(bool)
        
        # for i in range(self.return_len + self.offset):
            # token = self.nusc_infos[scene_name][idx + i]['token']
            # bevmap.update(scene_token=token)
            # bevmap.update(self.get_map_info(metas,scene_name,idx+i) )
            

        return input_occs, input_occs, metas, bevmaps

    # def get_meta_data(self, scene_name, idx):
    #     gt_modes = []
    #     xys = []
    #     for i in range(self.return_len + self.offset):
    #         xys.append(self.nusc_infos[scene_name][idx+i]['gt_ego_fut_trajs'][0]) #1*2
    #         gt_modes.append(self.nusc_infos[scene_name][idx+i]['pose_mode'])
    #     xys = np.asarray(xys)
    #     gt_modes = np.asarray(gt_modes)
    #     return {'rel_poses': xys, 'gt_mode': gt_modes}

    def get_meta_info(self, idx):
        """Get annotation info according to the given index.

        """
        # T = 6
        # idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[idx]
        fut_valid_flag = info['valid_flag']
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
                # print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        
        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
       

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            # attr_labels=attr_labels,
            fut_valid_flag=fut_valid_flag,)
        
        return anns_results

    def _project_dynamic_bbox(self, dynamic_mask, data):
        '''We use PIL for projection, while CVT use cv2. The results are
        slightly different due to anti-alias of line, but should be similar.
        '''
        for cls_id, cls_name in enumerate(self.object_classes):
            # pick boxes
            cls_mask = data['gt_labels_3d'] == cls_id
            boxes = data['gt_bboxes_3d'][cls_mask]
            if len(boxes) < 1:
                continue
            # get coordinates on canvas. the order of points matters.
            bottom_corners_lidar = boxes.corners[:, [0, 3, 7, 4], :2]

            bottom_corners_canvas = np.dot(
                # np.pad(flipped_points, ((0, 0), (0, 0), (0, 1)),
                np.pad(bottom_corners_lidar.numpy(), ((0, 0), (0, 0), (0, 1)),
                       constant_values=1.0),
                self.lidar2canvas.T)[..., :2]  # N, 4, xy
            # draw
            # Mod !!!
            points=bottom_corners_canvas
            centers = np.mean(points, axis=1, keepdims=True)
            points[:,:,1]= 2*centers[:,:,1]-points[:,:,1]
            bottom_corners_canvas=points

            render = Image.fromarray(dynamic_mask[cls_id])
            draw = ImageDraw.Draw(render)
            for box in bottom_corners_canvas:
                draw.polygon(
                    box.round().astype(np.int32).flatten().tolist(), fill=1)
            # save
            dynamic_mask[cls_id, :] = np.array(render)[:]
        return dynamic_mask

    def _project_dynamic(self, static_label, data):
        """for dynamic mask, one class per channel
        case 1: data is None, set all values to zeros
        """
        # setup
        ch = len(self.object_classes)
        dynamic_mask = np.zeros((ch, *self.canvas_size), dtype=np.uint8)

        # if int, set ch=object_classes with all zeros; otherwise, project
        if data is not None:
            dynamic_mask = self._project_dynamic_bbox(dynamic_mask, data)

        # combine with static_label
        dynamic_mask = dynamic_mask.transpose(0, 2, 1)
        combined_label = np.concatenate([static_label, dynamic_mask], axis=0)
        return combined_label

    # def get_image_info(self, scene_name, idx):
    #     T = 6
    #     idx = idx + self.return_len + self.offset - 1 - T
    #     info = self.nusc_infos[scene_name][idx]
    #     # import pdb; pdb.set_trace()
    #     input_dict = dict(
    #         sample_idx=info['token'],
    #         ego2global_translation = info['ego2global_translation'],
    #         ego2global_rotation = info['ego2global_rotation'],
    #     )
    #     f = 0.0055
    #     image_paths = []
    #     lidar2img_rts = []
    #     lidar2cam_rts = []
    #     cam_intrinsics = []
    #     cam_positions = []
    #     focal_positions = []
        
    #     lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    #     lidar2ego = np.eye(4)
    #     lidar2ego[:3, :3] = lidar2ego_r
    #     lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
    #     ego2lidar = np.linalg.inv(lidar2ego)
    #     for cam_type, cam_info in info['cams'].items():
    #         image_paths.append(cam_info['data_path'])
    #         # obtain lidar to image transformation matrix
    #         lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
    #         lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
    #         lidar2cam_rt = np.eye(4)
    #         lidar2cam_rt[:3, :3] = lidar2cam_r.T
    #         lidar2cam_rt[3, :3] = -lidar2cam_t
    #         intrinsic = cam_info['cam_intrinsic']
    #         viewpad = np.eye(4)
    #         viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    #         lidar2img_rt = (viewpad @ lidar2cam_rt.T)
    #         lidar2img_rts.append(lidar2img_rt)
    #         cam_intrinsics.append(viewpad)
    #         lidar2cam_rts.append(lidar2cam_rt.T)
    #         cam_intrinsics.append(viewpad)
    #         lidar2cam_rts.append(lidar2cam_rt.T)
    #         # import pdb; pdb.set_trace()
    #         ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
    #         ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
    #         ego2cam_rt = np.eye(4)
    #         ego2cam_rt[:3, :3] = ego2cam_r.T
    #         ego2cam_rt[3, :3] = -ego2cam_t
            
            
    #         cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
    #         focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
    #         #cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
    #         cam_positions.append(cam_position.flatten()[:3])
    #         #focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
    #         focal_positions.append(focal_position.flatten()[:3])
        
        
        
        
    #     input_dict.update(
    #         dict(
    #             img_filename=image_paths,
    #             lidar2img=lidar2img_rts,
    #             cam_intrinsic=cam_intrinsics,
    #             lidar2cam=lidar2cam_rts,
    #             ego2lidar=ego2lidar,
    #             cam_positions=cam_positions,
    #             focal_positions=focal_positions,
    #             lidar2ego=lidar2ego,
    #         ))
        
    #     return input_dict

    def get_map_info(self, data,idx):

        info = self.nusc_infos[idx]
        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        ego2global= np.eye(4)
        ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
        ego2global[:3, 3] = np.array(ego2global_translation).T

        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)

        lidar2global=ego2global @ lidar2ego

        map_pose = lidar2global[:2, 3]
        patch_box = (
            map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])  # angle between v and x-axis
        patch_angle = yaw / np.pi * 180

        

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        # cut semantics from nuscenesMap
        # sample_token=info["token"]
        # scene_token=self.nusc.field2token('scene', 'name', scene_name)
        # log_token= self.nusc.get('scene', scene_token[0])['log_token']
        # location = self.nusc.get('log', log_token)['location']
        # location = info["location"]
        location = self.nusc_infos[idx]['location']
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)  # TODO why need transpose here?
        masks = masks.astype(np.bool_)

        # here we handle possible combinations of semantics
        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.int64)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        # data={}
        bevmap={}
        if self.object_classes is not None:
            bevmap["gt_masks_bev_static"] = labels
            final_labels = self._project_dynamic(labels, data)
            # aux_labels = self._get_dynamic_aux(data)
            bevmap["gt_masks_bev"] = final_labels
            # data["gt_aux_bev"] = aux_labels
        else:
            bevmap["gt_masks_bev_static"] = labels
            bevmap["gt_masks_bev"] = labels

        # bevmap["gt_masks_bev"] = bevmap["gt_masks_bev"][:, ::-1, :].copy()
        
        return bevmap["gt_masks_bev"]



@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar_OpenScene:
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            nusc_dataroot=None,
            times=5,
            quantize_size=(200,200,16),
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts'
        ):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        self.nusc_infos = data['infos']
        self.occ_base_path = "/lpai/volumes/lmm-data-proc/hzhu/code/occ_gen/data/"
        # self.scene_names = list(self.nusc_infos.keys())
        # self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.quantize_size=quantize_size
        # self.nusc = nusc

        # self.nusc = NuScenes(version='advanced_12Hz_trainval', dataroot=nusc_dataroot, verbose=True)
        self.maps = {}
        LOCATIONS = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
        # LOCATIONS = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
        # for location in LOCATIONS:
        #     self.maps[location] = NuScenesMap(nusc_dataroot, location)

        self.classes= ['intersection','generic_drivable_areas','walkway','carpark_areas','crosswalks','lane_group_connectors','lane_group_polygons','road_segments']
        self.object_classes = ['vehicle','bicycle','pedestrian','traffic_cone','barrier','czone_sign','generic_object']
        self.times = times
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR

        # for 200 resolution
        xbound=[-50,50,0.5]
        ybound=[-50,50,0.5]
        zbound=[-4,4,0.5]
        self.point_cloud_range = [xbound[0],ybound[0],zbound[0],xbound[1],ybound[1],zbound[1]]
        voxel_size=0.5
        self.occ_xdim = int((xbound[1] - xbound[0]) / voxel_size)
        self.occ_ydim = int((ybound[1] - ybound[0]) / voxel_size)
        self.occ_zdim = int((zbound[1] - zbound[0]) / voxel_size)
        self.voxel_num = self.occ_xdim*self.occ_ydim*self.occ_zdim


        # for 800 resolution
        # xbound=[-50,50,0.125]
        # ybound=[-50,50,0.125]
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.use_valid_flag=True

        self.lidar2canvas = np.array([
            [canvas_h / patch_h, 0, canvas_h / 2],
            [0, canvas_w / patch_w, canvas_w / 2],
            [0, 0, 1]
        ])

        self.start_on_keyframe = True
        self.start_on_firstframe = False

        self.clip_infos=self.build_clips(self.nusc_infos,data['scene_tokens'])

    def fliter_clips(self,clip):
        for frame in clip:
            if self.nusc_infos[frame]['occ_gt_final_path'] is None:
                return 0
            file_path = os.path.join(self.occ_base_path,self.nusc_infos[frame]['occ_gt_final_path'])
            if not os.path.exists(file_path):  # 检查文件是否存在
                return 0
        return 1
            
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
                if self.fliter_clips(clip)==0:
                    continue
                all_clips.append(clip)
                if self.start_on_firstframe:
                    break
        # logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
        #              f"continuous scenes. Cut into {self.video_length}-clip, "
        #              f"which has {len(all_clips)} in total.")
        return all_clips
    
    def obtain_points_label(self, occ):
        occ_index, occ_cls = occ[:, 0], occ[:, 1]
        occ = np.ones(self.voxel_num, dtype=np.int8)*11
        occ[occ_index[:]] = occ_cls  # (voxel_num)
        points = []
        for i in range(len(occ_index)):
            indice = occ_index[i]
            x = indice % self.occ_xdim
            y = (indice // self.occ_xdim) % self.occ_xdim
            z = indice // (self.occ_xdim*self.occ_xdim)
            point_x = (x + 0.5) / self.occ_xdim * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
            point_y = (y + 0.5) / self.occ_ydim * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
            point_z = (z + 0.5) / self.occ_zdim * (self.point_cloud_range[5] - self.point_cloud_range[2]) + self.point_cloud_range[2]
            points.append([point_x, point_y, point_z])
        
        points = np.stack(points)
        point_label = occ_cls
        points_with_label = np.concatenate([points, point_label[:, None]], axis=-1)
        return points_with_label
    
    def convert_to_voxel_grid_int(self, voxels_):
        voxel = np.zeros((800, 800, 64), dtype=np.int32)
        voxel[voxels_[:, 0].astype(np.int32), voxels_[:, 1].astype(np.int32), voxels_[:, 2].astype(np.int32)] = voxels_[:, 3].astype(np.int32)+1
        return voxel
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.clip_infos)

    def __getitem__(self, index):
        # index = index % len(self.nusc_infos)
        # scene_name = self.scene_names[index]
        # scene_len = self.scene_lens[index]
        # # idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        # idx=0
        # self.return_len=scene_len
        clip = self.clip_infos[index]
        occs = []
        tokens=[]
        bevmaps=[]
        metas = {}
        

        for frame in clip:
            token = self.nusc_infos[frame]['token']
            tokens.append(token)
            occ_path = os.path.join(self.occ_base_path,self.nusc_infos[frame]['occ_gt_final_path'])
            occ = np.load(occ_path)

            # occ change to (200,200,16)
            occ_to_voxel = self.obtain_points_label(occ)
            occ_voxel_grid = self.convert_to_voxel_grid_int(occ_to_voxel)
            occs.append(occ_voxel_grid)

            metas.update(self.get_meta_info(frame))

            # bevmap=self.get_map_info(metas,frame) # 获得 bevmap
            # bevmaps.append(bevmap)
        
        metas.update(scene_token=tokens)
        input_occs = np.stack(occs).astype(np.int64)
        # bevmaps= np.stack(bevmaps).astype(bool)
        
        # for i in range(self.return_len + self.offset):
            # token = self.nusc_infos[scene_name][idx + i]['token']
            # bevmap.update(scene_token=token)
            # bevmap.update(self.get_map_info(metas,scene_name,idx+i) )
            

        # return input_occs, input_occs, metas, bevmaps
        return input_occs, input_occs, metas

    # def get_meta_data(self, scene_name, idx):
    #     gt_modes = []
    #     xys = []
    #     for i in range(self.return_len + self.offset):
    #         xys.append(self.nusc_infos[scene_name][idx+i]['gt_ego_fut_trajs'][0]) #1*2
    #         gt_modes.append(self.nusc_infos[scene_name][idx+i]['pose_mode'])
    #     xys = np.asarray(xys)
    #     gt_modes = np.asarray(gt_modes)
    #     return {'rel_poses': xys, 'gt_mode': gt_modes}

    def get_meta_info(self, idx):
        """Get annotation info according to the given index.

        """
        # T = 6
        # idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[idx]
        # fut_valid_flag = info['valid_flag']
        # filter out bbox containing no points
        # if self.use_valid_flag:
        #     mask = info['valid_flag']
        # else:
        #     mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info["anns"]['gt_boxes']
        gt_names_3d = info["anns"]['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d: 
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
                # print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        
        if self.with_velocity:
            gt_velocity = info["anns"]['gt_velocity_3d'][:, :2]

            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
       

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            # attr_labels=attr_labels,
            # fut_valid_flag=fut_valid_flag,
            )
        
        return anns_results

    def _project_dynamic_bbox(self, dynamic_mask, data):
        '''We use PIL for projection, while CVT use cv2. The results are
        slightly different due to anti-alias of line, but should be similar.
        '''
        for cls_id, cls_name in enumerate(self.object_classes):
            # pick boxes
            cls_mask = data['gt_labels_3d'] == cls_id
            boxes = data['gt_bboxes_3d'][cls_mask]
            if len(boxes) < 1:
                continue
            # get coordinates on canvas. the order of points matters.
            bottom_corners_lidar = boxes.corners[:, [0, 3, 7, 4], :2]

            bottom_corners_canvas = np.dot(
                # np.pad(flipped_points, ((0, 0), (0, 0), (0, 1)),
                np.pad(bottom_corners_lidar.numpy(), ((0, 0), (0, 0), (0, 1)),
                       constant_values=1.0),
                self.lidar2canvas.T)[..., :2]  # N, 4, xy
            # draw
            # Mod !!!
            points=bottom_corners_canvas
            centers = np.mean(points, axis=1, keepdims=True)
            points[:,:,1]= 2*centers[:,:,1]-points[:,:,1]
            bottom_corners_canvas=points

            render = Image.fromarray(dynamic_mask[cls_id])
            draw = ImageDraw.Draw(render)
            for box in bottom_corners_canvas:
                draw.polygon(
                    box.round().astype(np.int32).flatten().tolist(), fill=1)
            # save
            dynamic_mask[cls_id, :] = np.array(render)[:]
        return dynamic_mask

    def _project_dynamic(self, static_label, data):
        """for dynamic mask, one class per channel
        case 1: data is None, set all values to zeros
        """
        # setup
        ch = len(self.object_classes)
        dynamic_mask = np.zeros((ch, *self.canvas_size), dtype=np.uint8)

        # if int, set ch=object_classes with all zeros; otherwise, project
        if data is not None:
            dynamic_mask = self._project_dynamic_bbox(dynamic_mask, data)

        # combine with static_label
        dynamic_mask = dynamic_mask.transpose(0, 2, 1)
        combined_label = np.concatenate([static_label, dynamic_mask], axis=0)
        return combined_label

    # def get_image_info(self, scene_name, idx):
    #     T = 6
    #     idx = idx + self.return_len + self.offset - 1 - T
    #     info = self.nusc_infos[scene_name][idx]
    #     # import pdb; pdb.set_trace()
    #     input_dict = dict(
    #         sample_idx=info['token'],
    #         ego2global_translation = info['ego2global_translation'],
    #         ego2global_rotation = info['ego2global_rotation'],
    #     )
    #     f = 0.0055
    #     image_paths = []
    #     lidar2img_rts = []
    #     lidar2cam_rts = []
    #     cam_intrinsics = []
    #     cam_positions = []
    #     focal_positions = []
        
    #     lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    #     lidar2ego = np.eye(4)
    #     lidar2ego[:3, :3] = lidar2ego_r
    #     lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
    #     ego2lidar = np.linalg.inv(lidar2ego)
    #     for cam_type, cam_info in info['cams'].items():
    #         image_paths.append(cam_info['data_path'])
    #         # obtain lidar to image transformation matrix
    #         lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
    #         lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
    #         lidar2cam_rt = np.eye(4)
    #         lidar2cam_rt[:3, :3] = lidar2cam_r.T
    #         lidar2cam_rt[3, :3] = -lidar2cam_t
    #         intrinsic = cam_info['cam_intrinsic']
    #         viewpad = np.eye(4)
    #         viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    #         lidar2img_rt = (viewpad @ lidar2cam_rt.T)
    #         lidar2img_rts.append(lidar2img_rt)
    #         cam_intrinsics.append(viewpad)
    #         lidar2cam_rts.append(lidar2cam_rt.T)
    #         cam_intrinsics.append(viewpad)
    #         lidar2cam_rts.append(lidar2cam_rt.T)
    #         # import pdb; pdb.set_trace()
    #         ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
    #         ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
    #         ego2cam_rt = np.eye(4)
    #         ego2cam_rt[:3, :3] = ego2cam_r.T
    #         ego2cam_rt[3, :3] = -ego2cam_t
            
            
    #         cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
    #         focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
    #         #cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
    #         cam_positions.append(cam_position.flatten()[:3])
    #         #focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
    #         focal_positions.append(focal_position.flatten()[:3])
        
        
        
        
    #     input_dict.update(
    #         dict(
    #             img_filename=image_paths,
    #             lidar2img=lidar2img_rts,
    #             cam_intrinsic=cam_intrinsics,
    #             lidar2cam=lidar2cam_rts,
    #             ego2lidar=ego2lidar,
    #             cam_positions=cam_positions,
    #             focal_positions=focal_positions,
    #             lidar2ego=lidar2ego,
    #         ))
        
    #     return input_dict

    def get_map_info(self, data,idx):

        info = self.nusc_infos[idx]
        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        ego2global= np.eye(4)
        ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
        ego2global[:3, 3] = np.array(ego2global_translation).T

        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)

        lidar2global=ego2global @ lidar2ego

        map_pose = lidar2global[:2, 3]
        patch_box = (
            map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])  # angle between v and x-axis
        patch_angle = yaw / np.pi * 180

        

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        # cut semantics from nuscenesMap
        # sample_token=info["token"]
        # scene_token=self.nusc.field2token('scene', 'name', scene_name)
        # log_token= self.nusc.get('scene', scene_token[0])['log_token']
        # location = self.nusc.get('log', log_token)['location']
        # location = info["location"]
        location = self.nusc_infos[idx]['location']
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)  # TODO why need transpose here?
        masks = masks.astype(np.bool_)

        # here we handle possible combinations of semantics
        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.int64)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        # data={}
        bevmap={}
        if self.object_classes is not None:
            bevmap["gt_masks_bev_static"] = labels
            final_labels = self._project_dynamic(labels, data)
            # aux_labels = self._get_dynamic_aux(data)
            bevmap["gt_masks_bev"] = final_labels
            # data["gt_aux_bev"] = aux_labels
        else:
            bevmap["gt_masks_bev_static"] = labels
            bevmap["gt_masks_bev"] = labels

        # bevmap["gt_masks_bev"] = bevmap["gt_masks_bev"][:, ::-1, :].copy()
        
        return bevmap["gt_masks_bev"]


@OPENOCC_DATASET.register_module()
class Nuplan_HR_occ_mini:
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            occ_dataroot=None,
            bev_dataroot=None,
            times=5,
            quantize_size=(400,400,32),
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts',
            debug=False 
        ):

        with open(imageset, 'rb') as f:
            pkl_data = pickle.load(f)
        print(f"=> loaded pkl_data from {imageset}")

        self.nuplan_infos = pkl_data['infos']
        self.occ_base_path = occ_dataroot
        self.bev_base_path = bev_dataroot
        self.scene_names = pkl_data['scene_tokens']
        self.scene_lens = [len(sn) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.quantize_size=quantize_size


        self.object_classes = ['vehicle','bicycle','pedestrian','traffic_cone','barrier','czone_sign','generic_object']
        self.times = times
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR
        print(f"quantize_size: {self.quantize_size}")
        # for 200 resolution
        xbound=[-50,50,0.25]
        ybound=[-50,50,0.25]
        zbound=[-3,5,0.25]
        voxel_size=0.25

        self.point_cloud_range = [xbound[0],ybound[0],zbound[0],xbound[1],ybound[1],zbound[1]]
        self.occ_xdim = int((xbound[1] - xbound[0]) / voxel_size)
        self.occ_ydim = int((ybound[1] - ybound[0]) / voxel_size)
        self.occ_zdim = int((zbound[1] - zbound[0]) / voxel_size)
        self.voxel_num = self.occ_xdim*self.occ_ydim*self.occ_zdim
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.use_valid_flag=True

        self.lidar2canvas = np.array([
            [canvas_h / patch_h, 0, canvas_h / 2],
            [0, canvas_w / patch_w, canvas_w / 2],
            [0, 0, 1]
        ])

        self.start_on_keyframe = True
        self.start_on_firstframe = False

        if "clip_infos" in pkl_data:
            self.clip_infos = pkl_data['clip_infos']
        else:   
            if debug:
                self.clip_infos=self.build_clips(self.nuplan_infos, pkl_data['scene_tokens'][:1])
            else:
                self.clip_infos=self.build_clips(self.nuplan_infos, pkl_data['scene_tokens'])
            print(f"=> clip_infos: {len(self.clip_infos)}")
            # save clip_infos to new pkl
            # 🎯 只保留必要的数据，减小文件大小
            # 从 nuplan_infos 中只提取 token 信息
            lightweight_infos = []
            for info in self.nuplan_infos:
                lightweight_info = {'token': info['token']}
                lightweight_infos.append(lightweight_info)
            
            new_pkl_data = {
                'infos': lightweight_infos,  # 只包含token的轻量版本
                'scene_tokens': pkl_data['scene_tokens'],
                'clip_infos': self.clip_infos,
                'original_info_count': len(self.nuplan_infos),
            }
            
            # 计算文件大小对比
            original_size = len(pickle.dumps(self.nuplan_infos)) / 1024 / 1024
            lightweight_size = len(pickle.dumps(lightweight_infos)) / 1024 / 1024
            
            # 生成随机文件名，避免冲突
            random_id = str(uuid.uuid4())[:8]  # 取前8位UUID
            timestamp = int(time.time())
            clip_count = len(self.clip_infos)
            mode = "debug" if debug else "full"
            output_file = f'clip_infos_{mode}_{clip_count}clips_{timestamp}_{random_id}.pkl'
            
            with open(output_file, 'wb') as f:
                pickle.dump(new_pkl_data, f)
            
            saved_size = os.path.getsize(output_file) / 1024 / 1024
            print(f"=> 💾 Saved clip_infos to {output_file}")
            print(f"=> 📊 Size comparison:")
            print(f"   Original infos: {original_size:.2f} MB")
            print(f"   Lightweight infos: {lightweight_size:.2f} MB") 
            print(f"   Final file: {saved_size:.2f} MB")
            print(f"   Space saved: {((original_size - lightweight_size) / original_size * 100):.1f}%")


    def fliter_clips(self,clip):
        for frame in clip:
            if self.nuplan_infos[frame]["token"] is None:
                print(f"Warning: {self.nuplan_infos[frame]['token']} is None")
                return 0
            file_path = os.path.join(self.occ_base_path,self.nuplan_infos[frame]["token"]+".npy")
            if not os.path.exists(file_path):  # 检查文件是否存在
                print(f"Warning: {file_path} does not exist")
                return 0
        return 1
            
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
        self.token_data_dict = {item['token']: idx for idx, item in enumerate(data_infos)}
        all_clips = []
        for i, scene in enumerate(scene_tokens):
            print(f"building clips for scene: {i}/{len(scene_tokens)}")
            for start in range(len(scene) - self.return_len + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + self.return_len]]
                if self.fliter_clips(clip)==0:
                    continue
                all_clips.append(clip)
                if self.start_on_firstframe:
                    break
        return all_clips
    
    def obtain_points_label(self, occ):
        occ_index, occ_cls = occ[:, 0], occ[:, 1]
        occ = np.ones(self.voxel_num, dtype=np.int8)*11
        occ[occ_index[:]] = occ_cls  # (voxel_num)
        points = []
        for i in range(len(occ_index)):
            indice = occ_index[i]
            x = indice % self.occ_xdim
            y = (indice // self.occ_xdim) % self.occ_xdim
            z = indice // (self.occ_xdim*self.occ_xdim)
            point_x = (x + 0.5) / self.occ_xdim * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
            point_y = (y + 0.5) / self.occ_ydim * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
            point_z = (z + 0.5) / self.occ_zdim * (self.point_cloud_range[5] - self.point_cloud_range[2]) + self.point_cloud_range[2]
            points.append([point_x, point_y, point_z])
        
        points = np.stack(points)
        point_label = occ_cls
        points_with_label = np.concatenate([points, point_label[:, None]], axis=-1)
        return points_with_label
    
    def convert_to_voxel_grid_int(self, voxels_):
        voxel = np.zeros((800, 800, 64), dtype=np.int32)
        voxel[voxels_[:, 0].astype(np.int32), voxels_[:, 1].astype(np.int32), voxels_[:, 2].astype(np.int32)] = voxels_[:, 3].astype(np.int32)+1
        return voxel
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.clip_infos)

    def __getitem__(self, index):
        # index = index % len(self.scene_lens)
        # scene_name = self.scene_names[index]
        # scene_len = self.scene_lens[index]
        # # idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        # idx=0
        # self.return_len=scene_len
        clip = self.clip_infos[index]
        occs = []
        tokens=[]
        bevmaps=[]
        auxs=[]
        metas = {}
        
        
        layer_to_merge=[0,1,3,4,5,6,7]
        bev_ch_use=[1,2,8,9,10,11,12,13,14,15,17]

        for frame in clip:
            token = self.nuplan_infos[frame]['token']
            tokens.append(token)
            occ_path = os.path.join(self.occ_base_path,self.nuplan_infos[frame]["token"]+".npy") # h, w, d
            occ = np.load(occ_path)

            # occ change to (200,200,16)
            occ_voxel_grid = occ
            occs.append(occ_voxel_grid)

            metas.update(self.get_meta_info(frame))

            # bevmap 可选择
            bev_file_path=os.path.join(self.bev_base_path,self.nuplan_infos[frame]["token"]+".npz") 
            if os.path.exists(bev_file_path):
                bev_data=np.load(bev_file_path)['gt_bev_masks']
                aux_data=np.zeros(bev_data.shape)
            else:
                bev_data = np.zeros(occ_voxel_grid.shape)
                aux_data = np.zeros(bev_data.shape)

            # bev_data=np.load(bev_file_path)['gt_bev_masks']
            # aux_data=np.zeros(bev_data.shape)
            auxs.append(aux_data)

            bev_data[1,:,:] = np.any(bev_data[layer_to_merge, :, :], axis=0).astype(int)
            # occ_with_bev = replace_occ_grid_with_bev_nuplan(input_occ_data=occ_data, bevlayout=bev_data)
            bevmap=self.nBEV1(bev_data, bev_ch_use)
            bevmaps.append(bevmap)

        
        metas.update(scene_token=tokens)
        input_occs = np.stack(occs).astype(np.int64)
        bevmaps= np.stack(bevmaps).astype(bool)

        return input_occs, input_occs, bevmaps, metas, auxs

    def nBEV1(self, data_b,ch_use): # 18,200,200 -> 1,200,200
        data_b = data_b[ch_use]
        mask = data_b>0.01
        cumulative_mask = np.cumsum(mask, axis=0)
        max_index_map = np.argmax(cumulative_mask, axis=0)
        max_index_map = max_index_map / (len(ch_use) -1)
        all_zero_mask = np.all(mask == 0, axis=0)
        max_index_map[all_zero_mask] = -1
        data_b= np.array([max_index_map])
        return data_b

    def get_meta_info(self, idx):
        """Get annotation info according to the given index.

        """
        info = self.nuplan_infos[idx]
        gt_bboxes_3d = info["anns"]['gt_boxes']
        gt_names_3d = info["anns"]['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d: 
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
                # print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        
        if self.with_velocity:
            gt_velocity = info["anns"]['gt_velocity_3d'][:, :2]

            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
       

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            )
        
        return anns_results

    def _project_dynamic_bbox(self, dynamic_mask, data):
        '''We use PIL for projection, while CVT use cv2. The results are
        slightly different due to anti-alias of line, but should be similar.
        '''
        for cls_id, cls_name in enumerate(self.object_classes):
            # pick boxes
            cls_mask = data['gt_labels_3d'] == cls_id
            boxes = data['gt_bboxes_3d'][cls_mask]
            if len(boxes) < 1:
                continue
            # get coordinates on canvas. the order of points matters.
            bottom_corners_lidar = boxes.corners[:, [0, 3, 7, 4], :2]

            bottom_corners_canvas = np.dot(
                # np.pad(flipped_points, ((0, 0), (0, 0), (0, 1)),
                np.pad(bottom_corners_lidar.numpy(), ((0, 0), (0, 0), (0, 1)),
                       constant_values=1.0),
                self.lidar2canvas.T)[..., :2]  # N, 4, xy
            # draw
            points=bottom_corners_canvas
            centers = np.mean(points, axis=1, keepdims=True)
            points[:,:,1]= 2*centers[:,:,1]-points[:,:,1]
            bottom_corners_canvas=points

            render = Image.fromarray(dynamic_mask[cls_id])
            draw = ImageDraw.Draw(render)
            for box in bottom_corners_canvas:
                draw.polygon(
                    box.round().astype(np.int32).flatten().tolist(), fill=1)
            # save
            dynamic_mask[cls_id, :] = np.array(render)[:]
        return dynamic_mask

    def _project_dynamic(self, static_label, data):
        """for dynamic mask, one class per channel
        case 1: data is None, set all values to zeros
        """
        # setup
        ch = len(self.object_classes)
        dynamic_mask = np.zeros((ch, *self.canvas_size), dtype=np.uint8)

        # if int, set ch=object_classes with all zeros; otherwise, project
        if data is not None:
            dynamic_mask = self._project_dynamic_bbox(dynamic_mask, data)

        # combine with static_label
        dynamic_mask = dynamic_mask.transpose(0, 2, 1)
        combined_label = np.concatenate([static_label, dynamic_mask], axis=0)
        return combined_label

    


    def vectormap_pipeline(self, input_dict):
        """
        Process vector map data for input example, using transformation matrices and
        generating annotations.
        """

        lidar2ego = input_dict["lidar2ego"]
        ego2global = input_dict["ego2global"]
        lidar2global = ego2global @ lidar2ego
        # lidar2global = rotation_z_neg90 @ lidar2global
        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_rotation = Quaternion(matrix=lidar2global)

        anns_results = self.vector_map.gen_vectorized_samples_nuplan(
            self.vector_maps[input_dict["location"]], # 'us-nv-las-vegas-strip'
            lidar2global_translation, # [664465.0781396995, 3997749.3183807055, 616.683242571562]
            lidar2global_rotation, # Quaternion(0.6999239997105716, 0.0002516337328280848, 0.004921886360219306, 0.7142003264800971)
        )
        gt_vecs_label = to_tensor(anns_results["gt_vecs_label"])
        if isinstance(anns_results["gt_vecs_pts_loc"], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results["gt_vecs_pts_loc"]
            gt_lines_instance = gt_vecs_pts_loc.instance_list
            gt_map_pts = [np.array(list(line.coords)) for line in gt_lines_instance]
        input_dict["gt_vecs_label"] = DC(gt_vecs_label, cpu_only=False)
        input_dict["gt_vecs_pts_loc"] = DC(gt_map_pts, cpu_only=True)

        # Visualizing BEV map
        # drivable_mask = (
        #     input_dict["gt_masks_bev"][0, ...] + input_dict["gt_masks_bev"][-1, ...]
        # ).astype(bool)

        bev_map = visualize_bev_hdmap(
            input_dict["gt_vecs_pts_loc"].data,
            input_dict["gt_vecs_label"].data,
            self.canvas_size,
            # vis_format="polyline_pts",
            # drivable_mask=drivable_mask,
            # nuplan=True,
        )
        bev_map = bev_map.transpose(2, 0, 1)
        return bev_map


@OPENOCC_DATASET.register_module()
class Nuplan_HR_occ_full:
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            occ_dataroot=None,
            bev_dataroot=None,
            times=5,
            quantize_size=(400,400,32),
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts',
            debug=False 
        ):

        with open(imageset, 'rb') as f:
            pkl_data = pickle.load(f)
        print(f"=> loaded pkl_data from {imageset}")

        self.nuplan_infos = pkl_data['infos']
        self.occ_base_path = occ_dataroot
        self.bev_base_path = bev_dataroot
        self.scene_names = pkl_data['scene_tokens']
        self.scene_lens = [len(sn) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.quantize_size=quantize_size


        self.object_classes = ['vehicle','bicycle','pedestrian','traffic_cone','barrier','czone_sign','generic_object']
        self.times = times
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR
        print(f"quantize_size: {self.quantize_size}")
        # for 200 resolution
        xbound=[-50,50,0.25]
        ybound=[-50,50,0.25]
        zbound=[-3,5,0.25]
        voxel_size=0.25

        self.point_cloud_range = [xbound[0],ybound[0],zbound[0],xbound[1],ybound[1],zbound[1]]
        self.occ_xdim = int((xbound[1] - xbound[0]) / voxel_size)
        self.occ_ydim = int((ybound[1] - ybound[0]) / voxel_size)
        self.occ_zdim = int((zbound[1] - zbound[0]) / voxel_size)
        self.voxel_num = self.occ_xdim*self.occ_ydim*self.occ_zdim
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.use_valid_flag=True

        self.lidar2canvas = np.array([
            [canvas_h / patch_h, 0, canvas_h / 2],
            [0, canvas_w / patch_w, canvas_w / 2],
            [0, 0, 1]
        ])

        self.start_on_keyframe = True
        self.start_on_firstframe = False

        if "clip_infos" in pkl_data:
            self.clip_infos = pkl_data['clip_infos']
        else:   
            if debug:
                self.clip_infos=self.build_clips(self.nuplan_infos, pkl_data['scene_tokens'][:1])
            else:
                self.clip_infos=self.build_clips(self.nuplan_infos, pkl_data['scene_tokens'])
        print(f"=> clip_infos: {len(self.clip_infos)}, {self.clip_infos[0]}")
      

    def fliter_clips(self,clip):
        for frame in clip:
            if self.nuplan_infos[frame]["token"] is None:
                print(f"Warning: {self.nuplan_infos[frame]['token']} is None")
                return 0
            file_path = os.path.join(self.occ_base_path,self.nuplan_infos[frame]["token"]+".npy")
            if not os.path.exists(file_path):  # 检查文件是否存在
                print(f"Warning: {file_path} does not exist")
                return 0
        return 1
            
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
        self.token_data_dict = {item['token']: idx for idx, item in enumerate(data_infos)}
        all_clips = []
        for i, scene in enumerate(scene_tokens):
            print(f"building clips for scene: {i}/{len(scene_tokens)}")
            for start in range(len(scene) - self.return_len + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + self.return_len]]
                if self.fliter_clips(clip)==0:
                    continue
                all_clips.append(clip)
                if self.start_on_firstframe:
                    break
        return all_clips
    
    def obtain_points_label(self, occ):
        occ_index, occ_cls = occ[:, 0], occ[:, 1]
        occ = np.ones(self.voxel_num, dtype=np.int8)*11
        occ[occ_index[:]] = occ_cls  # (voxel_num)
        points = []
        for i in range(len(occ_index)):
            indice = occ_index[i]
            x = indice % self.occ_xdim
            y = (indice // self.occ_xdim) % self.occ_xdim
            z = indice // (self.occ_xdim*self.occ_xdim)
            point_x = (x + 0.5) / self.occ_xdim * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
            point_y = (y + 0.5) / self.occ_ydim * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
            point_z = (z + 0.5) / self.occ_zdim * (self.point_cloud_range[5] - self.point_cloud_range[2]) + self.point_cloud_range[2]
            points.append([point_x, point_y, point_z])
        
        points = np.stack(points)
        point_label = occ_cls
        points_with_label = np.concatenate([points, point_label[:, None]], axis=-1)
        return points_with_label
    
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
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.clip_infos)

    def __getitem__(self, index):
        # index = index % len(self.scene_lens)
        # scene_name = self.scene_names[index]
        # scene_len = self.scene_lens[index]
        # # idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        # idx=0
        # self.return_len=scene_len
        clip = self.clip_infos[index]
        occs = []
        tokens=[]
        bevmaps=[]
        auxs=[]
        metas = {}
        
        
        layer_to_merge=[0,1,3,4,5,6,7]
        bev_ch_use=[1,2,8,9,10,11,12,13,14,15,17]

        for frame in clip:
            token = self.nuplan_infos[frame]['token']
            tokens.append(token)
            occ_path = os.path.join(self.occ_base_path,self.nuplan_infos[frame]["token"],self.nuplan_infos[frame]["token"]+".npz") # h, w, d
            occ = np.load(occ_path)["occ"]
            occ_voxel_grid = self.convert_to_voxel_grid_int(occ)

            # occ change to (200,200,16)
            # print(occ_voxel_grid.shape)
            occs.append(occ_voxel_grid)

            # metas.update(self.get_meta_info(frame))

            # bevmap 可选择
            # if bev exist
            # bev_file_path=os.path.join(self.bev_base_path,self.nuplan_infos[frame]["token"]+".npz")
            # bev_data=np.load(bev_file_path)['gt_bev_masks']
            bev_data=np.zeros(occ_voxel_grid.shape)
            aux_data = np.zeros(bev_data.shape)
            # aux_data=np.zeros(bev_data.shape)
            auxs.append(aux_data)

            # bev_data[1,:,:] = np.any(bev_data[layer_to_merge, :, :], axis=0).astype(int)
            # occ_with_bev = replace_occ_grid_with_bev_nuplan(input_occ_data=occ_data, bevlayout=bev_data)
            # bevmap=self.nBEV1(bev_data, bev_ch_use)
            bevmaps.append(bev_data)

        
        metas.update(scene_token=tokens)
        input_occs = np.stack(occs).astype(np.int64)
        bevmaps= np.stack(bevmaps).astype(bool)

        return input_occs, input_occs, bevmaps, metas, auxs

    def nBEV1(self, data_b,ch_use): # 18,200,200 -> 1,200,200
        data_b = data_b[ch_use]
        mask = data_b>0.01
        cumulative_mask = np.cumsum(mask, axis=0)
        max_index_map = np.argmax(cumulative_mask, axis=0)
        max_index_map = max_index_map / (len(ch_use) -1)
        all_zero_mask = np.all(mask == 0, axis=0)
        max_index_map[all_zero_mask] = -1
        data_b= np.array([max_index_map])
        return data_b

    def get_meta_info(self, idx):
        """Get annotation info according to the given index.

        """
        info = self.nuplan_infos[idx]
        gt_bboxes_3d = info["anns"]['gt_boxes']
        gt_names_3d = info["anns"]['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d: 
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
                # print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        
        if self.with_velocity:
            gt_velocity = info["anns"]['gt_velocity_3d'][:, :2]

            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
       

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            )
        
        return anns_results

    def _project_dynamic_bbox(self, dynamic_mask, data):
        '''We use PIL for projection, while CVT use cv2. The results are
        slightly different due to anti-alias of line, but should be similar.
        '''
        for cls_id, cls_name in enumerate(self.object_classes):
            # pick boxes
            cls_mask = data['gt_labels_3d'] == cls_id
            boxes = data['gt_bboxes_3d'][cls_mask]
            if len(boxes) < 1:
                continue
            # get coordinates on canvas. the order of points matters.
            bottom_corners_lidar = boxes.corners[:, [0, 3, 7, 4], :2]

            bottom_corners_canvas = np.dot(
                # np.pad(flipped_points, ((0, 0), (0, 0), (0, 1)),
                np.pad(bottom_corners_lidar.numpy(), ((0, 0), (0, 0), (0, 1)),
                       constant_values=1.0),
                self.lidar2canvas.T)[..., :2]  # N, 4, xy
            # draw
            points=bottom_corners_canvas
            centers = np.mean(points, axis=1, keepdims=True)
            points[:,:,1]= 2*centers[:,:,1]-points[:,:,1]
            bottom_corners_canvas=points

            render = Image.fromarray(dynamic_mask[cls_id])
            draw = ImageDraw.Draw(render)
            for box in bottom_corners_canvas:
                draw.polygon(
                    box.round().astype(np.int32).flatten().tolist(), fill=1)
            # save
            dynamic_mask[cls_id, :] = np.array(render)[:]
        return dynamic_mask

    def _project_dynamic(self, static_label, data):
        """for dynamic mask, one class per channel
        case 1: data is None, set all values to zeros
        """
        # setup
        ch = len(self.object_classes)
        dynamic_mask = np.zeros((ch, *self.canvas_size), dtype=np.uint8)

        # if int, set ch=object_classes with all zeros; otherwise, project
        if data is not None:
            dynamic_mask = self._project_dynamic_bbox(dynamic_mask, data)

        # combine with static_label
        dynamic_mask = dynamic_mask.transpose(0, 2, 1)
        combined_label = np.concatenate([static_label, dynamic_mask], axis=0)
        return combined_label

    


    def vectormap_pipeline(self, input_dict):
        """
        Process vector map data for input example, using transformation matrices and
        generating annotations.
        """

        lidar2ego = input_dict["lidar2ego"]
        ego2global = input_dict["ego2global"]
        lidar2global = ego2global @ lidar2ego
        # lidar2global = rotation_z_neg90 @ lidar2global
        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_rotation = Quaternion(matrix=lidar2global)

        anns_results = self.vector_map.gen_vectorized_samples_nuplan(
            self.vector_maps[input_dict["location"]], # 'us-nv-las-vegas-strip'
            lidar2global_translation, # [664465.0781396995, 3997749.3183807055, 616.683242571562]
            lidar2global_rotation, # Quaternion(0.6999239997105716, 0.0002516337328280848, 0.004921886360219306, 0.7142003264800971)
        )
        gt_vecs_label = to_tensor(anns_results["gt_vecs_label"])
        if isinstance(anns_results["gt_vecs_pts_loc"], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results["gt_vecs_pts_loc"]
            gt_lines_instance = gt_vecs_pts_loc.instance_list
            gt_map_pts = [np.array(list(line.coords)) for line in gt_lines_instance]
        input_dict["gt_vecs_label"] = DC(gt_vecs_label, cpu_only=False)
        input_dict["gt_vecs_pts_loc"] = DC(gt_map_pts, cpu_only=True)

        # Visualizing BEV map
        # drivable_mask = (
        #     input_dict["gt_masks_bev"][0, ...] + input_dict["gt_masks_bev"][-1, ...]
        # ).astype(bool)

        bev_map = visualize_bev_hdmap(
            input_dict["gt_vecs_pts_loc"].data,
            input_dict["gt_vecs_label"].data,
            self.canvas_size,
            # vis_format="polyline_pts",
            # drivable_mask=drivable_mask,
            # nuplan=True,
        )
        bev_map = bev_map.transpose(2, 0, 1)
        return bev_map

@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar_Nuplan_dep:
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            occ_dataroot=None,
            times=5,
            # quantize_size=(400,400,32),
            quantize_size=(200,200,16),
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts',
            debug=False 
        ):

        with open(imageset, 'rb') as f:
            pkl_data = pickle.load(f)
        print(f"=> loaded pkl_data from {imageset}")

        # self.nusc_infos = pkl_data['infos']
        self.nuplan_infos = pkl_data['infos']
        # self.nuplan_infos = preprocess_infos_to_dict(pkl_data)
        self.occ_base_path = occ_dataroot
        self.scene_names = pkl_data['scene_tokens']
        self.scene_lens = [len(sn) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.quantize_size=quantize_size

        self.maps = {}
        LOCATIONS = ['us-ma-boston', 'us-nv-las-vegas-strip', 'sg-one-north', 'us-pa-pittsburgh-hazelwood']

        self.classes= ['lane_polygons','intersections','generic_drivable_areas','carpark_areas','lane_group_connectors','lane_group_polygons','road_segments']
        self.object_classes = ['vehicle','bicycle','pedestrian','traffic_cone','barrier','czone_sign','generic_object']
        self.times = times
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR
        print(f"quantize_size: {self.quantize_size}")
        # for 200 resolution
        if self.quantize_size == (400,400,32):
            xbound=[-50,50,0.25]
            ybound=[-50,50,0.25]
            zbound=[-3,5,0.25]
            voxel_size=0.25
        elif self.quantize_size == (200,200,16):
            xbound=[-50,50,0.5]
            ybound=[-50,50,0.5]
            zbound=[-3,5,0.5]
            voxel_size=0.5
        else:
            raise ValueError(f"quantize_size should be (400,400,32) or (200,200,16), but got {self.quantize_size}")
        self.point_cloud_range = [xbound[0],ybound[0],zbound[0],xbound[1],ybound[1],zbound[1]]
        self.occ_xdim = int((xbound[1] - xbound[0]) / voxel_size)
        self.occ_ydim = int((ybound[1] - ybound[0]) / voxel_size)
        self.occ_zdim = int((zbound[1] - zbound[0]) / voxel_size)
        self.voxel_num = self.occ_xdim*self.occ_ydim*self.occ_zdim
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.use_valid_flag=True

        self.lidar2canvas = np.array([
            [canvas_h / patch_h, 0, canvas_h / 2],
            [0, canvas_w / patch_w, canvas_w / 2],
            [0, 0, 1]
        ])

        self.start_on_keyframe = True
        self.start_on_firstframe = False
        if "clip_infos" in pkl_data:
            self.clip_infos = pkl_data["clip_infos"]
        else:
            if debug:
                self.clip_infos=self.build_clips(self.nuplan_infos, pkl_data['scene_tokens'][:1])
            else:
                self.clip_infos=self.build_clips(self.nuplan_infos, pkl_data['scene_tokens'])
        print(f"=> clip_infos: {len(self.clip_infos)} =={self.clip_infos[0]}")
        
        # 用于存储上次成功获取数据的index，用于重试机制
        self.last_valid_index = 0


    
    def obtain_points_label(self, occ):
        occ_index, occ_cls = occ[:, 0], occ[:, 1]
        occ = np.ones(self.voxel_num, dtype=np.int8)*11
        occ[occ_index[:]] = occ_cls  # (voxel_num)
        points = []
        for i in range(len(occ_index)):
            indice = occ_index[i]
            x = indice % self.occ_xdim
            y = (indice // self.occ_xdim) % self.occ_xdim
            z = indice // (self.occ_xdim*self.occ_xdim)
            point_x = (x + 0.5) / self.occ_xdim * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
            point_y = (y + 0.5) / self.occ_ydim * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
            point_z = (z + 0.5) / self.occ_zdim * (self.point_cloud_range[5] - self.point_cloud_range[2]) + self.point_cloud_range[2]
            points.append([point_x, point_y, point_z])
        
        points = np.stack(points)
        point_label = occ_cls
        points_with_label = np.concatenate([points, point_label[:, None]], axis=-1)
        return points_with_label
    
    def convert_to_voxel_grid_int(self, voxels_):
        voxel = np.zeros((800, 800, 64), dtype=np.int32)
        voxel[voxels_[:, 0].astype(np.int32), voxels_[:, 1].astype(np.int32), voxels_[:, 2].astype(np.int32)] = voxels_[:, 3].astype(np.int32)+1
        return voxel
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.clip_infos)

    def __getitem__(self, index):
        # index = index % len(self.scene_lens)
        # scene_name = self.scene_names[index]
        # scene_len = self.scene_lens[index]
        # # idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        # idx=0
        # self.return_len=scene_len
        clip = self.clip_infos[index]
        occs = []
        tokens=[]
        bevmaps=[]
        metas = {}
        

        for frame in clip:
            token = self.nuplan_infos[frame]['token']
            tokens.append(token)
            occ_path = os.path.join(self.occ_base_path,self.nuplan_infos[frame]["token"]+".npy") # h, w, d
            occ = np.load(occ_path)

            # occ change to (200,200,16)
            occ_voxel_grid = occ
            occs.append(occ_voxel_grid)

            # metas.update(self.get_meta_info(frame))

            # bevmap=self.get_map_info(metas,frame) # 获得 bevmap
            # bevmaps.append(bevmap)
        
        metas.update(scene_token=tokens)
        # metas.update(scene_name=scene_name)
        input_occs = np.stack(occs).astype(np.int64)
        # bevmaps= np.stack(bevmaps).astype(bool)
        # print(f"=>input_occs shape: {input_occs.shape}")

        return input_occs, input_occs, metas


    def get_meta_info(self, idx):
        """Get annotation info according to the given index.

        """
        info = self.nuplan_infos[idx]
        gt_bboxes_3d = info["anns"]['gt_boxes']
        gt_names_3d = info["anns"]['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d: 
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
                # print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        
        if self.with_velocity:
            gt_velocity = info["anns"]['gt_velocity_3d'][:, :2]

            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
       

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            # attr_labels=attr_labels,
            # fut_valid_flag=fut_valid_flag,
            )
        
        return anns_results

@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar_Nuplan:
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            occ_dataroot=None,
            times=5,
            # quantize_size=(400,400,32),
            quantize_size=(200,200,16),
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts',
            debug=False 
        ):

        with open(imageset, 'rb') as f:
            pkl_data = pickle.load(f)
        print(f"=> loaded pkl_data from {imageset}")

        # self.nusc_infos = pkl_data['infos']
        self.nuplan_infos = pkl_data['infos']
        # self.nuplan_infos = preprocess_infos_to_dict(pkl_data)
        self.occ_base_path = occ_dataroot
        self.scene_names = pkl_data['scene_tokens']
        self.scene_lens = [len(sn) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.quantize_size=quantize_size
        self.debug = debug

        self.maps = {}
        LOCATIONS = ['us-ma-boston', 'us-nv-las-vegas-strip', 'sg-one-north', 'us-pa-pittsburgh-hazelwood']

        self.classes= ['lane_polygons','intersections','generic_drivable_areas','carpark_areas','lane_group_connectors','lane_group_polygons','road_segments']
        self.object_classes = ['vehicle','bicycle','pedestrian','traffic_cone','barrier','czone_sign','generic_object']
        self.times = times
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR
        print(f"quantize_size: {self.quantize_size}")
        # for 200 resolution
        if self.quantize_size == (400,400,32):
            xbound=[-50,50,0.25]
            ybound=[-50,50,0.25]
            zbound=[-3,5,0.25]
            voxel_size=0.25
        elif self.quantize_size == (200,200,16):
            xbound=[-50,50,0.5]
            ybound=[-50,50,0.5]
            zbound=[-3,5,0.5]
            voxel_size=0.5
        else:
            raise ValueError(f"quantize_size should be (400,400,32) or (200,200,16), but got {self.quantize_size}")
        self.point_cloud_range = [xbound[0],ybound[0],zbound[0],xbound[1],ybound[1],zbound[1]]
        self.occ_xdim = int((xbound[1] - xbound[0]) / voxel_size)
        self.occ_ydim = int((ybound[1] - ybound[0]) / voxel_size)
        self.occ_zdim = int((zbound[1] - zbound[0]) / voxel_size)
        self.voxel_num = self.occ_xdim*self.occ_ydim*self.occ_zdim
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.use_valid_flag=True

        self.lidar2canvas = np.array([
            [canvas_h / patch_h, 0, canvas_h / 2],
            [0, canvas_w / patch_w, canvas_w / 2],
            [0, 0, 1]
        ])

        self.start_on_keyframe = True
        self.start_on_firstframe = False
        self.clip_infos = pkl_data["clip_infos"]

        print(f"=> clip_infos: {len(self.clip_infos)} =={self.clip_infos[0]}")
        
        # 用于存储上次成功获取数据的index，用于重试机制
        self.last_valid_index = 0

    # ------------------------- NEW: helpers for robust missing-file handling -------------------------
    def _occ_path_from_frame(self, frame_idx):
        token = self.nuplan_infos[frame_idx]['token']
        return os.path.join(self.occ_base_path, token + ".npy")

    def _clip_has_all_occs(self, clip):
        """Return True if every frame in this clip has its OCC .npy present."""
        for frame in clip:
            path = self._occ_path_from_frame(frame)
            if not os.path.isfile(path):
                return False
        return True

    def _pick_clip_index_with_retry(self, index, max_tries=1000):
        """If the clip at `index` is missing any OCCs, try up to max_tries random clips.
        If still missing, fall back to `self.last_valid_index`.
        Returns: (chosen_index, used_fallback: bool)
        """
        total = len(self.clip_infos)

        # Fast path: the requested index works
        if self._clip_has_all_occs(self.clip_infos[index]):
            return index, False

        # Retry randomly up to max_tries
        for _ in range(max_tries):
            rnd = random.randint(0, total - 1)
            if self._clip_has_all_occs(self.clip_infos[rnd]):
                if self.debug:
                    print(f"[retry-ok] Missing OCC at index {index}, using random valid index {rnd}.")
                return rnd, True

        # Fall back to last_valid_index
        if self.debug:
            print(f"[fallback] Could not find valid clip after {max_tries} tries; using last_valid_index={self.last_valid_index}.")
        return self.last_valid_index, True
    # -----------------------------------------------------------------------------------------------

    def obtain_points_label(self, occ):
        occ_index, occ_cls = occ[:, 0], occ[:, 1]
        occ = np.ones(self.voxel_num, dtype=np.int8)*11
        occ[occ_index[:]] = occ_cls  # (voxel_num)
        points = []
        for i in range(len(occ_index)):
            indice = occ_index[i]
            x = indice % self.occ_xdim
            y = (indice // self.occ_xdim) % self.occ_ydim  # fix: use occ_ydim
            z = indice // (self.occ_xdim*self.occ_xdim)
            point_x = (x + 0.5) / self.occ_xdim * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
            point_y = (y + 0.5) / self.occ_ydim * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
            point_z = (z + 0.5) / self.occ_zdim * (self.point_cloud_range[5] - self.point_cloud_range[2]) + self.point_cloud_range[2]
            points.append([point_x, point_y, point_z])
        
        points = np.stack(points)
        point_label = occ_cls
        points_with_label = np.concatenate([points, point_label[:, None]], axis=-1)
        return points_with_label
    
    def convert_to_voxel_grid_int(self, voxels_):
        voxel = np.zeros((800, 800, 64), dtype=np.int32)
        voxel[voxels_[:, 0].astype(np.int32), voxels_[:, 1].astype(np.int32), voxels_[:, 2].astype(np.int32)] = voxels_[:, 3].astype(np.int32)+1
        return voxel
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.clip_infos)

    def __getitem__(self, index):
        # Robustly pick a clip index with retry/fallback
        chosen_index, used_fallback = self._pick_clip_index_with_retry(index)
        clip = self.clip_infos[chosen_index]

        occs = []
        tokens=[]
        bevmaps=[]
        metas = {}
        
        # Now we are guaranteed that all OCC files exist for this clip
        for frame in clip:
            token = self.nuplan_infos[frame]['token']
            tokens.append(token)
            occ_path = os.path.join(self.occ_base_path, token+".npy")  # h, w, d
            occ = np.load(occ_path)

            # occ change to (200,200,16) (assumed preprocessed)
            occ_voxel_grid = occ
            occs.append(occ_voxel_grid)

            # metas.update(self.get_meta_info(frame))
            # bevmap=self.get_map_info(metas,frame) # 获得 bevmap
            # bevmaps.append(bevmap)
        
        # Update last_valid_index only after a successful load
        self.last_valid_index = chosen_index

        metas.update(scene_token=tokens)
        # metas.update(scene_name=scene_name)
        input_occs = np.stack(occs).astype(np.int64)
        # bevmaps= np.stack(bevmaps).astype(bool)
        
        if self.debug and used_fallback:
            print(f"[info] __getitem__ used fallback/random selection; final index={chosen_index}.")

        return input_occs, input_occs, metas


    def get_meta_info(self, idx):
        """Get annotation info according to the given index.

        """
        info = self.nuplan_infos[idx]
        gt_bboxes_3d = info["anns"]['gt_boxes']
        gt_names_3d = info["anns"]['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d: 
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
                # print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        
        if self.with_velocity:
            gt_velocity = info["anns"]['gt_velocity_3d'][:, :2]

            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
       

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            # attr_labels=attr_labels,
            # fut_valid_flag=fut_valid_flag,
            )
        
        return anns_results




@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar_Nuplan_bev:
    AUX_DATA_CH = {
        # "visibility": 1,
        "center_offset": 2,
        "center_ohw": 4,
        "height": 1,
    }
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            occ_dataroot=None,
            times=5,
            # quantize_size=(400,400,32),
            quantize_size=(200,200,16),
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts',
            debug=False 
        ):

        with open(imageset, 'rb') as f:
            pkl_data = pickle.load(f)
        print(f"=> loaded pkl_data from {imageset}")

        self.nuplan_infos = pkl_data['infos']
        self.occ_base_path = occ_dataroot
        self.scene_names = pkl_data['scene_tokens']
        self.scene_lens = [len(sn) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.quantize_size=quantize_size
        self.aux_data = self.AUX_DATA_CH

        self.maps = {}

        self.classes= ['intersections','generic_drivable_areas','walkways','carpark_areas','crosswalks','lane_group_connectors','lane_groups_polygons','road_segments']
        self.object_classes = ['vehicle','bicycle','pedestrian','traffic_cone','barrier','czone_sign','generic_object']
        self.times = times
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR
        print(f"quantize_size: {self.quantize_size}")
        # for 200 resolution
        if self.quantize_size == (400,400,32):
            xbound=[-50,50,0.25]
            ybound=[-50,50,0.25]
            zbound=[-3,5,0.25]
            voxel_size=0.25
        elif self.quantize_size == (200,200,16):
            xbound=[-50,50,0.5]
            ybound=[-50,50,0.5]
            zbound=[-3,5,0.5]
            voxel_size=0.5
        else:
            raise ValueError(f"quantize_size should be (400,400,32) or (200,200,16), but got {self.quantize_size}")
        self.point_cloud_range = [xbound[0],ybound[0],zbound[0],xbound[1],ybound[1],zbound[1]]
        self.occ_xdim = int((xbound[1] - xbound[0]) / voxel_size)
        self.occ_ydim = int((ybound[1] - ybound[0]) / voxel_size)
        self.occ_zdim = int((zbound[1] - zbound[0]) / voxel_size)
        self.voxel_num = self.occ_xdim*self.occ_ydim*self.occ_zdim
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.use_valid_flag=True

        self.lidar2canvas = np.array([
            [canvas_h / patch_h, 0, canvas_h / 2],
            [0, canvas_w / patch_w, canvas_w / 2],
            [0, 0, 1]
        ])

        self.start_on_keyframe = True
        self.start_on_firstframe = False

        self.mapdb = GPKGMapsDB("nuplan-maps-v1.0", f"/lpai/volumes/ad-lmm-data-proc-bd-ga/hzhu/data/maps")
        self.vector_maps = {}

        if debug:
            self.clip_infos=self.build_clips(self.nuplan_infos, pkl_data['scene_tokens'][:100])
        else:
            self.clip_infos=self.build_clips(self.nuplan_infos, pkl_data['scene_tokens'])
        print(f"=> clip_infos: {len(self.clip_infos)}")


    def fliter_clips(self,clip):
        for frame in clip:
            if self.nuplan_infos[frame]["token"] is None:
                print(f"Warning: {self.nuplan_infos[frame]['token']} is None")
                return 0
            file_path = os.path.join(self.occ_base_path,self.nuplan_infos[frame]["token"]+".npy")
            if not os.path.exists(file_path):  # 检查文件是否存在
                print(f"Warning: {file_path} does not exist")
                return 0
        return 1
            
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
        self.token_data_dict = {item['token']: idx for idx, item in enumerate(data_infos)}
        all_clips = []
        for i, scene in enumerate(scene_tokens):
            print(f"building clips for scene: {i}/{len(scene_tokens)}")
            for start in range(len(scene) - self.return_len + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + self.return_len]]
                if self.fliter_clips(clip)==0:
                    continue
                all_clips.append(clip)
                if self.start_on_firstframe:
                    break
        return all_clips
    
    def obtain_points_label(self, occ):
        occ_index, occ_cls = occ[:, 0], occ[:, 1]
        occ = np.ones(self.voxel_num, dtype=np.int8)*11
        occ[occ_index[:]] = occ_cls  # (voxel_num)
        points = []
        for i in range(len(occ_index)):
            indice = occ_index[i]
            x = indice % self.occ_xdim
            y = (indice // self.occ_xdim) % self.occ_xdim
            z = indice // (self.occ_xdim*self.occ_xdim)
            point_x = (x + 0.5) / self.occ_xdim * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
            point_y = (y + 0.5) / self.occ_ydim * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
            point_z = (z + 0.5) / self.occ_zdim * (self.point_cloud_range[5] - self.point_cloud_range[2]) + self.point_cloud_range[2]
            points.append([point_x, point_y, point_z])
        
        points = np.stack(points)
        point_label = occ_cls
        points_with_label = np.concatenate([points, point_label[:, None]], axis=-1)
        return points_with_label
    
    def convert_to_voxel_grid_int(self, voxels_):
        voxel = np.zeros((800, 800, 64), dtype=np.int32)
        voxel[voxels_[:, 0].astype(np.int32), voxels_[:, 1].astype(np.int32), voxels_[:, 2].astype(np.int32)] = voxels_[:, 3].astype(np.int32)+1
        return voxel
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.clip_infos)

    def __getitem__(self, index):
        # index = index % len(self.scene_lens)
        # scene_name = self.scene_names[index]
        # scene_len = self.scene_lens[index]
        # # idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        # idx=0
        # self.return_len=scene_len
        clip = self.clip_infos[index]
        occs = []
        tokens=[]
        bevmaps=[]
        metas = {}
        scene_token = self.nuplan_infos[clip[0]]['scene_token']

        for frame_idx in clip:
            token = self.nuplan_infos[frame_idx]['token']
            tokens.append(token)
            occ_path = os.path.join(self.occ_base_path,self.nuplan_infos[frame_idx]["token"]+".npy") # h, w, d
            occ = np.load(occ_path) # 200, 200, 16

            occs.append(occ)

            metas.update(self.get_meta_info(frame_idx))

            bevmap=self.get_map_info(frame_idx) # 获得 bevmap
            bevmaps.append(bevmap['gt_masks_bev'])
        
        metas.update(tokens=tokens)
        metas.update(token_idxs=clip)
        metas.update(scene_token=scene_token)
        input_occs = np.stack(occs).astype(np.int64)
        bevmaps= np.stack(bevmaps).astype(bool)
        # print(f"=>input_occs shape: {input_occs.shape}")

        return input_occs,input_occs, metas,bevmaps


    def get_meta_info(self, idx):
        """Get annotation info according to the given index.

        """
        info = self.nuplan_infos[idx]
        gt_bboxes_3d = info["anns"]['gt_boxes']
        gt_names_3d = info["anns"]['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d: 
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
                # print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        
        if self.with_velocity:
            gt_velocity = info["anns"]['gt_velocity_3d'][:, :2]

            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
       

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            token=info["token"],
            scene_token=info["scene_token"],
            # attr_labels=attr_labels,
            # fut_valid_flag=fut_valid_flag,
            )
        
        return anns_results

    def _project_dynamic_bbox(self, dynamic_mask, data_info):
        '''We use PIL for projection, while CVT use cv2. The results are
        slightly different due to anti-alias of line, but should be similar.
        '''
        # handle gt_boxes_3d, gt_names_3d, gt_labels_3d
        # info = self.data_infos[index]
        gt_bboxes_3d = data_info['anns']['gt_boxes']
        gt_names_3d = data_info['anns']['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            # 已经有 3d 的速度, 直接使用就行, 不用 z 轴分量
            gt_velocity = data_info['anns']['gt_velocity_3d'][:, :2]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        
        for cls_id, cls_name in enumerate(self.object_classes):
            # pick boxes
            cls_mask = gt_labels_3d == cls_id
            boxes = gt_bboxes_3d[cls_mask]

            if len(boxes) < 1:
                continue
            bottom_corners_lidar = boxes.corners[:, [0, 3, 7, 4], :2]

            bottom_corners_canvas = np.dot(
                np.pad(bottom_corners_lidar.numpy(), ((0, 0), (0, 0), (0, 1)),
                       constant_values=1.0),
                self.lidar2canvas.T)[..., :2]  # N, 4, xy
            # draw
            # points=bottom_corners_canvas
            # centers = np.mean(points, axis=1, keepdims=True)
            # points[:,:,1]= 2*centers[:,:,1]-points[:,:,1]
            # bottom_corners_canvas=points

            render = Image.fromarray(dynamic_mask[cls_id])
            draw = ImageDraw.Draw(render)
            for box in bottom_corners_canvas:
                draw.polygon(
                    box.round().astype(np.int32).flatten().tolist(), fill=1)
            # save
            dynamic_mask[cls_id, :] = np.array(render)[:]
        return dynamic_mask

    def _project_dynamic(self, static_label, data):
        """for dynamic mask, one class per channel
        case 1: data is None, set all values to zeros
        """
        # setup
        ch = len(self.object_classes)
        dynamic_mask = np.zeros((ch, *self.canvas_size), dtype=np.uint8)

        # if int, set ch=object_classes with all zeros; otherwise, project
        if data is not None:
            dynamic_mask = self._project_dynamic_bbox(dynamic_mask, data)

        # combine with static_label
        dynamic_mask = dynamic_mask.transpose(0, 2, 1)
        combined_label = np.concatenate([static_label, dynamic_mask], axis=0)
        return combined_label


    def get_map_info(self, index):
        # info = data_info
        data_info = self.nuplan_infos[index]
        box_info = self.get_meta_info(index)
        lidar2ego = data_info["lidar2ego"]
        ego2global = data_info["ego2global"]
        lidar2global = ego2global @ lidar2ego

        map_pose = lidar2global[:2, 3]
        patch_box = (
            map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        
        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])  # angle between v and x-axis
        patch_angle = yaw / np.pi * 180

        mappings = {}
        location = data_info["description"]["map_version"]
        # vector map
        fixed_ptsnum_per_line = -1
        padding_value = -10000
        vector_map_api = NuPlanMapWrapper(self.mapdb, map_name=location)
        self.vector_maps[location] = vector_map_api

        self.vector_map = VectorizedLocalMap(
            self.data_path,# '/baai-cwm-1/baai_cwm_ml/algorithm/bohan.li/code/DriveArena/WorldDreamer/data/dataset/nuplan-v1.1'
            patch_size=self.patch_size, # (100.0, 100.0)
            map_classes=["divider", "ped_crossing", "boundary"],
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,# -1
            padding_value=padding_value, # -10000
            nuplan_map_api = self.vector_maps
        )
        map_explorer = NuPlanMapExplorer(map_api=vector_map_api)
        self.maps[location] = map_explorer

        for name in self.classes:
            mappings[name] = [name]
        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        # cut semantics from nuscenesMap
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=self.classes,
            output_size=self.canvas_size,
        )

        masks = masks.transpose(0, 2, 1)  # TODO why need transpose here?
        masks = masks.astype(np.bool_)
        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.int64)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        # add dynamic labels
        bev_maps = dict()
        bev_maps["gt_masks_bev_static"]= labels
        final_labels = self._project_dynamic(labels, data_info)
        aux_labels = self._get_dynamic_aux(data=box_info)
        bev_maps['gt_aux_bev']=aux_labels
        bev_maps["gt_masks_bev"] = final_labels

        # add line dividers
        bev_lane = self.vectormap_pipeline(data_info)
        bev_maps["bev_lane"] = bev_lane
        bev_maps["gt_masks_bev"] = np.concatenate([bev_maps["gt_masks_bev"], bev_lane, aux_labels], axis=0)

        return bev_maps["gt_masks_bev"]
    

    def _get_dynamic_aux(self, data: Dict[str, Any] = None) -> Any:
        '''aux data
        case 1: self.aux_data is None, return None
        case 2: data=None, set all values to zeros
        '''
        if self.aux_data is None:
            print("aux_data is None!")
            return None  # there is no aux_data

        aux_ch = sum([self.AUX_DATA_CH[aux_k] for aux_k in self.aux_data])
        if aux_ch == 0:  # there is no available aux_data
            if len(self.aux_data) != 0:
                print(f"Your aux_data: {self.aux_data} is not available")
            return None

        aux_mask = np.zeros((*self.canvas_size, aux_ch), dtype=np.float32)
        if data is not None:
            aux_mask = self._get_dynamic_aux_bbox(aux_mask, data)

        # transpose x,y and channel first format
        aux_mask = aux_mask.transpose(2, 1, 0)
        return aux_mask

    def _get_dynamic_aux_bbox(self, aux_mask, data: Dict[str, Any]):
        '''Three aux data (7 channels in total), class-agnostic:
        1. visibility, 1-channel
        2. center-offset, 2-channel
        3. height/2, width/2, orientation, 4-channel, on bev canvas
        4. height of bbox, in lidar coordinate
        '''
        for _idx in range(len(data['gt_bboxes_3d'])):
            box = data['gt_bboxes_3d'][_idx]
            # get canvas coordinates
            # fmt:off
            _box_lidar = np.concatenate([
                box.corners[:, [0, 3, 7, 4], :2].numpy(),
                box.bottom_center[:, None, :2].numpy(),  # center
                box.corners[:, [4, 7], :2].mean(dim=1)[:, None].numpy(),  # front
                box.corners[:, [0, 4], :2].mean(dim=1)[:, None].numpy(),  # left
            ], axis=1)
            # fmt:on
            _box_canvas = np.dot(
                np.pad(_box_lidar, ((0, 0), (0, 0), (0, 1)), constant_values=1.0),
                self.lidar2canvas.T)[..., :2]  # N, 4, xy
            # in canvas coordinates
            box_canvas = _box_canvas[0, :4]
            center_canvas = _box_canvas[0, 4:5]
            front_canvas = _box_canvas[0, 5:6]
            left_canvas = _box_canvas[0, 6:7]
            # render mask
            render = Image.fromarray(np.zeros(self.canvas_size, dtype=np.uint8))
            draw = ImageDraw.Draw(render)
            draw.polygon(
                box_canvas.round().astype(np.int32).flatten().tolist(),
                fill=1)
            # construct
            tmp_mask = np.array(render) > 0
            coords = np.stack(np.meshgrid(
                np.arange(self.canvas_size[1]), np.arange(self.canvas_size[0])
            ), -1).astype(np.float32)
            _cur_ch = 0
            if "visibility" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['visibility']
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = data['visibility'][_idx]
                _cur_ch = _ch_stop
            if "center_offset" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['center_offset']
                center_offset = coords[tmp_mask] - center_canvas
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = center_offset
                _cur_ch = _ch_stop
            if "center_ohw" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['center_ohw']
                height = np.linalg.norm(front_canvas - center_canvas)
                width = np.linalg.norm(left_canvas - center_canvas)
                # yaw = box.yaw  # scaling aspect ratio, yaw does not change
                v = ((front_canvas - center_canvas) / (
                    np.linalg.norm(front_canvas - center_canvas) + 1e-6))[0]
                # yaw = - np.arctan2(v[1], v[0])  # add negative, align with mmdet coord
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = np.array([
                    height, width, v[0], v[1]])[None]
                _cur_ch = _ch_stop
            if "height" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['height']
                bbox_height = box.height.item()  # in lidar coordinate
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = np.array([
                    bbox_height])[None]
                _cur_ch = _ch_stop
        return aux_mask
    
    def vectormap_pipeline(self, input_dict):
        """
        Process vector map data for input example, using transformation matrices and
        generating annotations.
        """

        lidar2ego = input_dict["lidar2ego"]
        ego2global = input_dict["ego2global"]
        lidar2global = ego2global @ lidar2ego
        # lidar2global = rotation_z_neg90 @ lidar2global
        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_rotation = Quaternion(matrix=lidar2global)

        anns_results = self.vector_map.gen_vectorized_samples_nuplan(
            self.vector_maps[input_dict['description']["map_version"]], # 'us-nv-las-vegas-strip'
            lidar2global_translation, # [664465.0781396995, 3997749.3183807055, 616.683242571562]
            lidar2global_rotation, # Quaternion(0.6999239997105716, 0.0002516337328280848, 0.004921886360219306, 0.7142003264800971)
        )
        gt_vecs_label = to_tensor(anns_results["gt_vecs_label"])
        if isinstance(anns_results["gt_vecs_pts_loc"], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results["gt_vecs_pts_loc"]
            gt_lines_instance = gt_vecs_pts_loc.instance_list
            gt_map_pts = [np.array(list(line.coords)) for line in gt_lines_instance]
        input_dict["gt_vecs_label"] = DC(gt_vecs_label, cpu_only=False)
        input_dict["gt_vecs_pts_loc"] = DC(gt_map_pts, cpu_only=True)


        bev_map = visualize_bev_hdmap(
            input_dict["gt_vecs_pts_loc"].data,
            input_dict["gt_vecs_label"].data,
            self.canvas_size,
        )
        bev_map = bev_map.transpose(2, 0, 1)

        return bev_map

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


@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar_Nuplan_pro:
    # 删选数据之后的 nuplan Occupancy
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            occ_dataroot=None,
            times=5,
            # quantize_size=(400,400,32),
            quantize_size=(200,200,16),
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts',
            debug=False 
        ):

        with open(imageset, 'rb') as f:
            pkl_data = pickle.load(f)
        print(f"=> loaded pkl_data from {imageset}")

        # self.nusc_infos = pkl_data['infos']
        self.nuplan_infos = pkl_data['infos']
        # self.nuplan_infos = preprocess_infos_to_dict(pkl_data)
        self.occ_base_path = occ_dataroot
        self.scene_names_orig = pkl_data['scene_tokens']
        self.scene_names_static = pkl_data['scene_tokens_with_stationary_segments']
        # 1. 先把所有要去除的元素收集到一个 set 里
        to_remove = set()
        for sublist in self.scene_names_static:
            to_remove.update(sublist)

        # 2. 对 scene_tokens 里的每个子列表，去除这些元素
        scene_tokens_without_stationary_elements = [
            [item for item in sublist if item not in to_remove]
            for sublist in self.scene_names_orig
        ]
        self.scene_lists = scene_tokens_without_stationary_elements
        print(f"scene_lists: {len(self.scene_lists)}")

        self.scene_lens = [len(sn) for sn in self.scene_lists]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.quantize_size=quantize_size

        self.maps = {}

        self.classes= ['lane_polygons','intersections','generic_drivable_areas','carpark_areas','lane_group_connectors','lane_group_polygons','road_segments']
        self.object_classes = ['vehicle','bicycle','pedestrian','traffic_cone','barrier','czone_sign','generic_object']
        self.times = times
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR
        print(f"quantize_size: {self.quantize_size}")
        # for 200 resolution
        if self.quantize_size == (400,400,32):
            xbound=[-50,50,0.25]
            ybound=[-50,50,0.25]
            zbound=[-3,5,0.25]
            voxel_size=0.25
        elif self.quantize_size == (200,200,16):
            xbound=[-50,50,0.5]
            ybound=[-50,50,0.5]
            zbound=[-3,5,0.5]
            voxel_size=0.5
        else:
            raise ValueError(f"quantize_size should be (400,400,32) or (200,200,16), but got {self.quantize_size}")
        self.point_cloud_range = [xbound[0],ybound[0],zbound[0],xbound[1],ybound[1],zbound[1]]
        self.occ_xdim = int((xbound[1] - xbound[0]) / voxel_size)
        self.occ_ydim = int((ybound[1] - ybound[0]) / voxel_size)
        self.occ_zdim = int((zbound[1] - zbound[0]) / voxel_size)
        self.voxel_num = self.occ_xdim*self.occ_ydim*self.occ_zdim
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.use_valid_flag=True

        self.lidar2canvas = np.array([
            [canvas_h / patch_h, 0, canvas_h / 2],
            [0, canvas_w / patch_w, canvas_w / 2],
            [0, 0, 1]
        ])

        self.start_on_keyframe = True
        self.start_on_firstframe = False

        if debug:
            self.clip_infos=self.build_clips(self.nuplan_infos, self.scene_lists[:1])
        else:
            self.clip_infos=self.build_clips(self.nuplan_infos, self.scene_lists)
        print(f"=> clip_infos: {len(self.clip_infos)}")


    def fliter_clips(self,clip):
        for frame in clip:
            if self.nuplan_infos[frame]["token"] is None:
                print(f"Warning: {self.nuplan_infos[frame]['token']} is None")
                return 0
            file_path = os.path.join(self.occ_base_path,self.nuplan_infos[frame]["token"]+".npy")
            if not os.path.exists(file_path):  # 检查文件是否存在
                print(f"Warning: {file_path} does not exist")
                return 0
        return 1
            
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
        self.token_data_dict = {item['token']: idx for idx, item in enumerate(data_infos)}
        all_clips = []
        for i, scene in enumerate(scene_tokens):
            print(f"building clips for scene: {i}/{len(scene_tokens)}")
            for start in range(len(scene) - self.return_len + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + self.return_len]]
                if self.fliter_clips(clip)==0:
                    continue
                all_clips.append(clip)
                if self.start_on_firstframe:
                    break
        return all_clips
    
    def obtain_points_label(self, occ):
        occ_index, occ_cls = occ[:, 0], occ[:, 1]
        occ = np.ones(self.voxel_num, dtype=np.int8)*11
        occ[occ_index[:]] = occ_cls  # (voxel_num)
        points = []
        for i in range(len(occ_index)):
            indice = occ_index[i]
            x = indice % self.occ_xdim
            y = (indice // self.occ_xdim) % self.occ_xdim
            z = indice // (self.occ_xdim*self.occ_xdim)
            point_x = (x + 0.5) / self.occ_xdim * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
            point_y = (y + 0.5) / self.occ_ydim * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
            point_z = (z + 0.5) / self.occ_zdim * (self.point_cloud_range[5] - self.point_cloud_range[2]) + self.point_cloud_range[2]
            points.append([point_x, point_y, point_z])
        
        points = np.stack(points)
        point_label = occ_cls
        points_with_label = np.concatenate([points, point_label[:, None]], axis=-1)
        return points_with_label
    
    def convert_to_voxel_grid_int(self, voxels_):
        voxel = np.zeros((800, 800, 64), dtype=np.int32)
        voxel[voxels_[:, 0].astype(np.int32), voxels_[:, 1].astype(np.int32), voxels_[:, 2].astype(np.int32)] = voxels_[:, 3].astype(np.int32)+1
        return voxel
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.clip_infos)

    def __getitem__(self, index):
        # index = index % len(self.scene_lens)
        # scene_name = self.scene_names[index]
        # scene_len = self.scene_lens[index]
        # # idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        # idx=0
        # self.return_len=scene_len
        clip = self.clip_infos[index]
        occs = []
        tokens=[]
        bevmaps=[]
        metas = {}
        

        for frame in clip:
            token = self.nuplan_infos[frame]['token']
            tokens.append(token)
            occ_path = os.path.join(self.occ_base_path,self.nuplan_infos[frame]["token"]+".npy") # h, w, d
            occ = np.load(occ_path)

            # occ change to (200,200,16)
            occ_voxel_grid = occ
            occs.append(occ_voxel_grid)

            # metas.update(self.get_meta_info(frame))

            # bevmap=self.get_map_info(metas,frame) # 获得 bevmap
            # bevmaps.append(bevmap)
        
        metas.update(scene_token=tokens)
        # metas.update(scene_name=scene_name)
        input_occs = np.stack(occs).astype(np.int64)
        # bevmaps= np.stack(bevmaps).astype(bool)
        # print(f"=>input_occs shape: {input_occs.shape}")

        return input_occs, input_occs, metas


    def get_meta_info(self, idx):
        """Get annotation info according to the given index.

        """
        info = self.nuplan_infos[idx]
        gt_bboxes_3d = info["anns"]['gt_boxes']
        gt_names_3d = info["anns"]['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d: 
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
                # print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        
        if self.with_velocity:
            gt_velocity = info["anns"]['gt_velocity_3d'][:, :2]

            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
       

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            # attr_labels=attr_labels,
            # fut_valid_flag=fut_valid_flag,
            )
        
        return anns_results


    
@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar_Nuplan_pro_new_occ_bev:
    AUX_DATA_CH = {
        # "visibility": 1,
        "center_offset": 2,
        "center_ohw": 4,
        "height": 1,
    }
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            occ_dataroot=None,
            bev_dataroot=None,
            times=5,
            # quantize_size=(400,400,32),
            quantize_size=(200,200,16),
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts',
            debug=False,
            data_version='orig',# orig, ego_static, ego_dynamic
            map_path="data/maps",
        ):

        with open(imageset, 'rb') as f:
            pkl_data = pickle.load(f)
        print(f"=> loaded pkl_data from {imageset}")

        self.nuplan_infos = pkl_data['infos']
        # self.nuplan_infos = preprocess_infos_to_dict(pkl_data)
        self.occ_base_path = occ_dataroot
        self.bev_base_path = bev_dataroot

        if data_version=='orig':
            self.scene_lists=pkl_data['scene_tokens']
        elif data_version=='ego_static':
            self.scene_lists=pkl_data['scene_tokens_with_stationary_segments']
        elif data_version=='ego_dynamic':
            self.scene_names_orig = pkl_data['scene_tokens']
            self.scene_names_static = pkl_data['scene_tokens_with_stationary_segments']
            # 1. 先把所有要去除的元素收集到一个 set 里
            to_remove = set()
            for sublist in self.scene_names_static:
                to_remove.update(sublist)

            # 2. 对 scene_tokens 里的每个子列表，去除这些元素
            scene_tokens_without_stationary_elements = [
                [item for item in sublist if item not in to_remove]
                for sublist in self.scene_names_orig
            ]
            self.scene_lists = scene_tokens_without_stationary_elements

        self.scene_lens = [len(sn) for sn in self.scene_lists]

        print(f"=> data_version=={data_version} scene_len: {len(self.scene_lists)}==total frames: ={sum(self.scene_lens)}")

        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.quantize_size=quantize_size

        # map 
        self.maps = {}
        self.aux_data = self.AUX_DATA_CH

        self.classes= ['intersections','generic_drivable_areas','walkways','carpark_areas','crosswalks','lane_group_connectors','lane_groups_polygons','road_segments']
        self.object_classes = ['vehicle','bicycle','pedestrian','traffic_cone','barrier','czone_sign','generic_object']
        self.times = times
        # self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR
        # print(f"quantize_size: {self.quantize_size}")
        # for 200 resolution
        if self.quantize_size == (400,400,32):
            xbound=[-50,50,0.25]
            ybound=[-50,50,0.25]
            zbound=[-3,5,0.25]
            voxel_size=0.25
        elif self.quantize_size == (200,200,16):
            xbound=[-50,50,0.5]
            ybound=[-50,50,0.5]
            zbound=[-3,5,0.5]
            voxel_size=0.5
        else:
            raise ValueError(f"quantize_size should be (400,400,32) or (200,200,16), but got {self.quantize_size}")
        self.point_cloud_range = [xbound[0],ybound[0],zbound[0],xbound[1],ybound[1],zbound[1]]
        self.occ_xdim = int((xbound[1] - xbound[0]) / voxel_size)
        self.occ_ydim = int((ybound[1] - ybound[0]) / voxel_size)
        self.occ_zdim = int((zbound[1] - zbound[0]) / voxel_size)
        self.voxel_num = self.occ_xdim*self.occ_ydim*self.occ_zdim
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.use_valid_flag=True

        self.lidar2canvas = np.array([
            [canvas_h / patch_h, 0, canvas_h / 2],
            [0, canvas_w / patch_w, canvas_w / 2],
            [0, 0, 1]
        ])

        self.start_on_keyframe = True
        self.start_on_firstframe = False

        if "clip_infos" in pkl_data:
            self.clip_infos = pkl_data["clip_infos"]
        else:
            if debug:
                self.clip_infos=self.build_clips(self.nuplan_infos, pkl_data['scene_tokens'][:1])
            else:
                self.clip_infos=self.build_clips(self.nuplan_infos, pkl_data['scene_tokens'])
        print(f"=> clip_infos: {len(self.clip_infos)} =={self.clip_infos[0]}")


        # if debug:
        #     self.clip_infos=self.build_clips(self.nuplan_infos, self.scene_lists[:1])
        # else:
        #     self.clip_infos=self.build_clips(self.nuplan_infos, self.scene_lists)
        # print(f"=> clip_infos: {len(self.clip_infos)}")
        # # self.mapdb = GPKGMapsDB("nuplan-maps-v1.0", f"{map_path}")
        self.vector_maps = {}


    def fliter_clips(self,clip):
        for frame in clip:
            if self.nuplan_infos[frame]["token"] is None:
                print(f"Warning: {self.nuplan_infos[frame]['token']} is None")
                return 0
            file_path = os.path.join(self.occ_base_path,self.nuplan_infos[frame]["token"]+".npy")
            if not os.path.exists(file_path):  # 检查文件是否存在
                print(f"Warning: {file_path} does not exist")
                return 0
        return 1
            
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
        self.token_data_dict = {item['token']: idx for idx, item in enumerate(data_infos)}
        all_clips = []
        for i, scene in enumerate(scene_tokens):
            print(f"building clips for scene: {i}/{len(scene_tokens)}")
            for start in range(len(scene) - self.return_len + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + self.return_len]]
                if self.fliter_clips(clip)==0:
                    continue
                all_clips.append(clip)
                if self.start_on_firstframe:
                    break
        return all_clips
    
    def obtain_points_label(self, occ):
        occ_index, occ_cls = occ[:, 0], occ[:, 1]
        occ = np.ones(self.voxel_num, dtype=np.int8)*11
        occ[occ_index[:]] = occ_cls  # (voxel_num)
        points = []
        for i in range(len(occ_index)):
            indice = occ_index[i]
            x = indice % self.occ_xdim
            y = (indice // self.occ_xdim) % self.occ_xdim
            z = indice // (self.occ_xdim*self.occ_xdim)
            point_x = (x + 0.5) / self.occ_xdim * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
            point_y = (y + 0.5) / self.occ_ydim * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
            point_z = (z + 0.5) / self.occ_zdim * (self.point_cloud_range[5] - self.point_cloud_range[2]) + self.point_cloud_range[2]
            points.append([point_x, point_y, point_z])
        
        points = np.stack(points)
        point_label = occ_cls
        points_with_label = np.concatenate([points, point_label[:, None]], axis=-1)
        return points_with_label
    


    def convert_to_voxel_grid_int(self, voxels_):
        voxel = np.zeros((800, 800, 64), dtype=np.int32)
        voxel[voxels_[:, 0].astype(np.int32), voxels_[:, 1].astype(np.int32), voxels_[:, 2].astype(np.int32)] = voxels_[:, 3].astype(np.int32)+1
        return voxel
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.clip_infos)

    def __getitem__(self, index):
        # index = index % len(self.scene_lens)
        # scene_name = self.scene_names[index]
        # scene_len = self.scene_lens[index]
        # # idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        # idx=0
        # self.return_len=scene_len
        clip = self.clip_infos[index]
        occs = []
        tokens=[]
        bevmaps=[]
        auxs=[]
        metas = {}

        # bev 
        layer_to_merge=[0,1,3,4,5,6,7]
        bev_ch_use=[1,2,8,9,10,11,12,13,14,15,17]

        for frame in clip:
            token = self.nuplan_infos[frame]['token']
            tokens.append(token)
            occ_path = os.path.join(self.occ_base_path,self.nuplan_infos[frame]["token"]+".npy") # h, w, d
            occ_data = np.load(occ_path) # 200, 200, 16
            
            # metas.update(self.get_meta_info(frame))
            # if bev exist
            bev_file_path=os.path.join(self.bev_base_path,self.nuplan_infos[frame]["token"]+".npz")
            bev_data=np.load(bev_file_path)['gt_bev_masks']
            aux_data=np.zeros(bev_data.shape)

            bev_data[1,:,:] = np.any(bev_data[layer_to_merge, :, :], axis=0).astype(int)
            occ_with_bev = replace_occ_grid_with_bev_nuplan(input_occ_data=occ_data, bevlayout=bev_data)
            bevmap=self.nBEV1(bev_data, bev_ch_use)

            occs.append(occ_with_bev)
            bevmaps.append(bevmap)
            auxs.append(aux_data)
        
        metas.update(scene_token=tokens)
        # metas.update(scene_name=scene_name)
        input_occs = np.stack(occs).astype(np.int64)
        bevmaps= np.stack(bevmaps).astype(bool)
        auxs= np.stack(auxs).astype(np.int64)

        return input_occs, input_occs, metas, bevmaps, auxs

    def nBEV1(self, data_b,ch_use): # 18,200,200 -> 1,200,200
        data_b = data_b[ch_use]
        mask = data_b>0.01
        cumulative_mask = np.cumsum(mask, axis=0)
        max_index_map = np.argmax(cumulative_mask, axis=0)
        max_index_map = max_index_map / (len(ch_use) -1)
        all_zero_mask = np.all(mask == 0, axis=0)
        max_index_map[all_zero_mask] = -1
        data_b= np.array([max_index_map])
        return data_b
    

    def get_map_info(self, index):
        # info = data_info
        data_info = self.nuplan_infos[index]
        box_info = self.get_meta_info(index)
        lidar2ego = data_info["lidar2ego"]
        ego2global = data_info["ego2global"]
        lidar2global = ego2global @ lidar2ego

        map_pose = lidar2global[:2, 3]
        patch_box = (
            map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        
        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])  # angle between v and x-axis
        patch_angle = yaw / np.pi * 180

        mappings = {}
        location = data_info["description"]["map_version"]
        # vector map
        fixed_ptsnum_per_line = -1
        padding_value = -10000
        vector_map_api = NuPlanMapWrapper(self.mapdb, map_name=location)
        self.vector_maps[location] = vector_map_api

        self.vector_map = VectorizedLocalMap(
            self.data_path,# '/baai-cwm-1/baai_cwm_ml/algorithm/bohan.li/code/DriveArena/WorldDreamer/data/dataset/nuplan-v1.1'
            patch_size=self.patch_size, # (100.0, 100.0)
            map_classes=["divider", "ped_crossing", "boundary"],
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,# -1
            padding_value=padding_value, # -10000
            nuplan_map_api = self.vector_maps
        )
        map_explorer = NuPlanMapExplorer(map_api=vector_map_api)
        self.maps[location] = map_explorer

        for name in self.classes:
            mappings[name] = [name]
        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(layer_names)

        # cut semantics from nuscenesMap
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=self.classes,
            output_size=self.canvas_size,
        )

        masks = masks.transpose(0, 2, 1)  # TODO why need transpose here?
        masks = masks.astype(np.bool_)
        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.int64)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        # add dynamic labels
        bev_maps = dict()
        bev_maps["gt_masks_bev_static"]= labels
        final_labels = self._project_dynamic(labels, data_info)
        aux_labels = self._get_dynamic_aux(data=box_info)
        bev_maps['gt_aux_bev']=aux_labels
        bev_maps["gt_masks_bev"] = final_labels

        # add line dividers
        bev_lane = self.vectormap_pipeline(data_info)
        bev_maps["bev_lane"] = bev_lane
        bev_maps["gt_masks_bev"] = np.concatenate([bev_maps["gt_masks_bev"], bev_lane], axis=0)
        return bev_maps
    
    def get_meta_info(self, idx):
        """Get annotation info according to the given index.

        """
        info = self.nuplan_infos[idx]
        gt_bboxes_3d = info["anns"]['gt_boxes']
        gt_names_3d = info["anns"]['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d: 
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
                # print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        
        if self.with_velocity:
            gt_velocity = info["anns"]['gt_velocity_3d'][:, :2]

            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
       

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            # attr_labels=attr_labels,
            # fut_valid_flag=fut_valid_flag,
            )
        
        return anns_results


    def _get_dynamic_aux_bbox(self, aux_mask, data: Dict[str, Any]):
        '''Three aux data (7 channels in total), class-agnostic:
        1. visibility, 1-channel
        2. center-offset, 2-channel
        3. height/2, width/2, orientation, 4-channel, on bev canvas
        4. height of bbox, in lidar coordinate
        '''
        for _idx in range(len(data['gt_bboxes_3d'])):
            box = data['gt_bboxes_3d'][_idx]
            # get canvas coordinates
            # fmt:off
            _box_lidar = np.concatenate([
                box.corners[:, [0, 3, 7, 4], :2].numpy(),
                box.bottom_center[:, None, :2].numpy(),  # center
                box.corners[:, [4, 7], :2].mean(dim=1)[:, None].numpy(),  # front
                box.corners[:, [0, 4], :2].mean(dim=1)[:, None].numpy(),  # left
            ], axis=1)
            # fmt:on
            _box_canvas = np.dot(
                np.pad(_box_lidar, ((0, 0), (0, 0), (0, 1)), constant_values=1.0),
                self.lidar2canvas.T)[..., :2]  # N, 4, xy
            # in canvas coordinates
            box_canvas = _box_canvas[0, :4]
            center_canvas = _box_canvas[0, 4:5]
            front_canvas = _box_canvas[0, 5:6]
            left_canvas = _box_canvas[0, 6:7]
            # render mask
            render = Image.fromarray(np.zeros(self.canvas_size, dtype=np.uint8))
            draw = ImageDraw.Draw(render)
            draw.polygon(
                box_canvas.round().astype(np.int32).flatten().tolist(),
                fill=1)
            # construct
            tmp_mask = np.array(render) > 0
            coords = np.stack(np.meshgrid(
                np.arange(self.canvas_size[1]), np.arange(self.canvas_size[0])
            ), -1).astype(np.float32)
            _cur_ch = 0
            if "visibility" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['visibility']
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = data['visibility'][_idx]
                _cur_ch = _ch_stop
            if "center_offset" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['center_offset']
                center_offset = coords[tmp_mask] - center_canvas
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = center_offset
                _cur_ch = _ch_stop
            if "center_ohw" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['center_ohw']
                height = np.linalg.norm(front_canvas - center_canvas)
                width = np.linalg.norm(left_canvas - center_canvas)
                # yaw = box.yaw  # scaling aspect ratio, yaw does not change
                v = ((front_canvas - center_canvas) / (
                    np.linalg.norm(front_canvas - center_canvas) + 1e-6))[0]
                # yaw = - np.arctan2(v[1], v[0])  # add negative, align with mmdet coord
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = np.array([
                    height, width, v[0], v[1]])[None]
                _cur_ch = _ch_stop
            if "height" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['height']
                bbox_height = box.height.item()  # in lidar coordinate
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = np.array([
                    bbox_height])[None]
                _cur_ch = _ch_stop
        return aux_mask

    def _get_dynamic_aux(self, data: Dict[str, Any] = None) -> Any:
        '''aux data
        case 1: self.aux_data is None, return None
        case 2: data=None, set all values to zeros
        '''
        if self.aux_data is None:
            return None  # there is no aux_data

        aux_ch = sum([self.AUX_DATA_CH[aux_k] for aux_k in self.aux_data])
        if aux_ch == 0:  # there is no available aux_data
            if len(self.aux_data) != 0:
                print(f"Your aux_data: {self.aux_data} is not available")
            return None

        aux_mask = np.zeros((*self.canvas_size, aux_ch), dtype=np.float32)
        if data is not None:
            aux_mask = self._get_dynamic_aux_bbox(aux_mask, data)

        # transpose x,y and channel first format
        aux_mask = aux_mask.transpose(2, 1, 0)
        return aux_mask
    
    def _project_dynamic_bbox(self, dynamic_mask, data_info):
        '''We use PIL for projection, while CVT use cv2. The results are
        slightly different due to anti-alias of line, but should be similar.
        '''
        # handle gt_boxes_3d, gt_names_3d, gt_labels_3d
        # info = self.data_infos[index]
        gt_bboxes_3d = data_info['anns']['gt_boxes']
        gt_names_3d = data_info['anns']['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            # 已经有 3d 的速度, 直接使用就行, 不用 z 轴分量
            gt_velocity = data_info['anns']['gt_velocity_3d'][:, :2]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        # gt_bboxes_3d = data_info['anns']['gt_boxes']
        # gt_names_3d = data_info['anns']['gt_names']
        # gt_labels_3d = []
        # for cat in gt_names_3d: 
        #     if cat in self.object_classes:
        #         gt_labels_3d.append(self.object_classes.index(cat))
        #     else:   
        #         gt_labels_3d.append(-1)
        # gt_labels_3d = np.array(gt_labels_3d)
        
        for cls_id, cls_name in enumerate(self.object_classes):
            # pick boxes
            # cls_mask = data_info['gt_labels_3d'] == cls_id
            cls_mask = gt_labels_3d == cls_id
            boxes = gt_bboxes_3d[cls_mask]

            if len(boxes) < 1:
                continue
            bottom_corners_lidar = boxes.corners[:, [0, 3, 7, 4], :2]

            bottom_corners_canvas = np.dot(
                np.pad(bottom_corners_lidar.numpy(), ((0, 0), (0, 0), (0, 1)),
                       constant_values=1.0),
                self.lidar2canvas.T)[..., :2]  # N, 4, xy
            # draw
            # points=bottom_corners_canvas
            # centers = np.mean(points, axis=1, keepdims=True)
            # points[:,:,1]= 2*centers[:,:,1]-points[:,:,1]
            # bottom_corners_canvas=points

            render = Image.fromarray(dynamic_mask[cls_id])
            draw = ImageDraw.Draw(render)
            for box in bottom_corners_canvas:
                draw.polygon(
                    box.round().astype(np.int32).flatten().tolist(), fill=1)
            # save
            dynamic_mask[cls_id, :] = np.array(render)[:]
        return dynamic_mask

    def _project_dynamic(self, static_label, data_info):
        """for dynamic mask, one class per channel
        case 1: data is None, set all values to zeros
        """
        # setup
        ch = len(self.object_classes)
        dynamic_mask = np.zeros((ch, *self.canvas_size), dtype=np.uint8)

        # if int, set ch=object_classes with all zeros; otherwise, project
        if data_info is not None:
            dynamic_mask = self._project_dynamic_bbox(dynamic_mask, data_info)

        # combine with static_label
        dynamic_mask = dynamic_mask.transpose(0, 2, 1)
        combined_label = np.concatenate([static_label, dynamic_mask], axis=0)
        return combined_label


    def vectormap_pipeline(self, input_dict):
        """
        Process vector map data for input example, using transformation matrices and
        generating annotations.
        """

        lidar2ego = input_dict["lidar2ego"]
        ego2global = input_dict["ego2global"]
        lidar2global = ego2global @ lidar2ego
        # lidar2global = rotation_z_neg90 @ lidar2global
        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_rotation = Quaternion(matrix=lidar2global)

        anns_results = self.vector_map.gen_vectorized_samples_nuplan(
            self.vector_maps[input_dict['description']["map_version"]], # 'us-nv-las-vegas-strip'
            lidar2global_translation, # [664465.0781396995, 3997749.3183807055, 616.683242571562]
            lidar2global_rotation, # Quaternion(0.6999239997105716, 0.0002516337328280848, 0.004921886360219306, 0.7142003264800971)
        )
        gt_vecs_label = to_tensor(anns_results["gt_vecs_label"])
        if isinstance(anns_results["gt_vecs_pts_loc"], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results["gt_vecs_pts_loc"]
            gt_lines_instance = gt_vecs_pts_loc.instance_list
            gt_map_pts = [np.array(list(line.coords)) for line in gt_lines_instance]
        input_dict["gt_vecs_label"] = DC(gt_vecs_label, cpu_only=False)
        input_dict["gt_vecs_pts_loc"] = DC(gt_map_pts, cpu_only=True)


        bev_map = visualize_bev_hdmap(
            input_dict["gt_vecs_pts_loc"].data,
            input_dict["gt_vecs_label"].data,
            self.canvas_size,
        )
        bev_map = bev_map.transpose(2, 0, 1)

        return bev_map


    
@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar_Nuplan_pro_new_occ_bev_from_map:
    AUX_DATA_CH = {
        # "visibility": 1,
        "center_offset": 2,
        "center_ohw": 4,
        "height": 1,
    }
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            occ_dataroot=None,
            bev_dataroot=None,
            times=5,
            # quantize_size=(400,400,32),
            quantize_size=(200,200,16),
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts',
            debug=False,
            data_version='orig',# orig, ego_static, ego_dynamic
            map_path="data/maps",
        ):

        with open(imageset, 'rb') as f:
            pkl_data = pickle.load(f)
        print(f"=> loaded pkl_data from {imageset}")

        self.nuplan_infos = pkl_data['infos']
        # self.nuplan_infos = preprocess_infos_to_dict(pkl_data)
        self.occ_base_path = occ_dataroot
        self.bev_base_path = bev_dataroot

        if data_version=='orig':
            self.scene_lists=pkl_data['scene_tokens']
        elif data_version=='ego_static':
            self.scene_lists=pkl_data['scene_tokens_with_stationary_segments']
        elif data_version=='ego_dynamic':
            self.scene_names_orig = pkl_data['scene_tokens']
            self.scene_names_static = pkl_data['scene_tokens_with_stationary_segments']
            # 1. 先把所有要去除的元素收集到一个 set 里
            to_remove = set()
            for sublist in self.scene_names_static:
                to_remove.update(sublist)

            # 2. 对 scene_tokens 里的每个子列表，去除这些元素
            scene_tokens_without_stationary_elements = [
                [item for item in sublist if item not in to_remove]
                for sublist in self.scene_names_orig
            ]
            self.scene_lists = scene_tokens_without_stationary_elements

        self.scene_lens = [len(sn) for sn in self.scene_lists]

        print(f"=> data_version=={data_version} scene_len: {len(self.scene_lists)}==total frames: ={sum(self.scene_lens)}")

        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.quantize_size=quantize_size

        # map 
        self.maps = {}
        self.aux_data = self.AUX_DATA_CH

        self.classes= ['intersections','generic_drivable_areas','walkways','carpark_areas','crosswalks','lane_group_connectors','lane_groups_polygons','road_segments']
        self.object_classes = ['vehicle','bicycle','pedestrian','traffic_cone','barrier','czone_sign','generic_object']
        self.times = times
        # self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR
        # print(f"quantize_size: {self.quantize_size}")
        # for 200 resolution
        if self.quantize_size == (400,400,32):
            xbound=[-50,50,0.25]
            ybound=[-50,50,0.25]
            zbound=[-3,5,0.25]
            voxel_size=0.25
        elif self.quantize_size == (200,200,16):
            xbound=[-50,50,0.5]
            ybound=[-50,50,0.5]
            zbound=[-3,5,0.5]
            voxel_size=0.5
        else:
            raise ValueError(f"quantize_size should be (400,400,32) or (200,200,16), but got {self.quantize_size}")
        self.point_cloud_range = [xbound[0],ybound[0],zbound[0],xbound[1],ybound[1],zbound[1]]
        self.occ_xdim = int((xbound[1] - xbound[0]) / voxel_size)
        self.occ_ydim = int((ybound[1] - ybound[0]) / voxel_size)
        self.occ_zdim = int((zbound[1] - zbound[0]) / voxel_size)
        self.voxel_num = self.occ_xdim*self.occ_ydim*self.occ_zdim
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.use_valid_flag=True

        self.lidar2canvas = np.array([
            [canvas_h / patch_h, 0, canvas_h / 2],
            [0, canvas_w / patch_w, canvas_w / 2],
            [0, 0, 1]
        ])

        self.start_on_keyframe = True
        self.start_on_firstframe = False

        if debug:
            self.clip_infos=self.build_clips(self.nuplan_infos, self.scene_lists[:1])
        else:
            self.clip_infos=self.build_clips(self.nuplan_infos, self.scene_lists)
        print(f"=> clip_infos: {len(self.clip_infos)}")
        self.mapdb = GPKGMapsDB("nuplan-maps-v1.0", f"{map_path}")
        self.vector_maps = {}


    def fliter_clips(self,clip):
        for frame in clip:
            if self.nuplan_infos[frame]["token"] is None:
                print(f"Warning: {self.nuplan_infos[frame]['token']} is None")
                return 0
            file_path = os.path.join(self.occ_base_path,self.nuplan_infos[frame]["token"]+".npy")
            if not os.path.exists(file_path):  # 检查文件是否存在
                print(f"Warning: {file_path} does not exist")
                return 0
        return 1
            
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
        self.token_data_dict = {item['token']: idx for idx, item in enumerate(data_infos)}
        all_clips = []
        for i, scene in enumerate(scene_tokens):
            print(f"building clips for scene: {i}/{len(scene_tokens)}")
            for start in range(len(scene) - self.return_len + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + self.return_len]]
                if self.fliter_clips(clip)==0:
                    continue
                all_clips.append(clip)
                if self.start_on_firstframe:
                    break
        return all_clips
    
    def obtain_points_label(self, occ):
        occ_index, occ_cls = occ[:, 0], occ[:, 1]
        occ = np.ones(self.voxel_num, dtype=np.int8)*11
        occ[occ_index[:]] = occ_cls  # (voxel_num)
        points = []
        for i in range(len(occ_index)):
            indice = occ_index[i]
            x = indice % self.occ_xdim
            y = (indice // self.occ_xdim) % self.occ_xdim
            z = indice // (self.occ_xdim*self.occ_xdim)
            point_x = (x + 0.5) / self.occ_xdim * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
            point_y = (y + 0.5) / self.occ_ydim * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
            point_z = (z + 0.5) / self.occ_zdim * (self.point_cloud_range[5] - self.point_cloud_range[2]) + self.point_cloud_range[2]
            points.append([point_x, point_y, point_z])
        
        points = np.stack(points)
        point_label = occ_cls
        points_with_label = np.concatenate([points, point_label[:, None]], axis=-1)
        return points_with_label
    


    def convert_to_voxel_grid_int(self, voxels_):
        voxel = np.zeros((800, 800, 64), dtype=np.int32)
        voxel[voxels_[:, 0].astype(np.int32), voxels_[:, 1].astype(np.int32), voxels_[:, 2].astype(np.int32)] = voxels_[:, 3].astype(np.int32)+1
        return voxel
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.clip_infos)

    def __getitem__(self, index):
        # index = index % len(self.scene_lens)
        # scene_name = self.scene_names[index]
        # scene_len = self.scene_lens[index]
        # # idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        # idx=0
        # self.return_len=scene_len
        clip = self.clip_infos[index]
        occs = []
        tokens=[]
        bevmaps=[]
        auxs=[]
        metas = {}

        # bev 
        layer_to_merge=[0,1,3,4,5,6,7]
        bev_ch_use=[1,2,8,9,10,11,12,13,14,15,17]

        for frame in clip:
            token = self.nuplan_infos[frame]['token']
            tokens.append(token)
            occ_path = os.path.join(self.occ_base_path,self.nuplan_infos[frame]["token"]+".npy") # h, w, d
            occ_data = np.load(occ_path) # 200, 200, 16
            
            # metas.update(self.get_meta_info(frame))
            # if bev exist
            bev_file_path=os.path.join(self.bev_base_path,self.nuplan_infos[frame]["token"]+".npz")
            if os.path.exists(bev_file_path):
                # print("=> using saved bevmap")
                bev_data=np.load(bev_file_path)['gt_bev_masks']
                aux_data=np.zeros(bev_data.shape)
            else:
                map_info = self.get_map_info(frame) # 获得 bevmap
                bev_data = map_info['gt_masks_bev']
                aux_data = map_info["gt_aux_bev"]
                print("Get BEV map from maps")


            bev_data[1,:,:] = np.any(bev_data[layer_to_merge, :, :], axis=0).astype(int)
            occ_with_bev = replace_occ_grid_with_bev_nuplan(input_occ_data=occ_data, bevlayout=bev_data)
            bevmap=self.nBEV1(bev_data, bev_ch_use)

            occs.append(occ_with_bev)
            bevmaps.append(bevmap)
            auxs.append(aux_data)
        
        metas.update(scene_token=tokens)
        # metas.update(scene_name=scene_name)
        input_occs = np.stack(occs).astype(np.int64)
        bevmaps= np.stack(bevmaps).astype(bool)
        auxs= np.stack(auxs).astype(np.int64)

        return input_occs, input_occs, metas, bevmaps, auxs

    def nBEV1(self, data_b,ch_use): # 18,200,200 -> 1,200,200
        data_b = data_b[ch_use]
        mask = data_b>0.01
        cumulative_mask = np.cumsum(mask, axis=0)
        max_index_map = np.argmax(cumulative_mask, axis=0)
        max_index_map = max_index_map / (len(ch_use) -1)
        all_zero_mask = np.all(mask == 0, axis=0)
        max_index_map[all_zero_mask] = -1
        data_b= np.array([max_index_map])
        return data_b
    

    def get_map_info(self, index):
        # info = data_info
        data_info = self.nuplan_infos[index]
        box_info = self.get_meta_info(index)
        lidar2ego = data_info["lidar2ego"]
        ego2global = data_info["ego2global"]
        lidar2global = ego2global @ lidar2ego

        map_pose = lidar2global[:2, 3]
        patch_box = (
            map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        
        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])  # angle between v and x-axis
        patch_angle = yaw / np.pi * 180

        mappings = {}
        location = data_info["description"]["map_version"]
        # vector map
        fixed_ptsnum_per_line = -1
        padding_value = -10000
        vector_map_api = NuPlanMapWrapper(self.mapdb, map_name=location)
        self.vector_maps[location] = vector_map_api

        self.vector_map = VectorizedLocalMap(
            self.data_path,# '/baai-cwm-1/baai_cwm_ml/algorithm/bohan.li/code/DriveArena/WorldDreamer/data/dataset/nuplan-v1.1'
            patch_size=self.patch_size, # (100.0, 100.0)
            map_classes=["divider", "ped_crossing", "boundary"],
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,# -1
            padding_value=padding_value, # -10000
            nuplan_map_api = self.vector_maps
        )
        map_explorer = NuPlanMapExplorer(map_api=vector_map_api)
        self.maps[location] = map_explorer

        for name in self.classes:
            mappings[name] = [name]
        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(layer_names)

        # cut semantics from nuscenesMap
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=self.classes,
            output_size=self.canvas_size,
        )

        masks = masks.transpose(0, 2, 1)  # TODO why need transpose here?
        masks = masks.astype(np.bool_)
        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.int64)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        # add dynamic labels
        bev_maps = dict()
        bev_maps["gt_masks_bev_static"]= labels
        final_labels = self._project_dynamic(labels, data_info)
        aux_labels = self._get_dynamic_aux(data=box_info)
        bev_maps['gt_aux_bev']=aux_labels
        bev_maps["gt_masks_bev"] = final_labels

        # add line dividers
        bev_lane = self.vectormap_pipeline(data_info)
        bev_maps["bev_lane"] = bev_lane
        bev_maps["gt_masks_bev"] = np.concatenate([bev_maps["gt_masks_bev"], bev_lane], axis=0)
        return bev_maps
    
    def get_meta_info(self, idx):
        """Get annotation info according to the given index.

        """
        info = self.nuplan_infos[idx]
        gt_bboxes_3d = info["anns"]['gt_boxes']
        gt_names_3d = info["anns"]['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d: 
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
                # print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        
        if self.with_velocity:
            gt_velocity = info["anns"]['gt_velocity_3d'][:, :2]

            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
       

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            # attr_labels=attr_labels,
            # fut_valid_flag=fut_valid_flag,
            )
        
        return anns_results


    def _get_dynamic_aux_bbox(self, aux_mask, data: Dict[str, Any]):
        '''Three aux data (7 channels in total), class-agnostic:
        1. visibility, 1-channel
        2. center-offset, 2-channel
        3. height/2, width/2, orientation, 4-channel, on bev canvas
        4. height of bbox, in lidar coordinate
        '''
        for _idx in range(len(data['gt_bboxes_3d'])):
            box = data['gt_bboxes_3d'][_idx]
            # get canvas coordinates
            # fmt:off
            _box_lidar = np.concatenate([
                box.corners[:, [0, 3, 7, 4], :2].numpy(),
                box.bottom_center[:, None, :2].numpy(),  # center
                box.corners[:, [4, 7], :2].mean(dim=1)[:, None].numpy(),  # front
                box.corners[:, [0, 4], :2].mean(dim=1)[:, None].numpy(),  # left
            ], axis=1)
            # fmt:on
            _box_canvas = np.dot(
                np.pad(_box_lidar, ((0, 0), (0, 0), (0, 1)), constant_values=1.0),
                self.lidar2canvas.T)[..., :2]  # N, 4, xy
            # in canvas coordinates
            box_canvas = _box_canvas[0, :4]
            center_canvas = _box_canvas[0, 4:5]
            front_canvas = _box_canvas[0, 5:6]
            left_canvas = _box_canvas[0, 6:7]
            # render mask
            render = Image.fromarray(np.zeros(self.canvas_size, dtype=np.uint8))
            draw = ImageDraw.Draw(render)
            draw.polygon(
                box_canvas.round().astype(np.int32).flatten().tolist(),
                fill=1)
            # construct
            tmp_mask = np.array(render) > 0
            coords = np.stack(np.meshgrid(
                np.arange(self.canvas_size[1]), np.arange(self.canvas_size[0])
            ), -1).astype(np.float32)
            _cur_ch = 0
            if "visibility" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['visibility']
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = data['visibility'][_idx]
                _cur_ch = _ch_stop
            if "center_offset" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['center_offset']
                center_offset = coords[tmp_mask] - center_canvas
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = center_offset
                _cur_ch = _ch_stop
            if "center_ohw" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['center_ohw']
                height = np.linalg.norm(front_canvas - center_canvas)
                width = np.linalg.norm(left_canvas - center_canvas)
                # yaw = box.yaw  # scaling aspect ratio, yaw does not change
                v = ((front_canvas - center_canvas) / (
                    np.linalg.norm(front_canvas - center_canvas) + 1e-6))[0]
                # yaw = - np.arctan2(v[1], v[0])  # add negative, align with mmdet coord
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = np.array([
                    height, width, v[0], v[1]])[None]
                _cur_ch = _ch_stop
            if "height" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['height']
                bbox_height = box.height.item()  # in lidar coordinate
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = np.array([
                    bbox_height])[None]
                _cur_ch = _ch_stop
        return aux_mask

    def _get_dynamic_aux(self, data: Dict[str, Any] = None) -> Any:
        '''aux data
        case 1: self.aux_data is None, return None
        case 2: data=None, set all values to zeros
        '''
        if self.aux_data is None:
            return None  # there is no aux_data

        aux_ch = sum([self.AUX_DATA_CH[aux_k] for aux_k in self.aux_data])
        if aux_ch == 0:  # there is no available aux_data
            if len(self.aux_data) != 0:
                print(f"Your aux_data: {self.aux_data} is not available")
            return None

        aux_mask = np.zeros((*self.canvas_size, aux_ch), dtype=np.float32)
        if data is not None:
            aux_mask = self._get_dynamic_aux_bbox(aux_mask, data)

        # transpose x,y and channel first format
        aux_mask = aux_mask.transpose(2, 1, 0)
        return aux_mask
    
    def _project_dynamic_bbox(self, dynamic_mask, data_info):
        '''We use PIL for projection, while CVT use cv2. The results are
        slightly different due to anti-alias of line, but should be similar.
        '''
        # handle gt_boxes_3d, gt_names_3d, gt_labels_3d
        # info = self.data_infos[index]
        gt_bboxes_3d = data_info['anns']['gt_boxes']
        gt_names_3d = data_info['anns']['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            # 已经有 3d 的速度, 直接使用就行, 不用 z 轴分量
            gt_velocity = data_info['anns']['gt_velocity_3d'][:, :2]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        # gt_bboxes_3d = data_info['anns']['gt_boxes']
        # gt_names_3d = data_info['anns']['gt_names']
        # gt_labels_3d = []
        # for cat in gt_names_3d: 
        #     if cat in self.object_classes:
        #         gt_labels_3d.append(self.object_classes.index(cat))
        #     else:   
        #         gt_labels_3d.append(-1)
        # gt_labels_3d = np.array(gt_labels_3d)
        
        for cls_id, cls_name in enumerate(self.object_classes):
            # pick boxes
            # cls_mask = data_info['gt_labels_3d'] == cls_id
            cls_mask = gt_labels_3d == cls_id
            boxes = gt_bboxes_3d[cls_mask]

            if len(boxes) < 1:
                continue
            bottom_corners_lidar = boxes.corners[:, [0, 3, 7, 4], :2]

            bottom_corners_canvas = np.dot(
                np.pad(bottom_corners_lidar.numpy(), ((0, 0), (0, 0), (0, 1)),
                       constant_values=1.0),
                self.lidar2canvas.T)[..., :2]  # N, 4, xy
            # draw
            # points=bottom_corners_canvas
            # centers = np.mean(points, axis=1, keepdims=True)
            # points[:,:,1]= 2*centers[:,:,1]-points[:,:,1]
            # bottom_corners_canvas=points

            render = Image.fromarray(dynamic_mask[cls_id])
            draw = ImageDraw.Draw(render)
            for box in bottom_corners_canvas:
                draw.polygon(
                    box.round().astype(np.int32).flatten().tolist(), fill=1)
            # save
            dynamic_mask[cls_id, :] = np.array(render)[:]
        return dynamic_mask

    def _project_dynamic(self, static_label, data_info):
        """for dynamic mask, one class per channel
        case 1: data is None, set all values to zeros
        """
        # setup
        ch = len(self.object_classes)
        dynamic_mask = np.zeros((ch, *self.canvas_size), dtype=np.uint8)

        # if int, set ch=object_classes with all zeros; otherwise, project
        if data_info is not None:
            dynamic_mask = self._project_dynamic_bbox(dynamic_mask, data_info)

        # combine with static_label
        dynamic_mask = dynamic_mask.transpose(0, 2, 1)
        combined_label = np.concatenate([static_label, dynamic_mask], axis=0)
        return combined_label


    def vectormap_pipeline(self, input_dict):
        """
        Process vector map data for input example, using transformation matrices and
        generating annotations.
        """

        lidar2ego = input_dict["lidar2ego"]
        ego2global = input_dict["ego2global"]
        lidar2global = ego2global @ lidar2ego
        # lidar2global = rotation_z_neg90 @ lidar2global
        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_rotation = Quaternion(matrix=lidar2global)

        anns_results = self.vector_map.gen_vectorized_samples_nuplan(
            self.vector_maps[input_dict['description']["map_version"]], # 'us-nv-las-vegas-strip'
            lidar2global_translation, # [664465.0781396995, 3997749.3183807055, 616.683242571562]
            lidar2global_rotation, # Quaternion(0.6999239997105716, 0.0002516337328280848, 0.004921886360219306, 0.7142003264800971)
        )
        gt_vecs_label = to_tensor(anns_results["gt_vecs_label"])
        if isinstance(anns_results["gt_vecs_pts_loc"], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results["gt_vecs_pts_loc"]
            gt_lines_instance = gt_vecs_pts_loc.instance_list
            gt_map_pts = [np.array(list(line.coords)) for line in gt_lines_instance]
        input_dict["gt_vecs_label"] = DC(gt_vecs_label, cpu_only=False)
        input_dict["gt_vecs_pts_loc"] = DC(gt_map_pts, cpu_only=True)


        bev_map = visualize_bev_hdmap(
            input_dict["gt_vecs_pts_loc"].data,
            input_dict["gt_vecs_label"].data,
            self.canvas_size,
        )
        bev_map = bev_map.transpose(2, 0, 1)

        return bev_map

def replace_occ_grid_with_bev_nuplan(input_occ_data, bevlayout, driva_area_idx=1, bev_replace_idx=[1,15],
                                    occ_replace_new_idx=[12, 14]):
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
        roal_divider_mask, np.ones((1, 1), np.uint8))
    lane_divider_mask = cv2.dilate(
        lane_divider_mask, np.ones((1, 1), np.uint8))

    bevlayout[15, :, :] = roal_divider_mask.astype(bool)
    bevlayout[17, :, :] = lane_divider_mask.astype(bool)

    n = len(bev_replace_idx)
    x_max, y_max = input_occ_data.shape[0], input_occ_data.shape[1]
    output_occ = input_occ_data.copy()  # numpy copy() ; tensor clone()
    bev_replace_mask = []
    for i in range(n):
        bev_replace_mask.append(bevlayout[bev_replace_idx[i]] == 1)

    for x in range(x_max):
        for y in range(y_max):
            for i in range(n):
                if bev_replace_mask[i][x, y]:
                    occupancy_data = input_occ_data[x, y, :]

                    if driva_area_idx in occupancy_data:
                        max_11_index = np.where(
                            occupancy_data == driva_area_idx)
                        output_occ[x, y, max_11_index] = occ_replace_new_idx[i]
    return output_occ


if __name__ == '__main__':
    # test_dataset = Nuplan_HR_occ_mini(
    #     data_path='/mnt/datasets/nuplan-all/2-0-0/dataset/nuplan-v1.1',
    #     imageset='/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_train.pkl',
    #     occ_dataroot='/mnt/datasets/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_400_400_32',
    #     bev_dataroot='None',
    #     offset=0,
    #     return_len=5,
    #     quantize_size=(400,400,32),
    #     debug=True
    # )
    test_dataset = Nuplan_HR_occ_mini(
        data_path='/data/longhun/3D/nuplan',
        imageset='/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_train.pkl',
        occ_dataroot='/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ/nuplan_quantized_400_400_32',
        bev_dataroot='/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200',
        offset=0,
        return_len=5,
        quantize_size=(400,400,32),
        debug=True
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    from tqdm import tqdm
    for iter_i, (input_occs, output_occs, metas, bevmaps) in enumerate(tqdm(test_dataloader)): 
        if iter_i>1:
            break    
