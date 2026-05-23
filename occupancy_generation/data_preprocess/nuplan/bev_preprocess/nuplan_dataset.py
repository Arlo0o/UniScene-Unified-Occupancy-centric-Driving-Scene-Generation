import pickle
import os, numpy as np, pickle
from pyquaternion import Quaternion
from copy import deepcopy
from typing import Any, Dict, Tuple
import torch

from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.database.maps_db.map_api import NuPlanMapWrapper
from nuplan.database.maps_db.map_explorer import NuPlanMapExplorer
from nuplan.database.maps_db.gpkg_mapsdb import MAP_LOCATIONS

from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes, Box3DMode
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuplan.database.maps_db.map_api import NuPlanMapWrapper
from nuplan.database.maps_db.map_explorer import NuPlanMapExplorer

from collections.abc import Sequence
from map_utils import VectorizedLocalMap, visualize_bev_hdmap, DataContainer as DC, LiDARInstanceLines

from PIL import Image,ImageDraw

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



class nuplan_bev_dataset:
    AUX_DATA_CH = {
        # "visibility": 1,
        "center_offset": 2,
        "center_ohw": 4,
        "height": 1,
    }
    def __init__(
            self, 
            # loaded_data,
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            occ_dataroot=None,
            times=5,
            quantize_size=(200,200,16),
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts',
            aux_data: Tuple[str, ...] = None,  # aux_data for dynamic objects,
        ):

        with open(imageset, 'rb') as f:
            pkl_data = pickle.load(f)
        print(f"=> loaded pkl_data from {imageset}")

        self.data_infos = pkl_data['infos']
        self.occ_base_path = occ_dataroot
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.quantize_size=quantize_size
        self.aux_data = self.AUX_DATA_CH
        # self.nusc = nusc

        self.maps = {}
        # LOCATIONS = ['us-ma-boston', 'us-nv-las-vegas-strip', 'sg-one-north', 'us-pa-pittsburgh-hazelwood']

        self.classes= ['intersections','generic_drivable_areas','walkways','carpark_areas','crosswalks','lane_group_connectors','lane_groups_polygons','road_segments']
        self.object_classes = ['vehicle','bicycle','pedestrian','traffic_cone','barrier','czone_sign','generic_object']
        self.times = times

        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR

        # for 200 resolution
        if self.quantize_size == (400,400,32):
            xbound=[-50,50,0.25]
            ybound=[-50,50,0.25]
            zbound=[-5,3,0.25]
            voxel_size=0.25
        elif self.quantize_size == (200,200,16):
            xbound=[-50,50,0.5]
            ybound=[-50,50,0.5]
            zbound=[-5,3,0.5]
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

        self.mapdb = GPKGMapsDB("nuplan-maps-v1.0", "/data/zhuhu/3DVision_datasets/Occ/nuplan/dataset/maps")
        self.vector_maps = {}
        # self.map_components = dict()

    
    def convert_to_voxel_grid_int(self, voxels_):
        voxel = np.zeros((800, 800, 64), dtype=np.int32)
        voxel[voxels_[:, 0].astype(np.int32), voxels_[:, 1].astype(np.int32), voxels_[:, 2].astype(np.int32)] = voxels_[:, 3].astype(np.int32)+1
        return voxel
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_infos)

    def __getitem__(self, index):

        token = self.data_infos[index]['token']
        bevmap=self.get_map_info(index) # 获得 bevmap
        

        return token, bevmap


    def get_meta_info(self, idx):
        """Get annotation info according to the given index.

        """
        info = self.data_infos[idx]
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

    def get_ann_info(self, index):
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
        info = self.data_infos[index]
        gt_bboxes_3d = info["anns"]["gt_boxes"]
        gt_names_3d = info["anns"]["gt_names"]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            # 已经有 3d 的速度, 直接使用就行, 不用 z 轴分量
            gt_velocity = info["anns"]["gt_velocity_3d"][:, :2]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
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


    def get_map_info(self, index):
        # info = data_info
        data_info = self.data_infos[index]
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
