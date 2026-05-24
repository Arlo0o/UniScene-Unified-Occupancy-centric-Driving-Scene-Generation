from typing import Tuple, List
from functools import partial
import logging
from collections import OrderedDict, defaultdict
import cv2
import mmcv
import torch
import random
import numpy as np
from pyquaternion import Quaternion
from transformers import CLIPTokenizer
from mmengine.config import ConfigDict
from IPython import embed
from copy import deepcopy
from uniscenev2_video.registry import DATASETS, build_module
from mmcv.parallel import DataContainer

from uniscenev2_video.mmdet_plugin.datasets import NuScenesDataset
from uniscenev2_video.mmdet_plugin.core.bbox import LiDARInstance3DBoxes
from uniscenev2_video.datasets.utils import trans_boxes_to_views, IMG_FPS
from IPython import embed
from uniscenev2_video.datasets.nuscenes_base_datasets import NuScenesTDataset

@DATASETS.register_module()
class NuScenesVariableDataset(NuScenesTDataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length: list[int] = None,
        start_on_keyframe=True,
        next2topv2=True,
        trans_box2top=False,
        base_fps=12,
        fps: list[list[int]] = None,
        repeat_times: list[int] = None,
        img_collate_param={},
        micro_frame_size=None,
        balance_keywords=None,
        drop_ori_imgs=False,
        **kwargs,
    ) -> None:
        self.video_lengths = video_length
        self.start_on_keyframe = start_on_keyframe
        self.fps = fps
        self.micro_frame_size = micro_frame_size
        self.repeat_times = repeat_times
        self.balance_keywords = balance_keywords
        NuScenesDataset.__init__(
            self, ann_file, pipeline, dataset_root, object_classes, map_classes,
            load_interval, with_velocity, modality, box_type_3d,
            filter_empty_gt, test_mode, eval_version, use_valid_flag,
            force_all_boxes)

        if "12Hz" in ann_file and start_on_keyframe:
            logging.warning("12Hz should use all starting frame to train, please "
                         "double-check!")
            
        self.next2topv2 = next2topv2
        self.trans_box2top = trans_box2top
        self.allow_class = None
        self.del_box_ratio = 0.0
        self.drop_nearest_car = 0
        self.img_collate_param = img_collate_param
        if isinstance(self.img_collate_param, ConfigDict):
            self.img_collate_param = img_collate_param.to_dict()
        self.base_fps = base_fps
        self.drop_ori_imgs = drop_ori_imgs

        # index = '0-17-12'
        # sampled_data = self.prepare_train_data(
        #     index
        # )

    @property
    def num_frames(self):
        raise NotImplementedError()

    @property
    def possible_keys(self):
        keys = []
        for f, t in zip(self.fps, self.clip_infos.keys()):
            for fps in f:
                keys.append((fps, t))
        return keys

    def key_len(self, key):
        if isinstance(key, str):
            fps, t = key.split("-")
            fps = int(fps)
            t = t if t == "full" else int(t)
        elif isinstance(key, tuple):
            fps, t = key
        else:
            raise TypeError(key)
        return len(self.clip_infos[t])

    def __len__(self):
        return sum(self.key_len(key) for key in self.possible_keys)

    def build_clips(self, data_infos, scene_tokens, video_length, repeat_times=1):
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
        if self.balance_keywords is not None:
            data_infos, scene_tokens = self.balance_annotations(
                data_infos, scene_tokens)
        all_clips = []
        skip1, skip2 = 0, 0
        for scene in scene_tokens:
            if video_length == "full":
                clip = [self.token_data_dict[token] for token in scene]
                if self.micro_frame_size is not None:
                    # trim to micro_frame_size
                    res = len(clip) % self.micro_frame_size - 1
                    if res > 0:
                        clip = clip[:-res]
                all_clips.append(clip)
            else:
                for start in range(len(scene) - video_length + 1):
                    if self.start_on_keyframe and ";" in scene[start]:
                        skip1 += 1
                        continue  # this is not a keyframe
                    if self.start_on_keyframe and len(scene[start]) >= 33:
                        skip2 += 1
                        continue  # this is not a keyframe
                    clip = [self.token_data_dict[token]
                            for token in scene[start: start + video_length]]
                    if self.micro_frame_size is not None:
                        assert len(clip) % self.micro_frame_size <= 1
                    all_clips.append(clip)
        if repeat_times > 1:
            assert isinstance(repeat_times, int)
            all_clips = all_clips * repeat_times
        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Cut into {video_length}-clip, "
                     f"which has {len(all_clips)} in total. We skip {skip1} + "
                     f"{skip2} = {skip1 + skip2} possible starting frames.")
        return all_clips


    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        self.clip_infos = OrderedDict()
        for idx, video_length in enumerate(self.video_lengths):
            if self.repeat_times is not None:
                repeat_times = self.repeat_times[idx]
            else:
                repeat_times = 1
            self.clip_infos[video_length] = self.build_clips(
                data_infos, data['scene_tokens'], video_length, repeat_times)
        return data_infos

    def parse_index(self, index):
        idx, real_t, fps = index.split("-")
        idx, fps = map(int, [idx, fps])
        real_t = real_t if real_t == "full" else int(real_t)
        return idx, real_t, fps

    def _rand_another(self, index):
        idx, real_t, fps = self.parse_index(index)
        pool = list(range(len(self.clip_infos[real_t])))
        idx = np.random.choice(pool)
        return f"{idx}-{real_t}-{fps}"

    def get_data_info(self, idx, num_frames, interval):
        """We should sample from clip_infos
        """
        clip = self.clip_infos[num_frames][idx][0::interval]
        frames = self.load_clip(clip)
        return frames

    def prepare_train_data(self, index):
        idx, real_t, fps = self.parse_index(index)
        if isinstance(real_t, str) or real_t > 1:
            assert fps <= self.base_fps
            interval = self.base_fps // fps
        else:
            interval = 1
        frames = self.get_data_info(idx, real_t, interval=interval)
        real_t = len(frames)  # NOTE: we have load interval, real_t may change
        ret_dicts = self.load_frames(frames)
        if ret_dicts is None:
            return None
        ret_dicts['fps'] = IMG_FPS if real_t == 1 else fps
        ret_dicts['num_frames'] = real_t
        return ret_dicts


@DATASETS.register_module()
class NuScenesMultiResDataset(torch.utils.data.Dataset):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.datasets = OrderedDict()
        for key,d_cfg in cfg:
            dataset: NuScenesVariableDataset = build_module(d_cfg, DATASETS)
            self.datasets[key] = dataset

    def as_buckets(self):
        buckets = OrderedDict()  # str: list of indexes
        for res, v in self.datasets.items():
            for key in v.possible_keys:
                buckets["-".join(map(str, [*res, *key]))] = list(
                    range(v.key_len("-".join(map(str, key)))))
        return buckets

    def rand_another_key(self):
        buckets = self.as_buckets()
        key = np.random.choice(list(buckets.keys()))
        idx = np.random.choice(buckets[key])
        return f"{idx}-{key}"

    def parse_index(self, index: str):
        idx, real_h, real_w, fps = map(int, index.split("-")[:-1])
        real_t = index.split("-")[-1]
        real_t = real_t if real_t == "full" else int(real_t)
        return idx, real_h, real_w, fps, real_t

    def __len__(self):
        return sum(len(v) for v in self.datasets.values())

    def __getitem__(self, index):
        idx, real_h, real_w, fps, real_t = self.parse_index(index)
        sub_index = f"{idx}-{real_t}-{fps}"
        return self.datasets[(real_h, real_w)][sub_index]


if __name__ == "__main__":

    object_classes = [
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
    ]

    map_classes = [
        "drivable_area",
        "ped_crossing",
        "walkway",
        "stop_line",
        "carpark_area",
        "road_divider",
        "lane_divider",
        "road_block",
    ]

    template = "A driving scene image at {location}. {description}."

    input_modality = dict(
        use_lidar = False,
        use_camera = True,        
        use_radar = False,
        use_map = False,
        use_external = False,
    )
    micro_frame_size = 8

    bbox_mode = 'all-xyz'
    img_collate_param_train = dict(
        # template added by code.
        frame_emb = "next2top",
        bbox_mode = bbox_mode,
        bbox_view_shared = False,
        keyframe_rate = 6,  # work with `bbox_drop_ratio`
        bbox_drop_ratio = 0.4,
        bbox_add_ratio = 0.1,
        bbox_add_num = 3,
        bbox_processor_type = 2,
        template=template
    )
    balance_keywords = ["night", "rain", "none"]


    resize = [0.25, 0.25]
    rotate = "null"

    scale_3d = [1.0, 1.0]  # adjust the scale
    rotate_3d = [0.0, 0.0]  # rotation the lidar
    translate_3d =  0  # shift
    flip_ratio_3d = 0.0
    flip_direction_3d = "null"
    collect_meta_keys = [
        "camera_intrinsics",
        "lidar2ego",
        "lidar2camera",
        "camera2lidar",
        "lidar2image",
        "img_aug_matrix",
        "next2top"
    ]  # send to DataContainer

    collect_meta_lis_keys = [
        "timeofday",
        "location",
        "description",
        "filename",
        "token",
    ] # hold by one DataContainer

    view_order = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
    ]

    pipeline = [
        dict(
            type="LoadMultiViewImageFromFiles",
            to_float32=True
        ),
        dict(
            type="LoadAnnotations3D",
            with_bbox_3d = True,
            with_label_3d = True,
            with_attr_label = False
        ),
        dict(
            type="ImageAug3D",
            final_dim=[224, 400],
            resize_lim=resize,
            bot_pct_lim = [0.0, 0.0],
            rot_lim=rotate,
            rand_flip=False,
            is_train=False
        ),
        dict(
            type="GlobalRotScaleTrans",
            resize_lim=scale_3d,
            rot_lim=rotate_3d,
            trans_lim=translate_3d,
            is_train=True
        ),
        dict(
            type="ObjectNameFilter",
            classes=object_classes
        ),
        dict(
            type="ReorderMultiViewImages",
            order=view_order,
            safe=False
        ),
        dict(
            type="ImageNormalize",
            mean = [0.5, 0.5, 0.5],
            std = [0.5, 0.5, 0.5]
        ),
        dict(
            type="DefaultFormatBundle3D",
            classes=object_classes
        ),
        dict(
            type="Collect3D",
            keys=[
                "img",
                "lidar",
                "gt_bboxes_3d",
                "gt_labels_3d",
            ],
            meta_keys=collect_meta_keys,
            meta_lis_keys=collect_meta_lis_keys,
        ),
    ]

    pipeline2 = deepcopy(pipeline)

    pipeline2[2] = dict(
        type="ImageAug3D",
        final_dim=[424, 800],
        resize_lim=[0.5, 0.5],
        bot_pct_lim = [0.0, 0.0],
        rot_lim=rotate,
        rand_flip=False,
        is_train=False
    )

    MRD_cfg = {
        (224, 400):dict(
            type = "NuScenesVariableDataset",
            ann_file="./data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val_with_bid.pkl",
            pipeline=pipeline,
            dataset_root="./data/nuscenes/",
            object_classes=object_classes,
            map_classes=map_classes,
            load_interval=1,
            with_velocity=True,
            modality=input_modality,
            box_type_3d="LiDAR",
            filter_empty_gt=True,
            test_mode=False,
            eval_version="detection_cvpr_2019",
            use_valid_flag=False,
            force_all_boxes=True,
            video_length = [1, 17, "full"],
            start_on_keyframe=True,
            next2topv2=True,
            trans_box2top=False,
            base_fps=12,
            fps = [[120,], [12,], [12,],],
            repeat_times = [1, 1, 40],
            img_collate_param=img_collate_param_train,
            micro_frame_size=micro_frame_size,
            balance_keywords=balance_keywords,
            drop_ori_imgs=False,
        ),
        (424, 800):dict(
            type = "NuScenesVariableDataset",
            ann_file="./data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val_with_bid.pkl",
            pipeline=pipeline2,
            dataset_root="./data/nuscenes/",
            object_classes=object_classes,
            map_classes=map_classes,
            load_interval=1,
            with_velocity=True,
            modality=input_modality,
            box_type_3d="LiDAR",
            filter_empty_gt=True,
            test_mode=False,
            eval_version="detection_cvpr_2019",
            use_valid_flag=False,
            force_all_boxes=True,
            video_length = [1, 17, 33, 65, 129,],
            start_on_keyframe=True,
            next2topv2=True,
            trans_box2top=False,
            base_fps=12,
            fps = [[120,], [12,], [12,], [12], [12],],
            repeat_times = [1, 1, 1, 1, 1],
            img_collate_param=img_collate_param_train,
            micro_frame_size=micro_frame_size,
            balance_keywords=balance_keywords,
            drop_ori_imgs=False,
        )
    }


    NuScenesMultiResDataset(MRD_cfg)