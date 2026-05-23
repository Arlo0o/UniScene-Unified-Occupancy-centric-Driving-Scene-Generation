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
from uniscenev2.registry import DATASETS, build_module
from mmcv.parallel import DataContainer
from typing import Any, Dict
from uniscenev2.mmdet_plugin.datasets import  NuplanDataset
from uniscenev2.mmdet_plugin.core.bbox import LiDARInstance3DBoxes
from uniscenev2.datasets.utils import trans_boxes_to_views, IMG_FPS
from IPython import embed
from uniscenev2.datasets.nuplan_base_datasets import NuplanTDataset
import re
import gc
from tqdm import tqdm  


@DATASETS.register_module()
class NuplanVariableDataset(NuplanTDataset):
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
        NuplanDataset.__init__(
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

    # def build_clips(self, data_infos, scene_tokens, video_length, repeat_times=1):
    #     """Since the order in self.data_infos may change on loading, we
    #     calculate the index for clips after loading.

    #     Args:
    #         data_infos (list of dict): loaded data_infos
    #         scene_tokens (2-dim list of str): 2-dim list for tokens to each
    #         scene 

    #     Returns:
    #         2-dim list of int: int is the index in self.data_infos
    #     """
    #     self.token_data_dict = {
    #         item['token']: idx for idx, item in enumerate(data_infos)}
    #     if self.balance_keywords is not None:
    #         data_infos, scene_tokens = self.balance_annotations(
    #             data_infos, scene_tokens)
    #     all_clips = []
    #     skip1, skip2 = 0, 0
    #     for scene in scene_tokens:
    #         if video_length == "full":
    #             clip = [self.token_data_dict[token] for token in scene]
    #             if self.micro_frame_size is not None:
    #                 # trim to micro_frame_size
    #                 res = len(clip) % self.micro_frame_size - 1
    #                 if res > 0:
    #                     clip = clip[:-res]
    #             all_clips.append(clip)
    #         else:
    #             for start in range(len(scene) - video_length + 1):
    #                 if self.start_on_keyframe and ";" in scene[start]:
    #                     skip1 += 1
    #                     continue  # this is not a keyframe
    #                 if self.start_on_keyframe and len(scene[start]) >= 33:
    #                     skip2 += 1
    #                     continue  # this is not a keyframe
    #                 clip = [self.token_data_dict[token]
    #                         for token in scene[start: start + video_length]]
    #                 if self.micro_frame_size is not None:
    #                     assert len(clip) % self.micro_frame_size <= 1
    #                 all_clips.append(clip)
    #     if repeat_times > 1:
    #         assert isinstance(repeat_times, int)
    #         all_clips = all_clips * repeat_times
    #     logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
    #                  f"continuous scenes. Cut into {video_length}-clip, "
    #                  f"which has {len(all_clips)} in total. We skip {skip1} + "
    #                  f"{skip2} = {skip1 + skip2} possible starting frames.")
    #     return all_clips



    def build_clips(self, data_infos, scene_tokens, video_length, repeat_times=1):
        token_data_dict = {item['token']: idx for idx, item in enumerate(data_infos)}
        if self.balance_keywords is not None:
            data_infos, scene_tokens = self.balance_annotations( data_infos, scene_tokens)
        all_clips = []
        for scene in scene_tokens:
            if video_length == "full":
                clip = [token_data_dict[token] for token in scene]
                if self.micro_frame_size:
                    clip = clip[:-(len(clip) % self.micro_frame_size - 1) or None]
                all_clips.append(clip)
            else:
                for start in range(len(scene) - video_length + 1):
                    if self.start_on_keyframe and (";" in scene[start] or len(scene[start]) >= 33):
                        continue
                    clip = [token_data_dict[token] for token in scene[start:start+video_length]]
                    all_clips.append(clip)
        if repeat_times > 1:
            all_clips *= repeat_times
        del token_data_dict  # 释放临时字典
        gc.collect()
        return all_clips


    def load_merged_chunks(self, file_list):
        merged_data = {'metadata': None, 'scene_tokens': [], 'infos': []}
        for chunk_path in tqdm(sorted(file_list), desc="Loading chunks"):
            chunk_data = mmcv.load(chunk_path)
            if merged_data['metadata'] is None:
                merged_data['metadata'] = chunk_data['metadata']
            merged_data['scene_tokens'].extend(chunk_data['scene_tokens'])
            merged_data['infos'].extend(chunk_data['infos'])
            del chunk_data  # 立即释放分块内存
            gc.collect()     # 强制垃圾回收
        print(f"Total samples: {len(merged_data['infos']):,}")
        return merged_data



    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        # data = mmcv.load(ann_file)
        data = self.load_merged_chunks(ann_file)
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




    def super_get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]
        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            # lidar_path=info["lidar_path"],
            # sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            # location=info["location"],
        )
        add_key = [
            "description",
            "timeofday",
            "visibility",
            "flip_gt",
        ]
        for key in add_key:
            if key in info:
                data[key] = info[key]
        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global
        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego
        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []
            for _, camera_info in info["cams"].items():
                data["image_paths"].append(camera_info["data_path"])
                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
                lidar2camera_t = (
                    camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                data["lidar2camera"].append(lidar2camera_rt.T)
                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = camera_info["cam_intrinsic"]
                data["camera_intrinsics"].append(camera_intrinsics)
                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                data["lidar2image"].append(lidar2image)
                # # camera to ego transform
                # camera2ego = np.eye(4).astype(np.float32)
                # camera2ego[:3, :3] = Quaternion(
                #     camera_info["sensor2ego_rotation"]
                # ).rotation_matrix
                # camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                # data["camera2ego"].append(camera2ego)
                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                data["camera2lidar"].append(camera2lidar)
        annos, mask = self.get_ann_info(index)
        if "visibility" in data:
            data["visibility"] = data["visibility"][mask]
        data["ann_info"] = annos
        return data
    def load_clip(self, clip):
        frames = []
        first_info = self.data_infos[clip[0]]
        for frame in clip:
            frame_info = self.super_get_data_info(frame)
            info = self.data_infos[frame]
            next2top = obtain_next2top(first_info, info, v2=self.next2topv2)
            frame_info['next2top'] = next2top
            frames.append(frame_info)
        return frames
    def get_data_info(self, idx, num_frames, interval):
        """We should sample from clip_infos
        """
        clip = self.clip_infos[num_frames][idx][0::interval]
        frames = self.load_clip(clip)
        return frames




    # def prepare_train_data(self, index):
    #     idx, real_t, fps = self.parse_index(index)
    #     if isinstance(real_t, str) or real_t > 1:
    #         assert fps <= self.base_fps
    #         interval = self.base_fps // fps
    #     else:
    #         interval = 1
    #     frames = self.get_data_info(idx, real_t, interval=interval)
    #     real_t = len(frames)  # NOTE: we have load interval, real_t may change
    #     ret_dicts = self.load_frames(frames)
    #     if ret_dicts is None:
    #         return None
    #     ret_dicts['fps'] = IMG_FPS if real_t == 1 else fps
    #     ret_dicts['num_frames'] = real_t
    #     return ret_dicts

    def prepare_train_data(self, index):
        idx, real_t, fps = self.parse_index(index)
        interval = self.base_fps // fps if isinstance(real_t, str) or real_t > 1 else 1
        frames = self.get_data_info(idx, real_t, interval=interval)
        real_t = len(frames)  # NOTE: we have load interval, real_t may change
        ret_dicts = self.load_frames(frames)
        
        # 显式释放临时变量
        del frames
        gc.collect()
        
        if ret_dicts is None:
            return None
        ret_dicts['fps'] = IMG_FPS if real_t == 1 else fps
        ret_dicts['num_frames'] = len(ret_dicts)
        return ret_dicts


@DATASETS.register_module()
class NuplanMultiResDataset(torch.utils.data.Dataset):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.datasets = OrderedDict()
        for key,d_cfg in cfg:
            dataset: NuplanVariableDataset = build_module(d_cfg, DATASETS)
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
    





def obtain_next2top(first, current, epsilon=1e-6, v2=True):
    l2e_r = first["lidar2ego_rotation"]
    l2e_t = first["lidar2ego_translation"]
    e2g_r = first["ego2global_rotation"]
    e2g_t = first["ego2global_translation"]
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix
    l2e_r_s = current["lidar2ego_rotation"]
    l2e_t_s = current["lidar2ego_translation"]
    e2g_r_s = current["ego2global_rotation"]
    e2g_t_s = current["ego2global_translation"]
    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    next2lidar_rotation = R.T  # points @ R.T + T
    next2lidar_translation = T
    if v2:
        # inverse, point trans from lidar to next
        _R = np.concatenate([next2lidar_rotation.T, np.array(
            [[0.,] * 3], dtype=T.dtype)], axis=0)
        _T = -next2lidar_rotation.T @ next2lidar_translation
        _T = np.concatenate(
            [_T[..., np.newaxis], np.array([[1.]], dtype=T.dtype)], axis=0)
        # shape like:
        # | R T |
        # | 0 1 |
        # A @ point lidar -> point next
        next2lidar = np.concatenate([_R, _T], axis=1)
    else:
        _R = np.concatenate(
            [next2lidar_rotation, np.array([[0.,]] * 3, dtype=T.dtype)], axis=1)
        _T = np.concatenate(
            [next2lidar_translation, np.array([1.], dtype=T.dtype)], axis=0)
        # shape like:
        # | R 0 |
        # | T 1 |.T
        next2lidar = np.concatenate(
            [_R, _T[np.newaxis, ...]], axis=0,
        ).T  # A @ [points, 1].T
    if epsilon is not None:
        next2lidar[np.abs(next2lidar) < epsilon] = 0.
    return next2lidar
