from functools import partial

import numpy as np
from PIL import Image

from pcdet.utils import common_utils
from . import augmentor_utils#, database_sampler


class DataAugmentorOcc2LiDAR(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def disable_augmentation(self, augmentor_configs):
        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)
             
    # def gt_sampling(self, config=None):
    #     db_sampler = database_sampler.DataBaseSampler(
    #         root_path=self.root_path,
    #         sampler_cfg=config,
    #         class_names=self.class_names,
    #         logger=self.logger
    #     )
    #     return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        occ, points = data_dict['occ'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']

            if ('flip_%s'%cur_axis) in data_dict:
                enable = data_dict['flip_%s'%cur_axis]
            else:
                enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
            if enable:
                if cur_axis == 'x':
                    points[:, 1] = -points[:, 1]
                    occ[:, 1] = -occ[:, 1]
                    if 'sensor_loc' in data_dict:
                        data_dict['sensor_loc'][:, 1] = -data_dict['sensor_loc'][:, 1]
                elif cur_axis == 'y':
                    points[:, 0] = -points[:, 0]
                    occ[:, 0] = -occ[:, 0]
                    if 'sensor_loc' in data_dict:
                        data_dict['sensor_loc'][:, 0] = -data_dict['sensor_loc'][:, 0]
            data_dict['flip_%s'%cur_axis] = enable

        data_dict['points'] = points
        data_dict['occ'] = occ
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        occ, points = data_dict['occ'], data_dict['points']
        if 'noise_rot' in data_dict:
            noise_rotation = data_dict['noise_rot']
        else:
            noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
        points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
        occ = common_utils.rotate_points_along_z(occ[np.newaxis, :, :], np.array([noise_rotation]))[0]
        if 'sensor_loc' in data_dict:
            data_dict['sensor_loc'] = common_utils.rotate_points_along_z(data_dict['sensor_loc'][np.newaxis, :, :], np.array([noise_rotation]))[0]

        data_dict['occ'] = occ
        data_dict['points'] = points
        data_dict['noise_rot'] = noise_rotation
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)

        occ, points = data_dict['occ'], data_dict['points']

        scale_range = config['WORLD_SCALE_RANGE']
        if 'noise_scale' in data_dict:
            noise_scale = data_dict['noise_scale']
        else:
            noise_scale = np.random.uniform(scale_range[0], scale_range[1])
        points[:, :3] *= noise_scale
        occ[:, :3] *= noise_scale
        if 'sensor_loc' in data_dict:
            data_dict['sensor_loc'] *= noise_scale


        data_dict['occ'] = occ
        data_dict['points'] = points
        data_dict['noise_scale'] = noise_scale
        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        noise_translate_std = config['NOISE_TRANSLATE_STD']
        assert len(noise_translate_std) == 3
        if 'noise_translate' in data_dict:
            noise_translate = data_dict['noise_translate']
        else:
            noise_translate = np.array([
                np.random.normal(0, noise_translate_std[0], 1),
                np.random.normal(0, noise_translate_std[1], 1),
                np.random.normal(0, noise_translate_std[2], 1),
            ], dtype=np.float32).T

        occ, points = data_dict['occ'], data_dict['points']
        points[:, :3] += noise_translate
        occ[:, :3] += noise_translate
        if 'sensor_loc' in data_dict:
            data_dict['sensor_loc'] += noise_translate
                
        data_dict['occ'] = occ
        data_dict['points'] = points
        data_dict['noise_translate'] = noise_translate
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        return data_dict
