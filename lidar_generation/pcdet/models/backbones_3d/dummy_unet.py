from functools import partial

import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils
from .spconv_backbone import post_act_block
from .spconv_unet import UNetV2, SparseBasicBlock

#class DummyUNet(UNetV2):
class DummyUNet(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        #super().__init__(model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs)
        self.model_cfg = model_cfg
        input_channels = self.model_cfg.get('INPUT_CHANNELS', input_channels)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 32, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(32),
            nn.ReLU(),
        )
        # del self.conv1
        # del self.conv2
        # del self.conv3
        # del self.conv4
        # del self.conv5
        # del self.conv_up_m1
        # del self.conv_up_t1
        # del self.conv_up_m2
        # del self.conv_up_t2
        # del self.conv_up_m3
        # del self.conv_up_t3
        # del self.conv_up_m4
        # del self.conv_up_t4
        # del self.inv_conv2
        # del self.inv_conv3
        # del self.inv_conv4

        self.num_point_features = 32

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        batch_dict['point_features'] = x.features
        point_coords = common_utils.get_voxel_centers(
            x.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        batch_dict['point_coords'] = torch.cat((x.indices[:, 0:1].float(), point_coords), dim=1)
        batch_dict['feature_volumes'] = x
        return batch_dict

