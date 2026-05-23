from functools import partial

import torch
import torch.nn as nn
from ...utils.spconv_utils import replace_feature, spconv
from ..model_utils.ptv3_utils import Point
from .ptv3 import PointTransformerV3

class PTv3(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.ptv3 = PointTransformerV3(**model_cfg.PTv3)
        self.num_point_features = model_cfg.PTv3.dec_channels[-1]

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
        point_feats = voxel_features
        point_coords = voxel_coords.int()
        point_dict = dict(
            coord = point_feats[:, [2, 1, 0]],
            grid_coord = point_coords[:, 1:4],
            batch = point_coords[:, 0],
            feat = point_feats,
            sparse_shape = self.sparse_shape,
            sparse_conv_feat = input_sp_tensor,
        )
        point = Point(point_dict)
        point.serialization(order=self.ptv3.order, shuffle_orders=self.ptv3.shuffle_orders)
        #point.sparsify()
        point = self.ptv3.embedding(point)
        point = self.ptv3.enc(point)
        point = self.ptv3.dec(point)
        batch_dict['point_features'] = point['feat']
        batch_dict['feature_volumes'] = point.sparse_conv_feat
        return batch_dict
