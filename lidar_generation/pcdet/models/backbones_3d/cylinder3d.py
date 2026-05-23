# Copyright (c) OpenMMLab. All rights reserved.
r"""Modified from Cylinder3D.

Please refer to `Cylinder3D github page
<https://github.com/xinge008/Cylinder3D>`_ for details
"""

from typing import List, Optional

import numpy as np
import torch
from ...utils.spconv_utils import spconv
from torch import Tensor
import torch.nn as nn

def build_norm_layer(norm_cfg, num_features):
    _norm_cfg = norm_cfg.copy()
    assert _norm_cfg.pop('type') == 'BN1d'
    return None, nn.BatchNorm1d(num_features, **_norm_cfg)

def build_activation_layer(act_cfg):
    _act_cfg = act_cfg.copy()
    _type = _act_cfg.pop('type')
    _map = {'ReLU': nn.ReLU, 'LeakyReLU': nn.LeakyReLU, 'Sigmoid': nn.Sigmoid}
    assert _type in _map
    return _map[_type](**_act_cfg)


class AsymmResBlock(nn.Module):
    """Asymmetrical Residual Block.

    Args:
        in_channels (int): Input channels of the block.
        out_channels (int): Output channels of the block.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for
            normalization layer.
        act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
            Defaults to dict(type='LeakyReLU').
        indice_key (str, optional): Name of indice tables. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: dict,
                 act_cfg: dict = dict(type='LeakyReLU'),
                 indice_key: Optional[str] = None):
        super().__init__()

        self.conv0_0 = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 1),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef00')
        self.act0_0 = build_activation_layer(act_cfg)
        self.bn0_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv0_1 = spconv.SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef01')
        self.act0_1 = build_activation_layer(act_cfg)
        self.bn0_1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_0 = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef10')
        self.act1_0 = build_activation_layer(act_cfg)
        self.bn1_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_1 = spconv.SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 1),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef11')
        self.act1_1 = build_activation_layer(act_cfg)
        self.bn1_1 = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """Forward pass."""
        shortcut = self.conv0_0(x)

        shortcut = shortcut.replace_feature(self.act0_0(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_0(shortcut.features))

        shortcut = self.conv0_1(shortcut)
        shortcut = shortcut.replace_feature(self.act0_1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_1(shortcut.features))

        res = self.conv1_0(x)
        res = res.replace_feature(self.act1_0(res.features))
        res = res.replace_feature(self.bn1_0(res.features))

        res = self.conv1_1(res)
        res = res.replace_feature(self.act1_1(res.features))
        res = res.replace_feature(self.bn1_1(res.features))

        res = res.replace_feature(res.features + shortcut.features)

        return res


class AsymmeDownBlock(nn.Module):
    """Asymmetrical DownSample Block.

    Args:
       in_channels (int): Input channels of the block.
       out_channels (int): Output channels of the block.
       norm_cfg (:obj:`ConfigDict` or dict): Config dict for
            normalization layer.
       act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
            Defaults to dict(type='LeakyReLU').
       pooling (bool): Whether pooling features at the end of
           block. Defaults: True.
       height_pooling (bool): Whether pooling features at
           the height dimension. Defaults: False.
       indice_key (str, optional): Name of indice tables. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: dict,
                 act_cfg: dict = dict(type='LeakyReLU'),
                 pooling: bool = True,
                 height_pooling: bool = False,
                 indice_key: Optional[str] = None):
        super().__init__()
        self.pooling = pooling

        self.conv0_0 = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'asydown00')
        self.act0_0 = build_activation_layer(act_cfg)
        self.bn0_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv0_1 = spconv.SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 1),
            padding=1,
            bias=False,
            indice_key=indice_key + 'asydown01')
        self.act0_1 = build_activation_layer(act_cfg)
        self.bn0_1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_0 = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 1),
            padding=1,
            bias=False,
            indice_key=indice_key + 'asydown10')
        self.act1_0 = build_activation_layer(act_cfg)
        self.bn1_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_1 = spconv.SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'asydown11')
        self.act1_1 = build_activation_layer(act_cfg)
        self.bn1_1 = build_norm_layer(norm_cfg, out_channels)[1]

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    indice_key=indice_key,
                    bias=False)
            else:
                self.pool = spconv.SparseConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=(1, 2, 2),
                    padding=1,
                    indice_key=indice_key,
                    bias=False)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """Forward pass."""
        shortcut = self.conv0_0(x)
        shortcut = shortcut.replace_feature(self.act0_0(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_0(shortcut.features))

        shortcut = self.conv0_1(shortcut)
        shortcut = shortcut.replace_feature(self.act0_1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_1(shortcut.features))

        res = self.conv1_0(x)
        res = res.replace_feature(self.act1_0(res.features))
        res = res.replace_feature(self.bn1_0(res.features))

        res = self.conv1_1(res)
        res = res.replace_feature(self.act1_1(res.features))
        res = res.replace_feature(self.bn1_1(res.features))

        res = res.replace_feature(res.features + shortcut.features)

        if self.pooling:
            pooled_res = self.pool(res)
            return pooled_res, res
        else:
            return res


class AsymmeUpBlock(nn.Module):
    """Asymmetrical UpSample Block.

    Args:
        in_channels (int): Input channels of the block.
        out_channels (int): Output channels of the block.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for
                normalization layer.
        act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
                Defaults to dict(type='LeakyReLU').
        indice_key (str, optional): Name of indice tables. Defaults to None.
        up_key (str, optional): Name of indice tables used in
            SparseInverseConv3d. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: dict,
                 act_cfg: dict = dict(type='LeakyReLU'),
                 indice_key: Optional[str] = None,
                 up_key: Optional[str] = None):
        super().__init__()

        self.trans_conv = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'asyup_trans')
        self.trans_act = build_activation_layer(act_cfg)
        self.trans_bn = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1 = spconv.SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 1),
            padding=1,
            bias=False,
            indice_key=indice_key + 'asyup1')
        self.act1 = build_activation_layer(act_cfg)
        self.bn1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv2 = spconv.SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'asyup2')
        self.act2 = build_activation_layer(act_cfg)
        self.bn2 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv3 = spconv.SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'asyup3')
        self.act3 = build_activation_layer(act_cfg)
        self.bn3 = build_norm_layer(norm_cfg, out_channels)[1]

        self.up_subm = spconv.SparseInverseConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            indice_key=up_key,
            bias=False)

    def forward(self, x: spconv.SparseConvTensor,
                skip: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """Forward pass."""
        x_trans = self.trans_conv(x)
        x_trans = x_trans.replace_feature(self.trans_act(x_trans.features))
        x_trans = x_trans.replace_feature(self.trans_bn(x_trans.features))

        # upsample
        up = self.up_subm(x_trans)

        up = up.replace_feature(up.features + skip.features)

        up = self.conv1(up)
        up = up.replace_feature(self.act1(up.features))
        up = up.replace_feature(self.bn1(up.features))

        up = self.conv2(up)
        up = up.replace_feature(self.act2(up.features))
        up = up.replace_feature(self.bn2(up.features))

        up = self.conv3(up)
        up = up.replace_feature(self.act3(up.features))
        up = up.replace_feature(self.bn3(up.features))

        return up


class DDCMBlock(nn.Module):
    """Dimension-Decomposition based Context Modeling.

    Args:
        in_channels (int): Input channels of the block.
        out_channels (int): Output channels of the block.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for
            normalization layer.
        act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
            Defaults to dict(type='Sigmoid').
        indice_key (str, optional): Name of indice tables. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: dict,
                 act_cfg: dict = dict(type='Sigmoid'),
                 indice_key: Optional[str] = None):
        super().__init__()

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key+'ddcm1')
        self.bn1 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act1 = build_activation_layer(act_cfg)

        self.conv2 = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 1),
            padding=1,
            bias=False,
            indice_key=indice_key+'ddcm2')
        self.bn2 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act2 = build_activation_layer(act_cfg)

        self.conv3 = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            padding=1,
            bias=False,
            indice_key=indice_key+'ddcm3')
        self.bn3 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act3 = build_activation_layer(act_cfg)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """Forward pass."""
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.bn1(shortcut.features))
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        shortcut2 = self.conv2(x)
        shortcut2 = shortcut2.replace_feature(self.bn2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(self.act2(shortcut2.features))

        shortcut3 = self.conv3(x)
        shortcut3 = shortcut3.replace_feature(self.bn3(shortcut3.features))
        shortcut3 = shortcut3.replace_feature(self.act3(shortcut3.features))
        shortcut = shortcut.replace_feature(shortcut.features + \
            shortcut2.features + shortcut3.features)

        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut


class Asymm3DSpconv(nn.Module):
    """Asymmetrical 3D convolution networks.

    Args:
        grid_size (int): Size of voxel grids.
        input_channels (int): Input channels of the block.
        base_channels (int): Initial size of feature channels before
            feeding into Encoder-Decoder structure. Defaults to 16.
        backbone_depth (int): The depth of backbone. The backbone contains
            downblocks and upblocks with the number of backbone_depth.
        height_pooing (List[bool]): List indicating which downblocks perform
            height pooling.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01)).
        init_cfg (dict, optional): Initialization config.
            Defaults to None.
    """

    # def __init__(self,
    #              grid_size: int,
    #              input_channels: int,
    #              base_channels: int = 16,
    #              backbone_depth: int = 4,
    #              height_pooling: List[bool] = [True, True, False, False],
    #              norm_cfg: ConfigType = dict(
    #                  type='BN1d', eps=1e-3, momentum=0.01),
    #              init_cfg=None):

    # pcdet style __init__
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        base_channels = model_cfg['base_channels']
        backbone_depth = model_cfg.get('backbone_depth', 4)
        height_pooing = model_cfg.get('height_pooling', [True, True, False, False])
        norm_cfg = model_cfg.get('norm_cfg', dict(type='BN1d', eps=1e-3, momentum=0.01))
        input_channels = model_cfg.get('INPUT_CHANNELS', input_channels)

        self.grid_size = grid_size
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.backbone_depth = backbone_depth
        self.down_context = AsymmResBlock(
            input_channels, base_channels, indice_key='pre', norm_cfg=norm_cfg)

        self.down_block_list = torch.nn.ModuleList()
        self.up_block_list = torch.nn.ModuleList()
        for i in range(self.backbone_depth):
            self.down_block_list.append(
                AsymmeDownBlock(
                    2**i * base_channels,
                    2**(i + 1) * base_channels,
                    height_pooling=height_pooing[i],
                    indice_key='down' + str(i),
                    norm_cfg=norm_cfg))
            if i == self.backbone_depth - 1:
                self.up_block_list.append(
                    AsymmeUpBlock(
                        2**(i + 1) * base_channels,
                        2**(i + 1) * base_channels,
                        up_key='down' + str(i),
                        indice_key='up' + str(self.backbone_depth - 1 - i),
                        norm_cfg=norm_cfg))
            else:
                self.up_block_list.append(
                    AsymmeUpBlock(
                        2**(i + 2) * base_channels,
                        2**(i + 1) * base_channels,
                        up_key='down' + str(i),
                        indice_key='up' + str(self.backbone_depth - 1 - i),
                        norm_cfg=norm_cfg))

        self.ddcm = DDCMBlock(
            2 * base_channels,
            2 * base_channels,
            indice_key='ddcm',
            norm_cfg=norm_cfg)

        self.channel_reduction = spconv.SparseSequential(
            spconv.SubMConv3d(
                4 * base_channels, 2 * base_channels, kernel_size=1
            ),
            build_norm_layer(norm_cfg, 2 * base_channels)[1],
            nn.LeakyReLU(),
            spconv.SubMConv3d(
                2 * base_channels, 2 * base_channels, kernel_size=1
            ),
            build_norm_layer(norm_cfg, 2 * base_channels)[1],
            nn.LeakyReLU(),
        )
        self.num_point_features = 2 * base_channels

    # def forward(self, voxel_features: Tensor, coors: Tensor,
    #             batch_size: int) -> spconv.SparseConvTensor:
    def forward(self, batch_dict):
        """Forward pass."""
        # coors = coors.int()
        # ret = spconv.SparseConvTensor(voxel_features, coors, np.array(self.grid_size),
        #                        batch_size)
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        ret = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )


        ret = self.down_context(ret)

        down_skip_list = []
        down_pool = ret
        for i in range(self.backbone_depth):
            down_pool, down_skip = self.down_block_list[i](down_pool)
            down_skip_list.append(down_skip)

        up = down_pool
        for i in range(self.backbone_depth - 1, -1, -1):
            up = self.up_block_list[i](up, down_skip_list[i])

        ddcm = self.ddcm(up)
        ddcm = ddcm.replace_feature(torch.cat((ddcm.features, up.features), 1))

        ddcm = self.channel_reduction(ddcm)

        batch_dict['feature_volumes'] = ddcm
        return batch_dict