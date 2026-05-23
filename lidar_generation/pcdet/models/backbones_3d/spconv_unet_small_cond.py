from functools import partial

import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils
from .spconv_backbone import post_act_block
from .spconv_unet_small import UNetV2Small
from .lidar_embedding import LidarEmbedding, LidarEmbResampler

class LidarConditionalAttention(nn.Module):
    def __init__(self, in_dim, emb_dim, num_heads=8):
        super().__init__()
        q_dim = in_dim
        k_dim = emb_dim
        v_dim = emb_dim
        self.attn = nn.MultiheadAttention(q_dim, kdim=k_dim, vdim=v_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(q_dim)
    
    def forward(self, x, lidar_embeddings):
    # x: sparse tensor, lidar_embeddings: [B, nlidar, C]
        batch_size = x.batch_size
        new_feats = torch.zeros_like(x.features)
        for i in range(batch_size):
            batch_mask = x.indices[:, 0]==i
            attn_out, _ = self.attn(self.norm(x.features[batch_mask].unsqueeze(0)), lidar_embeddings[i:i+1], lidar_embeddings[i:i+1])
            new_feats[batch_mask] = attn_out
        x.replace_feature(new_feats + x.features)
        return x

class LidarConditionalModulated(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super().__init__()
        hidden_dims = 64
        self.scale_shift_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, in_dim * 3, bias=True)

            # nn.SiLU(),
            # nn.Linear(emb_dim, hidden_dims, bias=True),
            # nn.LayerNorm(hidden_dims),

            # nn.SiLU(),
            # nn.Linear(hidden_dims, hidden_dims, bias=True),
            # nn.LayerNorm(hidden_dims),

            # nn.SiLU(),
            # nn.Linear(hidden_dims, hidden_dims, bias=True),
            # nn.LayerNorm(hidden_dims),

            # nn.SiLU(),
            # nn.Linear(hidden_dims, in_dim * 3, bias=True)

            )
        self.conv = spconv.SubMConv3d(in_dim, in_dim, 3, padding=1, bias=False)
        nn.init.zeros_(self.scale_shift_gate[-1].weight)
        nn.init.zeros_(self.scale_shift_gate[-1].bias)


    def forward(self, x, lidar_embeddings):
        batch_size = x.batch_size
        new_feats = torch.zeros_like(x.features)
        _x = self.conv(x).features
        for i in range(batch_size):
            batch_mask = x.indices[:, 0]==i
            _lidar_embs = lidar_embeddings[i:i+1]
            scale, shift, gate = self.scale_shift_gate(_lidar_embs).chunk(3, dim=-1)
            new_feats[batch_mask] = (scale * _x[batch_mask] + shift) * gate
        x.replace_feature(new_feats + x.features)
        return x

class LidarConditionalContorlNet(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super().__init__()
        hidden_dims = 64
        self.conditioner = nn.Sequential(

            # nn.SiLU(),
            # nn.Linear(emb_dim, in_dim, bias=True)

            nn.SiLU(),
            nn.Linear(emb_dim, hidden_dims, bias=True),
            nn.LayerNorm(hidden_dims),

            nn.SiLU(),
            nn.Linear(hidden_dims, hidden_dims, bias=True),
            nn.LayerNorm(hidden_dims),

            nn.SiLU(),
            nn.Linear(hidden_dims, hidden_dims, bias=True),
            nn.LayerNorm(hidden_dims),

            nn.SiLU(),
            nn.Linear(hidden_dims, in_dim, bias=True)
            )
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)


    def forward(self, x, lidar_embeddings):
        batch_size = x.batch_size
        new_feats = torch.zeros_like(x.features)
        for i in range(batch_size):
            batch_mask = x.indices[:, 0]==i
            _lidar_embs = lidar_embeddings[i:i+1]
            new_feats[batch_mask] = self.conditioner(_lidar_embs)
        x.replace_feature(new_feats + x.features)
        return x

class LidarConditionalConcat(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super().__init__()
        hidden_dims = 64
        self.conditioner = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, hidden_dims, bias=True),
            nn.LayerNorm(hidden_dims),

            nn.SiLU(),
            nn.Linear(hidden_dims, hidden_dims, bias=True),
            nn.LayerNorm(hidden_dims),

            nn.SiLU(),
            nn.Linear(hidden_dims, in_dim, bias=True)
            )
        #nn.init.zeros_(self.conditioner[-1].weight)
        #nn.init.zeros_(self.conditioner[-1].bias)
        self.channel_reduce = nn.Linear(2*in_dim, in_dim)


    def forward(self, x, lidar_embeddings):
        batch_size = x.batch_size
        new_feats = torch.zeros_like(x.features)
        for i in range(batch_size):
            batch_mask = x.indices[:, 0]==i
            _lidar_embs = lidar_embeddings[i:i+1]
            new_feats[batch_mask] = self.conditioner(_lidar_embs)
        x.replace_feature(self.channel_reduce(torch.cat([new_feats, x.features], dim=-1)))
        return x

class UNetV2SmallCond(UNetV2Small):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_map = {'conv0': 32, 'conv1': 32, 'conv2': 64, 'conv3': 64, 'conv4': 64, 'deconv1': 64, 'deconv2': 64, 'deconv3': 32, 'deconv4': 32}
        lidar_emb_cfg = self.model_cfg.get('lidar_emb_cfg')
        lidar_cond_cfg = self.model_cfg.get('lidar_cond_cfg')
        self.cond_locs = lidar_cond_cfg.get('cond_locs', [0, 1, 2, 3])
        self.lidar_emb = LidarEmbedding(lidar_emb_cfg)
        cond_type = lidar_cond_cfg.pop('type')
        for loc in self.cond_locs:
            if loc > 4:
                in_dim = self.channel_map[f'deconv{loc-4}']
            else:   
                in_dim = self.channel_map[f'conv{loc}']
            if cond_type == 'cross_attn':
                self.__setattr__(f'lidar_attn_{loc}', LidarConditionalAttention(in_dim, self.lidar_emb.emb_dim))
            elif cond_type == 'film':
                self.__setattr__(f'lidar_attn_{loc}', LidarConditionalModulated(in_dim=in_dim, emb_dim=self.lidar_emb.emb_dim))
            elif cond_type == 'controlnet':
                self.__setattr__(f'lidar_attn_{loc}', LidarConditionalContorlNet(in_dim=in_dim, emb_dim=self.lidar_emb.emb_dim))
            elif cond_type == 'concat':
                self.__setattr__(f'lidar_attn_{loc}', LidarConditionalConcat(in_dim=in_dim, emb_dim=self.lidar_emb.emb_dim))
            else:
                raise NotImplementedError

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
        lidar_embeddings = self.lidar_emb(batch_dict['lidar_chosen_mask'].bool()) # [B, nlidar, C]
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        if 0 in self.cond_locs:
            x = self.__getattr__(f'lidar_attn_0')(x, lidar_embeddings)

        x_conv1 = self.conv1(x)

        if 1 in self.cond_locs:
            x_conv1 = self.__getattr__(f'lidar_attn_1')(x_conv1, lidar_embeddings)

        x_conv2 = self.conv2(x_conv1)

        if 2 in self.cond_locs:
            x_conv2 = self.__getattr__(f'lidar_attn_2')(x_conv2, lidar_embeddings)

        x_conv3 = self.conv3(x_conv2)

        if 3 in self.cond_locs:
            x_conv3 = self.__getattr__(f'lidar_attn_3')(x_conv3, lidar_embeddings)

        x_conv4 = self.conv4(x_conv3)

        if 4 in self.cond_locs:
            x_conv4 = self.__getattr__(f'lidar_attn_4')(x_conv4, lidar_embeddings)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)

        if 5 in self.cond_locs:
            x_up4 = self.__getattr__(f'lidar_attn_5')(x_up4, lidar_embeddings)

        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)

        if 6 in self.cond_locs:
            x_up3 = self.__getattr__(f'lidar_attn_6')(x_up3, lidar_embeddings)

        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)

        if 7 in self.cond_locs:
            x_up2 = self.__getattr__(f'lidar_attn_7')(x_up2, lidar_embeddings)

        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        if 8 in self.cond_locs:
            x_up1 = self.__getattr__(f'lidar_attn_8')(x_up1, lidar_embeddings)

        batch_dict['point_features'] = x_up1.features
        point_coords = common_utils.get_voxel_centers(
            x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        batch_dict['feature_volumes'] = x_up1
        return batch_dict