import torch
import torch.nn as nn
from ...utils.spconv_utils import spconv
from ..backbones_3d.ptv3 import VoxelAttnPTv3

class TemporalBlock(spconv.SparseModule):
    def __init__(self, block_cfg, residual=False):
        super().__init__()
        block_type = block_cfg.pop('type')
        if block_type == 'conv3d':
            self.block = TemporalSparseConv3d(**block_cfg)
        elif block_type == 'conv4d':
            self.block = TemporalSparseConv4d(**block_cfg)
        elif block_type == 'ptv3':
            self.block = TemporalAttention(**block_cfg)
        elif block_type == 'axial':
            self.block = TemporalAttentionAxial(**block_cfg)
        else:
            raise NotImplementedError
        self.residual = residual
    
    def forward(self, x):
        if not hasattr(x, 'real_batch_size'):
            x.__setattr__('real_batch_size', int((x.indices[-1, 0]+1) // self.block.n_frames))
        if self.residual:
            return self.block(x) + x
        else:
            return self.block(x)


class TemporalSparseConv3d(nn.Module):
    def __init__(self, n_frames, input_channels):
        super().__init__()
        self.n_frames = n_frames
        self.time_embd = nn.Embedding(self.n_frames, input_channels)
        self.conv = spconv.SubMConv3d(input_channels, input_channels, kernel_size=3)

    def forward(self, x):
        batch_size = x.real_batch_size
        # x: sparse tensor, coords [N, 4(b*f,z,y,x)] features [N, C]
        temporal_feats = x.features.clone()
        temporal_coords = x.indices.clone()
        ori_idxs = temporal_coords[:, 0].clone()
        for i in range(batch_size):
            temporal_mask = (ori_idxs >= i * self.n_frames) & (ori_idxs < (i+1) * self.n_frames)
            temporal_coords[temporal_mask, 0] = i
        for i in range(self.n_frames):
            temporal_mask = ori_idxs % self.n_frames
            temporal_feats[temporal_mask==i] += self.time_embd(torch.tensor(i, dtype=int, device=temporal_feats.device))
        input_sp_tensor = spconv.SparseConvTensor(
            features=temporal_feats,
            indices=temporal_coords.int(),
            spatial_shape=x.spatial_shape,
            batch_size=batch_size
        )
        x_new = self.conv(input_sp_tensor)
        x.replace_feature(x_new.features)
        return x


class TemporalSparseConv4d(nn.Module):
    def __init__(self, n_frames, input_channels):
        super().__init__()
        self.n_frames = n_frames
        self.conv = spconv.SubMConv4d(input_channels, input_channels, kernel_size=3)

    def forward(self, x):
        batch_size = x.real_batch_size
        ori_idxs = x.indices[:, 0].clone()
        temporal_coords = ori_idxs % self.n_frames
        x_tem_coords = torch.zeros([x.indices.shape[0], 5], device=x.features.device, dtype=x.features.dtype)
        for i in range(batch_size):
            temporal_mask = (ori_idxs >= i * self.n_frames) & (ori_idxs < (i+1) * self.n_frames)
            x_tem_coords[temporal_mask, 0] = i
        x_tem_coords[:, 1] = temporal_coords
        x_tem_coords[:, 2:] = x.indices[:, 1:]
        input_sp_tensor = spconv.SparseConvTensor(
            features=x.features,
            indices=x_tem_coords.int(),
            spatial_shape=[self.n_frames, *x.spatial_shape],
            batch_size=batch_size
        )
        x_new = self.conv(input_sp_tensor)
        x.replace_feature(x_new.features)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, n_frames, input_channels, attn_cfg=dict()):
        super().__init__()
        self.n_frames = n_frames
        default_attn_cfg = dict(
                enc_depths=(2,),
                enc_channels=(input_channels,),
                enc_num_head=(8,),
                enc_patch_size=(48,),
                enable_flash=False,
            )
        default_attn_cfg.update(**attn_cfg)
        self.voxel_attn = VoxelAttnPTv3(**default_attn_cfg)
        self.time_embd = nn.Embedding(self.n_frames, input_channels)

    def forward(self, x):
        batch_size = x.real_batch_size
        # x: sparse tensor, coords [N, 4(b*f,z,y,x)] features [N, C]
        temporal_feats = x.features.clone()
        temporal_coords = x.indices.clone()
        ori_idxs = temporal_coords[:, 0].clone()
        for i in range(batch_size):
            temporal_mask = (ori_idxs >= i * self.n_frames) & (ori_idxs < (i+1) * self.n_frames)
            temporal_coords[temporal_mask, 0] = i
        for i in range(self.n_frames):
            temporal_mask = ori_idxs % self.n_frames
            temporal_feats[temporal_mask==i] += self.time_embd(torch.tensor(i, dtype=int, device=temporal_feats.device))
        input_sp_tensor = spconv.SparseConvTensor(
            features=temporal_feats,
            indices=temporal_coords.int(),
            spatial_shape=x.spatial_shape,
            batch_size=batch_size
        )
        x_new = self.voxel_attn(input_sp_tensor)
        x.replace_feature(x_new.features)
        return x

def compute_group_positions(inverse: torch.Tensor, counts: torch.Tensor):
    """
    Args:
        inverse: [N], 每个点属于哪个 group
        counts: [B], 每个 group 的点数
    
    Returns:
        group_pos: [N], 每个点在 group 中的相对位置
    """
    device = inverse.device
    # 先构造每个 group 的起始索引
    group_start_idx = torch.zeros_like(counts)
    group_start_idx[1:] = torch.cumsum(counts, dim=0)[:-1]
    # 每个点的位置 = 全局索引 - group 起始索引
    point_indices = torch.arange(inverse.size(0), device=device)
    group_pos = point_indices - group_start_idx[inverse]
    return group_pos


class TemporalAttentionAxial(nn.Module):
    def __init__(self, n_frames, input_channels, axial, attn_cfg=dict()):
        super().__init__()
        self.n_frames = n_frames
        axial_map = dict(t=1, z=2, y=3, x=4)
        self.axial = axial_map[axial] # (b t z y x)
        attn_type = attn_cfg.pop('type', 'multiheadattn')
        
        if attn_type == 'multiheadattn':
            attn_cfg['embed_dim'] = input_channels
            self.attn = nn.MultiheadAttention(**attn_cfg)
        else:
            raise NotImplementedError
    
    def forward(self, x):
        batch_size = x.real_batch_size
        batch_size = x.real_batch_size
        ori_idxs = x.indices[:, 0].clone()
        temporal_coords = ori_idxs % self.n_frames
        x_tem_coords = torch.zeros([x.indices.shape[0], 5], device=x.features.device, dtype=x.features.dtype)
        for i in range(batch_size):
            temporal_mask = (ori_idxs >= i * self.n_frames) & (ori_idxs < (i+1) * self.n_frames)
            x_tem_coords[temporal_mask, 0] = i
        x_tem_coords[:, 1] = temporal_coords
        x_tem_coords[:, 2:] = x.indices[:, 1:]
        input_sp_tensor = spconv.SparseConvTensor(
            features=x.features,
            indices=x_tem_coords.int(),
            spatial_shape=[self.n_frames, *x.spatial_shape],
            batch_size=batch_size
        )
        
        axis_idx = self.axial

        features = input_sp_tensor.features  # [N, C]
        indices = input_sp_tensor.indices   # [N, 5]，[batch_idx, t, z, y, x]
        device = features.device
        N, C = features.shape

        # 构造 group_id：保留 batch 和除目标轴外的其他坐标
        group_ids = torch.cat([
            indices[:, :axis_idx],
            indices[:, axis_idx+1:]
        ], dim=1)  # [N, 3]

        # unique group id → [B, 3], inverse: [N], counts: [B]
        group_flat, inverse = torch.unique(group_ids, return_inverse=True, dim=0)
        counts = torch.bincount(inverse)
        B = group_flat.shape[0]
        L = counts.max().item()

        # 排序 (group 内连续)
        sorted_vals, sort_idx = torch.sort(inverse)
        sorted_features = features[sort_idx]
        inverse_sorted = inverse[sort_idx]

        # group 内相对位置
        group_pos = compute_group_positions(inverse_sorted, counts)

        # 分配 padded 特征和 mask
        grouped_features = torch.zeros((B, L, C), dtype=features.dtype, device=device)
        padding_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        grouped_features[inverse_sorted, group_pos] = sorted_features
        padding_mask[inverse_sorted, group_pos] = True

        # 送入 attention 模块
        feat = grouped_features.permute(1, 0, 2)  # [L, B, C]
        mask = ~padding_mask  # True 表示需要被 mask
        attn_out, _ = self.attn(feat, feat, feat, key_padding_mask=mask)
        out = attn_out.permute(1, 0, 2)

        # 映射回原始顺序
        unsort_idx = torch.argsort(sort_idx)
        attn_features = out[inverse_sorted, group_pos][unsort_idx]  # [N, C]
        x.replace_feature(attn_features)
        return x