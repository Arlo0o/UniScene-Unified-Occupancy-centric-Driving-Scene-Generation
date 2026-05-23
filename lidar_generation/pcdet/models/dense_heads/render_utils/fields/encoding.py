
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import json
import math
from .mlp import MLP
from pcdet.utils.common_utils import torch_unique

def get_1d_sin_cos_pos_encoding(pos: torch.Tensor, num_feats: int) -> torch.Tensor:
    """
    pos: [B, L, 1] - normalized scalar positions ∈ [0, 1]
    return: [B, L, num_feats]
    """
    div_term = torch.exp(
        torch.arange(0, num_feats, 2, device=pos.device) * (-torch.log(torch.tensor(10000.0)) / num_feats)
    )  # [num_feats // 2]

    sin = torch.sin(pos * div_term)
    cos = torch.cos(pos * div_term)
    return torch.cat([sin, cos], dim=-1)  # [B, L, num_feats]

def get_1d_fourier_encoding(pos: torch.Tensor, num_feats: int, scale: float = 1.0) -> torch.Tensor:
    assert num_feats % 2 == 0
    num_frequencies = num_feats // 2
    freq_bands = 2 ** torch.arange(num_frequencies) * scale        
    freqs = freq_bands.to(pos.device)  # [F]
    scaled = pos * freqs  # [B, L, F]
    pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # [B, L, 2F]
    return pe

def compute_batch_histogram(x_batch: torch.Tensor, num_bins=32, min_val=0.0, max_val=1.0):
    """
    x_batch: [B, N]，每行是一个样本，值不一定落在 [0, 1]
    num_bins: 直方图的bin数量
    min_val, max_val: 值范围（必须手动指定或从x_batch中计算）
    return: [B, num_bins]，每行是归一化后的概率直方图
    """
    B, N = x_batch.shape

    # 生成bin的边界（共 num_bins+1 个边界）
    bin_edges = torch.linspace(min_val, max_val, num_bins + 1, device=x_batch.device)

    # 每个值映射到bin编号（范围：[0, num_bins-1]）
    bin_ids = torch.bucketize(x_batch.contiguous(), bin_edges, right=False) - 1
    bin_ids = torch.clamp(bin_ids, 0, num_bins - 1)  # 防止越界

    # 构建 histogram
    hist = torch.zeros((B, num_bins), device=x_batch.device)
    hist.scatter_add_(1, bin_ids, torch.ones_like(bin_ids, dtype=torch.float))
    hist += 1e-6  # 避免KL中 log(0)
    hist /= hist.sum(dim=1, keepdim=True)  # 概率归一化

    return hist  # shape: [B, num_bins]

class HistWeightedEmbedding(torch.nn.Module):
    def __init__(self, n_bins=32, emb_dim=16):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.randn(n_bins, emb_dim))
        self.out_dim = emb_dim

    def forward(self, hist):
        # hist: [B, num_bins]，self.embedding: [num_bins, emb_dim]
        return hist @ self.embedding  # [B, emb_dim]

class RayHistEmbedding(nn.Module):
    def __init__(self, n_bins, vmin, vmax, encoder_cfg=dict(), ray_interaction_cfg=dict()):
        super().__init__()
        self.n_bins = n_bins
        self.vmin = vmin
        self.vmax = vmax
        encoder_type = encoder_cfg.pop('type', None)
        if encoder_type == 'mlp':
            self.encoder = MLP(**dict(**encoder_cfg, in_dim=n_bins))
        elif encoder_type == 'weighted_embedding':
            self.encoder = HistWeightedEmbedding(**dict(**encoder_cfg, n_bins=n_bins))
        else:
            self.encoder = None
        
        ray_interaction_type = ray_interaction_cfg.pop('type', None)
        if ray_interaction_type == 'conv2d':
            in_channels = self.encoder.out_dim if self.encoder is not None else n_bins
            self.ray_interaction = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.ray_interaction = None


    def ray_to_range_map(self, ray_d, lidar_idx, ring_idx, ray_feats, cell_size=1.0):
        unique_lidar_ids = torch.unique(lidar_idx)
        coords = []
        feat_maps = []
        valid_masks = []
        reverse_inds = []
        for lidar_id in unique_lidar_ids:
            lidar_mask = (lidar_idx == lidar_id)
            thetas = torch.rad2deg(torch.atan2(ray_d[lidar_mask, 1], ray_d[lidar_mask, 0]))
            thetas += 180 # [-180, 180] -> [0, 360]
            phis = ring_idx[lidar_mask]
            phis = torch.round(phis).int()
            thetas = torch.round(thetas / cell_size).int()
            _coords = torch.stack([phis, thetas], dim=-1)
            _, inverse, index, _ = torch_unique(_coords, dim=0)
            # print(ray_feats[lidar_mask].shape, thetas.shape, index.min(), index.max())
            thetas = thetas[index]
            phis = phis[index]
            feat_map = torch.zeros([int(phis.max()+1), int(thetas.max()+1), ray_feats.shape[-1]], dtype=ray_feats.dtype, device=ray_feats.device)
            # print(feat_map.shape, phis.min(), phis.max(), thetas.min(), thetas.max())
            feat_map[phis, thetas] = ray_feats[lidar_mask][index]
            valid_map = torch.zeros(feat_map.shape[:-1], dtype=torch.bool, device=feat_map.device)
            valid_map[phis, thetas] = True

            feat_maps.append(feat_map)
            valid_masks.append(valid_map)
            coords.append(torch.stack([phis, thetas], dim=-1))
            reverse_inds.append(inverse)
        return feat_maps, valid_masks, coords, reverse_inds, unique_lidar_ids

    def forward(self, ray_samples):
        # x: [N, C] (C is the number of samples along the ray)
        # hist: [N, n_bins]
        x = ray_samples.frustums.starts.squeeze(-1)
        hist = compute_batch_histogram(x, num_bins=self.n_bins, min_val=self.vmin, max_val=self.vmax)
        hist_features = hist
        if self.encoder is not None:
            hist_features = self.encoder(hist)

        if self.ray_interaction is not None:
            # TODO: need global interaction?
            feat_maps, valid_masks, coords, inverse_inds, lidar_ids = self.ray_to_range_map(
                ray_samples.frustums.directions[:, 0, :], # same direction along the ray 
                ray_samples.frustums.extra_info['lidar_idx'],
                ray_samples.frustums.extra_info['ring_idx'],
                hist_features,
                cell_size=0.4)

            new_hist_features = torch.zeros_like(hist_features)
            _feats = []
            for i in range(len(feat_maps)):
                feat_maps[i] = self.ray_interaction(feat_maps[i].permute(2, 0, 1)[None]).permute(0, 2, 3, 1).squeeze()

                _feats.append(feat_maps[i][coords[i][:, 0], coords[i][:, 1]][inverse_inds[i]])
            
            for i, lidar_id in enumerate(lidar_ids):
                lidar_mask = (ray_samples.frustums.extra_info['lidar_idx'] == lidar_id)
                new_hist_features[lidar_mask] = _feats[i]
            hist_features = new_hist_features

        return hist, hist_features

class PointInRayEmbedding(nn.Module):
    def __init__(self, emb_type, n_feats):
        super().__init__()
        self.emb_type = emb_type
        self.n_feats = n_feats
    def forward(self, x):
        # x: [B, L, 1]
        if self.emb_type == 'sin':
            return get_1d_sin_cos_pos_encoding(x, self.n_feats)
        elif self.emb_type == 'fourier':
            return get_1d_fourier_encoding(x, self.n_feats)
        else:
            raise NotImplementedError


def plucker_embedding(ray_o, ray_d):
    moments = torch.cross(ray_o, ray_d, dim=-1)
    embedding = torch.cat([ray_d, ray_o], dim=-1)
    return embedding

class XYZ_Encoder(nn.Module):
    encoder_type = "XYZ_Encoder"
    """Encode XYZ coordinates or directions to a vector."""

    def __init__(self, n_input_dims):
        super().__init__()
        self.n_input_dims = n_input_dims

    @property
    def n_output_dims(self) -> int:
        raise NotImplementedError



class SinusoidalEncoder(XYZ_Encoder):
    encoder_type = "SinusoidalEncoder"
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(
        self,
        n_input_dims: int = 3,
        min_deg: int = 0,
        max_deg: int = 10,
        enable_identity: bool = True,
    ):
        super().__init__(n_input_dims)
        self.n_input_dims = n_input_dims
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.enable_identity = enable_identity
        self.register_buffer("scales", Tensor([2 ** i for i in range(min_deg, max_deg + 1)]))

    @property
    def n_output_dims(self) -> int:
        return (int(self.enable_identity) + (self.max_deg - self.min_deg + 1) * 2) * self.n_input_dims

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., n_input_dims]
        Returns:
            encoded: [..., n_output_dims]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[..., None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg + 1) * self.n_input_dims],
        )
        encoded = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.enable_identity:
            encoded = torch.cat([x] + [encoded], dim=-1)
        return encoded

