import math
import torch
import torch.nn as nn
from uniscenev2_lidar.datasets.nuscenes_occ.nuplan_constants import NUPLAN_LIDAR_LOCS, NUPLAN_LIDAR_RAY_RANGE

class FourierPositionalEmbedding(nn.Module):
    def __init__(self, L=6):
        super().__init__()
        self.L = L
        self.emb_dim = L * 6
        self.out_dim = self.emb_dim

    def forward(self, xyz):
        # xyz: Tensor of shape [..., 3]
        freq_bands = 2 ** torch.arange(self.L, device=xyz.device).float() * math.pi  # [L]
        freqs = xyz[..., None, :] * freq_bands[:, None]  # [..., L, 3]
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        return torch.cat([sin, cos], dim=-1).view(*xyz.shape[:-1], -1)

class MLPEmbedding(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[64, 64], output_dim=32, activation=nn.ReLU):
        """
        MLP 映射编码器
        Args:
            input_dim: 输入维度（通常是 3D 坐标）
            hidden_dims: 隐藏层维度列表
            output_dim: 输出编码的维度
            activation: 激活函数（默认 ReLU）
        """
        super().__init__()
        self.emb_dim = output_dim
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())

        # 输出层
        layers.append(nn.Linear(dims[-1], output_dim))
        self.mlp = nn.Sequential(*layers)
        self.out_dim = output_dim

    def forward(self, x):
        """
        Args:
            x: 输入 tensor，形状为 [B, 3] 或 [..., 3]
        Returns:
            输出编码特征，形状为 [B, output_dim] 或 [..., output_dim]
        """
        original_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])  # flatten
        out = self.mlp(x)
        return out.view(*original_shape, -1)

class LearnableEmbedding(nn.Module):
    def __init__(self, num_positions, embed_dim):
        """
        Learnable 位置编码，基于索引查表
        Args:
            num_positions: 可用的位置数量（例如 1000 个位置）
            embed_dim: 输出的编码维度
        """
        super().__init__()
        self.emb_dim = embed_dim
        self.embedding = nn.Embedding(num_positions, embed_dim)
        self.out_dim = embed_dim

    def forward(self, position_idx):
        """
        Args:
            position_idx: [...], 每个位置是 [0, num_positions-1] 的整数索引
        Returns:
            [..., embed_dim] 的 learnable encoding
        """
        return self.embedding(position_idx)


class LidarEmbResampler(nn.Module):
    def __init__(self, emb_dim, use_full_transformer=False):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn((1, emb_dim)))
        self.use_full_transformer = use_full_transformer
        if use_full_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=2, batch_first=True)
            self.attn = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)
        else:
            self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=2, batch_first=True)
    
    def forward(self, x):
        batch_size = x.shape[0]
        _x = torch.cat([x, self.cls_token[None].expand(batch_size, -1, -1)], dim=1)
        if self.use_full_transformer:
            attn_out = self.attn(_x)
        else:
            attn_out, _ = self.attn(_x, _x, _x)
        return attn_out[:, -1, :]

class LidarRangeEmbedding(nn.Module):
    def __init__(self, emb_cfg=None):
        super().__init__()
        self.emb_dim = 6 + 6 + 1

    def plucker_embedding(self, ray_o, ray_d):
        moments = torch.cross(ray_o, ray_d, dim=-1)
        embedding = torch.cat([ray_d, ray_o], dim=-1)
        return embedding

    def forward(self, x):
        # x[i]: [x1, y1, z1, x2, y2, z2, angle]
        ray_o = torch.zeros_like(x[:, :3])
        ray_d1 = torch.nn.functional.normalize(x[:, :3], p=2, dim=-1)
        ray_d2 = torch.nn.functional.normalize(x[:, 3:6], p=2, dim=-1)
        emb1 = self.plucker_embedding(ray_o, ray_d1)
        emb2 = self.plucker_embedding(ray_o, ray_d2)
        emb3 = torch.deg2rad(x[:, 6:7]) / 6.283185307179586
        return torch.cat([emb1, emb2, emb3], dim=-1)

class LidarEmbedding(nn.Module):
    def __init__(self, emb_cfg):
        super().__init__()
        lidar_loc = list(NUPLAN_LIDAR_LOCS.values())
        self.register_buffer('lidar_loc', torch.tensor(lidar_loc).float())
        lidar_range = list(NUPLAN_LIDAR_RAY_RANGE.values())
        self.register_buffer('lidar_range', torch.tensor(lidar_range).float())
        emb_type = emb_cfg.pop('type')
        resampler_cfg = emb_cfg.pop('resampler_cfg', {})
        self.enabled = emb_cfg.pop('enabled', True)
        self.apply_resampler = emb_cfg.pop('apply_resampler', False)
        self.with_ray_range = emb_cfg.pop('with_ray_range', False)
        self.emb_proj_dim = emb_cfg.pop('emb_proj_dim', -1)
        if emb_type == 'fourier':
            self.lidar_emb = FourierPositionalEmbedding(**emb_cfg)
        elif emb_type == 'mlp':
            self.lidar_emb = MLPEmbedding(**emb_cfg)
        elif emb_type == 'learnable':
            self.lidar_emb = LearnableEmbedding(**emb_cfg)
        else:
            raise NotImplementedError
        self.emb_dim = self.lidar_emb.emb_dim

        if self.with_ray_range:
            self.ray_range_emb = LidarRangeEmbedding()
            self.emb_dim += self.ray_range_emb.emb_dim

        if self.emb_proj_dim != -1:
            self.emb_proj = nn.Sequential(
                nn.Linear(self.emb_dim, self.emb_proj_dim, bias=True),
                nn.LayerNorm(self.emb_proj_dim),
                nn.ReLU(),
            )
            self.emb_dim = self.emb_proj_dim
        
        if self.apply_resampler:
            self.emb_resampler = LidarEmbResampler(self.emb_dim, **resampler_cfg)
    
    def forward(self, lidar_chosen_mask):
        if isinstance(self.lidar_emb, LearnableEmbedding):
            lidar_embeddings = self.lidar_emb(torch.arange(0, self.lidar_loc.shape[0]))
        else:
            lidar_embeddings = self.lidar_emb(self.lidar_loc)

        if self.with_ray_range:
            range_embeddings = self.ray_range_emb(self.lidar_range)
            lidar_embeddings = torch.cat([lidar_embeddings, range_embeddings], dim=-1)

        if self.emb_proj_dim != -1:
            lidar_embeddings = self.emb_proj(lidar_embeddings)

        # B 5 C
        lidar_embeddings = lidar_embeddings.unsqueeze(0).repeat(lidar_chosen_mask.shape[0], 1 ,1)
        lidar_embeddings[~lidar_chosen_mask] = 0

        if self.apply_resampler:
            lidar_embeddings = self.emb_resampler(lidar_embeddings)

        if not self.enabled:
            lidar_embeddings = lidar_embeddings * 0.0
            
        return lidar_embeddings