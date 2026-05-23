import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class SDPASelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):  # [B, L, C]
        B, L, C = x.shape
        x_norm = self.norm(x)

        qkv = self.qkv(x_norm)  # [B, L, 3C]
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, L, head_dim]

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # [B, heads, L, head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, L, C)  # [B, L, C]

        return self.out_proj(attn_out) + x  # residual

class TokenMLPMixer(nn.Module):
    def __init__(self, L, C, hidden_dim=32):
        super().__init__()
        self.L = L
        self.C = C
        self.mlp = nn.Sequential(
            nn.Linear(L, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, L)
        )

    def forward(self, x):  # x: [B, L, C]
        input_L = x.shape[1] 
        if input_L != self.L: # NeuSSampler forward: nothing helf of samples
            assert x.shape[1] == self.L//2
            x = x.repeat(1, 2, 1)
        x = x.transpose(1, 2)  # -> [B, C, L]
        x = self.mlp(x)
        x = x.transpose(1, 2)  # -> [B, L, C]
        if input_L != self.L:
            x = x[:, :input_L, :]
        return x