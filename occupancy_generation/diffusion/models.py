# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

try:
    from diffusion.bev_cod import BEV_condition_net,BEV_concat_net,BEV_concat_net_s
    from diffusion.embedder import get_embedder

except ModuleNotFoundError:
    # 相对导入，适用于作为模块运行时
    # 绝对导入，适用于单独运行文件时
    from bev_cod import BEV_condition_net,BEV_concat_net,BEV_concat_net_s
    from embedder import get_embedder

# from bev_cod import BEV_condition_net,BEV_concat_net
# from embedder import get_embedder

############
import logging

import torch.nn.functional as F
from einops import rearrange, repeat





XYZ_MIN = [-200, -300, -20]
XYZ_RANGE = [350, 650, 80]


def normalizer(mode, data):
    if mode == 'cxyz' or mode == 'all-xyz':
        # data in format of (N, 4, 3):
        mins = torch.as_tensor(
            XYZ_MIN, dtype=data.dtype, device=data.device)[None, None]
        divider = torch.as_tensor(
            XYZ_RANGE, dtype=data.dtype, device=data.device)[None, None]
        data = (data - mins) / divider
    elif mode == 'owhr':
        raise NotImplementedError(f"wait for implementation on {mode}")
    else:
        raise NotImplementedError(f"not support {mode}")
    return data


class ContinuousBBoxWithTextEmbedding(nn.Module):
    """
    Use continuous bbox corrdicate and text embedding with CLIP encoder
    """

    def __init__(
        self,
        n_classes=18,
        class_token_dim=768,
        trainable_class_token=False,
        embedder_num_freq=4,
        proj_dims=[768, 512, 512, 2500],
        mode='cxyz',
        minmax_normalize=True,
        use_text_encoder_init=False,
        **kwargs,
    ):
        """
        Args:
            mode (str, optional): cxyz -> all points; all-xyz -> all points;
                owhr -> center, l, w, h, z-orientation.
        """
        super().__init__()

        self.mode = mode
        if self.mode == 'cxyz':
            input_dims = 3
            output_num = 4  # 4 points
        elif self.mode == 'all-xyz':
            input_dims = 3
            output_num = 8  # 8 points
        elif self.mode == 'owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {mode}")
        self.minmax_normalize = minmax_normalize
        self.use_text_encoder_init = use_text_encoder_init

        self.fourier_embedder = get_embedder(input_dims, embedder_num_freq)
        logging.info(
            f"[ContinuousBBoxWithTextEmbedding] bbox embedder has "
            f"{self.fourier_embedder.out_dim} dims.")

        self.bbox_proj = nn.Linear(
            self.fourier_embedder.out_dim * output_num, proj_dims[0])
        self.second_linear = nn.Sequential(
            nn.Linear(proj_dims[0] + class_token_dim, proj_dims[1]),
            nn.SiLU(),
            nn.Linear(proj_dims[1], proj_dims[2]),
            nn.SiLU(),
            nn.Linear(proj_dims[2], proj_dims[3]),
        )

        # for class token
        self._class_tokens_set_or_warned = not self.use_text_encoder_init
        if trainable_class_token:
            # parameter is trainable, buffer is not
            class_tokens = torch.randn(n_classes, class_token_dim)
            self.register_parameter("_class_tokens", nn.Parameter(class_tokens))
        else:
            class_tokens = torch.randn(n_classes, class_token_dim)
            self.register_buffer("_class_tokens", class_tokens)
            if not self.use_text_encoder_init:
                logging.warn(
                    "[ContinuousBBoxWithTextEmbedding] Your class_tokens is not"
                    " trainable but you set `use_text_encoder_init` to False. "
                    "Please check your config!")

        # null embedding
        self.null_class_feature = torch.nn.Parameter(
            torch.zeros([class_token_dim]))
        self.null_pos_feature = torch.nn.Parameter(
            torch.zeros([self.fourier_embedder.out_dim * output_num]))

    @property
    def class_tokens(self):
        if not self._class_tokens_set_or_warned:
            logging.warn(
                "[ContinuousBBoxWithTextEmbedding] Your class_tokens is not "
                "trainable and used without initialization. Please check your "
                "training code!")
            self._class_tokens_set_or_warned = True
        return self._class_tokens

    def prepare(self, cfg, **kwargs):
        if self.use_text_encoder_init:
            self.set_category_token(
                kwargs['tokenizer'], kwargs['text_encoder'],
                cfg.dataset.object_classes)
        else:
            logging.info("[ContinuousBBoxWithTextEmbedding] Your class_tokens "
                         "initilzed with random.")

    @torch.no_grad()
    def set_category_token(self, tokenizer, text_encoder, class_names):
        logging.info("[ContinuousBBoxWithTextEmbedding] Initialzing your "
                     "class_tokens with text_encoder")
        self._class_tokens_set_or_warned = True
        device = self.class_tokens.device
        for idx, name in enumerate(class_names):
            inputs = tokenizer(
                [name], padding='do_not_pad', return_tensors='pt')
            inputs = inputs.input_ids.to(device)
            # there are two outputs: last_hidden_state and pooler_output
            # we use the pooled version.
            hidden_state = text_encoder(inputs).pooler_output[0]  # 768
            self.class_tokens[idx].copy_(hidden_state)

    def add_n_uncond_tokens(self, hidden_states, token_num):
        B = hidden_states.shape[0]
        uncond_token = self.forward_feature(
            self.null_pos_feature[None], self.null_class_feature[None])
        uncond_token = repeat(uncond_token, 'c -> b n c', b=B, n=token_num)
        hidden_states = torch.cat([hidden_states, uncond_token], dim=1)
        return hidden_states

    def forward_feature(self, pos_emb, cls_emb):
        emb = self.bbox_proj(pos_emb)
        emb = F.silu(emb)

        # combine
        emb = torch.cat([emb, cls_emb], dim=-1)
        emb = self.second_linear(emb)
        return emb

    def forward(self, bboxes: torch.Tensor, classes: torch.LongTensor,
                masks=None, **kwargs):
        """Please do filter before input is needed.

        Args:
            bboxes (torch.Tensor): Expect (B, N, 4, 3) for cxyz mode.
            classes (torch.LongTensor): (B, N)

        Return:
            size B x N x emb_dim=768
        """
        print(bboxes.shape)
        print(classes.shape)
        (B, N) = classes.shape
        bboxes = rearrange(bboxes, 'b n ... -> (b n) ...')
        print(bboxes.shape)
        if masks is None:
            masks = torch.ones(len(bboxes))
        else:
            masks = masks.flatten()
        masks = masks.unsqueeze(-1).type_as(self.null_pos_feature)
        print(342352345)
        # box
        if self.minmax_normalize:
            bboxes = normalizer(self.mode, bboxes)
        pos_emb = self.fourier_embedder(bboxes)
        pos_emb = pos_emb.reshape(
            pos_emb.shape[0], -1).type_as(self.null_pos_feature)
        pos_emb = pos_emb * masks + self.null_pos_feature[None] * (1 - masks)
        print(7667876342352345)
        # class
        cls_emb = torch.stack([self.class_tokens[i] for i in classes.flatten()])
        cls_emb = cls_emb * masks + self.null_class_feature[None] * (1 - masks)

        # combine
        emb = self.forward_feature(pos_emb, cls_emb)
        emb = rearrange(emb, '(b n) ... -> b n ...', n=N)
        return emb


###########




def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class BEVDropout_layer(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, dropout_prob,use_3d=False,dsr=4): #dsr: down sample rate
        super().__init__()
        # use_cfg_embedding = dropout_prob > 0
        # self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        # self.num_classes = num_classes
        if use_3d:
            self.maxpool=nn.MaxPool3d(kernel_size=(1,dsr,dsr), stride=(1,dsr,dsr), padding=(0,0,0))
        else:
            self.maxpool=nn.MaxPool2d(kernel_size=dsr, stride=dsr, padding=0)
        self.dropout_prob = dropout_prob
        if dropout_prob>0:
            print("Use BEV Dropout!")

    def token_drop(self, BEV_layout):
        """
        Drops labels to enable classifier-free guidance.
        """
        if torch.rand(1) < self.dropout_prob:
            BEV_null = -torch.ones_like(BEV_layout,device=BEV_layout.device)
            BEV_layout = BEV_null
        # drop_mask = torch.rand_like(BEV_layout,device=BEV_layout.device) < self.dropout_prob
        # BEV_layout[drop_mask] =-1
        return BEV_layout

    def forward(self, BEV_layout):
        use_dropout = self.dropout_prob > 0

        BEV_layout = self.maxpool(BEV_layout)
        if self.training and use_dropout:
            BEV_layout = self.token_drop(BEV_layout)
        # embeddings = self.embedding_table(labels)
        return BEV_layout


class BEVDropout_layer_adaptive(nn.Module):
    """
    基于卷积的自适应BEV处理层，替代AdaptiveAvgPool3d以保持更好的空间信息
    """
    def __init__(self, dropout_prob, target_size=(334, 160)):
        super().__init__()
        self.target_h, self.target_w = target_size
        self.dropout_prob = dropout_prob
        
        # 使用卷积网络替代AdaptiveAvgPool3d
        self.conv_processor = nn.Sequential(
            # 第一层：保持通道数不变，进行特征提取
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 第二层：特征增强
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 第三层：特征融合
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 输出层：回到原始通道数
            nn.Conv2d(16, 1, kernel_size=1),
        )
        
        # 最终尺寸调整（仅在必要时使用）
        self.final_resize = nn.AdaptiveAvgPool2d(target_size)
        
        if dropout_prob > 0:
            print("Use BEV Dropout with Conv-based processing!")

    def token_drop(self, BEV_layout):
        """
        Drops labels to enable classifier-free guidance.
        """
        if torch.rand(1) < self.dropout_prob:
            BEV_null = -torch.ones_like(BEV_layout, device=BEV_layout.device)
            BEV_layout = BEV_null
        return BEV_layout

    def forward(self, BEV_layout):
        # BEV_layout shape: [N, T, C, H, W]
        batch_size, time_steps, channels, height, width = BEV_layout.shape
        
        # 重塑为 [N*T*C, 1, H, W] 以便逐个处理每个通道
        x = BEV_layout.view(batch_size * time_steps * channels, 1, height, width)
        
        # 通过卷积处理器
        x = self.conv_processor(x)
        
        # 调整到目标尺寸
        x = self.final_resize(x)
        
        # 恢复原始形状: [N*T*C, 1, H, W] -> [N, T, C, H, W]
        x = x.view(batch_size, time_steps, channels, self.target_h, self.target_w)
        
        # Apply dropout if training
        if self.training and self.dropout_prob > 0:
            x = self.token_drop(x)
            
        return x


class BEVConv_layer_adaptive(nn.Module):
    """基于卷积的BEV处理网络，替代AdaptiveAvgPool3d"""
    def __init__(self, input_channels=1, target_size=(334, 160), dropout_prob=0.0):
        super().__init__()
        self.target_size = target_size
        self.dropout_prob = dropout_prob
        
        # 使用卷积网络来处理BEV数据，保持更多空间信息
        self.conv_layers = nn.Sequential(
            # 第一层：保持空间信息的同时进行特征提取
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 第二层：轻微下采样，保持大部分空间信息
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第三层：特征融合
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 最后一层：输出目标通道数
            nn.Conv2d(32, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        # 自适应池化作为最后的尺寸调整
        self.final_pool = nn.AdaptiveAvgPool2d(target_size)
        
    def token_drop(self, BEV_layout):
        """
        Drops BEV data to enable classifier-free guidance.
        """
        if torch.rand(1) < self.dropout_prob:
            BEV_null = -torch.ones_like(BEV_layout, device=BEV_layout.device)
            BEV_layout = BEV_null
        return BEV_layout
        
    def forward(self, BEV_layout):
        # BEV_layout shape: [N, T, C, H, W]
        batch_size, time_steps, channels, height, width = BEV_layout.shape
        
        # 重塑为 [N*T, C, H, W] 以便处理
        x = BEV_layout.view(batch_size * time_steps, channels, height, width)
        
        # 通过卷积层处理
        x = self.conv_layers(x)
        
        # 最终池化到目标尺寸
        x = self.final_pool(x)
        
        # 恢复时间维度: [N*T, C, H, W] -> [N, T, C, H, W]
        x = x.view(batch_size, time_steps, -1, self.target_size[0], self.target_size[1])
        
        # Apply dropout if training
        if self.training and self.dropout_prob > 0:
            x = self.token_drop(x)
            
        return x


def visualize_x_y_correspondence(x, y, save_path="x_y_correspondence.png", batch_idx=0, time_idx=0):
    """
    可视化x和y的对应关系
    
    Args:
        x: [N, T, C, H, W] - latent特征
        y: [N, T, Cb, Hb, Wb] - BEV数据
        save_path: 保存路径
        batch_idx: 要可视化的batch索引
        time_idx: 要可视化的时间索引
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 提取要可视化的数据
    x_vis = x[batch_idx, time_idx].cpu().numpy()  # [C, H, W]
    y_vis = y[batch_idx, time_idx].cpu().numpy()  # [Cb, Hb, Wb]
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 可视化x的前3个通道
    for i in range(min(3, x_vis.shape[0])):
        im = axes[0, i].imshow(x_vis[i], cmap='viridis')
        axes[0, i].set_title(f'X Channel {i} ({x_vis[i].shape})')
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i])
    
    # 可视化y的前3个通道（如果有的话）
    if y_vis.shape[0] >= 3:
        for i in range(3):
            im = axes[1, i].imshow(y_vis[i], cmap='plasma')
            axes[1, i].set_title(f'Y Channel {i} ({y_vis[i].shape})')
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i])
    else:
        # 如果y只有1个通道，显示3次
        for i in range(3):
            im = axes[1, i].imshow(y_vis[0], cmap='plasma')
            axes[1, i].set_title(f'Y Channel 0 ({y_vis[0].shape})')
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"==> Saved X-Y correspondence visualization: {save_path}")
    print(f"    X shape: {x_vis.shape}, Y shape: {y_vis.shape}")


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()#lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,padding=1)
    def forward(self, x):
        
        x = self.conv1(x)  
        x = self.conv2(x) 
        return x


class BEVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=18, out_channels=64, kernel_size=3, stride=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=False,padding=1)
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.maxpool(x) 
        x = self.conv2(x) 
        x = self.maxpool(x) 
        x = self.conv3(x)
        return x

# class MLP_Trag(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, output_size)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class MLP_meta(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,dropout_prob):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        print(f"MLP_meta: {input_size}, {hidden_size}, {output_size}")

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        if self.training and self.dropout_prob > 0:
            if torch.rand(1) < self.dropout_prob:
                x_null = torch.zeros_like(x,device=x.device)
                x = x_null
        return x




class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=512,  #  1152
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        # self.mlp_trag = MLP_Trag(input_size=64, hidden_size=128, output_size=256)
        self.BEVnet= BEVNet()
        # self.ConditionNet=BEV_condition_net()
        self.my_net = MyNet()
        #self.bbox=ContinuousBBoxWithTextEmbedding()
        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = 2500   #self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=True)  #  num_patches=128 , hidden_size=1875

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, 64)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed_from_grid(256, torch.arange(1, 2501))    # self.x_embedder.num_patches 
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        

        
        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y ):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        x1 = x 
        # print(x1.shape)
        # x1 = torch.squeeze(x1, dim=1)
        x1 = self.my_net(x1)
        # print(x1.shape)
        y = self.BEVnet(y)
        # y= self.ConditionNet(y)
        # print(y.shape)
        y =y.reshape(-1,256)
        x1 =x1.reshape(-1,256,2500)
        x1 =x1.permute(0, 2, 1) # 16 2500 128
        x2 = self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        x =x1+x2
        t = self.t_embedder(t)                   # (N, D)

        c = t  + y                                # (N, D)

        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)


        x =x.permute(0, 2, 1) #8 128 2500
        
        # x =x.reshape(-1,256,4,25,25)
        x =x.reshape(-1,256,50,50)
        

        # print("x2",x.shape)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale  ):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y )
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class DiT_uncon(nn.Module):
    """
    Diffusion model with a Transformer backbone., multi_frame, also can use single frame
    """
    def __init__(
        self,
        input_size=50,
        patch_size=2,
        in_channels=128,
        hidden_size=256,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        bev_dropout_prob=0,
        # num_classes=1000,
        bev_in_ch = 18,
        bev_out_ch = 4,
        meta_num = 1,
        learn_sigma=True,
        use_label=False,
        use_meta=False,
        use_bev_concat=False,
        direct_concat=False,
        use_x_ref_concat=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels + bev_out_ch  if use_bev_concat else in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_label = use_label
        self.use_meta = use_meta
        self.use_bev_concat = use_bev_concat
        self.use_x_ref_concat = use_x_ref_concat
        self.x_embedder = PatchEmbed(input_size, patch_size, self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        if use_label:
            self.y_embedder = BEV_condition_net()
        if use_meta:
            print(f"Use Meta embedding! meta_num:{meta_num}")
            # self.meta_embedder = nn.Linear(meta_num,hidden_size)
            self.meta_embedder = MLP_meta(meta_num,64,hidden_size,bev_dropout_prob)
        if self.use_bev_concat:
            if direct_concat==False:
                # self.bev_concat = BEV_concat_net(BEV_in_ch=bev_in_ch,BEV_out_ch=bev_out_ch)
                self.bev_concat = BEV_concat_net_s(BEV_in_ch=bev_in_ch,BEV_out_ch=bev_out_ch)
            else:
                print("Use MaxPool2d!")
                # self.bev_concat = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
                self.bev_concat = BEVDropout_layer(bev_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # print(num_patches)
        # Will use fixed sin-cos embedding:
        self.pos_embed_m = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed_m.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed_m.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # if self.use_label:
        #     nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, x_ref, y, meta=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # x=x+y/20
        # x.shape : bs, 4, 50, 50
        if self.use_bev_concat: # 一般是不用 bev 的
            x = torch.cat([x, self.bev_concat(y)], dim=1)
        if self.use_x_ref_concat:
            x = torch.cat([x, x_ref], dim=1)
        x = self.x_embedder(x) + self.pos_embed_m  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)

        # if self.use_label:
        #     y = self.y_embedder(y)
        #     c = t + y                                # (N, D)
        # else:
        #     c=t
        if self.use_meta:
            pts_num=meta#['pts_num']
            meta_embd=self.meta_embedder(pts_num)
            c = t + meta_embd
        else:
            c = t
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, meta=None ,cfg_scale=1.0):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y ,meta)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :4], model_out[:, 4:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class DiT_2Frame(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=50,
        patch_size=2,
        in_channels=4,
        hidden_size=256,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        bev_dropout_prob=0,
        # num_classes=1000,
        bev_in_ch = 1,
        bev_out_ch = 1,
        meta_num = 1,
        learn_sigma=True,
        use_label=False,
        use_meta=False,
        use_bev_concat=True,
        direct_concat=False,
        Tframe = 6
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels + bev_out_ch  if use_bev_concat else in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.hidden_size=hidden_size
        self.num_heads = num_heads
        self.use_label = use_label
        self.use_meta = use_meta
        self.depth = depth
        self.Tframe = Tframe
        self.use_bev_concat = use_bev_concat
        self.x_embedder = PatchEmbed(input_size, patch_size, self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        if use_label:
            self.y_embedder = BEV_condition_net()
        if use_meta:
            print(f"Use Meta embedding! meta_num:{meta_num}")
            # self.meta_embedder = nn.Linear(meta_num,hidden_size)
            # self.meta_embedder = MLP_meta(meta_num,64,hidden_size,bev_dropout_prob)
            self.meta_embedder = MLP_meta(meta_num,int(hidden_size/2),hidden_size,bev_dropout_prob)
        if self.use_bev_concat:
            if direct_concat==False:
                # self.bev_concat = BEV_concat_net(BEV_in_ch=bev_in_ch,BEV_out_ch=bev_out_ch)
                self.bev_concat = BEV_concat_net_s(BEV_in_ch=bev_in_ch,BEV_out_ch=bev_out_ch)
            else:
                print("Use MaxPool3d!")
                # self.bev_concat = nn.MaxPool3d(kernel_size=(1,4,4), stride=(1,4,4), padding=(0,0,0))
                self.bev_concat = BEVDropout_layer(bev_dropout_prob,use_3d=True)
        num_patches = self.x_embedder.num_patches
        # print(num_patches)
        # Will use fixed sin-cos embedding:
        self.pos_embed_m = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, Tframe, hidden_size), requires_grad=False)

        # self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed_m.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed_m.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_pos_embed(self.hidden_size,self.Tframe)
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # if self.use_label:
        #     nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, meta=None):
        """
        x: [N,C,T,H,W]
        t: [N,]
        y: [N,T,Cb,Hb,Wb] (BEV)
        meta: [N,m]
        """
        x = x.permute(0,2,1,3,4) 
        print(x.shape,y.shape)
        if self.use_bev_concat:
            x = torch.cat([x, self.bev_concat(y)], dim=2)
        print(x.shape)
        B,T,C,H,W = x.shape
        # S = 625
        S = self.x_embedder.num_patches
        x = x.reshape(B*T,C,H,W)
        x = self.x_embedder(x)  #B*T,S,D
       
        x = x + self.pos_embed_m 
        x = rearrange(x, "(B T) S D -> (B S) T D", T=T, S=S)
        x = x + self.temp_embed
        x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)

        t = self.t_embedder(t)
        if self.use_meta:
            pts_num=meta
            meta_embd=self.meta_embedder(pts_num)
            c = t + meta_embd           #B D
        else:
            c = t
        
        c_s = c.repeat(1,1,T)
        c_s = c_s.reshape(-1,self.hidden_size)
        c_t = c.repeat(1,1,S)
        c_t = c_t.reshape(-1,self.hidden_size)
        # c_s = c.repeat((T,1))
        # c_t = c.repeat((S,1))
        for i,block in enumerate(self.blocks):
            if i %2 ==0:                          #(B*T,S,D) spatial 
                if x.shape[0] == B*S:
                    x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)
                x = block(x, c_s)                   
            else:                                  #(B*S,T,D) temporal
                if x.shape[0] == B*T:
                    x = rearrange(x, "(B T) S D -> (B S) T D", T=T, S=S)
                x = block(x, c_t)    

        x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)
        x = self.final_layer(x, c_s)          
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        # print(x.shape)
        x = x.reshape(B,T,self.out_channels,H,W)
        x = x.permute(0,2,1,3,4) 
        return x

    def forward_with_cfg(self, x, t, y, meta=None ,cfg_scale=1.0):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y ,meta)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class DiT_2Frame_lixiang_rectangular(nn.Module):
    """
    Diffusion model with a Transformer backbone, supports rectangular/non-square latent space.
    """
    def __init__(
        self,
        input_size=(334, 160),  # 支持非方形输入 (height, width)
        patch_size=2,
        in_channels=4,
        hidden_size=256,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        bev_dropout_prob=0,
        # num_classes=1000,
        bev_in_ch = 1,
        bev_out_ch = 16,  # 修改为16，匹配BEVConv_layer_adaptive的输出
        meta_num = 1,
        learn_sigma=True,
        use_label=False,
        use_meta=False,
        use_bev_concat=True,
        direct_concat=False,
        Tframe = 6
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels + bev_out_ch  if use_bev_concat else in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.hidden_size=hidden_size
        self.num_heads = num_heads
        self.use_label = use_label
        self.use_meta = use_meta
        self.depth = depth
        self.Tframe = Tframe
        self.use_bev_concat = use_bev_concat
        
        # 处理输入尺寸，支持非方形
        if isinstance(input_size, int):
            self.input_h = self.input_w = input_size
        else:
            self.input_h, self.input_w = input_size
        
        self.x_embedder = PatchEmbed((self.input_h, self.input_w), patch_size, self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        if use_label:
            self.y_embedder = BEV_condition_net()
        if use_meta:
            print(f"Use Meta embedding! meta_num:{meta_num}")
            # self.meta_embedder = nn.Linear(meta_num,hidden_size)
            # self.meta_embedder = MLP_meta(meta_num,64,hidden_size,bev_dropout_prob)
            self.meta_embedder = MLP_meta(meta_num,int(hidden_size/2),hidden_size,bev_dropout_prob)
        if self.use_bev_concat:
            if direct_concat==False:
                # 使用改进的卷积版本替代原来的网络
                print("使用改进的BEVConv_layer_adaptive!")
                self.bev_concat = BEVConv_layer_adaptive(
                    input_channels=bev_in_ch, 
                    target_size=(self.input_h, self.input_w), 
                    dropout_prob=bev_dropout_prob
                )
            else:
                print("Use MaxPool3d!")
                # self.bev_concat = nn.MaxPool3d(kernel_size=(1,4,4), stride=(1,4,4), padding=(0,0,0))
                self.bev_concat = BEVDropout_layer(bev_dropout_prob,use_3d=True)
        num_patches = self.x_embedder.num_patches
        # print(num_patches)
        # Will use fixed sin-cos embedding:
        self.pos_embed_m = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, Tframe, hidden_size), requires_grad=False)

        # self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # 支持非方形输入的位置编码
        h_patches = self.input_h // self.patch_size
        w_patches = self.input_w // self.patch_size
        pos_embed = get_2d_sincos_pos_embed_rectangular(self.pos_embed_m.shape[-1], h_patches, w_patches)
        self.pos_embed_m.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_pos_embed(self.hidden_size,self.Tframe)
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # if self.use_label:
        #     nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        # 支持非方形输入
        h = self.input_h // self.patch_size
        w = self.input_w // self.patch_size
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        
        # 如果重建的尺寸与原始输入尺寸不匹配，进行裁剪或填充
        if imgs.shape[2] != self.input_h or imgs.shape[3] != self.input_w:
            # 使用插值来调整到原始尺寸
            import torch.nn.functional as F
            imgs = F.interpolate(imgs, size=(self.input_h, self.input_w), mode='bilinear', align_corners=False)
        
        return imgs

    def forward(self, x, t, y, meta=None, save_visualization=False, vis_save_path="x_y_correspondence.png"):
        """
        x: [N,C,T,H,W]
        t: [N,]
        y: [N,T,Cb,Hb,Wb] (BEV)
        meta: [N,m]
        save_visualization: 是否保存可视化图像
        vis_save_path: 可视化图像保存路径
        """
        x = x.permute(0,2,1,3,4)  # [N,T,C,H,W]
        print(f"输入 x 形状: {x.shape}, y 形状: {y.shape}")
        
        if self.use_bev_concat:
            # 处理BEV数据
            processed_y = self.bev_concat(y)
            print(f"处理后的 BEV 形状: {processed_y.shape}")
            
            # 可视化 x 和 y 的对应关系
            if save_visualization:
                print("==> 生成 X-Y 对应关系可视化...")
                visualize_x_y_correspondence(
                    x, processed_y, 
                    save_path=vis_save_path, 
                    batch_idx=0, time_idx=0
                )
            
            # 拼接处理后的BEV数据
            x = torch.cat([x, processed_y], dim=2)
            print(f"拼接后的 x 形状: {x.shape}")
        
        B,T,C,H,W = x.shape
        # 动态计算patch数量，支持非方形输入
        S = (H // self.patch_size) * (W // self.patch_size)
        x = x.reshape(B*T,C,H,W)
        x = self.x_embedder(x)  #B*T,S,D
       
        x = x + self.pos_embed_m 
        x = rearrange(x, "(B T) S D -> (B S) T D", T=T, S=S)
        x = x + self.temp_embed
        x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)

        t = self.t_embedder(t)
        if self.use_meta:
            pts_num=meta
            meta_embd=self.meta_embedder(pts_num)
            c = t + meta_embd           #B D
        else:
            c = t
        
        c_s = c.repeat(1,1,T)
        c_s = c_s.reshape(-1,self.hidden_size)
        c_t = c.repeat(1,1,S)
        c_t = c_t.reshape(-1,self.hidden_size)
        # c_s = c.repeat((T,1))
        # c_t = c.repeat((S,1))
        for i,block in enumerate(self.blocks):
            if i %2 ==0:                          #(B*T,S,D) spatial 
                if x.shape[0] == B*S:
                    x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)
                x = block(x, c_s)                   
            else:                                  #(B*S,T,D) temporal
                if x.shape[0] == B*T:
                    x = rearrange(x, "(B T) S D -> (B S) T D", T=T, S=S)
                x = block(x, c_t)    

        x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)
        x = self.final_layer(x, c_s)          
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        print(x.shape)
        x = x.reshape(B,T,self.out_channels,H,W)
        x = x.permute(0,2,1,3,4) 
        return x

    def forward_with_cfg(self, x, t, y, meta=None ,cfg_scale=1.0):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y ,meta)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)



class DiT_multiframe(nn.Module):
    """
    Diffusion model with a Transformer backbone. TFrame occ. 
    """
    def __init__(
        self,
        input_size=50,
        patch_size=2,
        in_channels=4,
        hidden_size=256,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        bev_dropout_prob=0,
        # num_classes=1000,
        bev_in_ch = 1,
        bev_out_ch = 1,
        meta_num = 1,
        learn_sigma=True,
        use_label=False,
        use_meta=False,
        use_bev_concat=True,
        direct_concat=False,
        Tframe = 6,
        temp_attn = True,
        dsr = 4
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels + bev_out_ch  if use_bev_concat else in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.hidden_size=hidden_size
        self.num_heads = num_heads
        self.use_label = use_label
        self.use_meta = use_meta
        self.depth = depth
        self.Tframe = Tframe
        self.temp_attn = temp_attn
        self.use_bev_concat = use_bev_concat
        self.x_embedder = PatchEmbed(input_size, patch_size, self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if use_label:
            self.y_embedder = BEV_condition_net()
        if use_meta:
            print(f"Use Meta embedding! meta_num:{meta_num}")
            # self.meta_embedder = nn.Linear(meta_num,hidden_size)
            # self.meta_embedder = MLP_meta(meta_num,64,hidden_size,bev_dropout_prob)
            self.meta_embedder = MLP_meta(meta_num,int(hidden_size/2),hidden_size,bev_dropout_prob)
        if self.use_bev_concat:
            if direct_concat==False:
                # self.bev_concat = BEV_concat_net(BEV_in_ch=bev_in_ch,BEV_out_ch=bev_out_ch)
                self.bev_concat = BEV_concat_net_s(BEV_in_ch=bev_in_ch,BEV_out_ch=bev_out_ch)
            else:
                print("Use MaxPool3d!")
                # self.bev_concat = nn.MaxPool3d(kernel_size=(1,4,4), stride=(1,4,4), padding=(0,0,0))
                self.bev_concat = BEVDropout_layer(bev_dropout_prob,use_3d=True,dsr=dsr)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed_m = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, Tframe, hidden_size), requires_grad=False)

        # self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed_m.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed_m.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_pos_embed(self.hidden_size,self.Tframe)
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # if self.use_label:
        #     nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, x_ref=None, meta=None):
        """
        x: [N,C,T,H,W] - target frames to predict
        t: [N,]
        y: [N,T,Cb,Hb,Wb] (BEV) - can be None for unconditional
        x_ref: [N,C,T_ref,H,W] - reference frames (optional, not used if noise injection)
        meta: [N,m]
        """        
        x = x.permute(0,2,1,3,4) 
        # print(f"x shape: {x.shape}, y shape: {y.shape if y is not None else None}, type(y): {type(y)}")
            
        if self.use_bev_concat and y is not None:
            print(f"use bev concat: {self.use_bev_concat}, y shape: {y.shape}")
            x = torch.cat([x, self.bev_concat(y)], dim=2)
        
        # print(f"Final x shape: {x.shape}")
        B,T,C,H,W = x.shape
        S = 625
        x = x.reshape(B*T,C,H,W)
        x = self.x_embedder(x)  #B*T,S,D
       
        x = x + self.pos_embed_m 

        if self.temp_attn==True:
            x = rearrange(x, "(B T) S D -> (B S) T D", T=T, S=S)
            # Dynamically create temporal embedding if T doesn't match self.Tframe
            if T != self.Tframe:
                temp_embed = get_1d_sincos_pos_embed(self.hidden_size, T)
                temp_embed = torch.from_numpy(temp_embed).float().unsqueeze(0).to(x.device)
            else:
                temp_embed = self.temp_embed
            x = x + temp_embed
            x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)

        # x = x.reshape(B,-1,self.hidden_size) # B T*patch_num D
        # print(x.shape)
        t = self.t_embedder(t)                   # (B, D)
        # print("=> use meta: ", self.use_meta)
        if self.use_meta:
            pts_num=meta
            meta_embd=self.meta_embedder(pts_num)
            c = t + meta_embd           #B D
        else:
            c = t
        
        c_s = c.repeat(1,1,T)
        c_s = c_s.reshape(-1,self.hidden_size)
        c_t = c.repeat(1,1,S)
        c_t = c_t.reshape(-1,self.hidden_size)
        
        for i,block in enumerate(self.blocks):
            if self.temp_attn==True:
                if i %2 ==0:                          #(B*T,S,D) spatial 
                    if x.shape[0] == B*S:
                        x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)
                    x = block(x, c_s)                   
                else:                                  #(B*S,T,D) temporal
                    if x.shape[0] == B*T:
                        x = rearrange(x, "(B T) S D -> (B S) T D", T=T, S=S)
                    x = block(x, c_t)
            else:
                x = block(x, c_s) # only spatial 
                
        if x.shape[0] == B*S:
            x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)
        x = self.final_layer(x, c_s)          
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        # print(x.shape)
        x = x.reshape(B,T,self.out_channels,H,W)
        x = x.permute(0,2,1,3,4) 
        return x

    def forward_with_cfg(self, x, t, y, x_ref=None, meta=None, cfg_scale=1.0):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
            
        model_out = self.forward(combined, t, y, None, meta)  # x_ref=None since using noise injection
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

class DiT_WorldModel(nn.Module):
    """
    Diffusion model with a Transformer backbone. forcasting, ref is previous occ
    """
    def __init__(
        self,
        input_size=50,
        patch_size=2,
        in_channels=4,
        hidden_size=256,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        bev_dropout_prob=0,
        # num_classes=1000,
        bev_in_ch = 1,
        bev_out_ch = 1,
        meta_num = 1,
        learn_sigma=True,
        use_label=False,
        use_meta=False,
        use_bev_concat=True,
        direct_concat=False,
        T_pred = 6,
        T_condition = 1
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels + bev_out_ch  if use_bev_concat else in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.hidden_size=hidden_size
        self.num_heads = num_heads
        self.use_label = use_label
        self.use_meta = use_meta
        self.depth = depth
        self.T_pred =T_pred
        self.T_condition = T_condition
        self.Tframe = T_pred + T_condition
        self.use_bev_concat = use_bev_concat
        self.x_embedder = PatchEmbed(input_size, patch_size, self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if use_label:
            self.y_embedder = BEV_condition_net()
        if use_meta:
            print(f"Use Meta embedding! meta_num:{meta_num}")
            self.meta_embedder = MLP_meta(meta_num,int(hidden_size/2),hidden_size,bev_dropout_prob)
        if self.use_bev_concat:
            if direct_concat==False:
                # self.bev_concat = BEV_concat_net(BEV_in_ch=bev_in_ch,BEV_out_ch=bev_out_ch)
                self.bev_concat = BEV_concat_net_s(BEV_in_ch=bev_in_ch,BEV_out_ch=bev_out_ch)
            else:
                print("Use MaxPool3d!")
                # self.bev_concat = nn.MaxPool3d(kernel_size=(1,4,4), stride=(1,4,4), padding=(0,0,0))
                self.bev_concat = BEVDropout_layer(bev_dropout_prob,use_3d=True)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed_m = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, self.Tframe, hidden_size), requires_grad=False)

        # self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed_m.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed_m.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_pos_embed(self.hidden_size,self.Tframe)
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # if self.use_label:
        #     nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, x_ref, y, meta=None):
        """
        x: [N,C,Tp,H,W]
        t: [N,]
        x_ref: [N,C,Tc,H,W]
        y: [N,Tp,Cb,Hb,Wb] (BEV)
        meta: [N,m]
        """
        x = x.permute(0,2,1,3,4) 
        x_ref = x_ref.permute(0,2,1,3,4)
        
        # print(x.shape,y.shape)
        if self.use_bev_concat:
            print(f"==> use bev concat")
            x = torch.cat([x, self.bev_concat(y)], dim=2)
            y_ref = torch.zeros_like(x_ref[:,:,0:1],device=x.device)
            # print(f"x_ref shape: {x_ref.shape}, x shape: {x.shape}, condate shape{self.bev_concat(y).shape}")
            
            x_ref = torch.cat([x_ref, y_ref], dim=2) # B Tc C H W
            # print(f"x_ref shape: {x_ref.shape}, x shape: {x.shape}")
        
        x = torch.cat([x_ref,x], dim=1) # B (Tp + Tc) C H W --->
        # print(x.shape)
        B,T,C,H,W = x.shape
        S = 625
        x = x.reshape(B*T,C,H,W)
        x = self.x_embedder(x)  #B*T,S,D
       
        x = x + self.pos_embed_m 
        x = rearrange(x, "(B T) S D -> (B S) T D", T=T, S=S)
        print(f"==> x.shape={x.shape}==self.temp_embed.shape{self.temp_embed.shape}=={self.pos_embed_m.shape}")
        x = x + self.temp_embed
        x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)

        # x = x.reshape(B,-1,self.hidden_size) # B T*patch_num D
        # print(x.shape)
        t = self.t_embedder(t)                   # (B, D)
        # print(t.shape)
        if self.use_meta:
            pts_num=meta
            meta_embd=self.meta_embedder(pts_num)
            c = t + meta_embd           #B D
        else:
            c = t
        
        c_s = c.repeat(1,1,T)
        c_s = c_s.reshape(-1,self.hidden_size)
        c_t = c.repeat(1,1,S)
        c_t = c_t.reshape(-1,self.hidden_size)
        
        for i,block in enumerate(self.blocks):
            if i %2 ==0:                          #(B*T,S,D) spatial 
                if x.shape[0] == B*S:
                    x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)
                x = block(x, c_s)                   
            else:                                  #(B*S,T,D) temporal
                if x.shape[0] == B*T:
                    x = rearrange(x, "(B T) S D -> (B S) T D", T=T, S=S)
                x = block(x, c_t)    

        x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)
        x = self.final_layer(x, c_s)          
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        # print(x.shape)
        x = x.reshape(B,T,self.out_channels,H,W)
        x = x.permute(0,2,1,3,4) 

        x = x[:,:,self.T_condition:]
        return x

    def forward_with_cfg(self, x, t, x_ref, y, meta=None ,cfg_scale=1.0):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, x_ref, y ,meta)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)



class DiT_Occsora(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=50,
        patch_size=2,
        in_channels=4,
        hidden_size=256,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        bev_dropout_prob=0,
        # num_classes=1000,
        bev_in_ch = 1,
        bev_out_ch = 1,
        meta_num = 1,
        learn_sigma=True,
        use_label=False,
        use_meta=False,
        use_bev_concat=True,
        direct_concat=False,
        Tframe = 6
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels + bev_out_ch  if use_bev_concat else in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.hidden_size=hidden_size
        self.num_heads = num_heads
        self.use_label = use_label
        self.use_meta = use_meta
        self.depth = depth
        self.Tframe = Tframe
        self.use_bev_concat = use_bev_concat
        self.x_embedder = PatchEmbed(input_size, patch_size, self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if use_label:
            self.y_embedder = BEV_condition_net()
        if use_meta:
            print(f"Use Meta embedding! meta_num:{meta_num}")
            self.meta_embedder = MLP_meta(meta_num,int(hidden_size/2),hidden_size,bev_dropout_prob)
        if self.use_bev_concat:
            if direct_concat==False:
                # self.bev_concat = BEV_concat_net(BEV_in_ch=bev_in_ch,BEV_out_ch=bev_out_ch)
                self.bev_concat = BEV_concat_net_s(BEV_in_ch=bev_in_ch,BEV_out_ch=bev_out_ch)
            else:
                print("Use MaxPool3d!")
                # self.bev_concat = nn.MaxPool3d(kernel_size=(1,4,4), stride=(1,4,4), padding=(0,0,0))
                self.bev_concat = BEVDropout_layer(bev_dropout_prob,use_3d=True)

        # 根据输入尺寸与patch大小动态确定token数量（每帧的patch数 × 帧数）
        self.spatial_num_patches = self.x_embedder.num_patches
        total_tokens = Tframe * self.spatial_num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, hidden_size), requires_grad=True)
        # Will use fixed sin-cos embedding:
        # self.pos_embed_m = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding (length=Tframe*spatial_num_patches):
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, torch.arange(1, self.Tframe * self.spatial_num_patches + 1))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, meta=None):
        """
        x: [N,C,T,H,W]
        t: [N,]
        y: [N,T,Cb,Hb,Wb] (BEV)
        meta: [N,m]
        """
        x = x.permute(0,2,1,3,4) 
        # print(x.shape,y.shape)
        if self.use_bev_concat:
            x = torch.cat([x, self.bev_concat(y)], dim=2)
        # print(x.shape)
        B,T,C,H,W = x.shape
        S = self.x_embedder.num_patches
        x = x.reshape(B*T,C,H,W)
        x = self.x_embedder(x)  #B*T,S,D

        x = rearrange(x, "(B T) S D -> B (T S) D", T=T, S=S)
        x = x + self.pos_embed

        t = self.t_embedder(t)                   # (B, D)

        if self.use_meta:
            pts_num=meta
            meta_embd=self.meta_embedder(pts_num)
            c = t + meta_embd           #B D
        else:
            c = t
       
        for i,block in enumerate(self.blocks):
            x= block(x, c)
           
        x = self.final_layer(x, c)     
        x = rearrange(x, "B (T S) D -> (B T) S D", T=T, S=S)     
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        x = x.reshape(B,T,self.out_channels,H,W)
        x = x.permute(0,2,1,3,4) 
        return x

    def forward_with_cfg(self, x, t, y, meta=None,cfg_scale=1.0  ):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y,meta )
        
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    # grid = grid.reshape([2, 1, grid_size,grid_size])
    grid = grid.reshape([2, grid_size*grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_rectangular(embed_dim, grid_h, grid_w, cls_token=False, extra_tokens=0):
    """
    grid_h: int of the grid height
    grid_w: int of the grid width  
    return:
    pos_embed: [grid_h*grid_w, embed_dim] or [1+grid_h*grid_w, embed_dim] (w/ or w/o cls_token)
    """
    grid_h_coords = np.arange(grid_h, dtype=np.float32)
    grid_w_coords = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w_coords, grid_h_coords)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, grid_h*grid_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(length,dtype=np.float32)#[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
 
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):  #hidden_size=embed_dim 1152 pos=256 
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    # print("omega",omega.shape) #288  
    # print("pos",pos.shape)  

    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    return emb #  256 576


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=256, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=256, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=256, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)



DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("DiT_2Frame_lixiang_rectangular 模型测试")
    print("=" * 60)
    
    # 模型配置
    model_config = {
        "input_size": (334, 160),
        "in_channels":  16,
        "depth": 12,
        "hidden_size": 512,
        "bev_out_ch": 1,
        "use_bev_concat": True,
        "use_meta": True,
        "bev_dropout_prob": 0.1,
        "meta_num": 4,
        "direct_concat": True,
        "Tframe": 5
    }
    
    # 创建模型
    print("1. 创建模型...")
    model = DiT_2Frame_lixiang_rectangular(**model_config)
    
    # 临时修复BEV尺寸匹配问题
    if hasattr(model, 'bev_concat') and model.use_bev_concat:
        model.bev_concat = BEVDropout_layer_adaptive(model_config['bev_dropout_prob'], 
                                                   target_size=(model_config['input_size'][0], model_config['input_size'][1]))
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ 总参数数量: {total_params:,}")
    print(f"   ✓ 可训练参数数量: {trainable_params:,}")
    
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   ✓ 使用设备: {device}")
    model = model.to(device)
    
    # 测试不同批次大小
    print("\n2. 测试不同批次大小...")
    batch_sizes = [1, 2]
    
    for bz in batch_sizes:
        print(f"   测试批次大小: {bz}")
        
        # 准备测试数据
        input_tensor = torch.randn((bz, 16, 5, 334, 160), device=device)
        t = torch.randint(0, 100, (bz,), device=device)
        x_ref = torch.randn((bz, 16, 1, 334, 160), device=device)
        y = torch.randn((bz, 5, 1, 1000, 440), device=device)
        meta = torch.randn((bz, 4), device=device)
        
        # 前向传播测试
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            # 第一次调用时生成可视化
            save_vis = (bz == batch_sizes[0])  # 只在第一个批次大小时保存可视化
            vis_path = f"x_y_correspondence_batch_{bz}.png"
            output = model(input_tensor, t, y, meta, 
                         save_visualization=save_vis, 
                         vis_save_path=vis_path)
            end_time = time.time()
        
        print(f"     ✓ 输出形状: {output.shape}")
        print(f"     ✓ 推理时间: {end_time - start_time:.4f} 秒")
        
        # 检查输出是否合理
        if torch.isnan(output).any():
            print("     ✗ 警告: 输出包含NaN值!")
        elif torch.isinf(output).any():
            print("     ✗ 警告: 输出包含无穷值!")
        else:
            print(f"     ✓ 输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 梯度流测试
    print("\n3. 梯度流测试...")
    model.train()
    bz = 1
    input_tensor = torch.randn((bz, 16, 5, 334, 160), device=device, requires_grad=True)
    t = torch.randint(0, 100, (bz,), device=device)
    x_ref = torch.randn((bz, 16, 1, 334, 160), device=device)
    y = torch.randn((bz, 5, 1, 1000, 480), device=device)
    meta = torch.randn((bz, 4), device=device)
    
    output = model(input_tensor, t, y, meta, 
                 save_visualization=True, 
                 vis_save_path="x_y_correspondence_gradient_test.png")
    loss = output.mean()
    loss.backward()
    
    # 检查梯度
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_norms.append(grad_norm)
    
    print(f"   ✓ 损失值: {loss.item():.6f}")
    print(f"   ✓ 有梯度的参数数量: {len(grad_norms)}")
    if grad_norms:
        print(f"   ✓ 梯度范数范围: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
        print(f"   ✓ 平均梯度范数: {sum(grad_norms)/len(grad_norms):.6f}")
    
    # GPU内存使用情况
    if torch.cuda.is_available():
        print("\n4. GPU内存使用情况...")
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"   ✓ 已分配内存: {memory_allocated:.2f} MB")
        print(f"   ✓ 已保留内存: {memory_reserved:.2f} MB")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试完成!")
    print("=" * 60)