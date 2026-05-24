import torch
from torch import nn
from .base_surface_model import SurfaceModel
from functools import partial

import torch.nn.functional as F
from ..renderers import RGBRenderer, DepthRenderer
from .. import scene_colliders
from .. import fields
from .. import ray_samplers
from abc import abstractmethod
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import HEADS
from mmcv.runner.base_module import BaseModule
import numpy as np
from .neus_v2 import NeuSModelV2, BasicBlock


def conv_bn_relu(in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
    bias = not use_bn
    return nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
        torch.nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
        torch.nn.ReLU(inplace=True),
    )

class NeuSModelV3(NeuSModelV2):
    def __init__(
        self,
        pc_range,
        voxel_size,
        voxel_shape,
        field_cfg,
        collider_cfg,
        sampler_cfg,
        loss_cfg,
        norm_scene,
        **kwargs
    ):
        super().__init__(
            pc_range=pc_range,
            voxel_size=voxel_size,
            voxel_shape=voxel_shape,
            field_cfg=field_cfg,
            collider_cfg=collider_cfg,
            sampler_cfg=sampler_cfg,
            loss_cfg=loss_cfg,
            norm_scene=norm_scene,
            **kwargs
        )
        self.anneal_end = 50000
        rgb_upsample_factor = 2
        rgb_hidden_dim = 32
        in_dim = 16 + 32

        self.rgb_upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, rgb_hidden_dim, kernel_size=1, padding=0),
            torch.nn.ReLU(inplace=True),
            BasicBlock(rgb_hidden_dim, rgb_hidden_dim, kernel_size=7, padding=3, use_bn=True),
            conv_bn_relu(rgb_hidden_dim, rgb_hidden_dim * 2**2, kernel_size=3, padding=1, use_bn=True),
            nn.PixelShuffle(rgb_upsample_factor),
            BasicBlock(rgb_hidden_dim, rgb_hidden_dim, kernel_size=7, padding=3, use_bn=True),
            conv_bn_relu(rgb_hidden_dim, rgb_hidden_dim * 2**2, kernel_size=3, padding=1, use_bn=True),
            nn.PixelShuffle(rgb_upsample_factor),
            BasicBlock(rgb_hidden_dim, rgb_hidden_dim, kernel_size=7, padding=3, use_bn=True),
            conv_bn_relu(rgb_hidden_dim, rgb_hidden_dim * 2**2, kernel_size=3, padding=1, use_bn=True),
            nn.PixelShuffle(rgb_upsample_factor),
            BasicBlock(rgb_hidden_dim, rgb_hidden_dim, kernel_size=7, padding=3, use_bn=True),
            torch.nn.Conv2d(rgb_hidden_dim, 3, kernel_size=1, padding=0),
            torch.nn.Sigmoid(),
        ) # (0, 1)