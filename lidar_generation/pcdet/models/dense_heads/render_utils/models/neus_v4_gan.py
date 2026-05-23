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
from .neus_v3 import NeuSModelV3
from ..losses.gan_loss import VQLPIPSWithDiscriminator
from ..losses.perceptron_loss import VGGPerceptualLossPix2Pix
from .neus_v3 import conv_bn_relu, BasicBlock


class NeuSModelV4(NeuSModelV3):
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
        if kwargs.get("fp16_enabled", False):
            self.fp16_enabled = True
        self.disc_start = kwargs.get("disc_start", 3e3)
        self.discriminator = VQLPIPSWithDiscriminator(disc_start=self.disc_start)

        rgb_upsample_factor = 2
        rgb_hidden_dim = 32
        in_dim = 16 + 32
        self.use_sigmoid = kwargs.get("use_sigmoid", False)
        if self.use_sigmoid:
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
        else:
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
            ) # (0, 1)

    @force_fp32(apply_to=("preds_dict", "targets"))
    def loss(self, preds_dict, lidar_targets, img_targets):
        depth_pred = preds_dict["depth"]
        depth_gt = lidar_targets["depth"]
        rgb_pred = preds_dict["rgb"]
        rgb_gt = img_targets["rgb"]
        rgb_pred = rgb_pred.to(rgb_gt.dtype)

        last_layer = self.rgb_upsampler[-1].weight if not self.use_sigmoid else self.rgb_upsampler[-2].weight
        if self.cur_iter < self.disc_start or self.cur_iter % 2 == 0:
            loss_dict = self.g_loss(preds_dict, lidar_targets, img_targets)
            loss_dict = self.discriminator(
                loss_dict,
                inputs=rgb_gt,
                reconstructions=rgb_pred,
                optimizer_idx=0,
                global_step=self.cur_iter,
                last_layer=last_layer,
            )
        else:
            loss_dict = {}
            loss_dict = self.discriminator(
                loss_dict,
                inputs=rgb_gt,
                reconstructions=rgb_pred,
                optimizer_idx=1,
                global_step=self.cur_iter,
                last_layer=last_layer)
        return loss_dict