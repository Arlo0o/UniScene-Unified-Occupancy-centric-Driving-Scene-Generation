import torch
from torch import nn
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
from .neus_v3 import conv_bn_relu, BasicBlock


@HEADS.register_module()
class NerfModel(BaseModule):
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
        super().__init__()
        if kwargs.get("fp16_enabled", False):
            self.fp16_enabled = True
        self.scale_factor = (
            1.0 / np.max(np.abs(pc_range)) if norm_scene else 1.0
        )  # select the max length to scale scenes
        field_type = field_cfg.pop("type")
        self.field = getattr(fields, field_type)(
            voxel_size=voxel_size,
            pc_range=pc_range,
            voxel_shape=voxel_shape,
            scale_factor=self.scale_factor,
            **field_cfg
        )
        collider_type = collider_cfg.pop("type")
        self.collider = getattr(scene_colliders, collider_type)(
            scene_box=pc_range, scale_factor=self.scale_factor, **collider_cfg
        )
        sampler_type = sampler_cfg.pop("type")
        self.sampler = getattr(ray_samplers, sampler_type)(**sampler_cfg)
        self.rgb_renderer = RGBRenderer(background_color=(0., ), test_clamp=False)
        self.depth_renderer = DepthRenderer()
        self.loss_cfg = loss_cfg
        
        rgb_upsample_factor = 2
        rgb_hidden_dim = 32
        in_dim = 32

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

    def sample_and_forward_field(self, ray_bundle, feature_volume):
        ray_samples = self.sampler(
            ray_bundle,
        )
        field_outputs = self.field(ray_samples, feature_volume, return_alphas=True) # ray_samples:  feature_volume: (32, 5, 128, 128)
        weights, _ = ray_samples.get_weights_and_transmittance(
            field_outputs["density"]
        )

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "sampled_points": ray_samples.frustums.get_start_positions(),
        }
        return samples_and_field_outputs

    def get_outputs(self, lidar_ray_bundle, cam_ray_bundle, feature_volume, **kwargs):
        cam_samples_and_field_outputs = self.sample_and_forward_field(
            cam_ray_bundle, feature_volume
        )
        if lidar_ray_bundle is not None:
            lidar_samples_and_field_outputs = self.sample_and_forward_field(
                lidar_ray_bundle, feature_volume
            )
        else:
            lidar_samples_and_field_outputs = cam_samples_and_field_outputs
        # import ipdb; ipdb.set_trace()
        # lidar
        lidar_ray_samples = lidar_samples_and_field_outputs["ray_samples"]
        lidar_weights = lidar_samples_and_field_outputs["weights"]
        lidar_field_outputs = lidar_samples_and_field_outputs["field_outputs"]
        depth = self.depth_renderer(ray_samples=lidar_ray_samples, weights=lidar_weights)
        # cam
        cam_field_outputs = cam_samples_and_field_outputs["field_outputs"]
        cam_weights = cam_samples_and_field_outputs["weights"]

        rgb_feat = self.rgb_renderer(rgb=cam_field_outputs["rgb"], weights=cam_weights)
        # （n, 48)
        feat_H, feat_W = 48, 100
        rgb_feat_channel = rgb_feat.shape[-1]
        rgb_feat = rgb_feat.reshape(6, feat_H, feat_W, rgb_feat_channel).permute(0, 3, 1, 2)


        rgb = self.rgb_upsampler(rgb_feat)

        outputs = {
            "rgb": rgb,
            "depth": depth,
        }

        return outputs

    @auto_fp16(apply_to=("feature_volume"))
    def forward(self, lidar_ray_bundle, cam_ray_bundle, feature_volume, **kwargs):
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        if lidar_ray_bundle is not None:
            lidar_ray_bundle = self.collider(lidar_ray_bundle)  # set near and far
        cam_ray_bundle = self.collider(cam_ray_bundle)
        return self.get_outputs(lidar_ray_bundle, cam_ray_bundle, feature_volume, **kwargs)

    @force_fp32(apply_to=("preds_dict", "targets"))
    def loss(self, preds_dict, lidar_targets, img_targets):
        depth_pred = preds_dict["depth"]
        depth_gt = lidar_targets["depth"]
        rgb_pred = preds_dict["rgb"]
        rgb_gt = img_targets["rgb"]

        loss_dict = {}
        loss_weights = self.loss_cfg.weights

        if loss_weights.get("rgb_loss", 0.0) > 0:

            rgb_loss = F.l1_loss(rgb_pred, rgb_gt)
            loss_dict["rgb_loss"] = rgb_loss * loss_weights.rgb_loss

        valid_gt_mask = depth_gt > 0.0
        if loss_weights.get("depth_loss", 0.0) > 0:
            depth_loss = torch.sum(
                valid_gt_mask * torch.abs(depth_gt - depth_pred)
            ) / torch.clamp(valid_gt_mask.sum(), min=1.0)
            loss_dict["depth_loss"] = depth_loss * loss_weights.depth_loss

        # weight loss

        return loss_dict