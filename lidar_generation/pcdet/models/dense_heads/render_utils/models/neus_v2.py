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
import numpy as np
from ..losses.perceptron_loss import VGGPerceptualLossPix2Pix
# from threestudio.models.guidance.stable_diffusion_guidance_imgonly import StableDiffusionGuidance
from .res_block import ResidualBlock, BasicBlock

class NeuSModelV2(SurfaceModel):
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
        rgb_upsample_factor = 8
        rgb_hidden_dim = 32
        in_dim = 16 + 32

        self.rgb_upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, rgb_hidden_dim, kernel_size=1, padding=0),
            torch.nn.ReLU(inplace=True),
            BasicBlock(rgb_hidden_dim, rgb_hidden_dim, kernel_size=7, padding=3, use_bn=True),
            BasicBlock(rgb_hidden_dim, rgb_hidden_dim, kernel_size=7, padding=3, use_bn=True),
            torch.nn.ConvTranspose2d(
                rgb_hidden_dim,
                rgb_hidden_dim,
                kernel_size=rgb_upsample_factor,
                stride=rgb_upsample_factor,
            ),
            BasicBlock(rgb_hidden_dim, rgb_hidden_dim, kernel_size=7, padding=3, use_bn=True),
            BasicBlock(rgb_hidden_dim, rgb_hidden_dim, kernel_size=7, padding=3, use_bn=True),
            torch.nn.Conv2d(rgb_hidden_dim, 3, kernel_size=1, padding=0),
            torch.nn.Sigmoid(),
        ) # (0, 1)
        self.rgb_renderer = RGBRenderer(background_color=(0.,), test_clamp=True)

    def sample_and_forward_field(self, ray_bundle, feature_volume):
        sampler_out_dict = self.sampler(
            ray_bundle,
            occupancy_fn=self.field.get_occupancy,
            sdf_fn=partial(self.field.get_sdf, feature_volume=feature_volume),
        )
        ray_samples = sampler_out_dict.pop("ray_samples")
        field_outputs = self.field(ray_samples, feature_volume, return_alphas=True) # ray_samples:  feature_volume: (32, 5, 128, 128)
        weights, _ = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs["alphas"]
        )

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "sampled_points": ray_samples.frustums.get_start_positions(),
            **sampler_out_dict,
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

        # lidar
        lidar_ray_samples = lidar_samples_and_field_outputs["ray_samples"]
        lidar_weights = lidar_samples_and_field_outputs["weights"]
        lidar_field_outputs = lidar_samples_and_field_outputs["field_outputs"]
        depth = self.depth_renderer(ray_samples=lidar_ray_samples, weights=lidar_weights)
        # cam
        cam_field_outputs = cam_samples_and_field_outputs["field_outputs"]
        cam_weights = cam_samples_and_field_outputs["weights"]

        rgb_feat = self.rgb_renderer(rgb=cam_field_outputs["rgb"], weights=cam_weights) # （n, 48)
        feat_H, feat_W = 48, 100
        rgb_feat_channel = rgb_feat.shape[-1]
        rgb_feat = rgb_feat.reshape(6, feat_H, feat_W, rgb_feat_channel).permute(0, 3, 1, 2)

        
        rgb = self.rgb_upsampler(rgb_feat)

        cam_gradients = cam_field_outputs["gradients"] # torch.Size([28800, 192, 3])
        if lidar_ray_bundle is not None:
            lidar_gradients = lidar_field_outputs["gradients"] # torch.Size([21058, 192, 3])
            gradients = torch.cat([lidar_gradients, cam_gradients], dim=0)
        else:
            gradients = cam_gradients
        outputs = {
            "rgb": rgb,
            "depth": depth,
            # "weights": weights,
            "sdf": lidar_field_outputs["sdf"], # 用于和depth gt计算loss
            "gradients": gradients,
            "z_vals": lidar_ray_samples.frustums.starts,
        }

        """ add for visualization"""
        # outputs.update({"sampled_points": samples_and_field_outputs["sampled_points"]})
        # if samples_and_field_outputs.get("init_sampled_points", None) is not None:
        #     outputs.update(
        #         {
        #             "init_sampled_points": samples_and_field_outputs[
        #                 "init_sampled_points"
        #             ],
        #             "init_weights": samples_and_field_outputs["init_weights"],
        #             "new_sampled_points": samples_and_field_outputs[
        #                 "new_sampled_points"
        #             ],
        #         }
        #     )

        # if self.training:
        #     if self.loss_cfg.get("sparse_points_sdf_supervised", False):
        #         sparse_points_sdf, _, _ = self.field.get_sdf(
        #             kwargs["points"].unsqueeze(0), feature_volume
        #         )
        #         outputs["sparse_points_sdf"] = sparse_points_sdf.squeeze(0)

        return outputs

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

    def g_loss(self, preds_dict, lidar_targets, img_targets):
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

        # free space loss and sdf loss
        pred_sdf = preds_dict["sdf"][..., 0]
        z_vals = preds_dict["z_vals"][..., 0]
        truncation = self.loss_cfg.sensor_depth_truncation * self.scale_factor

        front_mask = valid_gt_mask & (z_vals < (depth_gt - truncation))
        back_mask = valid_gt_mask & (z_vals > (depth_gt + truncation))
        sdf_mask = valid_gt_mask & (~front_mask) & (~back_mask)

        if loss_weights.get("free_space_loss", 0.0) > 0:
            free_space_loss = (
                F.relu(truncation - pred_sdf) * front_mask
            ).sum() / torch.clamp(front_mask.sum(), min=1.0)
            loss_dict["free_space_loss"] = (
                free_space_loss * loss_weights.free_space_loss
            )

        if loss_weights.get("sdf_loss", 0.0) > 0:
            sdf_loss = (
                torch.abs(z_vals + pred_sdf - depth_gt) * sdf_mask
            ).sum() / torch.clamp(sdf_mask.sum(), min=1.0)
            loss_dict["sdf_loss"] = sdf_loss * loss_weights.sdf_loss

        if loss_weights.get("eikonal_loss", 0.0) > 0:
            gradients = preds_dict["gradients"]
            eikonal_loss = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
            loss_dict["eikonal_loss"] = eikonal_loss * loss_weights.eikonal_loss

        if self.loss_cfg.get("sparse_points_sdf_supervised", False):
            sparse_points_sdf_loss = torch.mean(
                torch.abs(preds_dict["sparse_points_sdf"])
            )
            loss_dict["sparse_points_sdf_loss"] = (
                sparse_points_sdf_loss * loss_weights.sparse_points_sdf_loss
            )
        if loss_weights.get("vgg_loss", 0.0) > 0:
            if not hasattr(self, "vgg_loss_module"):
                self.vgg_loss_module = VGGPerceptualLossPix2Pix()
                self.vgg_loss_module.to(rgb_gt.device)
            vgg_loss = self.vgg_loss_module(rgb_pred, rgb_gt)
            loss_dict["vgg_loss"] = vgg_loss * loss_weights.vgg_loss
        if loss_weights.get("sds_loss", 0.0) > 0:
            if not hasattr(self, "sds_loss_module"):
                from threestudio.utils.config import load_config
                cfg = load_config("configs/sds/sds.yaml")
                self.sds_loss_module = StableDiffusionGuidance(
                    cfg,
                )
                self.sds_loss_module
                self.sds_loss_module.to(rgb_gt.device)
            sds_loss = self.sds_loss_module(rgb_pred, rgb_gt)
            loss_dict["sds_loss"] = sds_loss * loss_weights.sds_loss
        return loss_dict

    def loss(self, preds_dict, lidar_targets, img_targets):
        return self.g_loss(preds_dict, lidar_targets, img_targets)