import torch
from torch import nn
import torch.nn.functional as F
from ..renderers import RGBRenderer, DepthRenderer
from .. import scene_colliders
from .. import fields
from .. import ray_samplers
from abc import abstractmethod
from .base_render_model import BaseRenderModel
import numpy as np


class SurfaceModel(BaseRenderModel):
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
        self.rgb_renderer = RGBRenderer()
        self.depth_renderer = DepthRenderer()
        self.loss_cfg = loss_cfg

    @abstractmethod
    def sample_and_forward_field(self, ray_bundle, feature_volume):
        """_summary_

        Args:
            ray_bundle (RayBundle): _description_
            return_samples (bool, optional): _description_. Defaults to False.
        """

    def get_outputs(self, ray_bundle, feature_volume, **kwargs):
        samples_and_field_outputs = self.sample_and_forward_field(
            ray_bundle, feature_volume
        )

        # Shotscuts
        field_outputs = samples_and_field_outputs["field_outputs"]
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]

        rgb = self.rgb_renderer(rgb=field_outputs["rgb"], weights=weights)
        depth = self.depth_renderer(ray_samples=ray_samples, weights=weights)

        outputs = {
            "rgb": rgb,
            "depth": depth,
            "weights": weights,
            "sdf": field_outputs["sdf"],
            "gradients": field_outputs["gradients"],
            "z_vals": ray_samples.frustums.starts,
        }

        """ add for visualization"""
        outputs.update({"sampled_points": samples_and_field_outputs["sampled_points"]})
        if samples_and_field_outputs.get("init_sampled_points", None) is not None:
            outputs.update(
                {
                    "init_sampled_points": samples_and_field_outputs[
                        "init_sampled_points"
                    ],
                    "init_weights": samples_and_field_outputs["init_weights"],
                    "new_sampled_points": samples_and_field_outputs[
                        "new_sampled_points"
                    ],
                }
            )

        if self.training:
            if self.loss_cfg.get("sparse_points_sdf_supervised", False):
                sparse_points_sdf, _, _ = self.field.get_sdf(
                    kwargs["points"].unsqueeze(0), feature_volume
                )
                outputs["sparse_points_sdf"] = sparse_points_sdf.squeeze(0)

        return outputs

    def forward(self, ray_bundle, feature_volume, **kwargs):
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        ray_bundle = self.collider(ray_bundle)  # set near and far
        return self.get_outputs(ray_bundle, feature_volume, **kwargs)

    def loss(self, preds_dict, targets):
        depth_pred = preds_dict["depth"]
        depth_gt = targets["depth"]
        rgb_pred = preds_dict["rgb"]
        rgb_gt = targets["rgb"]

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

        return loss_dict
