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
from ..fields.mlp import MLP
from .res_block import ResidualBlock, BasicBlock

    
def kl_div(p, q):
    return p * (torch.log(p+1e-6) - torch.log(q+1e-6))

def js_div(p, q):
    m = 0.5 * (p + q)
    return 0.5 * torch.sum(p * (torch.log(p+1e-6) - torch.log(m+1e-6)), dim=-1) + \
        0.5 * torch.sum(q * (torch.log(q+1e-6) - torch.log(m+1e-6)), dim=-1)

class NeuSModelV2LiDAROnly(SurfaceModel):
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
        self.pred_intensity = kwargs.get('pred_intensity', False)
        self.pred_raydrop = kwargs.get('pred_raydrop', False)
        if self.pred_intensity or self.pred_raydrop:
            #self.lidar_decoder = nn.Linear(32, 1)
            self.lidar_decoder = MLP(
                in_dim=field_cfg['sdf_decoder_cfg']['in_dim'],
                layer_width=32,
                out_dim=2 if (self.pred_intensity and self.pred_raydrop) else 1,
                num_layers=3,
                implementation='torch',
                out_activation=None,
                init_bias=-np.log((1 - 0.1) / 0.1)
            )


    def sample_and_forward_field(self, ray_bundle, feature_volume):
        sampler_out_dict = self.sampler(
            ray_bundle,
            occupancy_fn=self.field.get_occupancy,
            sdf_fn=partial(self.field.get_sdf, feature_volume=feature_volume),
            sdf_field=self.field, feature_volume=feature_volume
        )
        ray_samples = sampler_out_dict.pop("ray_samples")
        field_outputs = self.field(ray_samples, feature_volume, return_alphas=True, ray_bundle=ray_bundle) # ray_samples:  feature_volume: (32, 5, 128, 128)
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
    
    def get_outputs(self, lidar_ray_bundle, feature_volume, **kwargs):

        lidar_samples_and_field_outputs = self.sample_and_forward_field(
            lidar_ray_bundle, feature_volume
        )
        if self.pred_intensity or self.pred_raydrop:
            weighted_ray_features = (lidar_samples_and_field_outputs['field_outputs']['point_features'] * lidar_samples_and_field_outputs["weights"]).sum(-2)
            lidar_relative = self.lidar_decoder(weighted_ray_features)
        else:
            lidar_relative = None

        # lidar
        lidar_ray_samples = lidar_samples_and_field_outputs["ray_samples"]
        lidar_weights = lidar_samples_and_field_outputs["weights"]
        lidar_field_outputs = lidar_samples_and_field_outputs["field_outputs"]
        depth = self.depth_renderer(ray_samples=lidar_ray_samples, weights=lidar_weights)

        lidar_gradients = lidar_field_outputs["gradients"] # torch.Size([21058, 192, 3])
        gradients = lidar_gradients

        outputs = {
            "rgb": None,
            "depth": depth,
            # "weights": weights,
            # "sdf": lidar_field_outputs["sdf"], # 用于和depth gt计算loss
            # "gradients": gradients,
            # "z_vals": lidar_ray_samples.frustums.starts,
            'lidar_relative': lidar_relative,
            'ray_feats': lidar_field_outputs['ray_feats']
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

    def forward(self, lidar_ray_bundle, feature_volume, **kwargs):
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        lidar_ray_bundle = self.collider(lidar_ray_bundle)  # set near and far
        return self.get_outputs(lidar_ray_bundle, feature_volume, **kwargs)

    def g_loss(self, preds_dict, lidar_targets):
        depth_pred = preds_dict["depth"]
        depth_gt = lidar_targets["depth"]

        loss_dict = {}
        loss_weights = self.loss_cfg.weights

        valid_gt_mask = (depth_gt > 0.0)
        did_return = lidar_targets['did_return'].view(valid_gt_mask.shape)
        valid_gt_mask = valid_gt_mask & did_return
        if loss_weights.get("depth_loss", 0.0) > 0:
            depth_loss_type = self.loss_cfg.get('depth_loss_type', 'l1')
            if depth_loss_type == 'l1':
                depth_loss = torch.sum(
                    valid_gt_mask * torch.abs(depth_gt - depth_pred)
                ) / torch.clamp(valid_gt_mask.sum(), min=1.0)
            elif depth_loss_type == 'l2':
                depth_loss = torch.sum(
                    valid_gt_mask * (depth_gt - depth_pred)**2
                ) / torch.clamp(valid_gt_mask.sum(), min=1.0)
            elif depth_loss_type == 'smooth_l1':
                depth_loss = torch.sum(
                    valid_gt_mask * F.smooth_l1_loss(depth_pred, depth_gt, beta=0.2, reduction='none')
                ) / torch.clamp(valid_gt_mask.sum(), min=1.0)
            else:
                raise NotImplementedError
            loss_dict["depth_loss"] = depth_loss * loss_weights.depth_loss

        if loss_weights.get("smooth_loss", 0.0) > 0:
            # to range map -> calculate soomth loss on range map
            gt_points = lidar_targets['original_points']
            lidar_idx = gt_points[:, -1].int()
            ring_idx = gt_points[:, -2].int()
            lidar_loc = lidar_targets['ray_o']
            pred_depths = preds_dict['depth']
            pred_ray_feats = preds_dict['ray_feats']['ray_hist']
            smooth_loss_all = 0
            unique_lidar_ids = torch.unique(lidar_idx)
            for lidar_id in unique_lidar_ids:
                lidar_mask = (lidar_idx == lidar_id)
                thetas = torch.rad2deg(torch.atan2(gt_points[lidar_mask, 1]-lidar_loc[lidar_mask, 1], gt_points[lidar_mask, 0]-lidar_loc[lidar_mask, 0]))
                thetas += 180
                cell_size = 0.4
                phis = ring_idx[lidar_mask]
                phis = torch.round(phis).int()
                #thetas = torch.round(thetas).int()
                thetas = torch.round(thetas / cell_size).int()
                range_map = torch.zeros([int(phis.max()+1), int(thetas.max()+1)], dtype=pred_depths.dtype, device=pred_depths.device)
                feat_map = torch.zeros([int(phis.max()+1), int(thetas.max()+1), pred_ray_feats.shape[-1]], dtype=pred_ray_feats.dtype, device=pred_ray_feats.device)
                range_map[phis, thetas] = pred_depths[lidar_mask].squeeze()
                feat_map[phis, thetas] = pred_ray_feats[lidar_mask]
                valid_map = torch.zeros_like(range_map, dtype=torch.bool)
                valid_map[phis, thetas] = True
                smooth_loss = self.get_smooth_loss(range_map, feat_map, valid_map, sim_type=self.loss_cfg.get('smooth_loss_sim_type', 'js'))
                smooth_loss_all += smooth_loss
            loss_dict['smooth_loss'] = smooth_loss_all / len(unique_lidar_ids) * loss_weights.smooth_loss


        # free space loss and sdf loss
        # pred_sdf = preds_dict["sdf"][..., 0]
        # z_vals = preds_dict["z_vals"][..., 0]
        # truncation = self.loss_cfg.sensor_depth_truncation * self.scale_factor

        # front_mask = valid_gt_mask & (z_vals < (depth_gt - truncation))
        # back_mask = valid_gt_mask & (z_vals > (depth_gt + truncation))
        # sdf_mask = valid_gt_mask & (~front_mask) & (~back_mask)

        # if loss_weights.get("free_space_loss", 0.0) > 0:
        #     free_space_loss = (
        #         F.relu(truncation - pred_sdf) * front_mask
        #     ).sum() / torch.clamp(front_mask.sum(), min=1.0)
        #     loss_dict["free_space_loss"] = (
        #         free_space_loss * loss_weights.free_space_loss
        #     )

        # if loss_weights.get("sdf_loss", 0.0) > 0:
        #     sdf_loss = (
        #         torch.abs(z_vals + pred_sdf - depth_gt) * sdf_mask
        #     ).sum() / torch.clamp(sdf_mask.sum(), min=1.0)
        #     loss_dict["sdf_loss"] = sdf_loss * loss_weights.sdf_loss

        # if loss_weights.get("eikonal_loss", 0.0) > 0:
        #     gradients = preds_dict["gradients"]
        #     eikonal_loss = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
        #     loss_dict["eikonal_loss"] = eikonal_loss * loss_weights.eikonal_loss

        # if self.loss_cfg.get("sparse_points_sdf_supervised", False):
        #     sparse_points_sdf_loss = torch.mean(
        #         torch.abs(preds_dict["sparse_points_sdf"])
        #     )
        #     loss_dict["sparse_points_sdf_loss"] = (
        #         sparse_points_sdf_loss * loss_weights.sparse_points_sdf_loss
        #     )
        # if loss_weights.get("vgg_loss", 0.0) > 0:
        #     if not hasattr(self, "vgg_loss_module"):
        #         self.vgg_loss_module = VGGPerceptualLossPix2Pix()
        #         self.vgg_loss_module.to(rgb_gt.device)
        #     vgg_loss = self.vgg_loss_module(rgb_pred, rgb_gt)
        #     loss_dict["vgg_loss"] = vgg_loss * loss_weights.vgg_loss
        # if loss_weights.get("sds_loss", 0.0) > 0:
        #     if not hasattr(self, "sds_loss_module"):
        #         from threestudio.utils.config import load_config
        #         cfg = load_config("configs/sds/sds.yaml")
        #         self.sds_loss_module = StableDiffusionGuidance(
        #             cfg,
        #         )
        #         self.sds_loss_module
        #         self.sds_loss_module.to(rgb_gt.device)
        #     sds_loss = self.sds_loss_module(rgb_pred, rgb_gt)
        #     loss_dict["sds_loss"] = sds_loss * loss_weights.sds_loss

        if self.pred_intensity and self.pred_raydrop:
            intensity_pred, raydrop_pred = preds_dict['lidar_relative'].split(1, dim=-1)
        elif self.pred_intensity:
            intensity_pred = preds_dict['lidar_relative']
        elif self.pred_raydrop:
            raydrop_pred = preds_dict['lidar_relative']

        if self.pred_intensity:
            intensity_pred = intensity_pred.sigmoid()
            intensity_gt = lidar_targets['intensity']
            intensity_loss = torch.sum(
                #valid_gt_mask * torch.abs(intensity_gt - intensity_pred)
                valid_gt_mask * (intensity_gt - intensity_pred)**2
            ) / torch.clamp(valid_gt_mask.sum(), min=1.0)
            loss_dict["intensity_loss"] = intensity_loss * loss_weights.intensity_loss

        if self.pred_raydrop:
            valid_gt_mask = (depth_gt > 0.0)
            raydrop_loss = torch.nn.functional.binary_cross_entropy_with_logits(raydrop_pred[valid_gt_mask], (~did_return)[valid_gt_mask].to(raydrop_pred), reduction='sum') / torch.clamp(valid_gt_mask.sum(), min=1.0)
            loss_dict['raydrop_loss'] = raydrop_loss * loss_weights.raydrop_loss

        return loss_dict

    def loss(self, preds_dict, lidar_targets):
        return self.g_loss(preds_dict, lidar_targets)

    def get_smooth_loss(self, disp, feat_map, valid_map, sim_type='js'):
        grad_disp_x = torch.abs(disp[..., :, :-1] - disp[..., :, 1:])
        grad_disp_y = torch.abs(disp[..., :-1, :] - disp[..., 1:, :])

        sim_x, sim_y = self.get_sim(feat_map, sim_type=sim_type)

        grad_disp_x *= (torch.exp(-sim_x) * (valid_map[..., :, 1:] & valid_map[..., :, :-1]))
        grad_disp_y *= (torch.exp(-sim_y) * (valid_map[..., 1:, :] & valid_map[..., :-1, :]))

        return grad_disp_x.mean() # + grad_disp_y.mean()

    def get_sim(self, feat_map, sim_type):
        # feat_map [H, W, C]
        if sim_type == 'js':
            sim_x = js_div(feat_map[:, :-1], feat_map[:, 1:]) * 4
            sim_y = js_div(feat_map[:-1, :], feat_map[1:, :]) * 4
        elif sim_type == 'cosine':
            sim_x = (1 - F.cosine_similarity(feat_map[:, :-1], feat_map[:, 1:], dim=-1)) * 2.5
            sim_y = (1 - F.cosine_similarity(feat_map[:-1, :], feat_map[1:, :], dim=-1)) * 2.5
        else:
            raise NotImplementedError
        return sim_x, sim_y

