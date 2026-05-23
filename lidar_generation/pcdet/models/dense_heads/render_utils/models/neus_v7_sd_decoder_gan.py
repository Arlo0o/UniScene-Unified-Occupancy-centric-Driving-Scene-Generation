import torch
from mmcv.runner import force_fp32
from ..losses.gan_loss import VQLPIPSWithDiscriminator
from .neus_v6_sd_decoder import NeuSModelV6SDDecoder


class NeuSModelV7SDDecoderGan(NeuSModelV6SDDecoder):
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
        self.disc_start = kwargs.get("disc_start", 3e3)
        self.discriminator = VQLPIPSWithDiscriminator(disc_start=self.disc_start)

    @force_fp32(apply_to=("preds_dict", "targets"))
    def loss(self, preds_dict, lidar_targets, img_targets):
        depth_pred = preds_dict["depth"]
        depth_gt = lidar_targets["depth"]
        rgb_pred = preds_dict["rgb"]
        rgb_gt = img_targets["rgb"]
        rgb_pred = rgb_pred.to(rgb_gt.dtype)

        last_layer = self.rgb_upsampler.decoder.conv_out.weight
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
