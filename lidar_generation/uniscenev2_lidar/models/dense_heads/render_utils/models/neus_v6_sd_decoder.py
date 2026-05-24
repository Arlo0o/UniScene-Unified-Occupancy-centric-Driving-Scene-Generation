import torch

from .neus_v3 import NeuSModelV3
from diffusers.models.autoencoders.vae import Decoder
from mmcv.runner.base_module import BaseModule

class SD_Decoder(BaseModule):
    def __init__(self, latent_channels=16 + 32):
        super().__init__()

        act_fn="silu"
        block_out_channels=[32, 64, 128, 128]
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D']
        in_channels=3
        layers_per_block=2
        norm_num_groups=32
        out_channels=3
        sample_size=768
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D']

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

    def forward(self, x):
        """
        Returns:
            img_denorm: rgb format range in [0, 1].
        """
        img = self.decoder(x)
        img_denorm = img / 2 + 0.5
        return img_denorm
    
    def init_weights(self):
        super().init_weights()
        state_dict = torch.load("work_dirs/vae_decoder.pth", map_location="cpu")
        # drop conv_in
        key_list = ["conv_in.weight", "conv_in.bias"]
        for k in key_list:
            if k in state_dict.keys():
                state_dict.pop(k, None)
            else:
                print(f"{k} not in state_dict")
        self.decoder.load_state_dict(state_dict=state_dict, strict=False)


class NeuSModelV6SDDecoder(NeuSModelV3):
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
        self.rgb_upsampler = SD_Decoder(latent_channels=in_dim)
