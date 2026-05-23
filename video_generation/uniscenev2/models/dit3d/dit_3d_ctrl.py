import os
import numpy as np
from torch import nn
import torch
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
from diffusers.utils import is_torch_version, deprecate
from torch.nn import functional as F
from safetensors.torch import load_file as load_safetensors
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.models.embeddings import PixArtAlphaTextProjection
from uniscenev2.registry import MODELS
from IPython import embed
from transformers import PretrainedConfig, PreTrainedModel
from uniscenev2.models.layers.blocks import (
    SizeEmbedder,
    TimestepEmbedder,
    CaptionEmbedder,
    approx_gelu,
)
from packaging import version as pver
from uniscenev2.models.stdit.embedder import CamEmbedder, RayMapControlTempEmbedding, RayMapControlEmbedding, ContinuousBBoxWithTextTempEmbedding
from uniscenev2.acceleration.parallel_states import get_sequence_parallel_state, nccl_info
from uniscenev2.models.dit3d.modules import BasicTransformerBlock, PatchEmbed2D
from uniscenev2.acceleration.checkpoint import auto_grad_checkpoint


class DiT3DCTRLConfig(PretrainedConfig):
    model_type = "DIT3DCTRL"

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        use_additional_conditions: Optional[bool] = None,
        attention_mode: str = 'xformers', 
        downsampler: str = None, 
        use_rope: bool = False,
        use_stable_fp32: bool = False,
        # inpaint
        vae_scale_factor_t: int = 4,
        class_dropout_prob: float = 0.1,
        model_max_length: int = 300,
        num_camera = 6,
        with_depth = False,
        with_seg = False,
        use_attn_ctrl=False,
        **kwargs,
    ):
        self.use_attn_ctrl = use_attn_ctrl
        self.num_camera = num_camera
        self.class_dropout_prob = class_dropout_prob
        self.num_attention_heads=num_attention_heads
        self.attention_head_dim=attention_head_dim
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.num_layers=num_layers
        self.dropout=dropout
        self.norm_num_groups=norm_num_groups
        self.cross_attention_dim=cross_attention_dim
        self.attention_bias=attention_bias
        self.num_vector_embeds=num_vector_embeds
        self.patch_size=patch_size
        self.activation_fn=activation_fn
        self.num_embeds_ada_norm=num_embeds_ada_norm
        self.use_linear_projection=use_linear_projection
        self.only_cross_attention=only_cross_attention
        self.double_self_attention=double_self_attention
        self.upcast_attention=upcast_attention
        self.norm_type=norm_type
        self.norm_elementwise_affine=norm_elementwise_affine
        self.norm_eps=norm_eps
        self.attention_type=attention_type
        self.caption_channels=caption_channels
        self.use_additional_conditions=use_additional_conditions
        self.attention_mode=attention_mode
        self.downsampler=downsampler
        self.use_rope=use_rope
        self.use_stable_fp32=use_stable_fp32
        self.vae_scale_factor_t=vae_scale_factor_t
        self.model_max_length = model_max_length
        self.with_depth = with_depth
        self.with_seg = with_seg
        self.raymap_embedder_downsample_rate = 4.5
        super().__init__(**kwargs)





class DIT3DCTRL(PreTrainedModel):
    config_class = DiT3DCTRLConfig

    def __init__(
        self, config: DiT3DCTRLConfig
    ):
        super().__init__(config)
        # Validate inputs.
        if config.patch_size is not None:
            if config.norm_type not in ["ada_norm", "ada_norm_zero", "ada_norm_single"]:
                raise NotImplementedError(
                    f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{config.norm_type}'."
                )
            elif config.norm_type in ["ada_norm", "ada_norm_zero"] and config.num_embeds_ada_norm is None:
                raise ValueError(
                    f"When using a `patch_size` and this `norm_type` ({config.norm_type}), `num_embeds_ada_norm` cannot be None."
                )

        # Set some common variables used across the board.
        self.use_attn_ctrl = config.use_attn_ctrl
        self.raymap_embedder_downsample_rate = config.raymap_embedder_downsample_rate
        self.use_rope = config.use_rope
        self.downsampler = config.downsampler
        self.gradient_checkpointing = False
        use_additional_conditions = False
        self.use_additional_conditions = use_additional_conditions
        self.model_max_length = config.model_max_length
        self.class_dropout_prob = config.class_dropout_prob
        self.num_attention_heads=config.num_attention_heads
        self.attention_head_dim=config.attention_head_dim
        self.inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = self.inner_dim
        self.in_channels=config.in_channels
        self.out_channels = config.in_channels if config.out_channels is None else config.out_channels
        self.num_layers=config.num_layers
        self.dropout=config.dropout
        self.norm_num_groups=config.norm_num_groups
        self.cross_attention_dim=config.cross_attention_dim
        self.attention_bias=config.attention_bias
        self.num_vector_embeds=config.num_vector_embeds
        self.patch_size=config.patch_size
        self.activation_fn=config.activation_fn
        self.num_embeds_ada_norm=config.num_embeds_ada_norm
        self.use_linear_projection=config.use_linear_projection
        self.only_cross_attention=config.only_cross_attention
        self.double_self_attention=config.double_self_attention
        self.upcast_attention=config.upcast_attention
        self.norm_type=config.norm_type
        self.norm_elementwise_affine=config.norm_elementwise_affine
        self.norm_eps=config.norm_eps
        self.attention_type=config.attention_type
        self.caption_channels=config.caption_channels
        self.attention_mode=config.attention_mode
        self.use_stable_fp32=config.use_stable_fp32
        self.vae_scale_factor_t=config.vae_scale_factor_t
        self.num_camera = config.num_camera
        self.with_depth = config.with_depth
        self.with_seg = config.with_seg

        if config.norm_type == "layer_norm" and config.num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`. Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"
        # 2. Initialize the right blocks.
        # Initialize the output blocks and other projection blocks when necessary.
        self._init_patched_inputs(norm_type=config.norm_type)


    def _init_patched_inputs(self, norm_type):
        self.register_buffer("base_token", torch.randn(self.hidden_size))
        self.pos_embed = PatchEmbed2D(
            patch_size=self.patch_size[1],
            in_channels=self.in_channels,
            embed_dim=self.inner_dim,
            use_abs_pos=not self.use_rope, 
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    dropout=self.dropout,
                    cross_attention_dim=self.cross_attention_dim,
                    activation_fn=self.activation_fn,
                    num_embeds_ada_norm=self.num_embeds_ada_norm,
                    attention_bias=self.attention_bias,
                    only_cross_attention=self.only_cross_attention,
                    double_self_attention=self.double_self_attention,
                    upcast_attention=self.upcast_attention,
                    norm_type=self.norm_type,
                    norm_elementwise_affine=self.norm_elementwise_affine,
                    norm_eps=self.norm_eps,
                    attention_type=self.attention_type,
                    attention_mode=self.attention_mode, 
                    downsampler=self.downsampler, 
                    use_rope=self.use_rope, 
                    n_cam=self.num_camera,
                    with_seg=self.with_seg,
                    with_depth=self.with_depth,
                    use_attn_ctrl=self.use_attn_ctrl
                )
                for _ in range(self.config.num_layers)
            ]
        )
        if self.norm_type != "ada_norm_single":
            self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
            self.proj_out_2 = nn.Linear(
                self.inner_dim, self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * self.out_channels
            )
        elif self.norm_type == "ada_norm_single":
            self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
            self.proj_out = nn.Linear(
                self.inner_dim, self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * self.out_channels
            )


        # PixArt-Alpha blocks.
        self.adaln_single = None
        if self.norm_type == "ada_norm_single":
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(
                self.inner_dim, use_additional_conditions=self.use_additional_conditions
            )

        self.y_embedder = PixArtAlphaTextProjection(
            in_features=self.caption_channels, hidden_size=self.inner_dim
        )

        self.y_embedder.y_embedding = torch.zeros(self.model_max_length, self.caption_channels)

        # init seg_map 2D encoder
        self.segmap_embedder = None
        if self.with_seg:
            self.segmap_embedder = RayMapControlEmbedding(
                conditioning_embedding_channels=self.hidden_size // 2,
                conditioning_size = 1,
                block_out_channels=[16, 32, 96, 256],
            )

            self.segmap_embedder_temp = RayMapControlTempEmbedding(
                self.hidden_size, self.raymap_embedder_downsample_rate)
            self.segmap_patchifier = PatchEmbed2D(
                # self.patch_size, self.hidden_size, self.hidden_size
                patch_size=self.patch_size[1],
                in_channels=self.inner_dim,
                embed_dim=self.inner_dim,
                use_abs_pos=not self.use_rope, 
            )

        # init depth_map 2D encoder
        self.depthmap_embedder = None
        if self.with_depth:
            self.depthmap_embedder = RayMapControlEmbedding(
                conditioning_embedding_channels=self.hidden_size // 2,
                conditioning_size = 1,
                block_out_channels=[16, 32, 96, 256],
            )

            self.depthmap_embedder_temp = RayMapControlTempEmbedding(
                self.hidden_size, self.raymap_embedder_downsample_rate)
            self.depthmap_patchifier = PatchEmbed2D(
                # self.patch_size, self.hidden_size, self.hidden_size
                patch_size=self.patch_size[1],
                in_channels=self.inner_dim,
                embed_dim=self.inner_dim,
                use_abs_pos=not self.use_rope, 
            )

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)


    def encode_text(self, y, mask=None, drop_cond_mask=None):
        y = self.y_embedder(y)
        y_lens = [y.shape[2]] * y.shape[0]
        y = y.squeeze(1)
        return y, y_lens

    def prepare_text_embedding(self, text_encoder):
        @torch.no_grad()
        def text_to_embedding(text):
            ret = text_encoder.encode(text)
            hidden_state, _ = self.encode_text(ret['y'], mask=None)
            return hidden_state[:, :int(ret['mask'].sum(dim=1))]
        _training = self.training
        self.training = False
        self.training = _training

    def encode_seg_map(self, seg_embed, NC, H, W):
        B, T = seg_embed.shape[0], seg_embed.shape[2]
        controlnet_cond = self.segmap_embedder(seg_embed)
        controlnet_cond = rearrange(controlnet_cond, "BNC T C ... -> BNC C T ... ") 
        controlnet_cond = self.segmap_embedder_temp(controlnet_cond)
        controlnet_cond = self.segmap_patchifier(controlnet_cond)
        return controlnet_cond

    def encode_depth_map(self, depth_embed, NC, H, W):
        B, T = depth_embed.shape[0], depth_embed.shape[2]
        controlnet_cond = self.depthmap_embedder(depth_embed)
        controlnet_cond = rearrange(controlnet_cond, "BNC T C ... -> BNC C T ... ") 
        controlnet_cond = self.depthmap_embedder_temp(controlnet_cond)
        controlnet_cond = self.depthmap_patchifier(controlnet_cond)
        return controlnet_cond


    def encode_cond_sequence(self, cams, bbox, NC, T, y, mask, drop_cond_mask, drop_frame_mask):  # changed
        b = len(y)
        cond = []
        # encode y
        y, _ = self.encode_text(y, mask, drop_cond_mask)  # b, seq_len, dim
        # return y, None # change me!
        y = repeat(y, "b ... -> (b NC) ...", NC=NC)
        y = repeat(y, "B ... -> B T ...", T=T)
        cond = [y] + cond
        # cond = [cam_emb, y] + cond
        cond = torch.cat(cond, dim=2)  # B, T, len, dim
        cond = rearrange(cond, "B T L C -> B (T L) C")
        return cond, None

    def _operate_on_patched_inputs(
        self, 
        hidden_states, 
        encoder_hidden_states, 
        timestep, 
        fps,
        cams, 
        bbox,
        plucker_embed, 
        NC, 
        h, 
        w, 
        mask, 
        drop_cond_mask, 
        drop_frame_mask,
        added_cond_kwargs, 
        batch_size, 
        T,
        seg_map,
        depth_map
    ): 
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
        )

        raymap_embedding = None

        seg_map_embedding = None
        depth_map_embedding = None

        if self.segmap_embedder is not None:
            seg_map_embedding = self.encode_seg_map(seg_map, NC, h, w)

        if self.depthmap_embedder is not None:
            depth_map_embedding = self.encode_depth_map(depth_map, NC, h, w)

    
        hidden_states_vid = self.pos_embed(hidden_states)

        timestep_vid = timestep
        embedded_timestep_vid = embedded_timestep
        encoder_hidden_states_vid, _ = self.encode_cond_sequence(cams, bbox, NC,T, encoder_hidden_states, mask, drop_cond_mask, drop_frame_mask) 
        
        return timestep_vid,embedded_timestep_vid, hidden_states_vid, raymap_embedding, seg_map_embedding, depth_map_embedding, encoder_hidden_states_vid

    def _get_output_for_patched_inputs(
        self, hidden_states, timestep, class_labels, embedded_timestep, num_frames, height=None, width=None
    ):  
        # import ipdb;ipdb.set_trace()
        if self.norm_type != "ada_norm_single":
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)
        elif self.norm_type == "ada_norm_single":
            shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.squeeze(1)

        # unpatchify
        # if self.adaln_single is None:
        #     height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(-1, num_frames, height, width, self.patch_size[0], self.patch_size[1], self.patch_size[2], self.out_channels)
        )
        hidden_states = torch.einsum("nthwopqc->nctohpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, num_frames * self.patch_size[0], height * self.patch_size[1], width * self.patch_size[2])
        )
        return output


    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        y: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        height: Optional[torch.Tensor] = None,
        width: Optional[torch.Tensor] = None,
        drop_cond_mask: Optional[torch.Tensor] = None,
        drop_frame_mask: Optional[torch.Tensor] = None,
        mv_order_map = None,
        mask=None, 
        x_mask=None,
        bbox=None,
        cams=None,
        plucker_embed=None,
        seg_map=None,
        depth_map=None,
        **kwargs
    ):
        
        torch.autograd.set_detect_anomaly(True)
        B = x.size(0)
        if plucker_embed is not None:
            real_T = plucker_embed.size(2)
        elif seg_map is not None:
            real_T = seg_map.size(2)
        elif depth_map is not None:
            real_T = depth_map.size(2)
        else:
            real_T = int((x.size(2)-1)*4)+1


        if drop_cond_mask is None:  # camera
            drop_cond_mask = torch.ones((B), device=x.device, dtype=x.dtype)
        if drop_frame_mask is None:  # box & rel_pos
            drop_frame_mask = torch.ones((B, real_T), device=x.device, dtype=x.dtype)

        if mv_order_map is None:
            NC = 1
        else:
            NC = len(mv_order_map)

        # x = rearrange(x, "B (C NC) T ... -> (B NC) C T ...", NC=NC)
        batch_size, c, frame, h, w = x.shape
        T, H, W = self.get_dynamic_size(x)

        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                print.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        attention_mask_vid = None
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = attention_mask.to(self.dtype)
            if get_sequence_parallel_state():
                attention_mask_vid = attention_mask[:, :frame * nccl_info.world_size]  # b, frame, h, w
            else:
                attention_mask_vid = attention_mask[:, :frame]  # b, frame, h, w

            if attention_mask_vid.numel() > 0:
                attention_mask_vid_first_frame = attention_mask_vid[:, :1].repeat(1, self.patch_size[0]-1, 1, 1)
                attention_mask_vid = torch.cat([attention_mask_vid_first_frame, attention_mask_vid], dim=1)
                attention_mask_vid = attention_mask_vid.unsqueeze(1)  # b 1 t h w
                attention_mask_vid = F.max_pool3d(attention_mask_vid, kernel_size=(self.patch_size[0], self.patch_size[1], self.patch_size[2]), 
                                                  stride=(self.patch_size[0], self.patch_size[1], self.patch_size[2]))
                attention_mask_vid = rearrange(attention_mask_vid, 'b 1 t h w -> (b 1) 1 (t h w)') 


            attention_mask_vid = (1 - attention_mask_vid.bool().to(self.dtype)) * -10000.0 if attention_mask_vid.numel() > 0 else None


        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        encoder_attention_mask_vid = None
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0
            in_t = encoder_attention_mask.shape[1]
            encoder_attention_mask_vid = encoder_attention_mask[:, :in_t-use_image_num]  # b, 1, l
            encoder_attention_mask_vid = rearrange(encoder_attention_mask_vid, 'b 1 l -> (b 1) 1 l') if encoder_attention_mask_vid.numel() > 0 else None


        

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        timestep_vid,embedded_timestep_vid, hidden_states_vid, raymap_embedding, seg_map_embedding, depth_map_embedding, encoder_hidden_states_vid= self._operate_on_patched_inputs(
            x, y, timestep, fps, cams, bbox,plucker_embed, 
            NC, h, w, mask, drop_cond_mask, drop_frame_mask, added_cond_kwargs, B, T,
            seg_map,depth_map
        )



        if get_sequence_parallel_state():
            if hidden_states_vid is not None:
                print(333333333333333)
                hidden_states_vid = rearrange(hidden_states_vid, 'b s h -> s b h', b=batch_size).contiguous()
                encoder_hidden_states_vid = rearrange(encoder_hidden_states_vid, 'b s h -> s b h',
                                                      b=batch_size).contiguous()
                timestep_vid = timestep_vid.view(batch_size, 6, -1).transpose(0, 1).contiguous()


        for block in self.transformer_blocks:
            if True:
                hidden_states_vid = auto_grad_checkpoint(
                    block,
                    hidden_states_vid,
                    attention_mask_vid,
                    encoder_hidden_states_vid,
                    encoder_attention_mask_vid,
                    timestep_vid,
                    cross_attention_kwargs,
                    class_labels,
                    frame, 
                    H, 
                    W, 
                    mv_order_map,
                    raymap_embedding,
                    None,
                    seg_map_embedding,
                    depth_map_embedding
                )
                # hidden_states_vid=hidden_states_vid+seg_map_embedding+depth_map_embedding

            else:
                hidden_states_vid = block(
                    hidden_states_vid,
                    attention_mask=attention_mask_vid,
                    encoder_hidden_states=encoder_hidden_states_vid,
                    encoder_attention_mask=encoder_attention_mask_vid,
                    timestep=timestep_vid,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=frame, 
                    height=H, 
                    width=W, 
                    neighboring_view_pair=mv_order_map,
                    raymap_embedding=raymap_embedding,
                    seg_map=seg_map_embedding,
                    depth_map=depth_map_embedding
                )


        output_vid = self._get_output_for_patched_inputs(
            hidden_states=hidden_states_vid,
            timestep=timestep_vid,
            class_labels=class_labels,
            embedded_timestep=embedded_timestep_vid,
            num_frames=frame, 
            height=H,
            width=W,
        )  # b c t h w
        # output = rearrange(output_vid, "(b nc) c t h w -> b (c nc) t h w", nc = NC)
        # return output.to(torch.float32)
        return output_vid.to(torch.float32)


@MODELS.register_module("DIT3DCTRL")
def MVDIT3DCTRL(from_pretrained=None, force_huggingface=False, **kwargs):
    config = DiT3DCTRLConfig(
        **kwargs
    )
    model = DIT3DCTRL(config)

    return model



if __name__ == '__main__':
    from uniscenev2.utils.misc import get_model_numel, format_numel_str
    device = torch.device('cuda:0')
    num_layers = 28
    cross_attention_dim = 1152
    num_attention_heads = 16
    attention_head_dim = 72
    global_flash_attn = True
    global_layernorm = True
    global_xformers = True
    sp_size = 1
    ditcfg = DiT3DCTRLConfig(
        num_attention_heads = num_attention_heads,
        attention_head_dim = attention_head_dim,
        in_channels = 32,
        out_channels = 16,
        num_layers = num_layers,
        norm_num_groups = 32,
        cross_attention_dim = cross_attention_dim,
        attention_bias=True,
        num_vector_embeds = None,
        patch_size = (1,2,2),
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        use_linear_projection = False,
        only_cross_attention = False,
        double_self_attention = False,
        upcast_attention = False,
        norm_type="ada_norm_single", 
        norm_elementwise_affine=False,
        norm_eps = 1e-6,
        attention_type = "default",
        caption_channels = 4096,
        use_additional_conditions = None,
        attention_mode='flash',
        downsampler = None, 
        use_rope = True,
        use_stable_fp32 = False,
        # inpaint
        vae_scale_factor_t = 4,
        class_dropout_prob = 0.1,
        model_max_length = 300,
        num_camera = 3,
        with_depth = True,
        with_seg = True,
    )
    model = DIT3DCTRL(ditcfg).to(device).to(torch.bfloat16)
    model_numel, model_numel_trainable = get_model_numel(model)
    print(
        "[Diffusion] Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )
    model.eval()

    B = 1

    hidden_states = torch.rand(3,32,9,80,120).to(device,torch.bfloat16)
    timestep = torch.rand(3).to(device,torch.bfloat16)
    encoder_hidden_states = torch.rand(B,1,300,4096).to(device,torch.bfloat16)
    # cams_params = torch.rand(24, 17, 1, 3, 7).to(device,torch.bfloat16)
    fps = (torch.zeros((B))+12).to(device,torch.bfloat16)
    height = (torch.ones((B,1))+460).to(device,torch.bfloat16)
    width = (torch.ones((B,1))+960).to(device,torch.bfloat16)
    drop_cond_mask = torch.ones((3)).to(device,torch.bfloat16)
    drop_frame_mask=torch.ones((3, 33)).to(device,torch.bfloat16)
    mask=torch.ones(4, 300).to(device,torch.bfloat16)
    x_mask=torch.ones(3, 9).to(device,torch.bool)
    mv_order_map = {
        0: [1],
        1: [0, 2],
        2: [1]
    }
    # plucker_embed=torch.rand((24, 6, 17, 448, 800)).to(device,torch.bfloat16)
    seg_map = torch.rand((3, 1, 33, 640, 960)).to(device,torch.bfloat16)
    depth_map = torch.rand((3, 1, 33, 640, 960)).to(device,torch.bfloat16)

    import time
    with torch.no_grad(), torch.autocast('cuda'):
        start_time = time.time()
        res = model(
            x=hidden_states,
            timestep=timestep,
            y=encoder_hidden_states,
            fps=fps,
            height=height,
            width=width,
            drop_cond_mask=drop_cond_mask,
            drop_frame_mask=drop_frame_mask,
            mv_order_map=mv_order_map,
            mask=None,
            x_mask=x_mask,
            bbox=None,
            cams=None,
            plucker_embed=None,
            seg_map=seg_map,
            depth_map=depth_map
        )
        print(time.time()-start_time)

    embed()
    exit()