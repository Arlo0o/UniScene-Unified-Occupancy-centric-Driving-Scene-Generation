import os
import logging

DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")
from packaging import version as pver
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from transformers import PretrainedConfig, PreTrainedModel
from uniscenev2_video.acceleration.checkpoint import auto_grad_checkpoint
from uniscenev2_video.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
from uniscenev2_video.acceleration.parallel_states import get_sequence_parallel_group
from uniscenev2_video.models.layers.blocks import (
    Attention,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    PositionEmbedding2D,
    MultiHeadAttention,
    SeqParallelMultiHeadAttention,
    SeqParallelMultiHeadCrossAttention,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_layernorm,
    t2i_modulate,
)
from uniscenev2_video.registry import MODELS
from uniscenev2_video.utils.ckpt_utils import load_checkpoint
from uniscenev2_video.utils.misc import warn_once
from uniscenev2_video.models.stdit.embedder import RayMapControlTempEmbedding
from uniscenev2_video.models.utils import zero_module, load_module
from IPython import embed
from uniscenev2_video.models.layers.rope import RoPE3D

class MultiViewSTDiT3DBlock(nn.Module):
    """
    Adapt PixArt & STDiT3 block for multiview generation.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flash_attn=False,
        enable_xformers=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        sequence_parallelism_temporal=True,
        rope=None,
        qk_norm=False,
        raymap_proj=False
    ):
        super().__init__()
        self.raymap_proj = raymap_proj
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn
        self.enable_sequence_parallelism = enable_sequence_parallelism

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)

        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=rope,
            enable_flash_attn=enable_flash_attn,
            enable_xformers=enable_xformers,
            is_cross_attention=False,
        )

        self.cross_attn = MultiHeadAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            enable_flash_attn=enable_flash_attn,
            enable_xformers=enable_xformers,
            is_cross_attention=True,
            rope=False,
        )
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )

        self.norm3 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        # if split T, this is local attn; if split S, need full parallel.
        self.cross_view_attn = MultiHeadAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            enable_flash_attn=enable_flash_attn,
            enable_xformers=enable_xformers,
            is_cross_attention=True,
            rope=False,
        )
        self.mva_proj = zero_module(nn.Linear(hidden_size, hidden_size))

        # other helpers
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
        self.scale_shift_table_mva = nn.Parameter(torch.randn(3, hidden_size) / hidden_size**0.5)
        self.camera_pose_proj = None
        if self.raymap_proj:
            self.camera_pose_proj = nn.Linear(hidden_size, hidden_size)

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def _construct_attn_input_from_map(self, batch_size, norm_hidden_states, order_map: dict):
        hidden_states_in1 = []
        hidden_states_in2 = []
        cam_order = []
        for key, values in order_map.items():
            for value in values:
                hidden_states_in1.append(norm_hidden_states[:,key])
                hidden_states_in2.append(norm_hidden_states[:,value])
                cam_order+=[key]*batch_size

        hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
        hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
        cam_order = torch.LongTensor(cam_order)
        return hidden_states_in1, hidden_states_in2, cam_order


    def forward(
        self,
        x,
        y,
        t,  # this t
        mask=None,  # text mask
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0, for x_mask
        # dim param, we need them for dynamic input size
        T=None,  # number of frames
        height=None, 
        width=None,  # number of pixel patches
        NC=None,  # number of cameras
        # attn indexes, we need them for dynamic camera num/T
        mv_order_map=None,
        t_order_map=None,
        ray_map = None
    ):
        B, N, C = x.shape  # [6, 350, 1152]
        assert (N == T * height*width) and (B % NC == 0)
        b = B // NC
        if self.camera_pose_proj is not None:
            camera_pose_embed = self.camera_pose_proj(ray_map)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = repeat(
            self.scale_shift_table[None] + t.reshape(b, 6, -1),
            "b ... -> (b NC) ...", NC=NC,
        ).chunk(6, dim=1)

        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        ######################
        # attention
        ######################
        if self.camera_pose_proj is not None:
            x_m = x_m+camera_pose_embed
        x_m = self.attn(x_m,frame=T,height=height,width=width)

        # modulate (attention)
        x_m_s = gate_msa * x_m

        # residual
        x = x + self.drop_path(x_m_s)

        ######################
        # cross attn
        ######################
        # assert mask is None
        if y.shape[1] == 1:
            # x_c = self.cross_attn(x, y[:, 0], mask)
            x_c = self.cross_attn(x, y[:, 0])
        elif y.shape[1] == T:
            y_c = rearrange(y, "B T L C -> B (T L) C", T=T)
            if self.camera_pose_proj is not None:
                x = x+camera_pose_embed

            # x_c = self.cross_attn(x, y_c, mask)
            x_c = self.cross_attn(x, y_c,frame=T,height=height,width=width)
        else:
            raise RuntimeError(f"unsupported y.shape[1] = {y.shape[1]}")

        # residual, we skip drop_path here
        x = x + x_c

        ######################
        # multi-view cross attention
        ######################
        assert mv_order_map is not None
        # here we re-use the first 3 parameters from t and t0
        shift_mva, scale_mva, gate_mva = repeat(
            self.scale_shift_table_mva[None] + t[:, :3].reshape(b, 3, -1),
            "b ... -> (b NC) ...", NC=NC,
        ).chunk(3, dim=1)
        x_v = t2i_modulate(self.norm3(x), shift_mva, scale_mva)

        # Prepare inputs for multiview cross attention
        x_mv = rearrange(x_v, "(B NC) S C -> B NC S C", NC=NC)
        mv_batch_size = x_mv.shape[0]
        x_targets, x_neighbors, cam_order = self._construct_attn_input_from_map(
            mv_batch_size, x_mv, mv_order_map)
        # multi-view cross attention forward with batched neighbors
        cross_view_attn_output_raw = self.cross_view_attn(
            x_targets, x_neighbors,frame=T,height=height,width=width)
        # arrange output tensor for sum over neighbors
        cross_view_attn_output = torch.zeros_like(x_mv)
        # cross_view_attn_output_raw [400, 350, 1152] t=20 b=1 ， c=1152
        for cam_i in range(NC):
            attn_out_mv = rearrange(
                cross_view_attn_output_raw[cam_order == cam_i],
                "(n_neighbors b) ... -> b n_neighbors ...",
                b=mv_batch_size,
            )
            cross_view_attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
        cross_view_attn_output = rearrange(
            cross_view_attn_output, "B NC S C -> (B NC) S C", NC=NC)
        # modulate (cross-view attention)
        x_v_s = gate_mva * cross_view_attn_output
        # if x_mask is not None:
        #     x_v_s_zero = gate_mva_zero * cross_view_attn_output
        #     x_v_s = self.t_mask_select(x_mask, x_v_s, x_v_s_zero, T, S)
        # residual
        x_v_s = self.mva_proj(self.drop_path(x_v_s))
        x = x + x_v_s

        ######################
        # MLP
        ######################
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        # if x_mask is not None:
        #     x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
        #     x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m
        # if x_mask is not None:
        #     x_m_s_zero = gate_mlp_zero * x_m
        #     x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + self.drop_path(x_m_s)
        return x



class STDiT3DConfig(PretrainedConfig):
    model_type = "STDiT3D"

    def __init__(
        self,
        input_size=(1, 32, 32),
        input_sq_size=512,
        force_pad_h_for_sp_size=None,
        simulate_sp_size=[],
        in_channels=4,
        out_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path: float = 0.0,
        caption_channels=4096,
        model_max_length=300,
        qk_norm=True,
        enable_flash_attn=False,
        enable_xformers=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        freeze_y_embedder=False,

        with_temp_block=True,
        freeze_x_embedder=False,
        freeze_old_embedder=False,
        freeze_temporal_blocks=False,
        freeze_old_params=False,
        zero_and_train_embedder=None,
        only_train_base_blocks=False,
        only_train_temp_blocks=False,
        qk_norm_trainable=False,
        sequence_parallelism_temporal=False,
        control_depth=13,
        use_x_control_embedder=False,
        use_st_cross_attn=False,
        uncond_cam_in_dim=(3, 7),
        raymap_embedder_cls=None,
        raymap_embedder_param={},
        raymap_embedder_downsample_rate=4,
        micro_frame_size=17,
        bbox_embedder_cls=None,
        bbox_embedder_param={},
        **kwargs,
    ):
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.force_pad_h_for_sp_size = force_pad_h_for_sp_size
        self.simulate_sp_size = simulate_sp_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.caption_channels = caption_channels
        self.model_max_length = model_max_length
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.freeze_y_embedder = freeze_y_embedder
        
        self.bbox_embedder_cls = bbox_embedder_cls
        self.bbox_embedder_param = bbox_embedder_param
        self.zero_and_train_embedder = zero_and_train_embedder
        self.qk_norm_trainable = qk_norm_trainable
        self.enable_xformers = enable_xformers
        self.sequence_parallelism_temporal = sequence_parallelism_temporal
        self.uncond_cam_in_dim = uncond_cam_in_dim
        self.raymap_embedder_cls = raymap_embedder_cls
        self.raymap_embedder_param = raymap_embedder_param
        self.raymap_embedder_downsample_rate = raymap_embedder_downsample_rate
        self.micro_frame_size = micro_frame_size
        super().__init__(**kwargs)



class STDiT3D(PreTrainedModel):
    """
    Diffusion model with a Transformer backbone.
    """
    config_class = STDiT3DConfig

    def __init__(self, config: STDiT3DConfig):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        if config.pred_sigma:
            self.out_channels = config.in_channels * 2
        else:
            self.out_channels = config.out_channels

        # model size related
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        # computation related
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_xformers = config.enable_xformers
        self.enable_layernorm_kernel = config.enable_layernorm_kernel
        self.enable_sequence_parallelism = config.enable_sequence_parallelism
        self.sequence_parallelism_temporal = config.sequence_parallelism_temporal

        # input size related
        self.patch_size = config.patch_size
        self.input_sq_size = config.input_sq_size
        # self.pos_embed = PositionEmbedding2D(self.hidden_size)
        # self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)
        self.rope = RoPE3D(interpolation_scale_thw=(1,1,1))
        self.force_pad_h_for_sp_size = config.force_pad_h_for_sp_size
        self.simu_sp_size = config.simulate_sp_size

        # embedding
        self.x_embedder = PatchEmbed3D(self.patch_size, self.in_channels, self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )
        self.fps_embedder = SizeEmbedder(self.hidden_size)

        # base_token, should not be trainable
        self.register_buffer("base_token", torch.randn(self.hidden_size))
        self.camera_embedder = None
        if config.cam_encoder_cls is not None:
            # init camera encoder
            self.camera_embedder = load_module(config.cam_encoder_cls)(
                out_dim=config.hidden_size, **config.cam_encoder_param)

        # init bbox encoder
        self.bbox_embedder = None
        if config.bbox_embedder_cls is not None:
            self.bbox_embedder = load_module(config.bbox_embedder_cls)(
                **config.bbox_embedder_param)

        self.raymap_embedder = None
        if config.raymap_embedder_cls is not None:
            # init raymap 2D encoder
            self.raymap_embedder = load_module(config.raymap_embedder_cls)(
                conditioning_embedding_channels=self.hidden_size // 2,
                **config.raymap_embedder_param,
            )

        self.micro_frame_size = config.micro_frame_size  # should be the same as vae

        self.raymap_embedder_temp = RayMapControlTempEmbedding(
            self.hidden_size, config.raymap_embedder_downsample_rate)
        self.raymap_patchifier = PatchEmbed3D(self.patch_size, self.hidden_size, self.hidden_size)

        # base blocks
        drop_path = [x.item() for x in torch.linspace(0, config.drop_path, self.depth)]

        self.base_blocks = nn.ModuleList(
            [
                MultiViewSTDiT3DBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flash_attn=self.enable_flash_attn,
                    enable_xformers=self.enable_xformers,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    enable_sequence_parallelism=self.enable_sequence_parallelism,
                    sequence_parallelism_temporal=self.sequence_parallelism_temporal,
                    qk_norm=config.qk_norm,
                    raymap_proj=True if self.camera_embedder is not None else False,
                    rope=self.rope,
                )
                for i in range(self.depth)
            ]
        )
        # final layer
        self.final_layer = T2IFinalLayer(self.hidden_size, np.prod(self.patch_size), self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # NOTE: some proj layers are zero-initialized on creating.
        def _zero_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # new block in base
        for block in self.base_blocks:
            _zero_init(block.mva_proj)

        _zero_init(self.camera_embedder.after_proj)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d): cr. PixArt
        w = self.raymap_patchifier.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize caption embedding MLP: cr. PixArt
        nn.init.normal_(self.camera_embedder.emb2token.weight, std=0.02)

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

    def prepare_text_embedding(self, text_encoder):
        @torch.no_grad()
        def text_to_embedding(text):
            ret = text_encoder.encode(text)
            hidden_state, _ = self.encode_text(ret['y'], mask=None)
            return hidden_state[:, :int(ret['mask'].sum(dim=1))]
        _training = self.training
        self.training = False
        if self.bbox_embedder is not None:
            self.bbox_embedder.prepare(text_to_embedding)
        self.base_token[:] = text_to_embedding("").squeeze()
        self.training = _training

    def encode_text(self, y, mask=None, drop_cond_mask=None):
        # NOTE: we do not use y mask, but keep the batch dim.
        # NOTE: we do not use drop in y_embedder
        if drop_cond_mask is not None:
            y = self.y_embedder(y, False, force_drop_ids=1 - drop_cond_mask)  # [B, 1, N_token, C]
        else:
            y = self.y_embedder(y, False)  # [B, 1, N_token, C]
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            y_lens = [i + 1 for i in mask.sum(dim=1).tolist()]
            max_len = int(min(max(y_lens), y.shape[2]))  # we need min because of +1
            if drop_cond_mask is not None and not drop_cond_mask.all():  # on any drop, this should be the max
                assert max_len == y.shape[2]
            # y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
            y = y.squeeze(1)[:, :max_len]
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1)
        return y, y_lens


    def encode_cam(self, cam, embedder, drop_mask):
        B, T, S = cam.shape[:3]
        NC = B // drop_mask.shape[0]
        mask = repeat(drop_mask, "b T -> (b NC T S)", NC=NC, S=S)
        cam = rearrange(cam, "B T S ... -> (B T S) ...")
        cam_emb, _ = embedder.embed_cam(cam, mask, T=T, S=S)  # changed here
        return cam_emb

    def sample_box_latent(self, n_boxes, generator=None):
        if self.bbox_embedder.mean_var is None:
            latent = None
        else:
            latent = torch.randn(
                (n_boxes, self.bbox_embedder.box_latent_shape[1]),
                generator=generator,
            )
        return latent

    def encode_box(self, bboxes, drop_mask):  # changed
        B, T, seq_len = bboxes['bboxes'].shape[:3]
        bbox_embedder_kwargs = {}
        for k, v in bboxes.items():
            bbox_embedder_kwargs[k] = v.clone()
        # each key should have dim like: (b, seq_len, dim...)
        # bbox_embedder_kwargs["masks"]: 0 -> null, -1 -> mask, 1 -> keep
        # drop_mask: 0 -> mask, 1 -> keep
        drop_mask = repeat(drop_mask, "B T -> B T S", S=seq_len)
        _null_mask = torch.ones_like(bbox_embedder_kwargs["masks"])
        _null_mask[bbox_embedder_kwargs["masks"] == 0] = 0
        _mask = torch.ones_like(bbox_embedder_kwargs["masks"])
        _mask[bbox_embedder_kwargs["masks"] == -1] = 0
        _mask[torch.logical_and(
            bbox_embedder_kwargs["masks"] == 1,
            drop_mask == 0,  # only drop those real boxes
        )] = 0
        bbox_emb = self.bbox_embedder(
            bboxes=bbox_embedder_kwargs['bboxes'],
            classes=bbox_embedder_kwargs["classes"].type(torch.int32),
            null_mask=_null_mask,
            mask=_mask,
            box_latent=bbox_embedder_kwargs.get('box_latent', None),
        )
        # bbox_emb = rearrange(bbox_emb, "(B T) ... -> B T ...", T=T)
        return bbox_emb

    def encode_cond_sequence(self, bbox, cams, y, mask, drop_cond_mask, drop_frame_mask):  # changed
        b = len(y)
        NC, T = cams.shape[0] // b, cams.shape[1]
        cond = []

        # encode y
        y, _ = self.encode_text(y, mask, drop_cond_mask)  # b, seq_len, dim
        # return y, None # change me!
        y = repeat(y, "b ... -> (b NC) ...", NC=NC)

        # encode box
        if bbox is not None:
            drop_box_mask = torch.logical_and(drop_cond_mask[:, None], drop_frame_mask)  # b, T
            drop_box_mask = repeat(drop_box_mask, "b ... -> (b NC) ...", NC=NC)
            bbox_emb = self.encode_box(bbox, drop_mask=drop_box_mask)  # B, T, box_len, dim
            # bbox_emb = bbox_emb.mean(1)  # pooled token
            # zero proj on base token
            bbox_emb = self.base_token[None, None, None] + bbox_emb
            cond.append(bbox_emb)
            T = bbox_emb.shape[1]


        # encode cam, just take from first frame
        cam_emb = self.encode_cam(
            # cams, self.camera_embedder, repeat(drop_cond_mask, "b -> b T", T=T))
            cams[:, 0:1], self.camera_embedder, repeat(drop_cond_mask, "b -> b T", T=1))
        # frame_emb = self.encode_cam(rel_pos, self.frame_embedder, drop_frame_mask)
        cam_emb = rearrange(cam_emb, "(B 1 S) ... -> B 1 S ...", S=cams.shape[2])
        # frame_emb = frame_emb.mean(1)  # pooled token
        # zero proj on base token
        cam_emb = self.base_token[None, None, None] + cam_emb
        # frame_emb = self.base_token[None, None, None] + frame_emb

        cam_emb = repeat(cam_emb, 'B 1 S ... -> B T S ...', T=T)
        y = repeat(y, "B ... -> B T ...", T=T)

        cond = [cam_emb, y] + cond

        cond = torch.cat(cond, dim=2)  # B, T, len, dim
        return cond, None


    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def ray_condition(self, K, c2w, H, W, device, flip_flag=None):
        # c2w: B, V, 4, 4
        # K: B, V, 3, 3

        def custom_meshgrid(*args):
            # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
            if pver.parse(torch.__version__) < pver.parse('1.10'):
                return torch.meshgrid(*args)
            else:
                return torch.meshgrid(*args, indexing='ij')

        B, V = K.shape[:2]

        j, i = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
        )
        i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]
        j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]

        n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
        if n_flip > 0:
            j_flip, i_flip = custom_meshgrid(
                torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
                torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
            )
            i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
            j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
            i[:, flip_flag, ...] = i_flip
            j[:, flip_flag, ...] = j_flip

        fx = K[..., 0, 0].unsqueeze(-1)
        fy = K[..., 1, 1].unsqueeze(-1)
        cx = K[..., 0, 2].unsqueeze(-1)
        cy = K[..., 1, 2].unsqueeze(-1)

        zs = torch.ones_like(i)  # [B, V, HxW]
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        zs = zs.expand_as(ys)

        directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
        directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

        rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
        rays_o = c2w[..., :3, 3]  # B, V, 3
        rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, HW, 3
        # c2w @ dirctions
        rays_dxo = torch.cross(rays_o, rays_d)  # B, V, HW, 3
        plucker = torch.cat([rays_dxo, rays_d], dim=-1)
        plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
        # plucker = plucker.permute(0, 1, 4, 2, 3)
        plucker = rearrange(plucker, "b f h w c -> b c f h w")  # [b, 6, f, h, w]
        return plucker


    def encode_ray_map(self, plucker_embed, NC, h_pad_size, x_shape):
        B, T = plucker_embed.shape[0], plucker_embed.shape[2]
        controlnet_cond = self.raymap_embedder(plucker_embed,drop=True if self.training else False)
        controlnet_cond = rearrange(controlnet_cond, "B NC T C ... -> (B NC) C T ... ") 

        if self.micro_frame_size is None:
            controlnet_cond = self.raymap_embedder_temp(controlnet_cond)
        else:
            z_list = []
            for i in range(0, controlnet_cond.shape[2], self.micro_frame_size):
                x_z_bs = controlnet_cond[:, :, i: i + self.micro_frame_size]
                z = self.raymap_embedder_temp(x_z_bs)
                z_list.append(z)
            controlnet_cond = torch.cat(z_list, dim=2)

        if controlnet_cond.shape[-3:] != x_shape[-3:]:
            # [-3:] for (T, H, W)
            warn_once(
                f"For x_shape = {x_shape[-3:]}, we interpolate map cond from "
                f"{controlnet_cond.shape[-3:]}"
            )

            if np.prod(x_shape[-3:]) > np.prod([33, 106, 200]) and controlnet_cond.shape[0] > 1:
                # slice batch
                _controlnet_cond = []
                for ci in range(controlnet_cond.shape[0]):
                    _controlnet_cond.append(
                        F.interpolate(controlnet_cond[ci:ci + 1], x_shape[-3:])
                    )
                controlnet_cond = torch.cat(_controlnet_cond, dim=0)
            else:
                if np.prod(x_shape[-3:]) > np.prod([33, 106, 200]):
                    warn_once(f"shape={controlnet_cond.shape} cannot be splitted!")
                controlnet_cond = F.interpolate(controlnet_cond, x_shape[-3:])

        if h_pad_size > 0:
            hx_pad_size = h_pad_size * self.patch_size[1]
            # pad c along the H dimension
            controlnet_cond = F.pad(controlnet_cond, (0, 0, 0, hx_pad_size))


        controlnet_cond = self.raymap_patchifier(controlnet_cond)
        return controlnet_cond


    def forward(self, x, timestep, y, bbox, cams, fps,
                height, width, drop_cond_mask=None, drop_frame_mask=None,
                mv_order_map=None, t_order_map=None, mask=None, x_mask=None,plucker_embed=None,
                **kwargs):
        # torch.autograd.set_detect_anomaly(True)
        dtype = self.x_embedder.proj.weight.dtype
        B, real_T = x.size(0), plucker_embed.size(2)
        if drop_cond_mask is None:  # camera
            drop_cond_mask = torch.ones((B), device=x.device, dtype=x.dtype)
        if drop_frame_mask is None:  # box & rel_pos
            drop_frame_mask = torch.ones((B, real_T), device=x.device, dtype=x.dtype)

        if mv_order_map is None:
            NC = 1
        else:
            NC = len(mv_order_map)

        x = x.to(dtype)
        x = rearrange(x, "B (C NC) T ... -> (B NC) C T ...", NC=NC)

        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # === get pos embed ===
        _, _, Tx, Hx, Wx = x.size()
        x_in_shape = x.shape  # before pad
        T, H, W = self.get_dynamic_size(x)
        S = H * W
        # adjust for sequence parallelism
        # we need to ensure H * W is divisible by sequence parallel size
        # for simplicity, we can adjust the height to make it divisible
        h_pad_size = 0
        h_pad_size = 0
        if self.training:
            _simu_sp_size = self.simu_sp_size
        else:
            if len(self.simu_sp_size) > 0:
                warn_once(f"We will ignore `simu_sp_size` if not training.")
            _simu_sp_size = []

        if self.force_pad_h_for_sp_size is not None:
            if S % self.force_pad_h_for_sp_size != 0:
                h_pad_size = self.force_pad_h_for_sp_size - H % self.force_pad_h_for_sp_size
                warn_once(
                    f"Your input shape {x.shape} was rounded into {(T, H, W)}. "
                    f"With force_pad_h_for_sp_size={self.force_pad_h_for_sp_size}, "
                    f"it is padded by H with {h_pad_size}. "
                )
        elif len(_simu_sp_size) > 0:
            if self.enable_sequence_parallelism and not self.sequence_parallelism_temporal:
                # make sure the simulated is greater than real sp_size
                sp_size = dist.get_world_size(get_sequence_parallel_group())
                possible_sp_size = []
                for _sp_size in _simu_sp_size:
                    if _sp_size >= sp_size:
                        possible_sp_size.append(_sp_size)
            else:
                possible_sp_size = _simu_sp_size
            # random pick one
            simu_sp_size = random.choice(possible_sp_size)
            if S % simu_sp_size != 0:
                h_pad_size = simu_sp_size - H % simu_sp_size
            if h_pad_size > 0:
                warn_once(
                    f"Your input shape {x.shape} was rounded into {(T, H, W)}. "
                    f"For simu_sp_size={simu_sp_size} out of {possible_sp_size}, "
                    f"it is padded by H with {h_pad_size}. "
                    "Please pay attention to potential mismatch between w/ and w/o sp."
                )

        elif self.enable_sequence_parallelism and not self.sequence_parallelism_temporal:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            if S % sp_size != 0:
                h_pad_size = sp_size - H % sp_size
            if h_pad_size > 0:
                warn_once(
                    f"Your input shape {x.shape} was rounded into {(T, H, W)}. "
                    f"For sp_size={sp_size}, it is padded by H with {h_pad_size}. "
                    "Please pay attention to potential mismatch between w/ and w/o sp."
                )

        if h_pad_size > 0:
            # pad x along the H dimension
            hx_pad_size = h_pad_size * self.patch_size[1]
            x = F.pad(x, (0, 0, 0, hx_pad_size))
            # adjust parameters
            H += h_pad_size
            S = H * W
            if self.enable_sequence_parallelism and not self.sequence_parallelism_temporal:
                sp_size = dist.get_world_size(get_sequence_parallel_group())
                assert S % sp_size == 0, f"S={S} should be divisible by {sp_size}!"


        # base_size = round(S**0.5)
        # resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        # scale = resolution_sq / self.input_sq_size
        # pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)


        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        # if x_mask is not None:
        #     t0_timestep = torch.zeros_like(timestep)
        #     t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
        #     t0 = t0 + fps
        #     t0_mlp = self.t_block(t0)

        y, y_lens = self.encode_cond_sequence(bbox, cams, y, mask, drop_cond_mask, drop_frame_mask)  # (B, L, D)

        if y.shape[1] != T and y.shape[1] > 1:
            warn_once(f"Got y length {y.shape[1]}, will interpolate to {T}.")
            seq_len = y.shape[2]
            y = rearrange(y, "B T L D -> B (L D) T")
            y = F.interpolate(y, T)
            y = rearrange(y, "B (L D) T -> B T L D", L=seq_len)
        c = self.encode_ray_map(plucker_embed, NC, h_pad_size, x_in_shape)

        c = rearrange(c, "B (T S) C -> B T S C", T=T)

        # === get x embed ===
        x_b = self.x_embedder(x)  # [B, N, C]
        x_b = rearrange(x_b, "B (T S) C -> B T S C", T=T, S=S)
        # x_b = x_b + pos_emb
        x = x_b
        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            assert not self.sequence_parallelism_temporal, "not support!"
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="down")
            c = split_forward_gather_backward(c, get_sequence_parallel_group(), dim=2, grad_scale="down")
            S = S // dist.get_world_size(get_sequence_parallel_group())

        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
        c = rearrange(c, "B T S C -> B (T S) C", T=T, S=S)

        # === blocks ===
        # if x_mask is not None:
        #     x_mask = repeat(x_mask, "b ... -> (b NC) ...", NC=NC)

        for block_i in range(0, self.depth):
            x = auto_grad_checkpoint(
                self.base_blocks[block_i],
                x, y, t_mlp, y_lens, x_mask, t0_mlp, T, H, W, NC, mv_order_map, t_order_map,c)

        if self.enable_sequence_parallelism:
            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
            x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="up")
            S = S * dist.get_world_size(get_sequence_parallel_group())
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # === final layer ===
        x = self.final_layer(
            x, repeat(t, "b d -> (b NC) d", NC=NC),
            x_mask, repeat(t0, "b d -> (b NC) d", NC=NC) if t0 is not None else None,
            T, S,
        )
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        # HACK: to use scheduler, we never assume NC with C
        x = rearrange(x, "(B NC) C T ... -> B (C NC) T ...", NC=NC)
        return x


    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x

@MODELS.register_module("STDIT3D")
def MVSTDIT3D(from_pretrained=None, force_huggingface=False, **kwargs):

    config = STDiT3DConfig(
        **kwargs
    )
    model = STDiT3D(config)

    return model





if __name__ == "__main__":
    from uniscenev2_video.utils.misc import get_model_numel, format_numel_str
    device = torch.device('cuda:0')
    sp_size = 1
    global_flash_attn = True
    global_layernorm = True
    global_xformers = True
    config = STDiT3DConfig(
        input_size=(None, None, None),
        in_channels=32,
        out_channels=16,
        caption_channels=4096,
        text_encoder_model_max_length=300,
        depth=32, 
        hidden_size=2304, 
        patch_size=(1, 2, 2), 
        num_heads=24,
        simulate_sp_size=[4, 8],
        qk_norm=True,
        pred_sigma=False,
        enable_flash_attn=True and global_flash_attn,
        enable_layernorm_kernel=True and global_layernorm,
        enable_sequence_parallelism=sp_size > 1,
        freeze_y_embedder=False,
        with_temp_block=True,  # CHANGED
        use_x_control_embedder=True,
        enable_xformers = False and global_xformers,
        sequence_parallelism_temporal=False,
        use_st_cross_attn=False,
        uncond_cam_in_dim=(3, 7),
        cam_encoder_cls="uniscenev2_video.models.stdit.embedder.CamEmbedder",
        cam_encoder_param=dict(
            input_dim=3,
            # out_dim=1152,  # no need to set this.
            num=7,
            after_proj=True,
        ),
        micro_frame_size=None,
        control_skip_cross_view=True,
        control_skip_temporal=False,  # CHANGED
        raymap_embedder_cls="uniscenev2_video.models.stdit.embedder.RayMapControlEmbedding",
        raymap_embedder_param=dict(
            conditioning_size=6,
            block_out_channels=[16, 32, 96, 256],
            # conditioning_embedding_channels=1152,  # no need to set this.
        ),
        raymap_embedder_downsample_rate=4.5,  # CHANGED
        bbox_embedder_cls="uniscenev2_video.models.stdit.embedder.ContinuousBBoxWithTextTempEmbedding",
        bbox_embedder_param=dict(
            n_classes=10,
            class_token_dim=2304,
            trainable_class_token=False,
            embedder_num_freq=4,
            proj_dims=[2304, 512, 512, 2304],
            bbox_mode = 'all-xyz',
            minmax_normalize=False,
            use_text_encoder_init=True, 
            after_proj=True,
            sample_id=True,  # CHANGED
            # new
            num_heads=8,
            mlp_ratio=4.0,
            qk_norm=True,
            enable_flash_attn=False and global_flash_attn,
            enable_xformers=True and global_xformers,
            enable_layernorm_kernel=True and global_layernorm,
            use_scale_shift_table=True,
            time_downsample_factor=4.5,
        ),
    )
    model = STDiT3D(config).to(device).to(torch.bfloat16)

    model_numel, model_numel_trainable = get_model_numel(model)
    print(
        "[Diffusion] Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )
    model.eval()



    B = 1
    T = 17
    mv_order_map = {
        0: [5, 1],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3, 5],
        5: [4, 0],
    }


    x = torch.rand(1,192,5,28,50).to(device,torch.bfloat16)#torch.rand(6, 16, 17, 28, 50).to(device,torch.bfloat16)
    timestep = torch.rand(1).to(device,torch.bfloat16)
    y = torch.rand(1, 1, 300, 4096).to(device,torch.bfloat16)
    cams = torch.rand(6, 17, 1, 3, 7).to(device,torch.bfloat16)
    rel_pos = torch.rand(6, 17, 1, 4, 4).to(device,torch.bfloat16)
    fps = (torch.zeros((B))+12).to(device,torch.bfloat16)
    height = (torch.ones((B,1))+224).to(device,torch.bfloat16)
    width = (torch.ones((B,1))+400).to(device,torch.bfloat16)
    drop_cond_mask = torch.ones((B)).to(device,torch.bfloat16)
    drop_frame_mask=torch.ones((B, T)).to(device,torch.bfloat16)
    mv_order_map=mv_order_map
    t_order_map=None
    mask=torch.rand(1, 300).to(device,torch.bfloat16)
    x_mask=None#torch.ones(1, 5).to(device,torch.bool)
    bbox = None
    NC = 6
    cam_K = cams[:,:,0,:,:3]
    ego_to_world = rel_pos[:,:,0]
    cam_ext = torch.zeros_like(ego_to_world)
    cam_ext[:,:,0,0] = 1
    cam_ext[:,:,1,1] = 1
    cam_ext[:,:,2,2] = 1
    cam_ext[:,:,3,3] = 1
    cam_ext[:,:,:3] = cams[:,:,0,:,3:]
    c2w = ego_to_world @ cam_ext
    H, W = x.shape[-2:]
    plucker_embed = model.ray_condition(cam_K, c2w, H*8, W*8, device=cam_K.device)
    plucker_embed = rearrange(plucker_embed, "(B NC) C T ... -> B NC T C ...", NC=NC)


    import time
    embed()
    exit()
    with torch.no_grad():
        start_time = time.time()
        res = model(
            x=x, 
            timestep=timestep, 
            y=y, 
            bbox=bbox,
            cams=cams, 
            fps=fps,
            height=height, 
            width=width, 
            drop_cond_mask=drop_cond_mask, 
            drop_frame_mask=drop_frame_mask,
            mv_order_map=mv_order_map, 
            t_order_map=t_order_map, 
            mask=mask, 
            x_mask=x_mask,
            plucker_embed=plucker_embed

        )
        print(time.time()-start_time)