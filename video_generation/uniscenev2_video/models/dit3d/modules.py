from einops import rearrange
from torch import nn
import torch
import numpy as np

from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
from typing import Any, Dict, Optional
import re
import torch
import torch.nn.functional as F
from torch import nn
import diffusers
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward, GatedSelfAttentionDense
from diffusers.models.attention_processor import Attention as Attention_
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm
from uniscenev2_video.models.dit3d.rope import PositionGetter3D, RoPE3D
from uniscenev2_video.acceleration.parallel_states import get_sequence_parallel_state, nccl_info
from uniscenev2_video.acceleration.communications_plan import all_to_all_SBH
logger = logging.get_logger(__name__)
from IPython import embed
from diffusers.models.controlnet import zero_module


class PatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding but with 3D position embedding"""

    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        use_abs_pos=True, 
    ):
        super().__init__()
        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

    def forward(self, latent):
        b, _, _, _, _ = latent.shape
        # b c 1 h w
        latent = rearrange(latent, 'b c t h w -> (b t) c h w')
        latent = self.proj(latent)

        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BT C H W -> BT N C
        if self.layer_norm:
            latent = self.norm(latent)
        latent = rearrange(latent, '(b t) n c -> b t n c', b=b)
        return rearrange(latent, 'b t n c -> b (t n) c')
        # return rearrange(latent, '(b t) n c -> b (t n) c', b=b)
        # latent = rearrange(latent, '(b t) n c -> b t n c', b=b)
        # video_latent = rearrange(latent, 'b t n c -> b (t n) c')
        # return video_latent
    


class Attention(Attention_):
    def __init__(self, downsampler, attention_mode, use_rope, interpolation_scale_thw, **kwags):
        processor = AttnProcessor2_0(attention_mode=attention_mode, use_rope=use_rope, interpolation_scale_thw=interpolation_scale_thw)
        super().__init__(processor=processor, **kwags)

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if get_sequence_parallel_state():
            head_size = head_size // nccl_info.world_size
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, attention_mode='xformers', use_rope=False, interpolation_scale_thw=(1, 1, 1)):
        self.use_rope = use_rope
        self.interpolation_scale_thw = interpolation_scale_thw
        if self.use_rope:
            self._init_rope(interpolation_scale_thw)
        self.attention_mode = attention_mode
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")


    def _init_rope(self, interpolation_scale_thw):
        self.rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw)
        self.position_getter = PositionGetter3D()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        frame: int = 8, 
        height: int = 16, 
        width: int = 16, 
        *args,
        **kwargs,
    ) -> torch.FloatTensor:

        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if get_sequence_parallel_state():
            sequence_length, batch_size, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
        else:
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        if get_sequence_parallel_state():
            query = query.reshape(-1, attn.heads, head_dim)  # [s // sp, b, h * d] -> [s // sp * b, h, d]
            key = key.reshape(-1, attn.heads, head_dim)
            value = value.reshape(-1, attn.heads, head_dim)
            h_size = attn.heads * head_dim
            sp_size = nccl_info.world_size
            h_size_sp = h_size // sp_size
            query = all_to_all_SBH(query, scatter_dim=1, gather_dim=0).reshape(-1, batch_size, h_size_sp)
            key = all_to_all_SBH(key, scatter_dim=1, gather_dim=0).reshape(-1, batch_size, h_size_sp)
            value = all_to_all_SBH(value, scatter_dim=1, gather_dim=0).reshape(-1, batch_size, h_size_sp)
            query = query.reshape(-1, batch_size, attn.heads // sp_size, head_dim)
            key = key.reshape(-1, batch_size, attn.heads // sp_size, head_dim)
            value = value.reshape(-1, batch_size, attn.heads // sp_size, head_dim)
            if self.use_rope:
                # require the shape of (batch_size x nheads x ntokens x dim)
                pos_thw = self.position_getter(batch_size, t=frame * sp_size, h=height, w=width, device=query.device)
                query = self.rope(query, pos_thw)
                key = self.rope(key, pos_thw)
            # print('after rope query', query.shape, 'key', key.shape, 'value', value.shape)
            query = rearrange(query, 's b h d -> b h s d')
            key = rearrange(key, 's b h d -> b h s d')
            value = rearrange(value, 's b h d -> b h s d')
            if attention_mask is None or not torch.any(attention_mask.bool()):  # 0 mean visible
                attention_mask = None

            if self.attention_mode == 'flash':
                assert attention_mask is None, 'flash-attn do not support attention_mask'
                with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, dropout_p=attn.dropout, is_causal=False
                    )
            elif self.attention_mode == 'xformers':
                with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=attention_mask, dropout_p=attn.dropout, is_causal=False
                    )
            elif self.attention_mode == 'math':
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=attn.dropout, is_causal=False
                )
            else:
                raise NotImplementedError(f'Found attention_mode: {self.attention_mode}')

            hidden_states = rearrange(hidden_states, 'b h s d -> s b h d')
            hidden_states = hidden_states.reshape(-1, attn.heads // sp_size, head_dim)
            # [s * b, h // sp, d] -> [s // sp * b, h, d] -> [s // sp, b, h * d]
            hidden_states = all_to_all_SBH(hidden_states, scatter_dim=0, gather_dim=1).reshape(-1, batch_size, h_size)
        else:
            # query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            # key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            query = query.view(batch_size, -1, attn.heads, head_dim)
            key = key.view(batch_size, -1, attn.heads, head_dim)


            if self.use_rope:
                # require the shape of (batch_size x nheads x ntokens x dim)
                pos_thw = self.position_getter(batch_size, t=frame, h=height, w=width, device=query.device)
                query = self.rope(query, pos_thw)
                key = self.rope(key, pos_thw)

            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            if attention_mask is None or not torch.any(attention_mask.bool()):  # 0 mean visible
                attention_mask = None
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            # import ipdb;ipdb.set_trace()
            # print(attention_mask)
            if self.attention_mode == 'flash':
                assert attention_mask is None, 'flash-attn do not support attention_mask'
                with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, dropout_p=attn.dropout, is_causal=False
                    )
            elif self.attention_mode == 'xformers':
                with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=attention_mask, dropout_p=attn.dropout, is_causal=False
                    )
            elif self.attention_mode == 'math':
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=attn.dropout, is_causal=False
                )
            else:
                raise NotImplementedError(f'Found attention_mode: {self.attention_mode}')
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# # 需要定义零初始化包装器
# def zero_module(module):
#     # 将模块的参数初始化为零，同时保持结构
#     for p in module.parameters():
#         nn.init.zeros_(p)
#     return module

@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        attention_mode: str = "xformers", 
        downsampler: str = None, 
        use_rope: bool = False, 
        interpolation_scale_thw: Tuple[int] = (1, 1, 1), 
        n_cam=6,
        camera_control=False,
        with_seg=False,
        with_depth=False,
        use_attn_ctrl=False
    ):
        super().__init__()
        self.use_attn_ctrl = use_attn_ctrl
        self.with_seg = with_seg
        self.with_depth=with_depth
        self.n_cam = n_cam
        self.only_cross_attention = only_cross_attention
        self.downsampler = downsampler
        # We keep these boolean flags for backward-compatibility.
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"
        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None


        self.camera_pose_proj = None
        if camera_control:
            self.camera_pose_proj = nn.Linear(dim, dim)

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_zero":
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            self.norm1 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            attention_mode=attention_mode, 
            downsampler=downsampler, 
            use_rope=use_rope, 
            interpolation_scale_thw=interpolation_scale_thw, 
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if norm_type == "ada_norm":
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif norm_type == "ada_norm_continuous":
                self.norm2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            else:
                self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                attention_mode=attention_mode, 
                downsampler=False, 
                use_rope=False, 
                interpolation_scale_thw=interpolation_scale_thw, 
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )

        elif norm_type in ["ada_norm_zero", "ada_norm", "layer_norm", "ada_norm_continuous"]:
            self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)


        # # 6. Cross-View Attention
        if self.n_cam > 1:
            if norm_type == "ada_norm":
                self.norm4 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif norm_type == "ada_norm_zero":
                self.norm4 = AdaLayerNormZero(dim, num_embeds_ada_norm)
            elif norm_type == "ada_norm_continuous":
                self.norm4 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            else:
                self.norm4 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

            self.attn4 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                attention_mode=attention_mode, 
                downsampler=False, 
                use_rope=False, 
                interpolation_scale_thw=interpolation_scale_thw, 
            )  # is self-attn if encoder_hidden_states is none
            # self.connector = zero_module(nn.Linear(dim, dim))
            self.connector = nn.Linear(dim, dim)
        else:
            self.norm4 = None
            self.attn4 = None


        if self.with_seg:
            if self.use_attn_ctrl:
                self.norm_seg = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
                self.attn_seg = Attention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    out_bias=attention_out_bias,
                    attention_mode=attention_mode, 
                    downsampler=False, 
                    use_rope=False, 
                    interpolation_scale_thw=interpolation_scale_thw, 
                )  # is self-attn if encoder_hidden_states is none
            else:
                self.seg_proj = nn.Linear(dim, dim)
                self.norm_seg = None
                self.attn_seg = None
        else:
            self.norm_seg = None
            self.attn_seg = None

        if self.with_depth:
            if self.use_attn_ctrl:
                self.norm_depth = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
                self.attn_depth = Attention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    out_bias=attention_out_bias,
                    attention_mode=attention_mode, 
                    downsampler=False, 
                    use_rope=False, 
                    interpolation_scale_thw=interpolation_scale_thw, 
                )  # is self-attn if encoder_hidden_states is none
            else:
                self.depth_proj =  nn.Linear(dim, dim) 
                # self.depth_proj = zero_module( nn.Linear(dim, dim) )
                self.norm_depth = None
                self.attn_depth = None
        else:
            self.norm_depth = None
            self.attn_depth = None


        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        frame: int = None, 
        height: int = None, 
        width: int = None, 
        neighboring_view_pair: dict = None,
        raymap_embedding: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        seg_map: Optional[torch.LongTensor] = None,
        depth_map: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        if self.camera_pose_proj is not None and raymap_embedding is not None:
            camera_pose_embed = self.camera_pose_proj(raymap_embedding)

        if self.with_seg and not self.use_attn_ctrl and seg_map is not None:
            seg_embed = self.seg_proj(seg_map)

        if self.with_depth and not self.use_attn_ctrl and depth_map is not None:
            depth_embed = self.depth_proj(depth_map)

        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            # import ipdb;ipdb.set_trace()
            if get_sequence_parallel_state():
                batch_size = hidden_states.shape[1]
                # print('hidden_states', hidden_states.shape)
                # print('timestep', timestep.shape)
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                        self.scale_shift_table[:, None] + timestep.reshape(6, batch_size, -1)
                ).chunk(6, dim=0)
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                        self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            # norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)
        if self.camera_pose_proj is not None:
            norm_hidden_states = norm_hidden_states+camera_pose_embed
        
        if self.with_seg and not self.use_attn_ctrl and seg_map is not None:
            norm_hidden_states = norm_hidden_states+seg_embed

        if self.with_depth and not self.use_attn_ctrl and depth_map is not None:
            norm_hidden_states = norm_hidden_states+depth_embed
        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask, frame=frame, height=height, width=width, 
            **cross_attention_kwargs,
        )

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])


        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)
            if self.camera_pose_proj is not None:
                norm_hidden_states = norm_hidden_states+camera_pose_embed
            if self.with_seg and not self.use_attn_ctrl and seg_map is not None:
                norm_hidden_states = norm_hidden_states+seg_embed

            if self.with_depth and not self.use_attn_ctrl and depth_map is not None:
                norm_hidden_states = norm_hidden_states+depth_embed
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # Seg-Cross-Attention
        if self.attn_seg is not None and self.with_seg and self.use_attn_ctrl:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm_seg(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm_seg(hidden_states)
            elif self.norm_type == "ada_norm_single":
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm_seg(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            if self.camera_pose_proj is not None:
                norm_hidden_states = norm_hidden_states+camera_pose_embed

            attn_output = self.attn_seg(
                norm_hidden_states,
                encoder_hidden_states=seg_map,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states


        # Depth-Cross-Attention
        if self.attn_depth is not None and self.with_depth and self.use_attn_ctrl:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm_depth(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm_depth(hidden_states)
            elif self.norm_type == "ada_norm_single":
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm_depth(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            if self.camera_pose_proj is not None:
                norm_hidden_states = norm_hidden_states+camera_pose_embed

            attn_output = self.attn_depth(
                norm_hidden_states,
                encoder_hidden_states=depth_map,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states


        # 3. Cross-View Attention
        if self.attn4 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm4(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm4(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm4(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")
            
            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)
            # if self.camera_pose_proj is not None:
            #     norm_hidden_states = norm_hidden_states+camera_pose_embed
            if self.camera_pose_proj is not None:
                norm_hidden_states = norm_hidden_states+camera_pose_embed
            if self.with_seg and not self.use_attn_ctrl and seg_map is not None:
                norm_hidden_states = norm_hidden_states+seg_embed
    
            if self.with_depth and not self.use_attn_ctrl and depth_map is not None:
                norm_hidden_states = norm_hidden_states+depth_embed
            mv_norm_hidden_states = rearrange(norm_hidden_states, '(b n) ... -> b n ...', n=self.n_cam)
            B = len(mv_norm_hidden_states)
            hidden_states_in1, hidden_states_in2, cam_order = self._construct_attn_input(neighboring_view_pair, B, mv_norm_hidden_states)

            mv_attn_raw_output = self.attn4(
                hidden_states_in1,
                encoder_hidden_states=hidden_states_in2,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )

            mv_attn_output = torch.zeros_like(mv_norm_hidden_states)
            for cam_i in range(self.n_cam):
                attn_out_mv = rearrange(mv_attn_raw_output[cam_order == cam_i], '(n b) ... -> b n ...', b=B)
                mv_attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
            mv_attn_output = rearrange(mv_attn_output, 'b n ... -> (b n) ...')
            mv_attn_output = self.connector(mv_attn_output)
            hidden_states = mv_attn_output + hidden_states


        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

    def _construct_attn_input(self, neighboring_view_pair, batch_size, norm_hidden_states):
        hidden_states_in1 = []
        hidden_states_in2 = []
        cam_order = []
        for key, values in neighboring_view_pair.items():
            for value in values:
                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(norm_hidden_states[:, value])
                cam_order += [key] * batch_size
        
        hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
        hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
        cam_order = torch.LongTensor(cam_order)
        return hidden_states_in1, hidden_states_in2, cam_order




# @maybe_allow_in_graph
# class DIT3DBlock(nn.Module):
#     r"""
#     A basic DIT3D block.
#     """
#     def __init__(
#         self,
#         dim: int,
#         num_attention_heads: int,
#         attention_head_dim: int,
#         dropout=0.0,
#         cross_attention_dim: Optional[int] = None,
#         activation_fn: str = "geglu",
#         attention_bias: bool = False,
#         norm_type: str = "layer_norm",
#         norm_eps: float = 1e-5,
#         final_dropout: bool = False,
#         attention_type: str = "default",
#         ff_inner_dim: Optional[int] = None,
#         ff_bias: bool = True,
#         attention_out_bias: bool = True,
#         attention_mode: str = "xformers", 
#         use_rope: bool = False, 
#         interpolation_scale_thw: Tuple[int] = (1, 1, 1), 
#         n_cam=6,
#         camera_control=False,
#         with_seg=False,
#         with_depth=False
#     ):