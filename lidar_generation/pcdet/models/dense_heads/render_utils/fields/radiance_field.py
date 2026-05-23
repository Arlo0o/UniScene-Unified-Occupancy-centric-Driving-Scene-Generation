import torch
import torch.nn.functional as F
from torch import nn
from pcdet.ops.smooth_sampler import SmoothSampler, grid_sample_3d
from .encoding import SinusoidalEncoder
from typing import Optional, Set, Tuple, Union
from torch import FloatTensor, Tensor
from torch.cuda.amp import custom_bwd, custom_fwd


class _TruncExp(torch.autograd.Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply


def activation_to_tcnn_string(activation: Union[nn.Module, None]) -> str:
    """Converts a torch.nn activation function to a string that can be used to
    initialize a TCNN activation function.

    Args:
        activation: torch.nn activation function
    Returns:
        str: TCNN activation function string
    """

    if isinstance(activation, nn.ReLU):
        return "ReLU"
    if isinstance(activation, nn.LeakyReLU):
        return "Leaky ReLU"
    if isinstance(activation, nn.Sigmoid):
        return "Sigmoid"
    if isinstance(activation, nn.Softplus):
        return "Softplus"
    if isinstance(activation, nn.Tanh):
        return "Tanh"
    if isinstance(activation, type(None)):
        return "None"
    tcnn_documentation_url = "https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#activation-functions"
    raise ValueError(
        f"TCNN activation {activation} not supported for now.\nSee {tcnn_documentation_url} for TCNN documentation."
    )


class MLP(nn.Module):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        implementation = "torch",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None

        self.tcnn_encoding = None
        if implementation == "torch":
            self.build_nn_modules()
        elif implementation == "tcnn" and not TCNN_EXISTS:
            print_tcnn_speed_warning("MLP")
            self.build_nn_modules()
        elif implementation == "tcnn":
            network_config = self.get_tcnn_network_config(
                activation=self.activation,
                out_activation=self.out_activation,
                layer_width=self.layer_width,
                num_layers=self.num_layers,
            )
            self.tcnn_encoding = tcnn.Network(
                n_input_dims=in_dim,
                n_output_dims=self.out_dim,
                network_config=network_config,
            )

    @classmethod
    def get_tcnn_network_config(cls, activation, out_activation, layer_width, num_layers) -> dict:
        """Get the network configuration for tcnn if implemented"""
        activation_str = activation_to_tcnn_string(activation)
        output_activation_str = activation_to_tcnn_string(out_activation)
        if layer_width in [16, 32, 64, 128]:
            network_config = {
                "otype": "FullyFusedMLP",
                "activation": activation_str,
                "output_activation": output_activation_str,
                "n_neurons": layer_width,
                "n_hidden_layers": num_layers - 1,
            }
        else:
            CONSOLE.line()
            CONSOLE.print("[bold yellow]WARNING: Using slower TCNN CutlassMLP instead of TCNN FullyFusedMLP")
            CONSOLE.print("[bold yellow]Use layer width of 16, 32, 64, or 128 to use the faster TCNN FullyFusedMLP.")
            CONSOLE.line()
            network_config = {
                "otype": "CutlassMLP",
                "activation": activation_str,
                "output_activation": output_activation_str,
                "n_neurons": layer_width,
                "n_hidden_layers": num_layers - 1,
            }
        return network_config

    def build_nn_modules(self) -> None:
        """Initialize the torch version of the multi-layer perceptron."""
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    layers.append(nn.Linear(self.in_dim, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(nn.Linear(self.layer_width + self.in_dim, self.layer_width))
                else:
                    layers.append(nn.Linear(self.layer_width, self.layer_width))
            layers.append(nn.Linear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)

    def pytorch_fwd(self, in_tensor):
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

    def forward(self, in_tensor):
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)



class RGBDecoder(nn.Module):
    def __init__(self, in_dim, out_dim=3, hidden_size=256, n_blocks=5):
        super().__init__()

        dims = [hidden_size] + [hidden_size for _ in range(n_blocks)] + [out_dim]
        self.num_layers = len(dims)

        for l in range(self.num_layers - 1):
            lin = nn.Linear(dims[l], dims[l + 1])
            setattr(self, "lin" + str(l), lin)

        self.fc_c = nn.ModuleList(
            [nn.Linear(in_dim, hidden_size) for i in range(self.num_layers - 1)]
        )
        self.fc_p = nn.Linear(3, hidden_size)

        self.activation = nn.ReLU()

    def forward(self, points, point_feats):
        x = self.fc_p(points)
        for l in range(self.num_layers - 1):
            x = x + self.fc_c[l](point_feats)
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        x = torch.sigmoid(x)
        return x


class BasicField(nn.Module):
    def __init__(
        self,
        voxel_size,
        pc_range,
        voxel_shape,
        scale_factor,
        rgb_decoder_cfg,
        mlp_geo_cfg,
        mlp_feature_cfg,
        interpolate_cfg,
        beta_init,
        **kwargs
    ):
        super().__init__()
        self.fp16_enabled = kwargs.get("fp16_enabled", False)
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.voxel_shape = voxel_shape
        self.beta_init = beta_init
        self.interpolate_cfg = interpolate_cfg
        self.scale_factor = scale_factor
        self.rgb_decoder = RGBDecoder(**rgb_decoder_cfg)

        self._cos_anneal_ratio = 1.0
        self.geo_feat_dim = mlp_geo_cfg.out_dim
        self.direction_encoding = SinusoidalEncoder(n_input_dims=3, min_deg=0, max_deg=4)
        direction_dim = self.direction_encoding.n_output_dims
        self.mlp_geo = MLP(
            in_dim=mlp_geo_cfg.in_dim,
            num_layers=mlp_geo_cfg.num_layers,
            layer_width=mlp_geo_cfg.hidden_size,
            out_dim=self.geo_feat_dim + 1,
            activation=nn.ReLU(),
            out_activation=None,
            implementation='torch',
        )
        # import ipdb; ipdb.set_trace()
        self.mlp_feature = MLP(
            in_dim=direction_dim + self.geo_feat_dim,
            num_layers=mlp_feature_cfg.num_layers,
            layer_width=mlp_feature_cfg.hidden_size,
            out_dim=mlp_feature_cfg.out_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation='torch',
        )
        self.density_activation = lambda x: trunc_exp(x - 1)


    def set_cos_anneal_ratio(self, anneal):
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    def interpolate_feats(self, pts, feats_volume):
        pc_range = pts.new_tensor(self.pc_range)
        norm_coords = (pts / self.scale_factor - pc_range[:3]) / (
            pc_range[3:] - pc_range[:3]
        )
        assert (
            self.voxel_shape[0] == feats_volume.shape[3]
            and self.voxel_shape[1] == feats_volume.shape[2]
            and self.voxel_shape[2] == feats_volume.shape[1]
        )
        norm_coords = norm_coords * 2 - 1
        if self.interpolate_cfg["type"] == "SmoothSampler":
            feats = (
                SmoothSampler.apply(
                    feats_volume.unsqueeze(0),
                    norm_coords[None, None, ...],
                    self.interpolate_cfg["padding_mode"],
                    True,
                    False,
                )
                .squeeze(0)
                .squeeze(1)
                .permute(1, 2, 0)
            )
        else:
            feats = (
                grid_sample_3d(feats_volume.unsqueeze(0), norm_coords[None, None, ...])
                .squeeze(0)
                .squeeze(1)
                .permute(1, 2, 0)
            )
        return feats

    def forward(self, ray_samples, feature_volume, return_alphas=False):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        outputs = {}

        points = ray_samples.frustums.get_start_positions()
        if feature_volume.dtype == torch.float16:
            points = points.to(torch.float16)
        features = self.interpolate_feats(points, feature_volume)
        directions = ray_samples.frustums.directions

        # First output is the geometry, second is the pass-along vector.
        # import ipdb; ipdb.set_trace()
        sample_shape = points.shape[:-1]
        geo_out, geo_embedding = torch.split(self.mlp_geo(features), [1, self.geo_feat_dim], dim=-1)
        geo_out = geo_out.view(*sample_shape, 1)
        density = geo_out
        geo_embedding = geo_embedding.reshape(-1, geo_embedding.shape[-1])
        
        direction_embedding = self.direction_encoding(directions.reshape(-1, 3))
        # import ipdb; ipdb.set_trace()
        feature = geo_embedding + self.mlp_feature(torch.cat([geo_embedding, direction_embedding], dim=-1))
        feature = feature.view(*sample_shape, geo_embedding.shape[-1])
        outputs.update(
            {
                "rgb": feature,
                "density": density,
            }
        )

        return outputs