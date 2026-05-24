from typing import Literal, Optional, Set, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

try:
    import tcnn
except:
    TCNN_EXISTS = False

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
        init_bias = None,
        implementation: Literal["tcnn", "torch"] = "torch",
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
            #print_tcnn_speed_warning("MLP")
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

        if init_bias is not None:
            self.layers[-1].bias.data.fill_(init_bias)

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

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
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

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)