import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint as cp
import numpy as np
import math
from uniscenev2_lidar.ops.smooth_sampler import SmoothSampler, grid_sample_3d, bool_sample, padding_mode_enum
try:
    import MinkowskiEngine as ME
except:
    pass
from .encoding import plucker_embedding, RayHistEmbedding, PointInRayEmbedding
from .token_interaction import TokenMLPMixer, SDPASelfAttention

class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter(
            "beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False)
        )
        self.register_parameter(
            "beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True)
        )

    def forward(self, sdf, beta=None):
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SingleVarianceNetwork(nn.Module):
    """Variance network in NeuS"""

    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter(
            "variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True)
        )

    def forward(self, x):
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(
            self.variance * 10.0
        )

    def get_variance(self):
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


class SDFDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size=256, n_blocks=5):
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

        self.activation = nn.Softplus(beta=100)

    def forward(self, points, point_feats, checkpointing=False):
        x = self.fc_p(points)
        for l in range(self.num_layers - 1):
            x = x + self.fc_c[l](point_feats)
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        return x


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


class SDFField(nn.Module):
    def __init__(
        self,
        voxel_size,
        pc_range,
        voxel_shape,
        scale_factor,
        sdf_decoder_cfg,
        interpolate_cfg,
        beta_init,
        rgb_decoder_cfg=None,
        use_checkpointing=False,
        cat_plucker_embd=False,
        cylindrical_voxelize=False,

        calc_ray_hist_embd=False,
        cat_ray_hist_embd=False,
        ray_hist_embedding_cfg=dict(),
        cat_point_pos_embd=False,
        inner_ray_interaction_cfg=dict(enabled=False),
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
        self.sdf_decoder = SDFDecoder(**sdf_decoder_cfg)
        if rgb_decoder_cfg is not None:
            self.rgb_decoder = RGBDecoder(**rgb_decoder_cfg)
        else:
            self.rgb_decoder = None

        # laplace function for transform sdf to density from VolSDF
        self.laplace_density = LaplaceDensity(init_val=self.beta_init)

        # deviation_network to compute alpha from sdf from NeuS
        self.deviation_network = SingleVarianceNetwork(init_val=self.beta_init)

        self._cos_anneal_ratio = 1.0
        self.use_checkpointing = use_checkpointing
        self.cat_plucker_embd = cat_plucker_embd
        self.cat_ray_hist_embd = cat_ray_hist_embd
        self.cat_point_pos_embd = cat_point_pos_embd
        self.calc_ray_hist_embd = calc_ray_hist_embd
        if self.calc_ray_hist_embd:
            close_radius = ray_hist_embedding_cfg.pop('close_radius')
            far_radius = ray_hist_embedding_cfg.pop('far_radius')
            ray_hist_embedding_cfg['vmin'] = 0.0
            ray_hist_embedding_cfg['vmax'] = far_radius * self.scale_factor
            self.ray_hist_embedding = RayHistEmbedding(**ray_hist_embedding_cfg)

        self.inner_ray_interaction_cfg = inner_ray_interaction_cfg
        self.inner_ray_interaction = None
        if inner_ray_interaction_cfg['enabled']:
            _type = inner_ray_interaction_cfg['type']
            L = inner_ray_interaction_cfg['L']
            C = inner_ray_interaction_cfg['C']
            if _type == 'mlp_mixer':
                self.inner_ray_interaction = TokenMLPMixer(L=L, C=C)
            elif _type == 'selfattn':
                self.inner_ray_interaction = SDPASelfAttention(dim=C, num_heads=inner_ray_interaction_cfg.get('num_heads', 8))
            else:
                raise NotImplementedError
        self.cylindrical_voxelize = cylindrical_voxelize

    def set_cos_anneal_ratio(self, anneal):
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    def get_alpha(self, ray_samples, sdf, gradients):
        inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self._cos_anneal_ratio)
            + F.relu(-true_cos) * self._cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        return alpha

    def interpolate_feats(self, pts, feats_volume, use_bool_sample=False):
        pc_range = pts.new_tensor(self.pc_range)
        voxel_size = pts.new_tensor(self.voxel_size)
        # sparse interpolate
        if isinstance(feats_volume, dict):
            sparse_tensor = ME.SparseTensor(features=feats_volume['features'].float(), coordinates=feats_volume['indices'])
            interp_coords = (pts / self.scale_factor - pc_range[:3]) / voxel_size
            interp_coords = interp_coords.flip(-1) # xyz -> zyx
            *temp, _ = interp_coords.shape
            interp_coords = interp_coords.view(-1, 3)
            batch_indices = interp_coords.new_zeros((interp_coords.shape[0], 1)) + feats_volume['indices'][0,0]
            interp_coords = torch.cat([batch_indices, interp_coords], dim=-1)
            interpolator = ME.MinkowskiInterpolation()
            feats = interpolator(sparse_tensor, interp_coords)
            feats = feats.view(*temp, -1)
            return feats
        
        if not self.cylindrical_voxelize:
            norm_coords = (pts / self.scale_factor - pc_range[:3]) / (
                pc_range[3:] - pc_range[:3]
            )
        else:
            # to cylindrical first
            pts_unnorm = pts / self.scale_factor
            rho = torch.sqrt(pts_unnorm[..., 0]**2 + pts_unnorm[..., 1]**2)
            phi = torch.atan2(pts_unnorm[..., 1], pts_unnorm[..., 0])
            pts_cyl = torch.stack([rho, phi, pts_unnorm[..., 2]], dim=-1)
            polar_range = [0, -np.pi, self.pc_range[2], math.sqrt(((self.pc_range[3]-self.pc_range[0])/2)**2+ \
                                ((self.pc_range[4]-self.pc_range[1])/2)**2), np.pi, self.pc_range[-1]]
            polar_range = pts.new_tensor(polar_range)
            # grid_size = feats_volume.shape[[-1, -2, -3]]
            # polar_size = [(polar_range[3]-polar_range[0])/grid_size[0], (polar_range[4]-polar_range[1])/grid_size[1], (polar_range[5]-polar_range[2])/grid_size[2]]
            # polar_size = pts.new_tensor(polar_size)

            norm_coords = (pts_cyl - polar_range[:3]) / (
                polar_range[3:] - polar_range[:3]
            )

        assert (
            self.voxel_shape[0] == feats_volume.shape[-1]
            and self.voxel_shape[1] == feats_volume.shape[-2]
            and self.voxel_shape[2] == feats_volume.shape[-3]
        )
        norm_coords = norm_coords * 2 - 1
        interpolate_type = self.interpolate_cfg["type"]
        interpolate_type = 'bool_sample' if use_bool_sample else interpolate_type
        if interpolate_type == "SmoothSampler":
            feats = (
                SmoothSampler.apply(
                    feats_volume.unsqueeze(0) if feats_volume.ndim == 4 else feats_volume,
                    norm_coords[None, None, ...] if norm_coords.ndim == 3 else norm_coords.unsqueeze(1),
                    self.interpolate_cfg["padding_mode"],
                    True,
                    False,
                )
            )
        elif interpolate_type == "bool_sample":
            assert feats_volume.dtype == torch.bool
            feats = (
                bool_sample(
                    feats_volume.unsqueeze(0) if feats_volume.ndim == 4 else feats_volume,
                    norm_coords[None, None, ...] if norm_coords.ndim == 3 else norm_coords.unsqueeze(1),
                    padding_mode_enum(self.interpolate_cfg["padding_mode"]),
                    True,
                    False,
                )
            )
        else:
            feats = (
                grid_sample_3d(
                    feats_volume.unsqueeze(0) if feats_volume.ndim == 4 else feats_volume,
                    norm_coords[None, None, ...] if norm_coords.ndim == 3 else norm_coords.unsqueeze(1)
                )
            )
        if feats_volume.ndim == 4:
            feats = feats.squeeze(0).squeeze(1).permute(1, 2, 0)
        else:
            feats = feats.squeeze(2).permute(0, 2, 3, 1)
        return feats

    def get_sdf(self, points, feature_volume, ray_samples=None, ray_bundle=None, checkpointing=False):
        """predict the sdf value for ray samples"""
        with torch.cuda.amp.autocast(enabled=False):
            point_features = self.interpolate_feats(points.float(), feature_volume.float() if isinstance(feature_volume, torch.Tensor) else feature_volume)
        if self.cat_plucker_embd:
            assert ray_samples is not None
            emb = plucker_embedding(ray_samples.frustums.origins, ray_samples.frustums.directions) # [n_rays, n_samples, 6]
            point_features = torch.cat([point_features, emb], dim=-1)

        if self.calc_ray_hist_embd:
            assert ray_samples is not None
            ray_hist, ray_hist_features = self.ray_hist_embedding(ray_samples) # [n_rays, C]
            if self.cat_ray_hist_embd:
                point_features = torch.cat([point_features, ray_hist_features.unsqueeze(1).expand(-1, point_features.shape[1], -1)], dim=-1)
            if self.cat_point_pos_embd:
                point_pos_encoder = PointInRayEmbedding(emb_type='fourier', n_feats=16)
                point_pos_embd = point_pos_encoder(ray_samples.frustums.starts - ray_bundle.nears.unsqueeze(1) / ray_bundle.fars.unsqueeze(1))
                point_features = torch.cat([point_features, point_pos_embd], dim=-1)
        else:
            ray_hist, ray_hist_features = None, None

        if self.inner_ray_interaction_cfg['enabled']:
            point_features = self.inner_ray_interaction(point_features)

        assert not checkpointing, "checkpointing is not supported, as NeuS needs grad()"
        h = self.sdf_decoder(points, point_features, checkpointing=checkpointing)
        sdf, geo_features = h[..., :1], h[..., 1:]
        # return sdf, geo_features, point_features
        sdf_outputs = dict(
            sdf=sdf,
            geo_features=geo_features,
            point_features=point_features,
            ray_hist=ray_hist,
            ray_hist_features=ray_hist_features
        )
        return sdf_outputs

    def get_density(self, ray_samples, feature_volume):
        """Computes and returns the densities."""
        points = ray_samples.frustums.get_start_positions()
        sdf, _, _ = self.get_sdf(points, feature_volume)
        density = self.laplace_density(sdf)
        return density

    def get_occupancy(self, sdf):
        """compute occupancy as in UniSurf"""
        occupancy = torch.sigmoid(-10.0 * sdf)
        return occupancy

    def forward(self, ray_samples, feature_volume, return_alphas=False, ray_bundle=None):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        outputs = {}

        points = ray_samples.frustums.get_start_positions()
        points.requires_grad_(True)
        with torch.enable_grad():
            sdf_output = self.get_sdf(points, feature_volume, checkpointing=self.use_checkpointing, ray_samples=ray_samples, ray_bundle=ray_bundle)
            sdf, geo_features, point_features = sdf_output['sdf'], sdf_output['geo_features'], sdf_output['point_features']

        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        directions = ray_samples.frustums.directions
        if self.rgb_decoder is not None:
            rgb = self.rgb_decoder(
                points,
                torch.cat([directions, point_features, gradients, geo_features], dim=-1),
            )
        else:
            rgb = None

        density = self.laplace_density(sdf)

        outputs.update(
            {
                "rgb": rgb,
                "density": density,
                "sdf": sdf,
                "gradients": gradients,
                "point_features": point_features,
                "ray_feats": dict((k, v) for k, v in sdf_output.items() if 'ray' in k)
            }
        )

        if return_alphas:
            # TODO use mid point sdf for NeuS
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({"alphas": alphas})

        return outputs
