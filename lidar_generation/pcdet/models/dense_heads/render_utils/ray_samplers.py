from abc import abstractmethod
import torch
from torch import nn
import numpy as np

class Sampler(nn.Module):
    """Generate Samples

    Args:
        num_samples: number of samples to take
    """

    def __init__(self, num_samples=None):
        super().__init__()
        self.num_samples = num_samples

    @abstractmethod
    def generate_ray_samples(self):
        """Generate Ray Samples"""

    def forward(self, *args, **kwargs):
        """Generate ray samples"""
        return self.generate_ray_samples(*args, **kwargs)

class SpacedSampler(Sampler):
    """Sample points according to a function.

    Args:
        num_samples: Number of samples per ray
        spacing_fn: Function that dictates sample spacing (ie `lambda x : x` is uniform).
        spacing_fn_inv: The inverse of spacing_fn.
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        spacing_fn,
        spacing_fn_inv,
        num_samples=None,
        train_stratified=True,
        single_jitter=False,
    ):
        super().__init__(num_samples=num_samples)
        self.train_stratified = train_stratified
        self.single_jitter = single_jitter
        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv

    def generate_ray_samples(self, ray_bundle, num_samples=None, **kwargs):
        """Generates position samples accoring to spacing function.

        Args:
            ray_bundle: Rays to generate samples for
            num_samples: Number of samples per ray

        Returns:
            Positions and deltas for samples along a ray
        """
        assert ray_bundle is not None
        assert ray_bundle.nears is not None
        assert ray_bundle.fars is not None

        num_samples = num_samples or self.num_samples
        num_rays = ray_bundle.origins.shape[-2]

        bins = (
            torch.linspace(0.0, 1.0, num_samples + 1)
            .to(ray_bundle.origins.device)
            .expand(size=(*ray_bundle.origins.shape[:-1], -1))
        )

        if self.train_stratified and self.training:
            if self.single_jitter:
                t_rand = torch.rand((*ray_bundle.origins.shape[:-1], 1), dtype=bins.dtype, device=bins.device)
            else:
                t_rand = torch.rand(
                    (*ray_bundle.origins.shape[:-1], num_samples + 1), dtype=bins.dtype, device=bins.device
                )
            bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
            bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
            bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
            bins = bin_lower + (bin_upper - bin_lower) * t_rand

        s_near, s_far = (
            self.spacing_fn(x)
            for x in (ray_bundle.nears.clone(), ray_bundle.fars.clone())
        )
        spacing_to_euclidean_fn = lambda x: self.spacing_fn_inv(
            x * s_far + (1 - x) * s_near
        )
        euclidean_bins = spacing_to_euclidean_fn(bins)

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
        )

        return ray_samples


class UniformSampler(SpacedSampler):
    """Sample uniformly along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(self, num_samples=None, train_stratified=True, single_jitter=False, **kwargs):
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class LinearDisparitySampler(SpacedSampler):
    """Sample linearly in disparity along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(self, num_samples=None, train_stratified=True, single_jitter=False):
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: 1 / x,
            spacing_fn_inv=lambda x: 1 / x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class SqrtSampler(SpacedSampler):
    """Square root sampler along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
    """

    def __init__(self, num_samples=None, train_stratified=True, single_jitter=False):
        super().__init__(
            num_samples=num_samples,
            spacing_fn=torch.sqrt,
            spacing_fn_inv=lambda x: x**2,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class LogSampler(SpacedSampler):
    """Log sampler along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
    """

    def __init__(self, num_samples=None, train_stratified=True, single_jitter=False):
        super().__init__(
            num_samples=num_samples,
            spacing_fn=torch.log,
            spacing_fn_inv=torch.exp,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class UniformLinDispPiecewiseSampler(SpacedSampler):
    """Piecewise sampler along a ray that allocates the first half of the samples uniformly and the second half
    using linearly in disparity spacing.


    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(self, num_samples=None, train_stratified=True, single_jitter=False):
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x)),
            spacing_fn_inv=lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x)),
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class PDFSampler(Sampler):
    """Sample based on probability distribution

    Args:
        num_samples: Number of samples per ray
        train_stratified: Randomize location within each bin during training.
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
        include_original: Add original samples to ray.
        histogram_padding: Amount to weights prior to computing PDF.
    """

    def __init__(self, num_samples=None, train_stratified=True, single_jitter=False):
        super().__init__(num_samples=num_samples)
        self.train_stratified = train_stratified
        self.single_jitter = single_jitter

    def generate_ray_samples(
        self, ray_bundle, ray_samples, weights, num_samples=None, eps=1e-5
    ):
        """Generates position samples given a distribution.

        Args:
            ray_bundle: Rays to generate samples for
            ray_samples: Existing ray samples
            weights: Weights for each bin, (num_rays, num_samples)
            eps: Small value to prevent numerical issues.

        Returns:
            Positions and deltas for samples along a ray
        """
        num_samples = num_samples or self.num_samples
        num_bins = num_samples + 1

        weights = weights[..., 0]
        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(eps - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
        cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
        cdf = torch.cat(
            [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
        )  # (num_rays, num_samples+1)

        if self.train_stratified and self.training:
            # Stratified samples between 0 and 1
            u = torch.linspace(
                0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device
            )
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
            if self.single_jitter:
                rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
            else:
                rand = (
                    torch.rand((*cdf.shape[:-1], num_bins), device=cdf.device)
                    / num_bins
                )
            u = u + rand
        else:
            # Uniform samples between 0 and 1
            u = torch.linspace(
                0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device
            )
            u = u + 1.0 / (2 * num_bins)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
        u = u.contiguous()

        assert (
            ray_samples.spacing_starts is not None
            and ray_samples.spacing_ends is not None
        ), "ray_sample spacing_starts and spacing_ends must be provided"
        assert (
            ray_samples.spacing_to_euclidean_fn is not None
        ), "ray_samples.spacing_to_euclidean_fn must be provided"
        existing_bins = torch.cat(
            [
                ray_samples.spacing_starts[..., 0],
                ray_samples.spacing_ends[..., -1:, 0],
            ],
            dim=-1,
        )

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
        above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
        cdf_g0 = torch.gather(cdf, -1, below)
        bins_g0 = torch.gather(existing_bins, -1, below)
        cdf_g1 = torch.gather(cdf, -1, above)
        bins_g1 = torch.gather(existing_bins, -1, above)

        # t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        denom = cdf_g1 - cdf_g0
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = torch.clip((u - cdf_g0) / denom, 0, 1)
        bins = bins_g0 + t * (bins_g1 - bins_g0)

        # Stop gradients
        bins = bins.detach()

        euclidean_bins = ray_samples.spacing_to_euclidean_fn(bins)

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=ray_samples.spacing_to_euclidean_fn,
        )

        return ray_samples

class PriorSampler(Sampler):
    """Prior sampler that consider the sparse volume prior, but it's not originally sparse, remain to be improved."""

    def __init__(
        self,
        num_samples,
        num_samples_importance=None,
        train_stratified=True,
        single_jitter=True,
        use_bool_sample=False,
        **kwargs
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_samples_importance = 48 if num_samples_importance is None else num_samples_importance
        self.num_upsample_steps = 1
        self.use_bool_sample = use_bool_sample

        # samplers
        self.initial_sampler = UniformSampler(
            num_samples=num_samples,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )
        self.pdf_sampler = PDFSampler(
            train_stratified=train_stratified, single_jitter=single_jitter
        )
    
    def generate_ray_samples(self, ray_bundle, sdf_fn, sdf_field, feature_volume, **kwargs):

        ray_samples = self.initial_sampler(ray_bundle)

        total_iters = 0
        new_samples = ray_samples

        output_dict = {}
        with torch.no_grad():
            new_points = new_samples.frustums.get_start_positions()
            with torch.cuda.amp.autocast(enabled=False):
                if self.use_bool_sample:
                    point_features = sdf_field.interpolate_feats(new_points.float(), ray_bundle.grid, use_bool_sample=True)

                    weights = point_features.float()
                else:
                    point_features = sdf_field.interpolate_feats(new_points.float(), feature_volume.float() if isinstance(feature_volume, torch.Tensor) else feature_volume)

                    weights = torch.zeros([*new_points.shape[:-1], 1], dtype=new_points.dtype, device=new_points.device)
                    non_empty_points = point_features.any(-1)
                    weights[non_empty_points] = 1.0

        if total_iters == 0:
            output_dict.update(
                {"init_sampled_points": new_points, "init_weights": weights}
            )

        new_samples = self.pdf_sampler(
            ray_bundle,
            ray_samples,
            weights,
            num_samples=self.num_samples_importance // self.num_upsample_steps,
        )

        if output_dict.get("new_sampled_points", None) is None:
            output_dict[
                "new_sampled_points"
            ] = new_samples.frustums.get_start_positions()
        else:
            output_dict["new_sampled_points"] = torch.cat(
                [
                    output_dict["new_sampled_points"],
                    new_samples.frustums.get_start_positions(),
                ],
                dim=1,
            )

        total_iters += 1

        output_dict.update({"ray_samples": new_samples})
        return output_dict["ray_samples"]

class NeuSSampler(Sampler):
    """NeuS sampler that uses a sdf network to generate samples with fixed variance value in each iterations."""

    def __init__(
        self,
        initial_sampler,
        num_samples,
        num_samples_importance,
        num_upsample_steps,
        base_variance=64.0,
        train_stratified=True,
        single_jitter=True,
        num_samples_importance_prior=None,
        **kwargs
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_samples_importance = num_samples_importance
        self.num_upsample_steps = num_upsample_steps
        self.base_variance = base_variance

        # samplers
        self.initial_sampler = eval(initial_sampler)(
            num_samples=num_samples,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
            num_samples_importance=num_samples_importance_prior,
            **kwargs
        )
        self.pdf_sampler = PDFSampler(
            train_stratified=train_stratified, single_jitter=single_jitter
        )

    def generate_ray_samples(self, ray_bundle, sdf_fn, **kwargs):
        # Start with uniform sampling
        ray_samples = self.initial_sampler(ray_bundle, sdf_fn=sdf_fn, **kwargs)

        vis=False
        if vis:
            init_sampled_pts = ray_samples.frustums.get_start_positions()
            init_sampled_pts = init_sampled_pts[:100].view((-1, 3))
            occ = kwargs.get('feature_volume').any(0).nonzero()
            occ = occ.float()
            pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype='float32')
            scale_factor = (1.0 / np.max(np.abs(pc_range)))
            norm_coords = (init_sampled_pts / scale_factor - torch.tensor(pc_range[:3]).cuda()) / (
                torch.tensor(pc_range[3:]).cuda() - torch.tensor(pc_range[:3]).cuda()
            )
            norm_coords = norm_coords * 2 - 1
            grid = norm_coords
            IW, IH, ID = 512, 512, 40
            ix = grid[..., 0]
            iy = grid[..., 1]
            iz = grid[..., 2]

            ix = ((ix + 1) / 2) * (IW - 1)
            iy = ((iy + 1) / 2) * (IH - 1)
            iz = ((iz + 1) / 2) * (ID - 1)

            init_sampled_pts = torch.stack([iz, iy, ix], dim=-1)
            black = torch.zeros((init_sampled_pts.shape[0], 3)).cuda()
            red = torch.zeros((occ.shape[0], 3)).cuda()
            red[:, 0] = 255.0
            torch.concatenate([torch.concatenate([init_sampled_pts, black], dim=-1), torch.concatenate([occ, red], dim=-1)],dim=0).cpu().numpy().astype('float32').tofile('z.bin')
            
        total_iters = 0
        sorted_index = None
        new_samples = ray_samples

        base_variance = self.base_variance
        output_dict = {}
        while total_iters < self.num_upsample_steps:
            with torch.no_grad():
                new_points = new_samples.frustums.get_start_positions()
                new_sdf = sdf_fn(new_points, ray_samples=new_samples, ray_bundle=ray_bundle)# [0]
                new_sdf = new_sdf['sdf']

            # merge sdf predictions
            if sorted_index is not None:
                sdf_merge = torch.cat([sdf.squeeze(-1), new_sdf.squeeze(-1)], -1)
                sdf = torch.gather(sdf_merge, 1, sorted_index).unsqueeze(-1)
            else:
                sdf = new_sdf

            # compute with fix variances
            alphas = self.rendering_sdf_with_fixed_inv_s(
                ray_samples, sdf.squeeze(-1), inv_s=base_variance * 2**total_iters
            )

            weights, _ = ray_samples.get_weights_and_transmittance_from_alphas(
                alphas.unsqueeze(-1)
            )
            weights = torch.cat((weights, torch.zeros_like(weights[..., :1])), dim=-2)

            if total_iters == 0:
                output_dict.update(
                    {"init_sampled_points": new_points, "init_weights": weights}
                )

            new_samples = self.pdf_sampler(
                ray_bundle,
                ray_samples,
                weights,
                num_samples=self.num_samples_importance // self.num_upsample_steps,
            )

            if output_dict.get("new_sampled_points", None) is None:
                output_dict[
                    "new_sampled_points"
                ] = new_samples.frustums.get_start_positions()
            else:
                output_dict["new_sampled_points"] = torch.cat(
                    [
                        output_dict["new_sampled_points"],
                        new_samples.frustums.get_start_positions(),
                    ],
                    dim=1,
                )

            ray_samples, sorted_index = ray_bundle.merge_ray_samples(
                ray_samples, new_samples
            )

            total_iters += 1

        output_dict.update({"ray_samples": ray_samples})
        return output_dict

    def rendering_sdf_with_fixed_inv_s(self, ray_samples, sdf, inv_s):
        """rendering given a fixed inv_s as NeuS"""
        batch_size = ray_samples.deltas.shape[0]
        prev_sdf, next_sdf = sdf[..., :-1], sdf[..., 1:]
        deltas = ray_samples.deltas[..., :-1, 0]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (deltas + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat(
            [torch.zeros([*ray_samples.deltas.shape[:-2], 1], device=sdf.device), cos_val[..., :-1]], dim=-1
        )
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0)

        dist = deltas
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

        return alpha


class ErrorBoundedSampler(Sampler):
    """VolSDF's error bounded sampler that uses a sdf network to generate samples."""

    def __init__(
        self,
        initial_sampler,
        num_samples=64,
        num_samples_eval=128,
        num_samples_extra=32,
        eps=0.1,
        beta_iters=10,
        max_total_iters=5,
        train_stratified=True,
        single_jitter=True,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_samples_eval = num_samples_eval
        self.num_samples_extra = num_samples_extra
        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        # samplers
        self.initial_sampler = eval(initial_sampler)(
            train_stratified=train_stratified, single_jitter=single_jitter
        )
        self.pdf_sampler = PDFSampler(
            train_stratified=train_stratified, single_jitter=single_jitter
        )

    def generate_ray_samples(self, ray_bundle, density_fn, sdf_fn, **kwargs):
        beta0 = density_fn.get_beta().detach()

        # Start with uniform sampling
        ray_samples = self.initial_sampler(
            ray_bundle, num_samples=self.num_samples_eval
        )

        # Get maximum beta from the upper bound (Lemma 2)
        deltas = ray_samples.deltas.squeeze(-1)

        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (
            deltas**2.0
        ).sum(-1)
        beta = torch.sqrt(bound)

        total_iters, not_converge = 0, True
        sorted_index = None
        new_samples = ray_samples

        output_dict = {}
        # Algorithm 1
        while not_converge and total_iters < self.max_total_iters:
            with torch.no_grad():
                new_points = new_samples.frustums.get_start_positions()
                new_sdf = sdf_fn(new_points)[0]

            # merge sdf predictions
            if sorted_index is not None:
                sdf_merge = torch.cat([sdf.squeeze(-1), new_sdf.squeeze(-1)], -1)
                sdf = torch.gather(sdf_merge, 1, sorted_index).unsqueeze(-1)
            else:
                sdf = new_sdf

            # Calculating the bound d* (Theorem 1)
            d_star = self.get_dstar(sdf.squeeze(-1), ray_samples)

            # Updating beta using line search
            beta = self.get_updated_beta(
                beta0, beta, density_fn, sdf.squeeze(-1), d_star, ray_samples
            )

            # Upsample more points
            density = density_fn(sdf.squeeze(-1), beta=beta.unsqueeze(-1))

            weights, transmittance = ray_samples.get_weights_and_transmittance(
                density.unsqueeze(-1)
            )

            if total_iters == 0:
                output_dict.update(
                    {
                        "init_sampled_points": new_points,
                        "init_weights": weights,
                    }
                )

            #  Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0

            if not_converge and total_iters < self.max_total_iters:
                # Sample more points proportional to the current error bound
                deltas = ray_samples.deltas.squeeze(-1)

                error_per_section = (
                    torch.exp(-d_star / beta.unsqueeze(-1))
                    * (deltas**2.0)
                    / (4 * beta.unsqueeze(-1) ** 2)
                )

                error_integral = torch.cumsum(error_per_section, dim=-1)
                weights = (
                    torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0
                ) * transmittance[..., 0]

                new_samples = self.pdf_sampler(
                    ray_bundle,
                    ray_samples,
                    weights.unsqueeze(-1),
                    num_samples=self.num_samples_eval,
                )

                ray_samples, sorted_index = ray_bundle.merge_ray_samples(
                    ray_samples, new_samples
                )

            else:
                # Sample the final sample set to be used in the volume rendering integral
                ray_samples = self.pdf_sampler(
                    ray_bundle, ray_samples, weights, num_samples=self.num_samples
                )
                output_dict[
                    "new_sampled_points"
                ] = ray_samples.frustums.get_start_positions()

        # Add extra samples uniformly
        if self.num_samples_extra > 0:
            ray_samples_uniform = self.initial_sampler(
                ray_bundle, num_samples=self.num_samples_extra
            )
            ray_samples, _ = ray_bundle.merge_ray_samples(
                ray_samples, ray_samples_uniform
            )

        output_dict.update({"ray_samples": ray_samples})
        return output_dict

    def get_dstar(self, sdf, ray_samples):
        """Calculating the bound d* (Theorem 1) from VolSDF"""
        d = sdf
        dists = ray_samples.deltas.squeeze(-1)
        a, b, c = dists[:, :-1], d[:, :-1].abs(), d[:, 1:].abs()
        first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
        second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
        d_star = torch.zeros(
            ray_samples.deltas.shape[0], ray_samples.deltas.shape[1] - 1
        ).to(d)
        d_star[first_cond] = b[first_cond]
        d_star[second_cond] = c[second_cond]
        s = (a + b + c) / 2.0
        area_before_sqrt = s * (s - a) * (s - b) * (s - c)
        mask = ~first_cond & ~second_cond & (b + c - a > 0)
        d_star[mask] = ((2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])).to(d)
        d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

        # padding to make the same shape as ray_samples
        # d_star_left = torch.cat((d_star[:, :1], d_star), dim=-1)
        # d_star_right = torch.cat((d_star, d_star[:, -1:]), dim=-1)
        # d_star = torch.minimum(d_star_left, d_star_right)

        d_star = torch.cat((d_star, d_star[:, -1:]), dim=-1)
        return d_star

    def get_updated_beta(self, beta0, beta, density_fn, sdf, d_star, ray_samples):
        curr_error = self.get_error_bound(beta0, density_fn, sdf, d_star, ray_samples)
        beta[curr_error <= self.eps] = beta0
        beta_min, beta_max = beta0.repeat(ray_samples.deltas.shape[0]), beta
        for j in range(self.beta_iters):
            beta_mid = (beta_min + beta_max) / 2.0
            curr_error = self.get_error_bound(
                beta_mid.unsqueeze(-1), density_fn, sdf, d_star, ray_samples
            )
            beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
            beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
        beta = beta_max
        return beta

    def get_error_bound(self, beta, density_fn, sdf, d_star, ray_samples):
        """Get error bound from VolSDF"""
        densities = density_fn(sdf, beta=beta)

        deltas = ray_samples.deltas.squeeze(-1)
        delta_density = deltas * densities

        integral_estimation = torch.cumsum(delta_density[..., :-1], dim=-1)
        integral_estimation = torch.cat(
            [
                torch.zeros(
                    (*integral_estimation.shape[:1], 1), device=densities.device
                ),
                integral_estimation,
            ],
            dim=-1,
        )

        error_per_section = (
            torch.exp(-d_star / beta) * (deltas**2.0) / (4 * beta**2)
        )
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (
            torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0
        ) * torch.exp(-integral_estimation)

        return bound_opacity.max(-1)[0]


class UniSurfSampler(Sampler):
    """NeuS sampler that uses a sdf network to generate samples with fixed variance value in each iterations."""

    def __init__(
        self,
        initial_sampler,
        num_samples_importance,
        num_marching_steps,
        num_samples_interval,
        delta,
        train_stratified=True,
        single_jitter=True,
    ):
        super().__init__()
        self.num_samples_importance = num_samples_importance
        self.num_marching_steps = num_marching_steps
        self.num_samples_interval = num_samples_interval
        self.delta = delta
        self.single_jitter = single_jitter
        # samplers
        self.initial_sampler = eval(initial_sampler)(
            train_stratified=train_stratified, single_jitter=single_jitter
        )
        self.pdf_sampler = PDFSampler(
            train_stratified=train_stratified, single_jitter=single_jitter
        )

    def generate_ray_samples(self, ray_bundle, occupancy_fn, sdf_fn, **kwargs):
        output_dict = {}
        # Start with uniform sampling
        ray_samples = self.initial_sampler(
            ray_bundle, num_samples=self.num_marching_steps
        )
        points = ray_samples.frustums.get_start_positions()
        with torch.no_grad():
            sdf = sdf_fn(points)[0]

        # importance sampling
        occupancy = occupancy_fn(sdf)
        weights, _ = ray_samples.get_weights_and_transmittance_from_alphas(occupancy)

        output_dict.update(
            {
                "init_sampled_points": ray_samples.frustums.get_start_positions(),
                "init_weights": weights,
            }
        )

        importance_samples = self.pdf_sampler(
            ray_bundle,
            ray_samples,
            weights,
            num_samples=self.num_samples_importance,
        )

        # surface points
        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension
        n_rays, n_samples = ray_samples.deltas.shape[:2]
        starts = ray_samples.frustums.starts
        sign_matrix = torch.cat(
            [
                torch.sign(sdf[:, :-1, 0] * sdf[:, 1:, 0]),
                torch.ones(n_rays, 1).to(sdf.device),
            ],
            dim=-1,
        )
        cost_matrix = sign_matrix * torch.arange(n_samples, 0, -1).float().to(
            sdf.device
        )  # (n_rays, n_samples)

        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0  # (n_rays,)
        mask_pos_to_neg = sdf[torch.arange(n_rays), indices, 0] > 0

        # Define mask where a valid depth value is found
        mask = mask_sign_change & mask_pos_to_neg  # (n_rays,)

        # Get depth values and function values for the interval
        d_low = starts[torch.arange(n_rays), indices, 0][mask]
        v_low = sdf[torch.arange(n_rays), indices, 0][mask]

        indices = torch.clamp(indices + 1, max=n_samples - 1)
        d_high = starts[torch.arange(n_rays), indices, 0][mask]
        v_high = sdf[torch.arange(n_rays), indices, 0][mask]

        # TODO secant method
        # linear-interpolations, estimated depth values
        z = (v_low * d_high - v_high * d_low) / (v_low - v_high)

        # modify near and far values according current schedule
        nears, fars = ray_bundle.nears.clone(), ray_bundle.fars.clone()
        dists = fars - nears

        ray_bundle.nears[mask] = z[:, None] - dists[mask] * self.delta
        ray_bundle.fars[mask] = z[:, None] + dists[mask] * self.delta

        # min max bound
        ray_bundle.nears = torch.maximum(ray_bundle.nears, nears)
        ray_bundle.fars = torch.minimum(ray_bundle.fars, fars)

        # samples uniformly with new surface interval
        ray_samples_interval = self.initial_sampler(
            ray_bundle, num_samples=self.num_samples_interval
        )

        # change back to original values
        ray_bundle.nears = nears
        ray_bundle.fars = fars

        # merge sampled points
        ray_samples = ray_bundle.merge_ray_samples_in_eculidean(
            ray_samples_interval, importance_samples
        )

        output_dict["new_sampled_points"] = ray_samples.frustums.get_start_positions()
        output_dict.update({"ray_samples": ray_samples})
        return output_dict
