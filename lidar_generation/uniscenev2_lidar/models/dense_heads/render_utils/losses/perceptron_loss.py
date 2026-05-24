from enum import Enum
from typing import Dict, Literal, Optional, Tuple, cast

import torch
import torchvision
from torch import Tensor, nn

EPS = 1.0e-7


class VGGPerceptualLossPix2Pix(nn.Module):
    """From https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py"""

    def __init__(self, weights=None):
        super().__init__()
        self.vgg = Vgg19().eval()
        self.vgg.requires_grad_(False)
        self.criterion = nn.L1Loss()
        if weights is None:
            self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        else:
            assert len(weights) == 5, "Expected 5 weights for VGGPerceptualLossPix2Pix"
            self.weights = weights

    def forward(self, x, y):
        """
        Args:
            x: Input tensor of shape (N, C, H, W)
            y: Ground truth tensor of shape (N, C, H, W)
        Returns:
            Tensor containing the perceptual loss
        """
        assert x.shape == y.shape, "Input and ground truth tensors must have the same shape"
        assert x.ndim == 4, "Input and ground truth tensors must have 4 dimensions"
        
        # Assume input is in NCHW format
        vgg_out = self.vgg(torch.cat([x, y], dim=0))
        # x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(vgg_out)):
            x_vgg, y_vgg = vgg_out[i].chunk(2, dim=0)
            loss += self.weights[i] * self.criterion(x_vgg, y_vgg.detach())
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential(vgg_pretrained_features[:2])
        self.slice2 = torch.nn.Sequential(vgg_pretrained_features[2:7])
        self.slice3 = torch.nn.Sequential(vgg_pretrained_features[7:12])
        self.slice4 = torch.nn.Sequential(vgg_pretrained_features[12:21])
        self.slice5 = torch.nn.Sequential(vgg_pretrained_features[21:30])

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


def ray_samples_to_sdist(ray_samples):
    """Convert ray samples to s space"""
    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends
    sdist = torch.cat([starts[..., 0], ends[..., -1:, 0]], dim=-1)  # (num_rays, num_samples + 1)
    return sdist


def depth_ranking_loss(rendered_depth, gt_depth):
    """
    Depth ranking loss as described in the SparseNeRF paper
    Assumes that the layout of the batch comes from a PairPixelSampler, so that adjacent samples in the gt_depth
    and rendered_depth are from pixels with a radius of each other
    """
    m = 1e-4
    if rendered_depth.shape[0] % 2 != 0:
        # chop off one index
        rendered_depth = rendered_depth[:-1, :]
        gt_depth = gt_depth[:-1, :]
    dpt_diff = gt_depth[::2, :] - gt_depth[1::2, :]
    out_diff = rendered_depth[::2, :] - rendered_depth[1::2, :] + m
    differing_signs = torch.sign(dpt_diff) != torch.sign(out_diff)
    return torch.nanmean((out_diff[differing_signs] * torch.sign(out_diff[differing_signs])))


def _blur_stepfun(x: torch.Tensor, y: torch.Tensor, r: float) -> Tuple[torch.Tensor, torch.Tensor]:
    xr, xr_idx = torch.sort(torch.cat([x - r, x + r], dim=-1))
    y1 = (
        torch.cat([y, torch.zeros_like(y[..., :1])], dim=-1) - torch.cat([torch.zeros_like(y[..., :1]), y], dim=-1)
    ) / (2 * r)
    y2 = torch.cat([y1, -y1], dim=-1).take_along_dim(xr_idx[..., :-1], dim=-1)
    yr = torch.cumsum((xr[..., 1:] - xr[..., :-1]) * torch.cumsum(y2, dim=-1), dim=-1).clamp_min(0)
    yr = torch.cat([torch.zeros_like(yr[..., :1]), yr], dim=-1)
    return xr, yr


def _sorted_interp_quad(x, xp, fpdf, fcdf):
    right_idx = torch.searchsorted(xp, x)
    left_idx = (right_idx - 1).clamp_min(0)
    right_idx = right_idx.clamp_max(xp.shape[-1] - 1)

    xp0 = xp.take_along_dim(left_idx, dim=-1)
    xp1 = xp.take_along_dim(right_idx, dim=-1)
    fpdf0 = fpdf.take_along_dim(left_idx, dim=-1)
    fpdf1 = fpdf.take_along_dim(right_idx, dim=-1)
    fcdf0 = fcdf.take_along_dim(left_idx, dim=-1)

    offset = torch.clip(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    return fcdf0 + (x - xp0) * (fpdf0 + fpdf1 * offset + fpdf0 * (1 - offset)) * 0.5


def zipnerf_interlevel_loss(weights_list, ray_samples_list):
    """Anti-aliased interlevel loss proposed in ZipNeRF paper."""
    # ground truth s and w (real nerf samples)
    # This implementation matches ZipNeRF up to the scale.
    # In the paper the loss is computed as the sum over the ray samples.
    # Here we take the mean and the multiplier for this loss should be changed accordingly.
    pulse_widths = [0.03, 0.003]
    c = ray_samples_to_sdist(ray_samples_list[-1]).detach()
    w = weights_list[-1][..., 0].detach()
    accum_w = torch.sum(w, dim=-1, keepdim=True)
    w = torch.cat([w[..., :-1], w[..., -1:] + (1 - accum_w)], dim=-1)

    w_norm = w / (c[..., 1:] - c[..., :-1])
    loss = 0
    for i, (ray_samples, weights) in enumerate(zip(ray_samples_list[:-1], weights_list[:-1])):
        cp = ray_samples_to_sdist(ray_samples)
        wp = weights[..., 0]  # (num_rays, num_samples)
        c_, w_ = _blur_stepfun(c, w_norm, pulse_widths[i])

        # piecewise linear pdf to piecewise quadratic cdf
        area = 0.5 * (w_[..., 1:] + w_[..., :-1]) * (c_[..., 1:] - c_[..., :-1])
        cdf = torch.cat([torch.zeros_like(area[..., :1]), torch.cumsum(area, dim=-1)], dim=-1)

        # prepend 0 weight and append 1 weight
        c_ = torch.cat([torch.zeros_like(c_[..., :1]), c_, torch.ones_like(c_[..., :1])], dim=-1)
        w_ = torch.cat([torch.zeros_like(w_[..., :1]), w_, torch.zeros_like(w_[..., :1])], dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf, torch.ones_like(cdf[..., :1])], dim=-1)

        # query piecewise quadratic interpolation
        cdf_interp = _sorted_interp_quad(cp, c_, w_, cdf)

        # difference between adjacent interpolated values
        w_s = torch.diff(cdf_interp, dim=-1)
        loss += ((w_s - wp).clamp_min(0) ** 2 / (wp + 1e-5)).sum(dim=-1).mean()
    return loss
