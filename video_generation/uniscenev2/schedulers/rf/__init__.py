from functools import partial
from copy import deepcopy

import torch
from tqdm import tqdm
from einops import rearrange, repeat
from uniscenev2.registry import SCHEDULERS
from .rectified_flow import RFlowScheduler, timestep_transform
from uniscenev2.utils.train_utils import default
# from torchvision.io import write_video
import imageio
import os
import numpy as np
import matplotlib
import torch.nn.functional as F
from IPython import embed

def write_video(save_path,img_list,fps=10):
    img_numpy_list = img_list.numpy()
    videoWriter = imageio.get_writer(save_path, fps=fps)
    for idx in range(len(img_numpy_list)):
        videoWriter.append_data(img_numpy_list[idx])
    videoWriter.close()

# color the depth, kitti magma_r, nyu jet
def colorize(ori_value, cmap='magma_r', vmin=0, vmax=1):
    # TODO: remove hacks

    # for abs
    # vmin=1e-3
    # vmax=80

    # for relative
    # value[value<=vmin]=vmin

    # vmin=None
    # vmax=None
    # ori_value[ori_value<vmin] = 0
    # if vmax > 1:
    #     ori_value_depth = ori_value.clamp_(min=vmin, max=vmax).to(torch.float32).cpu().numpy()
    # else:
    ori_value = ori_value.clamp_(min=-1, max=1)
    ori_value_depth = (((ori_value*0.5)+0.5)*vmax).to(torch.float32).cpu().numpy()
    # ori_value_depth[ori_value_depth<0] = 0
    # ori_value_depth = ori_value_depth/(255/vmax)
    # normalize
    vmin = ori_value_depth.min() if vmin is None else vmin
    vmax = ori_value_depth.max() if vmax is None else vmax

    if vmin != vmax:
        value = (ori_value_depth - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = ori_value_depth * 0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # ((1)xhxwx4)

    value = value[:, :, :, :, :3] # bgr -> rgb
    # rgb_value = value[..., ::-1]
    rgb_value = value

    rgb_value[ori_value_depth==0] = [0,0,0]

    return torch.tensor(rgb_value[0].copy())

def latent2img(x, value_range=(-1, 1),is_multi_view=False):
    low, high = value_range
    x = x.to(torch.float32)
    x = x.clamp_(min=low, max=high)
    x = x.sub_(low).div_(max(high - low, 1e-5))
    if is_multi_view:
        x = x.mul(255).clamp_(0, 255).permute(0, 2, 3, 4, 1).to("cpu", torch.uint8)
    else:
        x = x.mul(255).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)
    return x



@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        use_extr_loss=False,
        use_vae_decode_loss=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform
        self.use_extr_loss = use_extr_loss
        self.use_vae_decode_loss = use_vae_decode_loss

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            use_extr_loss=self.use_extr_loss,
            use_vae_decode_loss=self.use_vae_decode_loss,
            **kwargs,
        )

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        neg_prompts=None,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
        m_cfg=False
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        if neg_prompts is not None:
            y_null = text_encoder.encode(neg_prompts)['y']
        else:
            y_null = text_encoder.null(n).to(device)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(
                t, additional_args, num_timesteps=self.num_timesteps,
                cog_style=self.scheduler.cog_style_trans,
            ) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = partial(tqdm, leave=False) if progress else (lambda x: x)
        cond_frame_x = model_args.pop('cond_frame_x', None)
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            if cond_frame_x is not None:
                z_in = torch.cat([z,cond_frame_x],dim=1)
                z_in = torch.cat([z_in]*2, 0)
            else:
                # classifier-free guidance
                z_in = torch.cat([z, z], 0)

            if m_cfg:
                embed()
                exit()
            else:
                t = torch.cat([t, t], 0)
                pred = model(z_in, t, **model_args)
                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                # update z
                dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
                dt = dt / self.num_timesteps
                z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None, vae=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, weights, t,vae)