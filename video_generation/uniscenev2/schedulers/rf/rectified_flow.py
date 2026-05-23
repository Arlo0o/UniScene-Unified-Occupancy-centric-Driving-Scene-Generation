from typing import List
import logging
import torch.fft as fft
import torch
from torch.distributions import LogisticNormal
from einops import rearrange
from IPython import embed
import torch.nn.functional as F

def fourier_filter(x, scale, d_s=0.25):
    dtype = x.dtype
    x = x.type(torch.float32)
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda()

    for h in range(H):
        for w in range(W):
            d_square = (2 * h / H - 1) ** 2 + (2 * w / W - 1) ** 2
            if d_square <= 2 * d_s:
                mask[..., h, w] = scale

    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    x_filtered = x_filtered.type(dtype)
    return x_filtered

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py


def mean_flat(tensor: torch.Tensor, mask=None):
    """
    Take the mean over all non-batch dimensions.
    """
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        assert tensor.dim() == 5
        assert tensor.shape[2] == mask.shape[1]
        tensor = rearrange(tensor, "b c t h w -> b t (c h w)")
        denom = mask.sum(dim=1) * tensor.shape[-1]
        loss = (tensor * mask.unsqueeze(2)).sum(dim=1).sum(dim=1) / denom
        return loss


def _extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: List[int]):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
    cog_style=False,
):
    # Force fp16 input to fp32 to avoid nan output
    for key in ["height", "width", "num_frames"]:
        if model_kwargs[key].dtype == torch.float16:
            model_kwargs[key] = model_kwargs[key].float()
    repeat_t = int(t.shape[0]/model_kwargs["num_frames"].shape[0])
    t = t / num_timesteps
    resolution = model_kwargs["height"].repeat(repeat_t) * model_kwargs["width"].repeat(repeat_t)
    ratio_space = (resolution / base_resolution).sqrt()
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    # TODO: hard-coded, may change later!
    
    if model_kwargs["num_frames"][0] == 1:
        num_frames = torch.ones_like(model_kwargs["num_frames"])
    else:
        if cog_style:
            num_frames = model_kwargs["num_frames"].repeat(repeat_t) // 4 + model_kwargs["num_frames"].repeat(repeat_t) % 2
        else:
            num_frames = model_kwargs["num_frames"] // 17 * 5
    assert (num_frames >= 1).all(), "num_frames cannot be less than 1"
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    assert (ratio > 0).all(), "ratio cannot be 0"
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
        cog_style_trans=False,
        use_extr_loss=False,
        additional_loss_weight=0.1,
        use_vae_decode_loss=False
    ):
        self.use_extr_loss = use_extr_loss
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps
        self.additional_loss_weight = additional_loss_weight
        self.use_vae_decode_loss = use_vae_decode_loss

        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale
        if cog_style_trans:
            logging.warning("Use `cog_style_trans`. Please make sure train&inference is consistent!")
        self.cog_style_trans = cog_style_trans

    def extr_loss(
        self,
        x_start,
        noise,
        velocity_pred,
        t_frames,
        batch_szie,
        latent_dim,
        mask,
    ):
        
        # print(velocity_pred.mean())
        target_seq = x_start - noise
        aux_loss = ((target_seq[:, :, 1:] - target_seq[:, :, :-1]) - (velocity_pred[:, :, 1:] - velocity_pred[:, :, :-1])) ** 2
        tmp_h, tmp_w = aux_loss.shape[-2], aux_loss.shape[-1]
        aux_loss = rearrange(aux_loss, "b c t h w -> b (t h w) c", c=latent_dim)
        aux_w = F.normalize(aux_loss, p=2)
        aux_w = rearrange(aux_w, "b (t h w) c -> b c t h w", t=t_frames - 1, h=tmp_h, w=tmp_w)
        aux_w = 1 + torch.cat((torch.zeros(batch_szie, aux_w.shape[1], 1, *aux_w.shape[3:]).to(aux_w), aux_w), dim=2)
        predict_hf = fourier_filter(rearrange(velocity_pred,"b c t h w -> (b t) c h w"), scale=0.)
        target_hf = fourier_filter(rearrange(target_seq,"b c t h w -> (b t) c h w"), scale=0.)
        hf_loss = torch.mean(((predict_hf - target_hf) ** 2).reshape(target_seq.shape[0], -1), 1)
        
        # print("-------------------------------------------------------------")
        loss = mean_flat(((((velocity_pred - target_seq).pow(2)))* aux_w.detach()), mask=mask) + self.additional_loss_weight * hf_loss
      
        # aux_w= torch.ones(1, device=hf_loss.device, dtype=hf_loss.dtype)
        # mask=None
        # loss = mean_flat(((((velocity_pred - target_seq).pow(2)))* aux_w.detach()), mask=mask) #+ self.additional_loss_weight * hf_loss
        # print( velocity_pred.max().item(), velocity_pred.min().item(), velocity_pred.mean().item(), target_seq.mean().item(),  loss.mean().item() )
        return loss



    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None, vae=None):
        """
        Compute training losses for a single timestep.
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps, cog_style=self.cog_style_trans)
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)
        terms = {}
        cond_frame_x = model_kwargs.pop('cond_frame_x', None)
        if cond_frame_x is not None:
            x_input = torch.cat([x_t,cond_frame_x],dim=1)
            model_output = model(x_input, t, **model_kwargs)
        else:
            model_output = model(x_t, t, **model_kwargs)
        if model_output.shape[1] == 2 * x_t.shape[1]:
            model_output = model_output.chunk(2, dim=1)[0]
        velocity_pred = model_output

        B,NC,T,H,W = velocity_pred.shape
        if weights is None:
            if self.use_extr_loss:
                loss = self.extr_loss(
                    x_start,
                    noise,
                    velocity_pred,
                    T,
                    B,
                    NC,
                    None if mask is None else mask,
                )
            else:
                loss = mean_flat((velocity_pred - (x_start - noise)).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (velocity_pred - (x_start - noise)).pow(2), mask=mask)
        terms["loss"] = loss

        return terms

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return timepoints * original_samples + (1 - timepoints) * noise





#################################################

    def dit3d_camera_ray_co_training_sample(
        self,
        model=None,
        vae=None,
        z=None,
        model_args=None,
        uc_model_args=None,
        device=None,
        mask=None,
        guidance_scale=None,
        progress=True,
        f_frames=44,
        vae_type='VAE_3D',
        diff_type='unet',
        scale_factor=1.0,
        overlap=3,
        round_num = 1,
        global_step=0,
        seed=None,
        return_pred_latent = False,
        batch = None,
        use_cfg=True,
        m_cfg = True,
        txt_scale = 1.0,
        camera_scale = 1.0
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale
        # dt = 1.0/self.num_sampling_steps
        # prepare timesteps
        if self.scheduler.sample_method == "uniform-norm-t":
            timesteps = [i / self.num_sampling_steps for i in range(self.num_sampling_steps)]
        else:
            timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            model_args['height'] = torch.tensor(352)
            model_args['width'] = torch.tensor(704)
            model_args['num_frames'] = torch.tensor(36)
            timesteps = [timestep_transform(t, model_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)
        n_frames = z.shape[2]
        batch_szie = z.shape[0]
        real_rec_list = []
        sample_rec_list = []
        noise_list = []
        for round_idx in range(round_num):
            if seed is not None:
                torch.manual_seed(seed)
            noise = torch.randn_like(z.to(torch.float32)).to(z.dtype)
            progress_wrap = tqdm if progress else (lambda x: x)
            for i, t in enumerate(progress_wrap(timesteps)):
                # noise[:,:,:self.cond_frame_index] = model_args['concat'][:,:,:self.cond_frame_index]
                # uc_z_in = torch.cat([noise, torch.zeros_like(model_args['concat'])], dim=1)
                z_in = torch.cat([noise, model_args['concat']], dim=1)
                condition_control = rearrange(batch['ray_map'],'b t c h w -> b c t h w')
                if use_cfg:
                    if m_cfg:
                        # pred = model(
                        #     torch.cat([z_in,z_in,z_in], 0),
                        #     timestep=torch.cat([t]*3, 0),
                        #     encoder_hidden_states=torch.cat([model_args['crossattn'][:,None,:,:],torch.zeros_like(model_args['crossattn'][:,None,:,:]),model_args['crossattn'][:,None,:,:]], 0),
                        #     condition_control = torch.cat([condition_control,condition_control,torch.zeros_like(condition_control)], 0),
                        # )
                        # pred_cond, pred_cam_uncond, pred_txt_uncond = pred.chunk(3, dim=0)
                        pred_cond = model(
                            z_in,
                            timestep=t,
                            encoder_hidden_states=model_args['crossattn'][:,None,:,:],
                            condition_control = condition_control,
                        )

                        pred_txt_uncond = model(
                            z_in,
                            timestep=t,
                            encoder_hidden_states=torch.zeros_like(model_args['crossattn'][:,None,:,:]),
                            condition_control = condition_control,
                        )

                        pred_cam_uncond = model(
                            z_in,
                            timestep=t,
                            encoder_hidden_states=model_args['crossattn'][:,None,:,:],
                            condition_control = torch.zeros_like(condition_control),
                        )
                        
                        v_pred = pred_cam_uncond + (camera_scale*(pred_cond-pred_cam_uncond))+ (txt_scale*(pred_cond-pred_txt_uncond))
                    else:
                        pred = model(
                            torch.cat([z_in,z_in], 0),
                            timestep=torch.cat([t]*2, 0),
                            encoder_hidden_states=torch.cat([model_args['crossattn'][:,None,:,:],torch.zeros_like(model_args['crossattn'][:,None,:,:])], 0),
                            condition_control = torch.cat([condition_control,torch.zeros_like(condition_control)], 0),
                        )
                        pred_cond, pred_uncond = pred.chunk(2, dim=0)
                        v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                else:
                    v_pred = model(
                        z_in,
                        timestep=t,
                        encoder_hidden_states=model_args['crossattn'][:,None,:,:],
                        condition_control = condition_control
                    )

                dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
                dt = dt / self.num_timesteps
                noise = noise + v_pred * dt[:, None, None, None, None]
            model_args['concat'][:,:,:self.cond_frame_index] = noise.clone()[:,:,n_frames-self.cond_frame_index:]
            torch.cuda.empty_cache()

            noise_list.append(noise)
            if vae_type == 'VAE_3D':
                real_rec = vae.decode(z,f_frames)
                sample_rec = vae.decode(noise.to(z.dtype),f_frames)
            elif vae_type == 'CausalVAE3D':
                real_rec = vae.decode(z)
                sample_rec = vae.decode(noise.to(z.dtype))
            elif 'CogVideo' in vae_type:
                real_rec = vae.decode(z)
                sample_rec = vae.decode(noise.to(z.dtype))
            elif 'VAE_2D' in vae_type:
                z_2d = rearrange(z,"b c t h w -> (b t) c h w")
                overlap = z_2d.shape[0]
                real_rec = self.decode_2d(z_2d, vae, overlap=overlap, scale_factor=scale_factor)
                noise_2d = rearrange(noise,"b c t h w -> (b t) c h w").to(z.dtype)
                sample_rec = self.decode_2d(noise_2d, vae, overlap=overlap, scale_factor=scale_factor)
                real_rec = rearrange(real_rec,"(b t) c h w -> b c t h w",b=batch_szie)
                sample_rec = rearrange(sample_rec,"(b t) c h w -> b c t h w",b=batch_szie)


            if self.num_camera > 1:
                real_rec_mv = rearrange(real_rec,"(b n) c t h w -> b n c t h w",n=self.num_camera)
                sample_rec_mv = rearrange(sample_rec,"(b n) c t h w -> b n c t h w",n=self.num_camera)
                mv_batch = real_rec_mv.shape[0]
                for i in range(mv_batch):
                    mv_real_rec_list = []
                    mv_sample_rec_list = []
                    for j in range(self.num_camera):
                        mv_real_video = latent2img(real_rec_mv[i][j])
                        mv_sample_video = latent2img(sample_rec_mv[i][j])
                        mv_real_rec_list.append(mv_real_video)
                        mv_sample_rec_list.append(mv_sample_video)

                    if self.num_camera == 3:
                        real_rec_list.append(torch.cat(mv_real_rec_list,dim=2))
                        sample_rec_list.append(torch.cat(mv_sample_rec_list,dim=2))
                    elif self.num_camera == 6:
                        real_h_front_list = mv_real_rec_list[:3]
                        real_h_back_list = mv_real_rec_list[3:]
                        real_rec_list.append(torch.cat([torch.cat(real_h_front_list,dim=2),torch.cat(real_h_back_list,dim=2)],dim=1))

                        sample_h_front_list = mv_sample_rec_list[:3]
                        sample_h_back_list = mv_sample_rec_list[3:]
                        sample_rec_list.append(torch.cat([torch.cat(sample_h_front_list,dim=2),torch.cat(sample_h_back_list,dim=2)],dim=1))
                    elif self.num_camera == 7:
                        real_h_front_list = mv_real_rec_list[:3]+[mv_real_rec_list[-1]]
                        real_h_back_list = mv_real_rec_list[3:]

                        sample_h_front_list = mv_sample_rec_list[:3]+[mv_sample_rec_list[-1]]
                        sample_h_back_list = mv_sample_rec_list[3:]

                        if round_idx == 0:
                            real_rec_list.append(torch.cat([torch.cat(real_h_front_list,dim=2),torch.cat(real_h_back_list,dim=2)],dim=1))
                            sample_rec_list.append(torch.cat([torch.cat(sample_h_front_list,dim=2),torch.cat(sample_h_back_list,dim=2)],dim=1))
                        else:
                            temp_real_rec_list = torch.cat([torch.cat(real_h_front_list,dim=2),torch.cat(real_h_back_list,dim=2)],dim=1)
                            temp_sample_rec_list = torch.cat([torch.cat(sample_h_front_list,dim=2),torch.cat(sample_h_back_list,dim=2)],dim=1)

                            real_rec_list[i] = torch.cat([
                                real_rec_list[i],
                                temp_real_rec_list[self.n_cond_frames:]
                            ],dim=0)

                            sample_rec_list[i] = torch.cat([
                                sample_rec_list[i],
                                temp_sample_rec_list[self.n_cond_frames:]
                            ],dim=0)
            else:
                for i in range(batch_szie):
                    real_video = latent2img(real_rec[i])
                    sample_video = latent2img(sample_rec[i])
                    real_rec_list.append(real_video)
                    sample_rec_list.append(sample_video)
            if round_num > 1:
                save_real_path = "debug_outputs/check_round_img/real"
                save_sample_path = "debug_outputs/check_round_img/sample"

                os.makedirs(save_real_path,exist_ok=True)
                os.makedirs(save_sample_path,exist_ok=True)

                for i in range(len(sample_rec_list)):
                    write_video(
                        os.path.join(
                            save_real_path,
                            str(global_step)+'_'+str(round_idx)+'_'+str(i)+'_real.mp4'
                        ),
                        real_rec_list[i],
                        fps=10
                    )
                    write_video(
                        os.path.join(
                            save_sample_path,
                            str(global_step)+'_'+str(round_idx)+'_'+str(i)+'_sample.mp4'
                        ),
                        sample_rec_list[i],
                        fps=10
                    )

 


        if return_pred_latent:
            return real_rec_list, sample_rec_list, noise_list

        return real_rec_list, sample_rec_list
    