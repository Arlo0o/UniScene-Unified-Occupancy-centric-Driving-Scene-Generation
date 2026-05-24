import torch
import torch.nn as nn
from uniscenev2_video.registry import SCHEDULERS
from uniscenev2_video.schedulers.rf import colorize, latent2img
from IPython import embed
from einops import rearrange, repeat
import torch.nn.functional as F



@SCHEDULERS.register_module("SigLoss")
class SigLoss(nn.Module):
    """SigLoss.

    Args:
        valid_mask (bool, optional): Whether filter invalid gt
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=None,
                 warm_up=False,
                 warm_iter=100,
                 num_camera = 7,
                 num_frames=1,
                 eps = 1e-8
                 ):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.num_camera = num_camera
        self.num_frames = num_frames

        self.eps = eps # avoid grad explode

        # HACK: a hack implement for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def decode_2d(self, z, vae, overlap=3, scale_factor=1):
        z = z / scale_factor
        # n_samples = default(1, z.shape[0])
        n_samples = z.shape[0]
        all_out = list()
        # with torch.autocast("cuda"):
        if overlap < n_samples:
            previous_z = z[:overlap]
            for current_z in z[overlap:].split(n_samples - overlap, dim=0):
                kwargs = {"timesteps": current_z.shape[0] + overlap}
                context_z = torch.cat((previous_z, current_z), dim=0)
                previous_z = current_z[-overlap:]
                out = vae.decode(context_z, **kwargs)
                if not all_out:
                    all_out.append(out)
                else:
                    all_out[-1][-overlap:] = (all_out[-1][-overlap:] + out[:overlap]) / 2
                    all_out.append(out[overlap:])
        else:
            # for current_z in z.split(n_samples, dim=0):
            #     kwargs = {"timesteps": current_z.shape[0]}
            #     out = vae.decode(current_z, **kwargs)
            #     all_out.append(out)
            for current_z in z.split(10, dim=0):
                kwargs = {"timesteps": 1}
                out = vae.decode(current_z, **kwargs)
                all_out.append(out)
            # return out
        out = torch.cat(all_out, dim=0)
        del all_out
        return out

    def vis(self, vae, x_start, pred_depth,sparse_depth_map,scale_factor=0.18215):
        # pred_depth/=self.max_depth
        # sparse_depth_map /= self.max_depth
        rgb_z,depth_z = x_start[:,:4], x_start[:,4:]
        overlap = rgb_z.shape[0]
        real_rec = self.decode_2d(rgb_z, vae, overlap=overlap, scale_factor=scale_factor)
        real_rec = rearrange(real_rec,"(b t) c h w -> b c t h w",t=self.num_frames)
        real_rec_mv = rearrange(real_rec,"(b n) c t h w -> b n c t h w",n=self.num_camera)
        
        real_depth = self.decode_2d(depth_z, vae, overlap=overlap, scale_factor=scale_factor)
        real_depth = rearrange(real_depth,"(b t) c h w -> b c t h w",t=self.num_frames).mean(dim=1).unsqueeze(1)
        real_depth_mv = rearrange(real_depth,"(b n) c t h w -> b n c t h w",n=self.num_camera)
        # real_depth_mv = 1 / real_depth_mv


        real_metric_depth_mv = rearrange(sparse_depth_map,"(b n t) c h w -> b n c t h w",t=self.num_frames,n=self.num_camera)
        sample_metric_depth_mv = rearrange(pred_depth,"(b n t) c h w -> b n c t h w",t=self.num_frames,n=self.num_camera)
        real_rec_list = []
        sample_rec_list = []
        mv_batch = real_rec_mv.shape[0]
        for i in range(mv_batch):
            mv_real_rec_list = []
            mv_real_depth_list = []
            mv_sample_metric_depth_list = []
            mv_real_metric_depth_list = []
            for j in range(self.num_camera):
                mv_real_video = latent2img(real_rec_mv[i][j])
                mv_real_rec_list.append(mv_real_video)
                mv_real_depth_video = colorize(real_depth_mv[i][j])
                mv_real_depth_list.append(mv_real_depth_video)
                mv_real_metric_depth_video = colorize(real_metric_depth_mv[i][j], vmin=0.1, vmax=self.max_depth)
                mv_real_metric_depth_list.append(mv_real_metric_depth_video)
                mv_sample_metric_depth_video = colorize(sample_metric_depth_mv[i][j], vmin=0.1, vmax=self.max_depth)
                mv_sample_metric_depth_list.append(mv_sample_metric_depth_video)

            real_h_front_list = mv_real_rec_list[:3]+[mv_real_rec_list[-1]]
            real_h_back_list = mv_real_rec_list[3:]
            real_hole_list = torch.cat([torch.cat(real_h_front_list,dim=2),torch.cat(real_h_back_list,dim=2)],dim=1)

            real_depth_h_front_list = mv_real_depth_list[:3]+[mv_real_depth_list[-1]]
            real_depth_h_back_list = mv_real_depth_list[3:]
            real_depth_hole_list = torch.cat([torch.cat(real_depth_h_front_list,dim=2),torch.cat(real_depth_h_back_list,dim=2)],dim=1)

            real_metirc_depth_h_front_list = mv_real_metric_depth_list[:3]+[mv_real_metric_depth_list[-1]]
            real_metirc_depth_h_back_list = mv_real_metric_depth_list[3:]
            real_metirc_depth_hole_list = torch.cat([torch.cat(real_metirc_depth_h_front_list,dim=2),torch.cat(real_metirc_depth_h_back_list,dim=2)],dim=1)


            real_rec_list.append(
                torch.cat(
                    [
                        real_hole_list,
                        real_depth_hole_list,
                        real_metirc_depth_hole_list
                    ],dim=1
                )
            )


            sample_metric_depth_h_front_list = mv_sample_metric_depth_list[:3]+[mv_sample_metric_depth_list[-1]]
            sample_metric_depth_h_back_list = mv_sample_metric_depth_list[3:]
            sample_metric_depth_hole_list = torch.cat([torch.cat(sample_metric_depth_h_front_list,dim=2),torch.cat(sample_metric_depth_h_back_list,dim=2)],dim=1)

            sample_rec_list.append(
                torch.cat(
                    [
                        real_hole_list,
                        real_depth_hole_list,
                        sample_metric_depth_hole_list
                    ],dim=1
                )
            )

        return real_rec_list, sample_rec_list



    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]

        # return F.smooth_l1_loss(input, target)

        # target/=self.max_depth

        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)+F.smooth_l1_loss(input, target)

    def forward(self,
                depth_pred,
                depth_gt,
                **kwargs):
        """Forward function."""
        
        loss_depth = self.loss_weight * self.sigloss(
            depth_pred,
            depth_gt,
            )
        return loss_depth