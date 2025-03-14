import copy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .render_utils import models
from .render_utils.rays import RayBundle
import pickle
from typing import Dict, List, Tuple, Union
from torch import Tensor
try:
    from dda3d_gpu import dda3d_gpu
except:
    print('disable drop_collisionless_rays')
    
def get_rays(x: Tensor, y: Tensor, c2w: Tensor, intrinsic: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        x: the horizontal coordinates of the pixels, shape: (num_rays,)
        y: the vertical coordinates of the pixels, shape: (num_rays,)
        c2w: the camera-to-world matrices, shape: (num_cams, 4, 4)
        intrinsic: the camera intrinsic matrices, shape: (num_cams, 3, 3)
    Returns:
        origins: the ray origins, shape: (num_rays, 3)
        viewdirs: the ray directions, shape: (num_rays, 3)
        direction_norm: the norm of the ray directions, shape: (num_rays, 1)
    """
    if len(intrinsic.shape) == 2:
        intrinsic = intrinsic[None, :, :]
    if len(c2w.shape) == 2:
        c2w = c2w[None, :, :]
    camera_dirs = torch.nn.functional.pad(
        torch.stack(
            [
                (x - intrinsic[:, 0, 2] + 0.5) / intrinsic[:, 0, 0],
                (y - intrinsic[:, 1, 2] + 0.5) / intrinsic[:, 1, 1],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [num_rays, 3]

    # rotate the camera rays w.r.t. the camera pose
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = (c2w[:, :3, -1]).expand(directions.shape)
    # TODO: not sure if we still need direction_norm
    direction_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
    # normalize the ray directions
    viewdirs = directions / (direction_norm + 1e-8)
    return origins, viewdirs, direction_norm



class Occ2LiDARRender(nn.Module):
    def __init__(
        self,
        model_cfg,
        # in_channels,
        # unified_voxel_size,
        # unified_voxel_shape,
        # pc_range,
        # render_conv_cfg,
        # view_cfg,
        # ray_sampler_cfg,
        # render_ssl_cfg,
        **kwargs
    ):
        super().__init__()
        in_channels = model_cfg.in_channels
        unified_voxel_size = model_cfg.unified_voxel_size
        unified_voxel_shape = model_cfg.unified_voxel_shape
        pc_range = model_cfg.pc_range
        render_conv_cfg = model_cfg.render_conv_cfg
        view_cfg = model_cfg.view_cfg
        ray_sampler_cfg = model_cfg.ray_sampler_cfg
        render_ssl_cfg = model_cfg.render_ssl_cfg
        self.drop_collisionless_rays = model_cfg.get('drop_collisionless_rays', False)
        self.drop_ray_when_pred = model_cfg.get('drop_ray_when_pred', True)
        self.use_gt_drop = model_cfg.get('use_gt_drop', False)
        if self.use_gt_drop:
            print('!!!!!!!!!! use gt drop !!!!!!!!!!!')
        if kwargs.get("fp16_enabled", False):
            self.fp16_enabled = True
        self.in_channels = in_channels
        self.pc_range = np.array(pc_range, dtype=np.float32)
        self.unified_voxel_shape = np.array(unified_voxel_shape, dtype=np.int32)
        self.unified_voxel_size = np.array(unified_voxel_size, dtype=np.float32)

        if render_conv_cfg is not None:
            self.render_conv = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    render_conv_cfg["out_channels"],
                    kernel_size=render_conv_cfg["kernel_size"],
                    padding=render_conv_cfg["padding"],
                    stride=1,
                ),
                nn.BatchNorm3d(render_conv_cfg["out_channels"]),
                nn.ReLU(inplace=True),
            )
        else:
            self.render_conv = None

        model_type = render_ssl_cfg.pop("type")
        self.render_model = getattr(models, model_type)(
            pc_range=self.pc_range,
            voxel_size=self.unified_voxel_size,
            voxel_shape=self.unified_voxel_shape,
            **render_ssl_cfg
        )
        render_ssl_cfg['type'] = model_type

        self.ray_sampler_cfg = ray_sampler_cfg
        self.part = 8192  # avoid out of GPU memory
        self.forward_ret_dict = {}


        # generate pre-defined rays
        self.use_predefine_rays = model_cfg.get('use_predefine_rays', False)
        if self.use_predefine_rays:
            self.predefine_rays_cfg = model_cfg.predefine_rays_cfg
            azimuth_range = self.predefine_rays_cfg.azimuth_range
            azimuth_res = self.predefine_rays_cfg.azimuth_res
            elevation_range = self.predefine_rays_cfg.elevation_range
            elevation_res = self.predefine_rays_cfg.elevation_res
            elevation_beams = self.predefine_rays_cfg.elevation_beams
            azi = np.arange(azimuth_range[0], azimuth_range[1], azimuth_res)
            #ele = np.arange(elevation_range[0], elevation_range[1], elevation_res)
            ele = np.linspace(elevation_range[0], elevation_range[1], elevation_beams)

            assert len(ele) == elevation_beams, f'number of beams is not {elevation_beams}'
            
            # create meshgrid of all possible combination of azi and ele
            ae = np.vstack(np.meshgrid(azi,ele)).reshape(2,-1)

            directions = np.vstack((np.cos(np.deg2rad(ae[1,:])) * np.cos(np.deg2rad(ae[0,:])), 
                        np.cos(np.deg2rad(ae[1,:])) * np.sin(np.deg2rad(ae[0,:])),
                        np.sin(np.deg2rad(ae[1,:])))).T
            directions = directions / np.linalg.norm(directions, axis=1)[:,np.newaxis]
            origins = np.zeros_like(directions)
            self.predefine_rays = {
                'ray_o': torch.from_numpy(origins).to(torch.float32).cuda() * self.render_model.scale_factor,
                'ray_d': torch.from_numpy(directions).to(torch.float32).cuda()
            }

    def get_loss(self):
        preds_dict = self.forward_ret_dict['preds_dict']
        targets = self.forward_ret_dict['targets']
        #lidar_targets, _ = targets
        lidar_targets = targets
        batch_size = len(lidar_targets)
        loss_dict = {}
        for bs_idx in range(batch_size):
            i_loss_dict = self.render_model.loss(preds_dict[bs_idx], lidar_targets[bs_idx])
            for k, v in i_loss_dict.items():
                if k not in loss_dict:
                    loss_dict[k] = []
                loss_dict[k].append(v)
        for k, v in loss_dict.items():
            loss_dict[k] = torch.stack(v, dim=0).mean()

        tb_dict = {}
        total_loss = 0
        for k, v in loss_dict.items():
            tb_dict[k] = v.item()
            total_loss += v
        return total_loss, tb_dict
    
    def generate_predicted_pc(self, batch_dict):
        #lidar_rays, _ = self.forward_ret_dict['targets']
        lidar_rays = self.forward_ret_dict['targets']
        batch_size = batch_dict['batch_size']
        pred_dicts = {'pc_out': [], 'gt_pts': []}
        for bs in range(batch_size):
            intensity_pred = None
            raydrop_pred = None
            if self.render_model.pred_intensity and self.render_model.pred_raydrop:
                intensity_pred, raydrop_pred = batch_dict['preds_dict'][bs]['lidar_relative'].split(1, dim=-1)
            elif self.render_model.pred_intensity:
                intensity_pred = batch_dict['preds_dict'][bs]['lidar_relative']
            elif self.render_model.pred_raydrop:
                raydrop_pred = batch_dict['preds_dict'][bs]['lidar_relative']

            pred_pts = batch_dict['preds_dict'][bs]['depth'] * lidar_rays[bs]['ray_d'] / self.render_model.scale_factor
            if intensity_pred is not None:
                pred_pts = torch.cat([pred_pts, intensity_pred.sigmoid()], dim=-1)
            if raydrop_pred is not None and self.drop_ray_when_pred:
                if self.use_gt_drop:
                    pred_pts = pred_pts[lidar_rays[bs]['did_return'].squeeze()]
                else:
                    pred_pts = pred_pts[(raydrop_pred.sigmoid()<0.5).squeeze()]
                    #print((raydrop_pred.sigmoid()>0.5).sum())
                
            pred_dicts['pc_out'].append(pred_pts)
            
            if 'points' in batch_dict:
                if 'pts_origin' in lidar_rays[bs]:
                    gt_pts = lidar_rays[bs]['pts_origin'][:, :3]
                else:
                    did_return = lidar_rays[bs]['did_return'].squeeze()
                    gt_pts = lidar_rays[bs]['depth'][did_return] * lidar_rays[bs]['ray_d'][did_return] / self.render_model.scale_factor
                pred_dicts['gt_pts'].append(gt_pts)


            vis = False
            if vis:
                pred_pts_np = pred_pts.cpu().numpy().astype('float32')
                colors1 = np.zeros_like(pred_pts_np)
                colors1[:, 0] = 255
                gt_pts_np = gt_pts.cpu().numpy().astype('float32')
                colors2 = np.zeros_like(gt_pts_np)
                colors2[:, 1] = 255
                np.concatenate([np.concatenate([pred_pts_np, colors1], axis=1), np.concatenate([gt_pts_np, colors2], axis=1)]).tofile('z.bin')
        return pred_dicts
        
    def forward(self, batch_dict):#pts_feats, rays):
        """
        Args:
            Currently only support single-frame, no 3D data augmentation, no 2D data augmentation
            ray_o: [(N*C*K, 3), ...]
            ray_d: [(N*C*K, 3), ...]
            img_feats: [(B, N*C, C', H, W), ...]
            img_depth: [(B*N*C, 64, H, W), ...]
        Returns:

        """
        uni_feats = []
        # if pts_feats is not None:
        #     uni_feats.append(pts_feats)

        # uni_feats = sum(uni_feats)
        # uni_feats = self.render_conv(uni_feats)
        pts_feats = batch_dict['feature_volumes'].dense()
        # B, C, Z, Y, X
        pts_feats = pts_feats[:, :, :self.unified_voxel_shape[2], :, :]
        batch_size = batch_dict['batch_size']

        for bs in range(batch_size):
            uni_feats.append(pts_feats[bs])
        
        if self.use_predefine_rays:
            rays = self.sample_rays(batch_size)
            lidar_rays, _ = rays
        else:
            pts_all = batch_dict['points']
            occ_grids = batch_dict.get('grid', None)
            pts = []
            did_returns = []
            for bs in range(batch_size):
                batch_mask = pts_all[:, 0].int()==bs
                pts.append(pts_all[batch_mask][:, 1:6])
                did_returns.append(batch_dict['did_return'][batch_mask][:, 1].bool())
            rays = self.sample_rays(batch_size, pts, did_returns, occ_grids=occ_grids)
            lidar_rays, _ = rays
        self.forward_ret_dict['targets'] = lidar_rays

        if self.render_model.pred_raydrop and self.training:
            for bs_idx in range(batch_size):
                dis = torch.norm(pts[bs_idx][:, :3], p=2, dim=-1)
                dis_mask = (dis > self.ray_sampler_cfg.close_radius) & (
                    dis < self.ray_sampler_cfg.get("far_radius", 100.0)
                ) & did_returns[bs_idx]
                self.forward_ret_dict['targets'][bs_idx]['pts_origin'] = pts[bs_idx][dis_mask]
        
        batch_ret = []              
        for bs_idx in range(batch_size):
            # i_cam_ray_o, i_cam_ray_d = (
            #     cam_rays[bs_idx]["ray_o"],
            #     cam_rays[bs_idx]["ray_d"],
            # )
            if self.training:
                i_ray_o, i_ray_d, i_ray_depth = (
                    lidar_rays[bs_idx]["ray_o"],
                    lidar_rays[bs_idx]["ray_d"],
                    lidar_rays[bs_idx].get("depth", None),
                )
                scaled_points = lidar_rays[bs_idx]["scaled_points"]
                lidar_ray_bundle = RayBundle(
                    origins=i_ray_o, directions=i_ray_d, depths=i_ray_depth
                )
                # cam_ray_bundle = RayBundle(
                #     origins=i_cam_ray_o, directions=i_cam_ray_d
                # )
                cam_ray_bundle = None
                preds_dict = self.render_model(
                    lidar_ray_bundle, uni_feats[bs_idx].contiguous(), points=scaled_points
                )

            else:
                # assert i_ray_o.shape[0] == i_ray_d.shape[0]
                # cam_ray_bundle = RayBundle(
                #     origins=i_cam_ray_o, directions=i_cam_ray_d
                # )
                cam_ray_bundle = None
                i_ray_o, i_ray_d, i_ray_depth = (
                    lidar_rays[bs_idx]["ray_o"],
                    lidar_rays[bs_idx]["ray_d"],
                    lidar_rays[bs_idx].get("depth", None),
                )
                # scaled_points = lidar_rays[bs_idx]["scaled_points"]
                lidar_ray_bundle = RayBundle(
                    origins=i_ray_o, directions=i_ray_d, depths=i_ray_depth
                )
                preds_dict = self.render_model(
                    lidar_ray_bundle, uni_feats[bs_idx].contiguous()
                )
            batch_ret.append(preds_dict)

        self.forward_ret_dict['preds_dict'] = batch_ret
        batch_dict['preds_dict'] = batch_ret

        vis=False
        if vis:
            pred_pts = batch_dict['preds_dict'][0]['depth'] * lidar_rays[0]['ray_d'] / self.render_model.scale_factor
            pred_pts.detach().cpu().numpy().astype('float32').tofile('z.bin')

            target_pts = lidar_rays[0]['depth'] * lidar_rays[0]['ray_d'] / self.render_model.scale_factor
            target_pts.detach().cpu().numpy().astype('float32').tofile('z_gt.bin')
        
        debug_pred=False
        if debug_pred:
            self.generate_predicted_pc(batch_dict)
        
        return batch_dict

    def sample_rays(self, batch_size, pts=None, did_returns=None, occ_grids=None):
        lidar_ret = self.sample_lidar_rays(batch_size, pts, did_returns, occ_grids=occ_grids)
        return lidar_ret, None

    def sample_lidar_rays(self, batch_size, pts=None, did_returns=None, test=False, occ_grids=None):
        """Get lidar ray
        Returns:
            lidar_ret: list of dict, each dict contains:
                ray_o: (num_rays, 3)
                ray_d: (num_rays, 3)
                depth: (num_rays, 1)
                scaled_points: (num_rays, 3)
        """
        lidar_ret = []
        if self.use_predefine_rays and (not self.training):
            for i in range(batch_size):
                if occ_grids is not None and self.drop_collisionless_rays:
                    grid = occ_grids[i]
                    lidar_directions = self.predefine_rays['ray_d'].clone().contiguous()
                    lidar_origins = self.predefine_rays['ray_o'].clone().contiguous()
                    num_rays = lidar_directions.shape[0]
                    intersection_mask = torch.zeros(num_rays, dtype=torch.bool, device=lidar_directions.device)
                    W, H, D = grid.shape
                    dda3d_gpu(
                        lidar_directions, grid, intersection_mask,
                        *self.pc_range[:3], *self.unified_voxel_size, W, H, D, num_rays,
                    )
                    lidar_origins = lidar_origins[intersection_mask]
                    lidar_directions = lidar_directions[intersection_mask]
                    rays = {
                        'ray_o': lidar_origins,
                        'ray_d': lidar_directions,
                        'intersection_mask': intersection_mask
                    }
                    lidar_ret.append(rays)
                else:
                    rays = copy.deepcopy(self.predefine_rays)
                    rays.update({'intersection_mask': None})
                    lidar_ret.append(rays)
            return lidar_ret

        for i in range(len(pts)):
            lidar_pc = pts[i]
            did_return = did_returns[i]
            dis = torch.norm(lidar_pc[:, :3], p=2, dim=-1)
            dis_mask = (dis > self.ray_sampler_cfg.close_radius) & (
                dis < self.ray_sampler_cfg.get("far_radius", 100.0)
            ) | (~did_return)
            lidar_pc = lidar_pc[dis_mask]
            did_return = did_return[dis_mask]
            lidar_points = lidar_pc[:, :3]
            lidar_origins = torch.zeros_like(lidar_points)
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)
            lidar_directions = lidar_directions / lidar_ranges
            lidar_intensity = lidar_pc[:, 3:4]

            if occ_grids is not None and self.drop_collisionless_rays:
                grid = occ_grids[i]
                num_rays = lidar_directions.shape[0]
                intersection_mask = torch.zeros(num_rays, dtype=torch.bool, device=lidar_directions.device)
                W, H, D = grid.shape
                dda3d_gpu(
                    lidar_directions, grid, intersection_mask,
                    *self.pc_range[:3], *self.unified_voxel_size, W, H, D, num_rays,
                )
                lidar_origins = lidar_origins[intersection_mask]
                lidar_directions = lidar_directions[intersection_mask]
                lidar_intensity = lidar_intensity[intersection_mask]
                lidar_ranges = lidar_ranges[intersection_mask]
                lidar_points = lidar_points[intersection_mask]
                did_return = did_return[intersection_mask]
            else:
                intersection_mask = None

            lidar_ret.append(
                {
                    "ray_o": lidar_origins * self.render_model.scale_factor,
                    "ray_d": lidar_directions,
                    "depth": lidar_ranges * self.render_model.scale_factor if not test else None,
                    "scaled_points": lidar_points * self.render_model.scale_factor,
                    "scale_factor": self.render_model.scale_factor,
                    'intensity': lidar_intensity,
                    'did_return': did_return,
                    'intersection_mask': intersection_mask
                }
            )
        return lidar_ret
