import copy
import shutil
import pickle
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .render_utils import models
from .render_utils.rays import RayBundle
import pickle
from typing import Dict, List, Tuple, Union
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from torch import Tensor
from dda3d_gpu import dda3d_gpu
from pcdet.datasets.nuscenes_occ.nuscenes_constants import CAM_FRONT_INTRINSIC

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



class Occ2LiDARRenderVis(nn.Module):
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
        self.save_folder = model_cfg.save_folder
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
        self.vis_raydrop = model_cfg.get('vis_raydrop', True)
        self.use_predefine_rays = model_cfg.get('use_predefine_rays', True)
        if self.use_predefine_rays:
            pass
            # self.predefine_rays_cfg = model_cfg.predefine_rays_cfg
            # azimuth_range = self.predefine_rays_cfg.azimuth_range
            # azimuth_res = self.predefine_rays_cfg.azimuth_res
            # elevation_range = self.predefine_rays_cfg.elevation_range
            # elevation_res = self.predefine_rays_cfg.elevation_res
            # elevation_beams = self.predefine_rays_cfg.elevation_beams
            # azi = np.arange(azimuth_range[0], azimuth_range[1], azimuth_res)
            # #ele = np.arange(elevation_range[0], elevation_range[1], elevation_res)
            # ele = np.linspace(elevation_range[0], elevation_range[1], elevation_beams)

            # assert len(ele) == elevation_beams, f'number of beams is not {elevation_beams}'
            
            # # create meshgrid of all possible combination of azi and ele
            # ae = np.vstack(np.meshgrid(azi,ele)).reshape(2,-1)

            # directions = np.vstack((np.cos(np.deg2rad(ae[1,:])) * np.cos(np.deg2rad(ae[0,:])), 
            #             np.cos(np.deg2rad(ae[1,:])) * np.sin(np.deg2rad(ae[0,:])),
            #             np.sin(np.deg2rad(ae[1,:])))).T
            # directions = directions / np.linalg.norm(directions, axis=1)[:,np.newaxis]
            # origins = np.zeros_like(directions)
            # self.predefine_rays = {
            #     'ray_o': torch.from_numpy(origins).to(torch.float32).cuda() * self.render_model.scale_factor,
            #     'ray_d': torch.from_numpy(directions).to(torch.float32).cuda()
            # }

            downsample_rate = 2
            self.downsample_rate = downsample_rate
            image_width = 1600 // downsample_rate
            image_height = 900 // downsample_rate
            pixel_offset = 0.0
            image_coords = torch.meshgrid(torch.arange(image_height), torch.arange(image_width))
            image_coords = torch.stack([image_coords[1], image_coords[0]], dim=-1) + pixel_offset  # stored as (y, x) coordinates
            image_coords = image_coords.view((-1, 2))
            self.image_coords = torch.cat([image_coords, image_coords.new_ones((image_coords.shape[0], 1))], dim=-1).cuda()

    def do_range_projection(self, points):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        proj_fov_down, proj_fov_up = [-30.67, 10.67]
        # laser parameters
        fov_up = proj_fov_up / 180.0 * torch.pi      # field of view up in rad
        fov_down = proj_fov_down / 180.0 * torch.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = torch.linalg.norm(points[:, :3], 2, dim=1)

        # get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        # get angles of all points
        yaw = -torch.arctan2(scan_y, scan_x)
        pitch = torch.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / torch.pi + 1.0)          # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

        return proj_x, proj_y

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
                if self.vis_raydrop:
                    pred_pts = torch.cat([pred_pts, raydrop_pred.sigmoid()], dim=-1)
                else:
                    if self.use_gt_drop:
                        pred_pts = pred_pts[lidar_rays[bs]['did_return'].squeeze()]
                    else:
                        pred_pts = pred_pts[(raydrop_pred.sigmoid()<0.5).squeeze()]
                
            pred_dicts['pc_out'].append(pred_pts)
            if self.vis_raydrop and raydrop_pred is not None:
                did_return = lidar_rays[bs]['did_return'][:, None]
                gt_pts = lidar_rays[bs]['depth'] * lidar_rays[bs]['ray_d'] / self.render_model.scale_factor
                if 'intensity' in lidar_rays[bs]:
                    gt_pts = torch.cat([gt_pts, lidar_rays[bs]['intensity']], dim=-1)
                gt_pts = torch.cat([gt_pts, did_return], dim=-1)
            else:
                if 'pts_origin' in lidar_rays[bs]:
                    gt_pts = lidar_rays[bs]['pts_origin'][:, :3]
                else:
                    if 'did_return' in lidar_rays[bs]:
                        did_return = lidar_rays[bs]['did_return'].squeeze()
                    else:
                        did_return = lidar_rays[bs]['depth'].new_ones((lidar_rays[bs]['depth'].shape[0], 1)).bool()
                    gt_pts = lidar_rays[bs]['depth'][did_return] * lidar_rays[bs]['ray_d'][did_return] / self.render_model.scale_factor
                if 'intensity' in lidar_rays[bs]:
                    gt_pts = torch.cat([gt_pts, lidar_rays[bs]['intensity'][did_return]], dim=-1)
            pred_dicts['gt_pts'].append(gt_pts)


            vis = True
            if vis:
                occ_filename = '/data/longhun/3D/nuscenes/data/pyramid_occ/nuscene_quantized_200_200_16/quantized'
                frame_id = batch_dict['frame_id'][0]
                token = batch_dict['frame_id'][0].split('-')[0]
                save_folder = os.path.join(self.save_folder, token)
                os.makedirs(save_folder, exist_ok=True)

                occ_filename = os.path.join(occ_filename, frame_id.split('-')[0], frame_id.split('-')[1]+'.npy')
                assert os.path.exists(occ_filename)
                shutil.copy(occ_filename, os.path.join(save_folder, 'occ.npy'))
                data_for_vis = {
                    'pred_pts': pred_pts.cpu().numpy(),
                    'gt_pts': gt_pts.cpu().numpy()
                }
                with open(os.path.join(save_folder, 'data_for_vis.pkl'), 'wb') as f:
                    pickle.dump(data_for_vis, f)

                # range map view
                if self.render_model.pred_raydrop:
                    proj_x_pred, proj_y_pred = self.do_range_projection(pred_pts)
                    proj_x_gt, proj_y_gt = self.do_range_projection(gt_pts)
                    
                    proj_x_pred *= 360
                    proj_y_pred = proj_y_pred * 40 - 30
                    proj_x_gt *= 360
                    proj_y_gt = proj_y_gt * 40 - 30
                    fig = plt.figure(dpi=500)
                    ax1 = plt.gca()
                    #ax1 = plt.subplot(211)
                    im1 = ax1.scatter(proj_x_pred.cpu().numpy(), proj_y_pred.cpu().numpy(), c=1-pred_pts[:,-1].cpu().numpy(), s=0.05, vmin=0.0, vmax=1.0)
                    plt.yticks([-30, -10, 10])
                    plt.xticks([0, 90, 180, 270, 360])
                    plt.xlabel('Azimuth (°)')
                    plt.ylabel('Elevation (°)')
                    #plt.colorbar(im1)
                    fig.colorbar(im1, ax=[ax1], orientation='vertical', fraction=0.015, pad=0.05)
                    plt.gca().set_aspect(2.0)
                    plt.title('Predicted Drop Probability')
                    plt.savefig(os.path.join(save_folder, 'raydrop_pred.png'), bbox_inches='tight', pad_inches=0)
                    plt.close()


                    fig=plt.figure(dpi=500)
                    ax2 = plt.gca()
                    #ax2 = plt.subplot(212)
                    im2 = ax2.scatter(proj_x_gt.cpu().numpy(), proj_y_gt.cpu().numpy(), c=gt_pts[:,-1].cpu().numpy(), s=0.05, vmin=0.0, vmax=1.0)
                    plt.yticks([-30, -10, 10])
                    plt.xticks([0, 90, 180, 270, 360])
                    plt.xlabel('Azimuth (°)')
                    plt.ylabel('Elevation (°)')
                    #plt.colorbar(im1)
                    fig.colorbar(im2, ax=[ax2], orientation='vertical', fraction=0.015, pad=0.05)
                    plt.gca().set_aspect(2.0)
                    plt.title('Ground Truth Drop Probability')
                    #fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical', fraction=0.025, pad=0.05)
                    plt.savefig(os.path.join(save_folder, 'raydrop_gt.png'), bbox_inches='tight', pad_inches=0)
                    plt.close()

                # bev view (intensity only)
                if self.render_model.pred_intensity:
                    intensity_idx = -1
                    if self.render_model.pred_raydrop:
                        gt_pts = gt_pts[gt_pts[:,-1].int().bool()]
                        pred_pts = pred_pts[pred_pts[:,-1]<0.5]
                        intensity_idx = -2
                    # proj_x_pred, proj_y_pred = self.do_range_projection(pred_pts)
                    # proj_x_gt, proj_y_gt = self.do_range_projection(gt_pts)
                    
                    # plt.figure(dpi=500)
                    # #plt.subplot(121)
                    # plt.scatter(proj_x_pred.cpu().numpy(), proj_y_pred.cpu().numpy(), c=pred_pts[:,-1].cpu().numpy(), s=0.05, vmin=0.0, vmax=1.0)
                    # plt.colorbar()
                    # plt.title('pred intensity')
                    # plt.savefig('z_int_pred_rangemap.png')

                    # plt.figure(dpi=500)
                    # #plt.subplot(122)
                    # plt.scatter(proj_x_gt.cpu().numpy(), proj_y_gt.cpu().numpy(), c=gt_pts[:,-1].cpu().numpy(), s=0.05, vmin=0.0, vmax=1.0)
                    # plt.colorbar()
                    # plt.title('gt intensity')
                    # plt.savefig('z_int_gt_rangemap.png')

                    # bev is better
                    fig = plt.figure(dpi=500)
                    ax1 = plt.gca()
                    #ax1 = plt.subplot(121)
                    # im1 = ax1.scatter(pred_pts[:,0].cpu().numpy(),pred_pts[:,1].cpu().numpy(),c=pred_pts[:,-1].cpu().numpy()+1e-6, s=0.01, norm=LogNorm(vmin=1e-6,vmax=1.0))
                    im1 = ax1.scatter(pred_pts[:,0].cpu().numpy(),pred_pts[:,1].cpu().numpy(),c=pred_pts[:,intensity_idx].cpu().numpy()+1e-6, s=0.01, vmin=1e-6,vmax=1.0)
                    plt.colorbar(im1)
                    plt.xticks([-50, -25, 0, 25, 50])
                    plt.yticks([-50, -25, 0, 25, 50])
                    plt.xlabel('x (m)')
                    plt.ylabel('y (m)')
                    ax1.set_aspect(1.0)
                    plt.title('Predicted Intensity')
                    plt.savefig(os.path.join(save_folder, 'intensity_pred_bev.png'), bbox_inches='tight', pad_inches=0)
                    plt.close()

                    plt.figure(dpi=500)
                    ax2 = plt.gca()
                    #ax2 = plt.subplot(122)
                    # im2 = ax2.scatter(gt_pts[:,0].cpu().numpy(),gt_pts[:,1].cpu().numpy(),c=gt_pts[:,-1].cpu().numpy()+1e-6, s=0.01, norm=LogNorm(vmin=1e-6,vmax=1.0))
                    im2 = ax2.scatter(gt_pts[:,0].cpu().numpy(),gt_pts[:,1].cpu().numpy(),c=gt_pts[:,intensity_idx].cpu().numpy()+1e-6, s=0.01, vmin=1e-6,vmax=1.0)
                    plt.xticks([-50, -25, 0, 25, 50])
                    plt.yticks([-50, -25, 0, 25, 50])
                    plt.xlabel('x (m)')
                    plt.ylabel('y (m)')
                    ax2.set_aspect(1.0)
                    plt.colorbar(im2)
                    #fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical', fraction=0.05, pad=0.05)
                    plt.title('Ground Truth Intensity')
                    plt.savefig(os.path.join(save_folder, 'intensity_gt_bev.png'), bbox_inches='tight', pad_inches=0)
                    plt.close()

                    # bev is better
                    fig = plt.figure(dpi=500)
                    ax1 = plt.gca()
                    #ax1 = plt.subplot(121)
                    im1 = ax1.scatter(pred_pts[:,0].cpu().numpy(),pred_pts[:,1].cpu().numpy(),c=pred_pts[:,intensity_idx].cpu().numpy()+1e-6, s=0.01, norm=LogNorm(vmin=1e-6,vmax=1.0))
                    #im1 = ax1.scatter(pred_pts[:,0].cpu().numpy(),pred_pts[:,1].cpu().numpy(),c=pred_pts[:,intensity_idx].cpu().numpy()+1e-6, s=0.01, vmin=1e-6,vmax=1.0)
                    plt.colorbar(im1)
                    plt.xticks([-50, -25, 0, 25, 50])
                    plt.yticks([-50, -25, 0, 25, 50])
                    plt.xlabel('x (m)')
                    plt.ylabel('y (m)')
                    ax1.set_aspect(1.0)
                    plt.title('Predicted Intensity')
                    plt.savefig(os.path.join(save_folder, 'intensity_pred_bev_logscale.png'), bbox_inches='tight', pad_inches=0)
                    plt.close()

                    plt.figure(dpi=500)
                    ax2 = plt.gca()
                    #ax2 = plt.subplot(122)
                    im2 = ax2.scatter(gt_pts[:,0].cpu().numpy(),gt_pts[:,1].cpu().numpy(),c=gt_pts[:,intensity_idx].cpu().numpy()+1e-6, s=0.01, norm=LogNorm(vmin=1e-6,vmax=1.0))
                    #im2 = ax2.scatter(gt_pts[:,0].cpu().numpy(),gt_pts[:,1].cpu().numpy(),c=gt_pts[:,intensity_idx].cpu().numpy()+1e-6, s=0.01, vmin=1e-6,vmax=1.0)
                    plt.xticks([-50, -25, 0, 25, 50])
                    plt.yticks([-50, -25, 0, 25, 50])
                    plt.xlabel('x (m)')
                    plt.ylabel('y (m)')
                    ax2.set_aspect(1.0)
                    plt.colorbar(im2)
                    #fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical', fraction=0.05, pad=0.05)
                    plt.title('Ground Truth Intensity')
                    plt.savefig(os.path.join(save_folder, 'intensity_gt_bev_logscale.png'), bbox_inches='tight', pad_inches=0)
                    plt.close()

                # pred_pts_np = pred_pts.cpu().numpy().astype('float32')
                # colors1 = np.zeros_like(pred_pts_np)
                # colors1[:, 0] = 255
                # gt_pts_np = gt_pts.cpu().numpy().astype('float32')
                # colors2 = np.zeros_like(gt_pts_np)
                # colors2[:, 1] = 255
                # np.concatenate([np.concatenate([pred_pts_np, colors1], axis=1), np.concatenate([gt_pts_np, colors2], axis=1)]).tofile('z.bin')
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
        pts_all = batch_dict['points']
        occ_grids = batch_dict.get('grid', None)
        pts = []
        did_returns = []
        for bs in range(batch_size):
            batch_mask = pts_all[:, 0].int()==bs
            uni_feats.append(pts_feats[bs])
            pts.append(pts_all[batch_mask][:, 1:6])
            did_returns.append(batch_dict['did_return'][batch_mask][:, 1].bool())

        batch_ret = []
        rays = self.sample_rays(pts, did_returns, None, img_metas={'cam_intrinsic': batch_dict['cam_intrinsic'], 'cam_extrinsic': batch_dict['cam_extrinsic']} if self.use_predefine_rays else None, occ_grids=occ_grids)
        lidar_rays, _ = rays
        self.forward_ret_dict['targets'] = lidar_rays

        if self.render_model.pred_raydrop:
            for bs_idx in range(batch_size):
                dis = torch.norm(pts[bs_idx][:, :3], p=2, dim=-1)
                dis_mask = (dis > self.ray_sampler_cfg.close_radius) & (
                    dis < self.ray_sampler_cfg.get("far_radius", 100.0)
                ) & did_returns[bs_idx]
                self.forward_ret_dict['targets'][bs_idx]['pts_origin'] = pts[bs_idx][dis_mask]
                            
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

    def sample_rays(self, pts, did_returns, imgs, img_metas, occ_grids=None):
        #if self.training:
        lidar_ret = self.sample_lidar_rays(pts, did_returns, img_metas, occ_grids=occ_grids)
        #else:
        #    lidar_ret = self.sample_lidar_rays(pts, img_metas, test=True)
        return lidar_ret, None

    def sample_rays_test(self, pts, imgs, img_metas):
        lidar_ret = self.sample_lidar_rays(pts, img_metas, test=True)
        return lidar_ret, None

    def sample_lidar_rays(self, pts, did_returns, img_metas, test=False, occ_grids=None):
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
            assert len(pts) == 1
            # generate pre-defined camera rays

            K = img_metas['cam_intrinsic'][0].to(torch.float32)
            K[:2, :] /= self.downsample_rate
            coords_3d = self.image_coords @ torch.linalg.inv(K).T
            coords_3d_hom = torch.cat([coords_3d, coords_3d.new_ones((coords_3d.shape[0], 1))], dim=-1)
            coords_lidar = coords_3d_hom @ img_metas['cam_extrinsic'][0].T
            coords_lidar = coords_lidar[:, :3]
            ray_o = torch.zeros_like(coords_lidar)
            ray_d = coords_lidar - ray_o
            ray_ranges = torch.norm(ray_d, dim=-1, keepdim=True)
            ray_d = ray_d / ray_ranges
            self.predefine_rays = {
                'ray_o': ray_o.to(torch.float32).cuda() * self.render_model.scale_factor,
                'ray_d': ray_d.to(torch.float32).cuda()
            }

            for i in range(len(pts)):
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
