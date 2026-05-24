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
from dda3d_gpu import dda3d_gpu, raycast_gpu, raycast_gpu_withorigins, raycast_gpu_withorigins_2
from uniscenev2_lidar.utils.transform_utils import cartesian_to_spherical
from uniscenev2_lidar.utils.common_utils import torch_lexsort

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
        self.use_raycast_prior_sampler = render_ssl_cfg['sampler_cfg']['initial_sampler'] == 'RaycastPriorSampler'
        self.filter_ambiguous_rays = model_cfg.get('filter_ambiguous_rays', None)
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
            if preds_dict[bs_idx]['depth'].numel() > 0:
                i_loss_dict = self.render_model.loss(preds_dict[bs_idx], lidar_targets[bs_idx])
            else:
                i_loss_dict = dict()
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

            pred_pts = (lidar_rays[bs]['ray_o'] + batch_dict['preds_dict'][bs]['depth'] * lidar_rays[bs]['ray_d']) / self.render_model.scale_factor
            if intensity_pred is not None:
                pred_pts = torch.cat([pred_pts, intensity_pred.sigmoid()], dim=-1)
            if raydrop_pred is not None and self.drop_ray_when_pred:
                if self.use_gt_drop:
                    pred_pts = pred_pts[lidar_rays[bs]['did_return'].squeeze()]
                else:
                    pred_pts = pred_pts[(raydrop_pred.sigmoid()<0.5).squeeze()]
                    #print((raydrop_pred.sigmoid()>0.5).sum())

            if 'lidar_idx' in lidar_rays[bs]:
                did_return = lidar_rays[bs]['did_return'].squeeze()
                lidar_idx = lidar_rays[bs]['lidar_idx'][did_return]
                pred_pts = torch.cat([pred_pts, lidar_idx.unsqueeze(-1)], dim=-1) 
            pred_dicts['pc_out'].append(pred_pts)

            if 0: # 'pts_origin' in lidar_rays[bs]:
                gt_pts = lidar_rays[bs]['pts_origin'][:, :3]
            else:
                did_return = lidar_rays[bs]['did_return'].squeeze()
                gt_pts = (lidar_rays[bs]['ray_o'][did_return] + lidar_rays[bs]['depth'][did_return] * lidar_rays[bs]['ray_d'][did_return]) / self.render_model.scale_factor

            if 'lidar_idx' in lidar_rays[bs]:
                lidar_idx = lidar_rays[bs]['lidar_idx'][did_return]
                gt_intensity = lidar_rays[bs]['intensity'][did_return].squeeze()
                gt_pts = torch.cat([gt_pts, gt_intensity.unsqueeze(-1), lidar_idx.unsqueeze(-1)], dim=-1)
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
        pts_all = batch_dict['points']
        sensor_loc = batch_dict.get('sensor_loc', torch.zeros_like(pts_all[:, :4]))
        occ_grids = batch_dict.get('grid', None)
        pts = []
        did_returns = []
        sensor_locs = []
        for bs in range(batch_size):
            batch_mask = pts_all[:, 0].int()==bs
            uni_feats.append(pts_feats[bs])
            pts.append(pts_all[batch_mask][:, 1:])
            sensor_locs.append(sensor_loc[batch_mask][:, 1:])
            did_returns.append(batch_dict['did_return'][batch_mask][:, 1].bool())

        batch_ret = []
        rays = self.sample_rays(pts, did_returns, None, None, occ_grids=occ_grids, sensor_locs=sensor_locs, lidar_chosen_mask=batch_dict.get('lidar_chosen_mask', None))
        lidar_rays, _ = rays
        self.forward_ret_dict['targets'] = lidar_rays

        # if self.render_model.pred_raydrop:
        #     for bs_idx in range(batch_size):
        #         dis = torch.norm(pts[bs_idx][:, :3], p=2, dim=-1)
        #         dis_mask = (dis > self.ray_sampler_cfg.close_radius) & (
        #             dis < self.ray_sampler_cfg.get("far_radius", 100.0)
        #         ) & did_returns[bs_idx]
        #         self.forward_ret_dict['targets'][bs_idx]['pts_origin'] = pts[bs_idx][dis_mask]
                            
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
                    origins=i_ray_o, directions=i_ray_d, depths=i_ray_depth, hits=lidar_rays[bs_idx]['hits'], grid=lidar_rays[bs_idx]['grid'], scale_factor=lidar_rays[bs_idx]['scale_factor'], ring_idx=lidar_rays[bs_idx]['ring_idx'], lidar_idx=lidar_rays[bs_idx]['lidar_idx'], nears=lidar_rays[bs_idx]['nears'], fars=lidar_rays[bs_idx]['fars']
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
                # lidar_ray_bundle = RayBundle(
                #     origins=i_ray_o, directions=i_ray_d, depths=i_ray_depth
                # )
                lidar_ray_bundle = RayBundle(
                    origins=i_ray_o, directions=i_ray_d, depths=i_ray_depth, hits=lidar_rays[bs_idx]['hits'], grid=lidar_rays[bs_idx]['grid'], scale_factor=lidar_rays[bs_idx]['scale_factor'], ring_idx=lidar_rays[bs_idx]['ring_idx'], lidar_idx=lidar_rays[bs_idx]['lidar_idx'], nears=lidar_rays[bs_idx]['nears'], fars=lidar_rays[bs_idx]['fars']
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
            pred_pts.detach().cpu().numpy().astype('float32').tofile('z_pred.bin')

            target_pts = lidar_rays[0]['depth'] * lidar_rays[0]['ray_d'] / self.render_model.scale_factor
            target_pts.detach().cpu().numpy().astype('float32').tofile('z_gt.bin')

            pred_pts, target_pts = pred_pts.detach().cpu().numpy(), target_pts.cpu().numpy()
            rad = np.zeros((target_pts.shape[0], 3))
            rad[:, 0] = 255
            white = 255 * np.ones((pred_pts.shape[0], 3))
            for_vis = np.concatenate([np.concatenate([target_pts, rad], axis=-1), np.concatenate([pred_pts, white], axis=-1)], axis=0)
            for_vis.astype('float32').tofile('z_gt_pred.bin')
        
        debug_pred=False
        if debug_pred:
            self.generate_predicted_pc(batch_dict)
        
        return batch_dict

    def sample_rays(self, pts, did_returns, imgs, img_metas, occ_grids=None, sensor_locs=None, lidar_chosen_mask=None):
        #if self.training:
        lidar_ret = self.sample_lidar_rays(pts, did_returns, img_metas, occ_grids=occ_grids, sensor_locs=sensor_locs, lidar_chosen_mask=lidar_chosen_mask)
        #else:
        #    lidar_ret = self.sample_lidar_rays(pts, img_metas, test=True)
        return lidar_ret, None

    def sample_rays_test(self, pts, imgs, img_metas):
        lidar_ret = self.sample_lidar_rays(pts, img_metas, test=True)
        return lidar_ret, None

    def sample_lidar_rays(self, pts, did_returns, img_metas, test=False, occ_grids=None, sensor_locs=None, lidar_chosen_mask=None):
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
            for i in range(len(pts)):
                if occ_grids is not None and self.drop_collisionless_rays:
                    grid = occ_grids[i]
                    lidar_directions = self.predefine_rays['ray_d'].clone().contiguous()
                    lidar_origins = self.predefine_rays['ray_o'].clone().contiguous()
                    ring_idx = None
                    if 'ring_idx' in self.predefine_rays:
                        ring_idx = self.predefine_rays['ring_idx'].clone().contiguous()
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
                        'intersection_mask': intersection_mask,
                        
                        "scale_factor": self.render_model.scale_factor,
                        'grid': grid.permute(2, 1, 0)[None].contiguous(), # [D W H]
                        'ring_idx': ring_idx[intersection_mask] if ring_idx is not None else None
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
            if lidar_chosen_mask is not None:
                lidar_mask = (lidar_chosen_mask[i][lidar_pc[:, -1].int()]).bool()
                dis_mask &= lidar_mask
            lidar_pc = lidar_pc[dis_mask]
            did_return = did_return[dis_mask]
            lidar_points = lidar_pc[:, :3]
            if sensor_locs is None:
                lidar_origins = torch.zeros_like(lidar_points)
            else:
                lidar_origins = sensor_locs[i][dis_mask]
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)
            lidar_directions = lidar_directions / lidar_ranges
            lidar_intensity = lidar_pc[:, 3:4]

            assert self.drop_collisionless_rays
            if occ_grids is not None and self.drop_collisionless_rays:
                grid = occ_grids[i]
                num_rays = lidar_directions.shape[0]
                intersection_mask = torch.zeros(num_rays, dtype=torch.bool, device=lidar_directions.device)
                W, H, D = grid.shape
                if self.use_raycast_prior_sampler:
                    raise NotImplementedError("deprecated implement")
                    # bug: 由于量化，不能用这种方式获得交点（可以不是正确的AABB），并且这种方式只能采样到第一次碰到的OCC，对噪声不鲁棒，弃用
                    # hits = torch.zeros_like(lidar_origins[:,0]).to(torch.float)
                    # raycast_gpu_withorigins_2(
                    #     lidar_directions, grid, intersection_mask, hits, lidar_origins,
                    #     *self.pc_range[:3], *self.unified_voxel_size, W, H, D, num_rays,
                    # )

                    # bug: 量化导致碰撞点不在射线上，并且也只能采样到第一次碰到的OCC，弃用
                    # hits = torch.zeros_like(lidar_origins).to(torch.int32)
                    # raycast_gpu_withorigins(
                    #     lidar_directions, grid, intersection_mask, hits, lidar_origins,
                    #     *self.pc_range[:3], *self.unified_voxel_size, W, H, D, num_rays,
                    # )
                    # # coord to real points
                    # hits = hits.to(torch.float32)
                    # hits = (hits + 0.5) * torch.from_numpy(self.unified_voxel_size.astype('float32')).cuda() + \
                    #     torch.from_numpy(self.pc_range[:3].astype('float32')).cuda()

                    # hits = hits[intersection_mask]
                else:
                    hits = None
                    dda3d_gpu(
                        lidar_directions, grid, intersection_mask,
                        *self.pc_range[:3], *self.unified_voxel_size, W, H, D, num_rays,
                    )
                
                nears, fars, collider_valid_mask = self.render_model.collider._intersect_with_aabb(lidar_origins * self.render_model.scale_factor, lidar_directions, self.render_model.collider.scene_box)
                nears = nears[..., None]
                fars = fars[..., None]
                intersection_mask &= collider_valid_mask


                if self.filter_ambiguous_rays is not None:
                    # filter out edge points
                    valid_mask = torch.zeros((lidar_directions.shape[0],), device=lidar_directions.device, dtype=torch.bool)
                    for lidar_id in range(5):
                        lidar_mask = lidar_pc[:, -1].int() == lidar_id
                        if not lidar_mask.any():
                            continue
                        sphere_coords = cartesian_to_spherical(lidar_directions[lidar_mask], to_deg=True)
                        sphere_coords[:, 0] += 180
                        sphere_coords[:, 1] += 90

                        if self.filter_ambiguous_rays == 'grid_filter':
                            _mask = self.grid_filter(sphere_coords, lidar_ranges[lidar_mask].squeeze(), cell_size=[0.2, 0.15], value_tol=0.5,  keep_repr='all')
                        elif self.filter_ambiguous_rays == 'edge_filter':
                            _sphere_coords = sphere_coords.clone()
                            _sphere_coords[:, 1] = lidar_pc[lidar_mask, -2]
                            _mask = self.edge_filter(_sphere_coords, lidar_ranges[lidar_mask].squeeze(), cell_size=0.5, value_tol=0.5)
                        else:
                            raise NotImplementedError
                        valid_mask[torch.nonzero(lidar_mask, as_tuple=False).squeeze()[_mask]] = True

                        # import matplotlib.pyplot as plt
                        # plt.figure(dpi=600)
                        # plt.scatter(sphere_coords[:, 0].cpu().numpy(), sphere_coords[:, 1].cpu().numpy(), s=0.1)
                        # plt.scatter(sphere_coords[~_mask, 0].cpu().numpy(), sphere_coords[~_mask, 1].cpu().numpy(), s=0.1, c='r')
                        # plt.savefig(f'z_grid_filter{lidar_id}.png')

                    intersection_mask &= valid_mask

                lidar_origins = lidar_origins[intersection_mask]
                lidar_directions = lidar_directions[intersection_mask]
                lidar_intensity = lidar_intensity[intersection_mask]
                lidar_ranges = lidar_ranges[intersection_mask]
                lidar_points = lidar_points[intersection_mask]
                did_return = did_return[intersection_mask]
                lidar_pc = lidar_pc[intersection_mask]
                nears = nears[intersection_mask]
                fars = fars[intersection_mask]

            else:
                raise NotImplementedError
                hits = None
                grid = None
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
                    'intersection_mask': intersection_mask,
                    'hits': hits * self.render_model.scale_factor if hits is not None else None,
                    'grid': grid.permute(2, 1, 0)[None].contiguous(), # [D W H]
                    'ring_idx': lidar_pc[:, -2].int(),
                    'lidar_idx': lidar_pc[:, -1].int(),
                    'original_points': lidar_pc,
                    "nears": nears,
                    "fars": fars
                }
            )
            # TODO: debug
            if lidar_origins.shape[0] == 0 and (lidar_chosen_mask is not None):
                print(lidar_chosen_mask[i])
        return lidar_ret

    def grid_filter(
        self,
        coords: torch.Tensor,      # [N, 2], float32/64, 连续坐标
        values: torch.Tensor,      # [N],    任意实数张量
        cell_size: float = 0.1,    # 网格边长 (世界坐标)
        value_tol: float = 1.0,    # 同格子内 “数值最大差” 允许范围
        keep_repr: str = "mean",   # same cell 时怎么保留代表: "all" | "mean" | "first"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert coords.ndim == 2 and coords.size(1) == 2
        assert values.ndim == 1 and values.size(0) == coords.size(0)
        assert keep_repr in {"all", "mean", "first"}

        N = coords.size(0)
        device = coords.device
        dtype  = coords.dtype

        # ① 量化到网格
        if isinstance(cell_size, float):
            cell_size_x = cell_size_y = cell_size
        else:
            cell_size_x, cell_size_y = cell_size
        grid_idx = torch.stack((
            torch.floor(coords[:, 0] / cell_size_x),  # X 方向量化
            torch.floor(coords[:, 1] / cell_size_y)   # Y 方向量化
        ), dim=1).to(torch.int64)  # [N, 2]

        # ② 建立哈希 (避免多列 unique 的开销)
        #   h = i + j * (max_i + 1)
        max_i = grid_idx[:, 0].max().item() + 1
        hash_id = grid_idx[:, 0] + grid_idx[:, 1] * max_i           # [N]

        uniq, inverse, counts = torch.unique(
            hash_id, return_inverse=True, return_counts=True
        )  # uniq[K], inverse[N]∈[0,K-1], counts[K]

        K = uniq.size(0)                      # 不同格子数
        # vals_sum = torch.zeros(
        #     K, dtype=values.dtype, device=device
        # ).scatter_add_(0, inverse, values)

        # ③ 每格子的均值与 (max - min)
        vals_max = torch.full((K,), -float("inf"), device=values.device)
        vals_min = torch.full((K,),  float("inf"), device=values.device)
        vals_max.scatter_reduce_(0, inverse, values, reduce="amax", include_self=True)
        vals_min.scatter_reduce_(0, inverse, values, reduce="amin", include_self=True)
        # vals_mean = vals_sum / counts.to(values.dtype)
        diff_per_cell = vals_max - vals_min                         # 极差

        # ④ 标记要留下的点
        #   策略 A: 仅当该格子的极差 ≤ tol 时，把整格都留下
        keep_cell = diff_per_cell <= value_tol                      # [K] bool
        mask_cell = keep_cell[inverse]                              # [N] bool

        return mask_cell
        if keep_repr == "all":
            mask = mask_cell
            out_coords = coords[mask]
            out_vals   = values[mask]


    def edge_filter(
        self,
        coords: torch.Tensor,      # [N, 2], float32/64, 连续坐标
        values: torch.Tensor,      # [N],    任意实数张量
        cell_size: float = 0.1,    # 网格边长 (世界坐标)
        value_tol: float = 1.0,    # 同格子内 “数值最大差” 允许范围
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 排序 (按 y 再按 x)
        sorted_idx = torch_lexsort((coords[:, 0], coords[:, 1]))
        coords_sorted = coords[sorted_idx]
        values_sorted = values[sorted_idx]

        # 判断相邻点是否同一行 (y 相同)
        same_row = coords_sorted[1:, 1] == coords_sorted[:-1, 1]
        
        # x 方向差与 value 差
        x_diff = torch.abs(coords_sorted[1:, 0] - coords_sorted[:-1, 0])
        value_diff = torch.abs(values_sorted[1:] - values_sorted[:-1])

        # 满足：同一行，x距离小于阈值，value差大于阈值
        bad_pairs = same_row & (x_diff < cell_size) & (value_diff > value_tol)

        # 构建 mask：滤除 bad_pairs 中的两个点
        mask = torch.ones(len(coords_sorted), dtype=torch.bool, device=coords.device)
        mask[1:][bad_pairs] = False
        mask[:-1][bad_pairs] = False

        # 恢复原顺序
        good_idx = sorted_idx[mask]
        mask = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.bool)
        mask[good_idx] = True
        return mask
        #return coords[good_idx], values[good_idx]