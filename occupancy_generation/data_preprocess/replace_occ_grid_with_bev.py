import sys
sys.path.append('/lpai/volumes/lmm-data-proc/liuhongsi/code/UniScene-V2/gs_render/diff-gaussian-rasterization')
import copy
import os
import shutil
import time
import random
import gc
from filelock import FileLock
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cv2
import numpy as np
import numba as nb
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import yaml
from pyquaternion import Quaternion
from gaussian_renderer import render
from gaussian_renderer import apply_depth_colormap, apply_semantic_colormap
from load_nuplan_map import load_occ_layout_nuplan

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import open3d
import dist_utils

cams = ['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0']

def generate_color_array(C=20, cmap_name="viridis"):
    cmap = cm.get_cmap(cmap_name, C)  # 获取色彩图并生成 C 个颜色
    colors = cmap(np.linspace(0, 1, C))[:, :3]  # 提取 RGB 部分
    colors = (colors * 255).astype(int)  # 转换到 0-255 范围
    return colors

nuscenes_cmap = generate_color_array(20)

# Normalize value to range 0-1
NORM = mcolors.Normalize(vmin=0, vmax=50)
CMAP = cm.get_cmap("viridis")
num_classes = 16
with open('gs_render/data_process/config_nuplan_r400.yaml', 'r') as stream:
    config = yaml.safe_load(stream)
point_cloud_range = config['pc_range']
occ_size = config['occ_size']
#point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
image_shape = (1080, 1920)
occupancy_size = [(point_cloud_range[3]-point_cloud_range[0])/occ_size[0], 
                (point_cloud_range[4]-point_cloud_range[1])/occ_size[1],
                (point_cloud_range[5]-point_cloud_range[2])/occ_size[2]]

occ_xdim = int((point_cloud_range[3] - point_cloud_range[0]) / occupancy_size[0])
occ_ydim = int((point_cloud_range[4] - point_cloud_range[1]) / occupancy_size[1])
occ_zdim = int((point_cloud_range[5] - point_cloud_range[2]) / occupancy_size[2])
voxel_num = occ_xdim*occ_ydim*occ_zdim
add_ego_car = True
occ_colors_map = np.array(
        [   
            [255, 158, 0, 255],  #  1 vehicle  orange
            [255, 99, 71, 255],  #  2 [place_holder]  Tomato
            [255, 140, 0, 255],  #  3 [place_holder]  Darkorange
            [255, 69, 0, 255],  #  4 [place_holder]  Orangered
            [233, 150, 70, 255],  #  5 czone_sign  Darksalmon
            [220, 20, 60, 255],  #  6 bicycle  Crimson
            [255, 61, 99, 255],  #  7 generic_object  Red
            [0, 0, 230, 255],  #  8 pedestrian  Blue
            [47, 79, 79, 255],  #  9 traffic_cone  Darkslategrey
            [112, 128, 144, 255],  #  10 barrier  Slategrey
            [0, 207, 191, 255],  # 11  background_surface  nuTonomy green  
            [175, 0, 75, 255],  #  12 None  
            [75, 0, 75, 255],  #  13  None 
            [112, 180, 60, 255],  # 14 None  
            [222, 184, 135, 255], # 15 None Burlywood 
            [0, 175, 0, 255],  # 16 None  Green
            [0, 0, 0, 255],  # unknown
        ]
    ).astype(np.uint8)

def number_to_rgb(value, vmin, vmax, cmap_name="viridis"):

    # Get the color from the colormap
    rgba = CMAP(NORM(value))
    
    # Convert RGBA to RGB in 0-255 range
    rgb = tuple(int(c * 255) for c in rgba[:3])
    return rgb


def replace_occ_grid_with_bev(input_occ, bevlayout, driva_area_idx=11, bev_replace_idx=[1, 5, 6],
                              occ_replace_new_idx=[17, 18, 19]):
    # self.classes= ['drivable_area','ped_crossing','walkway','stop_line','carpark_area','road_divider','lane_divider','road_block']
    # occ road [11] drivable area

    # default ped_crossing->18; stop_line->19 (del); roal_divider->20; lane_divider->21
    # default shape: input_occ: [200,200,16]; bevlayout: [18,200,200]

    roal_divider_mask = bevlayout[5, :, :].astype(np.uint8)
    lane_divider_mask = bevlayout[6, :, :].astype(np.uint8)

    roal_divider_mask = cv2.dilate(roal_divider_mask, np.ones((3, 3), np.uint8))
    lane_divider_mask = cv2.dilate(lane_divider_mask, np.ones((3, 3), np.uint8))

    bevlayout[5, :, :] = roal_divider_mask.astype(bool)
    bevlayout[6, :, :] = lane_divider_mask.astype(bool)

    n = len(bev_replace_idx)
    x_max, y_max = input_occ.shape[0], input_occ.shape[1]
    output_occ = input_occ.copy()  # numpy copy() ; tensor clone()
    bev_replace_mask = []
    for i in range(n):
        bev_replace_mask.append(bevlayout[bev_replace_idx[i]] == 1)

    for x in range(x_max):
        for y in range(y_max):
            for i in range(n):
                if bev_replace_mask[i][x, y]:
                    occupancy_data = input_occ[x, y, :]

                    if driva_area_idx in occupancy_data:
                        max_11_index = np.where(occupancy_data == driva_area_idx)
                        output_occ[x, y, max_11_index] = occ_replace_new_idx[i]
    return output_occ


def load_occ_layout(layout_path):
    # load layout data
    layout = np.load(open(layout_path, 'rb'), encoding='bytes', allow_pickle=True)
    #layout = layout['bev_map']
    layout = layout['arr_0']
    return layout

def obtain_points_label(occ):
    occ_index, occ_cls = occ[:, 0], occ[:, 1]
    occ = np.ones(voxel_num, dtype=np.int8)*11
    occ[occ_index[:]] = occ_cls  # (voxel_num)
    points = []
    for i in range(len(occ_index)):
        indice = occ_index[i]
        x = indice % occ_xdim
        y = (indice // occ_xdim) % occ_xdim
        z = indice // (occ_xdim*occ_xdim)
        point_x = (x + 0.5) / occ_xdim * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
        point_y = (y + 0.5) / occ_ydim * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
        point_z = (z + 0.5) / occ_zdim * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
        points.append([point_x, point_y, point_z])
    
    points = np.stack(points)
    points_label = occ_cls
    return points, points_label

class OccDataset(Dataset):
    def __init__(self, data_root, sample_tokens, infos, token_to_id):
        super().__init__()
        self.data_root = data_root
        self.sample_tokens = sample_tokens
        self.infos = infos
        self.token_to_id = token_to_id

    def __len__(self):
        return len(self.sample_tokens)
    
    @staticmethod
    def collect_fn(batch):
        assert len(batch) == 1
        return batch[0]

    def __getitem__(self, idx):
        sample_token = self.sample_tokens[idx]
        item_data = self.infos[self.token_to_id[sample_token]]

        data_root = self.data_root
        # cam_path = 'openscene_v1.1/sensor_blobs/mini'
        # occ_base_path = "/data/longhun/3D/nuscenes/data/nksr_occ"
        # #layout_base_path = "s3://guojiazhe/nuscenes/12hz_bevlayout_800_800/"
        # layout_base_path = "data/occ_gen/data/my_new_step2_12hz_800/train/bevmap_4"
        # is_vis = False


        occ_path = os.path.join(data_root, sample_token, sample_token+'.npz')
        #occ_label = load_occ_gt(occ_path=occ_path, grid_size=np.array([800, 800, 64]))
        
        #occ_label = np.load(occ_path)
        #xyz, semantics = obtain_points_label(occ_label)

        # pickle
        # with open(occ_path, 'rb') as f:
        #     occ = pickle.load(f)

        # npz
        try:
            occ = np.load(occ_path)['occ']
        except:
            with open('fail_files.txt', 'w') as f:
                f.write(f'{occ_path}\n')
            return dict(sample_token=item_data['token'])

        semantics = occ[:, -1]
        point_x = (occ[:, 0] + 0.5) / occ_xdim * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
        point_y = (occ[:, 1] + 0.5) / occ_ydim * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
        point_z = (occ[:, 2] + 0.5) / occ_zdim * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
        xyz = np.stack([point_x, point_y, point_z], axis=-1)

        # layout_path = os.path.join(layout_base_path, sample_token + '.npz')
        # bevlayout = load_occ_layout(layout_path=layout_path)

        # semantics = occ_label
        # semantics = replace_occ_grid_with_bev(semantics, bevlayout)

        # bevlayout = load_occ_layout_nuplan(os.path.join(layout_base_path, sample_token + '.npy'))
        

        semantics = torch.from_numpy(semantics) + 1
        xyz = torch.from_numpy(xyz).float()#.cuda()

        # l2e = Quaternion(item_data['lidar2ego_rotation']).transformation_matrix
        # l2e[:3, 3] = np.array(item_data['lidar2ego_translation'])
        # l2e = torch.from_numpy(l2e).cuda().float()
        # xyz = l2e[:3, :3] @ xyz.t() + l2e[:3, 3:4]
        # xyz = xyz.t()

        # add for filter floaters
        semantics_gt = semantics.view(-1, 1)  # (512, 512, 40) -> (10485760, 16)
        occ_mask = semantics_gt[:, 0] != 0
        pts = xyz[occ_mask].clone().cpu().numpy()
        colors = np.ones_like(pts)

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pts)
        pcd.colors = open3d.utility.Vector3dVector(colors)
        pcd, idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        semantics = semantics[idx]
        xyz = xyz[idx]

        # why not use idx?
        # inlier_xyz = torch.from_numpy(np.array(pcd.points))  # (n,3)
        # inlier_xyz_lidar = torch.inverse(l2e) @ torch.hstack(
        #     [inlier_xyz, torch.ones(inlier_xyz.shape[0], 1)]).cuda().float().T
        # inlier_xyz = inlier_xyz_lidar[:3, :].T

        # inlier_idx = torch.vstack(
        #     [torch.clamp(800 * (inlier_xyz[:, 0] - pc_range[0]) / (pc_range[3] - pc_range[0]), 0, 800 - 1),
        #      torch.clamp(800 * (inlier_xyz[:, 1] - pc_range[1]) / (pc_range[4] - pc_range[1]), 0, 800 - 1),
        #      torch.clamp(64 * (inlier_xyz[:, 2] - pc_range[2]) / (pc_range[5] - pc_range[2]), 0, 64 - 1)]).long()

        # filter_mask = torch.zeros((800, 800, 64)).cuda()
        # filter_mask[inlier_idx[0, :], inlier_idx[1, :], inlier_idx[2, :]] = 1.0
        # semantics[filter_mask.cpu() < 1] = 0

        # load semantic data ------------------------------------------------------------------------------
        semantics_gt = semantics.view(-1, 1)  # (200, 200, 16) -> (640000, 16)
        occ_mask = semantics_gt[:, 0] != 0
        semantics_gt = semantics_gt.permute(1, 0)    

        return dict(
            semantics_gt=semantics_gt,
            xyz=xyz,
            cam_infos=item_data['cams'],
            occ_mask=occ_mask,
            sample_token=item_data['token']
        )

def save_render_result(camera_semantic, camera_depth, base_path, sample_token, save_depth_as_uint16):
    if save_depth_as_uint16:
        max_depth = 100.0
        camera_depth = np.clip(camera_depth / max_depth * 65535, 0, 65535).astype(np.uint16)


    if not os.path.exists(os.path.join(base_path, sample_token)):
        os.makedirs(os.path.join(base_path, sample_token))
    sem_data_all_path = os.path.join(base_path, sample_token, "semantic.npz")
    if save_depth_as_uint16:
        depth_data_all_path = os.path.join(base_path, sample_token, "depth_data_uint16.npz")
    else:
        depth_data_all_path = os.path.join(base_path, sample_token, "depth_data.npz")
    np.savez_compressed(sem_data_all_path, camera_semantic)
    np.savez_compressed(depth_data_all_path, camera_depth)

#def render_occ_semantic_map(item_data, base_path, occ_base_path, layout_base_path, is_vis=False):
def render_occ_semantic_map(batch, data_root, base_path, save_executor, is_vis=False, gs_scale=0.125, save_depth_as_uint16=False):
    if 'xyz' not in batch:
        return None

    semantics_gt = batch['semantics_gt']
    xyz = batch['xyz'].cuda()
    cam_infos = batch['cam_infos']
    occ_mask = batch['occ_mask']
    sample_token = batch['sample_token']

    # if os.path.exists(os.path.join(base_path, sample_token, "semantic.npz")):
    #     return sample_token

    opacity = (semantics_gt.clone() != 0).float()
    opacity = opacity.permute(1, 0).cuda()

    semantics = torch.zeros((20, semantics_gt.shape[1])).cuda().float()
    color = torch.zeros((3, semantics_gt.shape[1])).cuda()
    for i in range(20):
        semantics[i] = semantics_gt == i

    rgb = color.permute(1, 0).float()
    feat = semantics.permute(1, 0).float()
    rot = torch.zeros((xyz.shape[0], 4)).cuda().float()
    rot[:, 0] = 1
    scale = torch.ones((xyz.shape[0], 3)).cuda().float() * gs_scale

    camera_semantic = []
    camera_depth = []

    for cam in cams:
        cam_info = cam_infos[cam] #item_data['cams'][cam]
        camera_intrinsic = np.eye(3).astype(np.float32)
        camera_intrinsic[:3, :3] = cam_info['cam_intrinsic']
        camera_intrinsic = torch.from_numpy(camera_intrinsic).cuda().float()

        c2e = np.eye(4).astype(np.float32)
        c2e[:3, :3] = cam_info['sensor2lidar_rotation']#Quaternion(cam_info['sensor2lidar_rotation']).transformation_matrix
        c2e[:3, 3] = np.array(cam_info['sensor2lidar_translation'])
        c2e = torch.from_numpy(c2e).cuda().float()

        camera_extrinsic = c2e

        render_pkg = render(
            camera_extrinsic, camera_intrinsic, image_shape,
            xyz[occ_mask], rgb[occ_mask], feat[occ_mask], rot[occ_mask], scale[occ_mask], opacity[occ_mask],
            bg_color=[0, 0, 0]
        )

        render_color = render_pkg['render_color']
        render_semantic = render_pkg['render_feat']
        render_depth = render_pkg['render_depth']
        render_alpha = render_pkg['render_alpha']

        
        if is_vis:
            cam_path = data_root
            vis_path = os.path.join(base_path, '..', 'vis')
            os.makedirs(vis_path, exist_ok=True)
            device = 'cuda'
            os.makedirs(os.path.join(vis_path, sample_token), exist_ok=True)
            pts_vis = np.zeros((xyz.shape[0], 6), dtype=np.float32)
            pts_vis[:, :3] = xyz.cpu().numpy()
            pts_vis[:, 3:] = nuscenes_cmap[semantics_gt.squeeze().cpu().numpy().astype(int)].astype('float32')
            pts_vis.tofile(os.path.join(vis_path, sample_token, 'colored_points.bin'))


            os.makedirs(os.path.join(vis_path, sample_token, "semantic_color"), exist_ok=True)
            sem_save_path = os.path.join(vis_path, sample_token, "semantic_color", cam + ".jpg")
            with open(sem_save_path, "wb") as f:
                sem_data = apply_semantic_colormap(render_semantic).cpu().permute(1, 2, 0).detach().numpy() * 255
                f.write(cv2.imencode('.jpg', sem_data)[1])

            os.makedirs(os.path.join(vis_path, sample_token, "depth_color"), exist_ok=True)
            depth_save_path = os.path.join(vis_path, sample_token, "depth_color", cam + ".jpg")
            with open(depth_save_path, "wb") as f:
                render_depth = torch.clamp(render_depth, min=0.1, max=40.0)
                dep_data = apply_depth_colormap(render_depth).cpu().permute(1, 2, 0).detach().numpy() * 255
                f.write(cv2.imencode('.jpg', dep_data)[1])

            os.makedirs(os.path.join(vis_path, sample_token, "image_color"), exist_ok=True)
            image_save_path = os.path.join(vis_path, sample_token, "image_color", cam + ".jpg")
            ori_img = cv2.imread(os.path.join(data_root, cam_info['data_path']))
            ori_img = cv2.resize(ori_img, (sem_data.shape[1], sem_data.shape[0]))
            blended = cv2.addWeighted(ori_img, 0.5, sem_data.astype('uint8'), 0.5, 0.0)
            cv2.imwrite(image_save_path, blended)
            #shutil.copy(os.path.join(data_root, '/'.join(cam_info['data_path'].split('/')[3:])), image_save_path)


            # visualize proj
            os.makedirs(os.path.join(vis_path, sample_token, "proj"), exist_ok=True)
            proj_save_path = os.path.join(vis_path, sample_token, "proj", cam + ".jpg")    
            # origin point cloud
            means3D_ = xyz
            means3D_h = torch.cat([means3D_, torch.ones(means3D_.shape[0], 1).type_as(means3D_)], dim=1).detach()

            # through ndc (principle point centered assumption)
            os.makedirs(os.path.join(vis_path, sample_token, "proj_ndc"), exist_ok=True)
            proj_save_path = os.path.join(vis_path, sample_token, "proj_ndc", cam + ".jpg") 
            width, height = image_shape[1], image_shape[0]
            fx = float(camera_intrinsic[0][0])
            fy = float(camera_intrinsic[1][1])
            cx = float(camera_intrinsic[0][2])
            cy = float(camera_intrinsic[1][2])
            from gaussian_renderer import focal2fov, get_projection_matrix_c
            from diff_gaussian_rasterization import _C
            import math
            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)
            tan_fov_x = math.tan(FovX * 0.5)
            tan_fov_y = math.tan(FovY * 0.5)

            extrinsics = torch.inverse(c2e) # w2c

            # projection_matrix = get_projection_matrix(near=0.1, far=200.0, fov_x=FovX, fov_y=FovY).transpose(0, 1).cuda()
            projection_matrix = get_projection_matrix_c(fx, fy, cx, cy, width, height, 0.1, 200.0).transpose(0, 1).cuda()
            world_view_transform = extrinsics.transpose(0, 1).cuda()
            full_projection = world_view_transform.float() @ projection_matrix
            cam_3D = means3D_h @ full_projection.to(device)
            cam_3D = cam_3D / cam_3D[:, 3:4]
            x = ((cam_3D[:, 0] + 1.0) * image_shape[1] - 1.0) * 0.5
            y = ((cam_3D[:, 1] + 1.0) * image_shape[0] - 1.0) * 0.5
            visiable_mask = _C.mark_visible(means3D_, world_view_transform.to(device), full_projection.to(device))
            visiable_mask = visiable_mask & (x < image_shape[1]) & (y < image_shape[0]) & (x > 0) & (y > 0)

            # through intrinsic
            # cam_3D = means3D_h @ torch.inverse(c2e).T.to(device)
            # uvs = cam_3D[:, :3] @ camera_intrinsic.T.to(device)
            # d = uvs[:, 2]
            # uvs = uvs / uvs[:, 2:3]
            # x, y = uvs[:, 0], uvs[:, 1]
            # visiable_mask = (d > 0.1) & (x < image_shape[1]) & (y < image_shape[0]) & (x > 0) & (y > 0)

            vis_x = x[visiable_mask].cpu().numpy()
            vis_y = y[visiable_mask].cpu().numpy()
            #vis_d = d[visiable_mask].cpu().numpy()
            vis_sem = (semantics_gt.squeeze())[visiable_mask.cpu()].numpy()
            ori_img = cv2.imread(os.path.join(data_root, cam_info['data_path']))
            for p in range(visiable_mask.sum()):
                #_rgb = number_to_rgb(vis_d[p], 0, 100)
                #_rgb = (255, 0, 0)
                _rgb = nuscenes_cmap[int(vis_sem[p])].tolist()
                cv2.circle(ori_img, (int(vis_x[p]), int(vis_y[p])), 2, _rgb, -1)
            cv2.imwrite(proj_save_path, ori_img)

        semantic = torch.max(render_semantic, dim=0)[1].squeeze().cpu().numpy().astype(np.int8)
        camera_semantic.append(semantic)

        depth_data = render_depth[0].detach().cpu().numpy()
        camera_depth.append(depth_data)

    ################################### update object to local ####################################
    # if not os.path.exists(os.path.join(base_path, sample_token)):
    #     os.makedirs(os.path.join(base_path, sample_token))
    # sem_data_all_path = os.path.join(base_path, sample_token, "semantic.npz")
    # depth_data_all_path = os.path.join(base_path, sample_token, "depth_data.npz")
    # np.savez(sem_data_all_path, camera_semantic)
    # np.savez(depth_data_all_path, camera_depth)
    ################################################################################################

    # fast save
    for i, cam in enumerate(cams):
        save_path = os.path.join(base_path, cam)
        # os.makedirs(save_path, exist_ok=True)
        save_executor.submit(save_render_result, camera_semantic[i], camera_depth[i], save_path, sample_token, save_depth_as_uint16)

    return sample_token
    #print(f"Rendered {sample_token} to {base_path}/{sample_token}")

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default="data/nuplan-all/sensor_blobs/trainval")
    parser.add_argument('--pkl_path', type=str, default='data/nuplan_pkls/trainval/nuplan_trainval_10hz_val.pkl')
    parser.add_argument('--version', type=str, default='trainval')
    parser.add_argument('--occ_path', type=str, default="/lpai/dataset/nuplan-occ/1-1-01/GT_occ_fast3_10hzval_r400/dense_voxels_with_semantic")
    parser.add_argument('--processed_path', type=str, default=None)
    parser.add_argument('--layout_path', type=str, default="/lpai/volumes/ad-lmm-data-proc-bd-ga/liuhongsi/code/occgen_dev/data/nuplan_bev/mini/train")
    parser.add_argument('--render_path', type=str, default="data/nuplan-occ-render-trainval_val/")
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_interval', type=int, default=200)
    parser.add_argument('--gs_scale', type=float, default=0.01)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=500)
    parser.add_argument('--save_depth_as_uint16', action='store_true', default=False)
    
    args = parser.parse_args()

    if dist_utils.is_dist():
        dist_utils.ddp_setup()
        rank = dist_utils.get_rank()
        world_size = dist_utils.get_world_size()
        print(f'Rank: {rank}, World size: {world_size}')
    else:
        rank = 0
        world_size = 1

    with open(args.pkl_path, 'rb') as f:
        infos = pickle.load(f)
    token_to_id = {}
    for i, info in enumerate(infos['infos']):
        token_to_id[info['token']] = i

    sample_tokens_list = infos['scene_tokens']
    render_base_path = os.path.join(args.render_path, args.version)
    occ_base_path = os.path.join(args.occ_path, args.version)
    layout_base_path = os.path.join(args.layout_path, args.version)

    all_train_items = len(sample_tokens_list)
    print('Total seq:', all_train_items)
    args.end_idx = min(all_train_items-1, args.end_idx)
    index_list = list(range(args.start_idx, args.end_idx+1))
    if len(index_list) == 0:
        index_list = list(range(all_train_items))

    if dist_utils.is_dist():
        index_list = np.array_split(index_list, world_size)[rank].tolist()
        #print(f'Rank: {rank}, proscess {index_list[0]} - {index_list[-1]}')
        print(f'Rank: {rank}, proscess {index_list}')

    save_executor = ThreadPoolExecutor(max_workers=8)
    #random.shuffle(index_list)
    n_processed = 0
    if os.path.exists(os.path.join(args.render_path, args.version, 'CAM_F0')) or (args.processed_path is not None):
        with FileLock('occrender_scan.lock'):
            if args.processed_path is None:
                exists_tokens = set(entry.name for entry in os.scandir(os.path.join(args.render_path, args.version, 'CAM_F0')))
            else:
                CAMS = ['CAM_F0', 'CAM_B0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_R0', 'CAM_R1', 'CAM_R2']
                exists_tokens = set()
                for i, cam in enumerate(CAMS):
                    print(f'checking processed samples... {cam}')
                    exists_single = set(entry.name for entry in os.scandir(os.path.join(args.processed_path, args.version, cam)))
                    if i == 0:
                        exists_tokens |= exists_single
                    else:
                        exists_tokens &= exists_single
    else:
        exists_tokens = set()
    print(f'Already processed: {len(exists_tokens)}')
    for index in index_list:
        sample_tokens = sample_tokens_list[index]
        # check exist
        unprocessed_tokens = []
        for sample_token in sample_tokens:
            if sample_token not in exists_tokens:
                unprocessed_tokens.append(sample_token)
        sample_tokens = unprocessed_tokens
        
        if len(sample_tokens) == 0:
            print(f'seq {index} processed')
            continue

        dataloader = DataLoader(OccDataset(args.occ_path, sample_tokens, infos['infos'], token_to_id), batch_size=1, collate_fn=OccDataset.collect_fn, shuffle=False, num_workers=0, pin_memory=True)
        #items = pickle.load(open(os.path.join(args.dataset_path, 'openscene_v1.1/meta_datas/mini', pkls[index]), 'rb'))
        #for item in items:

        for batch in tqdm(dataloader, desc=f'Rank: {rank}, Index: {index}'):
            
        #try:
            #item = items_data[index]

            #t1 = time.perf_counter()
            sample_token = render_occ_semantic_map(
                batch,
                data_root=args.dataset_path,
                base_path=render_base_path,
                save_executor=save_executor,
                is_vis=args.vis or (n_processed % args.vis_interval) == 0,
                gs_scale=args.gs_scale,
                save_depth_as_uint16=args.save_depth_as_uint16
            )
            if sample_token is None:
                print(f'Failed: {batch["sample_token"]}')
            else:
                n_processed += 1
            # if sample_token is None:
            #     with open('./error_list.txt', 'a') as f:
            #         f.write(str(sample_token), '\n')
                    
            #t2 = time.perf_counter()
            #print('Latency:', t2-t1)
        gc.collect()
        # except Exception as e:
        #     print(f"Error: {e}")
        #     with open("./error_list.txt", 'a') as f:
        #         f.write(str(index) + '\n')
        #     continue

        # with open("./success_list.txt", 'a') as f:
        #     f.write(str(index) + '\n')
    
    print(f'Rank: {rank} has processed {n_processed} frames')
    if dist_utils.is_dist():
        dist_utils.ddp_cleanup()
