import os
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
try:
    from gaussian_renderer import render
    from gaussian_renderer import apply_depth_colormap, apply_semantic_colormap
except:
    pass
from gsplat import rasterization

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
DEFAULT_CONFIG_FILE = 'gs_render/data_process/config_nuplan_r200.yaml'
num_classes = 16

def load_config(config_path: str = DEFAULT_CONFIG_FILE):
    global config, point_cloud_range, occ_size, occupancy_size
    global occ_xdim, occ_ydim, occ_zdim, occ_size, voxel_num, image_shape
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    point_cloud_range = config['pc_range']
    occ_size = config['occ_size']
    image_shape = (1080, 1920)
    occupancy_size = [(point_cloud_range[3]-point_cloud_range[0])/occ_size[0], 
                    (point_cloud_range[4]-point_cloud_range[1])/occ_size[1],
                    (point_cloud_range[5]-point_cloud_range[2])/occ_size[2]]

    occ_xdim = int((point_cloud_range[3] - point_cloud_range[0]) / occupancy_size[0])
    occ_ydim = int((point_cloud_range[4] - point_cloud_range[1]) / occupancy_size[1])
    occ_zdim = int((point_cloud_range[5] - point_cloud_range[2]) / occupancy_size[2])
    voxel_num = occ_xdim*occ_ydim*occ_zdim


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
    def __init__(self, data_root, sample_tokens, infos, token_to_id, occ_ext, is_gen_occ=False):
        super().__init__()
        self.data_root = data_root
        self.sample_tokens = sample_tokens
        self.infos = infos
        self.token_to_id = token_to_id
        self.occ_ext = occ_ext
        self.is_gen_occ = is_gen_occ

    def __len__(self):
        return len(self.sample_tokens)
    
    @staticmethod
    def collect_fn(batch):
        assert len(batch) == 1
        return batch[0]

    def __getitem__(self, idx):
        sample_token = self.sample_tokens[idx]
        if self.is_gen_occ:
            assert self.occ_ext == '.npy'
            item_data = self.infos[sample_token[0]]
        else:
            item_data = self.infos[self.token_to_id[sample_token]]

        data_root = self.data_root

        if self.occ_ext == '.npy':
            if self.is_gen_occ:
                occ_path = os.path.join(data_root, f'{item_data["token"]}_pred_occs.npy')
            else:
                occ_path = os.path.join(data_root, sample_token+self.occ_ext)
        else:
            occ_path = os.path.join(data_root, sample_token, sample_token+self.occ_ext)


        # npz or npy
        try:
            if self.occ_ext == '.npy':
                if self.is_gen_occ:
                    _occ = np.load(occ_path)[0]
                else:
                    _occ = np.load(occ_path)
            else:
                _occ = np.load(occ_path)['occ']
        except:
            with open('fail_files.txt', 'w') as f:
                f.write(f'{occ_path}\n')
            return dict(sample_token=item_data['token'])
        if self.occ_ext == '.npy':
            occ = np.stack(_occ.nonzero()).T
            semantics = _occ[occ[:, 0], occ[:, 1], occ[:, 2]] - 1
        else:
            occ = _occ
            semantics = occ[:, -1]
        point_x = (occ[:, 0] + 0.5) / occ_xdim * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
        point_y = (occ[:, 1] + 0.5) / occ_ydim * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
        point_z = (occ[:, 2] + 0.5) / occ_zdim * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
        xyz = np.stack([point_x, point_y, point_z], axis=-1)

        semantics = torch.from_numpy(semantics) + 1
        xyz = torch.from_numpy(xyz).float()#.cuda()

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

def save_render_result(camera_semantic, camera_depth, base_path, sample_token):
    if not os.path.exists(os.path.join(base_path, sample_token)):
        os.makedirs(os.path.join(base_path, sample_token))
    sem_data_all_path = os.path.join(base_path, sample_token, "semantic.npz")
    depth_data_all_path = os.path.join(base_path, sample_token, "depth_data.npz")
    np.savez_compressed(sem_data_all_path, camera_semantic)
    np.savez_compressed(depth_data_all_path, camera_depth)

def render_occ_semantic_map(batch, data_root, base_path, save_executor, is_vis=False, gs_scale=0.125, with_ut=False):
    if 'xyz' not in batch:
        return None

    semantics_gt = batch['semantics_gt']
    xyz = batch['xyz'].cuda()
    cam_infos = batch['cam_infos']
    occ_mask = batch['occ_mask']
    sample_token = batch['sample_token']

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
        cam_info = cam_infos[cam]
        camera_intrinsic = np.eye(3).astype(np.float32)
        camera_intrinsic[:3, :3] = cam_info['cam_intrinsic']
        camera_intrinsic = torch.from_numpy(camera_intrinsic).cuda().float()

        c2e = np.eye(4).astype(np.float32)
        c2e[:3, :3] = cam_info['sensor2lidar_rotation']
        c2e[:3, 3] = np.array(cam_info['sensor2lidar_translation'])
        c2e = torch.from_numpy(c2e).cuda().float()

        camera_extrinsic = c2e

        with torch.no_grad():
            if not with_ut:
                render_pkg = render(
                    camera_extrinsic, camera_intrinsic, image_shape,
                    xyz[occ_mask], rgb[occ_mask], feat[occ_mask], rot[occ_mask], scale[occ_mask], opacity[occ_mask],
                    bg_color=[0, 0, 0]
                )
                render_semantic = render_pkg['render_feat']
                render_depth = render_pkg['render_depth']
            else:
                k1,k2,p1,p2,k3 = cam_info['distortion']

                render_colors, render_alphas, metainfo = rasterization(
                    means=xyz[occ_mask], 
                    quats=rot[occ_mask], 
                    scales=scale[occ_mask], 
                    colors=feat[occ_mask],
                    opacities=opacity[occ_mask].squeeze(), 
                    viewmats=torch.linalg.inv(camera_extrinsic)[None],
                    Ks=camera_intrinsic[None],
                    width=image_shape[1],
                    height=image_shape[0],
                    render_mode='RGB+D',
                    camera_model='pinhole',
                    radial_coeffs=torch.tensor([[k1, k2, k3, 0.0, 0.0, 0.0]], device=xyz.device, dtype=torch.float32),
                    tangential_coeffs=torch.tensor([[p1, p2]], device=xyz.device, dtype=torch.float32),
                    thin_prism_coeffs=None,
                    with_ut=True,
                    packed=False
                )

                render_semantic = render_colors[..., :feat.shape[-1]][0].permute(2, 0, 1)
                render_depth = render_colors[..., -1][0][None]

                none_mask = render_alphas.squeeze() < 0.10
                none_label = torch.zeros(20).cuda()
                none_label[0] = 1
                render_semantic[:, none_mask] = none_label[:, None]
                render_depth[:, none_mask] = 51.2
        
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

            projection_matrix = get_projection_matrix_c(fx, fy, cx, cy, width, height, 0.1, 200.0).transpose(0, 1).cuda()
            world_view_transform = extrinsics.transpose(0, 1).cuda()
            full_projection = world_view_transform.float() @ projection_matrix
            cam_3D = means3D_h @ full_projection.to(device)
            cam_3D = cam_3D / cam_3D[:, 3:4]
            x = ((cam_3D[:, 0] + 1.0) * image_shape[1] - 1.0) * 0.5
            y = ((cam_3D[:, 1] + 1.0) * image_shape[0] - 1.0) * 0.5
            visiable_mask = _C.mark_visible(means3D_, world_view_transform.to(device), full_projection.to(device))
            visiable_mask = visiable_mask & (x < image_shape[1]) & (y < image_shape[0]) & (x > 0) & (y > 0)

            vis_x = x[visiable_mask].cpu().numpy()
            vis_y = y[visiable_mask].cpu().numpy()
            vis_sem = (semantics_gt.squeeze())[visiable_mask.cpu()].numpy()
            ori_img = cv2.imread(os.path.join(data_root, cam_info['data_path']))
            for p in range(visiable_mask.sum()):
                _rgb = nuscenes_cmap[int(vis_sem[p])].tolist()
                cv2.circle(ori_img, (int(vis_x[p]), int(vis_y[p])), 2, _rgb, -1)
            cv2.imwrite(proj_save_path, ori_img)

        semantic = torch.max(render_semantic, dim=0)[1].squeeze().cpu().numpy().astype(np.int8)
        camera_semantic.append(semantic)

        depth_data = render_depth[0].detach().cpu().numpy()
        camera_depth.append(depth_data)

    # fast save
    for i, cam in enumerate(cams):
        save_path = os.path.join(base_path, cam)
        save_executor.submit(save_render_result, camera_semantic[i], camera_depth[i], save_path, sample_token)

    return sample_token

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, default=DEFAULT_CONFIG_FILE)
    parser.add_argument('--dataset_path', type=str, default="data/nuplan_all/sensor_blobs/mini")
    parser.add_argument('--pkl_path', type=str, default='data/nuplan_pkls/mini/nuplan_mini_10hz_val.pkl')
    parser.add_argument('--version', type=str, default='mini')
    parser.add_argument('--occ_path', type=str, default="data/occ_quan/nuplan_quantized_400_400_32")
    # parser.add_argument('--occ_path', type=str, default="data/occ_gen_200/save_occ")
    parser.add_argument('--render_path', type=str, default="data/nuplan-occ-render-mini_val2/")
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_interval', type=int, default=200)
    parser.add_argument('--gs_scale', type=float, default=0.01)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=500)
    parser.add_argument('--with_ut', action='store_true', default=True)
    parser.add_argument('--occ_ext', type=str, default='.npy')
    parser.add_argument('--skip_exists', action='store_true', default=False)
    parser.add_argument('--is_gen_occ', action='store_true', default=False)
    parser.add_argument('--pkl_with_clip_info', type=str, default='data/nuplan_pkls_clipinfo/nuplan_mini_val_clip_infos.pkl')
    
    args = parser.parse_args()
    load_config(args.config_file)

    if dist_utils.is_dist():
        dist_utils.ddp_setup()
        rank = dist_utils.get_local_rank()
        world_size = dist_utils.get_local_world_size()
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

    if args.is_gen_occ:
        assert args.pkl_with_clip_info is not None, "Please provide pkl_with_clip_info when is_gen_occ is True"
        with open(args.pkl_with_clip_info, 'rb') as f:
            infos_withclip = pickle.load(f)
        infos['clip_infos'] = infos_withclip['clip_infos']
        sample_tokens_list = [infos['clip_infos']]
        

    render_base_path = os.path.join(args.render_path, args.version)
    occ_base_path = os.path.join(args.occ_path, args.version)
    if args.is_gen_occ:
        index_list = [0]
    else:
        all_train_items = len(sample_tokens_list)
        print('Total seq:', all_train_items)
        args.end_idx = min(all_train_items-1, args.end_idx)
        index_list = list(range(args.start_idx, args.end_idx+1))
        if len(index_list) == 0:
            index_list = list(range(all_train_items))

        if dist_utils.is_dist():
            index_list = np.array_split(index_list, world_size)[rank].tolist()
            print(f'Rank: {rank}, proscess {index_list}')

    save_executor = ThreadPoolExecutor(max_workers=8)
    n_processed = 0
    if os.path.exists(os.path.join(args.render_path, args.version, 'CAM_F0')):
        with FileLock('occrender_scan.lock'):
            exists_tokens = set(entry.name for entry in os.scandir(os.path.join(args.render_path, args.version, 'CAM_F0')))
    else:
        exists_tokens = set()
    for index in index_list:
        sample_tokens = sample_tokens_list[index]
        # check exist
        if args.skip_exists:
            unprocessed_tokens = []
            for sample_token in sample_tokens:
                if sample_token not in exists_tokens:
                    unprocessed_tokens.append(sample_token)
            sample_tokens = unprocessed_tokens
        
        if len(sample_tokens) == 0:
            print(f'seq {index} processed')
            continue

        dataloader = DataLoader(OccDataset(args.occ_path, sample_tokens, infos['infos'], token_to_id, args.occ_ext, args.is_gen_occ), batch_size=1, collate_fn=OccDataset.collect_fn, shuffle=False, num_workers=0, pin_memory=True)

        for batch in tqdm(dataloader, desc=f'Rank: {rank}, Index: {index}'):
            
            sample_token = render_occ_semantic_map(
                batch,
                data_root=args.dataset_path,
                base_path=render_base_path,
                save_executor=save_executor,
                is_vis=args.vis or (n_processed % args.vis_interval) == 0,
                gs_scale=args.gs_scale,
                with_ut=args.with_ut
            )
            if sample_token is None:
                print(f'Failed: {batch["sample_token"]}')
            else:
                n_processed += 1
        gc.collect()
    
    print(f'Rank: {rank} has processed {n_processed} frames')
    if dist_utils.is_dist():
        dist_utils.ddp_cleanup()
