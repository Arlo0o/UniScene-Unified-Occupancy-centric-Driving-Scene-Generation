import os

import gc
import torch
import json
import numpy as np
from tqdm import tqdm
import pickle
import nksr
import shutil
os.makedirs('/root/.cache/torch/hub/checkpoints', exist_ok=True)
shutil.copyfile('ks.pth', '/root/.cache/torch/hub/checkpoints/ks.pth')
import sys
import pdb
import time
import yaml
from concurrent.futures import ThreadPoolExecutor
from filelock import FileLock
import torch
from torch.utils.data import DataLoader
#import chamfer
# from chamfer_distance import ChamferDistance
# from pytorch3d.ops import knn_points
import numpy as np
#from nuscenes.nuscenes import NuScenes
#from nuscenes.utils import splits
from tqdm import tqdm
from argparse import Namespace
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
from voxelization import voxelize_point_cloud
from scipy.spatial.transform import Rotation
from megfile import smart_makedirs, smart_path_join, smart_listdir, smart_exists, smart_open

import pickle
import open3d as o3d
from pycg import vis
from copy import deepcopy
from argparse import ArgumentParser, Namespace
from nuplan import NuPlan, SingleScene, NKSRScene, MeshScene3, lidar_to_world_to_lidar_gpu
import dist_utils
from kiss_icp.kiss_icp import KissICP
from kiss_icp.config import load_config

def rotate_points_gpu(points, rot_angle):
    
    # 计算旋转角的正弦和余弦
    cos_theta = torch.cos(rot_angle)
    sin_theta = torch.sin(rot_angle)
    
    # 构建绕Z轴的旋转矩阵
    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0,          0,         1]
    ], dtype=torch.float32, device='cuda')
    
    # 应用旋转矩阵（行向量右乘R^T）
    rotated_points = torch.matmul(points, rotation_matrix.T)
    
    return rotated_points

def nksr_mesh_normal(input_xyz, input_normal, detail_level=0.5, mise_iter=1, cpu_=False, reconstructor=None, chunk_size=-1.0):
    if reconstructor is None:
        device = torch.device("cuda")
        reconstructor = nksr.Reconstructor(device)
        # reconstructor.chunk_tmp_device = torch.device("cuda:0")

    field = reconstructor.reconstruct(
        input_xyz,
        input_normal,
        chunk_size=chunk_size,
        # chunk_size=20.0,  # This could be smaller
        # chunk_size=50.0,
        detail_level=detail_level
    )

    if cpu_:
        # Put everything onto CPU.
        field.to_("cpu")
        reconstructor.network.to("cpu")

    mesh = field.extract_dual_mesh(mise_iter=mise_iter)
    return mesh


def nksr_mesh_sensor(input_xyz, input_sensor, detail_level=None, mise_iter=1, cpu_=False, reconstructor=None, chunk_size=-1.0):
    if reconstructor is None:
        device = torch.device("cuda")
        reconstructor = nksr.Reconstructor(device)

    field = reconstructor.reconstruct(
        input_xyz, sensor=input_sensor, detail_level=detail_level,
        # Minor configs for better efficiency (not necessary)
        approx_kernel_grad=True, solver_tol=1e-4, fused_mode=True, 
        # Chunked reconstruction (if OOM)
        #chunk_size=51.2,
        chunk_size=chunk_size,
        preprocess_fn=nksr.get_estimate_normal_preprocess_fn(64, 85.0)
    )
    if field is None:
        return None
    # Put everything onto CPU.
    if cpu_:
        field.to_("cpu")
        reconstructor.network.to("cpu")
    # [WARNING] Slow operation...
    try:
        mesh = field.extract_dual_mesh(mise_iter=mise_iter)
    except:
        return None
    return mesh


def run_poisson(pcd, depth, n_threads, min_density=None):
    # 转为 NumPy，再转为 Open3D 点云对象
    points_np = pcd.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # 估计法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    mesh_out = Namespace(v=torch.from_numpy(np.asarray(mesh.vertices)), f=torch.from_numpy(np.asarray(mesh.triangles)), c=None)

    return mesh_out, densities


def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original=None):
    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer)
    else:
        pcd = point_cloud_original

    return run_poisson(pcd, depth, n_threads, min_density)


def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()
    for cloud in buffer:
        pcd += cloud
    if compute_normals:
        pcd.estimate_normals()

    return pcd

def mesh_to_cpu(mesh, to_numpy=True):
    mesh.v = mesh.v.cpu()
    mesh.f = mesh.f.cpu()
    if mesh.c is not None:
        mesh.c = mesh.c.cpu()
    
    if to_numpy:
        mesh.v = mesh.v.numpy()
        mesh.f = mesh.f.numpy()
        if mesh.c is not None:
            mesh.c = mesh.c.numpy()
    return mesh

def preprocess_cloud(
        pcd,
        max_nn=20,
        normals=None,
):
    cloud = deepcopy(pcd)
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)
        cloud.orient_normals_towards_camera_location()

    return cloud


def preprocess(pcd, config):
    return preprocess_cloud(
        pcd,
        config['max_nn'],
        normals=True
    )


def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

        Args:
            nx3 np.array's
        Returns:
            ([indices], [distances])

    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances


def lidar_to_world_to_lidar(pc, lidar_calibrated_sensor, lidar_ego_pose,
                            cam_calibrated_sensor,
                            cam_ego_pose):
    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation'], dtype='float64'))

    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation'], dtype='float64'))

    pc.translate(-np.array(cam_ego_pose['translation'], dtype='float64'))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    pc.translate(-np.array(cam_calibrated_sensor['translation'], dtype='float64'))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    return pc

def save_occ(save_path, sample_token, lidar_token, occ):
    dirs = smart_path_join(save_path, 'dense_voxels_with_semantic/')

    # npz
    lidar_data_path = smart_path_join(dirs, sample_token, lidar_token + '.npz')
    os.makedirs(os.path.dirname(lidar_data_path), exist_ok=True)
    np.savez_compressed(lidar_data_path, occ=occ)

    # pkl
    # lidar_data_path = smart_path_join(dirs, sample_token, lidar_token + '.pkl')
    # with smart_open(lidar_data_path, "wb") as f:
    #     pickle.dump(occ, f)

def main(nusc, indice, nuscenesyaml, args, config):
    save_path = args.save_path
    data_root = args.dataroot
    learning_map = nuscenesyaml['learning_map']
    voxel_size = config['voxel_size']
    pc_range = config['pc_range']
    occ_size = config['occ_size']

    my_scene = nusc.scene[indice]
    sensor = 'LIDAR_TOP'


    # load the first sample to start
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)

    lidar_data = my_sample # nusc.get('sample_data', my_sample)
    lidar_data['filename'] = lidar_data['lidar_path']
    lidar_data['is_key_frame'] = True
    lidar_ego_pose0 = nusc.get('ego_pose', lidar_data)
    lidar_calibrated_sensor0 = nusc.get('calibrated_sensor', lidar_data)

    # collect LiDAR sequence
    dict_list = []
    cnt = 0
    dataloader = DataLoader(SingleScene(nusc, my_scene, nuscenesyaml, box_expand=1.0), collate_fn=SingleScene.collect_fn, batch_size=1, num_workers=8, shuffle=False)
    kicp = KissICP(config=load_config(None, max_range=None))
    icp_poses = np.zeros((len(dataloader), 4, 4))
    for frame_idx, batch in enumerate(dataloader):
        if batch is None:
            return False
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(f"Aggregating... {cnt}")
        sys.stdout.flush()

        #t1 = time.perf_counter()

        converted_object_category = batch['converted_object_category']
        flag_has_lidarseg = batch['flag_has_lidarseg']
        object_tokens = batch['object_tokens']
        pc_file_name = batch['pc_file_name']
        lidar_data = batch['lidar_data']
        if 'points_in_boxes' in batch:
            pc0 = batch['pc0']
            gt_bbox_3d = batch['gt_bbox_3d']
            points_in_boxes = batch['points_in_boxes']
        else:
            pc0 = torch.from_numpy(batch['pc0']).cuda()
            gt_bbox_3d = torch.from_numpy(batch['gt_bbox_3d']).to(torch.float32).cuda()

            #with SimpleProfile(name='agg-section3'):
            points_in_boxes = points_in_boxes_all((pc0[:, :3][None, :, :]),
                                                (gt_bbox_3d[None, :]))
            
            # expand 
            # x,y,z,l,w,h,yaw
            gt_bbox_3d_expanded = gt_bbox_3d.clone()
            gt_bbox_3d_expanded[:, 3:6] *= 1.1
            points_in_boxes_expand = points_in_boxes_all((pc0[:, :3][None, :, :]),
                                                (gt_bbox_3d_expanded[None, :]))

        
        # visualize points and gt_boxes
        # _boxes_vis = gt_bbox_3d.cpu().numpy().astype('float32').copy()
        # _boxes_vis[:, 2] += (_boxes_vis[:, 5] / 2)
        # _points_boxes = dict(points=pc0[:, :3].cpu().numpy().astype('float32').tolist(), boxes=_boxes_vis.tolist())
        # with open(f'vis/{frame_idx}.jsonpb', 'w') as f:
        #     json.dump(_points_boxes, f)
        
        ############################# cut out movable object points and masks ##########################
        object_points_list = []
        j = 0
        while j < points_in_boxes.shape[-1]:
            object_points_mask = points_in_boxes[0][:, j].bool()
            object_points = pc0[object_points_mask].cpu().numpy()
            object_points_list.append(object_points)
            j = j + 1

        moving_mask = torch.ones_like(points_in_boxes_expand)
        points_in_boxes_expand = torch.sum(points_in_boxes_expand * moving_mask, dim=-1).bool()
        points_mask = ~(points_in_boxes_expand[0])

        ############################# get point mask of the vehicle itself ##########################
        self_range = config['self_range']
        if isinstance(pc0, torch.Tensor):
            oneself_mask = (torch.abs(pc0[:, 0]) > self_range[0]) | \
                            (torch.abs(pc0[:, 1]) > self_range[1]) | \
                            (torch.abs(pc0[:, 2]) > self_range[2])
        else:
            oneself_mask = torch.from_numpy((np.abs(pc0[:, 0]) > self_range[0]) | \
                            (np.abs(pc0[:, 1]) > self_range[1]) | \
                            (np.abs(pc0[:, 2]) > self_range[2]))

        ############################# get static scene segment ##########################
        points_mask = points_mask & oneself_mask
        pc = pc0[points_mask]

        #with SimpleProfile(name='agg-section4'):
        ################## coordinate conversion to the same (first) LiDAR coordinate  ##################
        #lidar_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        #lidar_calibrated_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_ego_pose = nusc.get('ego_pose', lidar_data)
        lidar_calibrated_sensor = nusc.get('calibrated_sensor', lidar_data)
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().numpy()
            gt_bbox_3d = gt_bbox_3d.cpu().numpy()

        #print(f's1:{time.perf_counter()-t1}')
        #t1 = time.perf_counter()

        # 在这里去噪？还是聚合后再去噪？用什么去噪算法？
        use_pc_filter = config.get('use_pc_filter', True)
        if use_pc_filter:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
            pcd, idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
            pc = pc[idx]

        # # KISS-ICP
        use_icp = config.get('use_icp', True)
        if use_icp:
            source, keypoints = kicp.register_frame(pc[:,:3].copy().astype('float64'), np.array([]))
            icp_poses[frame_idx] = kicp.last_pose

            # print(f's2:{time.perf_counter()-t1}')
            # t1 = time.perf_counter()

            # 采用ICP估计出的位姿，世界坐标系远点变为0帧
            lidar_ego_pose0['translation'] = [0.0, 0.0, 0.0]
            lidar_ego_pose0['rotation'] = [1.0, 0.0, 0.0, 0.0]
            lidar_ego_pose['translation'] = kicp.last_pose[:3, -1]
            lidar_ego_pose['rotation'] = Quaternion._from_matrix(kicp.last_pose[:3, :3])

        lidar_pc = lidar_to_world_to_lidar(pc.copy().astype('float64'), lidar_calibrated_sensor.copy(), lidar_ego_pose.copy(),
                                        lidar_calibrated_sensor0,
                                        lidar_ego_pose0)

        #print(f's3:{time.perf_counter()-t1}')
        ################## record Non-key frame information into a dict  ########################
        frame_dict = {#"object_tokens": object_tokens,
                #"object_points_list": object_points_list,
                "lidar_pc": lidar_pc.points,
                "lidar_ego_pose": lidar_ego_pose,
                "lidar_calibrated_sensor": lidar_calibrated_sensor,
                "lidar_token": lidar_data['token'],
                "sample_token": lidar_data['token'],#lidar_data['sample_token'],
                "is_key_frame": lidar_data['is_key_frame'],
                "has_lidarseg": flag_has_lidarseg,
                "gt_bbox_3d": gt_bbox_3d,
                "converted_object_category": converted_object_category,
                #"pc_file_name": pc_file_name.split('/')[-1]
                "objects": dict(zip(object_tokens, zip(object_points_list, gt_bbox_3d.tolist(), converted_object_category))),
                "icp_pose": kicp.last_pose
                }
        ################# record semantic information into the dict if it's a key frame  ########################
        if lidar_data['is_key_frame'] and flag_has_lidarseg:
            pc_with_semantic = pc_with_semantic[points_mask]
            lidar_pc_with_semantic = lidar_to_world_to_lidar(pc_with_semantic.copy(),
                                                            lidar_calibrated_sensor.copy(),
                                                            lidar_ego_pose.copy(),
                                                            lidar_calibrated_sensor0,
                                                            lidar_ego_pose0)
            frame_dict["lidar_pc_with_semantic"] = lidar_pc_with_semantic.points

        dict_list.append(frame_dict)
        ################## go to next frame of the sequence  ########################
        #curr_sample_token = lidar_data['sample_token']
        #next_sample_token = nusc.get('sample', curr_sample_token)['next']

        # next_sample_token = lidar_data['sample_next']
        # if next_sample_token != '' and next_sample_token is not None:
        #     #next_lidar_token = nusc.get('sample', next_sample_token)['data'][sensor]
        #     #lidar_data = nusc.get('sample_data', next_lidar_token)
        #     lidar_data = nusc.get('sample', next_sample_token)
        #     lidar_data['filename'] = lidar_data['lidar_path']
        #     lidar_data['is_key_frame'] = True
        # else:
        #     break
        cnt += 1

        # next_token = lidar_data['next']
        # if next_token != '':
        #     lidar_data = nusc.get('sample_data', next_token)
        # else:
        #     break
    print('\nAggregation finished')

    ################## concatenate all static scene segments (including non-key frames)  ########################
    lidar_pc_list = [_dict['lidar_pc'] for _dict in dict_list]
    sensor_locs = np.concatenate([_dict['lidar_ego_pose']['translation'][None].repeat(_dict['lidar_pc'].shape[1], axis=0) for _dict in dict_list])
    lidar_pc = np.concatenate(lidar_pc_list, axis=1).T

    # free some memory
    del dataloader
    del kicp
    for _dict in dict_list:
        _dict.pop('lidar_pc')
    gc.collect()

    ################## concatenate all semantic scene segments (only key frames)  ########################
    lidar_pc_with_semantic_list = []
    for _dict in dict_list:
        if _dict['is_key_frame'] and _dict['has_lidarseg']:
            lidar_pc_with_semantic_list.append(_dict['lidar_pc_with_semantic'])
    if len(lidar_pc_with_semantic_list) != 0:
        lidar_pc_with_semantic = np.concatenate(lidar_pc_with_semantic_list, axis=1).T
    else:
        lidar_pc_with_semantic = np.zeros((0, lidar_pc.shape[1]))

    ################## Reimplemented  ########################
    ################## concatenate all object segments (including non-key frames)  ########################
    object_zoo = {}
    for _dict in dict_list:
        for i, (object_token, object_info) in enumerate(_dict['objects'].items()):
            object_points, object_box, object_semantic = object_info
            if (object_points.shape[0] > 0):
                if object_token not in object_zoo:
                    object_zoo[object_token] = {'points': [], 'label': [], 'traj': [], 'sensor_locs': []}
            
                object_zoo[object_token]['points'].append(object_points)
                object_zoo[object_token]['label'].append(object_semantic)
                object_zoo[object_token]['traj'].append(object_box)
                object_zoo[object_token]['sensor_locs'].append(np.zeros_like(object_points))
            else:
                continue
    for object_token, object_info in object_zoo.items():
        object_points_all = object_info['points']
        object_traj = object_info['traj']
        object_sensor_locs_all = object_info['sensor_locs']
        for j in range(len(object_points_all)):
            object_points_all[j] = object_points_all[j][:, :3] - object_traj[j][:3]
            rots = object_traj[j][6]
            Rot = Rotation.from_euler('z', -rots, degrees=False)
            object_points_all[j] = Rot.apply(object_points_all[j]).astype('float32')

            object_sensor_locs_all[j] = object_sensor_locs_all[j][:, :3] - object_traj[j][:3]
            Rot = Rotation.from_euler('z', -rots, degrees=False)
            object_sensor_locs_all[j] = Rot.apply(object_sensor_locs_all[j]).astype('float32')
            

        # ICP for object
        # kicp = KissICP(config=load_config("gs_render/data_process/kiss-icp/config/nuplan_object.yaml", max_range=None))
        # icp_poses = np.zeros((len(object_points_all), 4, 4))
        # for frame_idx, p in enumerate(object_points_all):
        #     _, _ = kicp.register_frame(p[:,:3].copy().astype('float64'), np.array([]))
        #     icp_poses[frame_idx] = kicp.last_pose

        # object_points_all_refined = []
        # for pts, pose in zip(object_points_all, icp_poses):
        #     pts_homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=-1)
        #     pts_world = pts_homo @ pose.T
        #     object_points_all_refined.append(pts_world[:, :3])
        

        object_zoo[object_token]['points'] = np.concatenate(object_points_all)
        assert (np.array(object_info['label'])==object_info['label'][0]).all()
        object_zoo[object_token]['label'] = object_info['label'][0]
        object_zoo[object_token]['sensor_locs'] = np.concatenate(object_sensor_locs_all)
    print('object finish')
    torch.cuda.empty_cache()
    gc.collect()


    # 先nksr重建大场景（前背景分开），避免重复重建
    try:
        reconstructor = nksr.Reconstructor('cuda')
    except:
        return False
    ################## sensor location for nksr ################
    sensor_locs_i = lidar_to_world_to_lidar(np.concatenate([sensor_locs.copy(), np.ones((sensor_locs.shape[0], 1), dtype=sensor_locs.dtype)], axis=-1),
                                            {'rotation': [1.0, 0.0, 0.0, 0.0], 'translation': np.zeros((3,), dtype=np.float64)},
                                            {'rotation': [1.0, 0.0, 0.0, 0.0], 'translation': np.zeros((3,), dtype=np.float64)},
                                            lidar_calibrated_sensor0,
                                            lidar_ego_pose0)
    sensor_locs = sensor_locs_i.points.T[:, :3]
    sensor_locs[:, -1] += 1.8   # FIXME: lidar height
    #mask = (lidar_pc[:, 2] > pc_range[2]) & (lidar_pc[:, 2] < pc_range[-1])# & (np.abs(lidar_pc[:, 0]) < 50) & (np.abs(lidar_pc[:, 1]) < 50)
    #lidar_pc = lidar_pc[mask]
    #sensor_locs = sensor_locs[mask]
    if args.use_poisson:
        nksr_mesh, _ = run_poisson(torch.from_numpy(lidar_pc[:,:3]).float(), depth=7, n_threads=32)
    else:
        try:
            nksr_mesh = nksr_mesh_sensor(torch.from_numpy(lidar_pc[:,:3]).float().cuda(), torch.from_numpy(sensor_locs).float().cuda(), detail_level=None, mise_iter=1, cpu_=False, reconstructor=reconstructor)#, chunk_size=51.2)
        except: # OOM, use chunk
            try:
                nksr_mesh = nksr_mesh_sensor(torch.from_numpy(lidar_pc[:,:3]).float().cuda(), torch.from_numpy(sensor_locs).float().cuda(), detail_level=None, mise_iter=1, cpu_=False, reconstructor=reconstructor, chunk_size=51.2)
            except:
                return False #:(

    if nksr_mesh is None:
        return False

    # visualization
    if 0:
        vis.to_file(vis.mesh(nksr_mesh.v, nksr_mesh.f), 'z.ply')

    nksr_mesh = mesh_to_cpu(nksr_mesh)

    # mesh后处理 TODO:前景物体需要吗？
    if args.fill_holes:
        mesh_o3d = o3d.t.geometry.TriangleMesh()
        mesh_o3d.vertex.positions = o3d.core.Tensor(nksr_mesh.v)
        mesh_o3d.triangle.indices = o3d.core.Tensor(nksr_mesh.f)
        mesh_o3d = mesh_o3d.fill_holes()
        mesh_o3d = mesh_o3d.to_legacy()
    else:
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(nksr_mesh.v)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(nksr_mesh.f)
    mesh_o3d = mesh_o3d.subdivide_midpoint()
    nksr_mesh.v = np.asarray(mesh_o3d.vertices)
    nksr_mesh.f = np.asarray(mesh_o3d.triangles)

    # 重建前景物体
    cnt = 0
    for object_token, object_info in tqdm(object_zoo.items(), desc=f'Seq {index} Reconstruct object...'):
        points = object_info['points']
        sensor_locs_obj = object_info['sensor_locs']
        sensor_locs_obj[:, -1] += 1.8   # FIXME: lidar height
        obj_mesh = nksr_mesh_sensor(torch.from_numpy(points[:,:3]).float().cuda(), torch.from_numpy(sensor_locs_obj).float().cuda(), detail_level=None, mise_iter=1, cpu_=False, reconstructor=reconstructor)
        object_zoo[object_token]['mesh'] = mesh_to_cpu(obj_mesh) if obj_mesh is not None else None
        # TODO: use original points instead?
        if obj_mesh is None:
            object_zoo[object_token]['mesh'] = Namespace(v=points[:,:3])
        del object_info['points']

        cnt += 1
        if cnt % 50 == 0:
           torch.cuda.empty_cache()
           gc.collect() 
    del cnt
    del lidar_pc
    lidar_pc = None
    print('reconstruction finish')
    torch.cuda.empty_cache()
    gc.collect()
    
    i = 0
    ds = MeshScene3(nusc, my_scene, nuscenesyaml, dict_list, lidar_pc, lidar_pc_with_semantic, nksr_mesh, object_zoo, sensor_locs, config)
    dataloader = DataLoader(ds, collate_fn=NKSRScene.collect_fn, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, persistent_workers=True)
    
    save_executor = ThreadPoolExecutor(max_workers=6)
    #chd = ChamferDistance()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Seq {indice} Gen occ...')):
        #t1 = time.perf_counter()
        mesh_points = torch.from_numpy(batch['mesh_points']).float().cuda()
        #_dict = batch['_dict']
        #point_cloud_with_semantic = torch.from_numpy(batch['semantic_points']).float().cuda()

        lidar_ego_pose0 = ds.frame_infos[0]['lidar_ego_pose']
        lidar_calibrated_sensor0 = ds.frame_infos[0]['lidar_calibrated_sensor']
        #lidar_calibrated_sensor = _dict['lidar_calibrated_sensor']
        #lidar_ego_pose = _dict['lidar_ego_pose']
        lidar_calibrated_sensor = batch['lidar_calibrated_sensor']
        lidar_ego_pose = batch['lidar_ego_pose']
        

        mesh_points = lidar_to_world_to_lidar_gpu(mesh_points,
                                             lidar_calibrated_sensor0.copy(),
                                             lidar_ego_pose0.copy(),
                                             lidar_calibrated_sensor,
                                             lidar_ego_pose)
        
        ################# bbox placement ##############
        cur_objects = batch['cur_objects']
        object_semantic_list = []
        object_mesh_list = []
        for j, (object_token, object_info) in enumerate(cur_objects.items()):
            if object_token in object_zoo:
                # 0x 1y 2z 3l 4w 5h 6yaw
                object_box = torch.tensor(object_info[1]).cuda()
                rot = object_box[6]
                loc = object_box[0:3]

                #points = torch.from_numpy(object_zoo[object_token]['points']).cuda()
                #Rot = Rotation.from_euler('z', rot, degrees=False)
                #rotated_object_points = Rot.apply(points)
                #rotated_object_points = rotate_points_gpu(points, rot)
                #points = rotated_object_points + loc

                # mesh
                obj_mesh = object_zoo[object_token]['mesh']
                if obj_mesh is not None:
                    obj_mesh_points = torch.from_numpy(obj_mesh.v).cuda()
                    #Rot = Rotation.from_euler('z', rot, degrees=False)
                    #obj_mesh_cur.v = Rot.apply(obj_mesh_cur.v)
                    obj_mesh_points = rotate_points_gpu(obj_mesh_points, rot)
                    obj_mesh_points = obj_mesh_points + loc
                else:
                    obj_mesh_points = None

                if obj_mesh_points.shape[0] >= 5:
                    points_in_boxes = points_in_boxes_all(obj_mesh_points[:, :3][None, :, :],
                                                        object_box[None][None, :])
                    points_in_boxes_mask = points_in_boxes[0, :, 0].bool()
                    obj_mesh_points = obj_mesh_points[points_in_boxes_mask]

                remove_exist_object_points = True
                if remove_exist_object_points:
                    # TODO:如果点数太多会报cuda错误
                    # points_in_boxes = points_in_boxes_all(point_cloud_with_semantic[:, :3][None, :, :],
                    #                                     object_box[None][None, :])
                    # points_in_boxes_mask = points_in_boxes[0, :, 0].bool()
                    # point_cloud_with_semantic = point_cloud_with_semantic[~points_in_boxes_mask]

                    chunk_size = 2097152  # 每次处理200万个点
                    N = mesh_points.shape[0]
                    mask_parts = []
                    # 分块处理点云数据
                    for i in range(0, N, chunk_size):
                        end = min(i + chunk_size, N)
                        # 提取当前分块的点坐标并增加必要的维度
                        chunk_points = mesh_points[i:end, :3].unsqueeze(0)
                        # 保持与原始代码相同的boxes维度
                        boxes = object_box[None, None, :]
                        
                        # 执行计算
                        points_in_chunk = points_in_boxes_all(chunk_points, boxes)
                        chunk_mask = points_in_chunk[0, :, 0].bool()  # 当前分块的mask
                        
                        mask_parts.append(chunk_mask)

                    # 合并所有分块的mask结果
                    points_in_boxes_mask = torch.cat(mask_parts, dim=0)

                    # 应用mask过滤原始点云
                    mesh_points = mesh_points[~points_in_boxes_mask]

                #semantics = torch.ones_like(points[:, 0:1]) * object_zoo[object_token]['label']
                #object_semantic_list.append(torch.cat([points[:, :3], semantics], dim=1))
                semantics = torch.zeros_like(obj_mesh_points[:, 0:1]) + (object_zoo[object_token]['label']+1)
                object_semantic_list.append(torch.cat([obj_mesh_points[:, :3], semantics], dim=1))
                if obj_mesh_points is not None:
                    object_mesh_list.append(obj_mesh_points)

        #print(f'trans obj time:{time.perf_counter()-t1}')
        #t1 = time.perf_counter()

        merged_mesh_points = mesh_points
        background_labels = torch.zeros_like(mesh_points[:, 0:1])
        merged_mesh_points = torch.cat([merged_mesh_points, background_labels], dim=-1)
        if object_semantic_list.__len__() != 0:
            merged_mesh_points = torch.cat([merged_mesh_points, torch.cat(object_semantic_list)])
        

        sample_token, lidar_token = batch['sample_token'], batch['lidar_token']

        scene_points = merged_mesh_points

        ################## remain points with a spatial range ##############
        mask = (torch.abs(scene_points[:, 0]) < abs(pc_range[0])) & (torch.abs(scene_points[:, 1]) < abs(pc_range[1])) \
               & (scene_points[:, 2] > pc_range[2]) & (scene_points[:, 2] < pc_range[-1])
        scene_points = scene_points[mask]

        if scene_points.numel() == 0:
            print('no points!!!!!!!!!!!!!!!!!!!!!')
            continue

        # directly voxelize scene points
        voxel_coords, voxel_labels = voxelize_point_cloud(scene_points, voxel_size, pc_range)

        dense_voxels_with_semantic = torch.cat([voxel_coords, voxel_labels[:, None]], dim=1)

        save_executor.submit(save_occ, save_path, sample_token, lidar_token, dense_voxels_with_semantic.cpu().numpy())

        #print(f'{time.perf_counter() - t1} s/sample')

    return True


def save_ply(points, name):
    point_cloud_original = o3d.geometry.PointCloud()
    point_cloud_original.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud("{}.ply".format(name), point_cloud_original)


if __name__ == '__main__':

    parse = ArgumentParser()
    parse.add_argument('--dataset', type=str, default='nuplan')
    parse.add_argument('--config_path', type=str, default='gs_render/data_process/config_nuplan_r400.yaml')
    parse.add_argument('--split', type=str, default='train')
    parse.add_argument('--save_path', type=str, default='./data/nuplan/GT_occ_fast3_r400')
    parse.add_argument('--label_mapping', type=str, default='gs_render/data_process/nuplan.yaml')
    
    parse.add_argument('--dataroot', type=str, default='data/nuplan_all/sensor_blobs/mini')
    parse.add_argument('--pkl_path', type=str, default='data/nuplan_pkls/mini/nuplan_mini_10hz_val.pkl')
    
    #parse.add_argument('--index_list', nargs='+', type=int, default=[])
    parse.add_argument('--start_idx', type=int, default=0)
    parse.add_argument('--end_idx', type=int, default=0)
    parse.add_argument('--fill_holes', action='store_true', default=False)
    parse.add_argument('--num_workers', type=int, default=4)
    parse.add_argument('--use_poisson', action='store_true', default=False)

    args = parse.parse_args()

    if dist_utils.is_dist():
        dist_utils.ddp_setup()
        rank, world_size = dist_utils.get_rank(), dist_utils.get_world_size()
    else:
        rank, world_size = 0, 1

    if args.dataset == 'nuplan':
        nusc = NuPlan(args.pkl_path, data_root=args.dataroot)
        # nusc = NuScenes(version='advanced_12Hz_trainval',
        #                 dataroot=args.dataroot,
        #                 verbose=True)
        # train_scenes = splits.train
        # val_scenes = splits.val

        # nusc.show_video_pc(nusc.scene[0]['token'], 'z.mp4')
        # nusc.show_video_pc(nusc.scene[34]['token'], 'z_pc.mp4')
        # nusc.show_video(nusc.scene[34]['token'], 'z_img.mp4')
    else:
        print('Dataset not supported')

    # load config
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # load learning map
    label_mapping = args.label_mapping
    with open(label_mapping, 'r') as stream:
        nuscenesyaml = yaml.safe_load(stream)

    args.index_list = list(range(args.start_idx, args.end_idx))
    if args.index_list.__len__() == 0:
        index_list = list(range(len(nusc.scene)))
    else:
        index_list = args.index_list
    # index_list = [2460] # debug: miniset_train: 1133个序列

    # check exist
    check_exist = True
    if check_exist:
        with FileLock('scan.lock'):
            if not os.path.exists(os.path.join(args.save_path, 'dense_voxels_with_semantic')):
                all_exists_files = set()
            else:
                all_exists_files = set(entry.name for entry in os.scandir(os.path.join(args.save_path, 'dense_voxels_with_semantic')))
            unprocessed_index_list = []
            for index in tqdm(index_list, desc='checking exist...'):
                sample_tokens = nusc.scene[index]['sample_tokens']
                for sample_token in sample_tokens:
                    # slow, if too many files
                    #if not os.path.exists(os.path.join(args.save_path, 'dense_voxels_with_semantic', sample_token, sample_token+'.npz')):
                    #    unprocessed_index_list.append(index)
                    #    break

                    # fast version
                    if sample_token not in all_exists_files:
                        unprocessed_index_list.append(index)
                        break
            print(f'Total seq: {len(index_list)}, unprocessed seq: {len(unprocessed_index_list)}')
            index_list = unprocessed_index_list
        

    rank_response = []
    if dist_utils.is_dist():
        index_list = np.array_split(index_list, world_size)[rank].tolist()
        rank_response = [index_list[0], index_list[-1]]
        print(f'Rank: {rank}, proscess {index_list[0]} - {index_list[-1]}')
    else:
        rank_response = [index_list[0], index_list[-1]]

    #try:
    for index in index_list:
        print(f'Rank{rank} processing sequecne ({rank_response[0]}-{rank_response[1]}):', index)
        #try:
        t1 = time.perf_counter()
        ok = main(nusc, indice=index, nuscenesyaml=nuscenesyaml, args=args, config=config)
        if not ok:
            with open('fail_to_gen_occ.txt', 'a') as f:
                f.write(f'{index} failed\n')
        t2 = time.perf_counter()
        print('Latency:', t2-t1)
        # except Exception as e:
        #     with open("./scene_error_nksr.txt", 'a') as f:
        #         f.write(str(index) + '\n')
        #     print(e)
        #     continue

        # with open("./scene_nksr.txt", 'a') as f:
        #     f.write(str(index) + '\n')

    # except Exception as e:
    #     print(f"Error on cleanup: {e}")
    # finally:
    #     if dist_utils.is_dist():
    #         dist_utils.ddp_cleanup()
    #         os._exit(0)