from collections import defaultdict
import os
import time
from copy import deepcopy
import pickle
import numpy as np
from PIL import Image
import imageio
import cv2
# import trimesh
from pypcd import pypcd
import open3d as o3d
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset
from nksr.fields.base_field import MeshingResult
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)

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

def lidar_to_world_to_lidar(pc, lidar_calibrated_sensor, lidar_ego_pose,
                            cam_calibrated_sensor,
                            cam_ego_pose):
    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    return pc

def apply_trans(pc, rot, trans, inv=False):
    if not inv:
        q = Quaternion(rot)
        R = torch.tensor(q.rotation_matrix, dtype=pc.dtype, device=pc.device)
        t = torch.tensor(trans, dtype=pc.dtype, device=pc.device)
        pc_ret = torch.matmul(pc, R.T)
        pc_ret = pc_ret + t
    else:   
        t = torch.tensor(trans, dtype=pc.dtype, device=pc.device)
        pc_ret = pc - t
        q = Quaternion(rot)
        R = torch.tensor(q.rotation_matrix, dtype=pc.dtype, device=pc.device)
        pc_ret = torch.matmul(pc_ret, R)
    return pc_ret

def lidar_to_world_to_lidar_gpu(pc, lidar_calibrated_sensor, lidar_ego_pose,
                                cam_calibrated_sensor,
                                cam_ego_pose):
    pc = apply_trans(pc, lidar_calibrated_sensor['rotation'], lidar_calibrated_sensor['translation'])
    pc = apply_trans(pc, lidar_ego_pose['rotation'], lidar_ego_pose['translation'])
    pc = apply_trans(pc, cam_ego_pose['rotation'], cam_ego_pose['translation'], inv=True)
    pc = apply_trans(pc, cam_calibrated_sensor['rotation'], cam_calibrated_sensor['translation'], inv=True)
    return pc

def create_3d_box(box_params: np.ndarray, color: tuple) -> o3d.geometry.LineSet:
    # 解析参数
    center = box_params[:3]
    size = box_params[3:6] / 2  # 转换为半长宽高
    yaw = box_params[6]

    # 创建方向包围盒
    obb = o3d.geometry.OrientedBoundingBox(
        center=center,
        R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw)),
        extent=size*2  # extent是全尺寸
    )
    
    # 转换为线框
    lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    lines.paint_uniform_color(color)
    
    return lines

class NuPlan:
    CAMS = ['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0']
    def __init__(self, pkl_file, data_root=None):
        super().__init__()
        self.data_root = data_root
        with open(pkl_file, 'rb') as f:
            self.infos = pickle.load(f)

        self.scenes = defaultdict(list)
        self.token_to_id = {}
        for i, info in enumerate(self.infos['infos']):
            self.scenes[info['scene_token']+'_'+info['scene_name']].append({'timestamp': info['timestamp'], 'token': info['token']})
            self.token_to_id[info['token']] = i
        for k, v in self.scenes.items():
            scene_token, scene_name = k.split('_')
            self.scenes[k] = sorted(v, key=lambda x: x['timestamp'])
            self.scenes[k] = {'first_sample_token': self.scenes[k][0]['token'], 'token': k, 'scene_name': scene_name, 'sample_tokens': [item['token'] for item in self.scenes[k]]}
        pass
        #self.scenes = list(self.scenes.values())
        #self.scenes = sorted(self.scenes, key=lambda x: x['scene_name'].split('-')[1]+x['scene_name'].split('-')[3])
    
    def show_video(self, scene_token, save_path, fps=10, downsample_rate=2):
        assert self.data_root is not None
        scene = self.scenes[scene_token]
        sample_token = scene['first_sample_token']
        video = []
        cnt = 0
        with imageio.get_writer(save_path, fps=fps) as writer:
            for sample_token in scene['sample_tokens']:
            #while sample_token != '' and sample_token is not None:
                sample = self.get('sample', sample_token)
                w, h = 1920//downsample_rate, 1080//downsample_rate
                canvas = np.zeros((h*3, w*3, 3), dtype=np.uint8)
                for cam in self.CAMS:
                    img_path = os.path.join(self.data_root, sample['cams'][cam]['data_path'])
                    img = np.asarray(Image.open(img_path).resize((w, h), Image.Resampling.LANCZOS))
                    if cam == 'CAM_F0':
                        canvas[:h, w:2*w, :] = img
                    elif cam == 'CAM_L0':
                        canvas[:h, :w, :] = img
                    elif cam == 'CAM_R0':
                        canvas[:h, 2*w:3*w, :] = img
                    elif cam == 'CAM_L1':
                        canvas[h:2*h, :w, :] = img
                    elif cam == 'CAM_R1':
                        canvas[h:2*h, 2*w:3*w, :] = img
                    elif cam == 'CAM_L2':
                        canvas[2*h:3*h, :w, :] = img
                    elif cam == 'CAM_B0':
                        canvas[2*h:3*h, w:2*w, :] = img
                    elif cam == 'CAM_R2':
                        canvas[2*h:3*h, 2*w:3*w, :] = img
                #video.append(canvas)
                writer.append_data(canvas)
                #sample_token = sample['sample_next']
                cnt += 1
                print(cnt, sample_token)
        pass

    def show_video_pc(self, scene_token, output_file, fps=10, width=1024, height=768, bboxes=None):
        from pyvirtualdisplay import Display
        H, W = 1080, 1920
        display = Display(visible=False, size=(W, H))
        display.start()
        vis = o3d.visualization.Visualizer()
        ok = vis.create_window(
            width=width,
            height=height,
            visible=False
        )
        if not ok:
            print('Open3D failed')
            return

        # 配置渲染参数
        render_opt = vis.get_render_option()
        render_opt.background_color = np.array((0, 0, 0))
        render_opt.point_size = 3
        render_opt.light_on = True

        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(coord_frame)

        # 配置固定相机参数
        ctr = vis.get_view_control()
        camera_params = ctr.convert_to_pinhole_camera_parameters()
        
        # 设置俯视视角
        camera_params.extrinsic = np.array([
            [1, 0, 0, 0],          # 相机位于Z轴正方向上方
            [0, 1, 0, 0],
            [0, 0, 1, 100],  # Z轴位置
            [0, 0, 0, 1]
        ])
        ctr.convert_from_pinhole_camera_parameters(camera_params)
        # 视角参数
        ctr.set_front([0, 0, -1])  # 朝向Z轴负方向
        ctr.set_up([1, 0, 0])      # X轴为上方向
        ctr.set_lookat([0, 0, 0])      # 注视点云中心
        ctr.set_zoom(30.0)

        # 设置第一人称视角(服务器上跑有问题，不知道为什么)
        # camera_params = o3d.io.read_pinhole_camera_parameters("camera_params.json")
        # ctr.convert_from_pinhole_camera_parameters(camera_params)
                
        # 准备视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # 遍历每一帧，渲染并写入视频
        assert self.data_root is not None
        scene = self.scenes[scene_token]
        sample_token = scene['first_sample_token']
        video = []
        current_boxes = []
        cnt = 0
        for i, sample_token in enumerate(scene['sample_tokens']):
            sample = self.get('sample', sample_token)
            pc_file_name = sample['lidar_path']
            pc0 = pypcd.PointCloud.from_path(os.path.join(self.data_root, pc_file_name))
            pc0 = np.stack([pc0.pc_data['x'], pc0.pc_data['y'], pc0.pc_data['z'], pc0.pc_data['intensity']/255], axis=-1).astype('float32')
            pts = pc0[:, :3]

            os.makedirs('vis/pc', exist_ok=True)
            pts.astype('float32').tofile(f'vis/pc/z_{i}.bin')
            # 创建点云对象
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pts))

            # 检测框
            for box in current_boxes:
                vis.remove_geometry(box, reset_bounding_box=False)
            current_boxes.clear()
            boxes = sample['anns']['gt_boxes']
            for box in boxes:
                box_lines = create_3d_box(box, (0, 1, 0))
                vis.add_geometry(box_lines, reset_bounding_box=False)
                current_boxes.append(box_lines)
            
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # 直接获取屏幕缓冲数据
            image_buffer = vis.capture_screen_float_buffer(do_render=True)
            
            # 转换为uint8数组并调整通道顺序
            frame = (np.asarray(image_buffer) * 255).astype(np.uint8)
            frame = frame[:, :, :3]  # 移除alpha通道
            frame = frame[:, :, ::-1]  # BGR转RGB
            frame = frame[:, ::-1, :]

            video_writer.write(frame)
            print(f"已渲染帧 {i+1}/{len(scene['sample_tokens'])}")
        
        video_writer.release()
        vis.destroy_window()
        print(f"视频已保存到 {output_file}")
        display.stop()

    @property
    def scene(self):
        ret = list(self.scenes.values())
        #ret = sorted(ret, key=lambda x: x['scene_name'].split('-')[1]+x['scene_name'].split('-')[3])
        return ret
    
    def get(self, table_name, item):
        if table_name == 'sample' or table_name == 'sample_data':
            return self.infos['infos'][self.token_to_id[item]]
        elif table_name == 'ego_pose':
            return {'translation': item['ego2global_translation'], 'rotation': item['ego2global_rotation'], 'ego2global': item['ego2global']}
        elif table_name == 'calibrated_sensor':
            return {'translation': item['lidar2ego_translation'], 'rotation': item['lidar2ego_rotation'], 'lidar2ego': item['lidar2ego']}
        elif table_name == 'instances':
            ret = {}
            ret['boxes'] = item['anns']['gt_boxes']
            ret['track_tokens'] = item['anns']['track_tokens']
            ret['category_names'] = item['anns']['gt_names']
            return ret
        else:
            raise KeyError


class SingleScene(Dataset):
    def __init__(self, nusc: NuPlan, scene, nuscenesyaml, box_expand=1.1):
        super().__init__()
        self.nusc = nusc
        self.scene = scene
        self.nuscenesyaml = nuscenesyaml
        
        first_sample_token = self.scene['first_sample_token']
        lidar_data = nusc.get('sample', first_sample_token)
        cur_scene_token = scene_token = lidar_data['scene_token']
        self.all_tokens = []
        self.all_tokens = self.scene['sample_tokens']
        self.box_expand = box_expand
        # sample_token = first_sample_token
        # while sample_token != '' and sample_token is not None and cur_scene_token == scene_token:
        #     self.all_tokens.append(sample_token)
        #     sample_token = lidar_data['sample_next']
        #     if sample_token != '' and sample_token is not None:
        #         lidar_data = nusc.get('sample', sample_token)
        #         cur_scene_token = lidar_data['scene_token']
    
    def __len__(self):
        return len(self.all_tokens)
    
    @staticmethod
    def collect_fn(batch):
        assert len(batch) == 1
        return batch[0]

    def __getitem__(self, idx):
        nusc = self.nusc
        data_root = self.nusc.data_root
        nuscenesyaml = self.nuscenesyaml
        learning_map = nuscenesyaml['learning_map']
        first_sample_token = self.scene['first_sample_token']

        my_sample = self.nusc.get('sample', self.all_tokens[idx])
        lidar_data = my_sample # nusc.get('sample_data', my_sample)
        lidar_data['filename'] = lidar_data['lidar_path']
        lidar_data['is_key_frame'] = True
        lidar_ego_pose0 = nusc.get('ego_pose', lidar_data)
        lidar_calibrated_sensor0 = nusc.get('calibrated_sensor', lidar_data)


        flag_has_lidarseg = False # nuplan do not provide lidarseg
        try:
            lidar_sd_token = lidar_data['token']
            seg_name = nusc.get('lidarseg', lidar_sd_token)
        except KeyError:
            flag_has_lidarseg = False

        ############################# get boxes ##########################
        #lidar_path, boxes, _ = nusc.get_sample_data(lidar_data['token'])
        #boxes_token = [box.token for box in boxes]
        #object_tokens = [nusc.get('sample_annotation', box_token)['instance_token'] for box_token in boxes_token]
        #object_category = [nusc.get('sample_annotation', box_token)['category_name'] for box_token in boxes_token]
        
        instances = nusc.get('instances', lidar_data)
        boxes = instances['boxes']
        object_tokens = instances['track_tokens']
        object_category = instances['category_names']

        ############################# get object categories ##########################
        converted_object_category = []
        for category in object_category:
            for (j, label) in enumerate(nuscenesyaml['labels']):
                if category == nuscenesyaml['labels'][label]:
                    converted_object_category.append(np.vectorize(learning_map.__getitem__)(label).item())

        ############################# get bbox attributes ##########################
        # locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        # dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        # rots = np.array([b.orientation.yaw_pitch_roll[0]
        #                  for b in boxes]).reshape(-1, 1)
        # gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d = boxes
        dims = boxes[:, 3:6]
        #gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        #gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # Move the bbox slightly down in the z direction
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * self.box_expand  # Slightly expand the bbox to wrap all object points
        ############################# get LiDAR points with semantics ##########################
        pc_file_name = lidar_data['filename']  # load LiDAR names
        # pc0 = np.fromfile(os.path.join(data_root, pc_file_name),
        #                   dtype=np.float32,
        #                   count=-1).reshape(-1, 5)[..., :4]
        #pc0 = o3d.io.read_point_cloud(os.path.join(data_root, pc_file_name))

        if not os.path.exists(os.path.join(data_root, pc_file_name)):
            return None
        pc0 = pypcd.PointCloud.from_path(os.path.join(data_root, pc_file_name))
        pc0 = np.stack([pc0.pc_data['x'], pc0.pc_data['y'], pc0.pc_data['z'], pc0.pc_data['intensity']/255], axis=-1).astype('float32')
        if lidar_data['is_key_frame'] and flag_has_lidarseg:  # only key frame has semantic annotations
            lidar_sd_token = lidar_data['token']
            lidarseg_labels_filename = os.path.join(nusc.data_root,
                                                    nusc.get('lidarseg', lidar_sd_token)['filename'])

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(learning_map.__getitem__)(points_label)

            pc_with_semantic = np.concatenate([pc0[:, :3], points_label], axis=1)  

        # compute in gpu is faster
        points_in_boxes = None
        # points_in_boxes = points_in_boxes_cpu(torch.from_numpy(pc0[:, :3][np.newaxis, :, :]),
        #                                       torch.from_numpy(gt_bbox_3d[np.newaxis, :]))
        
        ret = dict(
            scene_token=self.scene['token'],
            pc0=pc0,
            gt_bbox_3d=gt_bbox_3d,
            converted_object_category=converted_object_category,
            object_tokens=object_tokens,
            flag_has_lidarseg=flag_has_lidarseg,
            pc_file_name=pc_file_name,
            lidar_data=lidar_data,
            **dict(points_in_boxes=points_in_boxes) if points_in_boxes is not None else {}
        )
        return ret
            

class NKSRScene(Dataset):
    def __init__(self, nusc: NuPlan, scene, nuscenesyaml, frame_infos, agg_lidar, agg_lidar_semantic, object_zoo, sensor_locs, config):
        super().__init__()
        self.nusc = nusc
        self.scene = scene
        self.nuscenesyaml = nuscenesyaml
        self.frame_infos = frame_infos
        self.agg_lidar = agg_lidar
        self.agg_lidar_semantic = agg_lidar_semantic
        self.config = config
        self.object_zoo = object_zoo
        self.sensor_locs = sensor_locs
        
        first_sample_token = self.scene['first_sample_token']
        lidar_data = nusc.get('sample', first_sample_token)
        self.all_tokens = []
        self.all_tokens = self.scene['sample_tokens']
        # sample_token = first_sample_token
        # while sample_token != '' and sample_token is not None:
        #     self.all_tokens.append(sample_token)
        #     sample_token = lidar_data['sample_next']
        #     if sample_token != '' and sample_token is not None:
        #         lidar_data = nusc.get('sample', sample_token)


        # my_sample = self.nusc.get('sample', self.all_tokens[0])
        # lidar_data = my_sample
        # self.lidar_ego_pose0 = nusc.get('ego_pose', lidar_data)
        # self.lidar_calibrated_sensor0 = nusc.get('calibrated_sensor', lidar_data)

        self.lidar_height = 1.8
        self.remove_exist_object_points = True
    
    def __len__(self):
        return len(self.all_tokens)
    
    @staticmethod
    def collect_fn(batch):
        assert len(batch) == 1
        return batch[0]

    def __getitem__(self, idx):
        self.lidar_ego_pose0 = self.frame_infos[0]['lidar_ego_pose']
        self.lidar_calibrated_sensor0 = self.frame_infos[0]['lidar_calibrated_sensor']

        object_zoo = self.object_zoo
        dict_list = self.frame_infos
        lidar_pc = self.agg_lidar
        lidar_pc_with_semantic = self.agg_lidar_semantic
        nusc = self.nusc
        _dict = dict_list[idx]
        is_key_frame = _dict['is_key_frame']

        ################## convert the static scene to the target coordinate system ##############
        lidar_calibrated_sensor = _dict['lidar_calibrated_sensor']
        lidar_ego_pose = _dict['lidar_ego_pose']
        lidar_pc_i = lidar_to_world_to_lidar(lidar_pc.copy(),
                                             self.lidar_calibrated_sensor0.copy(),
                                             self.lidar_ego_pose0.copy(),
                                             lidar_calibrated_sensor,
                                             lidar_ego_pose)
        lidar_pc_i_semantic = lidar_to_world_to_lidar(lidar_pc_with_semantic.copy(),
                                                      self.lidar_calibrated_sensor0.copy(),
                                                      self.lidar_ego_pose0.copy(),
                                                      lidar_calibrated_sensor,
                                                      lidar_ego_pose)
        point_cloud = lidar_pc_i.points.T[:, :3]
        #point_cloud_with_semantic = lidar_pc_i_semantic.points.T
        point_cloud_with_semantic = point_cloud.copy()


        ################## sensor location for nksr ################
        sensor_locs_i = lidar_to_world_to_lidar(np.concatenate([self.sensor_locs.copy(), np.ones((self.sensor_locs.shape[0], 1), dtype=self.sensor_locs.dtype)], axis=-1),
                                             {'rotation': [1.0, 0.0, 0.0, 0.0], 'translation': np.zeros((3,), dtype=np.float64)},
                                             {'rotation': [1.0, 0.0, 0.0, 0.0], 'translation': np.zeros((3,), dtype=np.float64)},
                                             lidar_calibrated_sensor,
                                             lidar_ego_pose)
        sensor_locs = sensor_locs_i.points.T[:, :3]

        ################# load bbox of target frame ##############
        # lidar_path, boxes, _ = nusc.get_sample_data(dict['lidar_token'])
        # locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        # dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        # rots = np.array([b.orientation.yaw_pitch_roll[0]
        #                  for b in boxes]).reshape(-1, 1)
        # gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)

        ################# bbox placement ##############
        object_points_list = []
        object_semantic_list = []
        object_sensor_locs_list = []
        for j, (object_token, object_info) in enumerate(_dict['objects'].items()):
            if object_token in object_zoo:
                # 0x 1y 2z 3l 4w 5h 6yaw
                object_box = np.array(object_info[1])
                # not need, already done in previous 
                # object_box[2] -= object_box[5] / 2.
                # object_box[2] = object_box[2] - 0.1
                # object_box[3:6] = object_box[3:6] * 1.1
                rot = object_box[6]
                loc = object_box[0:3]

                points = object_zoo[object_token]['points']
                Rot = Rotation.from_euler('z', rot, degrees=False)
                rotated_object_points = Rot.apply(points)
                points = rotated_object_points + loc

                # sensor loc
                sensor_locs_object = object_zoo[object_token]['sensor_locs']
                Rot = Rotation.from_euler('z', rot, degrees=False)
                rotated_sensor_locs_object = Rot.apply(sensor_locs_object)
                sensor_locs_object = rotated_sensor_locs_object + loc

                if points.shape[0] >= 5:
                    points_in_boxes = points_in_boxes_cpu(torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                                                        torch.from_numpy(object_box[None][np.newaxis, :]))
                    points_in_boxes_mask = points_in_boxes[0, :, 0].bool()
                    points = points[points_in_boxes_mask]
                    sensor_locs_object = sensor_locs_object[points_in_boxes_mask]

                if self.remove_exist_object_points:
                    points_in_boxes = points_in_boxes_cpu(torch.from_numpy(point_cloud_with_semantic[:, :3][np.newaxis, :, :]),
                                                        torch.from_numpy(object_box[None][np.newaxis, :]))
                    points_in_boxes_mask = points_in_boxes[0, :, 0].bool()
                    point_cloud_with_semantic = point_cloud_with_semantic[~points_in_boxes_mask]

                object_points_list.append(points)
                object_sensor_locs_list.append(sensor_locs_object)
                semantics = np.ones_like(points[:, 0:1]) * object_zoo[object_token]['label']
                object_semantic_list.append(np.concatenate([points[:, :3], semantics], axis=1))

        try:  # avoid concatenate an empty array
            temp = np.concatenate(object_points_list)
            scene_points = np.concatenate([point_cloud, temp])
            sensor_locs = np.concatenate([sensor_locs, np.concatenate(object_sensor_locs_list)])

        except:
            scene_points = point_cloud
        
        temp_seg = np.concatenate(object_semantic_list)
        scene_points_label = np.zeros((point_cloud_with_semantic.shape[0],), dtype=scene_points.dtype)
        scene_points_label = np.concatenate([scene_points_label, temp_seg[:, -1]+1])
        point_cloud_with_semantic = np.concatenate([point_cloud_with_semantic, temp])
        scene_semantic_points = np.concatenate([point_cloud_with_semantic, scene_points_label[:, None]], axis=-1)
        # try:
        #     temp = np.concatenate(object_semantic_list)
        #     scene_semantic_points = np.concatenate([point_cloud_with_semantic, temp])
        # except:
        #     scene_semantic_points = point_cloud_with_semantic

        ################## remain points with a spatial range ##############
        mask = (np.abs(scene_points[:, 0]) < abs(self.config['pc_range'][0])) & (np.abs(scene_points[:, 1]) < abs(self.config['pc_range'][1])) \
               & (scene_points[:, 2] > self.config['pc_range'][2]) & (scene_points[:, 2] < self.config['pc_range'][-1])
        scene_points = scene_points[mask]
        sensor_locs = sensor_locs[mask]
        sensor_locs[:, -1] += self.lidar_height # FIXME: lidar height in nuplan?

        ################## get mesh via Possion Surface Reconstruction ##############
        # point_cloud_original = o3d.geometry.PointCloud()
        # with_normal2 = o3d.geometry.PointCloud()
        # point_cloud_original.points = o3d.utility.Vector3dVector(scene_points[:, :3])
        # with_normal = preprocess(point_cloud_original, self.normal_esti_config)
        # with_normal2.points = with_normal.points
        # with_normal2.normals = with_normal.normals
        # # mesh, _ = create_mesh_from_map(None, config['depth'], config['n_threads'], config['min_density'], with_normal2)

        # point = np.asarray(with_normal.points)
        # normal = np.asarray(with_normal.normals)

        point = scene_points

        ret = dict(
            point=point,
            #normal=normal,
            sensor_locs=sensor_locs,
            scene_points=scene_points,
            scene_semantic_points=scene_semantic_points,
            sample_token=_dict['sample_token'],
            lidar_token=_dict['lidar_token'],
            object_points_list=object_points_list,
            object_semantic_list=object_semantic_list
        )
        return ret

class MeshScene3(Dataset):
    def __init__(self, nusc: NuPlan, scene, nuscenesyaml, frame_infos, agg_lidar, agg_lidar_semantic, scene_mesh, object_zoo, sensor_locs, config):
        super().__init__()
        self.nusc = nusc
        self.scene = scene
        self.nuscenesyaml = nuscenesyaml
        self.frame_infos = frame_infos
        self.agg_lidar = agg_lidar
        self.agg_lidar_semantic = agg_lidar_semantic
        self.config = config
        self.object_zoo = object_zoo
        self.sensor_locs = sensor_locs
        self.scene_mesh = scene_mesh
        
        first_sample_token = self.scene['first_sample_token']
        lidar_data = nusc.get('sample', first_sample_token)
        self.all_tokens = []
        self.all_tokens = self.scene['sample_tokens']


        self.lidar_height = 1.8
        self.remove_exist_object_points = True
    
    def __len__(self):
        return len(self.all_tokens)
    
    @staticmethod
    def collect_fn(batch):
        assert len(batch) == 1
        return batch[0]

    def __getitem__(self, idx):
        dict_list = self.frame_infos
        _dict = dict_list[idx]
        ret = dict(
            sample_token=_dict['sample_token'],
            lidar_token=_dict['lidar_token'],
            mesh_points=self.scene_mesh.v,
            cur_objects=_dict['objects'],
            #_dict=_dict
            lidar_calibrated_sensor=_dict['lidar_calibrated_sensor'],
            lidar_ego_pose=_dict['lidar_ego_pose']
        )
        return ret
