import torch
import os
import pickle
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from monoscene.data.utils.helpers import (
    vox2pix,
    compute_local_frustums,
    compute_CP_mega_matrix,
)

import cv2
from typing import Tuple

def undistort(
        K: np.ndarray,
        dist_coeffs: list,
        image: Image.Image,
        alpha: float = 0.0,
        crop: bool = False,
        interpolation: int = cv2.INTER_LINEAR
) -> Tuple[np.ndarray, Image.Image]:
    if K.shape != (3, 3):
        raise ValueError("K must be a 3x3 matrix")
    
    #准备数据
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    K = K.astype(np.float64)
    dist = np.array(dist_coeffs).astype(np.float64).reshape(-1, 1)

    #计算新的相机矩阵
    K_new, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix=K,
        distCoeffs=dist,
        imageSize=(w, h),
        alpha=alpha,
        newImgSize=(w, h)
    )

    # 预计算映射并去畸变
    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix=K,
        distCoeffs=dist,
        R=np.eye(3, dtype=np.float64),
        newCameraMatrix=K_new,
        size=(w, h),
        m1type=cv2.CV_32FC1,
    )

    undist = cv2.remap(
        src=img_np,
        map1=map1,
        map2=map2,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT
    )

    if crop:
        x, y, rw, rh = roi
        undist = undist[y:y+rh, x:x+rw]
        K_new = K_new.copy()
        K_new[0, 2] -= x
        K_new[1, 2] -= y
    
    undist_pil = Image.fromarray(undist)
    return K_new, undist_pil


class NuplanDataset(Dataset):
    def __init__(
        self,
        root,
        pkls_path,
        occ_path,
        project_scale=2,
        frustum_size=4,
        color_jitter=None,
        fliplr=0.0,
        target_width=1920,
        target_height=1080,
        split='train'
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.occ_path = occ_path
        self.n_classes = 9
        self.img_H = target_height
        self.img_W = target_width        
        self.frustum_size = frustum_size
        self.project_scale = project_scale
        self.output_scale = int(self.project_scale / 2)
        self.voxel_size = 0.25
        self.fliplr = fliplr
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.vox_origin = np.array([0, -50, -5])
        self.pc_range = [0, -25, -5, 50, 25, 3]
        self.scene_size = (50, 50, 8)
        self.infos = pickle.load(open(os.path.join(pkls_path, "mini", f"nuplan_mini_10hz_{split}.pkl"), "rb"))["infos"]

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        scan = self.infos[index]
        sequence = scan['scene_token']
        frame_id = scan['token']
        cam_info = scan['cams']['CAM_F0']
        rgb_path = os.path.join(self.root, cam_info['data_path'])
        img = Image.open(rgb_path).convert("RGB")
        camera_intrinsic = np.eye(4).astype(np.float32)
        camera_intrinsic[:3, :3]= cam_info['cam_intrinsic']
        camera_intrinsic = np.array(camera_intrinsic)
        cam_k = camera_intrinsic[:3, :3]

        # 创建相机到激光雷达的变换矩阵
        c2e = np.eye(4).astype(np.float32)
        c2e[:3, :3] = cam_info['sensor2lidar_rotation'] #Quaternion(cam_info['sensor2lidar_rotation'])
        c2e[:3, 3] = np.array(cam_info['sensor2lidar_translation'])
        c2e = np.array(c2e)
        e2c = np.linalg.inv(c2e)
        T_velo_2_cam = e2c

        # 处理相机畸变参数
        k1, k2, p1, p2, k3 = cam_info['distortion']
        cam_k, img = undistort(cam_k, [k1, k2, p1, p2, k3], img)

        # 创建相机内参矩阵
        camera_intrinsic = np.eye(4).astype(np.float32)
        camera_intrinsic[:3, :3] = cam_k

        # 处理图像尺寸调整
        if self.img_W != img.size[0] or self.img_H != img.size[1]:
            ori_width, ori_height = img.size
            img = img.resize((self.img_W, self.img_H), Image.Resampling.LANCZOS)
            camera_intrinsic[0, :] *= (self.img_W / ori_width)
            camera_intrinsic[1, :] *= (self.img_H / ori_height)
            cam_k[0, :] *= (self.img_W / ori_width)
            cam_k[1, :] *= (self.img_H / ori_height)

        proj_matrix = camera_intrinsic @ e2c
        
        data = {
            "frame_id": frame_id,
            "sequence": sequence,
            "P": camera_intrinsic,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix": proj_matrix,
        }
        scale_3ds = [self.output_scale, self.project_scale]
        data["scale_3ds"] = scale_3ds
        data["cam_k"] = cam_k

        for scale_3d in scale_3ds:

            # compute the 3D-2D mapping
            projected_pix, fov_mask, pix_z = vox2pix(
                T_velo_2_cam,
                cam_k,
                self.vox_origin,
                self.voxel_size * scale_3d,
                self.img_W,
                self.img_H,
                self.scene_size,
            )            

            data["projected_pix_{}".format(scale_3d)] = projected_pix
            data["pix_z_{}".format(scale_3d)] = pix_z
            data["fov_mask_{}".format(scale_3d)] = fov_mask

        target_1_path = os.path.join(self.occ_path, frame_id + ".npy")
        target = np.load(target_1_path)
        if self.split != "train":
            target = np.concatenate([np.zeros((*target.shape[:2], 8), dtype=target.dtype), target[:, :, :24]], axis=-1)
        target = target[target.shape[0]//2:, 100:300, :]
        data["target"] = target
        if self.split == 'train':
            target_1_8 = np.load(os.path.join("data/nuplan_occ_d8", frame_id + ".npy"))

            CP_mega_matrix = compute_CP_mega_matrix(target_1_8)
            data["CP_mega_matrix"] = CP_mega_matrix

        # Compute the masks, each indicate the voxels of a local frustum
        if self.split != "test":
            projected_pix_output = data["projected_pix_{}".format(self.output_scale)]
            pix_z_output = data[
                "pix_z_{}".format(self.output_scale)
            ]
            frustums_masks, frustums_class_dists = compute_local_frustums(
                projected_pix_output,
                pix_z_output,
                target,
                self.img_W,
                self.img_H,
                dataset="kitti",
                n_classes=self.n_classes,
                size=self.frustum_size,
            )
        else:
            frustums_masks = None
            frustums_class_dists = None
        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        img = Image.open(rgb_path).convert("RGB")

        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0

        # Fliplr the image
        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            for scale in scale_3ds:
                key = "projected_pix_" + str(scale)
                data[key][:, 0] = img.shape[1] - 1 - data[key][:, 0]

        data["img"] = self.normalize_rgb(img)
        return data
