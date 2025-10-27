from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelizationSPConv():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points

def voxelize_point_cloud(points, voxel_size, point_cloud_range):
    """
    参数:
      points: [N, 4] 的张量，前3列为xyz，第四列为label（假设label为整数类型）。
      voxel_size: [vx, vy, vz] 的列表或张量，表示每个体素的大小。
      point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]，点云的范围。
      
    返回:
      unique_voxels: [M, 3] 的张量，每一行表示体素的索引坐标（整数）。
      voxel_labels: [M] 的张量，表示每个体素对应的 label（由体素内点的众数组成）。
    """
    device = points.device  # 确保所有操作在同一设备上
    voxel_size = torch.tensor(voxel_size, device=device, dtype=points.dtype)
    min_bound = torch.tensor(point_cloud_range[:3], device=device, dtype=points.dtype)
    max_bound = torch.tensor(point_cloud_range[3:], device=device, dtype=points.dtype)
    
    # 计算体素坐标 (floor操作)
    voxel_coords = torch.floor((points[:, :3] - min_bound) / voxel_size).to(torch.int32)
    
    # 获取每个体素的唯一坐标及每个点对应的体素索引
    unique_voxels, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
    
    # 获取标签（确保为整数类型）
    labels = points[:, 3].to(torch.int64)
    
    # 假设 label 的取值范围是连续的，这里确定类别数（也可以通过传入参数指定）
    num_classes = int(labels.max().item()) + 1

    # 对每个点做 one-hot 编码
    one_hot = F.one_hot(labels, num_classes=num_classes).to(torch.int32)  # 形状 [N, num_classes]
    
    # 对每个体素内的点进行聚合，统计每个类别出现次数
    num_voxels = unique_voxels.shape[0]
    voxel_label_counts = torch.zeros((num_voxels, num_classes), device=device, dtype=torch.int32)
    voxel_label_counts = voxel_label_counts.index_add(0, inverse_indices, one_hot)
    
    # 取众数作为体素 label（即取计数最大的类别）
    voxel_labels = torch.argmax(voxel_label_counts, dim=1)
    
    return unique_voxels, voxel_labels