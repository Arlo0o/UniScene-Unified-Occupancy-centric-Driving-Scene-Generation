# from xvfbwrapper import Xvfb
# vdisplay = Xvfb(width=1920, height=1080)
# vdisplay.start()

import numpy as np
from mayavi import mlab
# mlab.options.offscreen = True
# from nuscenes.nuscenes import NuScenes
import os
import glob
import imageio
import pickle
# from nuscenes.utils.splits import create_splits_scenes


classname_to_color = {  # RGB.
    0: (255, 255, 255),  # Black. noise
    1: (112, 128, 144),  # Slategrey barrier
    2: (220, 20, 60),  # Crimson bicycle
    3: (255, 127, 80),  # Orangered bus
    4: (255, 158, 0),  # Orange car
    5: (233, 150, 70),  # Darksalmon construction
    6: (255, 61, 99),  # Red motorcycle
    7: (0, 0, 230),  # Blue pedestrian
    8: (47, 79, 79),  # Darkslategrey trafficcone
    9: (255, 140, 0),  # Darkorange trailer
    10: (255, 99, 71),  # Tomato truck
    11: (0, 207, 191),  # nuTonomy green driveable_surface
    12: (175, 0, 75),  # flat other
    13: (75, 0, 75),  # sidewalk
    14: (112, 180, 60),  # terrain
    15: (222, 184, 135),  # Burlywood mannade
    16: (0, 175, 0),  # Green vegetation
}



def custom_colormap(plt_plot, colormap=classname_to_color):
    ori_colormap = plt_plot.module_manager.scalar_lut_manager.lut.table.to_array()
    for key, value in colormap.items():
        ori_colormap[key, :3] = value
    plt_plot.module_manager.scalar_lut_manager.lut.table = ori_colormap.astype(np.uint8)[:17]


def draw(voxels, voxel_size=0.2, vis_root=None, idx=0):
    figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))
    scene = figure.scene
    plt_plot = mlab.points3d(
        voxels[:, 0],
        voxels[:, 1],
        voxels[:, 2],
        voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.5 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=16,
    )
 
    scene.render()


    custom_colormap(plt_plot)

    mlab.show()

    # mlab.draw()
    # vis_path = os.path.join(vis_root, '{:0>4d}.png'.format(idx))
    # mlab.savefig(filename=vis_path)

def convert_to_voxel_grid_int(voxels_):
    voxel = np.zeros((800, 800, 64), dtype=np.int32)
    voxel[voxels_[:, 0].astype(np.int32), voxels_[:, 1].astype(np.int32), voxels_[:, 2].astype(np.int32)] = voxels_[:, 3].astype(np.int32)+1
    return voxel



def convert_to_voxel_grid(fov_voxels, range_bounds=[-50, -50, -5, 50, 50, 3], voxel_size=0.125):
    """
    将 (N, 4) 形式的体素点云数据转换为固定大小的 3D 体素网格。
    :param fov_voxels: (N, 4) 形式的 NumPy 数组, 每行是 (x, y, z, label)
    :param range_bounds: 体素空间的范围 [xmin, ymin, zmin, xmax, ymax, zmax]
    :param voxel_size: 体素的大小 (默认 0.6m)
    :return: 3D 体素网格 (X, Y, Z)
    """
    # 计算网格的大小 (确保固定为 200x200x16)
    occ_size = [800,800,64]
    x_min, y_min, z_min, x_max, y_max, z_max = range_bounds
    x_size = int((x_max - x_min) / 0.125)
    y_size = int((y_max - y_min) / 0.125)
    z_size = int((z_max - z_min) / 0.125)
    print(f"x_size: {x_size}, y_size: {y_size}, z_size: {z_size}")
    # assert x_size == 200 and y_size == 200 and z_size == 16, "体素网格大小应为 (200, 200, 16)"
    
    # 创建 3D 体素网格，并填充默认类别 0（空白体素）
    voxel_grid = np.zeros((x_size, y_size, z_size), dtype=np.int32)
    # 将 (N, 4) 数据转换到 3D 网格中
    for x, y, z, label in fov_voxels:
        # 计算体素索引
        x_idx = int((x - x_min) / voxel_size)
        y_idx = int((y - y_min) / voxel_size)
        z_idx = int((z - z_min) / voxel_size)
        
        # 确保索引在有效范围内
        if 0 <= x_idx < x_size and 0 <= y_idx < y_size and 0 <= z_idx < z_size:
            print(f"x_idx: {x_idx}, y_idx: {y_idx}, z_idx: {z_idx}")
            # if label == 0:
                # label = 1
            voxel_grid[x_idx, y_idx, z_idx] = int(label+1)
    
    return voxel_grid

if __name__ == '__main__':

    filepath = 'occ_n4/00009841879a5bb9.npz'

    # npz
    voxels_ = np.load(filepath)
    voxels_ = voxels_['occ']
    print(voxels_.shape)
    print(f"max x is {np.max(voxels_[:, 0])} min x is {np.min(voxels_[:, 0])}, max y is {np.max(voxels_[:, 1])} min y is {np.min(voxels_[:, 1])}, max z is {np.max(voxels_[:, 2])} min z is {np.min(voxels_[:, 2])}")

    voxels_[:,3]=voxels_[:,3]
    for i in range(20):
        if np.any(voxels_[:, 3] == i):
            print(f"label {i} in points_with_label {np.any(voxels_[:, 3] == i)} sum {np.sum(voxels_[:, 3] == i)}")

    voxel_grid = convert_to_voxel_grid_int(voxels_)
    # voxel_grid = convert_to_voxel_grid(voxels_)
    np.save('./occ_grid/occ_grid.npy', voxel_grid)
    print(voxel_grid.shape)
    for i in range(20):
        if np.any(voxel_grid == i):
            print(f"label {i} in points_with_label {np.any(voxel_grid == i)} sum {np.sum(voxel_grid == i)}")


    # with open(filepath, 'rb') as f:
    #     voxels_ = pickle.load(f)
    voxels_[:, -1] += 1

    draw(voxels_, voxel_size=2)