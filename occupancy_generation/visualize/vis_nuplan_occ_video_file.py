# 
import os
# os.environ["QT_DEBUG_PLUGINS"]="1"
from pyvirtualdisplay import Display
display = Display(visible=False, size=(900, 900))
display.start()

from mayavi import mlab
# mlab.options.offscreen = True

from pathlib import Path
import numpy as np
# from nuscenes.nuscenes import NuScenes
import os
import glob
import imageio


classname_to_color = {  # RGB.
    0: (255, 255, 255),  # Black. noise
    1: (112, 128, 144),  # Slategrey barrier
    2: (220, 20, 60),  # Crimson bicycle 深红色, vehicle
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

def get_grid_coords(dims, resolution):
	"""
	:param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
	:return coords_grid: is the center coords of voxels in the grid
	"""
	g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
	# g_xx = g_xx[::-1]
	g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
	# g_yy = g_yy[::-1]
	g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]
	# Obtaining the grid with coords...
	xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
	coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
	coords_grid = coords_grid.astype(np.float32)
	resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])
	coords_grid = (coords_grid * resolution) + resolution / 2
	return coords_grid

def custom_colormap(plt_plot, colormap=classname_to_color):
    ori_colormap = plt_plot.module_manager.scalar_lut_manager.lut.table.to_array()
    for key, value in colormap.items():
        ori_colormap[key, :3] = value
    plt_plot.module_manager.scalar_lut_manager.lut.table = ori_colormap.astype(np.uint8)[:17]


def draw_nusc_occupancy(
    voxels_input,
    vox_origin,
    voxel_bs=0,
    voxel_size=0.2,
    grid=None,
    save_folder=None,
    ):
    writer = imageio.get_writer(f'{voxel_bs}_forcasting.mp4', fps=10)

    for i in range(voxels_input.shape[0]):
        figure=mlab.figure(size=(900, 900), bgcolor=(1, 1, 1))

        voxels=voxels_input[i]

        # Compute the voxels coordinates
        grid_coords = get_grid_coords(
            [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
        ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T


        # Get the voxels inside FOV
        fov_grid_coords = grid_coords

        # Remove empty and unknown voxels
        fov_voxels = fov_grid_coords[
            (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
            # (fov_grid_coords[:, 3] == 0) | (fov_grid_coords[:, 3] > 1) | (fov_grid_coords[:, 3] < 19)
        ]

        draw(fov_voxels, voxel_size=voxel_size, vis_root=save_folder, idx=i, figure=figure, mlab=mlab)

        # frame = mlab.screenshot(mode="rgb", antialiased=True)
        # writer.append_data(frame)
        print(f"{i}/{voxels_input.shape[0]}")
    mlab.close(all=True)
    writer.close()

def draw(voxels, voxel_size=0.2, vis_root=None, idx=0, figure=None, mlab=None):

    x = voxels[:, 1]  # x position of point
    y = voxels[:, 0]  # y position of point
    z = -voxels[:, 2]  # z position of point
    point_color = np.zeros(voxels.shape[0])

    for cls_index in range(16):
        class_point = voxels[:, 3] == cls_index
        point_color[class_point] = cls_index+1 


    # figure = mlab.figure(size=(900, 900), bgcolor=(1, 1, 1))
    plt_plot = mlab.points3d(
        x,
        y,
        -z,
        point_color,
        scale_factor=0.5,
        scale_mode='vector',
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=17,
    )
    custom_colormap(plt_plot)

    # view_type ='back_view'
    # if view_type =='back_view':
    #     scene = figure
    #     scene.scene.z_plus_view()
    #     scene.scene.camera.position = [-1.1612566981665453, -63.271696093007456, 33.06645769267362]
    #     scene.scene.camera.focal_point = [-0.0828344205684326, -0.029545161654287222, -1.078433202901462]
    #     scene.scene.camera.view_angle = 45.0
    #     scene.scene.camera.view_up = [-0.011200801911309498, 0.4752037522484654, 0.879804487306994]
    #     scene.scene.camera.clipping_range = [0.18978054185107493, 189.78054185107493]
    #     scene.scene.camera.compute_view_plane_normal()
    #     scene.scene.render()

    save_file = f"outputs/nuplan_occ_dit/eval_uncon_occbev_2025-07-22-00-46-59/visualizations/png"
    os.makedirs(save_file, exist_ok=True)
    filename = str(save_file) + f'/{vis_root[0]}_{idx}_bev.png'
    mlab.draw()
    mlab.savefig(filename = filename)
    mlab.close()
    print(filename)
    # mlab.show()

# def render_one_npy()

if __name__ == '__main__':
    point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]

    folder_path = 'outputs/nuplan_occ_dit/eval_uncon_occbev_2025-07-22-00-46-59/visualizations'  # 替换为你的文件夹路径
    files_to_rename = []  # 存储符合条件的文件名列表
    # 遍历文件夹及其子文件夹中所有文件
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.npy'):
                files_to_rename.append(os.path.join(root, filename))
    # 按文件名排序
    files_to_rename.sort()
    
    for filepath in files_to_rename:
        print(filepath)
        pred_voxels =  np.load(filepath)
        vis_voxels = pred_voxels
        print(pred_voxels.shape)
        # if len(pred_voxels.shape) == 4:
        #     vis_voxels = pred_voxels[0]
        # elif len(pred_voxels.shape) == 5:
        #     vis_voxels = pred_voxels[0][0]
        # elif len(pred_voxels.shape) == 3:
        #     vis_voxels = pred_voxels
        
        # if vis_voxels.shape[0] == 200:
        #     occ_size = [200, 200, 16]
        # elif vis_voxels.shape[0] == 400:
        #     occ_size = [400, 400, 32]
        # else:
        #     raise ValueError(f"Invalid voxel size: {vis_voxels.shape[0]}")
        occ_size = [200, 200, 16]
        voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
        voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
        voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
        voxel_size = [voxel_x, voxel_y, voxel_z]

        # print(vis_voxels.shape)
        for i in range(vis_voxels.shape[0])[:1]:
            # for j in range(vis_voxels.shape[1]):
                # if i==1:
                draw_nusc_occupancy(
                        voxels_input=vis_voxels[i], 
                        voxel_bs = i,
                        vox_origin=np.array(point_cloud_range[:3]),
                        voxel_size=np.array(voxel_size),
                        grid=np.array(occ_size),
                        save_folder = filepath.split("/")[-1:],
                )