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
from PIL import Image


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
    voxels,
    vox_origin,
    voxel_size=0.2,
    grid=None,
    save_path=None,
    view_params=None,
    ):
    w, h, z = voxels.shape

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    grid_coords[grid_coords[:, 3] == 15, 3] = 12
    # car_vox_range = np.array([
    #     [w//2 - 2 - 4, w//2 - 2 + 4],
    #     [h//2 - 2 - 4, h//2 - 2 + 4],
    #     [z//2 - 2 - 3, z//2 - 2 + 3]
    # ], dtype=np.int32)

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
        # (fov_grid_coords[:, 3] == 0) | (fov_grid_coords[:, 3] > 1) | (fov_grid_coords[:, 3] < 19)
    ]
    return draw(fov_voxels, voxel_size=voxel_size, save_path=save_path, idx=0, view_params=view_params)


def draw(voxels, voxel_size=0.2, save_path=None, idx=0, view_params=None):
    x = voxels[:, 1]  # x position of point
    y = voxels[:, 0]  # y position of point
    z = -voxels[:, 2]  # z position of point
    point_color = np.zeros(voxels.shape[0])

    for cls_index in range(16):
        class_point = voxels[:, 3] == cls_index
        point_color[class_point] = cls_index+1 

    figure = mlab.figure(size=(900, 900), bgcolor=(1, 1, 1))
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

    if view_params is not None:
        mlab.view(*view_params)
    view_params = mlab.view()

    mlab.draw()
    mlab.savefig(filename = save_path)
    mlab.close(all=True)
    print(f'Saved {save_path}')
    # mlab.show()

    return view_params


if __name__ == '__main__':
    folder_path = 'outputs/nuplan_occ_dit/eval_uncon_occbev_2025-07-24-00-36-57/visualizations/'  # 替换为你的文件夹路径
    save_folder = 'outputs/nuplan_occ_dit/eval_uncon_occbev_2025-07-24-00-36-57/show_all/'  # 替换为你的文件夹路径
    os.makedirs(save_folder, exist_ok=True)

    occ_ids = list(map(lambda fn: int(fn.split('_')[0]), os.listdir(folder_path)))
    occ_ids = list(sorted(set(occ_ids)))
    print(f'Number of id: {len(occ_ids)}')

    for id in occ_ids:
        pred_filename = f'{id:05d}_pred.npy'
        gt_filename = f'{id:05d}_occ_ori.npy'

        pred_voxels = gt_voxels = None   

        # load pred
        try:
            pred_voxels = np.load(
                os.path.join(folder_path, pred_filename))  # [1, T, H, W, Z]
            pred_voxels = pred_voxels[0]  # -> [T, H, W, Z]
            occ_size_hw = pred_voxels.shape[1]
        except:
            print(f'Prediction occ {id=} does not exist')

        # load gt
        try:
            gt_voxels = np.load(
                os.path.join(folder_path, gt_filename))
            gt_voxels = gt_voxels[0][0]  # -> [H, W, Z]
            occ_size_hw = gt_voxels.shape[0]
        except:
            print(f'GT occ {id=} does not exist!')

        if pred_voxels is None and gt_voxels is None:
            continue

        if occ_size_hw in [200, 400, 600]:
            occ_size = [occ_size_hw, occ_size_hw, 16]
        else:
            raise ValueError(f"Invalid voxel size: {voxels_shape}")

        point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
        voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
        voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
        voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
        voxel_size = [voxel_x, voxel_y, voxel_z]

        # show pred
        _save_path = os.path.join(save_folder, f'{id:05d}_pred_occ.gif')
        if pred_voxels is not None and not os.path.exists(_save_path):

            T = pred_voxels.shape[0]
            _save_folder = os.path.join(save_folder, f'{id:05d}')
            os.makedirs(_save_folder, exist_ok=True)
            view_params = None
            for t in range(T):
                _save_path_t = os.path.join(_save_folder, f'{t:02d}.png')
                view_params = draw_nusc_occupancy(
                    voxels=pred_voxels[t],
                    vox_origin=np.array(point_cloud_range[:3]),
                    voxel_size=np.array(voxel_size),
                    grid=np.array(occ_size),
                    save_path=_save_path_t,
                    view_params=(None if t > 0 else view_params),
                )
            all_filenames = list(sorted(os.listdir(_save_folder)))
            all_frames = [Image.open(
                os.path.join(_save_folder, file)).convert('RGB') for file in all_filenames]
            all_frames[0].save(
                _save_path,
                save_all=True,
                append_images=all_frames[1:],
                duration=100,
                loop=0,
            )

        # show gt
        _save_path = os.path.join(save_folder, f'{id:05d}_ori_occ.png')
        if gt_voxels is not None and not os.path.exists(_save_path):
            draw_nusc_occupancy(
                voxels=gt_voxels, 
                vox_origin=np.array(point_cloud_range[:3]),
                voxel_size=np.array(voxel_size),
                grid=np.array(occ_size),
                save_path=_save_path,
                view_params=view_params,
            )

        mlab.close(all=True)
