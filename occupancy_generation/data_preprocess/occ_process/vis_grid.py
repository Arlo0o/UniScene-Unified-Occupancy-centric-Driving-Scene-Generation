# from pyvirtualdisplay import Display
# display = Display(visible=False, size=(1280, 1024))
# display.start()
'''
for shape like (1, 512, 512, 40)
'''
from mayavi import mlab
mlab.options.offscreen = False

from pathlib import Path
import numpy as np
# from nuscenes.nuscenes import NuScenes
import os
import glob
import imageio

# classname_to_color = np.array([
#     [255, 255, 255],  # Black. noise
#     [112, 128, 144],  # Slategrey barrier
#     [220, 20, 60],  # Crimson bicycle
#     [255, 127, 80],  # Orangered bus
#     [255, 158, 0],  # Orange car
#     [233, 150, 70],  # Darksalmon construction
#     [255, 61, 99],  # Red motorcycle
#     [0, 0, 230],  # Blue pedestrian
#     [47, 79, 79],  # Darkslategrey trafficcone
#     [255, 140, 0],  # Darkorange trailer
#     [255, 99, 71],  # Tomato truck
#     [0, 207, 191],  # nuTonomy green driveable_surface
#     [175, 0, 75],  # flat other
#     [75, 0, 75],  # sidewalk
#     [112, 180, 60],  # terrain
#     [222, 184, 135],  # Burlywood mannade
#     [0, 175, 0]   # Green vegetation
# ]).astype(np.uint8)

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

nuplan_occ_colors_map = np.array(
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

nuscenes_colors = np.array(
		[
			[255, 120,  50, 255],       # barrier              orange
			[255, 192, 203, 255],       # bicycle              pink
			[255, 255,   0, 255],       # bus                  yellow
			[  0, 150, 245, 255],       # car                  blue
			[  0, 255, 255, 255],       # construction_vehicle cyan
			[255, 127,   0, 255],       # motorcycle           dark orange
			[255,   0,   0, 255],       # pedestrian           red
			[255, 240, 150, 255],       # traffic_cone         light yellow
			[135,  60,   0, 255],       # trailer              brown
			[160,  32, 240, 255],       # truck                purple                
			[255,   0, 255, 255],       # driveable_surface    dark pink
			# [175,   0,  75, 255],       # other_flat           dark red
			[139, 137, 137, 255],
			[ 75,   0,  75, 255],       # sidewalk             dard purple
			[150, 240,  80, 255],       # terrain              light green          
			[230, 230, 250, 255],       # manmade              white
			[  0, 175,   0, 255],       # vegetation           green
			[  0, 255, 127, 255],       # ego car              dark cyan
			[255,  99,  71, 255],       # x       tomato
			[  0, 191, 255, 255]        # x        deep sky blue
		]
	).astype(np.uint8)



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
    save_folder=None,
    ):
    w, h, z = voxels.shape
    # grid = grid.astype(np.int32)
    for i in range(20):
        if np.any(voxels == i):
            print(f"label {i} in voxels {np.any(voxels == i)}")

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # grid_coords[grid_coords[:, 3] == 17, 3] = 20
    car_vox_range = np.array([
        [w//2 - 2 - 4, w//2 - 2 + 4],
        [h//2 - 2 - 4, h//2 - 2 + 4],
        [z//2 - 2 - 3, z//2 - 2 + 3]
    ], dtype=np.int32)

    # ''' draw the colorful ego-vehicle '''
    # car_x = np.arange(car_vox_range[0, 0], car_vox_range[0, 1])
    # car_y = np.arange(car_vox_range[1, 0], car_vox_range[1, 1])
    # car_z = np.arange(car_vox_range[2, 0], car_vox_range[2, 1])
    # car_xx, car_yy, car_zz = np.meshgrid(car_x, car_y, car_z)
    # car_label = np.zeros([8, 8, 6], dtype=np.int32)
    # car_label[:3, :, :2] = 17
    # car_label[3:6, :, :2] = 18
    # car_label[6:, :, :2] = 19
    # car_label[:3, :, 2:4] = 18
    # car_label[3:6, :, 2:4] = 19
    # car_label[6:, :, 2:4] = 17
    # car_label[:3, :, 4:] = 19
    # car_label[3:6, :, 4:] = 17
    # car_label[6:, :, 4:] = 18
    # car_grid = np.array([car_xx.flatten(), car_yy.flatten(), car_zz.flatten()]).T
    # car_indexes = car_grid[:, 0] * h * z + car_grid[:, 1] * z + car_grid[:, 2]
    # grid_coords[car_indexes, 3] = car_label.flatten()

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords
    for i in range(20):
        if np.any(fov_grid_coords[:, 3] == i):
            print(f"label {i} in fov_grid_coords {np.any(fov_grid_coords[:, 3] == i)} sum {np.sum(fov_grid_coords[:, 3] == i)}")

    # Remove empty and unknown voxels
    # fov_voxels = fov_grid_coords
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
        # (fov_grid_coords[:, 3] == 0) | (fov_grid_coords[:, 3] > 1) | (fov_grid_coords[:, 3] < 19)
    ]
    print(fov_voxels.shape)

    # np.save(str(Path(*save_folder) ) + '_fov_voxels.npy', fov_voxels)

    for i in range(20):
        if np.any(fov_voxels[:, 3] == i):
            print(f"label {i} in fov_voxels {np.any(fov_voxels[:, 3] == i)} sum {np.sum(fov_voxels[:, 3] == i)}")
    draw(fov_voxels, voxel_size=voxel_size, vis_root=save_folder, idx=0)

 

#     # figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
#     # # Draw occupied inside FOV voxels
#     # voxel_size = sum(voxel_size) / 3
#     # plt_plot_fov = mlab.points3d(
#     #     fov_voxels[:, 0],
#     #     fov_voxels[:, 1],
#     #     fov_voxels[:, 2],
#     #     fov_voxels[:, 3],
#     #     colormap="viridis",
#     #     scale_factor=voxel_size - 0.05 * voxel_size,
#     #     mode="cube",
#     #     opacity=1.0,
#     #     vmin=1,
#     #     vmax=19, # 16
#     # )

#     # plt_plot_fov.glyph.scale_mode = "scale_by_vector"
#     # plt_plot_fov.module_manager.scalar_lut_manager.lut.table = nuscenes_colors
#     # scene = figure.scene

#     # os.makedirs(save_folder, exist_ok=True)
#     # visualize_keys = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
#     #         'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'DRIVING_VIEW', 'BIRD_EYE_VIEW']


# # ------------------------------------------- 

#     scene.camera.position = [263.1592864706488, 269.11134236991353, 279.160342685614]
#     scene.camera.focal_point = [28.800004959106445, 28.800004959106445, -1.0000001192092896]
#     scene.camera.view_angle = 30.0
#     scene.camera.view_up = [-0.45283420212088804, -0.4533930811436847, 0.7677082123831782]
#     scene.camera.clipping_range = [254.9347300096278, 667.7401497754007]
#     scene.camera.compute_view_plane_normal()
#     scene.render()
#     scene.isometric_view()

#     save_file = Path(*save_folder) 
#     print(save_file)
#     # import pdb;pdb.set_trace()
#     mlab.draw()
#     # vis_path = os.path.join(save_file, 'bev.png' )
#     mlab.savefig(filename = str(save_file) + '_bev.png' )
#     mlab.show()
    # mlab.close()

def draw(voxels, voxel_size=0.2, vis_root=None, idx=0):
    x = -voxels[:, 1]  # x position of point
    y = voxels[:, 0]  # y position of point
    z = voxels[:, 2]  # z position of point
    point_color = np.zeros(voxels.shape[0])

    for cls_index in range(16):
        class_point = voxels[:, 3] == cls_index
        point_color[class_point] = cls_index+1 

    figure = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    plt_plot = mlab.points3d(
        x,
        y,
        z,
        point_color,
        scale_factor=0.7,
        scale_mode='vector',
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=17,
    )
    # plt_plot.glyph.scale_mode = "scale_by_vector"

    custom_colormap(plt_plot)

    view_type ='back_view'
    if view_type =='back_view':
        scene = figure
        scene.scene.z_plus_view()
        scene.scene.camera.position = [-1.1612566981665453, -63.271696093007456, 33.06645769267362]
        scene.scene.camera.focal_point = [-0.0828344205684326, -0.029545161654287222, -1.078433202901462]
        scene.scene.camera.view_angle = 45.0
        scene.scene.camera.view_up = [-0.011200801911309498, 0.4752037522484654, 0.879804487306994]
        scene.scene.camera.clipping_range = [0.18978054185107493, 189.78054185107493]
        scene.scene.camera.compute_view_plane_normal()
        scene.scene.render()

    save_file = Path(*vis_root) 
    print(save_file)
    # import pdb;pdb.set_trace()
    mlab.draw()
    # vis_path = os.path.join(save_file, 'bev.png' )
    mlab.savefig(filename = str(save_file) + '_bev.png' )
    mlab.show()
    mlab.close()
 
if __name__ == '__main__':
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    # point_cloud_range = [-25.600000381469727, -25.600000381469727, -2.0, 25.600000381469727, 25.600000381469727, 1.0]

    occ_size = [256, 256, 32]
    voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
    voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
    voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
    voxel_size = [voxel_x, voxel_y, voxel_z]
    
    # folder_path = '/Volumes/hu500/eval_dit_12hz'  # 替换为你的文件夹路径
    folder_path = '/lpai/volumes/ad-lmm-data-proc-bd-ga/hzhu/code/occ_process/occ_quan_sample/nuscene_quantized_200_200_16'  # 替换为你的文件夹路径
    # folder_path = '/Users/hu/code/occ-vis/occ_files/carla_occ/save_occ'  # sample data 
    files_to_rename = []  # 存储符合条件的文件名列表
    # 遍历文件夹及其子文件夹中所有文件
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.npy'):
                files_to_rename.append(os.path.join(root, filename))
    # 按文件名排序
    files_to_rename.sort()
    # print(files_to_rename)
    
    for filepath in files_to_rename:
        print(filepath)
        pred_voxels =  np.load(filepath)
        # load bin
        # pred_voxels = np.fromfile(filepath, dtype=np.uint8)
        # pred_voxels = pred_voxels.reshape(1, 192, 192, 16)
        vis_voxels = None
        print(pred_voxels.shape)
        if len(pred_voxels.shape) == 4:
            vis_voxels = pred_voxels[0]
        elif len(pred_voxels.shape) == 5:
            vis_voxels = pred_voxels[0][0]
        elif len(pred_voxels.shape) == 3:
            vis_voxels = pred_voxels

        print(vis_voxels.shape)
        draw_nusc_occupancy(
                voxels=vis_voxels, 
                vox_origin=np.array(point_cloud_range[:3]),
                voxel_size=np.array(voxel_size),
                grid=np.array(occ_size),
                save_folder = filepath.split(".")[:-1],
        )