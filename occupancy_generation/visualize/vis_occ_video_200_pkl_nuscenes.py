import os
#os.environ['QT_DEBUG_PLUGINS'] = "1"
import pickle
import multiprocessing
import numpy as np
from pyvirtualdisplay import Display
H, W = 1080, 1920
display = Display(visible=False, size=(W, H))
display.start()
import imageio

# occ_root = '/lpai/dataset/nuplan-occ/1-1-01/GT_occ_fast_val/dense_voxels_with_semantic'
occ_root = '/data/zhuhu/3DVision_datasets/Occ/nuscenes/gts'
#pkl_file = '/lpai/volumes/lmm-data-proc/hzhu/data/nuplan_pkls/nuplan_trainval_10hz_pkl/nuplan_10hz_trainval_part/nuplan_trainval_val_part1.pkl'
pkl_file = "/data/zhuhu/3DVision_datasets/Occ/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl"
vis_root = "outputs/temp_concate_occ_vis"
vis_root_frame = "outputs/temp_concate_occ_vis/occ_frame"
os.makedirs(vis_root_frame, exist_ok=True)

with open(pkl_file, 'rb') as f:
    pkl_data = pickle.load(f)
    
colors = np.array(
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
			[255,  99,  71, 255],
			[  0, 191, 255, 255]
		]
	).astype(np.uint8)

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


def custom_colormap(plt_plot, colormap=colors):
    ori_colormap = plt_plot.module_manager.scalar_lut_manager.lut.table.to_array()
    for key, value in colormap.items():
        ori_colormap[key, :3] = value
    plt_plot.module_manager.scalar_lut_manager.lut.table = ori_colormap.astype(np.uint8)[:17]


def draw(voxels, voxel_size=0.2, vis_root=None, idx=0, figure=None, mlab=None):
    if figure is None:
        figure = mlab.figure(size=(W, H), bgcolor=(1, 1, 1))
    scene = figure.scene

    camera = mlab.gcf().scene.camera
    camera.position = [0, 0, 10]
    camera.focal_point = [0, 0, 0]
    camera.view_up = [1, 0, 0]
    camera.compute_view_plane_normal()
    print(voxel_size)

    plt_plot = mlab.points3d(
        voxels[:, 0],
        voxels[:, 1],
        voxels[:, 2],
        voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )
 
    scene.render()


    plt_plot.module_manager.scalar_lut_manager.lut.table = colors
    plt_plot.glyph.scale_mode = "scale_by_vector"

    #mlab.show()


    mlab.draw()
    vis_path = os.path.join(vis_root, 'occ_frame/{:0>4d}_bev.png'.format(idx))
    mlab.savefig(filename=vis_path)


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

def render_one_scene(scene_name, scene_data, voxel_size, voxel_origin, grid):
    from mayavi import mlab
    print(f'Start render scene{scene_name}')

    os.makedirs(vis_root, exist_ok=True)
    writer = imageio.get_writer(f'{vis_root}/{scene_name}.mp4', fps=10)
    figure = mlab.figure(size=(W, H), bgcolor=(1, 1, 1))
    for i, sample in enumerate(scene_data):
        occ_path = os.path.join(occ_root, f"{scene_name}/{sample['token']}/labels.npz")
        print(occ_path)
        if not os.path.exists(occ_path):
            print(f'scene {scene_name} is not completed')
            assert "scene is not completed"
        voxels_ = np.load(occ_path)
        voxels_ = voxels_['semantics']

        grid_coords = get_grid_coords(
            [voxels_.shape[0], voxels_.shape[1], voxels_.shape[2]], voxel_size
        ) + np.array(voxel_origin, dtype=np.float32).reshape([1, 3])
        grid_coords = np.vstack([grid_coords.T, voxels_.reshape(-1)]).T
        
        grid_coords[grid_coords[:, 3] == 17, 3] = 20
        grid_coords = grid_coords[
            (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 20)
        ]
        print(f"==> {grid_coords.shape}")

        mlab.clf()
        draw(grid_coords, voxel_size=0.5, figure=figure, mlab=mlab, vis_root=vis_root, idx=i)

        #mlab.savefig("test_output.png")

        frame = mlab.screenshot(mode='rgb', antialiased=True)
        writer.append_data(frame)
        
        print(f"{scene_name} 渲染{i}/{len(scene_data)}")
    mlab.close(all=True)
    writer.close()

if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
    occ_size = [200, 200, 16]
    voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
    voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
    voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
    voxel_size = [voxel_x, voxel_y, voxel_z]

    scene_names = pkl_data['infos']

    args = []
    for scene_name, scene_data in scene_names.items():
        # if scene_id == 1:
        args.append((
        scene_name,
        scene_data,
        voxel_size,
        np.array(point_cloud_range[:3]),
        np.array(occ_size))
        )

        # print(scene_name)
        # render_one_scene(scene_name, scene_data, 
        # voxel_size, 
        # voxel_origin=np.array(point_cloud_range[:3]),
        # grid=np.array(occ_size),
        # )

    with multiprocessing.Pool(processes=12) as pool:
        # args = list(zip(list(range(len(scene_names))), scene_names))
        pool.starmap(render_one_scene, args)


    display.stop()