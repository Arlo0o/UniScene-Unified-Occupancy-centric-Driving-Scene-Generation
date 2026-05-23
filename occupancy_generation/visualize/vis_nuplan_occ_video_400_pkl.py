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
# occ_root = '/mnt/datasets/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_200_200_16'
occ_root = '/mnt/datasets/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_400_400_32'
#pkl_file = '/lpai/volumes/lmm-data-proc/hzhu/data/nuplan_pkls/nuplan_trainval_10hz_pkl/nuplan_10hz_trainval_part/nuplan_trainval_val_part1.pkl'
pkl_file = "/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_train_noted_with_ego_stationary_dynamic_others.pkl"
vis_root = "/mnt/volumes/ad-lmm-data-proc-bd-ga/hzhu/code/occ_gen/out/vis/nuplan_vis_dynamic_ego_occ_400"
vis_root_frame = "/mnt/volumes/ad-lmm-data-proc-bd-ga/hzhu/code/occ_gen/out/vis/nuplan_vis_dynamic_ego_occ_400/occ_frame"
os.makedirs(vis_root_frame, exist_ok=True)

with open(pkl_file, 'rb') as f:
    pkl_data = pickle.load(f)
# downsample_rate = 2

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


def draw(voxels, voxel_size=0.2, vis_root=None, idx=0, figure=None, mlab=None):
    if figure is None:
        figure = mlab.figure(size=(W, H), bgcolor=(1, 1, 1))
    scene = figure.scene

    camera = mlab.gcf().scene.camera
    camera.position = [0, 0, 10]
    camera.focal_point = [0, 0, 0]
    camera.view_up = [1, 0, 0]
    camera.compute_view_plane_normal()

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

def render_one_scene(scene_id, sample_tokens, voxel_size, voxel_origin, grid):
    from mayavi import mlab
    print(f'Start render scene{scene_id}')

    os.makedirs(vis_root, exist_ok=True)
    writer = imageio.get_writer(f'{vis_root}/{scene_id}.mp4', fps=10)
    figure = mlab.figure(size=(W, H), bgcolor=(1, 1, 1))
    for i, sample_token in enumerate(sample_tokens):
        occ_path = os.path.join(occ_root, sample_token+'.npy')
        print(occ_path)
        if not os.path.exists(occ_path):
            print(f'scene {scene_id} is not completed')
            break
        voxels_ = np.load(occ_path)

        grid_coords = get_grid_coords(
            [voxels_.shape[0], voxels_.shape[1], voxels_.shape[2]], voxel_size
        ) + np.array(voxel_origin, dtype=np.float32).reshape([1, 3])
        grid_coords = np.vstack([grid_coords.T, voxels_.reshape(-1)]).T
        
        grid_coords = grid_coords[
            (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 20)
        ]
        print(f"==> {grid_coords.shape}")

        mlab.clf()
        draw(grid_coords, voxel_size=1, figure=figure, mlab=mlab, vis_root=vis_root, idx=i)

        #mlab.savefig("test_output.png")

        frame = mlab.screenshot(mode='rgb', antialiased=True)
        writer.append_data(frame)
        
        print(f"{scene_id} 渲染{i}/{len(sample_tokens)}")
    mlab.close(all=True)
    writer.close()

if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
    occ_size = [400, 400, 32]
    voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
    voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
    voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
    voxel_size = [voxel_x, voxel_y, voxel_z]

    scene_names_orig = pkl_data['scene_tokens']
    scene_names_static = pkl_data['scene_tokens_with_stationary_segments']
    # 1. 先把所有要去除的元素收集到一个 set 里
    to_remove = set()
    for sublist in scene_names_static:
        to_remove.update(sublist)

    # 2. 对 scene_tokens 里的每个子列表，去除这些元素
    scene_tokens_without_stationary_elements = [
        [item for item in sublist if item not in to_remove]
        for sublist in scene_names_orig
    ]

    args = []
    for scene_id, sample_tokens in enumerate(scene_tokens_without_stationary_elements):
        # if scene_id == 1:
        args.append((
        scene_id,
        sample_tokens,
        voxel_size,
        np.array(point_cloud_range[:3]),
        np.array(occ_size)
        ))
        # print(scene_id)
        # render_one_scene(scene_id, sample_tokens, 
        # voxel_size, 
        # voxel_origin=np.array(point_cloud_range[:3]),
        # grid=np.array(occ_size),
        # )

    with multiprocessing.Pool(processes=12) as pool:
        # args = list(zip(list(range(len(infos['scene_tokens']))), infos['scene_tokens']))
        pool.starmap(render_one_scene, args)


    display.stop()