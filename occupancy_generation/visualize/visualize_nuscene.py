import pickle, argparse, os
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.map_visualizer import visualize_map

from pyvirtualdisplay import Display
display = Display(visible=False, size=(1280, 1024))
display.start()

from mayavi import mlab
mlab.options.offscreen = True


import pdb

camera_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
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

def draw_nusc_occupancy(
	voxels,
	vox_origin,
	voxel_size=0.2,
	grid=None,
	pred_lidarseg=None,
	target_lidarseg=None,
	save_folder=None,
	cat_save_file=None,
	cam_positions=None,
	focal_positions=None,
):
	w, h, z = voxels.shape
	grid = grid.astype(np.int32)
	
	# Compute the voxels coordinates
	grid_coords = get_grid_coords(
		[voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
	) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
	grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
	
	grid_coords[grid_coords[:, 3] == 17, 3] = 20
	car_vox_range = np.array([
		[w//2 - 2 - 4, w//2 - 2 + 4],
		[h//2 - 2 - 4, h//2 - 2 + 4],
		[z//2 - 2 - 3, z//2 - 2 + 3]
	], dtype=np.int32)
	
	''' draw the colorful ego-vehicle '''
	car_x = np.arange(car_vox_range[0, 0], car_vox_range[0, 1])
	car_y = np.arange(car_vox_range[1, 0], car_vox_range[1, 1])
	car_z = np.arange(car_vox_range[2, 0], car_vox_range[2, 1])
	car_xx, car_yy, car_zz = np.meshgrid(car_x, car_y, car_z)
	car_label = np.zeros([8, 8, 6], dtype=np.int32)
	car_label[:3, :, :2] = 17
	car_label[3:6, :, :2] = 18
	car_label[6:, :, :2] = 19
	car_label[:3, :, 2:4] = 18
	car_label[3:6, :, 2:4] = 19
	car_label[6:, :, 2:4] = 17
	car_label[:3, :, 4:] = 19
	car_label[3:6, :, 4:] = 17
	car_label[6:, :, 4:] = 18
	car_grid = np.array([car_xx.flatten(), car_yy.flatten(), car_zz.flatten()]).T
	car_indexes = car_grid[:, 0] * h * z + car_grid[:, 1] * z + car_grid[:, 2]
	grid_coords[car_indexes, 3] = car_label.flatten()

	# Get the voxels inside FOV
	fov_grid_coords = grid_coords

	# Remove empty and unknown voxels
	fov_voxels = fov_grid_coords[
		(fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
	]
	# print(len(fov_voxels))
	
	figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
	# Draw occupied inside FOV voxels
	voxel_size = sum(voxel_size) / 3
	plt_plot_fov = mlab.points3d(
		fov_voxels[:, 1],
		fov_voxels[:, 0],
		fov_voxels[:, 2],
		fov_voxels[:, 3],
		colormap="viridis",
		scale_factor=voxel_size - 0.05 * voxel_size,
		mode="cube",
		opacity=1.0,
		vmin=1,
		vmax=19, # 16
	)
	
	plt_plot_fov.glyph.scale_mode = "scale_by_vector"
	plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
	scene = figure.scene

	os.makedirs(save_folder, exist_ok=True)
	visualize_keys = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
			'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'DRIVING_VIEW', 'BIRD_EYE_VIEW']
	
	for i in range(8):
		# from six cameras
		if i < 6:
			scene.camera.position = cam_positions[i] - np.array([0.7, 1.3, 0.])
			scene.camera.focal_point = focal_positions[i] - np.array([0.7, 1.3, 0.])
			scene.camera.view_angle = 35 if i != 3 else 60
			scene.camera.view_up = [0.0, 0.0, 1.0]
			scene.camera.clipping_range = [0.01, 300.]
			scene.camera.compute_view_plane_normal()
			scene.render()
		
		# bird-eye-view and facing front 
		elif i == 6:
			scene.camera.position = [  0.75131739, -35.08337438,  16.71378558]
			scene.camera.focal_point = [  0.75131739, -34.21734897,  16.21378558]
			scene.camera.view_angle = 40.0
			scene.camera.view_up = [0.0, 0.0, 1.0]
			scene.camera.clipping_range = [0.01, 300.]
			scene.camera.compute_view_plane_normal()
			scene.render()
		
		# bird-eye-view
		else:
			scene.camera.position = [ 0.75131739,  0.78265103, 93.21378558]
			scene.camera.focal_point = [ 0.75131739,  0.78265103, 92.21378558]
			scene.camera.view_angle = 40.0
			scene.camera.view_up = [0., 1., 0.]
			scene.camera.clipping_range = [0.01, 400.]
			scene.camera.compute_view_plane_normal()
			scene.render()

		save_file = os.path.join(save_folder, '{}.png'.format(visualize_keys[i]))
		mlab.savefig(save_file)
	
	mlab.close()
	
	# read rendered images, combine, and create the predictions
	cam_img_size = [480, 270]
	pred_img_size = [1920, 1080]
	spacing = 10

	cam_w, cam_h = cam_img_size
	pred_w, pred_h = pred_img_size
	result_w = cam_w * 6 + 5 * spacing
	result_h = cam_h * 2 + pred_h + 2 * spacing

	pred_imgs = []
	for cam_name in camera_names:
		pred_img_file = os.path.join(save_folder, '{}.png'.format(cam_name))
		pred_img = Image.open(pred_img_file).resize(cam_img_size, Image.BILINEAR)
		pred_imgs.append(pred_img)

	drive_view_occ = Image.open(os.path.join(save_folder, 'DRIVING_VIEW.png')).resize(pred_img_size, Image.BILINEAR)
	bev_occ = Image.open(os.path.join(save_folder, 'BIRD_EYE_VIEW.png')).resize(pred_img_size, Image.BILINEAR).crop([460, 0, 1460, 1080])

	# create the output image
	result = Image.new(pred_imgs[0].mode, (result_w, result_h), (0, 0, 0))

	result.paste(pred_imgs[0], box=(3*cam_w+3*spacing, 0))
	result.paste(pred_imgs[1], box=(4*cam_w+4*spacing, 0))
	result.paste(pred_imgs[2], box=(5*cam_w+5*spacing, 0))

	result.paste(pred_imgs[3], box=(3*cam_w+3*spacing, 1*cam_h+1*spacing))
	result.paste(pred_imgs[4], box=(4*cam_w+4*spacing, 1*cam_h+1*spacing))
	result.paste(pred_imgs[5], box=(5*cam_w+5*spacing, 1*cam_h+1*spacing))

	result.paste(drive_view_occ, box=(0, 2*cam_h+2*spacing))
	result.paste(bev_occ, box=(1*pred_w+1*spacing, 2*cam_h+2*spacing))

	result.save(cat_save_file)


def draw_nusc_occupancy_Bev_Front(
	voxels,
	vox_origin,
	voxel_size=0.2,
	grid=None,
	pred_lidarseg=None,
	target_lidarseg=None,
	save_folder=None,
	cat_save_file=None,
	cam_positions=None,
	focal_positions=None,
	bev_layout=None,
):
	w, h, z = voxels.shape
	grid = grid.astype(np.int32)
	
	# Compute the voxels coordinates
	grid_coords = get_grid_coords(
		[voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
	) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
	grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
	
	grid_coords[grid_coords[:, 3] == 17, 3] = 20
	car_vox_range = np.array([
		[w//2 - 2 - 4, w//2 - 2 + 4],
		[h//2 - 2 - 4, h//2 - 2 + 4],
		[z//2 - 2 - 3, z//2 - 2 + 3]
	], dtype=np.int32)
	
	''' draw the colorful ego-vehicle '''
	car_x = np.arange(car_vox_range[0, 0], car_vox_range[0, 1])
	car_y = np.arange(car_vox_range[1, 0], car_vox_range[1, 1])
	car_z = np.arange(car_vox_range[2, 0], car_vox_range[2, 1])
	car_xx, car_yy, car_zz = np.meshgrid(car_x, car_y, car_z)
	car_label = np.zeros([8, 8, 6], dtype=np.int32)
	car_label[:3, :, :2] = 17
	car_label[3:6, :, :2] = 18
	car_label[6:, :, :2] = 19
	car_label[:3, :, 2:4] = 18
	car_label[3:6, :, 2:4] = 19
	car_label[6:, :, 2:4] = 17
	car_label[:3, :, 4:] = 19
	car_label[3:6, :, 4:] = 17
	car_label[6:, :, 4:] = 18
	car_grid = np.array([car_xx.flatten(), car_yy.flatten(), car_zz.flatten()]).T
	car_indexes = car_grid[:, 0] * h * z + car_grid[:, 1] * z + car_grid[:, 2]
	grid_coords[car_indexes, 3] = car_label.flatten()

	# Get the voxels inside FOV
	fov_grid_coords = grid_coords

	# Remove empty and unknown voxels
	fov_voxels = fov_grid_coords[
		(fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
	]
	# print(len(fov_voxels))
	
	figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
	# Draw occupied inside FOV voxels
	voxel_size = sum(voxel_size) / 3
	plt_plot_fov = mlab.points3d(
		fov_voxels[:, 1],
		fov_voxels[:, 0],
		fov_voxels[:, 2],
		fov_voxels[:, 3],
		colormap="viridis",
		scale_factor=voxel_size - 0.05 * voxel_size,
		mode="cube",
		opacity=1.0,
		vmin=1,
		vmax=19, # 16
	)
	
	plt_plot_fov.glyph.scale_mode = "scale_by_vector"
	plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
	scene = figure.scene

	os.makedirs(save_folder, exist_ok=True)
	# visualize_keys = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
	# 		'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'DRIVING_VIEW', 'BIRD_EYE_VIEW']
	visualize_keys = ['DRIVING_VIEW', 'BIRD_EYE_VIEW']
 
	for i in range(2):
		# from six cameras
		# if i < 6:
		# 	scene.camera.position = cam_positions[i] - np.array([0.7, 1.3, 0.])
		# 	scene.camera.focal_point = focal_positions[i] - np.array([0.7, 1.3, 0.])
		# 	scene.camera.view_angle = 35 if i != 3 else 60
		# 	scene.camera.view_up = [0.0, 0.0, 1.0]
		# 	scene.camera.clipping_range = [0.01, 300.]
		# 	scene.camera.compute_view_plane_normal()
		# 	scene.render()
		
		# bird-eye-view and facing front 
		if i == 0:
			scene.camera.position = [  0.75131739, -35.08337438,  16.71378558]
			scene.camera.focal_point = [  0.75131739, -34.21734897,  16.21378558]
			scene.camera.view_angle = 40.0
			scene.camera.view_up = [0.0, 0.0, 1.0]
			scene.camera.clipping_range = [0.01, 300.]
			scene.camera.compute_view_plane_normal()
			scene.render()
		
		# bird-eye-view
		else:
			scene.camera.position = [ 0.75131739,  0.78265103, 93.21378558]
			scene.camera.focal_point = [ 0.75131739,  0.78265103, 92.21378558]
			scene.camera.view_angle = 40.0
			scene.camera.view_up = [0., 1., 0.]
			scene.camera.clipping_range = [0.01, 400.]
			scene.camera.compute_view_plane_normal()
			scene.render()

		save_file = os.path.join(save_folder, '{}.png'.format(visualize_keys[i]))
		mlab.savefig(save_file)
	
	mlab.close()
	
	map_img_np=visualize_map(bev_layout)
	map_img=Image.fromarray(map_img_np)
	map_img.save(os.path.join(save_folder, "BEV_Layout.png"))

	# read rendered images, combine, and create the predictions
	cam_img_size = [480, 270]
	pred_img_size = [1920, 1080]
	spacing = 10

	cam_w, cam_h = cam_img_size
	pred_w, pred_h = pred_img_size
	result_w = cam_w * 6 + 5 * spacing + 1000 + spacing
	result_h = cam_h * 2 + pred_h + 2 * spacing

	# pred_imgs = []
	# for cam_name in camera_names:
	# 	pred_img_file = os.path.join(save_folder, '{}.png'.format(cam_name))
	# 	pred_img = Image.open(pred_img_file).resize(cam_img_size, Image.BILINEAR)
	# 	pred_imgs.append(pred_img)

	drive_view_occ = Image.open(os.path.join(save_folder, 'DRIVING_VIEW.png')).resize(pred_img_size, Image.BILINEAR)
	bev_occ = Image.open(os.path.join(save_folder, 'BIRD_EYE_VIEW.png')).resize(pred_img_size, Image.BILINEAR).crop([460, 0, 1460, 1080])
	bev_layout = Image.open(os.path.join(save_folder, "BEV_Layout.png")).resize([1000,1080],Image.BILINEAR)
	# create the output image
	# result = Image.new(pred_imgs[0].mode, (result_w, result_h), (0, 0, 0))

	# result.paste(pred_imgs[0], box=(3*cam_w+3*spacing, 0))
	# result.paste(pred_imgs[1], box=(4*cam_w+4*spacing, 0))
	# result.paste(pred_imgs[2], box=(5*cam_w+5*spacing, 0))

	# result.paste(pred_imgs[3], box=(3*cam_w+3*spacing, 1*cam_h+1*spacing))
	# result.paste(pred_imgs[4], box=(4*cam_w+4*spacing, 1*cam_h+1*spacing))
	# result.paste(pred_imgs[5], box=(5*cam_w+5*spacing, 1*cam_h+1*spacing))

	result = Image.new(drive_view_occ.mode, (result_w, result_h), (0, 0, 0))
	result.paste(drive_view_occ, box=(0, 2*cam_h+2*spacing))
	result.paste(bev_occ, box=(1*pred_w+1*spacing, 2*cam_h+2*spacing))
	result.paste(bev_layout, box=(1*pred_w+2*spacing+1000, 2*cam_h+2*spacing))
	return result
	# result.save(cat_save_file)
 
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--pred_dir', default='./')
	parser.add_argument('--save_path', default='outputs' )
	args = parser.parse_args()
	
	point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
	occ_size = [256, 256, 32]
	voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
	voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
	voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
	voxel_size = [voxel_x, voxel_y, voxel_z]
	
	# noqa
	constant_f = 0.0055
	sample_files = os.listdir(args.pred_dir)
	save_path = args.save_path

	for index, sample_file in tqdm(enumerate(sample_files), total=len(sample_files)):
		if not sample_file[-3:] == 'pkl':
			continue

		sample_token = sample_file.split('.')[0]
		sample_file = os.path.join(args.pred_dir, sample_file)

		save_folder = os.path.join(save_path, '{}_assets'.format(sample_token))
		cat_save_file = os.path.join(save_path, '{}_cat_vis.png'.format(sample_token))

		if os.path.exists(cat_save_file):
			continue
		
		with open(sample_file, 'rb') as f:
			sample_data = pickle.load(f)
		
		pred_voxels = sample_data['pred_voxels']
		cam2lidar = sample_data['cam2lidar']
		cam_positions = cam2lidar @ np.array([0., 0., 0., 1.])
		cam_positions = cam_positions[:, :3]
		focal_positions = cam2lidar @ np.array( [0., 0., constant_f, 1.] )   
		focal_positions = focal_positions[:, :3]

		draw_nusc_occupancy(
			voxels=pred_voxels, 
			vox_origin=np.array(point_cloud_range[:3]),
			voxel_size=np.array(voxel_size),
			grid=np.array(occ_size),
			pred_lidarseg=None,
			target_lidarseg=None,
			save_folder=save_folder,
			cat_save_file=cat_save_file,
			cam_positions=cam_positions,
			focal_positions=focal_positions,
		)