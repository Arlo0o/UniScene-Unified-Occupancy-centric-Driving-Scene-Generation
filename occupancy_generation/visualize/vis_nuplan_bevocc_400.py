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
import cv2

occ_dir = 'data_preprocess/be_cc_sample/200/occ'
bev_dir = 'data_preprocess/be_cc_sample/200/bev'
occ_with_bev = True  # 是否使用 BEV 替换占据网格


classname_to_color = {  # RGB.
    0: (255, 255, 255),  # Black. noise
    1: (0,175,0),  # other-ground
    2: (255, 158, 0),  # vehicle
    3: (220,20,60),  # bicycle
    4: (0,0,230),  # pedestrian 
    5: (47,79,79),  # traffic-cone
    6: (112,128,144),  # barrier
    7: (255, 200, 0),  # construction-zones
    8: (222,184,13 ),  # generic-object
    12: (0,207,191), # road
    14: (150 , 240, 80 ),  # road-line
}

# ===== BEV visualization palettes (from your notebook) =====
MAP_PALETTE_NUPLAN = {
    "intersections": (166, 206, 227),
    "generic_drivable_areas": (166, 206, 227),
    "walkways": (227, 26, 28),
    "carpark_areas": (166, 206, 227),
    "crosswalks": (166, 206, 227),
    "lane_group_connectors": (166, 206, 227),
    "lane_groups_polygons": (166, 206, 227),
    "road_segments": (166, 206, 227),
}

OBJECT_PALETTE_NUPLAN = {
    "vehicle": (255, 158, 0),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
    "barrier": (112, 128, 144),
    "czone_sign": (255, 99, 71),
    "generic_object": (233, 150, 70),
}

LANE_PALETTE_NUPLAN = {
    "lane": (255, 30, 30),
}

def render_bev_canvas(masks: np.ndarray) -> np.ndarray:
    # masks shape expected: (C, H, W). Values 0/1
    if masks.ndim == 2:
        masks = masks[None, ...]
    canvas = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
    canvas[:] = (240, 240, 240)
    classes = list(MAP_PALETTE_NUPLAN.keys()) + list(OBJECT_PALETTE_NUPLAN.keys()) + list(LANE_PALETTE_NUPLAN.keys())
    max_c = min(len(classes), masks.shape[0])
    for k in range(max_c):
        name = classes[k]
        if name in OBJECT_PALETTE_NUPLAN:
            color = OBJECT_PALETTE_NUPLAN[name]
        elif name in MAP_PALETTE_NUPLAN:
            color = MAP_PALETTE_NUPLAN[name]
        elif name in LANE_PALETTE_NUPLAN:
            color = LANE_PALETTE_NUPLAN[name]
        else:
            continue
        maskk = masks[k] == 1
        canvas[maskk] = color
    return canvas
def replace_occ_grid_with_bev_nuplan(input_occ, bevlayout, driva_area_idx=1, bev_replace_idx=[1,15, 17],
                                     occ_replace_new_idx=[12, 14, 15]):
                                    #  需要更换的: 
    # self.classes= ['drivable_area','ped_crossing','walkway','stop_line','carpark_area','road_divider','lane_divider','road_block']
    # nuplan
    # self.classes= ['intersections','generic_drivable_areas','walkways','carpark_areas','crosswalks','lane_group_connectors','lane_groups_polygons','road_segments']
    # 需要把 walkways 换成另外一种颜色,10
    # 需要把 drivable_area 换成另外一种颜色,11
    # lane_divider 换成另外一种颜色,12
    # stop_line del
    # road_block 13
    # occ road [11] drivable area

    # default ped_crossing->18; stop_line->19 (del); roal_divider->20; lane_divider->21
    # default shape: input_occ: [200,200,16]; bevlayout: [18,200,200]

    roal_divider_mask = bevlayout[15, :, :].astype(np.uint8)
    lane_divider_mask = bevlayout[17, :, :].astype(np.uint8)

    roal_divider_mask = cv2.dilate(roal_divider_mask, np.ones((1, 1), np.uint8))
    lane_divider_mask = cv2.dilate(lane_divider_mask, np.ones((1, 1), np.uint8))

    # roal_divider_mask = cv2.dilate(
    #     roal_divider_mask, np.ones((2, 2), np.uint8))
    # lane_divider_mask = cv2.dilate(
    #     lane_divider_mask, np.ones((2, 2), np.uint8))

    bevlayout[15, :, :] = roal_divider_mask.astype(bool)
    bevlayout[17, :, :] = lane_divider_mask.astype(bool)

    n = len(bev_replace_idx)
    x_max, y_max = input_occ.shape[0], input_occ.shape[1]
    output_occ = input_occ.copy()  # numpy copy() ; tensor clone()
    bev_replace_mask = []
    for i in range(n):
        bev_replace_mask.append(bevlayout[bev_replace_idx[i]] == 1)

    for x in range(x_max):
        for y in range(y_max):
            for i in range(n):
                if bev_replace_mask[i][x, y]:
                    occupancy_data = input_occ[x, y, :]

                    if driva_area_idx in occupancy_data:
                        max_11_index = np.where(
                            occupancy_data == driva_area_idx)
                        output_occ[x, y, max_11_index] = occ_replace_new_idx[i]
    return output_occ

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
    draw(fov_voxels, voxel_size=voxel_size, vis_root=save_folder, idx=0)


def draw(voxels, voxel_size=0.2, vis_root=None, idx=0):
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

    # save_file = f"/mnt/volumes/ad-cl-rl-train-vol-ga/hzhu/hzhu/code/occ_gen/result_vis/vae_bev"
    save_file = f"{occ_dir}_vis"
    os.makedirs(save_file, exist_ok=True)
    filename = str(save_file) + f'/{vis_root[0]}_{idx}_bev.png'
    mlab.draw()
    mlab.savefig(filename = filename)
    mlab.close()
    mlab.clf()
    print(filename)
    # mlab.show()

if __name__ == '__main__':
    

    # folder_path = 'outputs/train_3dVAE_filled'  # 替换为你的文件夹路径 finetune filled vae
    folder_path = f'{occ_dir}'  # 替换为你的文件夹路径
    files_to_rename = []  # 存储符合条件的文件名列表
    # 遍历文件夹及其子文件夹中所有文件
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.npy'):
                files_to_rename.append(os.path.join(root, filename))
    # 按文件名排序
    files_to_rename.sort()
    # breakpoint()

    for filepath in files_to_rename:
        print(filepath)
        pred_voxels =  np.load(filepath)
        vis_voxels = None
        print(pred_voxels.shape)
        if len(pred_voxels.shape) == 4:
            vis_voxels = pred_voxels[0]
        elif len(pred_voxels.shape) == 5:
            vis_voxels = pred_voxels[0][0]
        elif len(pred_voxels.shape) == 3:
            vis_voxels = pred_voxels
        point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
        
        if vis_voxels.shape[0] == 200:
            occ_size = [200, 200, 16]
        elif vis_voxels.shape[0] == 400:
            occ_size = [400, 400, 16]
        elif vis_voxels.shape[0] == 600:
            occ_size = [600, 600, 16]
        else:
            continue
            # raise ValueError(f"Invalid voxel size: {vis_voxels.shape[0]}")
        
        if occ_with_bev:
            print("replace occ with bev")
            layer_to_merge=[0,1,3,4,5,6,7]
            # 得到 token (文件名，不包含扩展名和路径)
            token = os.path.splitext(os.path.basename(filepath))[0]
            bev_file_name = os.path.join(bev_dir, f"{token}.npz")
            print(bev_file_name)
            bev_layout = np.load(bev_file_name)['gt_bev_masks']
            bev_layout[1,:,:] = np.any(bev_layout[layer_to_merge, :, :], axis=0).astype(int)
            
            # Use a copy for OCC replacement so dilation inside doesn't affect BEV 2D rendering thickness
            vis_voxels = replace_occ_grid_with_bev_nuplan(
                vis_voxels, bev_layout.copy(), bev_replace_idx=[1,15], occ_replace_new_idx=[12,14]
            )

            # Also save 2D BEV visualization
            bev_canvas = render_bev_canvas(bev_layout)
            bev_vis_dir = f"{bev_dir}_vis"
            os.makedirs(bev_vis_dir, exist_ok=True)
            bev_png_path = os.path.join(bev_vis_dir, f"{token}_bev2d.png")
            imageio.imwrite(bev_png_path, bev_canvas)

        voxel_size = [0.5, 0.5, 0.5]
        counts, elements = np.unique(vis_voxels, return_counts=True)
        print(f"Unique elements: {elements}, Counts: {counts}")

        print(vis_voxels.shape)
        draw_nusc_occupancy(
                voxels=vis_voxels, 
                vox_origin=np.array(point_cloud_range[:3]),
                voxel_size=np.array(voxel_size),
                grid=np.array(occ_size),
                save_folder = filepath.split("/")[-1:],
        )