import os
import numpy as np
import pickle
from pyvirtualdisplay import Display
from natsort import natsorted
display = Display(visible=0, size=(1920, 1080))
display.start()
import mayavi
from mayavi import mlab
from bev_util import replace_occ_grid_with_bev_nuplan

MAP_PALETTE_NUPLAN = {
    # "lane_polygons": (106, 61, 154),         # 深紫色（对应 lane_divider）
    "intersections": (166, 206, 227),       # 柔和粉红色（对应 ped_crossing）
    "generic_drivable_areas": (166, 206, 227),  # 浅蓝色（对应 drivable_area）
    "walkways": (227, 26, 28),              # 鲜艳红色（对应 walkway）
    # "walkways": (166, 206, 227),              # 鲜艳红色（对应 walkway）
    "carpark_areas": (166, 206, 227),         # 橙色（对应 carpark_area）
    "crosswalks": (166, 206, 227),          # 浅橙色（对应 stop_line）
    "lane_group_connectors": (166, 206, 227),  # 柔和浅绿（对应 road_block）
    "lane_groups_polygons": (166, 206, 227),  # 绿色（对应 lane）
    "road_segments": (166, 206, 227),        # 深蓝色（对应 road_segment）
    # "stop_polygons": (202, 178, 214),        # 浅紫色（对应 road_divider）
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
    "lane": (255, 30, 30),  # 让红色更深
    # "stop_line": (0, 230, 0),  # 让绿色更接近荧光绿
    # "road_block": (0, 0, 220),  # 让蓝色稍微降低亮度
}

def visualize_object_and_map(
    masks: np.ndarray,
    *,
    classes: list,
    background=(240, 240, 240),
) -> None:
    """
    可视化 8 通道的掩码数据，并使用伪彩色标注
    :param masks: (8, 200, 200) 形状的 int64 数组，每个通道代表一个类别的掩码 (0 或 1)
    :param classes: 类别名称列表
    :param background: 背景颜色
    """
    # assert masks.dtype == np.int64, masks.dtype  # 确保数据类型为 int64

    # 创建空白画布
    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    # 遍历每个类别，并应用颜色
    for k, name in enumerate(classes):
        if name in OBJECT_PALETTE_NUPLAN:
            canvas[masks[k] == 1, :] = OBJECT_PALETTE_NUPLAN[name]  # 只填充值为 1 的区域
        elif name in MAP_PALETTE_NUPLAN:
            canvas[masks[k] == 1, :] = MAP_PALETTE_NUPLAN[name]
        elif name in LANE_PALETTE_NUPLAN:
            canvas[masks[k] == 1, :] = LANE_PALETTE_NUPLAN[name]

    # 显示图像
    canvas = np.flipud(canvas)
    canvas = np.fliplr(canvas)

    return canvas

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
    9: (255, 140, 0),  # Darkorange trailer
    10: (255, 99, 71),  # Tomato truck
    11: (0, 207, 191),  # nuTonomy green driveable_surface
    12: (0,207,191), # road
    13: (75, 0, 75),  # sidewalk
    14: (150 , 240, 80 ),  # road-line
    15: (222, 184, 135),  # Burlywood mannade
    16: (0, 175, 0),  # Green vegetation
}


# openscene map
occ_colors_map = np.array(
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

openscene_map_to_ours = {
    0: 0,
    4: 5,
    5: 1,
    6: 6,
    7: 2,
    8: 3,
    9: 4,
    10: -1
}

def obtain_points_label(occ):

    num_classes = 16
    point_cloud_range = [-50.0, -50.0, -4.0, 50.0, 50.0, 4.0]
    occ_resolution ='coarse'
    if occ_resolution == 'coarse':
        occupancy_size = [0.5, 0.5, 0.5]
        voxel_size = 0.5
    else:
        occupancy_size = [0.2, 0.2, 0.2]
        voxel_size = 0.2
    occ_xdim = int((point_cloud_range[3] - point_cloud_range[0]) / occupancy_size[0])
    occ_ydim = int((point_cloud_range[4] - point_cloud_range[1]) / occupancy_size[1])
    occ_zdim = int((point_cloud_range[5] - point_cloud_range[2]) / occupancy_size[2])
    voxel_num = occ_xdim*occ_ydim*occ_zdim

    occ_index, occ_cls = occ[:, 0], occ[:, 1]
    occ = np.ones(voxel_num, dtype=np.int8)*11
    occ[occ_index[:]] = occ_cls  # (voxel_num)
    points = []
    for i in range(len(occ_index)):
        indice = occ_index[i]
        x = indice % occ_xdim
        y = (indice // occ_xdim) % occ_xdim
        z = indice // (occ_xdim*occ_xdim)
        # point_x = (x + 0.5) / occ_xdim * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
        # point_y = (y + 0.5) / occ_ydim * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
        # point_z = (z + 0.5) / occ_zdim * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
        # points.append([point_x, point_y, point_z])
        points.append([x, y, z])
    
    points = np.stack(points)
    points_label = occ_cls
    new_points_label = np.zeros_like(points_label)
    for k, v in openscene_map_to_ours.items():
        new_points_label[points_label == k] = v
    return points, new_points_label

# mlab.options.offscreen = True
fig = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
# 使用 mlab.pipeline.open 加载 PLY 模型
ours_path = 'data/occ_quan/nuplan_quantized_400_400_32'
bev_r200_path = 'data/nuplan_bev/sample_200'
bev_r400_path = 'data/nuplan_bev/sample_400'
bev_dir = bev_r400_path
save_path = 'data/vis_occ'
pkl_path = 'data/nuplan_pkls/mini/nuplan_mini_10hz_val.pkl'
with open(pkl_path, 'rb') as f:
    infos = pickle.load(f)
os.makedirs(save_path, exist_ok=True)
for item in infos['infos']:
    token = item['token']
    occ_path = os.path.join(ours_path, token+'.npy')
    # occ_path = os.path.join(ours_path, token, token+'.npz')
    try:
        if not os.path.exists(occ_path):
            continue
        
        # occ = np.load( occ_path, encoding='bytes', allow_pickle=True)['occ']
        _occ = np.load(occ_path)
        occ = np.stack(_occ.nonzero()).T
        semantics = _occ[occ[:, 0], occ[:, 1], occ[:, 2]] - 1
        occ = np.concatenate([occ, semantics[:, None]], axis=-1)
        
        # # occ with bev
        bev_path = os.path.join(bev_dir, token+'.npz')
        if os.path.exists(bev_path):
            bev_data = np.load(bev_path)['gt_bev_masks']
            bev = bev_data.copy()
            layer_to_merge=[0,1,3,4,5,6,7]
            bev[1,:,:] = np.any(bev[layer_to_merge, :, :], axis=0).astype(int)
            grid = np.zeros((400, 400, 32), dtype=np.uint8)
            grid[occ[:, 0], occ[:, 1], occ[:, 2]] = occ[:, 3]+1
            occ_with_bev = replace_occ_grid_with_bev_nuplan(grid, bev, bev_replace_idx=[1,15],occ_replace_new_idx=[12,14])
            _loc = np.stack(occ_with_bev.nonzero(), axis=-1)
            _label = occ_with_bev[_loc[:, 0], _loc[:, 1], _loc[:, 2]]
            occ_with_bev = np.concatenate([_loc, _label[:, None]], axis=-1)



            layer_to_merge=[0,1,3,4,5,6,7]
            bev_data[1,:,:] = np.any(bev_data[layer_to_merge, :, :], axis=0).astype(int)
            map_classes = list(MAP_PALETTE_NUPLAN.keys())
            object_classes = list(OBJECT_PALETTE_NUPLAN.keys())
            lane_classes = list(LANE_PALETTE_NUPLAN.keys())
            masks_class = map_classes + object_classes
            masks_class_with_line = map_classes + object_classes + lane_classes
            # bev[8]=bev_hight.

            bev_canvas=visualize_object_and_map(bev_data, classes=masks_class_with_line)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,6))
            plt.imshow(bev_canvas)
            plt.axis("off")
            plt.savefig(os.path.join(save_path, f'bev_{token}.png'))


        occ[:, :2] -= occ[:,:2].mean(0).astype('int64') # to center
        occ[:, -1] += 1
        voxels_ = occ
        voxels=voxels_
        voxel_size=2
        scene = fig.scene
        plt_plot = mlab.points3d(
            voxels[:, 0],
            voxels[:, 1],
            voxels[:, 2],
            voxels[:, 3],
            # colormap="viridis",
            scale_mode='none',
            scale_factor=voxel_size - 0.5 * voxel_size,
            mode="cube",
            opacity=1.0,
            transparent=False,
            vmin=0,
            vmax=256,
        )

        lut = np.array(list(classname_to_color.values())).astype('uint8')

        # 把 LUT 填充为 256 行（Mayavi 要求 LUT 是 256 行的）
        # 这里只使用前几行即可
        full_lut = np.zeros((256, 4), dtype=np.uint8)
        full_lut[:, -1] = 255
        full_lut[:len(lut), :3] = lut

        # 应用自定义 LUT
        plt_plot.module_manager.scalar_lut_manager.lut.table = full_lut

        scene.camera.position = [-50/0.2, 50/0.2, 40/0.2]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, -2]
        scene.camera.view_angle = 40.0
        scene.camera.view_up = [0, 0, 1]
        #scene.camera.clipping_range = [0, 1000000]
        scene.camera.compute_view_plane_normal()
        scene.camera.zoom(1.0)
        scene.camera.clipping_range = [1, 1000]
        scene.render()

        mlab.savefig(os.path.join(save_path, f'{token}.png'))
        mlab.clf(fig)

        # occ with bev
        if os.path.exists(bev_path):
            occ_with_bev[:, :2] -= occ_with_bev[:,:2].mean(0).astype('int64')
            voxels_ = occ_with_bev
            voxels=voxels_
            voxel_size=2
            scene = fig.scene
            plt_plot = mlab.points3d(
                voxels[:, 0],
                voxels[:, 1],
                voxels[:, 2],
                voxels[:, 3],
                # colormap="viridis",
                scale_mode='none',
                scale_factor=voxel_size - 0.5 * voxel_size,
                mode="cube",
                opacity=1.0,
                transparent=False,
                vmin=0,
                vmax=256,
            )

            lut = np.array(list(classname_to_color.values())).astype('uint8')

            # 把 LUT 填充为 256 行（Mayavi 要求 LUT 是 256 行的）
            # 这里只使用前几行即可
            full_lut = np.zeros((256, 4), dtype=np.uint8)
            full_lut[:, -1] = 255
            full_lut[:len(lut), :3] = lut

            # 应用自定义 LUT
            plt_plot.module_manager.scalar_lut_manager.lut.table = full_lut

            scene.camera.position = [-50/0.2, 50/0.2, 40/0.2]
            scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, -2]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [0, 0, 1]
            #scene.camera.clipping_range = [0, 1000000]
            scene.camera.compute_view_plane_normal()
            scene.camera.zoom(1.0)
            scene.camera.clipping_range = [1, 1000]
            scene.render()

            #mlab.draw()
            #mlab.show()
            mlab.savefig(os.path.join(save_path, f'{token}_withbev.png'))
            mlab.clf(fig)


        print(f'Finished occ visualization for {token}')

    except Exception as e:
        print(f'Fail to visualize {token}, as:')
        print(e)
    # break
display.stop()