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
# ours map
# labels:
#  0: vehicle
#  1: bicycle
#  2: pedestrian
#  3: traffic_cone
#  4: barrier
#  5: czone_sign
#  6: generic_object
#  7: background

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
ours_path = 'data/nuplan/GT_occ_fast3_r400_vis/vis'
baseline_path = 'data/nuplan/GT_occ_fast3_r400_vis/vis_baseline'
openscene_path = 'data/openscene/openscene-v1.0/occupancy/mini'
bev_r200_path = '/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200'
bev_r400_path = '/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_400'
bev_path = bev_r400_path
downsampled_occ_path = 'data/occ_quan/nuplan_quantized_400_400_32'
save_path = 'data/vis_occ_pipeline'
pkl_path = 'data/nuplan_pkls/mini/nuplan_mini_10hz_train.pkl'
with open(pkl_path, 'rb') as f:
    infos = pickle.load(f)
seq_id_to_token = {}
for item in infos['infos']:
    seq_id_to_token[item['scene_name']+'_'+str(item['frame_idx'])] = item['token']
all_seqs = os.path.join(ours_path)
for seq in os.listdir(all_seqs):
    if not os.path.exists(os.path.join(baseline_path, seq)):
        continue
    
    try:
        save_path_cur = os.path.join(save_path, seq)
        os.makedirs(save_path_cur, exist_ok=True)

        mesh_ours = mlab.pipeline.open(os.path.join(ours_path, seq, 'scene_mesh.ply'))
        mesh_baseline = mlab.pipeline.open(os.path.join(baseline_path, seq, 'scene_mesh.ply'))
        points_path = os.path.join(ours_path, seq, 'agg_pc.bin')
        points_path_baseline = os.path.join(baseline_path, seq, 'agg_pc.bin')
        object_files = os.listdir(os.path.join(ours_path, seq, 'object_points'))
        occ_files = natsorted(os.listdir(os.path.join(ours_path, seq, 'occ_final')))

        # mesh可视化
        if 1:
            ########################### ours ###########################
            # 渲染为表面模型，支持自定义 colormap、透明度等参数
            surf = mlab.pipeline.surface(mesh_ours, colormap='viridis', opacity=1.0)

            # mlab.show()

            scene = fig.scene

            # scene.camera.position = [-70, -70, 60]
            # scene.camera.position = [-50, -50, 40]
            scene.camera.position = [-50, 50, 40]
            scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, -2]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [0, 0, 1]
            scene.camera.clipping_range = [10, 222.91192666552377]
            scene.camera.compute_view_plane_normal()
            scene.render()

            mlab.savefig(os.path.join(save_path_cur, 'ours_mesh.png'))
            mlab.clf(fig)

            ########################### baseline ###########################
            # 渲染为表面模型，支持自定义 colormap、透明度等参数
            surf = mlab.pipeline.surface(mesh_baseline, colormap='viridis', opacity=1.0)

            # mlab.show()

            scene = fig.scene

            # scene.camera.position = [-70, -70, 60]
            # scene.camera.position = [-50, -50, 40]
            scene.camera.position = [-50, 50, 40]
            scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, -2]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [0, 0, 1]
            scene.camera.clipping_range = [10, 222.91192666552377]
            scene.camera.compute_view_plane_normal()
            scene.render()

            mlab.savefig(os.path.join(save_path_cur, 'baseline_mesh.png'))
            mlab.clf(fig)
            print(f'Finished mesh visualization for {seq}')

        # occ可视化
        if 1:
            save_path_occ_cur = os.path.join(save_path, seq, 'occ')
            os.makedirs(save_path_occ_cur, exist_ok=True)
            # occ_files_to_vis = occ_files[::4]
            occ_files_to_vis = [occ_files[len(occ_files)//2]]
            # ours
            for occ_id, occ_file in enumerate(occ_files_to_vis):
                occ_id = occ_file.split('.')[0]
                occ_path = os.path.join(ours_path, seq, 'occ_final', occ_file)
                if not os.path.exists(occ_path):
                    continue

                occ = np.load( occ_path, encoding='bytes', allow_pickle=True)['occ']
                
                # # occ with bev
                token = seq_id_to_token[seq+'_'+occ_id]
                bev_path = os.path.join(bev_path, token+'.npz')
                if os.path.exists(bev_path):
                    bev = np.load(os.path.join(bev_path, token+'.npz'))['gt_bev_masks']
                    bev = bev
                    layer_to_merge=[0,1,3,4,5,6,7]
                    bev[1,:,:] = np.any(bev[layer_to_merge, :, :], axis=0).astype(int)
                    grid = np.zeros((400, 400, 32), dtype=np.uint8)
                    grid[occ[:, 0], occ[:, 1], occ[:, 2]] = occ[:, 3]+1
                    occ_with_bev = replace_occ_grid_with_bev_nuplan(grid, bev)
                    _loc = np.stack(occ_with_bev.nonzero(), axis=-1)
                    _label = occ_with_bev[_loc[:, 0], _loc[:, 1], _loc[:, 2]]
                    occ_with_bev = np.concatenate([_loc, _label[:, None]], axis=-1)


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

                # 构造自定义 LUT（颜色可根据需要修改）
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
                mlab.savefig(os.path.join(save_path_occ_cur, f'ours_{occ_id}.png'))
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

                    # 构造自定义 LUT（颜色可根据需要修改）
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
                    mlab.savefig(os.path.join(save_path_occ_cur, f'ours_{occ_id}_withbev.png'))
                    mlab.clf(fig)

            # baseline
            for occ_id, occ_file in enumerate(occ_files_to_vis):
                occ_id = occ_file.split('.')[0]
                occ_path = os.path.join(baseline_path, seq, 'occ_final', occ_file)
                if not os.path.exists(occ_path):
                    continue

                occ = np.load( occ_path, encoding='bytes', allow_pickle=True)['occ']
                occ[:, :2] -= occ[:,:2].mean(0).astype('int64')
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
                    colormap="viridis",
                    scale_mode='none',
                    scale_factor=voxel_size - 0.5 * voxel_size,
                    mode="cube",
                    opacity=1.0,
                    transparent=False,
                    vmin=0,
                    vmax=256,
                )

                # 构造自定义 LUT（颜色可根据需要修改）
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
                mlab.savefig(os.path.join(save_path_occ_cur, f'baseline_{occ_id}.png'))
                mlab.clf(fig)

            # openscene
            for occ_id, occ_file in enumerate(occ_files_to_vis):
                occ_id = occ_file.split('.')[0]
                occ_path = os.path.join(openscene_path, seq, 'occ_gt', f'{int(occ_file.split(".")[0])//5:03d}_occ_final.npy')
                if not os.path.exists(occ_path):
                    continue

                occ = np.load(occ_path)
                points, label = obtain_points_label(occ)
                points = points.astype('int64')
                label = label.astype('int64')
                points[:, :2] -= points[:,:2].mean(0).astype('int64')
                voxel_size=2
                scene = fig.scene
                plt_plot = mlab.points3d(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    label+2,
                    colormap="viridis",
                    scale_mode='none',
                    scale_factor=voxel_size - 0.5 * voxel_size,
                    mode="cube",
                    opacity=1.0,
                    transparent=False,
                    vmin=0,
                    vmax=256,
                )

                # 构造自定义 LUT（颜色可根据需要修改）
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
                scene.camera.zoom(2.0)
                scene.camera.clipping_range = [1, 1000]
                scene.render()

                #mlab.draw()
                #mlab.show()
                mlab.savefig(os.path.join(save_path_occ_cur, f'openscene_{occ_id}.png'))
                mlab.clf(fig)
            print(f'Finished occ visualization for {seq}')

        # points可视化
        if 1:
            # ours
            points = np.fromfile(points_path, dtype='float32').reshape(-1, 3)
            voxel_size = 0.1
            #scene = figure.scene
            plt_plot = mlab.points3d(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                points[:, 2],
                # voxels[:, 3],
                colormap="viridis",
                scale_mode='none',
                scale_factor=voxel_size - 0.5 * voxel_size,
                mode="point",
                opacity=1.0,
                transparent=False,
                #vmin=0,
                #vmax=256,
            )
            # 构造自定义 LUT（颜色可根据需要修改）
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
            lut = np.array(list(classname_to_color.values())).astype('uint8')

            # 把 LUT 填充为 256 行（Mayavi 要求 LUT 是 256 行的）
            # 这里只使用前几行即可
            full_lut = np.zeros((256, 4), dtype=np.uint8)
            full_lut[:, -1] = 255
            full_lut[:len(lut), :3] = lut

            # 应用自定义 LUT
            # plt_plot.module_manager.scalar_lut_manager.lut.table = full_lut
            
            scene = fig.scene
            scene.camera.position = [-50, 50, 40]
            scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, -2]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [0, 0, 1]
            scene.camera.clipping_range = [10, 222.91192666552377]
            scene.camera.compute_view_plane_normal()
            scene.render()

            mlab.draw()
            mlab.savefig(os.path.join(save_path_cur, 'our_points.png'))
            mlab.clf(fig)

            # baseline
            points = np.fromfile(points_path_baseline, dtype='float32').reshape(-1, 3)
            voxel_size = 0.1
            #scene = figure.scene
            plt_plot = mlab.points3d(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                points[:, 2],
                # voxels[:, 3],
                colormap="viridis",
                scale_mode='none',
                scale_factor=voxel_size - 0.5 * voxel_size,
                mode="point",
                opacity=1.0,
                transparent=False,
                #vmin=0,
                #vmax=256,
            )
            # 构造自定义 LUT（颜色可根据需要修改）
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
            lut = np.array(list(classname_to_color.values())).astype('uint8')

            # 把 LUT 填充为 256 行（Mayavi 要求 LUT 是 256 行的）
            # 这里只使用前几行即可
            full_lut = np.zeros((256, 4), dtype=np.uint8)
            full_lut[:, -1] = 255
            full_lut[:len(lut), :3] = lut

            # 应用自定义 LUT
            # plt_plot.module_manager.scalar_lut_manager.lut.table = full_lut
            
            scene = fig.scene
            scene.camera.position = [-50, 50, 40]
            scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, -2]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [0, 0, 1]
            scene.camera.clipping_range = [10, 222.91192666552377]
            scene.camera.compute_view_plane_normal()
            scene.render()

            mlab.draw()
            mlab.savefig(os.path.join(save_path_cur, 'baseline_points.png'))
            mlab.clf(fig)
            print(f'Finished points visualization for {seq}')

        # object points and object mesh 可视化
        if 1:
            save_path_points_cur = os.path.join(save_path, seq, 'object_points')
            os.makedirs(save_path_points_cur, exist_ok=True)

            save_path_mesh_cur = os.path.join(save_path, seq, 'object_meshs')
            os.makedirs(save_path_mesh_cur, exist_ok=True)
            # ours
            for object_file in object_files:
                occ = np.fromfile(os.path.join(ours_path, seq, 'object_points', object_file), dtype='float32').reshape(-1, 3)
                if occ.shape[0] < 8192:
                    continue

                # occ_loc = np.stack(occ.nonzero(), axis=-1)[:, [2, 1, 0]]
                # occ = np.concatenate([occ_loc, occ[occ_loc[:, 2], occ_loc[:, 1], occ_loc[:, 0]][:, None]], axis=-1)
                voxels_ = occ
                voxels=voxels_

                ######################################################
                #scene = figure.scene
                plt_plot = mlab.points3d(
                    voxels[:, 0],
                    voxels[:, 1],
                    voxels[:, 2],
                    np.ones_like(voxels[:, 0]),
                    # voxels[:, 3],
                    colormap="viridis",
                    scale_mode='none',
                    scale_factor=0.02,
                    #mode="point",
                    opacity=1.0,
                    transparent=False,
                    vmin=0,
                    vmax=256,
                )
                # mlab.points3d(8, 99, 125, mode='sphere', opacity=1.0, scale_factor=5, color=(1.0,0.0,0.0))
                #mlab.axes(xlabel='x',ylabel='y',zlabel='z',ranges= (-5,5,-5,5,-5,5), color=(1,0,0))
            
                #scene.render()

                # 构造自定义 LUT（颜色可根据需要修改）
                classname_to_color = {  # RGB.
                    0: (255, 255, 255),  # Black. noise
                    # 1: (112, 128, 144),  # Slategrey barrier 
                    1: (255, 31, 69),  # Red motorcycle
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
                lut = np.array(list(classname_to_color.values())).astype('uint8')

                # 把 LUT 填充为 256 行（Mayavi 要求 LUT 是 256 行的）
                # 这里只使用前几行即可
                full_lut = np.zeros((256, 4), dtype=np.uint8)
                full_lut[:, -1] = 255
                full_lut[:len(lut), :3] = lut

                scene = fig.scene
                scene.camera.zoom(1.5)

                # 应用自定义 LUT
                plt_plot.module_manager.scalar_lut_manager.lut.table = full_lut
                mlab.savefig(os.path.join(save_path_points_cur, f'ours_{object_file.split(".")[0]}.png'))
                mlab.clf(fig)

                ################### mesh ##################### 
                # 使用 mlab.pipeline.open 加载 PLY 模型
                src = mlab.pipeline.open(os.path.join(ours_path, seq, 'object_mesh', object_file.replace('.bin', '.ply')))

                # 渲染为表面模型，支持自定义 colormap、透明度等参数
                surf = mlab.pipeline.surface(src, colormap='viridis', opacity=1.0)

                # mlab.show()

                scene = fig.scene
                scene.camera.zoom(1.5)

                mlab.savefig(os.path.join(save_path_mesh_cur, f'ours_{object_file.split(".")[0]}.png'))
                mlab.clf(fig)
            print(f'Finished object points visualization for {seq}')
        
    except Exception as e:
        print(f'Fail to visualize {seq}, as:')
        print(e)
    # break
display.stop()