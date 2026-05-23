import os
from PIL import Image
#os.environ['QT_DEBUG_PLUGINS'] = "1"
import pickle
import multiprocessing
import numpy as np
# from pypcd import pypcd
from scipy.spatial.transform import Rotation as R
from pyvirtualdisplay import Display
H, W = 1080, 1920
display = Display(visible=False, size=(W, H))
display.start()
from mayavi import mlab
import imageio

# data_root = 'data/lixiang_data'
# pkl_file = 'data/lixiang_data.pkl'
# save_path = 'vis/lixiang_occ_preview'
# os.makedirs(save_path, exist_ok=True)
# with open(pkl_file, 'rb') as f:
    # infos = pickle.load(f)
# token_to_info_id = {}
# for i, info in enumerate(infos['infos']):
    # token_to_info_id[info['token']] = i
downsample_rate = 2

import yaml

# 读取 YAML 文件
with open('data/lixiang_meta.yaml', 'r', encoding='utf-8') as f:
    lixiang_meta = yaml.safe_load(f)['voxelizer']

voxel_size = lixiang_meta['upsample_voxel_size']
pc_range = np.array([*lixiang_meta['min_extent'], *lixiang_meta['max_extent']])
classname_to_color = lixiang_meta['color_map_mapped_v2']

def custom_colormap(plt_plot, colormap=classname_to_color):
    ori_colormap = plt_plot.module_manager.scalar_lut_manager.lut.table.to_array()
    for key, value in colormap.items():
        ori_colormap[key, :3] = value
    plt_plot.module_manager.scalar_lut_manager.lut.table = ori_colormap.astype(np.uint8)[:17]


def draw(voxels, voxel_size=0.2, vis_root=None, idx=0, figure=None, mlab=None):
    if figure is None:
        figure = mlab.figure(size=(W, H), bgcolor=(1, 1, 1))
    scene = figure.scene

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
    lut = np.array(list(classname_to_color.values())).astype('uint8')

    # 把 LUT 填充为 256 行（Mayavi 要求 LUT 是 256 行的）
    # 这里只使用前几行即可
    full_lut = np.zeros((256, 4), dtype=np.uint8)
    full_lut[:, -1] = 255
    full_lut[np.array(list(classname_to_color.keys())), :3] = lut

    # 应用自定义 LUT
    plt_plot.module_manager.scalar_lut_manager.lut.table = full_lut

    scene.camera.position = [-30/0.2, 50/0.2, 20/0.2]
    scene.camera.focal_point = [668/2, 320/2, -20]
    scene.camera.view_angle = 40.0
    scene.camera.view_up = [0, 0, 1]
    #scene.camera.clipping_range = [0, 1000000]
    scene.camera.compute_view_plane_normal()
    scene.camera.zoom(1.0)
    scene.camera.clipping_range = [1, 1000]
    scene.render()

    #mlab.show()


    mlab.draw()
    # vis_path = os.path.join(vis_root, '{:0>4d}.png'.format(idx))
    # mlab.savefig(filename=vis_path)

import av

def is_valid_mp4(filepath):
    try:
        container = av.open(filepath)
        video_streams = [s for s in container.streams if s.type == 'video']
        return len(video_streams) > 0
    except av.AVError:
        return False

# def render_one_scene(scene_id, sample_tokens):
#     # from mayavi import mlab
#     print(f'Start render scene{scene_id}')
#     mp4_path = os.path.join(save_path, f'{scene_id}.mp4')
#     if os.path.exists(mp4_path) and is_valid_mp4(mp4_path):
#         print(f'{mp4_path} already processed!')
#         return
    
#     writer = imageio.get_writer(mp4_path, fps=10//downsample_rate)
#     figure = mlab.figure(size=(W, H), bgcolor=(1, 1, 1))
#     for i, sample_token in enumerate(sample_tokens[::downsample_rate]):
#         cur_info = infos['infos'][token_to_info_id[sample_token]]
#         occ_path = os.path.join(data_root, cur_info['occ_path'])
#         if not os.path.exists(occ_path):
#             print(f'scene {scene_id} is not completed')
#             break

#         # npy
#         #voxels_ = np.load(occ_path)
        
#         # npz
#         voxels_ = np.load(occ_path)['upsample'].reshape([668, 320, 32])
#         locs = np.stack((voxels_!=255).nonzero(), axis=-1)
#         labels = voxels_[locs[:, 0], locs[:, 1], locs[:, 2]]
#         voxels = np.concatenate([locs, labels.reshape((-1, 1))], axis=-1)

#         mlab.clf()
#         draw(voxels, voxel_size=2, figure=figure, mlab=mlab)

#         mlab.savefig("test_output.png")

#         frame = mlab.screenshot(mode='rgb', antialiased=True)
#         writer.append_data(frame)
        
#         print(f"{scene_id} 渲染{i}/{len(sample_tokens[::downsample_rate])}")
#     mlab.close(all=True)
#     #writer.close()

def reverse_map_labels(mapped_array):
    """
    从映射后的类别（0-9）反投影回原始类别，但只保留特定类别：
    0,1,2,3,4,5,6,7,8,100,255
    其他类别（如253,254）设为0（或其他默认值）
    """
    # 初始化输出，默认设为0（或255，取决于你的需求）
    original_labels = np.zeros_like(mapped_array)
    
    # 直接映射的类别（唯一映射）
    original_labels[mapped_array == 0] = 0   # 0 → 255（255也可以，但这里选择0）
    original_labels[mapped_array == 1] = 1   # 1 → 1（253被丢弃，设为0）
    original_labels[mapped_array == 2] = 2
    original_labels[mapped_array == 3] = 3
    original_labels[mapped_array == 4] = 4   # 4 → 4（254被丢弃，设为0）
    original_labels[mapped_array == 5] = 5
    original_labels[mapped_array == 6] = 6
    original_labels[mapped_array == 7] = 7
    original_labels[mapped_array == 8] = 8
    original_labels[mapped_array == 9] = 100  # 9 → 100
    
    
    return original_labels

def extract_first_volume(array_like):
    """
    提取体素体第一帧/第一个样本：
    - 5D: [B, T, X, Y, Z] -> [0, 0]
    - 4D: [B, X, Y, Z] -> [0]
    - 3D: [X, Y, Z] -> 本体
    其他维度抛出异常
    """
    if array_like.ndim == 5:
        return array_like[0, 0]
    if array_like.ndim == 4:
        return array_like[0]
    if array_like.ndim == 3:
        return array_like
    raise ValueError(f"Unsupported ndim {array_like.ndim} for volume extraction")

def visualize_and_save(volume_3d, figure, save_path):
    """
    对单个 3D 体素数组进行反投影上色并渲染保存
    """
    unique_elements, counts = np.unique(volume_3d, return_counts=True)
    print(f"before:=={volume_3d.shape}=={unique_elements}={counts}")

    occ_data_restored = reverse_map_labels(volume_3d)

    unique_elements, counts = np.unique(occ_data_restored, return_counts=True)
    print(f"after =={occ_data_restored.shape}=={unique_elements}={counts}")

    locs = np.stack((occ_data_restored != 0).nonzero(), axis=-1)
    if locs.size == 0:
        print(f"skip empty volume -> {save_path}")
        return
    labels = occ_data_restored[locs[:, 0], locs[:, 1], locs[:, 2]]
    voxels = np.concatenate([locs, labels.reshape((-1, 1))], axis=-1)
    mlab.clf()
    draw(voxels, voxel_size=2, figure=figure, mlab=mlab)
    mlab.savefig(save_path)

if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    #filepath = 'z_nuplan_occ.npy'
    # folder_path = 'out/lixiang/2025_08_05_19_52_38/save_occ'  # 替换为你的文件夹径
    folder_path = 'out/sync_lixiang/save_occ'  # 替换为你的文件夹径
    # out/eval_lixiang_dit/eval_checkpoints_2025-08-04-16-27-56/visualizations
    vis_save_dir = f'{folder_path}_vis/'
    os.makedirs(vis_save_dir, exist_ok=True)
    files_to_rename = []  # 存储符合条件的文件名列表
    # 遍历文件夹及其子文件夹中所有文件
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.npz'):
                files_to_rename.append(os.path.join(root, filename))
    # 按文件名排序
    files_to_rename.sort()

    figure = mlab.figure(size=(W, H), bgcolor=(1, 1, 1))

    for filepath in files_to_rename:
        print(filepath)
        basename = os.path.splitext(os.path.basename(filepath))[0]
        data = np.load(filepath)

        for key in ['input_occs', 'pred_occs']:
            if key in data:
                try:
                    volume = extract_first_volume(data[key])
                except Exception as e:
                    print(f"extract volume failed for {key}: {e}")
                    continue
                save_path = f"{folder_path}_vis/{basename}_{key}.png"
                visualize_and_save(volume, figure, save_path)
            else:
                print(f"{key} not found in {filepath}")



    display.stop()