#!/usr/bin/env python3
"""
NuPlan占用网格可视化脚本 - 视频和图片生成版本
为每个场景生成GT、Pred和对比的MP4视频，同时保存每个token的单张图片
"""

import os
import pickle
import multiprocessing
import numpy as np
import glob
import re
from collections import defaultdict
from scipy import ndimage  # 用于替代cv2的膨胀操作

# 简化的显示设置 - 参考NuScenes版本
from pyvirtualdisplay import Display
H, W = 1080, 1920
display = Display(visible=False, size=(W, H))
display.start()

import imageio  # 用于生成MP4视频
from PIL import Image  # 用于保存对比图片

pkl_file = '/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_val.pkl'
occ_root = 'outputs/occ_generation/save_occ'
vis_root = "outputs/occ_generation/save_occ_mp4"
vis_root_frame = "outputs/occ_generation/save_occ_frame"
bev_root = '/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200'
os.makedirs(vis_root_frame, exist_ok=True)

# 读取pkl文件
with open(pkl_file, 'rb') as f:
    pkl_data = pickle.load(f)


def get_scene_tokens_from_pkl(pkl_data):
    """从pkl文件获取场景tokens"""
    scene_tokens = pkl_data['scene_tokens']  # 这应该是一个场景token的列表
    print(f"从pkl文件中找到 {len(scene_tokens)} 个场景")
    return scene_tokens

def find_occ_file(token, occ_root, file_type):
    """根据token查找对应的occ文件"""
    if file_type == 'gt':
        pattern = f"{token}_gt_occ.npy"
    elif file_type == 'pred':
        pattern = f"{token}_pred.npy"
    else:
        return None
    
    files = glob.glob(os.path.join(occ_root, pattern))
    return files[0] if files else None

def replace_occ_grid_with_bev_nuplan(input_occ_data, bevlayout, driva_area_idx=1, bev_replace_idx=[1,15],
                                    occ_replace_new_idx=[12, 14]):
    """
    用BEV数据替换occupancy grid中的特定区域
    参数:
    - input_occ_data: 输入的occupancy数据 [200,200,16]
    - bevlayout: BEV布局数据 [18,200,200]  
    - driva_area_idx: 可驾驶区域在occ中的索引
    - bev_replace_idx: BEV中需要替换的层索引
    - occ_replace_new_idx: 替换后在occ中的新索引
    """
    
    # 处理road_divider和lane_divider的mask
    roal_divider_mask = bevlayout[15, :, :].astype(np.uint8)
    lane_divider_mask = bevlayout[17, :, :].astype(np.uint8)

    # kernel = np.ones((1, 1), dtype=bool)
    # roal_divider_mask = ndimage.binary_dilation(roal_divider_mask, structure=kernel).astype(np.uint8)
    # lane_divider_mask = ndimage.binary_dilation(lane_divider_mask, structure=kernel).astype(np.uint8)

    bevlayout[15, :, :] = roal_divider_mask.astype(bool)
    bevlayout[17, :, :] = lane_divider_mask.astype(bool)

    n = len(bev_replace_idx)
    x_max, y_max = input_occ_data.shape[0], input_occ_data.shape[1]
    output_occ = input_occ_data.copy().astype(np.float32)  # 确保输出类型一致
    bev_replace_mask = []
    for i in range(n):
        bev_replace_mask.append(bevlayout[bev_replace_idx[i]] == 1)
        print(f"🔧 BEV mask {i}: {np.sum(bev_replace_mask[i])} 个像素")

    # 遍历每个像素点进行替换
    replace_count = 0
    for x in range(x_max):
        for y in range(y_max):
            for i in range(n):
                if bev_replace_mask[i][x, y]:
                    occupancy_data = input_occ_data[x, y, :]
                    if driva_area_idx in occupancy_data:
                        max_11_index = np.where(occupancy_data == driva_area_idx)
                        output_occ[x, y, max_11_index] = occ_replace_new_idx[i]
                        replace_count += len(max_11_index[0])
    
    return output_occ

def load_bev_data(token, bev_root):
    """加载BEV数据并进行预处理"""
    bev_file_path = os.path.join(bev_root, f'{token}.npz')
    if not os.path.exists(bev_file_path):
        print(f"BEV文件不存在: {bev_file_path}")
        return None
    
    bev_data = np.load(bev_file_path)['gt_bev_masks']  # [18, 200, 200]
    
    # 合并特定层
    layer_to_merge = [0, 1, 3, 4, 5, 6, 7]
    bev_data[1, :, :] = np.any(bev_data[layer_to_merge, :, :], axis=0).astype(int)
    
    return bev_data

classname_to_color = {  # RGB.
    0: (255, 255, 255),  # free, white
    1: (0,175,0),        # other-ground, green
    2: (255, 158, 0),    # vehicle, orange
    3: (220,20,60),      # bicycle, crimson red
    4: (0,0,230),        # pedestrian, blue
    5: (47,79,79),       # traffic-cone, dark slate gray
    6: (112,128,144),    # barrier, slate gray
    7: (255, 200, 0),    # construction-zones, yellow
    8: (222,184,13),     # generic-object, goldenrod
    12: (0,207,191),     # road, cyan-green
    14: (150 , 240, 80), # road-line, lime green
}


def custom_colormap(plt_plot, colormap=classname_to_color):
    ori_colormap = plt_plot.module_manager.scalar_lut_manager.lut.table.to_array()
    for key, value in colormap.items():
        ori_colormap[key, :3] = value
    plt_plot.module_manager.scalar_lut_manager.lut.table = ori_colormap.astype(np.uint8)[:17]


def draw(voxels, figure, mlab, voxel_size=0.5):
    """绘制3D占用网格"""
    scene = figure.scene

    # 确保背景为白色
    scene.background = (1.0, 1.0, 1.0)
    scene.foreground = (0.0, 0.0, 0.0)

    # 绘制3D点云 - 使用实际voxel尺寸作为scale_factor
    plt_plot = mlab.points3d(
        voxels[:, 0],
        voxels[:, 1],
        voxels[:, 2],
        voxels[:, 3],
        scale_factor=voxel_size,  # 使用实际的voxel尺寸
        scale_mode='vector',
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=16,
    )
 
    scene.render()
    custom_colormap(plt_plot)
    mlab.draw()


def get_grid_coords(dims, resolution):
	"""
	:param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
	:return coords_grid: is the center coords of voxels in the grid
	"""
	g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
	g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
	g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]
	# Obtaining the grid with coords...
	xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
	coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
	coords_grid = coords_grid.astype(np.float32)
	resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])
	coords_grid = (coords_grid * resolution) + resolution / 2
	return coords_grid

def render_one_scene(scene_id, scene_tokens, voxel_size, voxel_origin, grid):

    from mayavi import mlab
    print(f'开始渲染场景 {scene_id}')

    os.makedirs(vis_root, exist_ok=True)
    os.makedirs(vis_root_frame, exist_ok=True)
    
    # 为每个场景创建三个视频：GT、Pred、对比
    gt_video_path = os.path.join(vis_root, f'scene_{scene_id}_gt.mp4')
    pred_video_path = os.path.join(vis_root, f'scene_{scene_id}_pred.mp4')
    compare_video_path = os.path.join(vis_root, f'scene_{scene_id}_compare.mp4')
    
    gt_writer = imageio.get_writer(gt_video_path, fps=10)
    pred_writer = imageio.get_writer(pred_video_path, fps=10)
    compare_writer = imageio.get_writer(compare_video_path, fps=10)
    
    # 创建mayavi图形窗口
    figure_gt = mlab.figure(size=(W, H), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    figure_pred = mlab.figure(size=(W, H), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    
    valid_tokens = 0
    for i, token in enumerate(scene_tokens):
        print(f"\n处理token {i+1}/{len(scene_tokens)}: {token}")
        
        # 根据token查找对应的GT和Pred文件
        gt_path = find_occ_file(token, occ_root, 'gt')
        pred_path = find_occ_file(token, occ_root, 'pred')
        
        if not gt_path or not pred_path:
            print(f"跳过token {token}: GT文件={gt_path is not None}, Pred文件={pred_path is not None}")
            continue
            
        valid_tokens += 1
        
        # 加载GT和Pred数据
        gt_voxels_raw = np.load(gt_path)
        pred_voxels_raw = np.load(pred_path)
        print(f"数据形状 - GT: {gt_voxels_raw.shape}, Pred: {pred_voxels_raw.shape}")
        
        # 处理数据形状：如果是5D数据就取第一个时间步，如果是4D就直接使用
        if len(gt_voxels_raw.shape) == 5:
            # 从 (1, 5, 200, 200, 16) 取第一个时间步 -> (200, 200, 16)
            gt_voxels = gt_voxels_raw[0, 0]  # 取第一个batch的第一个时间步
        elif len(gt_voxels_raw.shape) == 4:
            # 从 (5, 200, 200, 16) 取第一个时间步 -> (200, 200, 16)
            gt_voxels = gt_voxels_raw[0]
        else:
            gt_voxels = gt_voxels_raw
            
        if len(pred_voxels_raw.shape) == 5:
            pred_voxels = pred_voxels_raw[0, 0]  # 取第一个batch的第一个时间步
        elif len(pred_voxels_raw.shape) == 4:
            pred_voxels = pred_voxels_raw[0]
        else:
            pred_voxels = pred_voxels_raw
        
        print(f"处理后数据形状 - GT: {gt_voxels.shape}, Pred: {pred_voxels.shape}")
        
        # 加载BEV数据并应用替换
        bev_data = load_bev_data(token, bev_root)
        
        # BEV数据替换处理
        if bev_data is not None:
            print(f"BEV数据形状: {bev_data.shape}")
            print(f"  BEV替换前 - GT类别: {np.unique(gt_voxels)}")
            print(f"  BEV替换前 - Pred类别: {np.unique(pred_voxels)}")
            
            gt_voxels = replace_occ_grid_with_bev_nuplan(gt_voxels, bev_data.copy())
            pred_voxels = replace_occ_grid_with_bev_nuplan(pred_voxels, bev_data.copy())
            
        else:
            print("未找到BEV数据，使用原始占用网格")
        
        # 处理GT数据
        gt_grid_coords = get_grid_coords(
            [gt_voxels.shape[0], gt_voxels.shape[1], gt_voxels.shape[2]], voxel_size
        ) + np.array(voxel_origin, dtype=np.float32).reshape([1, 3])
        gt_grid_coords = np.vstack([gt_grid_coords.T, gt_voxels.reshape(-1)]).T
        
        gt_grid_coords = gt_grid_coords[
            (gt_grid_coords[:, 3] > 0) & (gt_grid_coords[:, 3] < 20)
        ]
        
        # 处理Pred数据
        pred_grid_coords = get_grid_coords(
            [pred_voxels.shape[0], pred_voxels.shape[1], pred_voxels.shape[2]], voxel_size
        ) + np.array(voxel_origin, dtype=np.float32).reshape([1, 3])
        pred_grid_coords = np.vstack([pred_grid_coords.T, pred_voxels.reshape(-1)]).T
        
        pred_grid_coords = pred_grid_coords[
            (pred_grid_coords[:, 3] > 0) & (pred_grid_coords[:, 3] < 20)
        ]
        
        # 显示类别分布信息
        if len(gt_grid_coords) > 0:
            gt_unique, gt_counts = np.unique(gt_grid_coords[:, 3], return_counts=True)
            print(f"  GT类别: {dict(zip(gt_unique.astype(int), gt_counts))}")
        
        if len(pred_grid_coords) > 0:
            pred_unique, pred_counts = np.unique(pred_grid_coords[:, 3], return_counts=True)
            print(f"  Pred类别: {dict(zip(pred_unique.astype(int), pred_counts))}")
        
        # 渲染GT帧
        mlab.figure(figure_gt)
        mlab.clf()  # 清空之前的内容
        draw(gt_grid_coords, figure_gt, mlab, voxel_size=voxel_size[0])
        gt_frame = mlab.screenshot(mode='rgb', antialiased=True)
        gt_writer.append_data(gt_frame)
        
        # 保存GT单张图片
        gt_image_path = os.path.join(vis_root_frame, f'{token}_gt.png')
        mlab.savefig(filename=gt_image_path, size=(W, H))
        
        # 渲染Pred帧
        mlab.figure(figure_pred)
        mlab.clf()  # 清空之前的内容
        draw(pred_grid_coords, figure_pred, mlab, voxel_size=voxel_size[0])
        pred_frame = mlab.screenshot(mode='rgb', antialiased=True)
        pred_writer.append_data(pred_frame)
        
        # 保存Pred单张图片
        pred_image_path = os.path.join(vis_root_frame, f'{token}_pred.png')
        mlab.savefig(filename=pred_image_path, size=(W, H))
        
        # 创建对比帧（左GT，右Pred）
        compare_frame = np.concatenate([gt_frame, pred_frame], axis=1)
        compare_writer.append_data(compare_frame)
        
        # 保存对比图片
        compare_image_path = os.path.join(vis_root_frame, f'{token}_compare.png')
        compare_img = Image.fromarray(compare_frame)
        compare_img.save(compare_image_path)
        
        print(f"Token {token} 帧 {valid_tokens} 渲染完成 ✅")
    
    # 关闭视频写入器和清理资源
    gt_writer.close()
    pred_writer.close()
    compare_writer.close()
    
    # 关闭mayavi图形窗口
    mlab.close(figure_gt)
    mlab.close(figure_pred)
    
    print(f'✅ 场景 {scene_id} 渲染完成，处理了 {valid_tokens} 个有效tokens')
    print(f'  📹 视频: {gt_video_path}')
    print(f'  🖼️ 图片: {vis_root_frame}/ (共 {valid_tokens * 3} 张)')

if __name__ == '__main__':
    print("=== NuPlan 占用网格可视化工具 ===")
    print("🎬 生成场景视频 + 🖼️ 保存单张图片")
    
    # 设置体素参数
    point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]  # 点云范围
    occ_size = [200, 200, 16]  # 占用网格尺寸
    
    # 计算体素分辨率
    voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
    voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
    voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
    voxel_size = [voxel_x, voxel_y, voxel_z]
    

    # 获取场景tokens
    print("获取场景tokens...")
    scene_tokens = get_scene_tokens_from_pkl(pkl_data)
    
    # 准备参数
    args = []
    valid_scenes = []
    for scene_id, token_list in enumerate(scene_tokens):
        if len(token_list) > 0:
            valid_scenes.append((scene_id, len(token_list)))
            args.append((
                scene_id,
                token_list,
                voxel_size,
                np.array(point_cloud_range[:3]),
                np.array(occ_size)
            ))
    
    # 显示有效场景统计
    print(f"\n📋 发现的有效场景:")
    for scene_id, token_count in valid_scenes[:5]:  # 显示前5个
        print(f"  场景 {scene_id}: {token_count} 个tokens")
    if len(valid_scenes) > 5:
        print(f"  ... 还有 {len(valid_scenes) - 5} 个场景")
    
    # 测试模式
    test_mode = input(f"\n🎬 发现 {len(valid_scenes)} 个有效场景。是否先测试前1个场景？(y/N): ").lower().strip()
    if test_mode == 'y':
        args = args[:1]
        print(f"✅ 测试模式：只处理前 {len(args)} 个场景")
    
    
    # 单进程处理避免mayavi多进程问题
    for i, arg in enumerate(args):
        print(f"\n=== 开始渲染场景 {arg[0]} ({i+1}/{len(args)}) ===")
        try:
            render_one_scene(*arg)
            print(f"=== 场景 {arg[0]} 视频生成完成 ({i+1}/{len(args)}) ===\n")
        except Exception as e:
            print(f"❌ 场景 {arg[0]} 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    display.stop()