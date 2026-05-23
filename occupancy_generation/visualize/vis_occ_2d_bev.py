#!/usr/bin/env python3
"""
基于matplotlib的2D可视化方案 - 避免Mayavi的segfault问题
生成BEV视角的occupancy对比图和视频
"""

import os
import numpy as np
import glob
import re
from collections import defaultdict
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import imageio
from tqdm import tqdm

# 配置
occ_root = 'outputs/nuplan_occ_dit_eval/2025_09_11_15_32_19_ok/save_occ'
bev_root = '/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200'
output_root = "outputs/visualize/vis_out_2d"
os.makedirs(output_root, exist_ok=True)

# 类别颜色映射 (RGB, 归一化到0-1)
classname_to_color = {
    0: (1.0, 1.0, 1.0),      # free, white
    1: (0.0, 0.686, 0.0),    # other-ground, green
    2: (1.0, 0.620, 0.0),    # vehicle, orange
    3: (0.863, 0.078, 0.235), # bicycle, crimson red
    4: (0.0, 0.0, 0.902),    # pedestrian, blue
    5: (0.184, 0.310, 0.310), # traffic-cone, dark slate gray
    6: (0.439, 0.502, 0.565), # barrier, slate gray
    7: (1.0, 0.784, 0.0),    # construction-zones, yellow
    8: (0.871, 0.722, 0.051), # generic-object, goldenrod
    12: (0.0, 0.812, 0.749), # road, cyan-green
    14: (0.588, 0.941, 0.314), # road-line, lime green
}

def parse_occ_files(occ_root):
    """解析occupancy文件，按场景组织数据"""
    scene_data = defaultdict(lambda: {'gt': [], 'pred': []})
    
    npy_files = glob.glob(os.path.join(occ_root, '*.npy'))
    
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        match = re.match(r'(\d+)_([a-f0-9]+)_(gt_occ|pred)\.npy', filename)
        if match:
            scene_id = int(match.group(1))
            token = match.group(2)
            file_type = match.group(3)
            
            if file_type == 'gt_occ':
                scene_data[scene_id]['gt'].append((token, file_path))
            elif file_type == 'pred':
                scene_data[scene_id]['pred'].append((token, file_path))
    
    for scene_id in scene_data:
        scene_data[scene_id]['gt'].sort(key=lambda x: x[0])
        scene_data[scene_id]['pred'].sort(key=lambda x: x[0])
    
    return scene_data

def load_bev_data(token, bev_root):
    """加载BEV数据并进行预处理"""
    bev_file_path = os.path.join(bev_root, f'{token}.npz')
    if not os.path.exists(bev_file_path):
        return None
    
    bev_data = np.load(bev_file_path)['gt_bev_masks']  # [18, 200, 200]
    
    # 合并特定层
    layer_to_merge = [0, 1, 3, 4, 5, 6, 7]
    bev_data[1, :, :] = np.any(bev_data[layer_to_merge, :, :], axis=0).astype(int)
    
    return bev_data

def replace_occ_grid_with_bev_nuplan(input_occ_data, bevlayout, driva_area_idx=1, bev_replace_idx=[1,15],
                                    occ_replace_new_idx=[12, 14]):
    """用BEV数据替换occupancy grid中的特定区域"""
    roal_divider_mask = bevlayout[15, :, :].astype(np.uint8)
    lane_divider_mask = bevlayout[17, :, :].astype(np.uint8)

    roal_divider_mask = cv2.dilate(roal_divider_mask, np.ones((1, 1), np.uint8))
    lane_divider_mask = cv2.dilate(lane_divider_mask, np.ones((1, 1), np.uint8))

    bevlayout[15, :, :] = roal_divider_mask.astype(bool)
    bevlayout[17, :, :] = lane_divider_mask.astype(bool)

    n = len(bev_replace_idx)
    x_max, y_max = input_occ_data.shape[0], input_occ_data.shape[1]
    output_occ = input_occ_data.copy()
    bev_replace_mask = []
    for i in range(n):
        bev_replace_mask.append(bevlayout[bev_replace_idx[i]] == 1)

    for x in range(x_max):
        for y in range(y_max):
            for i in range(n):
                if bev_replace_mask[i][x, y]:
                    occupancy_data = input_occ_data[x, y, :]
                    if driva_area_idx in occupancy_data:
                        max_11_index = np.where(occupancy_data == driva_area_idx)
                        output_occ[x, y, max_11_index] = occ_replace_new_idx[i]
    return output_occ

def create_bev_image(occ_data, title="Occupancy BEV"):
    """创建BEV视角的occupancy图像"""
    # 投影到BEV：取每个(x,y)位置的最高非零值
    bev_map = np.zeros((occ_data.shape[0], occ_data.shape[1]), dtype=np.int32)
    
    for x in range(occ_data.shape[0]):
        for y in range(occ_data.shape[1]):
            # 从上到下寻找第一个非零值
            for z in range(occ_data.shape[2]-1, -1, -1):
                if occ_data[x, y, z] != 0:
                    bev_map[x, y] = occ_data[x, y, z]
                    break
    
    # 创建RGB图像
    rgb_image = np.ones((bev_map.shape[0], bev_map.shape[1], 3), dtype=np.float32)
    
    for class_id, color in classname_to_color.items():
        mask = bev_map == class_id
        if np.any(mask):
            rgb_image[mask] = color
    
    return rgb_image, bev_map

def create_comparison_plot(gt_rgb, pred_rgb, gt_map, pred_map, scene_id, frame_idx, save_path=None):
    """创建GT和Pred的对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'Scene {scene_id} - Frame {frame_idx}', fontsize=16)
    
    # GT图像
    axes[0, 0].imshow(gt_rgb)
    axes[0, 0].set_title('Ground Truth', fontsize=14)
    axes[0, 0].axis('off')
    
    # Pred图像
    axes[0, 1].imshow(pred_rgb)
    axes[0, 1].set_title('Prediction', fontsize=14)
    axes[0, 1].axis('off')
    
    # GT类别统计
    gt_unique, gt_counts = np.unique(gt_map[gt_map > 0], return_counts=True)
    axes[1, 0].bar(gt_unique, gt_counts, color=[classname_to_color.get(c, (0.5, 0.5, 0.5)) for c in gt_unique])
    axes[1, 0].set_title('GT Class Distribution', fontsize=14)
    axes[1, 0].set_xlabel('Class ID')
    axes[1, 0].set_ylabel('Pixel Count')
    
    # Pred类别统计
    pred_unique, pred_counts = np.unique(pred_map[pred_map > 0], return_counts=True)
    axes[1, 1].bar(pred_unique, pred_counts, color=[classname_to_color.get(c, (0.5, 0.5, 0.5)) for c in pred_unique])
    axes[1, 1].set_title('Pred Class Distribution', fontsize=14)
    axes[1, 1].set_xlabel('Class ID')
    axes[1, 1].set_ylabel('Pixel Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"保存对比图: {save_path}")
    
    # 转换为numpy数组用于视频
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return img_array

def process_scene(scene_id, gt_files, pred_files):
    """处理单个场景，生成图像和视频"""
    print(f"\n=== 处理场景 {scene_id} ===")
    
    if len(gt_files) == 0 or len(pred_files) == 0:
        print(f"场景 {scene_id} 没有有效的GT/Pred文件对")
        return False
    
    scene_output_dir = os.path.join(output_root, f"scene_{scene_id}")
    os.makedirs(scene_output_dir, exist_ok=True)
    
    # 处理所有帧
    min_frames = min(len(gt_files), len(pred_files))
    video_frames = []
    
    for i in range(min_frames):
        gt_token, gt_path = gt_files[i]
        pred_token, pred_path = pred_files[i]
        
        print(f"处理帧 {i+1}/{min_frames}: {gt_token}")
        
        try:
            # 加载数据
            gt_voxels_raw = np.load(gt_path)
            pred_voxels_raw = np.load(pred_path)
            
            # 处理数据形状
            if len(gt_voxels_raw.shape) == 5:
                gt_voxels = gt_voxels_raw[0, -1]
            else:
                gt_voxels = gt_voxels_raw
                
            if len(pred_voxels_raw.shape) == 5:
                pred_voxels = pred_voxels_raw[0, -1]
            else:
                pred_voxels = pred_voxels_raw
            
            # 加载并应用BEV数据
            bev_data = load_bev_data(gt_token, bev_root)
            if bev_data is not None:
                gt_voxels = replace_occ_grid_with_bev_nuplan(gt_voxels, bev_data.copy())
                pred_voxels = replace_occ_grid_with_bev_nuplan(pred_voxels, bev_data.copy())
            
            # 创建BEV图像
            gt_rgb, gt_map = create_bev_image(gt_voxels, f"GT - Scene {scene_id}")
            pred_rgb, pred_map = create_bev_image(pred_voxels, f"Pred - Scene {scene_id}")
            
            # 创建对比图
            img_path = os.path.join(scene_output_dir, f"frame_{i:04d}.png")
            frame_img = create_comparison_plot(gt_rgb, pred_rgb, gt_map, pred_map, scene_id, i, img_path)
            video_frames.append(frame_img)
            
        except Exception as e:
            print(f"处理帧 {i} 时出错: {e}")
            continue
    
    # 生成视频
    if len(video_frames) > 0:
        video_path = os.path.join(scene_output_dir, f"scene_{scene_id}_comparison.mp4")
        print(f"生成视频: {video_path}")
        
        with imageio.get_writer(video_path, fps=2, quality=8) as writer:
            for frame in video_frames:
                writer.append_data(frame)
        
        print(f"✅ 场景 {scene_id} 处理完成")
        print(f"   - 生成了 {len(video_frames)} 帧图像")
        print(f"   - 保存路径: {scene_output_dir}")
        return True
    else:
        print(f"❌ 场景 {scene_id} 处理失败")
        return False

def main():
    """主函数"""
    print("=== NuPlan Occupancy 2D可视化 (BEV视角) ===")
    
    # 解析occupancy文件
    print("解析occupancy文件...")
    scene_data = parse_occ_files(occ_root)
    print(f"找到 {len(scene_data)} 个场景")
    
    # 询问处理模式
    test_mode = input(f"发现 {len(scene_data)} 个有效场景。是否只处理前5个场景进行测试？(y/N): ").lower().strip()
    
    if test_mode == 'y':
        test_scenes = sorted(list(scene_data.keys()))[:5]
        print(f"✅ 测试模式：处理前 {len(test_scenes)} 个场景")
    else:
        test_scenes = sorted(list(scene_data.keys()))
        print(f"✅ 完整模式：处理所有 {len(test_scenes)} 个场景")
    
    # 处理场景
    success_count = 0
    with tqdm(test_scenes, desc="处理场景") as pbar:
        for scene_id in pbar:
            pbar.set_description(f"处理场景 {scene_id}")
            
            gt_files = scene_data[scene_id]['gt']
            pred_files = scene_data[scene_id]['pred']
            
            if process_scene(scene_id, gt_files, pred_files):
                success_count += 1
            
            pbar.set_postfix({"成功": f"{success_count}/{len(test_scenes)}"})
    
    print(f"\n🎉 处理完成！")
    print(f"   成功处理: {success_count}/{len(test_scenes)} 个场景")
    print(f"   输出目录: {output_root}")
    
    # 显示生成的文件
    print(f"\n📁 生成的文件:")
    for scene_id in sorted(test_scenes)[:3]:  # 显示前3个场景的输出
        scene_dir = os.path.join(output_root, f"scene_{scene_id}")
        if os.path.exists(scene_dir):
            files = os.listdir(scene_dir)
            print(f"   场景 {scene_id}: {len([f for f in files if f.endswith('.png')])} 张图片, {len([f for f in files if f.endswith('.mp4')])} 个视频")

if __name__ == "__main__":
    main()
