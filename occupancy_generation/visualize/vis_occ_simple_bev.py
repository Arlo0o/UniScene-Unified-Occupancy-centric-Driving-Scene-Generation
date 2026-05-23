#!/usr/bin/env python3
"""
简化版本 - 基于工作版本的核心逻辑
只处理单个文件，避免复杂的多进程和视频生成
"""

import os
import sys
import gc
import atexit

# ===== 完全复制工作版本的环境设置 =====
print("Setting up environment variables...")

if "HOME" not in os.environ:
    os.environ["HOME"] = "/tmp"
    print(f"Set HOME to: {os.environ['HOME']}")

if "XDG_RUNTIME_DIR" not in os.environ:
    os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"
    os.makedirs("/tmp/runtime-root", exist_ok=True)
    print(f"Set XDG_RUNTIME_DIR to: {os.environ['XDG_RUNTIME_DIR']}")

# Qt和显示相关设置
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ":99"
os.environ["QT_LOGGING_RULES"] = "*.debug=false"
os.environ["QT_DEBUG_PLUGINS"] = "0"

# VTK和Mayavi设置
os.environ["VTK_USE_X"] = "0"
os.environ["MAYAVI_USE_ENVISAGE"] = "0"
os.environ["ETS_TOOLKIT"] = "null"
os.environ["MPLBACKEND"] = "Agg"

# 设置临时目录权限
os.environ["TMPDIR"] = "/tmp"

print("Environment setup completed.")

# ===== 导入库 =====
try:
    from pyvirtualdisplay import Display
    print("pyvirtualdisplay imported successfully")
except ImportError as e:
    print(f"Error importing pyvirtualdisplay: {e}")
    sys.exit(1)

# 全局显示对象
display = None

def initialize_display():
    """安全初始化虚拟显示"""
    global display
    try:
        if display is None:
            print("Initializing virtual display...")
            display = Display(visible=False, size=(900, 900), backend='xvfb')
            display.start()
            print("Virtual display started successfully")
        return True
    except Exception as e:
        print(f"Warning: Could not start virtual display: {e}")
        return False

def cleanup_display():
    """清理显示资源"""
    global display
    try:
        if display is not None:
            display.stop()
            display = None
            print("Virtual display stopped")
    except Exception as e:
        print(f"Warning: Error stopping display: {e}")

# 注册退出时的清理函数
atexit.register(cleanup_display)

# 初始化显示
display_ok = initialize_display()

# 导入mayavi (在显示初始化之后)
try:
    from mayavi import mlab
    # 强制设置离屏渲染
    mlab.options.offscreen = True
    print(f"Mayavi imported, offscreen mode: {mlab.options.offscreen}")
except ImportError as e:
    print(f"Error importing mayavi: {e}")
    sys.exit(1)

# 其他导入
import numpy as np
import glob
import re
from collections import defaultdict
import cv2

# 配置
occ_root = 'outputs/nuplan_occ_dit_eval/2025_09_11_15_32_19_ok/save_occ'
bev_root = '/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200'
output_dir = f"{occ_root}_vis_simple"
os.makedirs(output_dir, exist_ok=True)

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

def custom_colormap(plt_plot, colormap=classname_to_color):
    """自定义颜色映射"""
    try:
        ori_colormap = plt_plot.module_manager.scalar_lut_manager.lut.table.to_array()
        for key, value in colormap.items():
            ori_colormap[key, :3] = value
        plt_plot.module_manager.scalar_lut_manager.lut.table = ori_colormap.astype(np.uint8)[:17]
    except Exception as e:
        print(f"Warning: Error setting colormap: {e}")

def replace_occ_grid_with_bev_nuplan(input_occ_data, bevlayout, driva_area_idx=1, bev_replace_idx=[1,15],
                                    occ_replace_new_idx=[12, 14]):
    """用BEV数据替换occupancy grid中的特定区域"""
    # 处理road_divider和lane_divider的mask
    roal_divider_mask = bevlayout[15, :, :].astype(np.uint8)
    lane_divider_mask = bevlayout[17, :, :].astype(np.uint8)

    # 对mask进行膨胀操作
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

    # 遍历每个像素点进行替换
    for x in range(x_max):
        for y in range(y_max):
            for i in range(n):
                if bev_replace_mask[i][x, y]:
                    occupancy_data = input_occ_data[x, y, :]
                    if driva_area_idx in occupancy_data:
                        max_11_index = np.where(occupancy_data == driva_area_idx)
                        output_occ[x, y, max_11_index] = occ_replace_new_idx[i]
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

def draw_nusc_occupancy(voxels, vox_origin, voxel_size=0.2, save_name="test"):
    """绘制NuScenes占用网格 - 完全复制工作版本的逻辑"""
    w, h, z = voxels.shape

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    grid_coords[grid_coords[:, 3] == 15, 3] = 12

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
    ]
    return draw(fov_voxels, voxel_size=voxel_size, save_name=save_name)

def draw(voxels, voxel_size=0.2, save_name="test"):
    """安全地绘制3D占用体素并保存图像 - 基于工作版本"""
    figure = None
    success = False
    
    try:
        print(f"Drawing {len(voxels)} voxels...")
        
        # 检查数据有效性
        if len(voxels) == 0:
            print("No voxels to draw")
            return False
            
        x = voxels[:, 1]  # x position of point
        y = voxels[:, 0]  # y position of point
        z = -voxels[:, 2]  # z position of point
        point_color = np.zeros(voxels.shape[0])

        for cls_index in range(16):
            class_point = voxels[:, 3] == cls_index
            point_color[class_point] = cls_index+1 

        # 创建新的图形
        print("Creating mayavi figure...")
        figure = mlab.figure(size=(900, 900), bgcolor=(1, 1, 1))
        
        print("Creating 3D points...")
        plt_plot = mlab.points3d(
            x, y, -z, point_color,
            scale_factor=0.5,
            scale_mode='vector',
            mode="cube",
            opacity=1.0,
            vmin=1,
            vmax=17,
        )
        
        # 应用自定义颜色映射
        custom_colormap(plt_plot)

        # 保存文件
        filename = os.path.join(output_dir, f'{save_name}.png')
        
        print(f"Rendering and saving to: {filename}")
        # 渲染和保存
        mlab.draw()
        mlab.savefig(filename=filename)
        print(f"Successfully saved: {filename}")
        success = True
        
    except Exception as e:
        print(f"Error in draw function: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 确保资源得到清理
        try:
            if figure is not None:
                mlab.close(figure)
            mlab.clf()  # 清除当前图形
            gc.collect()  # 强制垃圾回收
            print("Resources cleaned up")
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
    
    return success

def parse_occ_files(occ_root):
    """解析occupancy文件，按场景组织数据"""
    scene_data = defaultdict(lambda: {'gt': [], 'pred': []})
    
    # 获取所有.npy文件
    npy_files = glob.glob(os.path.join(occ_root, '*.npy'))
    
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        # 解析文件名格式：序号_token_类型.npy
        match = re.match(r'(\d+)_([a-f0-9]+)_(gt_occ|pred)\.npy', filename)
        if match:
            scene_id = int(match.group(1))
            token = match.group(2)
            file_type = match.group(3)
            
            # 根据类型分类
            if file_type == 'gt_occ':
                scene_data[scene_id]['gt'].append((token, file_path))
            elif file_type == 'pred':
                scene_data[scene_id]['pred'].append((token, file_path))
    
    # 按token排序每个场景的数据
    for scene_id in scene_data:
        scene_data[scene_id]['gt'].sort(key=lambda x: x[0])
        scene_data[scene_id]['pred'].sort(key=lambda x: x[0])
    
    return scene_data

def process_single_scene(scene_id, gt_files, pred_files):
    """处理单个场景 - 只生成图片，不生成视频"""
    try:
        print(f"\n=== 处理场景 {scene_id} ===")
        
        if len(gt_files) == 0 or len(pred_files) == 0:
            print(f"场景 {scene_id} 没有有效的GT/Pred文件对")
            return False
            
        # 只处理第一个文件对
        gt_token, gt_path = gt_files[0]
        pred_token, pred_path = pred_files[0]
        
        print(f"处理token: {gt_token}")
        
        # 加载数据
        gt_voxels_raw = np.load(gt_path)
        pred_voxels_raw = np.load(pred_path)
        
        # 处理数据形状
        if len(gt_voxels_raw.shape) == 5:
            gt_voxels = gt_voxels_raw[0, -1]  # 取第一个batch的最后一帧
        else:
            gt_voxels = gt_voxels_raw
            
        if len(pred_voxels_raw.shape) == 5:
            pred_voxels = pred_voxels_raw[0, -1]  # 取第一个batch的最后一帧
        else:
            pred_voxels = pred_voxels_raw
        
        print(f"GT形状: {gt_voxels.shape}, Pred形状: {pred_voxels.shape}")
        
        # 加载并处理BEV数据
        bev_data = load_bev_data(gt_token, bev_root)
        if bev_data is not None:
            print(f"BEV数据形状: {bev_data.shape}")
            # 用BEV数据替换GT和Pred中的occupancy
            gt_voxels = replace_occ_grid_with_bev_nuplan(gt_voxels, bev_data)
            pred_voxels = replace_occ_grid_with_bev_nuplan(pred_voxels, bev_data)
            print("已应用BEV数据替换")
        else:
            print(f"警告: 未找到token {gt_token}的BEV数据，使用原始occupancy数据")
        
        # 设置参数
        point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
        voxel_size = [0.5, 0.5, 0.5]

        print(f"GT唯一值: {np.unique(gt_voxels)}")
        print(f"Pred唯一值: {np.unique(pred_voxels)}")
        
        # 处理GT可视化
        print("处理GT可视化...")
        gt_success = draw_nusc_occupancy(
            voxels=gt_voxels, 
            vox_origin=np.array(point_cloud_range[:3]),
            voxel_size=np.array(voxel_size),
            save_name=f"scene_{scene_id}_gt",
        )
        
        # 处理Pred可视化
        print("处理Pred可视化...")
        pred_success = draw_nusc_occupancy(
            voxels=pred_voxels, 
            vox_origin=np.array(point_cloud_range[:3]),
            voxel_size=np.array(voxel_size),
            save_name=f"scene_{scene_id}_pred",
        )
        
        success = gt_success and pred_success
        print(f"场景 {scene_id} 处理{'成功' if success else '失败'}")
        return success
        
    except Exception as e:
        print(f"Error processing scene {scene_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=== 简化版NuPlan Occupancy可视化（集成BEV） ===")
    
    if not display_ok:
        print("Warning: Virtual display not available, proceeding anyway...")
    
    try:
        # 解析occupancy文件
        print("解析occupancy文件...")
        scene_data = parse_occ_files(occ_root)
        print(f"找到 {len(scene_data)} 个场景")
        
        # 只处理前3个场景进行测试
        test_scenes = sorted(list(scene_data.keys()))[:3]
        print(f"测试前 {len(test_scenes)} 个场景: {test_scenes}")
        
        success_count = 0
        for scene_id in test_scenes:
            gt_files = scene_data[scene_id]['gt']
            pred_files = scene_data[scene_id]['pred']
            
            if process_single_scene(scene_id, gt_files, pred_files):
                success_count += 1
        
        print(f"\n=== 完成！成功处理 {success_count}/{len(test_scenes)} 个场景 ===")
        print(f"输出目录: {output_dir}")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 最终清理
        cleanup_display()
        gc.collect()

if __name__ == "__main__":
    main()
