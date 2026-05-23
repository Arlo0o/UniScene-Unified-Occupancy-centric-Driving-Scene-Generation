#!/usr/bin/env python3
"""
修复版的NuPlan占用网格可视化脚本
彻底解决segmentation fault和客户端连接问题
"""

import os
import sys
import gc
import atexit
import tempfile

# ===== 关键环境变量设置 - 必须在导入任何图形库之前设置 =====
print("Setting up environment variables...")

# 修复缺失的环境变量，这是导致segmentation fault的关键原因
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

# ===== 现在可以安全导入图形库 =====
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
        # 尝试不使用虚拟显示
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
from pathlib import Path
import numpy as np
import glob
import imageio

# 配置
occ_dir = 'outputs/nuplan_occ_dit_eval/2025_09_11_15_32_19_ok/save_occ'
pkl_file = "/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_val.pkl"


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
            if key < len(ori_colormap):
                ori_colormap[key, :3] = value
        plt_plot.module_manager.scalar_lut_manager.lut.table = ori_colormap.astype(np.uint8)[:17]
    except Exception as e:
        print(f"Warning: Error setting colormap: {e}")

def draw_nusc_occupancy(voxels, vox_origin, voxel_size=0.2, grid=None, save_folder=None):
    """绘制NuScenes占用网格"""
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
    return draw(fov_voxels, voxel_size=voxel_size, vis_root=save_folder, idx=0)

def draw(voxels, voxel_size=0.2, vis_root=None, idx=0):
    """安全地绘制3D占用体素并保存图像"""
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
        save_file = f"{occ_dir}_vis"
        os.makedirs(save_file, exist_ok=True)
        filename = str(save_file) + f'/{vis_root[0]}_{idx}_bev.png'
        
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

def process_single_file(filepath):
    """处理单个文件"""
    try:
        print(f"\nProcessing: {os.path.basename(filepath)}")
        
        # 加载数据
        pred_voxels = np.load(filepath)
        print(f"Loaded shape: {pred_voxels.shape}")
        
        # 处理不同维度的数据
        vis_voxels = None
        if len(pred_voxels.shape) == 4:
            vis_voxels = pred_voxels[0]
        elif len(pred_voxels.shape) == 5:
            vis_voxels = pred_voxels[0][0]
        elif len(pred_voxels.shape) == 3:
            vis_voxels = pred_voxels
        else:
            print(f"Unsupported shape: {pred_voxels.shape}, skipping...")
            return False
        
        point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
        voxel_size = [0.5, 0.5, 0.5]

        print(f"Visualization shape: {vis_voxels.shape}")
        elements, counts = np.unique(vis_voxels, return_counts=True)
        print(f"Unique elements: {elements}, Counts: {counts}")
        
        # 处理可视化
        success = draw_nusc_occupancy(
            voxels=vis_voxels, 
            vox_origin=np.array(point_cloud_range[:3]),
            voxel_size=np.array(voxel_size),
            save_folder=filepath.split("/")[-1:],
        )
        
        return success
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=== NuPlan Occupancy Visualization (Fixed Version) ===")
    
    if not display_ok:
        print("Warning: Virtual display not available, proceeding anyway...")
    
    # 收集文件
    folder_path = f'{occ_dir}'
    files_to_rename = []
    print(f"Processing folder: {folder_path}")
    
    # 遍历文件夹及其子文件夹中所有文件
    for root, dirs, files in os.walk(folder_path):
        print(f"Scanning: {root}")
        for filename in files:
            if filename.endswith('.npy'):
                files_to_rename.append(os.path.join(root, filename))
    
    # 按文件名排序
    files_to_rename.sort()
    print(f"Found {len(files_to_rename)} .npy files to process")
    
    if len(files_to_rename) == 0:
        print("No .npy files found!")
        return
    
    # 处理设置
    BATCH_SIZE = 1  # 每次只处理1个文件，最安全
    processed_count = 0
    failed_count = 0
    
    try:
        for i, filepath in enumerate(files_to_rename):
            print(f"\n{'='*60}")
            print(f"Progress: [{i+1}/{len(files_to_rename)}]")
            
            success = process_single_file(filepath)
            
            if success:
                processed_count += 1
                print("✅ SUCCESS")
            else:
                failed_count += 1
                print("❌ FAILED")
            
            # 每处理BATCH_SIZE个文件后进行清理
            if (i + 1) % BATCH_SIZE == 0:
                print(f"Batch completed. Performing cleanup...")
                gc.collect()  # 强制垃圾回收
                
            # 如果失败太多，提前退出
            if failed_count > 5:
                print("Too many failures, stopping...")
                break
                
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 最终清理
        print(f"\n{'='*60}")
        print(f"Processing Summary:")
        print(f"  Successfully processed: {processed_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Total: {len(files_to_rename)}")
        print(f"{'='*60}")
        
        # 清理显示资源
        cleanup_display()
        gc.collect()

if __name__ == '__main__':
    main()
