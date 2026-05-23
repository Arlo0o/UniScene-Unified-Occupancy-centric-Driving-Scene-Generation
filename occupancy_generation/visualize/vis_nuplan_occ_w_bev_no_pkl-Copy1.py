from pyvirtualdisplay import Display
display = Display(visible=False, size=(1800, 1800))
display.start()

from mayavi import mlab
# mlab.options.offscreen = True

from pathlib import Path
import numpy as np
# from nuscenes.nuscenes import NuScenes
import os
import glob
import imageio
# import pickle  # 不再需要pkl文件
import gc  # 添加垃圾回收模块
from scipy import ndimage
# from PIL import Image, ImageDraw, ImageFont  # 将在函数内部导入


occ_dir ='outputs/occ_generation/save_occ'
vis_dir = f"/code/world_model/occ_for_combined_videos_nuplan"
# pkl_file = 'data/nuplan_mini_val_clip_infos.pkl'  # 不再需要pkl文件
os.makedirs(vis_dir, exist_ok=True)

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
    g_xx = np.arange(0, dims[0])
    g_yy = np.arange(0, dims[1])
    g_zz = np.arange(0, dims[2])
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])
    coords_grid = (coords_grid * resolution) + resolution / 2
    return coords_grid

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

def custom_colormap(plt_plot, colormap=classname_to_color):
    ori_colormap = plt_plot.module_manager.scalar_lut_manager.lut.table.to_array()
    for key, value in colormap.items():
        ori_colormap[key, :3] = value
    plt_plot.module_manager.scalar_lut_manager.lut.table = ori_colormap.astype(np.uint8)[:17]

def draw_nusc_occupancy(
    voxels,
    vox_origin,
    voxel_size=[0.5, 0.5, 0.5],
    grid=None,
    output_path=None,  # 修改为输出文件路径
    ):
    w, h, z = voxels.shape

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
    ]
    draw(fov_voxels, voxel_size=voxel_size, output_path=output_path)


_GLOBAL_FIGURE = None

def _get_global_figure():
    global _GLOBAL_FIGURE
    if _GLOBAL_FIGURE is None:
        _GLOBAL_FIGURE = mlab.figure(size=(1800, 1800), bgcolor=(1, 1, 1))
    return _GLOBAL_FIGURE

def draw(voxels, voxel_size=[0.5, 0.5, 0.5], output_path=None):
    figure = _get_global_figure()
    mlab.clf(figure)

    x = voxels[:, 1]  # x position of point
    y = voxels[:, 0]  # y position of point
    z = -voxels[:, 2]  # z position of point
    point_color = np.zeros(voxels.shape[0])

    for cls_index in range(16):
        class_point = voxels[:, 3] == cls_index
        point_color[class_point] = cls_index + 1 

    # 使用全局figure并关闭交互渲染
    figure.scene.disable_render = True  # 禁用交互式渲染以提高性能
    
    plt_plot = mlab.points3d(
        x,
        y,
        -z,
        point_color,
        scale_factor=0.8,
        scale_mode='vector',
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=17,
    )
    custom_colormap(plt_plot)
    
    # 确保输出目录存在
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 启用渲染并保存
    figure.scene.disable_render = False
    mlab.draw()
    mlab.savefig(filename=output_path)
    
    # 清理图元但不关闭全局figure，避免反复创建销毁
    mlab.clf(figure)
    gc.collect()
    
    print(f"Saved visualization to: {output_path}")

def merge_images_left_right(image_left_path, image_right_path, output_path, spacing=10):
    """
    将左右两张图片合并成一张图片
    参数:
    - image_left_path: 左侧图片路径
    - image_right_path: 右侧图片路径  
    - output_path: 输出合并图片路径
    - spacing: 两张图片之间的间距
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # 打开两张图片
    try:
        img_left = Image.open(image_left_path)
        img_right = Image.open(image_right_path)
    except Exception as e:
        print(f"❌ 无法打开图片: {e}")
        return False
    
    # 确保两张图片尺寸一致（以较小的为准）
    left_size = img_left.size
    right_size = img_right.size
    
    # 调整到相同尺寸
    target_size = (min(left_size[0], right_size[0]), min(left_size[1], right_size[1]))
    img_left = img_left.resize(target_size, Image.Resampling.LANCZOS)
    img_right = img_right.resize(target_size, Image.Resampling.LANCZOS)
    
    # 创建新的合并图片
    merged_width = target_size[0] * 2 + spacing
    merged_height = target_size[1]
    merged_img = Image.new('RGB', (merged_width, merged_height), color=(255, 255, 255))
    
    # 粘贴左侧图片
    merged_img.paste(img_left, (0, 0))
    
    # 粘贴右侧图片
    merged_img.paste(img_right, (target_size[0] + spacing, 0))
    
    # 添加文字标签
    try:
        draw = ImageDraw.Draw(merged_img)
        # 尝试使用默认字体，如果失败则使用基本字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            except:
                font = ImageFont.load_default()
        
        # 添加标签
        draw.text((target_size[0]//4, 20), "GT", fill=(0, 0, 0), font=font)
        draw.text((target_size[0] + spacing + target_size[0]//4, 20), "Pred", fill=(0, 0, 0), font=font)
        
        # 添加分隔线
        draw.line([(target_size[0], 0), (target_size[0], merged_height)], fill=(200, 200, 200), width=2)
        draw.line([(target_size[0] + spacing, 0), (target_size[0] + spacing, merged_height)], fill=(200, 200, 200), width=2)
        
    except Exception as e:
        print(f"⚠️ 添加标签失败，但图片合并成功: {e}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存合并后的图片
    merged_img.save(output_path, quality=95)
    
    # 清理临时文件
    try:
        os.remove(image_left_path)
        os.remove(image_right_path)
        print(f"🗑️ 已删除临时文件: {os.path.basename(image_left_path)}, {os.path.basename(image_right_path)}")
    except:
        pass
    
    print(f"✅ 合并图片已保存: {output_path}")
    return True

if __name__ == '__main__':
    # 配置路径
    # occ_root = 'outputs/occ_generation/save_occ'
    # pkl_file = '/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_val.pkl'
    vis_root = vis_dir
    bev_root = '/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200'
    os.makedirs(vis_root, exist_ok=True)
    
    # 不再需要pkl文件，直接从文件系统获取tokens
    
    # 定义点云范围
    point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
    occ_size = [200, 200, 16]
    
    # 计算体素尺寸
    voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
    voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
    voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
    voxel_size = [voxel_x, voxel_y, voxel_z]
    
    # 直接从occ_dir扫描获取所有tokens
    print(f"🔍 开始扫描目录: {occ_dir}")
    
    # 扫描所有npy文件
    # npy_files = glob.glob(os.path.join(occ_dir, "*.npy"))
    specific_names = [
    "0009ee7de70853bc_gt_occ.npy",
    "30cd1611dbf357e1_gt_occ.npy",
    "7d6192cd8d445a3c_gt_occ.npy",
    "356761c2243952be_gt_occ.npy",
    "b7de0f352bb35dfd_gt_occ.npy",
    "1461c4442fd95593_gt_occ.npy",
    "22da0658305959c2_gt_occ.npy",
    "3993ce18984f518b_gt_occ.npy",
    "472a42b6f9da56d3_gt_occ.npy",
    "5979c08182ab5643_gt_occ.npy",
    "63b4e06179e25dfe_gt_occ.npy",
    "5543b81c45825324_gt_occ.npy",
    "c9ee9387f55050cd_gt_occ.npy",
    "8aab118f844152ba_gt_occ.npy",
    "5ec0546fbc5a533a_gt_occ.npy",
    "d313c0362f365011_gt_occ.npy",
    "6ef53aecb73e5a86_gt_occ.npy",
    ]
    npy_files = [os.path.join(occ_dir, name) for name in specific_names if os.path.isfile(os.path.join(occ_dir, name))]

    print(f"📁 找到 {len(npy_files)} 个npy文件")
    
    # 提取所有唯一的tokens（从文件名中提取）
    tokens_set = set()
    for file_path in npy_files:
        filename = os.path.basename(file_path)  # 例如: 3532b3b3091852ab_input_occs.npy
        # 提取token：从文件名的开头到第一个下划线之前
        token = filename.split('_')[0]
        tokens_set.add(token)
    
    all_tokens = list(tokens_set)
    all_tokens.sort()  # 按字母顺序排序以便一致性
    
    print(f"📋 总共找到 {len(all_tokens)} 个唯一tokens")
    print(f"🎯 Tokens: {all_tokens[:5]}..." if len(all_tokens) > 5 else f"🎯 Tokens: {all_tokens}")
    
    # 处理每个token
    for i, sample_token in enumerate(all_tokens):
        print(f'Processing token {i+1}/{len(all_tokens)}: {sample_token}')
        
        # 每处理10个token强制清理一次内存
        if i % 10 == 0 and i > 0:
            gc.collect()
            print(f"🔧 Memory cleanup at token {i}")
        
        # 构造具体的文件路径（基于观察到的文件模式）
        gt_path = os.path.join(occ_dir, f'{sample_token}_gt.npy')
        pred_path = os.path.join(occ_dir, f'{sample_token}_occworld.npy')
        
        # 验证文件是否存在
        gt_exists = os.path.exists(gt_path)
        pred_exists = os.path.exists(pred_path)
        
        if not gt_exists or not pred_exists:
            print(f'❌ Token {sample_token} files missing:')
            print(f'   GT ({gt_exists}): {os.path.basename(gt_path)}')
            print(f'   Pred ({pred_exists}): {os.path.basename(pred_path)}')
            continue
            
        # 打印找到的文件路径
        print(f'✅ Token {sample_token} files found:')
        print(f'   GT: {os.path.basename(gt_path)}')
        print(f'   Pred: {os.path.basename(pred_path)}')
        
        # 加载OCC数据
        gt_voxels = np.load(gt_path)
        pred_voxels = np.load(pred_path)
        
        # 处理数据形状
        if len(gt_voxels.shape) == 5:
            gt_voxels = gt_voxels[0, 0]
        elif len(gt_voxels.shape) == 4:
            gt_voxels = gt_voxels[0]
            
        if len(pred_voxels.shape) == 5:
            pred_voxels = pred_voxels[0, 0]
        elif len(pred_voxels.shape) == 4:
            pred_voxels = pred_voxels[0]
        
        # 加载BEV数据并应用替换
        bev_data = load_bev_data(sample_token, bev_root)
        
        if bev_data is not None:
            print(f"🗺️ BEV数据已加载: {bev_data.shape}")
            print(f"  📊 BEV替换前 - GT类别: {np.unique(gt_voxels)}")
            print(f"  📊 BEV替换前 - Pred类别: {np.unique(pred_voxels)}")
            
            # 使用同一个BEV数据替换GT和Pred
            gt_voxels = replace_occ_grid_with_bev_nuplan(gt_voxels, bev_data.copy())
            pred_voxels = replace_occ_grid_with_bev_nuplan(pred_voxels, bev_data.copy())
            
            print(f"  📊 BEV替换后 - GT类别: {np.unique(gt_voxels)}")
            print(f"  📊 BEV替换后 - Pred类别: {np.unique(pred_voxels)}")
        else:
            print("⚠️ 未找到BEV数据，使用原始占用网格")
        
        # 构建输出路径（先分别保存为临时文件）
        gt_temp_path = os.path.join(vis_root, f'{sample_token}_gt_temp.png')
        pred_temp_path = os.path.join(vis_root, f'{sample_token}_pred_temp.png')
        combined_output_path = os.path.join(vis_root, f'{sample_token}_combined_bev.png')
        
        # 执行可视化（分别渲染GT和Pred）
        draw_nusc_occupancy(
            voxels=gt_voxels, 
            vox_origin=np.array(point_cloud_range[:3]),
            voxel_size=voxel_size,
            grid=np.array(occ_size),
            output_path=gt_temp_path,
        )
        
        draw_nusc_occupancy(
            voxels=pred_voxels, 
            vox_origin=np.array(point_cloud_range[:3]),
            voxel_size=voxel_size,
            grid=np.array(occ_size),
            output_path=pred_temp_path,
        )
        
        # 合并两张图片
        print(f"🖼️ 开始合并图片: {os.path.basename(gt_temp_path)} + {os.path.basename(pred_temp_path)}")
        success = merge_images_left_right(gt_temp_path, pred_temp_path, combined_output_path)
        if success:
            print(f"✅ 图片合并成功: {os.path.basename(combined_output_path)}")
        else:
            print(f"❌ 图片合并失败: {sample_token}")
        
        print(f"Processed token {i+1}/{len(all_tokens)}: {sample_token}")
    
    print("🎉 All tokens processed successfully!")
    print(f"📈 处理统计:")
    print(f"   - 总tokens: {len(all_tokens)}")
    print(f"   - 成功处理: {len(all_tokens)} tokens") 
    print(f"   - 输出目录: {vis_root}")
    
    # 最终清理：仅关闭一次全局figure
    try:
        if '_GLOBAL_FIGURE' in globals() and _GLOBAL_FIGURE is not None:
            mlab.close(_GLOBAL_FIGURE)
    except Exception:
        pass
    gc.collect()
    print("🔧 Final memory cleanup completed")

