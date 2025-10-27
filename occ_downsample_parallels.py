import numpy as np
from scipy.ndimage import zoom
import yaml
from numba import njit
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import numba as nb
import copy
import pickle
from multiprocessing import Pool, cpu_count
import functools



# -------------------- 核心处理函数 --------------------
@njit
def get_max_occurrence_label(block):
    counts = np.bincount(block)
    if len(counts) < 2:
        return 0
    return np.argmax(counts[1:]) + 1

@njit
def resample(labels, quantize_size):
    original_shape = labels.shape
    result = np.zeros(quantize_size, dtype=labels.dtype)
    
    z_step = original_shape[0] // quantize_size[0]
    y_step = original_shape[1] // quantize_size[1]
    x_step = original_shape[2] // quantize_size[2]
    
    for i in range(quantize_size[0]):
        for j in range(quantize_size[1]):
            for k in range(quantize_size[2]):
                z_start = i * z_step
                z_end = (i+1) * z_step
                y_start = j * y_step
                y_end = (j+1) * y_step
                x_start = k * x_step
                x_end = (k+1) * x_step
                
                block = labels[z_start:z_end, y_start:y_end, x_start:x_end]
                result[i,j,k] = get_max_occurrence_label(block.flatten())
    return result

# -------------------- 数据加载与转换 --------------------
def voxel2world(voxel, voxel_size=(0.125, 0.125, 0.125), pc_range=(-50., -50., -5.)):
    return voxel * np.array(voxel_size) + np.array(pc_range)

def world2voxel(world, voxel_size=(0.125, 0.125, 0.125), pc_range=(-50., -50., -5.)):
    return (world - np.array(pc_range)) / np.array(voxel_size)

@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True)
def nb_process_label(processed_label, sorted_pairs):
    current_index = sorted_pairs[0, :3]
    counter = np.zeros(256, dtype=np.uint16)
    for row in sorted_pairs:
        if not np.array_equal(row[:3], current_index):
            processed_label[current_index[0], current_index[1], current_index[2]] = np.argmax(counter)
            counter.fill(0)
            current_index = row[:3]
        counter[row[3]] += 1
    processed_label[current_index[0], current_index[1], current_index[2]] = np.argmax(counter)
    return processed_label

def load_occ_gt(occ_path, grid_size=(800,800,64), voxel_size=(0.125,0.125,0.125)):
    data = np.load(occ_path)['occ']
    data[:,3] += 1  # 调整类别索引
    
    # 坐标转换
    world_coords = voxel2world(data[:,:3] + 0.5, voxel_size)
    voxel_coords = np.clip(world2voxel(world_coords, voxel_size), 0, np.array(grid_size)-1)
    
    # 创建标签矩阵
    label_matrix = np.zeros(grid_size, dtype=np.uint8)
    sorted_indices = np.lexsort((voxel_coords[:,2], voxel_coords[:,1], voxel_coords[:,0]))
    sorted_data = np.hstack((voxel_coords[sorted_indices], data[sorted_indices, 3:4]))
    return nb_process_label(label_matrix, sorted_data.astype(np.int64))

# -------------------- 并行处理框架 --------------------
def generate_save_path(input_path, save_base, quant_size):
    """生成输出文件路径"""
    rel_path = os.path.relpath(input_path, start=os.path.dirname(input_path))
    base_name = os.path.splitext(rel_path)[0]
    return os.path.join(
        save_base,
        f"nuplan_quantized_{quant_size[0]}_{quant_size[1]}_{quant_size[2]}",
        base_name + ".npy"
    )

class QuantizationProcessor:
    def __init__(self, config_path, pkl_path, data_dir, output_dir, target_size, origin_size, voxel_size, method, workers):
        # 初始化配置
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.label_map = np.array(list(config['learning_map'].values()), dtype=np.uint8)
        
        # 获取文件列表并过滤已处理
        self.file_list = self._discover_files(pkl_path=pkl_path, data_dir=data_dir)
        self.target_size = tuple(target_size)
        self.origin_size = tuple(origin_size)
        self.voxel_size = tuple(voxel_size)
        self.output_dir = output_dir
        self.method = method
        self.workers = workers if workers > 0 else cpu_count()
        
        # 预过滤已处理文件
        self._filter_processed_files()
        
    def _discover_files(self, pkl_path, data_dir):
        """发现所有需要处理的npz文件"""
        with open(pkl_path, 'rb') as f:
            infos = pickle.load(f)['infos']
        return [os.path.join(data_dir, f"{info['token']}/{info['token']}.npz") for info in infos]
    
    def _filter_processed_files(self):
        """过滤已处理的文件"""
        valid_files = []
        for fpath in self.file_list:
            save_path = generate_save_path(fpath, self.output_dir, self.target_size)
            if not os.path.exists(save_path):
                valid_files.append(fpath)
        print(f"原始文件数: {len(self.file_list)} | 待处理: {len(valid_files)}")
        self.file_list = valid_files
    
    def _process_single(self, file_path):
        """处理单个文件"""
        save_path = generate_save_path(file_path, self.output_dir, self.target_size)
        if os.path.exists(save_path):
            return "skipped"
        
        try:
            # 加载并转换标签
            orig_labels = load_occ_gt(file_path, grid_size=self.origin_size, voxel_size=self.voxel_size)
            mapped_labels = self.label_map[orig_labels]
            
            # 降采样
            if self.method == 'max':
                resampled = resample(mapped_labels, self.target_size)
            elif self.method == 'nearest':
                scales = np.array(self.target_size) / np.array(orig_labels.shape)
                resampled = zoom(mapped_labels, scales, order=0)
            else:
                raise ValueError(f"未知方法: {self.method}")
            
            # 保存结果
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, resampled)
            return "success"
        except Exception as e:
            print(f"处理失败 {file_path}: {str(e)}")
            return "failed"
    
    def run(self):
        """启动并行处理"""
        if not self.file_list:
            print("没有需要处理的文件")
            return
        
        processor = functools.partial(
            self._process_single
        )
        
        with Pool(processes=self.workers) as pool:
            results = []
            with tqdm(total=len(self.file_list), desc="处理进度") as pbar:
                for status in pool.imap_unordered(processor, self.file_list):
                    results.append(status)
                    pbar.update()
        
        # 统计结果
        stats = {
            'total': len(results),
            'success': results.count('success'),
            'skipped': results.count('skipped'),
            'failed': results.count('failed')
        }
        print(
            f"处理完成 | 成功: {stats['success']} "
            f"跳过: {stats['skipped']} "
            f"失败: {stats['failed']}"
        )

# -------------------- 主程序入口 --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D体素数据量化处理器")
    parser.add_argument('--config_path', type=str, default='gs_render/data_process/nuplan.yaml', help='YAML配置文件路径')
    parser.add_argument('--pkl_path', type=str, default='data/nuplan_pkls/mini/nuplan_mini_10hz_val.pkl', help='pkl文件目录')
    parser.add_argument('--data_base_path', type=str, default='data/GT_occ_fast_val/dense_voxels_with_semantic', help='输入数据目录')
    parser.add_argument('--save_base_path', type=str, default='data/GT_occ_fast_val_400', help='输出目录')
    parser.add_argument('--quantize_size', type=int, nargs=3, default=[400, 400, 32], help='目标尺寸 (X Y Z)')
    parser.add_argument('--method', choices=['max', 'nearest'], default='max', help='降采样方法')
    parser.add_argument('--workers', type=int, default=64, help='并行工作进程数')
    parser.add_argument('--origin_size', type=int, nargs=3, default=[800, 800, 64], help='原始体素尺寸 (X Y Z)')
    parser.add_argument('--voxel_size', type=float, nargs=3, default=[0.125, 0.125, 0.125], help='体素尺寸 (X Y Z)')

    
    args = parser.parse_args()
    
    processor = QuantizationProcessor(
        config_path=args.config_path,
        pkl_path=args.pkl_path,
        data_dir=args.data_base_path,
        output_dir=args.save_base_path,
        target_size=args.quantize_size,
        origin_size=args.origin_size,
        voxel_size=args.voxel_size,
        method=args.method,
        workers=args.workers
    )
    processor.run()