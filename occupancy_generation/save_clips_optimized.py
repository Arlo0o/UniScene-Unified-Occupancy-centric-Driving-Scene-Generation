#!/usr/bin/env python3
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
from pathlib import Path


def save_clip_infos_ultimate_mini_npy(imageset_path, occ_dataroot, return_len=8, output_path=None, debug=False):
    """
    终极性能版本 - 针对370万infos + {token}.npy结构的最快实现
    """
    import os
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    start_time = time.time()
    
    # 加载原始数据
    with open(imageset_path, 'rb') as f:
        pkl_data = pickle.load(f)
    print(f"=> loaded pkl_data from {imageset_path}")
    
    nuplan_infos = pkl_data['infos']
    scene_tokens = pkl_data['scene_tokens']
    
    print(f"=> 总共 {len(nuplan_infos)} 个infos, {len(scene_tokens)} 个scenes")
    
    # 参数
    start_on_keyframe = True
    start_on_firstframe = False
    
    # 构建token到索引的映射
    token_data_dict = {item['token']: idx for idx, item in enumerate(nuplan_infos)}
    
    # 终极优化: 直接扫描.npy文件
    print("=> ⚡ 终极文件扫描...")
    occ_path = Path(occ_dataroot)
    
    # 直接扫描occ_dataroot目录下的.npy文件
    print("   扫描.npy文件...")
    all_token_files = set()
    
    with os.scandir(occ_dataroot) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.npy'):
                # 从文件名提取token (去掉.npy后缀)
                token = entry.name[:-4]
                all_token_files.add(token)
    
    print(f"=> ✅ 找到 {len(all_token_files)} 个有效token文件")
    
    # 构建clips - 最优化版本
    print("=> 🔨 构建clips...")
    all_clips = []
    scenes_to_process = scene_tokens[:1] if debug else scene_tokens
    
    # 预计算，避免重复计算
    valid_clips = 0
    
    for i, scene in enumerate(scenes_to_process):
        if i % 500 == 0:
            print(f"   处理scene: {i}/{len(scenes_to_process)}")
        
        scene_len = len(scene)
        for start in range(scene_len - return_len + 1):
            # 快速跳过检查
            start_token = scene[start]
            if start_on_keyframe and (";" in start_token or len(start_token) >= 33):
                continue
            
            # 一次性获取所有tokens并检查
            tokens_in_clip = scene[start: start + return_len]
            
            # 使用all()的短路特性 - 一旦有一个不存在就立即跳出
            if all(token in all_token_files for token in tokens_in_clip):
                clip = [token_data_dict[token] for token in tokens_in_clip]
                all_clips.append(clip)
                valid_clips += 1
                
                if start_on_firstframe:
                    break
    
    print(f"=> clip_infos: {len(all_clips)}")
    
    # 最快的lightweight_infos构建
    lightweight_infos = [{'token': info['token'],
                          "anns":info['anns']
                          } for info in nuplan_infos]
    
    new_pkl_data = {
        'infos': lightweight_infos,
        'scene_tokens': pkl_data['scene_tokens'],
        'clip_infos': all_clips,
        'original_info_count': len(nuplan_infos),
    }
    
    # 保存文件
    if output_path is None:
        output_path = 'ultimate_fast_clip_infos.pkl'
    
    with open(output_path, 'wb') as f:
        pickle.dump(new_pkl_data, f)
    
    total_time = time.time() - start_time
    print(f"=> 🎉 终极版本完成! 总耗时: {total_time:.2f}秒")
    print(f"=> 💾 Saved to {output_path}")
    print(f"=> ⚡ 处理速度: {len(nuplan_infos)/total_time:.0f} infos/sec")
    
    return output_path

def save_clip_infos_ultimate(imageset_path, occ_dataroot, return_len=8, output_path=None, debug=False):
    """
    终极性能版本 - 针对370万infos + {token}/{token}.npz结构的最快实现
    """
    import os
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    start_time = time.time()
    
    # 加载原始数据
    with open(imageset_path, 'rb') as f:
        pkl_data = pickle.load(f)
    print(f"=> loaded pkl_data from {imageset_path}")
    
    nuplan_infos = pkl_data['infos']
    scene_tokens = pkl_data['scene_tokens']
    
    print(f"=> 总共 {len(nuplan_infos)} 个infos, {len(scene_tokens)} 个scenes")
    
    # 参数
    start_on_keyframe = True
    start_on_firstframe = False
    
    # 构建token到索引的映射
    token_data_dict = {item['token']: idx for idx, item in enumerate(nuplan_infos)}
    
    # 终极优化: 直接扫描{token}/{token}.npz文件
    print("=> ⚡ 终极文件扫描...")
    scan_start_time = time.time()
    occ_path = Path(occ_dataroot)
    
    # 扫描occ_dataroot目录下的{token}/{token}.npz文件
    print("   扫描{token}/{token}.npz文件...")
    all_token_files = set()
    scanned_count = 0
    
    # 使用多线程扫描以加速处理
    def scan_token_directory(entry_path):
        """扫描单个token目录"""
        try:
            token_dir = os.path.basename(entry_path)
            npz_file_path = os.path.join(entry_path, f"{token_dir}.npz")
            if os.path.isfile(npz_file_path):
                return token_dir
        except:
            pass
        return None
    
    # 先快速获取所有目录
    all_dirs = []
    with os.scandir(occ_dataroot) as entries:
        for entry in entries:
            if entry.is_dir():
                all_dirs.append(entry.path)
    
    print(f"   找到 {len(all_dirs)} 个目录，开始多线程扫描...")
    
    # 使用线程池并行扫描
    max_workers = min(256, os.cpu_count() * 4)  # 限制最大线程数
    batch_size = 10000  # 批处理大小
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(all_dirs), batch_size):
            batch = all_dirs[i:i + batch_size]
            
            # 提交批处理任务
            futures = {executor.submit(scan_token_directory, dir_path): dir_path for dir_path in batch}
            
            # 收集结果
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_token_files.add(result)
                
                scanned_count += 1
                
                # 每10万个打印一次进度
                if scanned_count % 100000 == 0:
                    elapsed = time.time() - scan_start_time
                    rate = scanned_count / elapsed
                    total_estimate = len(all_dirs) / rate
                    remaining = total_estimate - elapsed
                    print(f"   已扫描: {scanned_count:,}/{len(all_dirs):,} | 速度: {rate:.0f} dirs/sec | 预估总时间: {total_estimate/60:.1f}分钟 | 剩余: {remaining/60:.1f}分钟")
    
    scan_time = time.time() - scan_start_time
    final_rate = scanned_count / scan_time
    estimated_370w_time = scanned_count / final_rate  # 使用实际扫描数量
    
    print(f"=> ✅ 找到 {len(all_token_files)} 个有效token文件")
    print(f"=> 📊 扫描统计: 扫描了{scanned_count:,}个目录, 耗时{scan_time:.2f}秒, 速度{final_rate:.0f} dirs/sec")
    print(f"=> ⏱️  370万文件预估扫描时间: {estimated_370w_time/60:.1f}分钟 ({estimated_370w_time/3600:.1f}小时)")
    
    # 构建clips - 最优化版本
    print("=> 🔨 构建clips...")
    all_clips = []
    scenes_to_process = scene_tokens[:1] if debug else scene_tokens
    
    # 预计算，避免重复计算
    valid_clips = 0
    
    for i, scene in enumerate(scenes_to_process):
        if i % 500 == 0:
            print(f"   处理scene: {i}/{len(scenes_to_process)}")
        
        scene_len = len(scene)
        for start in range(scene_len - return_len + 1):
            # 快速跳过检查
            start_token = scene[start]
            if start_on_keyframe and (";" in start_token or len(start_token) >= 33):
                continue
            
            # 一次性获取所有tokens并检查
            tokens_in_clip = scene[start: start + return_len]
            
            # 使用all()的短路特性 - 一旦有一个不存在就立即跳出
            if all(token in all_token_files for token in tokens_in_clip):
                clip = [token_data_dict[token] for token in tokens_in_clip]
                all_clips.append(clip)
                valid_clips += 1
                
                if start_on_firstframe:
                    break
    
    print(f"=> clip_infos: {len(all_clips)}")
    
    # 最快的lightweight_infos构建
    lightweight_infos = [{'token': info['token']} for info in nuplan_infos]
    
    new_pkl_data = {
        'infos': lightweight_infos,
        'scene_tokens': pkl_data['scene_tokens'],
        'clip_infos': all_clips,
        'original_info_count': len(nuplan_infos),
    }
    
    # 保存文件
    if output_path is None:
        output_path = 'ultimate_fast_clip_infos.pkl'
    
    with open(output_path, 'wb') as f:
        pickle.dump(new_pkl_data, f)
    
    total_time = time.time() - start_time
    print(f"=> 🎉 终极版本完成! 总耗时: {total_time:.2f}秒")
    print(f"=> 💾 Saved to {output_path}")
    print(f"=> ⚡ 处理速度: {len(nuplan_infos)/total_time:.0f} infos/sec")
    
    return output_path


def save_clip_infos_ultimate_exist(imageset_path, occ_dataroot, return_len=8, output_path=None, debug=False):
    """
    终极性能版本 - 假设所有文件都存在，跳过文件扫描直接生成clips
    """
    start_time = time.time()
    
    # 加载原始数据
    with open(imageset_path, 'rb') as f:
        pkl_data = pickle.load(f)
    print(f"=> loaded pkl_data from {imageset_path}")
    
    nuplan_infos = pkl_data['infos']
    scene_tokens = pkl_data['scene_tokens']
    
    print(f"=> 总共 {len(nuplan_infos)} 个infos, {len(scene_tokens)} 个scenes")
    
    # 参数
    start_on_keyframe = True
    start_on_firstframe = False
    
    # 构建token到索引的映射
    token_data_dict = {item['token']: idx for idx, item in enumerate(nuplan_infos)}
    
    print("=> ⚡ 跳过文件扫描，假设所有文件都存在...")
    
    # 构建clips - 直接生成，不检查文件存在性
    print("=> 🔨 构建clips...")
    all_clips = []
    scenes_to_process = scene_tokens[:1] if debug else scene_tokens
    
    valid_clips = 0
    
    for i, scene in enumerate(scenes_to_process):
        if i % 500 == 0:
            print(f"   处理scene: {i}/{len(scenes_to_process)}")
        
        scene_len = len(scene)
        for start in range(scene_len - return_len + 1):
            # 快速跳过检查
            start_token = scene[start]
            if start_on_keyframe and (";" in start_token or len(start_token) >= 33):
                continue
            
            # 直接生成clip，不检查文件存在性
            tokens_in_clip = scene[start: start + return_len]
            clip = [token_data_dict[token] for token in tokens_in_clip]
            all_clips.append(clip)
            valid_clips += 1
            
            if start_on_firstframe:
                break
    
    print(f"=> clip_infos: {len(all_clips)}")
    
    # 最快的lightweight_infos构建
    lightweight_infos = [{
        'token': info['token'], 
        'anns': info['anns'], 
        'ego2global_translation':info['ego2global_translation'], 
        'ego2global_rotation':info['ego2global_rotation'],
        'lidar2ego_rotation':info['lidar2ego_rotation'], 
        'lidar2ego_translation':info['lidar2ego_translation'],
        'driving_command':info['driving_command'],
        } for info in nuplan_infos]
    
    new_pkl_data = {
        'infos': lightweight_infos,
        'scene_tokens': pkl_data['scene_tokens'],
        'clip_infos': all_clips,
        'original_info_count': len(nuplan_infos),
    }
    
    # 保存文件
    if output_path is None:
        output_path = 'ultimate_fast_clip_infos.pkl'
    
    with open(output_path, 'wb') as f:
        pickle.dump(new_pkl_data, f)
    
    total_time = time.time() - start_time
    print(f"=> 🎉 终极版本完成! 总耗时: {total_time:.2f}秒")
    print(f"=> 💾 Saved to {output_path}")
    print(f"=> ⚡ 处理速度: {len(nuplan_infos)/total_time:.0f} infos/sec")
    
    return output_path

if __name__ == "__main__":
    # 使用示例
    imageset_path = "/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_val.pkl"
    # imageset_path = "/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/trainval/nuplan_trainval_10hz_val.pkl"
    # imageset_path = "/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_train.pkl"
    # occ_dataroot = "/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_val/GT_occ_fast3_10hzval_r400/dense_voxels_with_semantic"
    # occ_dataroot = "/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ/nuplan_quantized_400_400_32"
    occ_dataroot = "/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ/nuplan_quantized_200_200_16"
    
    print("🚀 针对370万infos的终极优化处理...")
    
    # 推荐使用终极优化版本
    output_file = save_clip_infos_ultimate_exist(
        imageset_path, 
        occ_dataroot, 
        return_len=32, 
        debug=False,
        output_path="data/nuplan_mini_val_clip_infos_dit_32.pkl"
    )
    print(f"终极优化版本完成! 文件保存在: {output_file}")
