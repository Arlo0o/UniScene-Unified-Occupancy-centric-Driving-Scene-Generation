#!/usr/bin/env python3
import pickle

def load_preprocessed_clips(clip_file_path):
    """
    直接加载预处理好的clip_infos，无需重新构建
    """
    with open(clip_file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"=> 从 {clip_file_path} 加载预处理数据")
    print(f"=> 总共 {len(data['clip_infos'])} 个clips")
    print(f"=> 原始info数量: {data['original_info_count']}")
    
    return data

if __name__ == "__main__":
    # 使用示例
    clip_file = "nuplan_mini_10hz_train_clip_infos_lightweight.pkl"  # 修改为你的预处理文件路径
    
    # 直接加载，无需dataloader重建
    data = load_preprocessed_clips(clip_file)
    clip_infos = data['clip_infos']
    
    print(f"可以直接使用 clip_infos: {len(clip_infos)} 个clips")
    print("第一个clip示例:", clip_infos[0] if clip_infos else "无clips")
