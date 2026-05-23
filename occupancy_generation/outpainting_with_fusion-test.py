#!/usr/bin/env python3
"""
基于第一帧数据的Outpainting - 使用is_fusion策略支持无限扩充

基于eval_OccDiT_nuplan_uncon_bevocc.py的结构，实现outpainting功能
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffusion.models import *
from diffusion import create_diffusion
from tqdm import tqdm
import datetime

from mmengine import Config
from mmengine.registry import MODELS
import model_vae

 
import math
from scipy import ndimage
import numpy as np
import torch.distributions as dist
import torch.nn.functional as F
import random


# def fill_large_empty_regions_gpu2(
#     occ_volume, 
#     min_fill_height=6,  # 明确最低填充高度要求
#     base_center=8,      # 基础高度设为min_fill_height+2确保最低为10
#     std_dev=0.1,         # 保持自然起伏
#     kernel_size=3,       # 核大小设为1（相当于禁用形态学过滤）
#     fill_class=1,
#     reverse_height=False
# ):
#     """
#     连续填充空洞区域 - 创建实心柱体且确保最低高度
    
#     参数:
#         occ_volume (torch.Tensor): [B, T, H, W, D] 3D语义占据体积
#         min_fill_height (int): 最低填充高度（原始坐标系：0=地面）
#         base_center (int): 基础高度位置（应 >= min_fill_height）
#         std_dev (float): 高斯分布标准差
#         kernel_size (int): 形态学核大小（设为1禁用过滤）
#         fill_class (int): 填充类别
#         reverse_height (bool): 是否反转高度方向
        
#     返回:
#         torch.Tensor: 填充后的体积
#     """
#     B, T, H, W, D = occ_volume.shape
#     device = occ_volume.device
    
#     # 1. 检测所有空洞区域（包括小区域）
#     # kernel_size=1 禁用形态学过滤，保留所有空洞
#     empty_mask = (occ_volume == 0).all(dim=-1)  # [B, T, H, W]
    
#     # 2. 生成随机高度（确保 >= min_fill_height）
#     # 基础高度 = min_fill_height + std_dev*2 (确保95%区域>min_fill_height)
#     base_height = min_fill_height + 2 * std_dev
#     base_height = max(base_height, min_fill_height + 1)  # 安全边界
    
#     # 生成高度分布（原始坐标系：0=地面）
#     random_heights = base_height + torch.normal(
#         mean=0.0, 
#         std=std_dev, 
#         size=(B, T, H, W), 
#         device=device
#     )
    
#     # 强制高度 >= min_fill_height
#     random_heights = torch.clamp(random_heights, min=min_fill_height, max=D-1)
    
#     # 3. 创建连续填充掩码（实心柱体）
#     z_indices = torch.arange(D, device=device).view(1, 1, 1, 1, D).float()
    
#     if reverse_height:
#         # 反转坐标系：填充条件 z >= (D-1 - random_heights)
#         fill_condition = z_indices >= (D - 1 - random_heights).unsqueeze(-1)
#     else:
#         # 标准坐标系：填充条件 z <= random_heights
#         fill_condition = z_indices <= random_heights.unsqueeze(-1)
    
#     # 仅对空洞区域应用填充
#     fill_mask = fill_condition & empty_mask.unsqueeze(-1)

    
#     # real_height_mask= torch.ones_like(fill_mask)
#     # real_height_mask[..., :6] = 0
#     # fill_mask = fill_mask & real_height_mask

    
    
#     # 5. 执行填充（创建实心柱体）
#     result = occ_volume.clone()
#     result[fill_mask] = fill_class
    
#     return result



def fill_large_empty_regions_gpu(
    occ_volume,
    min_height=8,   
    max_height=9,  
    fill_class=1,
    min_area=3,
    roughness=0.3,
    seed=42,
    reverse_height=False,  # 新增参数控制高度方向
     position=None
):
    """
    GPU加速的随机高度填充（近似实现）
    """
    B, T, H, W, D = occ_volume.shape
    device = occ_volume.device

    
    # 使用位置信息创建唯一种子
    if seed is None:
        seed = int(time.time() * 1000) % 1000000
    if position is not None:
        seed = (seed + position[0]*1000 + position[1]) % 1000000

        
    # 1. 检测空位置
    empty_mask = (occ_volume == 0).all(dim=-1)  # [B, T, H, W]
    
    # 2. 形态学开运算（GPU）
    kernel_size = int(np.sqrt(min_area)) * 2 + 1
    padding = kernel_size // 2
    
    # 合并批次和时间维度
    mask_4d = empty_mask.float().view(-1, 1, H, W)
    
    # 腐蚀 + 膨胀 (开运算)
    eroded = -F.max_pool2d(-mask_4d, kernel_size=kernel_size, stride=1, padding=padding)
    opened = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=padding)
    large_regions_mask = (opened > 0.5).view(B, T, H, W)
    
    # 3. 生成随机高度场（GPU）
    torch.manual_seed(seed)
    base_noise = torch.rand(B, T, H, W, device=device)

    
    # 应用多次模糊创建分形效果
    for _ in range(3):
        base_noise = F.avg_pool2d(base_noise, kernel_size=3, stride=1, padding=1)
        new_noise = torch.rand(B, T, H, W, device=device)
        base_noise = base_noise * 0.7 + new_noise * 0.3 * roughness
    
    # 映射到高度范围
    height_map = min_height + (max_height - min_height) * base_noise



    # 添加基于位置的高度变化
    if position is not None:
        pos_factor = 0.3 * (position[0]/H + position[1]/W)
        height_map = height_map * (1.0 - pos_factor) + min_height * pos_factor


        
    # 4. 创建高度索引
    height_indices = torch.arange( D, device=device).view(1, 1, 1, 1, D)

    # random_integer = int(random.randint( int(min_height), D))
    # # 根据reverse_height参数调整高度方向
    if reverse_height:
        height_indices = torch.arange(D-1, -1, -1, dtype=torch.float32, device=device).view(1, 1, 1, 1, D)
    else:
        height_indices = torch.arange(D, dtype=torch.float32, device=device).view(1, 1, 1, 1, D)

        
    # 5. 创建填充掩码
    height_mask = height_indices < height_map.unsqueeze(-1)  # [B, T, H, W, D]
    region_mask = large_regions_mask.unsqueeze(-1)  # [B, T, H, W, 1]

    clipped_heright = torch.ones_like(height_mask)
    clipped_heright[:, :, :, :, 12:]=0

    
    fill_mask = height_mask & region_mask

    
    real_height_mask= torch.ones_like(fill_mask)
    real_height_mask[..., :8] = 0
    fill_mask = fill_mask & real_height_mask

    
    # 6. 应用填充
    result = occ_volume.clone()
    result[fill_mask] = fill_class
    
    # # 7. 添加随机空洞（GPU）
    # hole_mask = (torch.rand(B, T, H, W, D, device=device) )< (0.05 * roughness)
    # result[fill_mask & hole_mask] = 0
    # import pdb; pdb.set_trace()
    
    return result


def create_logger(logging_dir, rank):
    """创建日志记录器"""
    if rank == 0:
        logging.basicConfig(
            level=logging.DEBUG,
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def load_first_frame_data(data_path):
    """
    加载第一帧数据
    
    Args:
        data_path: 第一帧数据文件路径 (.npy)
    
    Returns:
        first_frame_occ: [B, T, H, W, D]
        first_frame_bev: [B, T, C, Hb, Wb] 
        first_frame_meta: [B, meta_num]
    """
    print(f"正在加载第一帧数据: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"第一帧数据文件不存在: {data_path}")
    
    # 加载occupancy数据
    occ_data = np.load(data_path)
    print(f"第一帧原始形状: {occ_data.shape}")
    
    # 确保是5维：[B, T, H, W, D]
    if occ_data.ndim == 3:  # [H, W, D]
        occ_data = occ_data[None, None, ...]  # [1, 1, H, W, D]
        T = 1
    elif occ_data.ndim == 4:  # [T, H, W, D]
        occ_data = occ_data[None, ...]  # [1, T, H, W, D]
        T = occ_data.shape[1]
    elif occ_data.ndim == 5:  # [B, T, H, W, D]
        T = occ_data.shape[1]
    else:
        raise ValueError(f"不支持的数据维度: {occ_data.ndim}")
        
    B, T, H, W, D = occ_data.shape
    
    # 创建对应的BEV数据（模拟）
    bev_data = np.zeros((B, T, 1, 200, 200), dtype=np.float32)
    
    # 创建meta数据（模拟occupancy metadata）
    # 根据eval代码，使用meta_num=4
    meta_data = np.random.randn(B, 4).astype(np.float32)
    
    print(f"最终第一帧形状: occ={occ_data.shape}, bev={bev_data.shape}, meta={meta_data.shape}")
    
    return torch.from_numpy(occ_data), torch.from_numpy(bev_data), torch.from_numpy(meta_data)


# def get_patch_positions(scene_shape, patch_shape=(200, 200, 16), overlap=0.5):
#     """
#     计算patch位置 - is_fusion策略
#     使用固定200x200x16的patch大小，参考lt3sd
    
#     Args:
#         scene_shape: (H, W, D) 目标场景大小
#         patch_shape: (200, 200, 16) 固定patch大小 - 与模型训练时一致
#         overlap: 重叠比例
        
#     Returns:
#         patch_configs: [(h_start, h_end, w_start, w_end), ...]
#     """
#     H, W, D = scene_shape
#     patch_H, patch_W, patch_D = patch_shape
    
#     # 计算步长（参考lt3sd的step_size计算）
#     step_h = int(patch_H * (1 - overlap))
#     step_w = int(patch_W * (1 - overlap))
    
#     print(f"is_fusion配置: 场景{scene_shape}, patch{patch_shape}, 步长({step_h}, {step_w})")
    
#     patch_configs = []
#     for h_start in range(0, H, step_h):
#         for w_start in range(0, W, step_w):
#             h_end = min(h_start + patch_H, H)
#             w_end = min(w_start + patch_W, W)
            
#             # 添加所有patch，即使是边界的小patch也处理
#             patch_configs.append((h_start, h_end, w_start, w_end))
    
#     print(f"总共{len(patch_configs)}个patches需要处理")
#     return patch_configs
# 修改get_patch_positions函数
def get_patch_positions(scene_shape, patch_shape=(200, 200, 16), overlap=0.5):
    H, W, D = scene_shape
    patch_H, patch_W, patch_D = patch_shape
    
    step_h = int(patch_H * (1 - overlap))
    step_w = int(patch_W * (1 - overlap))
    
    patch_configs = []
    
    # 从右下角开始处理（左上扩展方向）
    for h_start in range(H - patch_H, -1, -step_h):
        for w_start in range(W - patch_W, -1, -step_w):
            h_end = min(h_start + patch_H, H)
            w_end = min(w_start + patch_W, W)
            patch_configs.append((max(0, h_start), h_end, max(0, w_start), w_end))
    
    # 添加未覆盖的左上角区域
    for h_start in range(0, H - patch_H, step_h):
        for w_start in range(0, W - patch_W, step_w):
            # 跳过已处理区域
            if (h_start, h_start + patch_H, w_start, w_start + patch_W) not in patch_configs:
                patch_configs.append((h_start, h_start + patch_H, w_start, w_start + patch_W))
    
    print(f"总共{len(patch_configs)}个patches需要处理（右下优先）")
    return patch_configs

def perform_outpainting_fusion(first_frame_occ, first_frame_bev, first_frame_meta, 
                              target_scene_shape, model, vae, diffusion, device,
                              scale_factor=70.0, use_noise_prior=True, lambda_np=0.03,
                              patch_overlap=0.5):
    """
    使用is_fusion策略进行outpainting - 核心patch fusion逻辑
    
    参考lt3sd的is_fusion策略：
    1. 将第一帧放在左上角
    2. 分块处理，每个200x200x16 patch独立生成
    3. 使用known_mask融合：已知区域保持原值，未知区域使用生成值
    
    Args:
        first_frame_occ: 第一帧占用数据 [B, T, H, W, D]
        first_frame_bev: 第一帧BEV数据 [B, T, C, Hb, Wb]  
        first_frame_meta: 第一帧meta数据 [B, meta_num]
        target_scene_shape: 目标场景大小 (H, W, D)
        其他: 模型参数
    
    Returns:
        generated_occ: 生成的occupancy [B, T, target_H, target_W, target_D]
    """
    print(f"\n=== 开始 is_fusion Outpainting ===")
    print(f"第一帧形状: {first_frame_occ.shape}")
    print(f"目标场景: {target_scene_shape}")
    
    B, T, orig_H, orig_W, orig_D = first_frame_occ.shape
    target_H, target_W, target_D = target_scene_shape
    
    # 移动到设备
    first_frame_occ = first_frame_occ.to(device)

    print("************************   ")
    first_frame_occ = fill_large_empty_regions_gpu( first_frame_occ )
    
    
    first_frame_bev = first_frame_bev.to(device)
    first_frame_meta = first_frame_meta.to(device)
    
    # 初始化full scene和known mask（is_fusion核心）
    full_scene = torch.zeros((B, T, target_H, target_W, target_D), device=device, dtype=torch.long)
    known_mask = torch.zeros((B, T, target_H, target_W, target_D), device=device, dtype=torch.float32)
    
    # # is_fusion策略：将第一帧放在左上角（0,0位置开始）
    # place_h = min(orig_H, target_H)
    # place_w = min(orig_W, target_W) 
    # place_d = min(orig_D, target_D)
 
    # full_scene[:, :, :place_h, :place_w, :place_d] = first_frame_occ[:, :, :place_h, :place_w, :place_d]
    # known_mask[:, :, :place_h, :place_w, :place_d] = 1.0
    
    # print(f"第一帧已放置在左上角位置: (0, 0) - ({place_h}, {place_w}, {place_d})")

    # 修改位置：将第一帧放在右下角
    place_h = min(orig_H, target_H)
    place_w = min(orig_W, target_W)
    place_d = min(orig_D, target_D)
    
    # 计算右下角起始位置
    h_start = max(0, target_H - place_h)
    w_start = max(0, target_W - place_w)
    
    # 放置到右下角
    full_scene[:, :, h_start:h_start+place_h, w_start:w_start+place_w, :place_d] = \
        first_frame_occ[:, :, :place_h, :place_w, :place_d]
    known_mask[:, :, h_start:h_start+place_h, w_start:w_start+place_w, :place_d] = 1.0
    
    print(f"第一帧已放置在右下角位置: ({h_start}, {w_start}) - "
          f"({h_start+place_h}, {w_start+place_w}, {place_d})")

    
    
    # 获取patch配置
    patch_configs = get_patch_positions(target_scene_shape, overlap=patch_overlap)
    
    # is_fusion策略：独立生成每个patch
    for i, (patch_h_start, patch_h_end, patch_w_start, patch_w_end) in enumerate(tqdm(patch_configs, desc="生成patches")):
        print(f"\n处理patch {i+1}/{len(patch_configs)}: H[{patch_h_start}:{patch_h_end}], W[{patch_w_start}:{patch_w_end}]")
        
        
        # 计算patch中心位置（用于位置编码）
        patch_center_h = (patch_h_start + patch_h_end) / 2.0 / target_H
        patch_center_w = (patch_w_start + patch_w_end) / 2.0 / target_W

        
        # 提取当前patch区域
        patch_H = patch_h_end - patch_h_start
        patch_W = patch_w_end - patch_w_start
        patch_D = target_D
        
        # 当前区域的occupancy和mask
        current_region = full_scene[:, :, patch_h_start:patch_h_end, patch_w_start:patch_w_end, :]
        current_mask = known_mask[:, :, patch_h_start:patch_h_end, patch_w_start:patch_w_end, :]
        
        # is_fusion核心逻辑：跳过完全已知的patch
        if torch.all(current_mask == 1.0):
            print("  patch完全已知，跳过")
            continue
        
        # is_fusion策略：始终使用标准200x200x16生成，确保模型输入一致
        print(f"  patch区域: ({patch_H}, {patch_W}, {patch_D}) -> 生成标准200x200x16")
        
        # 创建标准大小的输入（200x200x16）
        standard_region = torch.zeros((B, T, 200, 200, 16), device=device, dtype=torch.long)
        standard_mask = torch.zeros((B, T, 200, 200, 16), device=device, dtype=torch.float32)
        
        # 将实际区域复制到标准大小patch中
        actual_h = min(patch_H, 200)
        actual_w = min(patch_W, 200)
        actual_d = min(patch_D, 16)
        
        standard_region[:, :, :actual_h, :actual_w, :actual_d] = current_region[:, :, :actual_h, :actual_w, :actual_d]
        standard_mask[:, :, :actual_h, :actual_w, :actual_d] = current_mask[:, :, :actual_h, :actual_w, :actual_d]
        
        # 1. 编码标准大小的patch  
        if torch.any(standard_mask > 0):
            with torch.no_grad():
                # 对包含已知区域的patch进行编码
                x_encoded = vae.encode(standard_region) * scale_factor
        else:
            # 完全未知区域，创建标准大小的随机噪声
            # x_encoded = torch.randn(B, 4, T, 200//4, 200//4, device=device) * scale_factor
            # 基于位置生成不同的噪声
            position_noise = torch.randn(B, 4, T, 200//4, 200//4, device=device)
            position_noise = 0.7 * position_noise +  0.3 * torch.randn(1, device=device) * patch_center_h
            position_noise = 0.7 * position_noise +  0.3 * torch.randn(1, device=device) * patch_center_w
            x_encoded = position_noise * scale_factor
            
        
        # 2. 准备模型条件和噪声 - 修复噪声先验问题
        model_kwargs = dict(y=first_frame_bev, meta=first_frame_meta)
        noise = torch.randn_like(x_encoded)
        
        # if use_noise_prior and torch.any(standard_mask > 0):
        #     known_ratio = standard_mask.float()
            
        #     if known_ratio.mean().item() > 0.1: 
        #         adjusted_lambda = lambda_np * known_ratio
        #         import pdb; pdb.set_trace()
        #         noise = (torch.ones_like(adjusted_lambda) - adjusted_lambda) * noise + adjusted_lambda * x_encoded
        #         print(f"  应用噪声先验: known_ratio={known_ratio:.3f}, adjusted_lambda={adjusted_lambda:.4f}")
        #     else:
        #         print(f"  已知区域过少({known_ratio:.3f})，跳过噪声先验")

        if use_noise_prior and torch.any(standard_mask > 0):
            known_ratio = standard_mask.float().mean().item()
            known_ratio=1.0
            if known_ratio > 0.1: 
                
                adjusted_lambda = lambda_np * known_ratio
                noise = (1-adjusted_lambda) * noise + adjusted_lambda * x_encoded ##latent_mask -->modify
                print(f"  应用噪声先验: known_ratio={known_ratio:.3f}, adjusted_lambda={adjusted_lambda:.4f}")
            else:
                print(f"  已知区域过少({known_ratio:.3f})，跳过噪声先验")

                
            
            noise = torch.clamp(noise, -3*scale_factor, 3*scale_factor)
        
        # 3. 对标准200x200x16 patch进行独立扩散采样
        with torch.no_grad():
            model_fn = model.module.forward if hasattr(model, 'module') else model.forward
            samples = diffusion.p_sample_loop(
                model_fn,
                noise.shape,
                noise=noise,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
                clip_denoised=True,  # 添加clip增强稳定性
            )
            
            # 验证采样结果的范围
            sample_std = samples.std().item()
            if sample_std > scale_factor * 2:
                print(f"  警告: 采样结果方差过大 std={sample_std:.2f}, 进行调整")
                samples = torch.clamp(samples, -scale_factor, scale_factor)
        
        # 4. VAE解码为标准200x200x16
        samples = samples / scale_factor  
        rec_shape = [B, T, 200, 200, 16] 
        
        with torch.no_grad():
            result = vae.generate(samples, rec_shape)
            logits = result["logits"]
            
            # 验证logits质量
            logit_max = logits.max().item()
            logit_min = logits.min().item()
            if logit_max - logit_min < 1.0:  # logits动态范围过小
                print(f"  警告: logits动态范围过小 [{logit_min:.2f}, {logit_max:.2f}]")
            
            standard_pred = logits.argmax(dim=-1).long()  # [B, T, 200, 200, 16]
            print( "  **********************  standard_pred shape", standard_pred.shape )
            standard_pred = fill_large_empty_regions_gpu( standard_pred , position=(patch_h_start, patch_w_start)  )
            
            # 验证生成结果的分布
            unique_vals = torch.unique(standard_pred).cpu().numpy()
            print(f"  生成值分布: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
            
        
        # 5. is_fusion融合：
        
        patch_pred = standard_pred[:, :, :actual_h, :actual_w, :actual_d]
        
        # is_fusion核心
        current_region_actual = current_region[:, :, :actual_h, :actual_w, :actual_d]
        current_mask_actual = current_mask[:, :, :actual_h, :actual_w, :actual_d]
        
        fused_patch = patch_pred * (1 - current_mask_actual).long() + current_region_actual * current_mask_actual.long()
        
        
        # 6. 更新全局场景（只更新实际区域）
        actual_h_end = patch_h_start + actual_h
        actual_w_end = patch_w_start + actual_w
        actual_d_end = min(patch_D, target_D)
        
        full_scene[:, :, patch_h_start:actual_h_end, patch_w_start:actual_w_end, :actual_d_end] = fused_patch
        known_mask[:, :, patch_h_start:actual_h_end, patch_w_start:actual_w_end, :actual_d_end] = 1.0
    
    # 确保所有区域都被处理
    print(f"\n最终known_mask覆盖率: {known_mask.mean().item():.4f}")
    
    return full_scene  # full_scene已经是long类型


def main(args):
    print(" 基于第一帧数据的is_fusion Outpainting")
    print("=" * 60)
    
    # 设备设置
    if not args.no_distributed:
        dist.init_process_group("nccl")
        rank = args.local_rank
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}, rank: {rank}")
    
    # 创建输出目录
    if rank == 0:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        experiment_dir = f"{args.results_dir}/outpainting_fusion_{timestamp}"
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir, rank)
        logger.info(f"Outpainting目录: {experiment_dir}")
    else:
        logger = create_logger(None, rank)
        experiment_dir = None
    
    # 模型配置（与eval代码保持一致）
    use_bev_concat = False
    use_occ_meta = True
    in_ch = 4
    Tframe = 5
    meta_num = 4  # meta_num_mode = 3
    ref_idx = 0
    scale_factor = 70
    use_noise_prior = True
    lambda_np = args.lambda_noise_prior
    
    # 创建DiT模型
    DiT_cfg = {
        "depth": 12, 
        "in_channels": in_ch, 
        "hidden_size": 512,
        "use_label": False, 
        "use_bev_concat": use_bev_concat,
        "bev_in_ch": 1,
        "bev_out_ch": 1,
        "use_meta": use_occ_meta, 
        "bev_dropout_prob": 0.1,
        "meta_num": meta_num,
        "direct_concat": True,
        # "Tframe": Tframe
    }
    model = DiT_multiframe(**DiT_cfg).to(device)
    
    # 加载DiT checkpoint
    if args.ckpt:
        logger.info(f"Loading DiT model from: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        if 'ema' in checkpoint:
            logger.info("Using EMA weights")
            model.load_state_dict(checkpoint['ema'], strict=False)
        else:
            logger.info("Using model weights")
            model.load_state_dict(checkpoint['model'], strict=False)
        del checkpoint
    else:
        logger.warning("No DiT checkpoint provided!")
    
    # 创建扩散模型
    diffusion = create_diffusion(timestep_respacing="")
    
    # 创建VAE模型
    logger.info("Loading VAE model...")
    cfg = Config.fromfile(args.vae_config)
    vae = MODELS.build(cfg.model)
    vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu')
    vae.load_state_dict(vae_ckpt['state_dict'], strict=True)
    vae = vae.to(device)
    vae.eval()
    
    # DDP
    if not args.no_distributed:
        model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    model.eval()
    
    logger.info(f"DiT参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 加载第一帧数据
    logger.info(f"Loading first frame from: {args.first_frame_path}")
    first_frame_occ, first_frame_bev, first_frame_meta = load_first_frame_data(args.first_frame_path)
    
    # 定义目标场景大小
    original_shape = first_frame_occ.shape[2:5]  # (H, W, D)
    target_shapes = []
    
    for scale in args.scale_factors:
        target_H = int(original_shape[0] * scale)
        target_W = int(original_shape[1] * scale)
        target_D = original_shape[2]  # 深度保持不变
        target_shapes.append((target_H, target_W, target_D))
    
    logger.info(f"原始形状: {original_shape}")
    logger.info(f"目标形状: {target_shapes}")
    
    # 对每个目标大小进行outpainting
    for i, target_shape in enumerate(target_shapes):
        scale = args.scale_factors[i]
        logger.info(f"\n开始 {scale}x outpainting to {target_shape}")
        
        try:
            generated_occ = perform_outpainting_fusion(
                first_frame_occ=first_frame_occ,
                first_frame_bev=first_frame_bev,
                first_frame_meta=first_frame_meta,
                target_scene_shape=target_shape,
                model=model,
                vae=vae,
                diffusion=diffusion,
                device=device,
                scale_factor=scale_factor,
                use_noise_prior=use_noise_prior,
                lambda_np=lambda_np,
                patch_overlap=args.patch_overlap
            )
            
            # 保存结果
            if rank == 0:
                save_dir = f"{experiment_dir}/scale_{scale}x"
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存生成结果
                for b in range(generated_occ.shape[0]):
                    for t in range(generated_occ.shape[1]):
                        occ_frame = generated_occ[b, t].detach().cpu().numpy().astype(np.int8)
                        print("occ_frame shape", occ_frame.shape )
                        save_path = os.path.join(save_dir, f"outpainted_{scale}x_batch{b}_frame{t}.npy")
                        np.save(save_path, occ_frame)
                
                # 保存第一帧参考
                first_frame_np = first_frame_occ[0, 0].detach().cpu().numpy().astype(np.int8)
                ref_path = os.path.join(save_dir, "reference_first_frame.npy")
                np.save(ref_path, first_frame_np)
                
                # 保存信息
                info_path = os.path.join(save_dir, f"info_{scale}x.txt")
                with open(info_path, 'w') as f:
                    f.write(f"Scale Factor: {scale}x\n")
                    f.write(f"Original Shape: {original_shape}\n")
                    f.write(f"Target Shape: {target_shape}\n")
                    f.write(f"Generated Shape: {generated_occ.shape}\n")
                    f.write(f"Unique Values: {torch.unique(generated_occ).cpu().numpy()}\n")
                
                logger.info(f" {scale}x outpainting完成! 保存到: {save_dir}")
            
        except Exception as e:
            logger.error(f" {scale}x outpainting失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not args.no_distributed:
        dist.barrier()
        dist.destroy_process_group()
    
    logger.info(" 所有outpainting任务完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 必需参数
    parser.add_argument("--first-frame-path", type=str, required=True,
                       default="out/nuplan_occ_dit/eval_uncon_occbev_2025-07-22-00-46-59/visualizations/00000_occ_ori.npy",
                       help="第一帧数据路径")
    parser.add_argument("--ckpt", type=str, required=True,
                       default="checkpoint/occ_generation/dit.pt",
                       help="DiT模型checkpoint路径")
    parser.add_argument("--vae-ckpt", type=str, required=True,
                       default="checkpoint/occ_generation/3dvae.pth",
                       help="VAE checkpoint路径")
    parser.add_argument("--vae-config", type=str, required=True,
                       default="config/train_3dvae_nuplan_200_pro_occ_bev.py",
                       help="VAE配置文件路径")
    
    # 可选参数
    parser.add_argument("--results-dir", type=str, default="./outputs/outpainting_fusion",
                       help="结果保存目录")
    parser.add_argument("--scale-factors", type=float, nargs='+', default=[2.0, 3.0, 4.0],
                       help="放大倍数列表")
    parser.add_argument("--patch-overlap", type=float, default=0.5,
                       help="patch重叠比例")
    parser.add_argument("--lambda-noise-prior", type=float, default=0.5,
                       help="噪声先验权重")
    parser.add_argument("--no-distributed", action='store_true',
                       help="不使用分布式训练")
    parser.add_argument("--local-rank", type=int, default=0,
                       help="本地rank")
    
    args = parser.parse_args()
    main(args) 
