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

from diffusion.models import DiT_WorldModel
from diffusion import create_diffusion
from tqdm import tqdm
import datetime

from mmengine import Config
from mmengine.registry import MODELS
import model_vae


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

def fill_large_zero_height_regions_gpu(
    occ_volume, 
    target_layer=8,        # 目标填充层（索引位置）
    fill_class=1,
    reverse_height=False
):
    """
    在指定高度层填充空洞区域

    参数:
        occ_volume (torch.Tensor): [B, T, H, W, D] 3D语义占据体积
        target_layer (int): 目标填充层的索引位置（默认为6）
        fill_class (int): 填充类别
        reverse_height (bool): 是否反转高度方向（暂未使用）

    返回:
        torch.Tensor: 填充后的体积
    """
    B, T, H, W, D = occ_volume.shape
    device = occ_volume.device

    # 1. 检测所有空洞区域（包括小区域）
    # kernel_size=1 禁用形态学过滤，保留所有空洞, nuscenes 是 17, nuplan 是 0
    empty_mask = (occ_volume == 0).all(dim=-1)  # [B, T, H, W]

    # 2. 只填充第6层（索引为6的位置）
    # target_layer = 6
    
    # 3. 创建只填充第6层的掩码
    z_indices = torch.arange(D, device=device).view(1, 1, 1, 1, D)
    
    # 只选择第6层
    layer_mask = (z_indices == target_layer)
    
    # 仅对空洞区域的第6层应用填充
    fill_mask = layer_mask & empty_mask.unsqueeze(-1)
    
    # 5. 执行填充（创建实心柱体）
    result = occ_volume.clone()
    result[fill_mask] = fill_class

    return result

def load_first_frame_data(data_path, max_time_steps=5):
    """
    加载第一帧数据
    
    Args:
        data_path: 第一帧数据文件路径 (.npy)
        max_time_steps: 最大时间步数，默认为5（匹配T_pred）
    
    Returns:
        first_frame_occ: [B, T, H, W, D]
        first_frame_bev: [B, T, C, Hb, Wb] 
        first_frame_meta: [B, meta_num]
    """
    print(f"正在加载第一帧数据: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"第一帧数据文件不存在: {data_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载occupancy数据
    occ_data = np.load(data_path)
    print(f"原始数据类型: {type(occ_data)}, 形状: {occ_data.shape}")
    occ_data = torch.from_numpy(occ_data).to(device)
    occ_data = fill_large_zero_height_regions_gpu(occ_data, target_layer=8, fill_class=1)

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
    
    # 限制时间维度以匹配模型要求
    if T > max_time_steps:
        print(f"警告：数据有{T}个时间步，截取前{max_time_steps}个时间步")
        occ_data = occ_data[:, :max_time_steps, :, :, :]
        T = max_time_steps
    elif T < max_time_steps:
        print(f"警告：数据只有{T}个时间步，复制最后一帧到{max_time_steps}个时间步")
        # 复制最后一帧来填充
        last_frame = occ_data[:, -1:, :, :, :]  # [B, 1, H, W, D]
        repeat_frames = last_frame.repeat(1, max_time_steps - T, 1, 1, 1)
        occ_data = torch.cat([occ_data, repeat_frames], dim=1)
        T = max_time_steps
    
    B, T, H, W, D = occ_data.shape
    print(f"调整后时间维度: T={T}")
    
    # 创建对应的BEV数据（模拟）
    bev_data = np.zeros((B, T, 1, 200, 200), dtype=np.float32)
    
    # 创建meta数据（模拟occupancy metadata）
    # 根据eval代码，使用meta_num=4
    meta_data = np.random.randn(B, 4).astype(np.float32)
    
    print(f"最终第一帧形状: occ={occ_data.shape}, bev={bev_data.shape}, meta={meta_data.shape}")
    
    return occ_data, torch.from_numpy(bev_data).to(device), torch.from_numpy(meta_data).to(device)


def get_patch_positions(scene_shape, patch_shape=(200, 200, 16), overlap=0.5):
    """
    计算patch位置 - is_fusion策略
    使用固定200x200x16的patch大小，参考lt3sd
    
    Args:
        scene_shape: (H, W, D) 目标场景大小
        patch_shape: (200, 200, 16) 固定patch大小 - 与模型训练时一致
        overlap: 重叠比例
        
    Returns:
        patch_configs: [(h_start, h_end, w_start, w_end), ...]
    """
    H, W, D = scene_shape
    patch_H, patch_W, patch_D = patch_shape
    
    # 计算步长（参考lt3sd的step_size计算）
    step_h = int(patch_H * (1 - overlap))
    step_w = int(patch_W * (1 - overlap))
    
    print(f"is_fusion配置: 场景{scene_shape}, patch{patch_shape}, 步长({step_h}, {step_w})")
    
    patch_configs = []
    for h_start in range(0, H, step_h):
        for w_start in range(0, W, step_w):
            h_end = min(h_start + patch_H, H)
            w_end = min(w_start + patch_W, W)
            
            # 添加所有patch，即使是边界的小patch也处理
            patch_configs.append((h_start, h_end, w_start, w_end))
    
    print(f"总共{len(patch_configs)}个patches需要处理")
    return patch_configs


def create_position_aware_noise(base_noise, patch_pos, target_scene_shape, patch_index):
    """
    创建位置感知的噪声，增加patch间的多样性
    
    Args:
        base_noise: 基础噪声张量
        patch_pos: patch位置
        target_scene_shape: 目标场景形状
        patch_index: patch索引
    
    Returns:
        position_noise: 位置感知的噪声
    """
    h_start, h_end, w_start, w_end = patch_pos
    target_H, target_W, target_D = target_scene_shape
    
    # 基于patch位置和索引创建唯一种子
    position_seed = hash((h_start, h_end, w_start, w_end, patch_index)) % 2147483647
    
    # 创建独立的随机生成器
    generator = torch.Generator(device=base_noise.device)
    generator.manual_seed(position_seed)
    
    # 生成位置相关的噪声
    position_noise = torch.randn_like(base_noise)
    
    # 添加空间频率调制
    B, C, T, H, W = base_noise.shape
    
    # 创建空间坐标网格
    h_coords = torch.linspace(0, 1, H, device=base_noise.device).view(1, 1, 1, H, 1)
    w_coords = torch.linspace(0, 1, W, device=base_noise.device).view(1, 1, 1, 1, W)
    
    # 基于patch全局位置的频率调制
    center_h_norm = (h_start + h_end) / 2 / target_H
    center_w_norm = (w_start + w_end) / 2 / target_W
    
    # 创建多频率的空间调制
    freq1 = 2 * np.pi * (center_h_norm + 0.1)
    freq2 = 2 * np.pi * (center_w_norm + 0.1)
    
    spatial_mod = (torch.sin(h_coords * freq1) * torch.cos(w_coords * freq2) + 
                   torch.cos(h_coords * freq2) * torch.sin(w_coords * freq1)) * 0.3
    
    # 混合基础噪声、位置噪声和空间调制
    final_noise = 0.6 * base_noise + 0.3 * position_noise + 0.1 * spatial_mod
    
    return final_noise


def perform_outpainting_fusion(first_frame_occ, first_frame_bev, first_frame_meta, 
                              target_scene_shape, model, vae, diffusion, device,
                              scale_factor=70.0, use_noise_prior=True, lambda_np=0.03,
                              patch_overlap=0.5, half_mask_dir='right', 
                              diversity_strength=0.3, use_position_encoding=True):
    """
    使用is_fusion策略进行outpainting - 核心patch fusion逻辑
    
    参考lt3sd的is_fusion策略：
    1. 将第一帧放在左上角    ##右下角 -->modify
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
    first_frame_bev = first_frame_bev.to(device)
    first_frame_meta = first_frame_meta.to(device)
    
    # 初始化full scene和known mask（is_fusion核心）
    full_scene = torch.zeros((B, T, target_H, target_W, target_D), device=device, dtype=torch.long)
    known_mask = torch.zeros((B, T, target_H, target_W, target_D), device=device, dtype=torch.float32)

    # 将第一帧放置在左上角（一次外扩）
    place_h = min(orig_H, target_H)
    place_w = min(orig_W, target_W)
    place_d = min(orig_D, target_D)
    full_scene[:, :, :place_h, :place_w, :place_d] = first_frame_occ[:, :, :place_h, :place_w, :place_d]

    # 半遮罩: 仅保留一半为已知，其余即使在原区域里也设为未知用于重绘 + 扩展
    if half_mask_dir not in ['left','right','top','bottom','none']:
        half_mask_dir = 'right'
    if half_mask_dir == 'none':
        known_mask[:, :, :place_h, :place_w, :place_d] = 1.0
        print("未启用半遮罩 (half-mask-dir=none)")
    else:
        if half_mask_dir in ['left','right']:
            mid = place_w // 2
            if half_mask_dir == 'left':
                known_mask[:, :, :place_h, :mid, :place_d] = 1.0  # 保留左半
                # 右半视为未知 -> 可被重绘/扩展衔接
            else:  # right
                known_mask[:, :, :place_h, mid:place_w, :place_d] = 1.0  # 保留右半
        else:  # top/bottom
            mid = place_h // 2
            if half_mask_dir == 'top':
                known_mask[:, :, :mid, :place_w, :place_d] = 1.0
            else:
                known_mask[:, :, mid:place_h, :place_w, :place_d] = 1.0
        print(f"启用半遮罩方向: {half_mask_dir}")

    print(f"第一帧放置并应用半遮罩: 已知占比 {known_mask.mean().item():.4f}")
    
    # 获取全局patch配置
    patch_configs_all = get_patch_positions(target_scene_shape, overlap=patch_overlap)

    # 仅保留与未知区域相交的 patch，减少无效遍历
    with torch.no_grad():
        unknown_mask = (known_mask == 0).any(dim=1).any(dim=-1)  # [B, H, W]
        # 投影到 2D (H,W) 有任一帧或高度未知即视为需要
        need_coords = unknown_mask[0]  # 取 batch0 代表
    filtered_patches = []
    for (hs, he, ws, we) in patch_configs_all:
        if need_coords[hs:he, ws:we].any():
            filtered_patches.append((hs, he, ws, we))
    patch_configs = filtered_patches
    print(f"过滤后实际需要生成的patch数量: {len(patch_configs)}/{len(patch_configs_all)}")
    
    # is_fusion策略：独立生成每个patch
    for i, (patch_h_start, patch_h_end, patch_w_start, patch_w_end) in enumerate(tqdm(patch_configs, desc="生成patches")):
        print(f"处理patch {i+1}/{len(patch_configs)}: H[{patch_h_start}:{patch_h_end}], W[{patch_w_start}:{patch_w_end}]")
        
        # 当前patch的位置信息
        current_patch_pos = (patch_h_start, patch_h_end, patch_w_start, patch_w_end)
        
        # 提取当前patch区域
        patch_H = patch_h_end - patch_h_start
        patch_W = patch_w_end - patch_w_start
        patch_D = target_D
        
        # 当前区域的occupancy和mask
        current_region = full_scene[:, :, patch_h_start:patch_h_end, patch_w_start:patch_w_end, :]
        current_mask = known_mask[:, :, patch_h_start:patch_h_end, patch_w_start:patch_w_end, :]
        
        # is_fusion核心逻辑：跳过完全已知的patch
        if torch.all(current_mask == 1.0):
            continue
        
        # is_fusion策略：始终使用标准200x200x16生成，确保模型输入一致
    # print(f"  patch区域: ({patch_H}, {patch_W}, {patch_D})")
        
        # 创建标准大小的输入（200x200x16）- 使用实际的时间维度T
        actual_T = current_region.shape[1]  # 获取实际时间维度
        standard_region = torch.zeros((B, actual_T, 200, 200, 16), device=device, dtype=torch.long)
        standard_mask = torch.zeros((B, actual_T, 200, 200, 16), device=device, dtype=torch.float32)
        
        # 将实际区域复制到标准大小patch中
        actual_h = min(patch_H, 200)
        actual_w = min(patch_W, 200)
        actual_d = min(patch_D, 16)
        
        standard_region[:, :, :actual_h, :actual_w, :actual_d] = current_region[:, :, :actual_h, :actual_w, :actual_d]
        standard_mask[:, :, :actual_h, :actual_w, :actual_d] = current_mask[:, :, :actual_h, :actual_w, :actual_d]
        
        # 1. 编码标准大小的patch - 修复VAE编码问题
        # 使用所有T帧来匹配模型预期
        if torch.any(standard_mask > 0):
            with torch.no_grad():
                # 对包含已知区域的patch进行编码，确保数据类型为long
                standard_region_long = standard_region.long()
                x_encoded = vae.encode(standard_region_long) * scale_factor
        else:
            # 完全未知区域，创建标准大小的随机噪声 - 使用实际T维度
            actual_T = standard_region.shape[1]
            x_encoded = torch.randn(B, 4, actual_T, 200//4, 200//4, device=device) * scale_factor
        
        # 2. 准备模型条件和噪声 - 修复噪声先验问题
        # 准备x_ref：使用第一帧的第一个时间步作为参考（T_condition=1）
        first_frame_single = first_frame_occ[:, :1, :, :, :]  # 只取第一个时间步 [B, 1, H, W, D]
        first_frame_single_long = first_frame_single.long()
        x_ref_encoded = vae.encode(first_frame_single_long) * scale_factor  # 编码参考帧
        
        # 3. 准备模型输入 (保持原始输入不变，training-free)
        # BEV数据应该对应T_pred个时间步，与x_encoded匹配
        first_frame_bev_pred = first_frame_bev[:, :actual_T, :, :, :]  # 匹配x_encoded的时间维度
        model_kwargs = dict(y=first_frame_bev_pred, meta=first_frame_meta, x_ref=x_ref_encoded)
        
        # 创建位置感知的噪声
        base_noise = torch.randn_like(x_encoded)
        if diversity_strength > 0:
            noise = create_position_aware_noise(base_noise, current_patch_pos, target_scene_shape, i)
            # 根据diversity_strength调整噪声强度
            noise = (1 - diversity_strength) * base_noise + diversity_strength * noise
        else:
            noise = base_noise
        
        if use_noise_prior and torch.any(standard_mask > 0):
            known_ratio = standard_mask.float().mean().item()
            
            if known_ratio > 0.1: 
                adjusted_lambda = lambda_np * known_ratio
                noise = (1-adjusted_lambda) * noise + adjusted_lambda * x_encoded ##latent_mask -->modify
                pass
            else:
                pass
                
            noise = torch.clamp(noise, -3*scale_factor, 3*scale_factor)
        
        # 4. 对标准200x200x16 patch进行独立扩散采样
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
                samples = torch.clamp(samples, -scale_factor, scale_factor)
        
        # 5. VAE解码为标准200x200x16
        samples = samples / scale_factor  
        actual_T = standard_region.shape[1]  # 使用实际的时间维度
        rec_shape = [B, actual_T, 200, 200, 16] 
        
        with torch.no_grad():
            result = vae.generate(samples, rec_shape)
            logits = result["logits"]
            
            # 验证logits质量
            logit_max = logits.max().item()
            logit_min = logits.min().item()
            # if logit_max - logit_min < 1.0:  # 可选日志
            #     print(f"  警告: logits动态范围过小 [{logit_min:.2f}, {logit_max:.2f}]")
            
            standard_pred = logits.argmax(dim=-1).long()  # [B, T, 200, 200, 16]
            
            # 验证生成结果的分布
            # unique_vals = torch.unique(standard_pred).cpu().numpy()
            # print(f"  生成值分布: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
        
        # 6. is_fusion融合：
        
        patch_pred = standard_pred[:, :, :actual_h, :actual_w, :actual_d]
        
        # is_fusion核心
        current_region_actual = current_region[:, :, :actual_h, :actual_w, :actual_d]
        current_mask_actual = current_mask[:, :, :actual_h, :actual_w, :actual_d]

        fused_patch = patch_pred * (1 - current_mask_actual).long() + current_region_actual * current_mask_actual.long()
        
        # 7. 更新全局场景（只更新实际区域）
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
    use_occ_meta = False
    in_ch = 4
    Tframe = 6  # 根据调试信息，模型是基于6个时间步训练的
    T_pred = 5  # 修改为5以匹配数据
    T_condition = 1
    meta_num = 4+12*(T_pred-1)  # meta_num_mode = 3
    ref_idx = 0
    scale_factor = 70
    use_noise_prior = True
    lambda_np = args.lambda_noise_prior
    
    # 创建DiT模型 - 与评估脚本保持一致的配置
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
        "T_pred": T_pred,
        "T_condition": T_condition
    }
    model = DiT_WorldModel(**DiT_cfg).to(device)
    
    # 加载DiT checkpoint - 与评估脚本保持一致的加载方式
    if args.ckpt:
        logger.info(f"Loading DiT model from: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        if 'ema' in checkpoint:
            logger.info("Using EMA weights")
            model.load_state_dict(checkpoint['ema'], strict=False)
        elif 'model' in checkpoint:
            logger.info("Using model weights")
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            logger.info("Using direct state_dict")
            model.load_state_dict(checkpoint, strict=False)
        del checkpoint
    else:
        logger.warning("No DiT checkpoint provided!")
    
    # 创建扩散模型 - 与评估脚本保持一致使用ddim25
    diffusion = create_diffusion("ddim25")
    
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
    first_frame_occ, first_frame_bev, first_frame_meta = load_first_frame_data(args.first_frame_path, max_time_steps=T_pred)
    
    # 定义多个目标场景大小，支持多个scale factors
    original_shape = first_frame_occ.shape[2:5]  # (H, W, D)
    scale_factors = [args.scale_factor] if not hasattr(args, 'scale_factors') else args.scale_factors
    
    for scale in scale_factors:
        target_H = int(original_shape[0] * scale)
        target_W = int(original_shape[1] * scale)
        target_D = original_shape[2]  # 深度保持不变
        target_shape = (target_H, target_W, target_D)

        logger.info(f"原始形状: {original_shape}")
        logger.info(f"目标形状({scale}x扩充): {target_shape}")

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
                patch_overlap=args.patch_overlap,
                half_mask_dir=args.half_mask_dir,
                diversity_strength=args.diversity_strength,
                use_position_encoding=args.use_position_encoding
            )

            # 保存结果
            if rank == 0:
                save_dir = f"{experiment_dir}/scale_{scale}x"
                os.makedirs(save_dir, exist_ok=True)

                for b in range(generated_occ.shape[0]):
                    for t in range(generated_occ.shape[1]):
                        occ_frame = generated_occ[b, t].detach().cpu().numpy().astype(np.int8)
                        save_path = os.path.join(save_dir, f"outpainted_{scale}x_batch{b}_frame{t}.npy")
                        np.save(save_path, occ_frame)

                first_frame_np = first_frame_occ[0, 0].detach().cpu().numpy().astype(np.int8)
                ref_path = os.path.join(save_dir, "reference_first_frame.npy")
                np.save(ref_path, first_frame_np)

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
    parser.add_argument("--scale-factor", type=float, default=2.0,
                       help="单次放大倍数 (原H,W按该倍数扩充)")
    parser.add_argument("--patch-overlap", type=float, default=0.5,
                       help="patch重叠比例")
    parser.add_argument("--lambda-noise-prior", type=float, default=0.03,
                       help="噪声先验权重")
    parser.add_argument("--half-mask-dir", type=str, default='right', choices=['left','right','top','bottom','none'],
                       help="半遮罩方向: 只保留这一半为已知，其余重绘 (none 表示不裁半)")
    parser.add_argument("--diversity-strength", type=float, default=0.3,
                       help="多样性强度: 控制patch间的变化程度 (0.0-1.0)")
    parser.add_argument("--use-position-encoding", action='store_true', default=True,
                       help="是否使用位置编码增加空间多样性")
    parser.add_argument("--no-position-encoding", dest='use_position_encoding', action='store_false',
                       help="禁用位置编码")
    parser.add_argument("--no-distributed", action='store_true',
                       help="不使用分布式训练")
    parser.add_argument("--local-rank", type=int, default=0,
                       help="本地rank")
    
    args = parser.parse_args()
    main(args) 