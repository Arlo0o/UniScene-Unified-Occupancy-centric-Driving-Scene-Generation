# Evaluation script for Occupancy Outpainting DiT model
# 基于outpainting训练脚本修改的评估版本

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.download import find_model
from diffusion.models import DiT_models, DiT_2Frame
from diffusion import create_diffusion
from tqdm import tqdm
import datetime
import shutil
from dataset.dataload_util import Nuplan_Occbev_bev_Dataset
import random
from mmengine import Config
from mmengine.registry import MODELS

from dataset import get_nuScenes_label_name
from utils.metric_util import MeanIoU, multi_step_MeanIou, multi_step_fid_mmd, multi_step_TemporalConsistency

def create_outpainting_mask(x, mask_ratio=0.5, mask_type="center"):
    """
    创建outpainting的mask
    
    Args:
        x: [B, C, T, H, W] 输入tensor
        mask_ratio: 保留区域的比例
        mask_type: "center" (保留中心) 或 "random" (随机保留)
    
    Returns:
        known_mask: True表示已知区域，False表示需要生成的区域
    """
    B, C, T, H, W = x.shape
    known_mask = torch.zeros_like(x, dtype=torch.bool)
    
    if mask_type == "center":
        # 保留中心区域
        center_h, center_w = H // 2, W // 2
        keep_h = int(H * mask_ratio ** 0.5)
        keep_w = int(W * mask_ratio ** 0.5)
        
        h_start = center_h - keep_h // 2
        h_end = h_start + keep_h
        w_start = center_w - keep_w // 2
        w_end = w_start + keep_w
        
        known_mask[:, :, :, h_start:h_end, w_start:w_end] = True
        
    elif mask_type == "random":
        # 随机保留区域
        total_pixels = H * W
        keep_pixels = int(total_pixels * mask_ratio)
        
        for b in range(B):
            for t in range(T):
                # 为每个batch和时间步创建随机mask
                flat_mask = torch.zeros(H * W, dtype=torch.bool)
                keep_indices = torch.randperm(H * W)[:keep_pixels]
                flat_mask[keep_indices] = True
                known_mask[b, :, t, :, :] = flat_mask.reshape(H, W).unsqueeze(0).expand(C, -1, -1)
    
    return known_mask

def fill_large_zero_height_regions_gpu(occ_volume, target_layer=0, fill_class=15, reverse_height=False):
    """保持与原始代码一致的填充函数"""
    B, T, H, W, D = occ_volume.shape
    device = occ_volume.device
    empty_mask = (occ_volume == 0).all(dim=-1)
    z_indices = torch.arange(D, device=device).view(1, 1, 1, 1, D)
    layer_mask = (z_indices == target_layer)
    fill_mask = layer_mask & empty_mask.unsqueeze(-1)
    result = occ_volume.clone()
    result[fill_mask] = fill_class
    return result

def cleanup():
    """
    End DDP evaluation.
    """
    dist.destroy_process_group()

def create_logger(logging_dir, rank):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0:  # real logger
        logging.basicConfig(
            level=logging.DEBUG,
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def main(args):
    """
    评估Occupancy outpainting模型
    """
    assert torch.cuda.is_available(), "Evaluation currently requires at least one GPU."
    
    torch.set_grad_enabled(False)  # 评估时禁用梯度计算
    
    # DDP设置
    dist.init_process_group("nccl")
    rank = args.local_rank
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    # 实验文件夹设置
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        ct_str = f"dit_outpainting-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_dir = f"{args.results_dir}/{ct_str}"
        os.makedirs(experiment_dir, exist_ok=True)
        vis_dir = f"{experiment_dir}/visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        logger = create_logger(experiment_dir, rank)
        logger.info(f"Evaluation directory created at {experiment_dir}")
    else:
        logger = create_logger(None, rank)
    
    # 模型配置 - 适应outpainting
    use_bev_concat = False
    use_occ_meta = False  # outpainting通常不需要外部条件
    in_ch = 4
    Tframe = 5
    use_vae = True
    scale_factor = 30
    use_noise_prior = True  # 使用已知区域作为先验
    lambda_np = args.lambda_noise_prior
    
    # outpainting特定参数
    mask_ratio = 0.5  # 增加到50%的区域作为已知条件
    mask_type = "center"  # 保留中心区域
    
    DiT_cfg = {
        "depth": 12, "in_channels": in_ch, "hidden_size": 512,
        "use_label": False, "use_bev_concat": use_bev_concat,
        "bev_in_ch": 1, "bev_out_ch": 1, "use_meta": use_occ_meta,
        "bev_dropout_prob": 0.1, "meta_num": 4, "direct_concat": True,
        "Tframe": Tframe
    }
    
    model = DiT_2Frame(**DiT_cfg).to(device)
    
    # 加载检查点
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        logger.info(f"Checkpoint keys: {checkpoint.keys()}")
        
        # 选择加载model或ema权重
        if args.use_ema and 'ema' in checkpoint:
            model.load_state_dict(checkpoint['ema'], strict=False)
            logger.info(f"Using EMA weights from: {args.ckpt}")
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
            logger.info(f"Using model weights from: {args.ckpt}")
        else:
            raise ValueError(f"No valid model weights found in checkpoint: {list(checkpoint.keys())}")
        
        # 打印检查点信息，确认是否为outpainting模型
        if 'args' in checkpoint:
            ckpt_args = checkpoint['args']
            logger.info(f"Checkpoint args: {ckpt_args}")
        
    else:
        raise ValueError("Must specify --ckpt for evaluation")

    model.eval()  # 设置为评估模式

    # VAE加载
    if use_vae:
        import model_vae
        cfg = Config.fromfile(args.vae_config)
        vae = MODELS.build(cfg.model)
        vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu')
        vae.load_state_dict(vae_ckpt['state_dict'], strict=True)
        vae = vae.to(device)
        vae.eval()
    
    # 创建diffusion
    diffusion = create_diffusion(timestep_respacing=str(args.num_sampling_steps))
    
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 数据集设置
    imageset = "/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_train.pkl"
    bev_path = '/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200'
    gts_path = "/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ/nuplan_quantized_200_200_16"
    bev_ch_use = [1,2,8,9,10,11,12,13,14,15,17]
    
    dataset = Nuplan_Occbev_bev_Dataset(imageset, gts_path, bev_path, bev_ch_use, meta_num=4, Tframe=Tframe, training=False)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=False, seed=args.global_seed)
    loader = DataLoader(dataset,    
                        # batch_size=int(args.dit_batch_size // dist.get_world_size()), 
                        batch_size=1, 
                       shuffle=False, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    
    logger.info(f"Dataset contains {len(dataset):,} samples")
    

    # 评估循环
    logger.info(f"开始outpainting评估，mask_ratio={mask_ratio}, mask_type={mask_type}")
    start_time = time()
    
    with torch.no_grad():
        for i_iter, (occ_ori, occ_gt, y, occ_meta, pose_meta) in enumerate(tqdm(loader, desc="Evaluating")):
            if args.max_eval_samples > 0 and i_iter >= args.max_eval_samples:
                break
                
            occ_ori = occ_ori.to(device)
            occ_gt = occ_gt.to(device)
            occ_ori = fill_large_zero_height_regions_gpu(occ_ori, target_layer=8)
            y = y.to(device)
            
            # VAE编码
            if use_vae:
                x = vae.encode(occ_ori) * scale_factor  # [B, C, T, H, W]
            
            # 创建outpainting mask
            known_mask = create_outpainting_mask(x, mask_ratio=mask_ratio, mask_type=mask_type)
            unknown_mask = ~known_mask  # 需要生成的区域
            
            # 调试信息
            if i_iter == 0 and rank == 0:
                logger.info(f"Input shape: {x.shape}")
                logger.info(f"Known mask ratio: {known_mask.float().mean():.3f}")
                logger.info(f"Unknown mask ratio: {unknown_mask.float().mean():.3f}")
                logger.info(f"VAE latent stats - mean: {x.mean():.3f}, std: {x.std():.3f}")
            
            # 创建先验噪声：改进的策略
            if use_noise_prior:
                noise_base = torch.randn_like(x)
                # 已知区域使用强先验，未知区域使用弱先验
                # 增加已知区域的影响权重
                known_prior = x * 0.8  # 增加已知区域的权重
                unknown_prior = noise_base * 0.1  # 减少未知区域的随机性
                noise_prior = torch.where(known_mask, known_prior, unknown_prior)
                
                # 添加全局的弱先验，帮助空间连续性
                global_prior = torch.mean(x[known_mask]).item()
                noise_prior = noise_prior + global_prior * 0.1
            else:
                noise_prior = torch.randn_like(x)
            
            # 模型输入：不使用外部条件（纯outpainting）
            model_kwargs = dict(y=y)
            
            # 使用DDIM采样生成outpainting结果
            samples = diffusion.ddim_sample_loop(
                model,
                x.shape,
                noise=noise_prior,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                progress=False,
                device=device
            )
            
            # 将已知区域替换为原始值（确保已知部分不变）
            # 使用更强的约束确保已知区域完全保持原值
            samples = torch.where(known_mask, x, samples)
            
            # 调试：检查采样结果
            if i_iter == 0 and rank == 0:
                logger.info(f"Samples stats - mean: {samples.mean():.3f}, std: {samples.std():.3f}")
                logger.info(f"Known region preserved: {torch.allclose(samples[known_mask], x[known_mask])}")
            
            # VAE解码
            if use_vae:
                samples = samples / scale_factor
                # 解码每一帧
                B, C, T, H, W = samples.shape
                decoded_samples = []
                for t in range(T):
                    frame_sample = samples[:, :, t, :, :]  # [B, C, H, W]
                    # 创建输入形状用于解码器
                    input_shape = (B, 1, H*4, W*4, 16)  # 根据你的配置调整
                    decoded_frame = vae.generate(frame_sample.unsqueeze(2), input_shape)['logits']
                    decoded_samples.append(decoded_frame.argmax(dim=-1).squeeze(1))  # [B, H, W, D]
                
                pred_occ = torch.stack(decoded_samples, dim=1)  # [B, T, H, W, D]
            else:
                pred_occ = samples.permute(0, 2, 3, 4, 1).argmax(dim=-1)



            # 可视化（每隔一定间隔保存）
            print(f"i_iter: {i_iter}")
            if args.vis and i_iter % args.vis_every == 0:
                sample_idx = 0  # 保存batch中的第一个样本
                
                # 保存真实值
                gt_save_path = f"{vis_dir}/{i_iter:05d}_gt.npy"
                np.save(gt_save_path, occ_gt[sample_idx].cpu().numpy())
                
                # 保存预测值
                pred_save_path = f"{vis_dir}/{i_iter:05d}_pred.npy"
                np.save(pred_save_path, pred_occ[sample_idx].cpu().numpy())
                
                # 保存原始输入
                ori_save_path = f"{vis_dir}/{i_iter:05d}_ori.npy"
                np.save(ori_save_path, occ_ori[sample_idx].cpu().numpy())
                
                # 保存mask信息
                print(f"occ_gt.shape: {occ_gt.shape}, pred.shape: {pred_occ.shape}, ori.shape: {occ_ori.shape}")

                if use_vae:
                    # 将latent space的mask转换回原始空间进行可视化
                    mask_vis = known_mask[sample_idx, 0].cpu().numpy()  # [T, H, W]
                    mask_save_path = f"{vis_dir}/{i_iter:05d}_mask.npy"
                    np.save(mask_save_path, mask_vis)

    logger.info("Outpainting evaluation done!")
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="outputs/eval_outpainting")
    parser.add_argument("--dit-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--vae_config", type=str, required=True)
    parser.add_argument("--lambda_noise_prior", type=float, default=0.3)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--use-ema", action='store_true', default=False, help="Use EMA weights instead of model weights")
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--vis", action='store_true', default=False)
    parser.add_argument("--vis-every", type=int, default=10)
    parser.add_argument("--max-eval-samples", type=int, default=-1, help="Maximum number of samples to evaluate (-1 for all)")
    
    # Outpainting特定参数
    # parser.add_argument("--mask-ratio", type=float, default=0.25, help="Ratio of known region")
    # parser.add_argument("--mask-type", type=str, default="center", choices=["center", "random"], help="Type of mask")
    
    args = parser.parse_args()
    main(args)