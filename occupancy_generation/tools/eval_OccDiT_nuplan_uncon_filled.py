# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluation script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 将父目录添加到 sys.path 中

from utils.download import find_model

from diffusion.models import DiT_models, DiT_uncon, DiT_multiframe, DiT_2Frame
from diffusion import create_diffusion
from tqdm import tqdm
# from diffusers.models import AutoencoderKL

import datetime
import shutil

from dataset.dataload_util import Nuplan_Occbev_bev_Dataset
from torch.optim.lr_scheduler import StepLR
import random

from mmengine import Config
from mmengine.registry import MODELS

from dataset import get_nuScenes_label_name
from utils.metric_util import MeanIoU, multi_step_MeanIou,multi_step_fid_mmd,multi_step_TemporalConsistency

#################################################################################
#                             Evaluation Helper Functions                       #
#################################################################################

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir, rank):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0:  # real logger
        logging.basicConfig(
            level=logging.DEBUG,
            # format='[\033[34m%(asctime)s\033[0m] %(message)s',

            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def fill_large_zero_height_regions_gpu(
    occ_volume, 
    target_layer=0,        # 目标填充层（索引位置）
    fill_class=15,
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

#################################################################################
#                                  Evaluation Loop                              #
#################################################################################

def main(args):
    """
    Evaluates a DiT model.
    """
    assert torch.cuda.is_available(), "Evaluation currently requires at least one GPU."

    torch.set_grad_enabled(False)  # 评估时禁用梯度计算

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = args.local_rank
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        ct_str = f"eval_uncon_filled-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        experiment_dir = f"{args.results_dir}/{ct_str}"
        os.makedirs(experiment_dir, exist_ok=True)
        vis_dir = f"{experiment_dir}/visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        logger = create_logger(experiment_dir, rank)
        logger.info(f"Evaluation directory created at {experiment_dir}")
    else:
        logger = create_logger(None, rank)

    # Create model:
    use_bev_concat = False
    use_occ_meta = False  # 不使用外部条件（BEV、pose等），但保留参考帧先验
    in_ch = 4
    Tframe = 5
    meta_num_mode = 3
    use_vae = True
    
    if meta_num_mode == 1:
        meta_num = 4*Tframe + 12*(Tframe-1)
    elif meta_num_mode == 2:
        meta_num = 4 + 12*(Tframe-1)
    elif meta_num_mode == 3:
        meta_num = 4

    ref_idx = 0  # 参考帧索引，用于forecasting
    scale_factor = 30
    use_noise_prior = True  # 使用参考帧作为噪声先验，用于forecasting
    lambda_np = args.lambda_noise_prior
    bev_ch_use = [1,2,8,9,10,11,12,13,14,15,17]

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
        "Tframe": Tframe
    }
    
    model = DiT_2Frame(**DiT_cfg).to(device)
    
    # Load checkpoint
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
    else:
        raise ValueError("Must specify --ckpt for evaluation")

    model.eval()  # 设置为评估模式

    diffusion = create_diffusion('ddim50')  # default: 1000 steps, linear noise schedule
    # diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    
    # 加载VAE
    if use_vae:
        import model_vae
        cfg = Config.fromfile(args.vae_config)
        vae = MODELS.build(cfg.model)
        vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu')
        vae.load_state_dict(vae_ckpt['state_dict'], strict=True)
        vae = vae.to(device)
        vae.eval()
    
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")



    imageset = "/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_train.pkl"
    bev_path = '/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200'
    gts_path = "/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ/nuplan_quantized_200_200_16"

    dataset = Nuplan_Occbev_bev_Dataset(imageset, gts_path, bev_path, bev_ch_use, meta_num=4, Tframe=Tframe, training=False)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,  # 评估时不打乱
        seed=args.global_seed
    )
    
    loader = DataLoader(
        dataset,
        # batch_size=int(args.dit_batch_size // dist.get_world_size()),
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False  # 评估时不丢弃最后一个batch
    )
    
    logger.info(f"Dataset contains {len(dataset):,} samples")

    # 这是评估代码，不需要训练相关的设置
    logger.info(f"Model loaded for evaluation from: {args.ckpt}")
    
    # 确保模型处于评估模式
    model.eval()

    # 评估循环
    logger.info("Starting evaluation...")
    start_time = time()
    
    with torch.no_grad():
        for i_iter, (occ_ori, occ_gt, y, occ_meta, pose_meta) in enumerate(tqdm(loader, desc="Evaluating")):
            if args.max_eval_samples > 0 and i_iter >= args.max_eval_samples:
                break
                
            occ_ori = occ_ori.to(device)
            occ_gt = occ_gt.to(device)
            occ_ori = fill_large_zero_height_regions_gpu(occ_ori, target_layer=8)
            y = y.to(device)

            if use_vae:
                # VAE编码
                x = vae.encode(occ_ori) * scale_factor
                z_ref = x[:, :, ref_idx].unsqueeze(2)  # 取出参考帧
                z_ref = z_ref.repeat(1, 1, Tframe, 1, 1)  # 复制到所有时间步作为先验
            else:
                x = occ_ori.permute(0, 4, 1, 2, 3) * scale_factor
                z_ref = x[:, :, ref_idx].unsqueeze(2)
                z_ref = z_ref.repeat(1, 1, Tframe, 1, 1)

            # 准备模型输入
            if use_occ_meta:
                if meta_num_mode == 1:
                    new_meta = torch.cat((occ_meta.reshape(-1, 4*Tframe), pose_meta.reshape(-1, 12*(Tframe-1))), dim=1)
                elif meta_num_mode == 2:
                    new_meta = torch.cat((occ_meta[:, ref_idx], pose_meta.reshape(-1, 12*(Tframe-1))), dim=1)
                elif meta_num_mode == 3:
                    new_meta = occ_meta[:, ref_idx]
                new_meta = new_meta.to(device)
                model_kwargs = dict(y=y, meta=new_meta)
            else:
                model_kwargs = dict(y=y)

            # 生成样本
            if use_noise_prior:
                noise = torch.randn_like(x) + lambda_np * z_ref
            else:
                noise = torch.randn_like(x)

            # 使用DDIM采样
            samples = diffusion.ddim_sample_loop(
                model,
                x.shape,
                noise=noise,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                progress=False,
                device=device
            )

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

            # 更新评估指标
            # multi_step_iou.update(pred_occ.cpu(), occ_gt.cpu())
            # multi_step_fid.update(pred_occ.cpu(), occ_gt.cpu())
            # temporal_consistency.update(pred_occ.cpu())

            # 可视化（每隔一定间隔保存）
            if args.vis and i_iter % args.vis_every == 0 and rank == 0:
                # 保存原始和预测的occupancy
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

    # # 计算最终指标
    # if rank == 0:
    #     end_time = time()
    #     eval_time = end_time - start_time
        
    #     # 获取IoU结果
    #     iou_results = multi_step_iou.compute()
    #     logger.info("=== IoU Results ===")
    #     for step, iou_dict in iou_results.items():
    #         logger.info(f"Step {step}: mIoU = {iou_dict['mIoU']:.4f}")
    #         for class_name, iou in iou_dict['class_iou'].items():
    #             logger.info(f"  {class_name}: {iou:.4f}")

    #     # 获取FID/MMD结果
    #     fid_results = multi_step_fid.compute()
    #     logger.info("=== FID/MMD Results ===")
    #     for step, fid_dict in fid_results.items():
    #         logger.info(f"Step {step}: FID = {fid_dict['fid']:.4f}, MMD = {fid_dict['mmd']:.4f}")

    # 评估完成
    if rank == 0:
        end_time = time()
        eval_time = end_time - start_time
        logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        logger.info(f"Processed {len(dataset)} samples")

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="outputs/eval_uncon_filled")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 128], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--dit-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--vae_config", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--use-ema", action='store_true', default=False, help="Use EMA weights instead of model weights")
    parser.add_argument("--lambda_noise_prior", type=float, default=0.3)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--vis", action='store_true', default=False)
    parser.add_argument("--vis-every", type=int, default=10)
    parser.add_argument("--max-eval-samples", type=int, default=-1, help="Maximum number of samples to evaluate (-1 for all)")
    
    args = parser.parse_args()
    main(args)