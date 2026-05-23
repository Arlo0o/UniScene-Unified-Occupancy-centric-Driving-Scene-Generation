# 训练单帧的 unconditional 模型. 

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# 设置CUDNN优化选项
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# 抑制CUDNN相关警告（可选）
import warnings
warnings.filterwarnings("ignore", message=".*CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*")

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
import os, sys
from pathlib import Path

# 获取当前文件的父目录的父目录（上一级目录）
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.download import find_model

from diffusion.models import DiT_WorldModel
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
import matplotlib.pyplot as plt
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir,rank):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0:  # real logger
        logging.basicConfig(
            level=logging.DEBUG,
            # format='[\033[34m%(asctime)s\033[0m] %(message)s',

            datefmt='%Y%m%d_%H%M%S',
            # filename=f"{logging_dir}/log.txt"
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
            # handlers=logging.StreamHandler()
        )
        print("logger")
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def filter_state_dict(state_dict, ignore_keys):
    """
    """
    filtered_dict = OrderedDict()
    for k, v in state_dict.items():
        if k in ignore_keys:
            continue
        filtered_dict[k] = v
    return filtered_dict


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

def save_occ_bev_visualization(occ_data, save_path, title="Occupancy BEV"):
    """
    保存occupancy的BEV可视化图像
    occ_data: numpy array of shape [H, W, D] or other shapes
    """
    try:
        # 确保数据是numpy数组
        if torch.is_tensor(occ_data):
            occ_data = occ_data.cpu().numpy()
        
        # 处理不同的输入形状
        if occ_data.ndim == 3:  # [H, W, D]
            # 创建BEV图像：沿着高度维度取最大值
            bev_image = np.max(occ_data, axis=2)  # [H, W]
        elif occ_data.ndim == 4:  # [T, H, W, D] 或 [B, H, W, D]
            # 取第一个时间帧或batch
            bev_image = np.max(occ_data[0], axis=2)  # [H, W]
        elif occ_data.ndim == 5:  # [B, T, H, W, D]
            # 取第一个batch的第一个时间帧
            bev_image = np.max(occ_data[0, 0], axis=2)  # [H, W]
        else:
            print(f"==> Unsupported shape for BEV visualization: {occ_data.shape}")
            return
        
        plt.figure(figsize=(10, 10))
        plt.imshow(bev_image, cmap='viridis', origin='lower')
        plt.colorbar(label='Occupancy Class')
        plt.title(title)
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"==> Saved BEV visualization: {save_path}")
    except Exception as e:
        print(f"==> Failed to save BEV visualization: {e}")
        print(f"==> Data shape: {occ_data.shape if hasattr(occ_data, 'shape') else type(occ_data)}")


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Variables for monitoring/logging purposes:
    start_epoch = 0
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    scale_factor = 70

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    
    # 正确获取rank和local_rank
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    seed = args.global_seed * dist.get_world_size() + rank
    
    torch.manual_seed(seed)
    # 统一设备设置
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"Starting rank={rank}, local_rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    # Setup an experiment folder:
        # os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        # experiment_index = len(glob(f"{args.results_dir}/*"))
        # model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        
        # ct_str=datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
        # experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    experiment_dir = f"{args.results_dir}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if rank == 0:
    
        logger = create_logger(experiment_dir,rank)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None,rank)

    # Create model:
    
    use_bev_concat=False
    use_occ_meta=False
    in_ch=4
    T_pred = 6
    T_condition = 2
    Tframe = 5  # 添加缺失的Tframe定义
    meta_num= 4 + 12*(T_pred-1)
    use_noise_prior = False
    # bev_ch_use=[0,2,5,6,8,9,10,11,12,13,14,15,16,17]
    bev_ch_use=[0,2,8,9,10,11,12,13,14,15,16,17]
    

    DiT_cfg={"depth":12, "in_channels":in_ch, "hidden_size":512,"use_label":False, "use_bev_concat":use_bev_concat,"bev_in_ch":1,"bev_out_ch":1,"use_meta":use_occ_meta, "bev_dropout_prob":0.1,"meta_num":meta_num,"T_pred":5,
        "T_condition":1}
    # model = DiT_Occsora(**DiT_cfg).to(device)
    model = DiT_WorldModel(**DiT_cfg).to(device)
    # model = DiT_multiframe(**DiT_cfg).to(device)
    if rank==0:
        shutil.copy(os.path.abspath(__file__),experiment_dir)


    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    # model = DDP(model.to(device), device_ids=[rank])

    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
    scheduler = StepLR(opt, step_size=10000, gamma=0.9)
    # opt = torch.optim.AdamW([{"params":model.parameters()},{"params":y_net.parameters()}], lr=1e-5, weight_decay=0)

    # Load checkpoint
    if args.ckpt_preDIT:
        state_dict = find_model(args.ckpt_preDIT)
        ignore_dict={"x_embedder.proj.weight"}
        state_dict_F=filter_state_dict(state_dict,ignore_dict)
        model.load_state_dict(state_dict_F,strict=False)

    if args.ckpt:
    #      -- Re train ---
        checkpoint = torch.load(args.ckpt,map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        ema.load_state_dict(checkpoint['ema'], strict=False)
        opt.load_state_dict(checkpoint['opt'])
        del checkpoint
        logger.info(f"Using checkpoint: {args.ckpt}")


    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    
    import model_vae
    cfg = Config.fromfile(args.vae_config)
    vae = MODELS.build(cfg.model)
    vae_ckpt = torch.load(args.vae_ckpt,map_location='cpu')
    vae.load_state_dict(vae_ckpt['state_dict'], strict=True)
    vae = vae.to(device)
    # 移除重复的设备设置代码，使用之前设置的device和local_rank
    model = DDP(model.to(device), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    imageset = "/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_train.pkl"
    bev_path = '/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200'
    gts_path = "/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ/nuplan_quantized_200_200_16"

    dataset = Nuplan_Occbev_bev_Dataset(imageset, gts_path, bev_path, bev_ch_use, meta_num=4, Tframe=5, training=True)

    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size if args.batch_size else int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")


   # Initial state
    if args.ckpt:
        train_steps = int(args.ckpt.split('/')[-1].split('.')[0])
        start_epoch = int(train_steps / (len(dataset) / args.global_batch_size))
        logger.info(f"Initial state: step={train_steps}, epoch={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        # 只在没有checkpoint时初始化EMA
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights    

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    save_occ_dir = f"{experiment_dir}/occ_vis"
    os.makedirs(save_occ_dir, exist_ok=True)

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for i_iter, (occ_ori, occ_gt, y, occ_meta, pose_meta) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            occ_ori = occ_ori.to(device)
            occ_ori = fill_large_zero_height_regions_gpu(occ_ori, target_layer=8, fill_class=1)

            with torch.no_grad():
                x=vae.encode(occ_ori)*scale_factor # bs, 4, t, 50, 50
                x_ref = x[:, :, 0:1, :, :] # bs, 4,1, 50, 50
                # x_ref = None # bs, 4,1, 50, 50
                
                # 添加解码查看步骤：每1000步保存一次解码结果
                if train_steps % 1000 == 0 and rank == 0:
                    try:
                        # 重建VAE输入：需要扩展单帧到5帧来匹配VAE的期望输入
                        batch_size = x.shape[0]
                        Tframe = 5  # VAE期望的帧数
                        # 使用VAE的generate方法解码
                        rec_shape = [batch_size, Tframe, 200, 200, 16]
                        result = vae.generate(x / scale_factor, rec_shape)
                        logit = result["logits"]
                        decoded_original = logit.argmax(dim=-1)  # [bs, 5, 200, 200, 16]
                        print(f"==> decoded_original shape: {decoded_original.shape}")
                        
                        # 保存解码结果用于可视化
                        save_dir = f"{experiment_dir}/decoded_vis"
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # 保存第一个batch的第一帧结果
                        # decoded_original shape: [batch_size, time_frames, H, W, D] -> 取第一个batch的第一帧
                        decoded_np = decoded_original[0, 0].cpu().numpy().astype(np.int8)  # [200, 200, 16]
                        original_np = occ_ori[0].cpu().numpy().astype(np.int8)  # [200, 200, 16] 原始输入的第一帧
                        
                        # 保存numpy文件 (按照参考代码的格式)
                        np.save(f"{save_dir}/step_{train_steps:07d}_decoded.npy", decoded_np)
                        np.save(f"{save_dir}/step_{train_steps:07d}_original.npy", original_np)
                        
                        # 保存BEV可视化图像
                        save_occ_bev_visualization(
                            decoded_np, 
                            f"{save_dir}/step_{train_steps:07d}_decoded_bev.png",
                            f"Decoded Occupancy BEV (Step {train_steps})"
                        )
                        save_occ_bev_visualization(
                            original_np, 
                            f"{save_dir}/step_{train_steps:07d}_original_bev.png",
                            f"Original Occupancy BEV (Step {train_steps})"
                        )
                        
                        logger.info(f"Saved decoded occupancy at step {train_steps}")
                    except Exception as e:
                        logger.warning(f"Failed to save visualization at step {train_steps}: {e}")
                        print(f"==> Visualization error: {e}")
                        # 不中断训练，继续执行

            x = x.to(device)
            y = y.to(device)
            x_ref = x_ref.to(device)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            if use_occ_meta:  # 这个分支不会执行，因为use_occ_meta=False
                occ_meta = occ_meta.to(device)
                pose_meta = pose_meta.to(device)
                ref_idx = 0
                new_meta = torch.cat((occ_meta[:,ref_idx],pose_meta.reshape(-1,12*(T_pred-1))),dim=1)
                new_meta = new_meta.to(device)
                model_kwargs = dict(x_ref=x_ref, y=y, meta=new_meta)
            else:
                model_kwargs = dict(x_ref=x_ref, y=y)

            
            lambda_z = 0.3#0.03
            # z_ref = 0
            noise_prior = torch.randn_like(x) + lambda_z * x

            if use_noise_prior:
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs, noise=noise_prior)
            else:
                print(f"model_kwargs.keys(){model_kwargs.keys()}")
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

            if args.confidence:
                occ_meta = occ_meta.to(device)
                w_loss = loss_dict["loss"] * occ_meta.squeeze()
                # print(w_loss.shape)
                loss=w_loss.mean()
            else:
                loss = loss_dict["loss"].mean()
            

            opt.zero_grad()
            loss.backward()
            opt.step()

            scheduler.step()
            update_ema(ema, model.module)
            
            # 添加模型生成结果的解码查看：每2000步生成一次样本
            if train_steps % 20 == 0 and train_steps > 0 and rank == 0:
                with torch.no_grad():
                    print(f"==> Generating samples at step {train_steps}")
                    # 使用EMA模型生成样本
                    ema.eval()
                    
                    # 准备采样参数
                    sample_batch_size = 2  # 生成两个样本
                    sample_shape = (sample_batch_size, x.shape[1], x.shape[2], x.shape[3], x.shape[4])  # [2, 4, 5, 50, 50]
                    
                    # 从纯噪声开始
                    sample_noise = torch.randn(sample_shape, device=device)
                    
                    # 使用当前批次的条件作为采样条件
                    sample_x_ref = x_ref[:sample_batch_size]  # [2, 4, 1, 50, 50] - 参考帧
                    sample_y = y[:sample_batch_size]          # [2, ...] - 可能是BEV或其他条件，但我们不使用
                    
                    # 准备CFG采样条件（只使用x_ref，不使用BEV）
                    sample_noise_cfg = torch.cat([sample_noise, sample_noise], 0)  # [4, 4, 5, 50, 50]
                    sample_x_ref_null = torch.zeros_like(sample_x_ref)
                    sample_x_ref_cfg = torch.cat([sample_x_ref, sample_x_ref_null], 0)  # [4, 4, 1, 50, 50]
                    
                    # 只使用x_ref作为条件，不使用BEV（y可能是占位符或未使用）
                    if use_bev_concat:  # 如果使用BEV
                        sample_y_null = torch.zeros_like(sample_y) if sample_y is not None else None
                        sample_y_cfg = torch.cat([sample_y, sample_y_null], 0) if sample_y is not None else None
                        sample_kwargs = dict(x_ref=sample_x_ref_cfg, y=sample_y_cfg, cfg_scale=2.0)
                    else:  # 不使用BEV
                        sample_kwargs = dict(x_ref=sample_x_ref_cfg, y=sample_y[:sample_batch_size], cfg_scale=2.0)
                    
                    # 创建简化的DDIM采样器
                    sample_diffusion = create_diffusion('ddim25')  # 25步DDIM采样
                    
                    try:
                        # 使用DDIM采样
                        generated_latent = sample_diffusion.ddim_sample_loop(
                            ema.forward_with_cfg,
                            sample_noise_cfg.shape,
                            sample_noise_cfg,
                            clip_denoised=False,
                            model_kwargs=sample_kwargs,
                            progress=False,
                            device=device
                        )
                        
                        # 只使用conditional的结果
                        generated_latent, _ = generated_latent.chunk(2, dim=0)  # [2, 4, 5, 50, 50]
                        
                        print(f"==> Generated latent shape: {generated_latent.shape}")
                        print(f"==> Generated latent range: [{generated_latent.min().item():.3f}, {generated_latent.max().item():.3f}]")
                        
                        # 解码生成的潜在表示
                        rec_shape = [sample_batch_size, Tframe, 200, 200, 16]
                        result = vae.generate(generated_latent / scale_factor, rec_shape)
                        logit = result["logits"]
                        generated_occ = logit.argmax(dim=-1)  # [2, 5, 200, 200, 16]
                        
                        # 保存生成结果
                        save_dir = f"{experiment_dir}/generated_samples"
                        os.makedirs(save_dir, exist_ok=True)
                        
                        for sample_idx in range(sample_batch_size):
                            # 保存latent
                            np.save(f"{save_dir}/step_{train_steps:07d}_sample_{sample_idx}_latent.npy", 
                                   generated_latent[sample_idx].cpu().numpy())
                            
                            # 保存occupancy
                            np.save(f"{save_dir}/step_{train_steps:07d}_sample_{sample_idx}_occ.npy", 
                                   generated_occ[sample_idx].cpu().numpy())
                            
                            # 保存条件（只保存参考帧）
                            np.save(f"{save_dir}/step_{train_steps:07d}_sample_{sample_idx}_x_ref.npy", 
                                   sample_x_ref[sample_idx].cpu().numpy())
                            
                            # 保存所有帧的occupancy BEV投影可视化
                            for frame_idx in range(Tframe):
                                save_occ_bev_visualization(
                                    generated_occ[sample_idx, frame_idx].cpu().numpy(), 
                                    f"{save_dir}/step_{train_steps:07d}_sample_{sample_idx}_frame{frame_idx}_occ.png",
                                    f"Generated Occupancy Sample {sample_idx} Frame {frame_idx} (Step {train_steps})"
                                )
                            
                            # 保存参考帧的occupancy可视化
                            try:
                                ref_decoded = vae.generate(sample_x_ref[sample_idx:sample_idx+1] / scale_factor, [1, 1, 200, 200, 16])
                                ref_occ = ref_decoded["logits"].argmax(dim=-1)[0, 0]  # [200, 200, 16]
                                save_occ_bev_visualization(
                                    ref_occ.cpu().numpy(), 
                                    f"{save_dir}/step_{train_steps:07d}_sample_{sample_idx}_ref_condition_occ.png",
                                    f"Reference Condition Occupancy Sample {sample_idx} (Step {train_steps})"
                                )
                            except Exception as e:
                                print(f"Failed to decode reference frame: {e}")
                        
                        # 统计生成结果的类别分布
                        unique_classes = torch.unique(generated_occ).cpu().numpy()
                        class_counts = [(c, (generated_occ == c).sum().item()) for c in unique_classes]
                        
                        # 保存统计信息到文件
                        stats_file = f"{save_dir}/step_{train_steps:07d}_stats.txt"
                        with open(stats_file, 'w') as f:
                            f.write(f"Generated samples at step {train_steps}\n")
                            f.write(f"Sample shape: {generated_occ.shape}\n")
                            f.write(f"Latent shape: {generated_latent.shape}\n")
                            f.write(f"Unique classes: {unique_classes}\n")
                            f.write(f"Class distribution: {class_counts}\n")
                            f.write(f"Latent stats: mean={generated_latent.mean().item():.3f}, std={generated_latent.std().item():.3f}\n")
                            f.write(f"CFG scale used: 2.0\n")
                            f.write(f"DDIM steps: 25\n")
                            f.write(f"Conditions used: x_ref (reference frame) only, no BEV\n")
                        
                        logger.info(f"Generated samples at step {train_steps}:")
                        logger.info(f"  Unique classes: {unique_classes}")
                        logger.info(f"  Class distribution: {class_counts}")
                        logger.info(f"  Latent stats: mean={generated_latent.mean().item():.3f}, std={generated_latent.std().item():.3f}")
                        logger.info(f"  Stats saved to: {stats_file}")
                        
                        # 创建一个综合的时间序列可视化
                        try:
                            import matplotlib.pyplot as plt
                            fig, axes = plt.subplots(sample_batch_size, Tframe, figsize=(Tframe*3, sample_batch_size*3))
                            if sample_batch_size == 1:
                                axes = axes.reshape(1, -1)
                            
                            for sample_idx in range(sample_batch_size):
                                for frame_idx in range(Tframe):
                                    occ_frame = generated_occ[sample_idx, frame_idx].cpu().numpy()
                                    occ_bev = np.max(occ_frame, axis=2)  # Occupancy BEV projection
                                    
                                    ax = axes[sample_idx, frame_idx]
                                    im = ax.imshow(occ_bev, cmap='viridis', origin='lower')
                                    ax.set_title(f'Occ Sample {sample_idx} Frame {frame_idx}')
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            
                            plt.suptitle(f'Generated Occupancy Sequence (Step {train_steps})', fontsize=16)
                            plt.tight_layout()
                            plt.savefig(f"{save_dir}/step_{train_steps:07d}_sequence_overview.png", dpi=150, bbox_inches='tight')
                            plt.close()
                            
                        except Exception as e:
                            print(f"Failed to create sequence visualization: {e}")
                        
                        print(f"==> Saved generated samples to {save_dir}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate samples at step {train_steps}: {e}")
                        print(f"==> Sampling failed: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    ema.train()  # 切回训练模式
 
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if rank==0:
                    with open(f"{experiment_dir}/log.log", 'a') as file:
                        current_lr = opt.param_groups[0]['lr']
                        print(f"(step={train_steps:07d}) lr:{current_lr:.6f} Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}", file=file)
                    
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    # torch.save(y_net.module.state_dict(),f"{checkpoint_dir}/BEV_cond_{train_steps:07d}.pt")
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="path")
    parser.add_argument("--results-dir", type=str, default="out/wm_results")
    parser.add_argument("--image-size", type=int, choices=[256, 128], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--global-batch-size", type=int, default=72)  #72
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--vae-ckpt", type=str, default=None)
    parser.add_argument("--vae-config", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpt-preDIT", type=str, default=None)
    parser.add_argument("--confidence", type=int,default=0)
    parser.add_argument("--local-rank", type=int,default=0)
    parser.add_argument("--batch-size", type=int,default=None)
    args = parser.parse_args()
    main(args)

    # 正确的启动命令示例:
    # torchrun --nproc_per_node=8 train_OccDiT_nuplan_outpainting_concat.py
    # 或者旧版本: python -m torch.distributed.launch --nproc_per_node=8 train_OccDiT_nuplan_outpainting_concat.py 