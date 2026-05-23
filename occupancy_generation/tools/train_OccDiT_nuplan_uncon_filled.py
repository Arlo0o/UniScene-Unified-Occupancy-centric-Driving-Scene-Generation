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
from torch.utils.tensorboard import SummaryWriter
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

            datefmt='%Y-%m-%d %H:%M:%S',
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



    # Setup DDP:
    dist.init_process_group("nccl")
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    rank=args.local_rank
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        
        ct_str=f"dit_uncon-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}" 
        experiment_dir = f"{args.results_dir}/{ct_str}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        # 添加tensorboard writer
        tensorboard_dir = f"{experiment_dir}/tensorboard"
        writer = SummaryWriter(tensorboard_dir)
        logger = create_logger(experiment_dir,rank)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None,rank)

    # Create model:
    
    use_bev_concat=False
    use_occ_meta=False  # 不使用外部条件（BEV、pose等），但保留参考帧先验
    in_ch=4
    Tframe=5
    meta_num_mode = 3
    use_vae = True
    if meta_num_mode == 1:
        meta_num=4*Tframe + 12*(Tframe-1)
    elif meta_num_mode == 2:
        meta_num= 4 + 12*(Tframe-1)
    elif meta_num_mode == 3:
        meta_num = 4

    ref_idx = 0  # 参考帧索引，用于forecasting（通常使用第0帧作为已知条件预测后续帧）
    # scale_factor = 70
    scale_factor = 30
    use_noise_prior = True  # 使用参考帧作为噪声先验，用于forecasting
    lambda_np = args.lambda_noise_prior
    # bev_ch_use=[0,2,5,6,8,9,10,11,12,13,14,15,16,17]
    bev_ch_use=[1,2,8,9,10,11,12,13,14,15,17]

    DiT_cfg={"depth":12, "in_channels":in_ch, "hidden_size":512,"use_label":False, "use_bev_concat":use_bev_concat,"bev_in_ch":1,"bev_out_ch":1,"use_meta":use_occ_meta, "bev_dropout_prob":0.1,"meta_num":meta_num,"direct_concat":True,"Tframe":Tframe}
    # model = DiT_Occsora(**DiT_cfg).to(device)
    model = DiT_2Frame(**DiT_cfg).to(device)
    
    if rank == 0:
        current_file = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
        shutil.copy(current_file, experiment_dir)  # 复制当前脚本到 experiment_dir


    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    # model = DDP(model.to(device), device_ids=[rank])

    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
    scheduler = StepLR(opt, step_size=10000, gamma=0.9)

    # opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.001)
    # scheduler = StepLR(opt, step_size=50000, gamma=0.9)

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
        # opt.load_state_dict(checkpoint['opt'])
        del checkpoint
        logger.info(f"Using checkpoint: {args.ckpt}")


    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    
    if use_vae:
        import model_vae
        cfg = Config.fromfile(args.vae_config)
        vae = MODELS.build(cfg.model)
        vae_ckpt = torch.load(args.vae_ckpt,map_location='cpu')
        vae.load_state_dict(vae_ckpt['state_dict'], strict=True)
        vae = vae.to(device)
        vae.eval()
    
    model = DDP(model.to(device), device_ids=[rank],find_unused_parameters=True)
    
    
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")



    imageset = "/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_train.pkl"
    bev_path = '/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200'
    gts_path = "/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ/nuplan_quantized_200_200_16"

    dataset = Nuplan_Occbev_bev_Dataset(imageset,gts_path,bev_path,bev_ch_use,meta_num=4,Tframe=Tframe,training=True)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    
    print(f"dist.get_world_size(): {dist.get_world_size()}")
    loader = DataLoader(
        dataset,
        batch_size=int(args.dit_batch_size // dist.get_world_size()),
        # batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")


   # Initial state
    if args.ckpt:
        train_steps = int(args.ckpt.split('/')[-1].split('.')[0])
        logger.info(f"len(dataset) {len(dataset)}, and train_steps is {train_steps}, and arg.dit_batch_size{args.dit_batch_size}")
        start_epoch = int(train_steps / (len(dataset) / args.dit_batch_size))
        logger.info(f"Initial state: step={train_steps}, epoch={start_epoch}")
    else:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights    


    # Prepare models for training:
    # update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    #train_steps = 0
    #log_steps = 0
    #running_loss = 0
    #start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for i_iter, (occ_ori, occ_gt, y, occ_meta, pose_meta) in enumerate(tqdm(loader, desc=f"Epoch {epoch}", total=len(loader))):
            # if i_iter>100:
            #     break
            occ_ori = occ_ori.to(device)
            occ_ori = fill_large_zero_height_regions_gpu(occ_ori, target_layer=8)


            y = y.to(device)
            # ref_idx = random.randint(0, Tframe-1)

            if use_vae:
                with torch.no_grad():
                    # VAE编码5帧occupancy到潜在空间
                    x=vae.encode(occ_ori) *scale_factor # x: B C T H W  *30 make std = 1
                # 使用参考帧（ref_idx=0，即第0帧）作为forecasting的起点
                z_ref = x[:,:,ref_idx].unsqueeze(2)  # 取出参考帧
                z_ref = z_ref.repeat(1, 1, Tframe, 1, 1)  # 复制到所有时间步作为先验
                z_ref = z_ref.to(device)
            else:
                x = x.to(device) * scale_factor #50 # B T C H W
                z_ref = x[:,ref_idx].unsqueeze(1)
                z_ref = z_ref.repeat(1, Tframe, 1, 1, 1)
                z_ref = z_ref.permute(0,2,1,3,4).to(device)
                x = x.permute(0,2,1,3,4) 
 

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            # Occupancy无条件生成：不使用外部条件（BEV、pose），但保留时序结构
            if use_occ_meta:
                if meta_num_mode == 1:
                        new_meta = torch.cat((occ_meta.reshape(-1,4*Tframe),pose_meta.reshape(-1,12*(Tframe-1))),dim=1)
                elif meta_num_mode == 2:
                    new_meta = torch.cat((occ_meta[:,ref_idx],pose_meta.reshape(-1,12*(Tframe-1))),dim=1)
                elif meta_num_mode == 3:
                    new_meta = occ_meta[:,ref_idx]

                new_meta = new_meta.to(device)
                # print(new_meta.shape)
                model_kwargs = dict(y=y, meta=new_meta)
            else:
                model_kwargs = dict(y=y)

            
            # Occupancy forecasting：自定义损失计算，专注于未来帧预测
            if use_noise_prior:
                noise_prior = torch.randn_like(x) + lambda_np * z_ref
            else:
                noise_prior = torch.randn_like(x)
            
            # 手动计算forecasting损失，而不使用标准的training_losses
            x_t = diffusion.q_sample(x, t, noise=noise_prior)
            model_output = model(x_t, t, **model_kwargs)
            
            # 处理learn_sigma=True的情况：模型输出包含均值和方差
            if model_output.shape[1] == x.shape[1] * 2:  # learn_sigma=True
                # 分离均值和方差预测
                B, C_double, T, H, W = model_output.shape
                C = C_double // 2
                model_mean, model_var_values = torch.split(model_output, C, dim=1)
                model_output = model_mean  # 只使用均值部分进行损失计算
            
            # 确定预测目标（通常是noise）
            if diffusion.model_mean_type.name == "EPSILON":
                target = noise_prior
            elif diffusion.model_mean_type.name == "START_X":
                target = x
            else:
                target = noise_prior  # 默认预测噪声
            
            # Forecasting关键：只对未来帧计算损失
            B, C, T, H, W = x.shape
            
            # 创建未来帧mask（排除参考帧）
            future_mask = torch.ones_like(x, dtype=torch.bool)
            future_mask[:, :, ref_idx, :, :] = False  # 排除参考帧
            
            # 只对未来帧计算MSE损失
            mse_loss = (target - model_output) ** 2
            forecasting_loss = mse_loss[future_mask].mean()  # 只对未来帧求平均
            
            # 创建兼容的loss_dict格式
            loss_dict = {"loss": forecasting_loss.unsqueeze(0).expand(B)}
            
            # 可选：添加参考帧的重构损失（权重较小）
            if args.lambda_noise_prior > 0:
                ref_mask = torch.zeros_like(x, dtype=torch.bool)
                ref_mask[:, :, ref_idx, :, :] = True
                ref_recon_loss = mse_loss[ref_mask].mean() if ref_mask.any() else 0
                # 总损失 = 主要的forecasting损失 + 少量参考帧重构损失
                total_loss = forecasting_loss + 0.1 * ref_recon_loss
                loss_dict = {"loss": total_loss.unsqueeze(0).expand(B)}

            # 无条件生成通常不使用confidence weighting
            if args.confidence and use_occ_meta:
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
                    with open(f"{experiment_dir}/log.txt", 'a') as file:
                        current_lr = opt.param_groups[0]['lr']
                        print(f"(step={train_steps:07d}) lr:{current_lr:.6f} Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}", file=file)
                    
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

                # 添加tensorboard记录
                if rank == 0:
                    current_lr = opt.param_groups[0]['lr']
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                    writer.add_scalar('Learning_rate', current_lr, train_steps)
                    # 如果使用了confidence weighted loss
                    if args.confidence:
                        writer.add_scalar('Loss/weighted_train', loss.item(), train_steps)
                    # 添加noise prior相关的loss
                    if use_noise_prior:
                        writer.add_scalar('Loss/noise_prior', lambda_np, train_steps)
                    # 新增的指标
                    # 1. 记录梯度范数，用于监控梯度稳定性
                    total_grad_norm = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            total_grad_norm += param.grad.data.norm(2).item() ** 2
                    total_grad_norm = total_grad_norm ** 0.5
                    writer.add_scalar('Gradients/norm', total_grad_norm, train_steps)
                    
                    # 2. 记录EMA和原始模型的参数差异
                    ema_param_diff = 0
                    for ema_param, model_param in zip(ema.parameters(), model.module.parameters()):
                        ema_param_diff += torch.norm(ema_param.data - model_param.data).item()
                    writer.add_scalar('Model/ema_diff', ema_param_diff, train_steps)

                    # 3. 记录噪声水平
                    writer.add_scalar('Diffusion/timestep_mean', t.float().mean(), train_steps)
                    
                    # 4. 如果使用noise_prior，记录noise和prior的比例
                    if use_noise_prior:
                        noise_std = torch.std(noise_prior).item()
                        z_ref_std = torch.std(z_ref).item()
                        writer.add_scalar('Diffusion/noise_std', noise_std, train_steps)
                        writer.add_scalar('Diffusion/z_ref_std', z_ref_std, train_steps)

                    # 5. 记录VAE编码的统计信息
                    with torch.no_grad():
                        vae_latent = vae.encode(occ_ori)
                        writer.add_scalar('VAE/latent_mean', vae_latent.mean().item(), train_steps)
                        writer.add_scalar('VAE/latent_std', vae_latent.std().item(), train_steps)

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

    if rank == 0:
        writer.close()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="xx")
    parser.add_argument("--results-dir", type=str, default="out/nuplan_occ_dit")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 128], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--dit-batch-size", type=int, default=108)  #72
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--vae_ckpt", type=str, default=None)
    parser.add_argument("--vae_config", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpt-preDIT", type=str, default=None)
    parser.add_argument("--confidence", type=int,default=0)
    parser.add_argument("--lambda_noise_prior", type=float, default=0.3)
    parser.add_argument("--local-rank", type=int,default=0)
    args = parser.parse_args()
    main(args)
    # python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 train_continuous_mVAE.py --vae_ckpt="out/vae_4_DwT_L_c16r2me/epoch_296.pth" --vae_config="out/vae_4_DwT_L_c16r2me/train_vae_4_DwT_L_me.py"
    # python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 train_continuous_mVAE.py --vae_ckpt="out/vae_4_DwoT_L_c16r2me/epoch_196.pth" --vae_config="out/vae_4_DwoT_L_c16r2me/train_vae_4_DwoT_L_me.py"