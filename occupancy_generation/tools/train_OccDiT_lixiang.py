# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
lixiang latent space shape: 334, 160, 16
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

from diffusion.models import DiT_2Frame_lixiang_rectangular
from diffusion import create_diffusion
from tqdm import tqdm
# from diffusers.models import AutoencoderKL

import datetime
import shutil

from dataset.dataload_util import CustomDataset_2frame_time,CustomDataset_2frame_continuous,CustomDataset_Tframe_continuous,Nuplan_Occ_bev_Dataset
from torch.optim.lr_scheduler import StepLR
import random

from mmengine import Config
from mmengine.registry import MODELS
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
        
        ct_str=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        # experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        experiment_dir = f"{args.results_dir}/{ct_str}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir,rank)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None,rank)

    # Create model:
    
    use_bev_concat=True
    use_occ_meta=True
    in_ch=4
    Tframe=5
    meta_num_mode = 2
    use_vae = True
    if meta_num_mode == 1:
        meta_num=4*Tframe + 12*(Tframe-1)
    elif meta_num_mode == 2:
        meta_num= 4 + 12*(Tframe-1)
    elif meta_num_mode == 3:
        meta_num = 4

    ref_idx = 0
    scale_factor = 15  # 降低scale_factor，避免训练不稳定
    use_noise_prior = True
    lambda_np = args.lambda_noise_prior
    bev_ch_use=[1,2,8,9,10,11,12,13,14,15,17]

    DiT_cfg={"depth":12, "in_channels":in_ch, "hidden_size":512,"use_label":False, "use_bev_concat":use_bev_concat,"bev_in_ch":1,"bev_out_ch":1,"use_meta":use_occ_meta, "bev_dropout_prob":0.1,"meta_num":meta_num,"direct_concat":True,"Tframe":Tframe}
    # model = DiT_Occsora(**DiT_cfg).to(device)
    model = DiT_2Frame_lixiang_rectangular(**DiT_cfg).to(device)
    
    if rank == 0:
        current_file = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
        shutil.copy(current_file, experiment_dir)  # 复制当前脚本到 experiment_dir


    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.001)  # 降低学习率
    # scheduler = StepLR(opt, step_size=10000, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)  # 更快的学习率衰减

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
    
    if use_vae:
        import model_vae
        cfg = Config.fromfile(args.vae_config)
        vae = MODELS.build(cfg.model)
        vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu')
        vae.load_state_dict(vae_ckpt['state_dict'], strict=True)
        vae = vae.to(device)
        vae.eval()
    
    model = DDP(model.to(device), device_ids=[rank],find_unused_parameters=True)
    
    
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")



    imageset = "/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_train.pkl"
    bev_path = '/lpai/dataset/nuplan-bev/0-1-0/nuplan_bev_new/mini/train'
    gts_path = "/lpai/dataset/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_200_200_16/"

    dataset = Nuplan_Occ_bev_Dataset(imageset,gts_path,bev_path,bev_ch_use,meta_num=4,Tframe=Tframe,training=True)

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
        for occ_in, occ_out,y,occ_meta,pose_meta in tqdm(loader):
            occ_ori = occ_out.to(device)  # 使用输出的occupancy作为训练目标

            y = y.to(device)
            # ref_idx = random.randint(0, Tframe-1)

            if use_vae:
                with torch.no_grad():
                    x=vae.encode(occ_ori) *scale_factor # x: B C T H W  *20 make std = 1
                z_ref = x[:,:,ref_idx].unsqueeze(2)
                z_ref = z_ref.repeat(1, 1, Tframe, 1, 1)
                z_ref = z_ref.to(device)
            else:
                x = x.to(device) * scale_factor #50 # B T C H W
                z_ref = x[:,ref_idx].unsqueeze(1)
                z_ref = z_ref.repeat(1, Tframe, 1, 1, 1)
                z_ref = z_ref.permute(0,2,1,3,4).to(device)
                x = x.permute(0,2,1,3,4) 
 

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

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

            # lambda_z = 0.03
            noise_prior = torch.randn_like(x) + lambda_np * z_ref

            # 关键维度检查 - 确保损失计算正确
            if train_steps % 100 == 0:  # 每100步打印一次
                print(f"=> x.shape: {x.shape}")  # 应该是 [B, C, T, H, W]
                print(f"=> y.shape: {y.shape}")  # 应该是 [B, T, C, H, W] 
                print(f"=> new_meta.shape: {new_meta.shape}")  # 应该是 [B, meta_num]
                print(f"=> t.shape: {t.shape}")  # 应该是 [B]
                print(f"=> noise_prior.shape: {noise_prior.shape}")  # 应该与x相同
                print(f"=> z_ref.shape: {z_ref.shape}")  # 应该与x相同
            
            if use_noise_prior:
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs, noise=noise_prior)
            else:
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

            if args.confidence:
                occ_meta = occ_meta.to(device)
                w_loss = loss_dict["loss"] * occ_meta.squeeze()
                # print(w_loss.shape)
                loss=w_loss.mean()
            else:
                loss = loss_dict["loss"].mean()
            

            opt.zero_grad()
            print(f'==> loss is {loss}')
            loss.backward()
            
            # 添加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()

            scheduler.step()
            # 使用更保守的EMA衰减率
            update_ema(ema, model.module, decay=0.999)

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
    parser.add_argument("--data-path", type=str, default="xx")
    parser.add_argument("--results-dir", type=str, default="out/nuplan_occ_dit")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 128], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--dit-batch-size", type=int, default=40)  #72
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
    parser.add_argument("--lambda_noise_prior", type=float, default=0.15)
    parser.add_argument("--local-rank", type=int,default=0)
    args = parser.parse_args()
    main(args)
    # python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 train_continuous_mVAE.py --vae_ckpt="out/vae_4_DwT_L_c16r2me/epoch_296.pth" --vae_config="out/vae_4_DwT_L_c16r2me/train_vae_4_DwT_L_me.py"
    # python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 train_continuous_mVAE.py --vae_ckpt="out/vae_4_DwoT_L_c16r2me/epoch_196.pth" --vae_config="out/vae_4_DwoT_L_c16r2me/train_vae_4_DwoT_L_me.py"