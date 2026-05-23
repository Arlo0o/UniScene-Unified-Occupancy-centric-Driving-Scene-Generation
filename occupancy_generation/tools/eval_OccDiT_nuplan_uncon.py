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

from dataset.dataload_util import CustomDataset_2frame_time,CustomDataset_2frame_continuous,CustomDataset_Tframe_continuous,Nuplan_Occ_bev_Dataset
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



#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Evaluates a pre-trained DiT model.
    """
    assert torch.cuda.is_available(), "Evaluation currently requires at least one GPU."

    # Variables for results:
    torch.set_grad_enabled(False)  # 确保不计算梯度
    
    # Setup DDP:
    dist.init_process_group("nccl")
    rank = args.local_rank
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    # Setup output directories:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder
        if args.ckpt:
            model_name = os.path.basename(os.path.dirname(args.ckpt))
            experiment_dir = f"{args.results_dir}/eval_{model_name}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        else:
            experiment_dir = f"{args.results_dir}/eval_unknown_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        
        vis_dir = f"{experiment_dir}/visualizations"  # 存储可视化结果
        os.makedirs(vis_dir, exist_ok=True)
        
        logger = create_logger(experiment_dir, rank)
        logger.info(f"Evaluation directory created at {experiment_dir}")
    else:
        logger = create_logger(None, rank)
        vis_dir = None

    # Create model:
    use_bev_concat = False
    use_occ_meta = True
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

    ref_idx = 0
    scale_factor = 70
    use_noise_prior = True
    lambda_np = args.lambda_noise_prior
    bev_ch_use=[1,2,8,9,10,11,12,13,14,15,17]

    DiT_cfg={"depth":12, "in_channels":in_ch, "hidden_size":512,"use_label":False, "use_bev_concat":use_bev_concat,"bev_in_ch":1,"bev_out_ch":1,"use_meta":use_occ_meta, "bev_dropout_prob":0.1,"meta_num":meta_num,"direct_concat":True,"Tframe":Tframe}
    # model = DiT_Occsora(**DiT_cfg).to(device)
    model = DiT_2Frame(**DiT_cfg).to(device)
    
    # Load checkpoint
    if args.ckpt:
        logger.info(f"Loading model from checkpoint: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        if 'ema' in checkpoint:
            logger.info("Using EMA weights for evaluation")
            model.load_state_dict(checkpoint['ema'], strict=False)
        else:
            logger.info("Using model weights for evaluation")
            model.load_state_dict(checkpoint['model'], strict=False)
        del checkpoint
    else:
        logger.warning("No checkpoint provided. Using randomly initialized model.")


    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    
    if use_vae:
        import model_vae
        cfg = Config.fromfile(args.vae_config)
        vae = MODELS.build(cfg.model)
        vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu')
        vae.load_state_dict(vae_ckpt['state_dict'], strict=True)
        vae = vae.to(device)
        vae.eval()
    
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    model.eval()  # 确保模型在评估模式
    
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup dataset
    imageset = "/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_train.pkl"
    bev_path = '/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200'
    gts_path = "/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ/nuplan_quantized_200_200_16"
    npy_save_path = args.save_path
    os.makedirs(npy_save_path,exist_ok=True)

    dataset = Nuplan_Occ_bev_Dataset(imageset, gts_path, bev_path, bev_ch_use, meta_num=4, Tframe=Tframe, training=False)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,  # 评估时不需要打乱数据
        seed=args.global_seed
    )
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        # batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False  # 评估时不丢弃最后一个批次
    )
    logger.info(f"Dataset contains {len(dataset):,} samples")

    num_samples_processed = 0
    
    # Evaluation loop
    logger.info("Beginning evaluation...")
    for i_iter, (occ_ori,occ_out,  y, occ_meta, pose_meta) in enumerate(tqdm(loader, desc="Evaluating", total=len(loader))):
        # if args.num_samples and num_samples_processed >= args.num_samples:
        #     break
            
        # 将数据移动到设备
        occ_ori = occ_ori.to(device)
        y = y.to(device)
        
        batch_size = occ_ori.shape[0]
        
        # VAE编码
        if use_vae:
            with torch.no_grad():
                x = vae.encode(occ_ori) * scale_factor
            z_ref = x[:, :, ref_idx].unsqueeze(2)
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
            model_kwargs = dict(y=y, meta=new_meta)
        else:
            model_kwargs = dict(y=y)

        # 生成初始噪声（带先验）
        noise = torch.randn_like(x)
        if use_noise_prior:
            noise = noise + lambda_np * z_ref
        print(f"=> noise shape: {noise.shape}")
        # 使用扩散模型进行采样
        with torch.no_grad():
            samples = diffusion.p_sample_loop(
                # model.module.forward_with_cfg,
                model.module.forward,
                noise.shape,
                noise=noise,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
            )
        # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        print(f"=> samples shape{samples.shape}")

        samples = samples / scale_factor
        rec_shape=[batch_size,Tframe,200,200,16]
        if use_vae==True:
            result=vae.generate(samples,rec_shape)
        else:
            result=vae.generate_vq(samples,rec_shape)
        logit=result["logits"]
        pred = logit.argmax(dim=-1) #  200, 200, 16

        pred_save = pred.cpu().numpy()
        occ_ori_save = occ_ori.cpu().numpy()
        np.save(f'{npy_save_path}/{i_iter:05d}_pred.npy', pred_save.astype(np.int8))
        np.save(f'{npy_save_path}/{i_iter:05d}_occ_ori.npy', occ_ori_save.astype(np.int8))
        
    
    dist.barrier()  # 等待所有进程
    
    logger.info(f"Evaluation completed. Processed {num_samples_processed} samples.")
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
    parser.add_argument("--save_path", type=str, default=None, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpt-preDIT", type=str, default=None)
    parser.add_argument("--confidence", type=int,default=0)
    parser.add_argument("--lambda_noise_prior", type=float, default=0.03)
    parser.add_argument("--local-rank", type=int,default=0)
    args = parser.parse_args()
    main(args)
    # python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 train_continuous_mVAE.py --vae_ckpt="out/vae_4_DwT_L_c16r2me/epoch_296.pth" --vae_config="out/vae_4_DwT_L_c16r2me/train_vae_4_DwT_L_me.py"
    # python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 train_continuous_mVAE.py --vae_ckpt="out/vae_4_DwoT_L_c16r2me/epoch_196.pth" --vae_config="out/vae_4_DwoT_L_c16r2me/train_vae_4_DwoT_L_me.py"