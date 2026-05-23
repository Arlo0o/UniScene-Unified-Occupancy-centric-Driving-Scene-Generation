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

from diffusion.models import DiT_models,DiT_2Frame
from diffusion import create_diffusion
from tqdm import tqdm
# from diffusers.models import AutoencoderKL

import datetime
import shutil

from dataset.dataload_util import Nuplan_Occ_bev_HR_mini
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



def topdown_projection_and_save(occ: torch.Tensor, out_path="topdown.png"):
    """
    将3D occupancy转换为top-down投影并保存
    occ: [H, W, Z] 的整型类别张量（例如 0=空，其它为语义类）
    """
    import torch
    import numpy as np
    from PIL import Image
    
    H, W, Z = occ.shape

    # 从顶层开始找第一个非零类别
    flip_occ  = torch.flip(occ, dims=[2])          # [H,W,Z]
    flip_mask = (flip_occ != 0)                    # True=占据
    has_any   = flip_mask.any(dim=2)               # [H,W]

    # 每个像素的"第一个 True"的索引（无则为0，占位）
    first_idx = flip_mask.float().argmax(dim=2)    # [H,W], 0..Z-1

    # 取对应类别
    top_cls = torch.gather(flip_occ, 2, first_idx.unsqueeze(2)).squeeze(2)  # [H,W]
    top_cls = torch.where(has_any, top_cls, torch.zeros_like(top_cls))       # 无占据→0

    # 自定义调色表（按需补充/修改）
    classname_to_color = {
        0: (255, 255, 255),  # 空
        1: (0, 175, 0),      # other-ground
        2: (255, 158, 0),    # vehicle
        3: (220, 20, 60),    # bicycle
        4: (0, 0, 230),      # pedestrian
        5: (47, 79, 79),     # traffic-cone
        6: (112, 128, 144),  # barrier
        7: (255, 200, 0),    # construction-zones
        8: (222, 184, 13),   # generic-object
        12: (0, 207, 191),   # road
        14: (150, 240, 80),  # road-line
    }

    # 映射到 RGB
    top_np = top_cls.cpu().numpy().astype(np.int64)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for k, rgb in classname_to_color.items():
        img[top_np == k] = rgb

    Image.fromarray(img).save(out_path)



def create_generation_visualization(latent_sample, decoded_sample, save_path, bev_sample=None):
    """
    创建生成结果的可视化 - 使用top-down投影，包含BEV条件
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    print(f"Debug: latent_sample.shape={latent_sample.shape}, decoded_sample.shape={decoded_sample.shape}")
    if bev_sample is not None:
        print(f"Debug: bev_sample.shape={bev_sample.shape}")
    
    # 如果有BEV，增加一行显示BEV
    if bev_sample is not None:
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))  # 3行：BEV, Latent, Occupancy
    else:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))   # 2行：Latent, Occupancy
    
    Tframe = min(5, latent_sample.shape[1])  # 最多显示5个时间步
    
    try:
        for t in range(Tframe):
            # BEV可视化（如果有的话）
            if bev_sample is not None:
                # BEV数据形状是 [T, C, H, W] = [5, 1, 400, 400] (已经取了第一个batch)
                if len(bev_sample.shape) == 4:  # [T, C, H, W]
                    bev_vis = bev_sample[t, 0]  # 取第t个时间步，第一个通道
                elif len(bev_sample.shape) == 3:  # [T, H, W]
                    bev_vis = bev_sample[t]  # 取第t个时间步
                else:
                    bev_vis = bev_sample[t] if len(bev_sample.shape) > 2 else bev_sample
                
                # BEV直接可视化，不需要top-down投影
                axes[0, t].imshow(bev_vis, cmap='viridis')  # 使用viridis颜色映射
                axes[0, t].set_title(f'BEV Condition T={t}')
                axes[0, t].axis('off')
            
            # Latent space可视化 - 根据实际维度处理
            latent_row = 0 if bev_sample is None else 1
            if len(latent_sample.shape) == 4:  # [C, T, H, W]
                latent_vis = latent_sample[0, t]  # 取第一个通道
            elif len(latent_sample.shape) == 3:  # [T, H, W]
                latent_vis = latent_sample[t]
            else:
                latent_vis = latent_sample[t] if len(latent_sample.shape) > 2 else latent_sample
            
            axes[latent_row, t].imshow(latent_vis, cmap='coolwarm')
            axes[latent_row, t].set_title(f'Generated Latent T={t}')
            axes[latent_row, t].axis('off')
            
            # Decoded occupancy可视化 - 使用top-down投影
            occ_row = 1 if bev_sample is None else 2
            if len(decoded_sample.shape) == 5:  # [T, C, H, W, D] - 3D occupancy
                decoded_3d = decoded_sample[t, 0]  # 取第一个通道 [H, W, D]
            elif len(decoded_sample.shape) == 4:  # [T, H, W, D]
                decoded_3d = decoded_sample[t]  # [H, W, D]
            else:  # [H, W, D]
                decoded_3d = decoded_sample
            
            # 转换为torch tensor并做top-down投影
            decoded_tensor = torch.from_numpy(decoded_3d).long()
            topdown_path = save_path.replace('.png', f'_topdown_t{t}.png')
            topdown_projection_and_save(decoded_tensor, topdown_path)
            
            # 读取top-down图像并显示
            from PIL import Image
            decoded_img = np.array(Image.open(topdown_path))
            axes[occ_row, t].imshow(decoded_img)
            axes[occ_row, t].set_title(f'Generated Occupancy T={t} (Top-down)')
            axes[occ_row, t].axis('off')
            
    except Exception as e:
        print(f"生成可视化过程中出错: {e}")
        import traceback
        traceback.print_exc()
        # 在最后一个子图中显示错误信息
        error_row = 1 if bev_sample is None else 2
        axes[error_row, 4].text(0.1, 0.5, f"Error: {str(e)}", transform=axes[error_row, 4].transAxes, 
                               fontsize=10, verticalalignment='center')
        axes[error_row, 4].set_title('Error')
        axes[error_row, 4].axis('off')
    
    plt.suptitle('DiT Generated Samples Visualization (BEV + Latent + Occupancy)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()




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
    # scale_factor = 15  # 降低scale_factor，避免训练不稳定
    use_noise_prior = True
    lambda_np = args.lambda_noise_prior
    # bev_ch_use=[1,2,8,9,10,11,12,13,14,15,17]
    bev_ch_use=[1,8,9,10,11,12,13]

    DiT_cfg={"depth":12, "in_channels":in_ch, "hidden_size":512, "input_size":100, "patch_size":4, "use_label":False, "use_bev_concat":use_bev_concat, "bev_in_ch":1, "bev_out_ch":1, "use_meta":use_occ_meta, "bev_dropout_prob":0.1, "meta_num":meta_num, "direct_concat":True, "Tframe":Tframe}
    model = DiT_2Frame(**DiT_cfg).to(device)
    
    if rank == 0:
        current_file = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
        shutil.copy(current_file, experiment_dir)  # 复制当前脚本到 experiment_dir


    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.001)  # 降低学习率
    # 使用更稳定的学习率调度
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)  # 每5000步学习率减半

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



    # imageset = "/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_train.pkl"
    # bev_path = '/lpai/dataset/nuplan-bev/0-1-0/nuplan_bev_new/mini/train'
    # gts_path = "/lpai/dataset/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_200_200_16/"

    # imageset = "data/nuplan_mini_val_clip_infos_dit.pkl"
    # bev_path = "/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200"
    # gts_path = "/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ/nuplan_quantized_200_200_16"

    # full 400
    imageset = "data/nuplan_mini_val_clip_infos_dit.pkl"
    bev_path = "/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_400"
    gts_path = "/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ/nuplan_quantized_400_400_32"
    dataset = Nuplan_Occ_bev_HR_mini(imageset,gts_path,bev_path,bev_ch_use,meta_num=4,Tframe=Tframe,training=True, debug=args.debug)

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
        num_workers=0,
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


    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for occ_in, bev, occ_meta, pose_meta, scene_metas in tqdm(loader):
            occ_ori = occ_in.to(device)  # 使用输入的occupancy作为训练目标

            y = bev.to(device)
            # ref_idx = random.randint(0, Tframe-1)


            x=vae.encode(occ_ori)  # x: B C T H W  *20 make std = 1

            # 测试 latent space 加噪过程
            if rank == 0:
                os.makedirs(f"{experiment_dir}/latent_noise_test", exist_ok=True)
                # 选择不同的噪声步数进行测试
                noise_steps = [0, 100, 200, 500, 800, 999]  # 不同的噪声级别
                B, T, H, W, D = occ_ori.shape
                max_b = min(B, 2)
                t = 0  # 仅可视化第一个时刻
                
                for b in range(max_b):
                    # 原始图像
                    ori_3d = occ_ori[b, t].detach().cpu().long()
                    ori_path = f"{experiment_dir}/latent_noise_test/ori_b{b}_t{t}.png"
                    topdown_projection_and_save(ori_3d, ori_path)
                    
                    # 对每个噪声级别进行测试
                    for noise_step in noise_steps:
                        # 按照 diffusion 过程添加噪声
                        t_tensor = torch.tensor([noise_step], device=device)
                        noise = torch.randn_like(x[b:b+1])  # 生成噪声
                        
                        # 使用 diffusion 的噪声调度
                        alpha_t = torch.tensor(diffusion.alphas_cumprod[noise_step], device=device)
                        sqrt_alpha_t = torch.sqrt(alpha_t)
                        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                        
                        # 添加噪声到 latent space
                        noisy_x = sqrt_alpha_t * x[b:b+1] + sqrt_one_minus_alpha_t * noise
                        
                        # 解码加噪后的 latent
                        with torch.no_grad():
                            noisy_occ_recon = vae.generate(noisy_x, occ_ori[b:b+1].shape)
                            noisy_pred = noisy_occ_recon['logits'].argmax(dim=-1).detach().cpu()
                        
                        # 保存加噪后的结果
                        noisy_3d = noisy_pred[0, t].long()  # [H, W, D]
                        noisy_path = f"{experiment_dir}/latent_noise_test/noisy_step{noise_step}_b{b}_t{t}.png"
                        topdown_projection_and_save(noisy_3d, noisy_path)
                        
                        # 创建对比图：原始 vs 加噪
                        from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFont as _PILFont
                        ori_img = _PILImage.open(ori_path)
                        noisy_img = _PILImage.open(noisy_path)
                        w1, h1 = ori_img.size
                        w2, h2 = noisy_img.size
                        canvas = _PILImage.new('RGB', (w1 + w2, max(h1, h2)), (255, 255, 255))
                        canvas.paste(ori_img, (0, 0))
                        canvas.paste(noisy_img, (w1, 0))
                        
                        # 绘制图注
                        drawer = _PILDraw.Draw(canvas)
                        try:
                            font = _PILFont.load_default()
                        except Exception:
                            font = None
                        # 左侧标签
                        drawer.rectangle([(5, 5), (5 + 60, 5 + 22)], fill=(0, 0, 0, 127))
                        drawer.text((10, 8), "GT", fill=(255, 255, 255), font=font)
                        # 右侧标签
                        drawer.rectangle([(w1 + 5, 5), (w1 + 5 + 100, 5 + 22)], fill=(0, 0, 0, 127))
                        drawer.text((w1 + 10, 8), f"Noise T={noise_step}", fill=(255, 255, 255), font=font)
                        
                        cmp_path = f"{experiment_dir}/latent_noise_test/compare_step{noise_step}_b{b}_t{t}.png"
                        canvas.save(cmp_path)
                        print(f"已保存噪声步数 {noise_step} 的对比图: {cmp_path}")

            # 正常的 VAE 重建（无噪声）
            occ_recon = vae.generate(x, occ_in.shape)
            pred = occ_recon['logits'].argmax(dim=-1).detach().cuda()
            print(pred.shape)

            # 可视化保存 VAE 重建结果（Top-down 视角）：只可视化第一个时刻，并与原始对比拼接
            if rank == 0:
                os.makedirs(f"{experiment_dir}/vae_recon_vis", exist_ok=True)
                # pred: [B, T, H, W, D]，occ_ori: [B, T, H, W, D]
                B, T, H, W, D = pred.shape
                max_b = min(B, 2)
                t = 0  # 仅可视化第一个时刻
                for b in range(max_b):
                    # 生成原始与重建的 top-down 图
                    ori_3d = occ_ori[b, t].detach().cpu().long()   # [H, W, D]
                    recon_3d = pred[b, t].detach().cpu().long()    # [H, W, D]
                    ori_path = f"{experiment_dir}/vae_recon_vis/ori_b{b}_t{t}.png"
                    recon_path = f"{experiment_dir}/vae_recon_vis/recon_b{b}_t{t}.png"
                    topdown_projection_and_save(ori_3d, ori_path)
                    topdown_projection_and_save(recon_3d, recon_path)

                    # 横向拼接对比图
                    from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFont as _PILFont
                    ori_img = _PILImage.open(ori_path)
                    recon_img = _PILImage.open(recon_path)
                    w1, h1 = ori_img.size
                    w2, h2 = recon_img.size
                    canvas = _PILImage.new('RGB', (w1 + w2, max(h1, h2)), (255, 255, 255))
                    canvas.paste(ori_img, (0, 0))
                    canvas.paste(recon_img, (w1, 0))
                    # 绘制图注：左侧为 GT，右侧为 Recon
                    drawer = _PILDraw.Draw(canvas)
                    try:
                        font = _PILFont.load_default()
                    except Exception:
                        font = None
                    # 左侧标签
                    drawer.rectangle([(5, 5), (5 + 60, 5 + 22)], fill=(0, 0, 0, 127))
                    drawer.text((10, 8), "GT", fill=(255, 255, 255), font=font)
                    # 右侧标签
                    drawer.rectangle([(w1 + 5, 5), (w1 + 5 + 80, 5 + 22)], fill=(0, 0, 0, 127))
                    drawer.text((w1 + 10, 8), "Recon", fill=(255, 255, 255), font=font)
                    cmp_path = f"{experiment_dir}/vae_recon_vis/compare_b{b}_t{t}.png"
                    canvas.save(cmp_path)



    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="xx")
    parser.add_argument("--results-dir", type=str, default="out_1/nuplan_occ_dit")
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
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()
    main(args)
    # python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 train_continuous_mVAE.py --vae_ckpt="out/vae_4_DwT_L_c16r2me/epoch_296.pth" --vae_config="out/vae_4_DwT_L_c16r2me/train_vae_4_DwT_L_me.py"
    # python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 train_continuous_mVAE.py --vae_ckpt="out/vae_4_DwoT_L_c16r2me/epoch_196.pth" --vae_config="out/vae_4_DwoT_L_c16r2me/train_vae_4_DwoT_L_me.py"