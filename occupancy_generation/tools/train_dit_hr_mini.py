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


def save_latent_samples(x, z_ref, noise_prior, occ_ori, experiment_dir, train_steps, vae=None, bev_sample=None):
    """
    保存latent space样本用于可视化分析，包含BEV数据
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 创建保存目录
    latent_dir = f"{experiment_dir}/latent_samples"
    os.makedirs(latent_dir, exist_ok=True)
    
    # 取第一个样本进行可视化
    x_sample = x[0].detach().cpu().numpy()  # [C, T, H, W]
    z_ref_sample = z_ref[0].detach().cpu().numpy()  # [C, T, H, W]
    noise_sample = noise_prior[0].detach().cpu().numpy()  # [C, T, H, W]
    occ_sample = occ_ori[0].detach().cpu().numpy()  # [T, C, H, W]
    
    # 保存原始数据
    np.save(f"{latent_dir}/step_{train_steps:07d}_x.npy", x_sample)
    np.save(f"{latent_dir}/step_{train_steps:07d}_z_ref.npy", z_ref_sample)
    np.save(f"{latent_dir}/step_{train_steps:07d}_noise.npy", noise_sample)
    np.save(f"{latent_dir}/step_{train_steps:07d}_occ_ori.npy", occ_sample)
    
    # 如果有BEV数据，也保存
    bev_sample_np = None
    if bev_sample is not None:
        bev_sample_np = bev_sample[0].detach().cpu().numpy()  # [T, C, H, W]
        np.save(f"{latent_dir}/step_{train_steps:07d}_bev.npy", bev_sample_np)
    
    # 如果使用VAE，检查编码-解码一致性
    if vae is not None:
        try:
            with torch.no_grad():
                # 正确的检查：原始occupancy → VAE编码 → VAE解码 → 对比原始occupancy
                # 重新编码原始occupancy
                encoded_occ = vae.encode(occ_ori[0:1])  # 重新编码
                # 解码回occupancy - 使用正确的形状
                decoded_occ = vae.generate(encoded_occ, occ_ori[0:1].shape)  # 使用正确的形状
                # VAE的generate方法返回字典{'logits': logits}，需要取logits并应用argmax
                decoded_sample = decoded_occ['logits'][0].argmax(dim=-1).detach().cpu().numpy()
                np.save(f"{latent_dir}/step_{train_steps:07d}_decoded.npy", decoded_sample)
                
                # 创建可视化对比图：原始 vs 编码-解码，包含BEV
                create_latent_visualization(x_sample, z_ref_sample, noise_sample, 
                                          occ_sample, decoded_sample, 
                                          f"{latent_dir}/step_{train_steps:07d}_vis.png",
                                          bev_sample_np if bev_sample is not None else None)
        except Exception as e:
            print(f"VAE编码-解码检查失败: {e}")
            import traceback
            traceback.print_exc()
            # 不使用VAE时的可视化
            create_latent_visualization(x_sample, z_ref_sample, noise_sample, 
                                      occ_sample, None, 
                                      f"{latent_dir}/step_{train_steps:07d}_vis.png",
                                      bev_sample_np if bev_sample is not None else None)
    else:
        # 不使用VAE时的可视化
        create_latent_visualization(x_sample, z_ref_sample, noise_sample, 
                                  occ_sample, None, 
                                  f"{latent_dir}/step_{train_steps:07d}_vis.png",
                                  bev_sample_np if bev_sample is not None else None)
    
    print(f"已保存latent samples到 {latent_dir}/step_{train_steps:07d}_*")


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


def create_latent_visualization(x, z_ref, noise, occ_ori, decoded=None, save_path=None, bev_sample=None):
    """
    创建latent space可视化 - 重点检查VAE编码-解码一致性
    使用top-down投影可视化occupancy，包含BEV数据
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    print(f"Debug: x.shape={x.shape}, z_ref.shape={z_ref.shape}, noise.shape={noise.shape}, occ_ori.shape={occ_ori.shape}")
    if bev_sample is not None:
        print(f"Debug: bev_sample.shape={bev_sample.shape}")
    
    # 如果有BEV，增加一行显示BEV
    if bev_sample is not None:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # 3行：BEV, Latent, Occupancy
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))   # 2行：Latent, Occupancy
    
    # 选择第一个时间步进行可视化
    t_idx = 0
    
    try:
        # BEV可视化（如果有的话）
        if bev_sample is not None:
            # BEV数据形状是 [T, C, H, W] = [5, 1, 400, 400]
            if len(bev_sample.shape) == 4:  # [T, C, H, W]
                bev_vis = bev_sample[t_idx, 0]  # 取第t个时间步，第一个通道
            elif len(bev_sample.shape) == 3:  # [T, H, W]
                bev_vis = bev_sample[t_idx]  # 取第t个时间步
            else:
                bev_vis = bev_sample[t_idx] if len(bev_sample.shape) > 2 else bev_sample
            
            # BEV直接可视化，不需要top-down投影
            axes[0, 0].imshow(bev_vis, cmap='viridis')  # 使用viridis颜色映射
            axes[0, 0].set_title('BEV Condition')
            axes[0, 0].axis('off')
        
        # 原始occupancy - 转换为top-down投影
        if len(occ_ori.shape) == 5:  # [T, C, H, W, D] - 3D occupancy
            occ_3d = occ_ori[t_idx, 0]  # 取第一个通道 [H, W, D]
        elif len(occ_ori.shape) == 4:  # [T, H, W, D]
            occ_3d = occ_ori[t_idx]  # [H, W, D]
        else:  # [H, W, D]
            occ_3d = occ_ori
        
        # 转换为torch tensor并做top-down投影
        occ_tensor = torch.from_numpy(occ_3d).long()
        topdown_path = save_path.replace('.png', '_topdown_ori.png') if save_path else 'topdown_ori.png'
        topdown_projection_and_save(occ_tensor, topdown_path)
        
        # 读取top-down图像并显示
        from PIL import Image
        topdown_img = np.array(Image.open(topdown_path))
        occ_row = 0 if bev_sample is None else 1
        axes[occ_row, 0].imshow(topdown_img)
        axes[occ_row, 0].set_title('Original Occupancy (Top-down)')
        axes[occ_row, 0].axis('off')
        
        # Latent space (x) - 根据实际维度处理
        if len(x.shape) == 4:  # [C, T, H, W]
            x_vis = x[0, t_idx]  # 取第一个通道
        elif len(x.shape) == 3:  # [T, H, W]
            x_vis = x[t_idx]
        else:
            x_vis = x[t_idx] if len(x.shape) > 2 else x
        
        latent_row = 0 if bev_sample is None else 1
        axes[latent_row, 1].imshow(x_vis, cmap='coolwarm')
        axes[latent_row, 1].set_title('VAE Encoded Latent')
        axes[latent_row, 1].axis('off')
        
        # Reference latent (z_ref) - 根据实际维度处理
        if len(z_ref.shape) == 4:  # [C, T, H, W]
            z_ref_vis = z_ref[0, t_idx]
        elif len(z_ref.shape) == 3:  # [T, H, W]
            z_ref_vis = z_ref[t_idx]
        else:
            z_ref_vis = z_ref[t_idx] if len(z_ref.shape) > 2 else z_ref
        
        axes[latent_row, 2].imshow(z_ref_vis, cmap='coolwarm')
        axes[latent_row, 2].set_title('Reference Latent (z_ref)')
        axes[latent_row, 2].axis('off')
        
        # Noise prior - 根据实际维度处理
        if len(noise.shape) == 4:  # [C, T, H, W]
            noise_vis = noise[0, t_idx]
        elif len(noise.shape) == 3:  # [T, H, W]
            noise_vis = noise[t_idx]
        else:
            noise_vis = noise[t_idx] if len(noise.shape) > 2 else noise
        
        noise_row = 1 if bev_sample is None else 2
        axes[noise_row, 0].imshow(noise_vis, cmap='coolwarm')
        axes[noise_row, 0].set_title('Noise Prior')
        axes[noise_row, 0].axis('off')
        
        # 关键检查：VAE解码结果 - 也使用top-down投影
        if decoded is not None:
            if len(decoded.shape) == 5:  # [T, C, H, W, D] - 3D occupancy
                decoded_3d = decoded[t_idx, 0]  # 取第一个通道 [H, W, D]
            elif len(decoded.shape) == 4:  # [T, H, W, D]
                decoded_3d = decoded[t_idx]  # [H, W, D]
            else:  # [H, W, D]
                decoded_3d = decoded
            
            # 转换为torch tensor并做top-down投影
            decoded_tensor = torch.from_numpy(decoded_3d).long()
            topdown_decoded_path = save_path.replace('.png', '_topdown_decoded.png') if save_path else 'topdown_decoded.png'
            topdown_projection_and_save(decoded_tensor, topdown_decoded_path)
            
            # 读取top-down图像并显示
            decoded_img = np.array(Image.open(topdown_decoded_path))
            decoded_row = 1 if bev_sample is None else 2
            axes[decoded_row, 1].imshow(decoded_img)
            axes[decoded_row, 1].set_title('VAE Decoded Occupancy (Top-down)')
            axes[decoded_row, 1].axis('off')
            
            # 计算编码-解码一致性
            if len(occ_ori.shape) == len(decoded.shape):
                mse = np.mean((occ_ori - decoded) ** 2)
                stats_row = 1 if bev_sample is None else 2
                axes[stats_row, 2].text(0.1, 0.7, f"VAE Reconstruction MSE: {mse:.6f}", 
                                       transform=axes[stats_row, 2].transAxes, fontsize=10)
            else:
                stats_row = 1 if bev_sample is None else 2
                axes[stats_row, 2].text(0.1, 0.7, "Shape mismatch for MSE", 
                                       transform=axes[stats_row, 2].transAxes, fontsize=10)
        else:
            decoded_row = 1 if bev_sample is None else 2
            axes[decoded_row, 1].text(0.5, 0.5, 'No VAE Decode', transform=axes[decoded_row, 1].transAxes, 
                                     ha='center', va='center', fontsize=12)
            axes[decoded_row, 1].set_title('No VAE Decode')
            axes[decoded_row, 1].axis('off')
        
        # 统计信息
        stats_text = f"""
        Latent stats: mean={x.mean():.3f}, std={x.std():.3f}
        z_ref stats: mean={z_ref.mean():.3f}, std={z_ref.std():.3f}
        noise stats: mean={noise.mean():.3f}, std={noise.std():.3f}
        """
        stats_row = 1 if bev_sample is None else 2
        axes[stats_row, 2].text(0.1, 0.3, stats_text, transform=axes[stats_row, 2].transAxes, 
                                fontsize=10, verticalalignment='center')
        axes[stats_row, 2].set_title('Statistics')
        axes[stats_row, 2].axis('off')
        
    except Exception as e:
        print(f"可视化过程中出错: {e}")
        # 显示错误信息
        error_row = 1 if bev_sample is None else 2
        axes[error_row, 2].text(0.1, 0.5, f"Error: {str(e)}", transform=axes[error_row, 2].transAxes, 
                                fontsize=10, verticalalignment='center')
        axes[error_row, 2].set_title('Error')
        axes[error_row, 2].axis('off')
    
    plt.suptitle('VAE Encoding-Decoding Consistency Check', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_training_samples(ema_model, diffusion, vae, experiment_dir, train_steps, device, use_vae, scale_factor=70, dataloader=None, real_bev=None):
    """
    在训练过程中生成样本用于检查训练效果
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 创建生成样本目录
    samples_dir = f"{experiment_dir}/generated_samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    # 设置模型为评估模式
    ema_model.eval()
    
    try:
        with torch.no_grad():
            # 生成随机噪声 - 修复batch_size问题
            batch_size = 2  # 改为2，因为forward_with_cfg需要偶数batch_size
            Tframe = 5
            in_ch = 4
            
            if use_vae:
                # 使用VAE时的latent space尺寸
                # 根据训练时的实际形状：x.shape = [B, C, T, H, W] = [1, 4, 5, 100, 100]
                z_shape = (batch_size, in_ch, Tframe, 100, 100)  # [B, C, T, H, W]
            else:
                # 直接使用occupancy尺寸
                z_shape = (batch_size, in_ch, Tframe, 200, 200)
            
            print(f"Debug: batch_size={batch_size}, z_shape={z_shape}")
            z = torch.randn(z_shape, device=device)
            print(f"Debug: z.shape={z.shape}")
            
            # 优先使用训练循环中的真实BEV数据
            if real_bev is not None:
                # 使用训练循环中的真实BEV数据，需要重复到batch_size
                if real_bev.shape[0] >= batch_size:
                    y = real_bev[:batch_size].to(device)  # 取前batch_size个样本
                else:
                    # 如果真实BEV的batch_size小于需要的batch_size，重复第一个样本
                    y = real_bev[0:1].repeat(batch_size, 1, 1, 1, 1).to(device)
                print(f"使用训练循环中的真实BEV数据: {y.shape}")
            elif dataloader is not None:
                try:
                    # 从数据加载器中获取一个真实的BEV样本
                    for batch in dataloader:
                        # 获取BEV数据
                        if 'bev' in batch:
                            y_real = batch['bev'].to(device)  # 真实的BEV数据
                            # 确保维度匹配
                            if y_real.shape[0] >= batch_size:
                                y = y_real[:batch_size]  # 取前batch_size个样本
                                print(f"使用数据加载器中的BEV数据: {y.shape}")
                                break
                        elif 'y' in batch:
                            y_real = batch['y'].to(device)  # 真实的BEV数据
                            if y_real.shape[0] >= batch_size:
                                y = y_real[:batch_size]  # 取前batch_size个样本
                                print(f"使用数据加载器中的BEV数据: {y.shape}")
                                break
                    else:
                        # 如果没有找到BEV数据，使用构造的布局
                        raise ValueError("未找到BEV数据")
                        
                except Exception as e:
                    print(f"从数据加载器获取BEV失败: {e}")
                    # 回退到构造的布局
                    y = torch.zeros(batch_size, Tframe, 1, 400, 400, device=device)
                    # 创建更复杂的道路布局
                    y[:, :, :, 180:220, 50:350] = 12  # 主道路
                    y[:, :, :, 50:350, 180:220] = 12  # 辅道路
                    y[:, :, :, 170:230, 170:230] = 12  # 交叉路口
                    y[:, :, :, 175:185, 50:350] = 14  # 道路线
                    y[:, :, :, 50:350, 175:185] = 14  # 道路线
                    print("使用构造的BEV布局作为条件")
            else:
                # 如果没有提供数据加载器，使用构造的布局
                y = torch.zeros(batch_size, Tframe, 1, 400, 400, device=device)
                # 创建更复杂的道路布局
                y[:, :, :, 180:220, 50:350] = 12  # 主道路
                y[:, :, :, 50:350, 180:220] = 12  # 辅道路
                y[:, :, :, 170:230, 170:230] = 12  # 交叉路口
                y[:, :, :, 175:185, 50:350] = 14  # 道路线
                y[:, :, :, 50:350, 175:185] = 14  # 道路线
                print("使用构造的BEV布局作为条件")
            
            meta = torch.randn(batch_size, 52, device=device)  # meta条件保持随机
            
            print(f"Debug: y.shape={y.shape}, meta.shape={meta.shape}")
            
            model_kwargs = dict(y=y, meta=meta)
            
            # 使用DDIM采样
            print(f"Debug: 调用ddim_sample_loop前 z.shape={z.shape}")
            samples = diffusion.ddim_sample_loop(
                ema_model.forward_with_cfg, z.shape, z, 
                clip_denoised=False, model_kwargs=model_kwargs, 
                progress=False, device=device
            )
            print(f"Debug: ddim_sample_loop后 samples.shape={samples.shape}")
            
            # 保存生成的latent - 只保存第一个样本
            samples_np = samples[0].detach().cpu().numpy()  # 只取第一个样本
            np.save(f"{samples_dir}/step_{train_steps:07d}_generated_latent.npy", samples_np)
            
            # 如果使用VAE，解码DiT生成的latent到occupancy space
            if use_vae and vae is not None:
                try:
                    # 使用VAE的generate方法解码DiT生成的latent space
                    # 使用真实的occupancy形状，只处理第一个样本
                    # 注意：VAE期望的input_shape是5个维度 [bs, F, H, W, D]
                    target_shape = (1, Tframe, 400, 400, 32)  # [bs, F, H, W, D]
                    decoded_occ = vae.generate(samples[0:1], target_shape)  # 只处理第一个样本
                    # 从logits中获取类别预测 [bs, F, H, W, D, num_classes] -> [bs, F, H, W, D]
                    pred_classes = decoded_occ['logits'][0].argmax(dim=-1).detach().cpu().numpy()
                    np.save(f"{samples_dir}/step_{train_steps:07d}_generated_occ.npy", pred_classes)
                    
                    # 创建可视化：DiT生成的latent vs 解码的occupancy，包含BEV条件
                    bev_sample = y[0].detach().cpu().numpy()  # 取第一个样本的BEV数据
                    create_generation_visualization(samples_np, pred_classes, 
                                                  f"{samples_dir}/step_{train_steps:07d}_vis.png", 
                                                  bev_sample=bev_sample)
                except Exception as e:
                    print(f"DiT生成样本VAE解码失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 只可视化latent space
                    create_latent_only_visualization(samples_np, 
                                                   f"{samples_dir}/step_{train_steps:07d}_latent_vis.png")
            else:
                # 不使用VAE时，直接可视化latent space
                create_latent_only_visualization(samples_np, 
                                               f"{samples_dir}/step_{train_steps:07d}_latent_vis.png")
            
            print(f"已生成训练样本到 {samples_dir}/step_{train_steps:07d}_*")
            
    except Exception as e:
        print(f"生成训练样本时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复训练模式
        ema_model.train()


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


def create_latent_only_visualization(latent_sample, save_path):
    """
    只可视化latent space
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"Debug: latent_sample.shape={latent_sample.shape}")
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    Tframe = min(5, latent_sample.shape[1])  # 最多显示5个时间步
    
    try:
        for t in range(Tframe):
            if len(latent_sample.shape) == 4:  # [C, T, H, W]
                latent_vis = latent_sample[0, t]  # 取第一个通道
            elif len(latent_sample.shape) == 3:  # [T, H, W]
                latent_vis = latent_sample[t]
            else:
                latent_vis = latent_sample[t] if len(latent_sample.shape) > 2 else latent_sample
            
            axes[t].imshow(latent_vis, cmap='coolwarm')
            axes[t].set_title(f'Generated Latent T={t}')
            axes[t].axis('off')
            
    except Exception as e:
        print(f"Latent可视化过程中出错: {e}")
        # 在最后一个子图中显示错误信息
        axes[4].text(0.1, 0.5, f"Error: {str(e)}", transform=axes[4].transAxes, 
                     fontsize=10, verticalalignment='center')
        axes[4].set_title('Error')
        axes[4].axis('off')
    
    plt.suptitle('Generated Latent Space Visualization', fontsize=16)
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
    # model = DiT_Occsora(**DiT_cfg).to(device)
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


    diffusion = create_diffusion(timestep_respacing="ddim50")  # default: 1000 steps, linear noise schedule
    
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
    imageset = "data/nuplan_mini_10hz_train_dit.pkl"
    bev_path = "/mnt/datasets/nuplan-bev/0-1-0/nuplan_bev_400/bev_400"
    gts_path = "/mnt/datasets/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_400_400_32"
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
        for occ_in, bev, occ_meta, pose_meta, scene_metas in tqdm(loader):
            occ_ori = occ_in.to(device)  # 使用输入的occupancy作为训练目标

            y = bev.to(device)
            # ref_idx = random.randint(0, Tframe-1)

            if use_vae:
                with torch.no_grad():
                    x=vae.encode(occ_ori)  # x: B C T H W  *20 make std = 1
                # import ipdb; ipdb.set_trace()
                z_ref = x[:,:,ref_idx].unsqueeze(2)
                z_ref = z_ref.repeat(1, 1, Tframe, 1, 1)
                z_ref = z_ref.to(device)
            else:
                # 修复：直接使用occupancy作为输入
                x = occ_ori.to(device)  # B T C H W
                z_ref = x[:,ref_idx].unsqueeze(1)
                z_ref = z_ref.repeat(1, Tframe, 1, 1, 1)
                z_ref = z_ref.permute(0,2,1,3,4).to(device)
                x = x.permute(0,2,1,3,4)  # B C T H W 


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
                
                # 保存latent space用于可视化
                if rank == 0 and train_steps % 1000 == 0:  # 每1000步保存一次
                    save_latent_samples(x, z_ref, noise_prior, occ_ori, experiment_dir, train_steps, vae if use_vae else None, y)
                    
                    # 每1000步也生成样本用于检查训练效果
                    try:
                        generate_training_samples(ema, diffusion, vae if use_vae else None, 
                                               experiment_dir, train_steps, device, 
                                               use_vae, scale_factor=70, dataloader=loader, real_bev=y)
                    except Exception as e:
                        print(f"生成训练样本失败: {e}")
                        import traceback
                        traceback.print_exc()
            
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
                    
                    # 生成样本用于检查训练效果
                    try:
                        generate_training_samples(ema, diffusion, vae if use_vae else None, 
                                               experiment_dir, train_steps, device, 
                                               use_vae, scale_factor=70, dataloader=loader, real_bev=y)
                    except Exception as e:
                        logger.warning(f"生成训练样本失败: {e}")
                        
                dist.barrier()

    # model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

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
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=10000)
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