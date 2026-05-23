#!/usr/bin/env python3
"""
DiT训练监控和可视化脚本
用于检查训练过程中的loss计算和生成效果
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob
import torch
from mmengine import Config
from mmengine.registry import MODELS
import model_vae

def analyze_training_logs(log_dir):
    """
    分析训练日志，提取loss信息
    """
    log_file = os.path.join(log_dir, "log.txt")
    
    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        return None
    
    steps = []
    losses = []
    lrs = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if "Train Loss:" in line and "step=" in line:
                    # 解析日志行
                    parts = line.split()
                    step_idx = -1
                    loss_idx = -1
                    lr_idx = -1
                    
                    for i, part in enumerate(parts):
                        if "step=" in part:
                            step_idx = i
                        elif "lr:" in part:
                            lr_idx = i
                        elif "Train Loss:" in part:
                            loss_idx = i
                    
                    if step_idx != -1 and loss_idx != -1:
                        try:
                            step = int(parts[step_idx].split('=')[1].rstrip(','))
                            loss = float(parts[loss_idx + 1].rstrip(','))
                            
                            steps.append(step)
                            losses.append(loss)
                            
                            if lr_idx != -1:
                                lr = float(parts[lr_idx + 1].rstrip(','))
                                lrs.append(lr)
                        except (ValueError, IndexError) as e:
                            continue
    except Exception as e:
        print(f"❌ 读取日志文件失败: {e}")
        return None
    
    if not steps:
        print("❌ 没有找到有效的训练记录")
        return None
    
    return {
        'steps': steps,
        'losses': losses,
        'lrs': lrs
    }

def plot_training_curves(log_data, save_path):
    """
    绘制训练曲线
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Loss曲线
    ax1.plot(log_data['steps'], log_data['losses'], 'b-', linewidth=2)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Curve')
    ax1.grid(True, alpha=0.3)
    
    # 学习率曲线
    if log_data['lrs']:
        ax2.plot(log_data['steps'][:len(log_data['lrs'])], log_data['lrs'], 'r-', linewidth=2)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 训练曲线已保存到: {save_path}")

def visualize_latent_samples(latent_dir, save_path):
    """
    可视化保存的latent samples
    """
    if not os.path.exists(latent_dir):
        print(f"❌ Latent samples目录不存在: {latent_dir}")
        return
    
    # 查找所有latent sample文件
    x_files = sorted(glob(os.path.join(latent_dir, "*_x.npy")))
    z_ref_files = sorted(glob(os.path.join(latent_dir, "*_z_ref.npy")))
    noise_files = sorted(glob(os.path.join(latent_dir, "*_noise.npy")))
    
    if not x_files:
        print("❌ 没有找到latent sample文件")
        return
    
    # 选择几个时间点进行可视化
    selected_indices = [0, len(x_files)//2, len(x_files)-1] if len(x_files) > 2 else [0]
    
    fig, axes = plt.subplots(len(selected_indices), 3, figsize=(15, 5*len(selected_indices)))
    if len(selected_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(selected_indices):
        # 加载数据
        x_data = np.load(x_files[idx])
        z_ref_data = np.load(z_ref_files[idx]) if idx < len(z_ref_files) else None
        noise_data = np.load(noise_files[idx]) if idx < len(noise_files) else None
        
        # 提取步骤信息
        step = os.path.basename(x_files[idx]).split('_')[1]
        
        # 可视化第一个时间步和第一个通道
        if len(x_data.shape) == 4:  # [C, T, H, W]
            x_vis = x_data[0, 0]  # 第一个通道，第一个时间步
        else:
            x_vis = x_data[0]
        
        axes[i, 0].imshow(x_vis, cmap='coolwarm')
        axes[i, 0].set_title(f'Latent X (Step {step})')
        axes[i, 0].axis('off')
        
        if z_ref_data is not None:
            if len(z_ref_data.shape) == 4:
                z_ref_vis = z_ref_data[0, 0]
            else:
                z_ref_vis = z_ref_data[0]
            
            axes[i, 1].imshow(z_ref_vis, cmap='coolwarm')
            axes[i, 1].set_title(f'Z_ref (Step {step})')
            axes[i, 1].axis('off')
        
        if noise_data is not None:
            if len(noise_data.shape) == 4:
                noise_vis = noise_data[0, 0]
            else:
                noise_vis = noise_data[0]
            
            axes[i, 2].imshow(noise_vis, cmap='coolwarm')
            axes[i, 2].set_title(f'Noise Prior (Step {step})')
            axes[i, 2].axis('off')
    
    plt.suptitle('Latent Space Evolution During Training', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Latent space可视化已保存到: {save_path}")

def visualize_generated_samples(samples_dir, save_path):
    """
    可视化生成的样本
    """
    if not os.path.exists(samples_dir):
        print(f"❌ Generated samples目录不存在: {samples_dir}")
        return
    
    # 查找生成的样本文件
    latent_files = sorted(glob(os.path.join(samples_dir, "*_generated_latent.npy")))
    occ_files = sorted(glob(os.path.join(samples_dir, "*_generated_occ.npy")))
    
    if not latent_files:
        print("❌ 没有找到生成的样本文件")
        return
    
    # 选择几个时间点
    selected_indices = [0, len(latent_files)//2, len(latent_files)-1] if len(latent_files) > 2 else [0]
    
    fig, axes = plt.subplots(len(selected_indices), 5, figsize=(20, 4*len(selected_indices)))
    if len(selected_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(selected_indices):
        # 加载latent数据
        latent_data = np.load(latent_files[idx])
        step = os.path.basename(latent_files[idx]).split('_')[1]
        
        # 可视化5个时间步
        Tframe = min(5, latent_data.shape[1])
        for t in range(Tframe):
            if len(latent_data.shape) == 4:  # [C, T, H, W]
                latent_vis = latent_data[0, t]  # 第一个通道
            else:  # [T, H, W]
                latent_vis = latent_data[t]
            
            axes[i, t].imshow(latent_vis, cmap='coolwarm')
            axes[i, t].set_title(f'T={t} (Step {step})')
            axes[i, t].axis('off')
    
    plt.suptitle('Generated Samples Evolution During Training', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 生成样本可视化已保存到: {save_path}")

def decode_latent_to_occupancy(latent_file, vae_config, vae_ckpt, output_file):
    """
    使用VAE将latent space解码为occupancy
    """
    try:
        # 加载VAE模型
        cfg = Config.fromfile(vae_config)
        vae = MODELS.build(cfg.model)
        vae_ckpt_data = torch.load(vae_ckpt, map_location='cpu')
        vae.load_state_dict(vae_ckpt_data['state_dict'], strict=True)
        vae.eval()
        
        # 加载latent数据
        latent_data = np.load(latent_file)
        latent_tensor = torch.from_numpy(latent_data).float().unsqueeze(0)  # 添加batch维度
        
        # 解码
        with torch.no_grad():
            decoded_occ = vae.decode(latent_tensor)
            decoded_np = decoded_occ[0].detach().cpu().numpy()
        
        # 保存解码结果
        np.save(output_file, decoded_np)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        Tframe = min(5, decoded_np.shape[0])
        
        for t in range(Tframe):
            if len(decoded_np.shape) == 4:  # [T, C, H, W]
                occ_vis = decoded_np[t, 0]  # 第一个通道
            else:  # [T, H, W]
                occ_vis = decoded_np[t]
            
            axes[t].imshow(occ_vis, cmap='viridis')
            axes[t].set_title(f'Decoded Occupancy T={t}')
            axes[t].axis('off')
        
        plt.suptitle('Decoded Occupancy from Latent Space', fontsize=16)
        plt.tight_layout()
        
        vis_file = output_file.replace('.npy', '_vis.png')
        plt.savefig(vis_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 解码结果已保存到: {output_file}")
        print(f"✅ 可视化结果已保存到: {vis_file}")
        
    except Exception as e:
        print(f"❌ VAE解码失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='DiT训练监控和可视化')
    parser.add_argument('--experiment_dir', type=str, required=True, help='实验目录路径')
    parser.add_argument('--vae_config', type=str, help='VAE配置文件路径')
    parser.add_argument('--vae_ckpt', type=str, help='VAE检查点路径')
    parser.add_argument('--output_dir', type=str, default='training_analysis', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔍 开始分析DiT训练过程...")
    
    # 1. 分析训练日志
    print("\n📊 分析训练日志...")
    log_data = analyze_training_logs(args.experiment_dir)
    if log_data:
        plot_training_curves(log_data, os.path.join(args.output_dir, 'training_curves.png'))
        
        # 打印训练统计信息
        print(f"✅ 训练步数: {len(log_data['steps'])}")
        print(f"✅ 最终loss: {log_data['losses'][-1]:.4f}")
        print(f"✅ 最小loss: {min(log_data['losses']):.4f}")
        if log_data['lrs']:
            print(f"✅ 最终学习率: {log_data['lrs'][-1]:.6f}")
    else:
        print("❌ 无法分析训练日志")
    
    # 2. 可视化latent samples
    print("\n🎨 可视化latent samples...")
    latent_dir = os.path.join(args.experiment_dir, 'latent_samples')
    visualize_latent_samples(latent_dir, os.path.join(args.output_dir, 'latent_evolution.png'))
    
    # 3. 可视化生成的样本
    print("\n🎯 可视化生成的样本...")
    samples_dir = os.path.join(args.experiment_dir, 'generated_samples')
    visualize_generated_samples(samples_dir, os.path.join(args.output_dir, 'generated_evolution.png'))
    
    # 4. 如果有VAE，解码latent space
    if args.vae_config and args.vae_ckpt:
        print("\n🔄 解码latent space到occupancy...")
        latent_files = glob(os.path.join(latent_dir, "*_x.npy"))
        if latent_files:
            # 选择最新的latent文件
            latest_latent = sorted(latent_files)[-1]
            output_file = os.path.join(args.output_dir, 'decoded_occupancy.npy')
            decode_latent_to_occupancy(latest_latent, args.vae_config, args.vae_ckpt, output_file)
    
    print(f"\n✅ 分析完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
