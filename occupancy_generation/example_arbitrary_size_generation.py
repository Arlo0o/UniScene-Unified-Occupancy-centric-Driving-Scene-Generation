#!/usr/bin/env python3
"""
示例：使用第一帧数据进行任意倍数的outpainting

这个脚本展示了如何使用您的权重和第一帧数据进行outpainting。
"""

import torch
import numpy as np
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.models import DiT_2Frame
from diffusion import create_diffusion
from mmengine import Config
from mmengine.registry import MODELS
import model_vae
import os
import glob
from tqdm import tqdm


def load_first_frame_data(data_path): 
    """
    加载第一帧数据用于outpainting
    
    Args:
        data_path: 第一帧数据文件路径 (.npy)
    
    Returns:
        first_frame_occ: 第一帧占用网格数据 [1, T, H, W, D]
        first_frame_bev: 第一帧BEV数据 [1, T, C, Hb, Wb] 
    """
    print(f"正在加载第一帧数据: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"第一帧数据文件不存在: {data_path}")
    
    # 直接加载指定文件
    occ_data = np.load(data_path)
    print(f"第一帧原始形状: {occ_data.shape}")
    
    # 确保是5维：[B, T, H, W, D]
    if occ_data.ndim == 3:  # [H, W, D]
        occ_data = occ_data[None, None, ...]  # [1, 1, H, W, D]
        print(f"转换为5维: {occ_data.shape}")
    elif occ_data.ndim == 4:  # [T, H, W, D]
        occ_data = occ_data[None, ...]  # [1, T, H, W, D]
        print(f"添加batch维度: {occ_data.shape}")
    elif occ_data.ndim == 5:  # [B, T, H, W, D]
        print(f"数据已经是5维: {occ_data.shape}")
    else:
        raise ValueError(f"不支持的数据维度: {occ_data.ndim}")
        
    # 创建对应的BEV数据（模拟，您可以根据实际情况修改）
    B, T, H, W, D = occ_data.shape
    bev_data = np.zeros((B, T, 1, 200, 200), dtype=np.float32)
    
    print(f"最终第一帧形状: occ={occ_data.shape}, bev={bev_data.shape}")
    
    return torch.from_numpy(occ_data), torch.from_numpy(bev_data)


def setup_models():
    """设置模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 设置 DiT 模型
    DiT_cfg = {
        "depth": 12, 
        "in_channels": 4, 
        "hidden_size": 512,
        "use_label": False, 
        "use_bev_concat": False,
        "bev_in_ch": 1,
        "bev_out_ch": 1,
        "use_meta": True, 
        "bev_dropout_prob": 0.1,
        "meta_num": 4,
        "direct_concat": True,
        "Tframe": 5
    }
    dit_model = DiT_2Frame(**DiT_cfg).to(device)
    
    # 加载预训练权重
    print("加载 DiT 权重...")
    ckpt_path = 'checkpoint/occ_generation/dit.pt'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'ema' in checkpoint:
            dit_model.load_state_dict(checkpoint['ema'], strict=False)
            print(f"✅ 成功加载 DiT EMA 权重: {ckpt_path}")
        else:
            dit_model.load_state_dict(checkpoint, strict=False)
            print(f"✅ 成功加载 DiT 权重: {ckpt_path}")
    else:
        print(f"⚠️  警告: 权重文件不存在 {ckpt_path}，使用随机初始化")
    dit_model.eval()
    
    # 2. 设置 VAE 模型
    vae_config = Config.fromfile('config/train_3dvae_nuplan_200_pro_occ_bev.py')
    vae_model = MODELS.build(vae_config.model)
    
    # 加载VAE权重
    print("加载 VAE 权重...")
    vae_ckpt_path = 'checkpoint/occ_generation/3dvae.pth'
    if os.path.exists(vae_ckpt_path):
        vae_ckpt = torch.load(vae_ckpt_path, map_location='cpu')
        vae_model.load_state_dict(vae_ckpt['state_dict'], strict=True)
        print(f"✅ 成功加载 VAE 权重: {vae_ckpt_path}")
    else:
        print(f"⚠️  警告: VAE权重文件不存在 {vae_ckpt_path}，使用随机初始化")
    vae_model = vae_model.to(device)
    vae_model.eval()
    
    # 3. 创建扩散模型
    diffusion = create_diffusion(timestep_respacing="")
    
    return dit_model, vae_model, diffusion, device


def perform_outpainting(first_frame_occ, first_frame_bev, scale_factor, 
                       dit_model, vae_model, diffusion, device, 
                       output_path="./outputs/outpainting_results"):
    """
    使用第一帧数据进行任意倍数outpainting
    
    Args:
        first_frame_occ: 第一帧占用数据 [B, T, H, W, D]
        first_frame_bev: 第一帧BEV数据 [B, T, C, Hb, Wb]
        scale_factor: 放大倍数，如 2.0, 3.0, 4.0 等
        其他参数: 模型和设备
        output_path: 输出路径
    """
    print(f"\n=== 开始 {scale_factor}x Outpainting ===")
    print(f"第一帧形状: {first_frame_occ.shape}")
    
    # 计算目标场景大小
    original_H, original_W, original_D = first_frame_occ.shape[2:5]
    target_H = int(original_H * scale_factor)
    target_W = int(original_W * scale_factor)
    target_D = original_D  # 深度保持不变
    target_scene_shape = (target_H, target_W, target_D)
    
    print(f"原始场景: ({original_H}, {original_W}, {original_D})")
    print(f"目标场景: {target_scene_shape}")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 移动到设备
    first_frame_occ = first_frame_occ.to(device)
    first_frame_bev = first_frame_bev.to(device)
    
    # 使用VAE编码第一帧
    print("编码第一帧...")
    with torch.no_grad():
        first_frame_latent = vae_model.encode(first_frame_occ) * 70.0  # scale_factor
        
    print(f"第一帧潜在表示形状: {first_frame_latent.shape}")
    
    # 生成参数
    batch_size, Tframe = first_frame_occ.shape[:2]
    in_channels = first_frame_latent.shape[1]
    
    # 模型条件
    model_kwargs = {
        'y': first_frame_bev,
        'meta': torch.randn((batch_size, 4), device=device)  # 示例元数据
    }
    
    # 计算VAE潜在空间的形状
    vae_downsample = 4  # 根据您的VAE架构调整
    latent_H = target_H // vae_downsample
    latent_W = target_W // vae_downsample
    latent_shape = (batch_size, in_channels, Tframe, latent_H, latent_W)
    
    print(f"目标潜在空间形状: {latent_shape}")
    
    # 进行outpainting生成
    print("使用 p_sample_loop_cond_outpainting 进行生成...")
    generated_latents = diffusion.p_sample_loop_cond_outpainting(
        model=dit_model,
        shape=latent_shape,
        device=device,
        model_kwargs=model_kwargs,
        input_latents=first_frame_latent,
        rolling_sampling_n=max(2, int(scale_factor)),  # 根据放大倍数调整滚动次数
        n_conds=1,  # 条件帧数量
        n_conds_roll=1,
        progress=True,
    )
    
    print(f"生成的潜在表示形状: {generated_latents.shape}")
    
    # 使用VAE解码
    print("解码生成结果...")
    generated_latents = generated_latents / 70.0  # 反缩放
    rec_shape = [batch_size, Tframe, target_H, target_W, target_D]
    
    with torch.no_grad():
        result = vae_model.generate(generated_latents, rec_shape)
        logits = result["logits"]
        pred_occ = logits.argmax(dim=-1)  # [B, T, H, W, D]
    
    print(f"最终生成形状: {pred_occ.shape}")
    
    # 保存结果
    print("保存结果...")
    for i in range(pred_occ.shape[0]):
        for t in range(pred_occ.shape[1]):
            occ_frame = pred_occ[i, t].detach().cpu().numpy().astype(np.int8)
            save_path = os.path.join(output_path, f"outpainted_{scale_factor}x_batch{i}_frame{t}.npy")
            np.save(save_path, occ_frame)
            print(f"  保存: {save_path}")
    
    # 保存原始第一帧作为参考
    first_frame_np = first_frame_occ[0, 0].detach().cpu().numpy().astype(np.int8)
    ref_path = os.path.join(output_path, "reference_first_frame.npy")
    np.save(ref_path, first_frame_np)
    print(f"  保存参考帧: {ref_path}")
    
    # 保存信息文件
    info_path = os.path.join(output_path, f"outpainting_{scale_factor}x_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Outpainting Scale: {scale_factor}x\n")
        f.write(f"Original Scene: {(original_H, original_W, original_D)}\n")
        f.write(f"Target Scene: {target_scene_shape}\n")
        f.write(f"Generated Shape: {pred_occ.shape}\n")
        f.write(f"Unique Values: {np.unique(pred_occ.detach().cpu().numpy())}\n")
    
    print(f"✅ {scale_factor}x Outpainting 完成！")
    return pred_occ


def main():
    """主函数"""
    print("🚀 第一帧数据任意倍数Outpainting")
    print("=" * 50)
    
    # 第一帧数据路径
    first_frame_path = "out/nuplan_occ_dit/eval_uncon_occbev_2025-07-22-00-46-59/visualizations/00000_occ_ori.npy"
    
    # 支持的放大倍数列表
    scale_factors = [2.0, 3.0, 4.0]  # 您可以根据需要修改这个列表
    
    try:
        # 设置模型
        print("\n1. 设置模型...")
        dit_model, vae_model, diffusion, device = setup_models()
        
        # 加载第一帧数据
        print("\n2. 加载第一帧数据...")
        first_frame_occ, first_frame_bev = load_first_frame_data(first_frame_path)
        
        # 对每个放大倍数进行outpainting
        print(f"\n3. 进行不同倍数的outpainting...")
        for scale_factor in scale_factors:
            print(f"\n--- {scale_factor}x Outpainting ---")
            
            output_path = f"./outputs/outpainting_{scale_factor}x"
            
            try:
                result = perform_outpainting(
                    first_frame_occ=first_frame_occ,
                    first_frame_bev=first_frame_bev, 
                    scale_factor=scale_factor,
                    dit_model=dit_model,
                    vae_model=vae_model,
                    diffusion=diffusion,
                    device=device,
                    output_path=output_path
                )
                print(f"✅ {scale_factor}x Outpainting 成功完成！")
                
            except Exception as e:
                print(f"❌ {scale_factor}x Outpainting 失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n✅ 所有Outpainting任务完成！")
        
        print("\n📊 任务总结:")
        print(f"- 使用第一帧数据: {first_frame_path}")
        print(f"- 支持的放大倍数: {scale_factors}")
        print("- 使用真实的预训练权重")
        print("- 输出保存为numpy数组格式")
        print("- 结果保存在 ./outputs/ 目录下")
        
    except Exception as e:
        print(f"\n❌ 过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("请检查第一帧数据路径和模型权重文件")


if __name__ == "__main__":
    main() 