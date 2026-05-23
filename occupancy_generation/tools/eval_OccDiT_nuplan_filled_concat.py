# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A testing script for DiT occupancy prediction evaluation.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader, Dataset
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
sys.path.append('.')
from utils.download import find_model

from diffusion.models import DiT_2Frame, DiT_models, DiT_Occsora, DiT_WorldModel
from diffusion import create_diffusion
from tqdm import tqdm
# from diffusers.models import AutoencoderKL

# from diffusion.bev_cod import BEV_concat_net  # 不需要BEV相关功能
import datetime
import shutil

from mmengine import Config
from mmengine.registry import MODELS

from dataset.dataload_util import CustomDataset_2frame_time,CustomDataset_2frame_continuous,CustomDataset_Tframe_continuous,Nuplan_Occ_bev_Dataset,Nuplan_Occ_bev_Dataset_pro,Nuplan_Occbev_bev_Dataset


from dataset import get_nuScenes_label_name
from utils.metric_util import MeanIoU, multi_step_MeanIou,multi_step_fid_mmd,multi_step_TemporalConsistency


from visualize.vis_Dit_time import draw_return,figure_to_array
# from sample import vis_matrix
import imageio
import matplotlib.pyplot as plt
#################################################################################
#                                                   #
#################################################################################

def vis_matrix(matrix,save_path):
    plt.clf()
    plt.imshow(matrix, origin="lower",cmap='coolwarm')
    plt.colorbar()
    plt.title("Matrix Visualization")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.savefig(save_path)
    # plt.close()

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



# find token in gts

#################################################################################
#                                  Eval Loop                                    #
#################################################################################

def main(args):
    """
    Test DiT model for occupancy prediction.
    """
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    torch.set_grad_enabled(False)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.global_seed)
    np.random.seed(args.global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.global_seed)
    
    print(f"Starting evaluation with seed={args.global_seed}")
    print(f"Evaluation mode: {'DDIM Inversion + Reconstruction' if args.inversion else 'Pure Generation from Noise'}")
    print(f"Model uses x_ref (first frame) as condition for generating subsequent frames")
    
    # 匹配训练代码的配置
    use_bev_concat=False  # 不使用BEV concat
    use_occ_meta=False    # 不使用occupancy metadata
    in_ch=4
    T_pred = 6
    T_condition = 2
    Tframe = 5  # 匹配训练代码
    meta_num= 4 + 12*(T_pred-1)
    use_noise_prior = False
    vis=args.vis
    ref_idx = 0
    scale_factor = 70
    use_vq=False

    # 匹配训练代码的bev_ch_use配置
    bev_ch_use=[0,2,8,9,10,11,12,13,14,15,16,17]
    lambda_np = getattr(args, 'lambda_noise_prior', getattr(args, 'lambda-noise-prior', 0.05))

    # 匹配训练代码的DiT配置
    DiT_cfg={"depth":12, "in_channels":in_ch, "hidden_size":512,"use_label":False, "use_bev_concat":use_bev_concat,"bev_in_ch":1,"bev_out_ch":1,"use_meta":use_occ_meta, "bev_dropout_prob":0.1,"meta_num":meta_num,"T_pred":5,"T_condition":1}
    
    # 匹配训练代码使用DiT_WorldModel
    Dit_model = DiT_WorldModel(**DiT_cfg).to(device)
    # Dit_model = DiT_2Frame(**DiT_cfg).to(device)
    # Dit_model = DiT_Occsora(**DiT_cfg).to(device)
    
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    
    # 加载checkpoint，区分EMA和主模型权重
    if ckpt_path.endswith('.pt') and 'ema' not in ckpt_path:
        # 如果是训练checkpoint，包含EMA权重
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'ema' in checkpoint:
            print("Loading EMA weights from checkpoint for evaluation")
            Dit_model.load_state_dict(checkpoint['ema'], strict=True)
        elif 'model' in checkpoint:
            print("Loading model weights from checkpoint for evaluation")  
            Dit_model.load_state_dict(checkpoint['model'], strict=True)
        else:
            print("Loading direct state_dict from checkpoint")
            Dit_model.load_state_dict(checkpoint, strict=True)
    else:
        # 传统方式加载
        state_dict = find_model(ckpt_path)
        Dit_model.load_state_dict(state_dict, strict=True)
    
    Dit_model.eval()  # important!
    
    # 调试信息：输出模型参数统计
    total_params = sum(p.numel() for p in Dit_model.parameters())
    print(f"DiT Model loaded with {total_params:,} parameters")
    print(f"Model device: {next(Dit_model.parameters()).device}")
    print(f"Using CFG scale: {args.cfg_scale if hasattr(args, 'cfg_scale') else 'default'}")

    ckpt_path_list = ckpt_path.split('/')
    # result_dir = f"outputs/nuplan_occ_dit_eval"
    result_dir = args.results_dir
    ct_str=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") 
    log_path = f"{result_dir}/{ct_str}"
    os.makedirs(log_path, exist_ok=True)
    vis_root=f"{result_dir}/{ct_str}/vis_continuous_{args.inversion}_{args.cfg_scale}/video/"
    zvis_root = f"{result_dir}/{ct_str}/vis_continuous_{args.inversion}_{args.cfg_scale}/zvis/"
    save_occ_root = f"{result_dir}/{ct_str}/save_occ/"
    # os.makedirs(zvis_root, exist_ok=True)
    os.makedirs(save_occ_root, exist_ok=True)
    # 匹配训练时的采样设置：使用25步DDIM采样
    diffusion = create_diffusion('ddim25')  # 匹配训练时采样步数
    iter_num = ckpt_path_list[-1].split('.')[0]
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    # logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
 
    # imageset = "data/nuplan_mini_10hz_val_occ.pkl"

    # imageset = "/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_val.pkl"
    imageset = "/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_train.pkl"
    # imageset = "/lpai/volumes/ad-lmm-data-proc-bd-ga/hzhu/code/occ_gen/data/nuplan_mini_10hz_val_nes_keys.pkl"
    # file1_path = './step2/train/Zmid_4_me/'
    bev_path = '/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200'
    # bev_path = '/lpai/dataset/nuplan-bev/0-1-0/nuplan_bev/mini/train'
    gts_path = "/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ/nuplan_quantized_200_200_16"

    # 匹配训练代码使用Nuplan_Occbev_bev_Dataset
    dataset = Nuplan_Occbev_bev_Dataset(imageset, gts_path, bev_path, bev_ch_use, meta_num=4, Tframe=5, training=False)
    # dataset = Nuplan_Occ_bev_Dataset(imageset,gts_path,bev_path,bev_ch_use,meta_num=4,Tframe=Tframe)
    # dataset = Nuplan_Occ_bev_Dataset_pro(imageset,gts_path,bev_path,bev_ch_use,meta_num=4,Tframe=Tframe)

    p_time = Tframe 
    
    bz = args.dit_batch_size
    if vis:
        bz = 1
    
    loader = DataLoader(
        dataset,
        batch_size=bz,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )  


    # Prepare model for evaluation:
    Dit_model.eval()  # important! This enables embedding dropout for classifier-free guidance

    cfg1 = Config.fromfile(args.vae_config)
    import model_vae
    my_VQVAE = MODELS.build(cfg1.model)
    my_VQVAE = my_VQVAE.to(device)

    resume_from = args.vae_ckpt
    vae_ckpt = torch.load(resume_from,map_location='cpu')
    my_VQVAE.load_state_dict(vae_ckpt['state_dict'], strict=True)

    my_VQVAE.eval()


    from model_vae.VAE.AE_eval import Autoencoder_2D
    ae_eval = Autoencoder_2D(num_classes=18,expansion=4)

    ae_ckpt_path = getattr(args, 'ae_ckpt', getattr(args, 'ae-ckpt', 'ckpts/AE_eval/epoch_196.pth'))
    ae_ckpt = torch.load(ae_ckpt_path,map_location='cpu')
    ae_eval.load_state_dict(ae_ckpt['state_dict'], strict=True)
    ae_eval = ae_eval.to(device)
    ae_eval.eval()

    #logger
    from mmengine.logging import MMLogger
    # if not vis:
    log_file = os.path.join(log_path, f'Dit_eval_ddim_continuous_mVAE_{iter_num}_{args.cfg_scale}_{lambda_np}.log')
    logger = MMLogger('genocc', log_file=log_file)
    MMLogger._instance_dict['genocc'] = logger
    logger.info(f'Cfg scale:{args.cfg_scale}')
    logger.info(f'Evaluation mode: {"DDIM Inversion + Reconstruction" if args.inversion else "Pure Generation from Noise"}')
    logger.info(f'Model: DiT_WorldModel with x_ref as condition')

    label_name = get_nuScenes_label_name(cfg1.label_mapping)
    unique_label = np.asarray(cfg1.unique_label)
    unique_label_str = [label_name[l] for l in unique_label]
    CalMeanIou_sem = multi_step_MeanIou(unique_label, cfg1.get('ignore_label', -100), unique_label_str, 'sem', times=p_time)
    CalMeanIou_sem.reset()

    CalMeanIou_vox = multi_step_MeanIou([1], cfg1.get('ignore_label', -100), ['occupied'], 'vox', times=p_time)
    CalMeanIou_vox.reset()

    Cal_fid_mmd = multi_step_fid_mmd()

    Cal_TC = multi_step_TemporalConsistency("TC",times=p_time)
    Cal_TC.reset()
    # Cal_TC_real = multi_step_TemporalConsistency("TC_real",times=p_time)
    # Cal_TC_real.reset()
    with torch.no_grad():
        # for epoch in range(start_epoch, args.epochs):
            # sampler.set_epoch(epoch)

        for i_iter_val, (occ_ori, occ_gt, y, occ_meta, pose_meta) in enumerate(tqdm(loader)):
            # scene_idx = [2,7,14,21,26,36,40]
            # if i_iter_val not in scene_idx and vis==True:
            #     continue

            occ_ori = occ_ori.to(device)
            occ_ori = fill_large_zero_height_regions_gpu(occ_ori, target_layer=8, fill_class=1)
            y = y.to(device)
            images=[]
            # n_time=z_ori.shape[1]-1
            
            with torch.no_grad():
                z_ori=my_VQVAE.encode(occ_ori) *scale_factor # x: B C T H W
                x_ref = z_ori[:, :, 0:1, :, :] # bs, 4,1, 50, 50 - 参考帧（condition）
            
            # 根据是否使用inversion来决定起始点
            if args.inversion:
                # 使用DDIM inversion：从真实数据开始，用于测试reconstruction能力
                z = z_ori  
            else:
                # 纯生成：从噪声开始，测试真正的生成能力
                z = torch.randn_like(z_ori)
                # 或者使用noise prior（可选）
                # z = torch.randn_like(z_ori) + lambda_np * z_ori  

            z = torch.cat([z, z], 0)
            y_null = -torch.ones_like(y)
            y = torch.cat([y, y_null], 0)

            x_ref = x_ref.to(device)
            x_ref_null = torch.zeros_like(x_ref)  # 或者使用 -torch.ones_like(x_ref)
            x_ref_combined = torch.cat([x_ref, x_ref_null], 0)
            
            if use_occ_meta:  # 这个分支不会执行，因为use_occ_meta=False
                occ_meta = occ_meta.to(device)
                pose_meta = pose_meta.to(device)
                new_meta = torch.cat((occ_meta[:,ref_idx],pose_meta.reshape(-1,12*(T_pred-1))),dim=1)
                new_meta = new_meta.to(device)
                new_meta = torch.cat([new_meta, new_meta], 0)
                model_kwargs = dict(x_ref=torch.cat([x_ref, x_ref], 0), y=y, meta=new_meta, cfg_scale=args.cfg_scale)
            else:
                model_kwargs = dict(x_ref=x_ref_combined, y=y, cfg_scale=args.cfg_scale)
            
            # 调试信息：检查CFG设置
            if i_iter_val == 0:
                print(f"CFG Debug Info:")
                print(f"  z shape: {z.shape} (should be [2*batch_size, ...])")
                print(f"  y shape: {y.shape} (conditional + unconditional)")
                print(f"  x_ref_combined shape: {x_ref_combined.shape}")
                print(f"  cfg_scale: {args.cfg_scale}")
                print(f"  y[0] range: [{y[0].min().item():.2f}, {y[0].max().item():.2f}] (conditional)")
                print(f"  y[{y.shape[0]//2}] range: [{y[y.shape[0]//2].min().item():.2f}, {y[y.shape[0]//2].max().item():.2f}] (unconditional)")
            
            if args.inversion==True:
                z_ori_inv = torch.cat([z_ori, z_ori], 0)  #ddim inversion
                latent = z_ori_inv.clone().detach()
                model_kwargs_inv = model_kwargs.copy()  # 使用相同的model_kwargs
                for t in range(20,args.num_sampling_steps):
                    t_inv = torch.full((z.shape[0],), t).to(device)
                    # t_inv = torch.randint(0, 1, (z.shape[0],), device=device)
                    ddim_inversion_samples = diffusion.ddim_reverse_sample(Dit_model.forward_with_cfg,latent,t_inv,model_kwargs=model_kwargs_inv)
                    # c_latent,uc_latent = ddim_inversion_samples['sample'].chunk(2, dim=0)
                    # latent = torch.cat([c_latent, c_latent], 0)
                    latent = ddim_inversion_samples['sample']
                    # vis_matrix(latent[0,1,0].cpu().numpy(),zvis_root+f"{i_iter_val}_01_{t}.png")
                inversion_noise,_ = latent.chunk(2, dim=0)
                z = torch.cat([inversion_noise, inversion_noise], 0)

            samples = diffusion.ddim_sample_loop(
                Dit_model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device)
            
            # t_ = torch.tensor([999] * z.shape[0], device=device)
            # # t_ = t_.to(device)
            # one_step_sample = diffusion.ddim_sample(
            #     Dit_model.forward_with_cfg, z, t_, clip_denoised=False, model_kwargs=model_kwargs)
            # samples = one_step_sample["pred_xstart"]
            
            samples, samples_uncond = samples.chunk(2, dim=0)  # 分离conditional和unconditional结果
            
            # 调试信息：检查CFG结果
            if i_iter_val == 0:
                print(f"CFG Results Debug:")
                print(f"  conditional samples shape: {samples.shape}")
                print(f"  unconditional samples shape: {samples_uncond.shape}")
                print(f"  conditional samples range: [{samples.min().item():.3f}, {samples.max().item():.3f}]")
                print(f"  unconditional samples range: [{samples_uncond.min().item():.3f}, {samples_uncond.max().item():.3f}]")
            
            # 使用conditional结果进行后续处理
            # samples = samples  # 已经是conditional结果
            
            # samples = samples.permute(0,2,1,3,4)
            # samples = samples.reshape(-1,4,50,50) 
            
            # shapes=[torch.Size([200, 200]), torch.Size([100, 100])]
            samples = samples / scale_factor
            
            # 调试信息：检查latent统计
            if i_iter_val < 3:  # 只对前几个样本输出调试信息
                print(f"Sample {i_iter_val} - Latent stats:")
                print(f"  Shape: {samples.shape}")
                print(f"  Range: [{samples.min().item():.3f}, {samples.max().item():.3f}]")
                print(f"  Mean: {samples.mean().item():.3f}, Std: {samples.std().item():.3f}")

            # Use actual batch/time dims to avoid shape mismatch on last batch
            b_cur = samples.shape[0]
            t_cur = samples.shape[2]
            rec_shape=[b_cur, t_cur, 200, 200, 16]
            if use_vq==False:
                result=my_VQVAE.generate(samples,rec_shape)
            else:
                result=my_VQVAE.generate_vq(samples,rec_shape)
            logit=result["logits"]
            pred = logit.argmax(dim=-1) #  200, 200, 16
            
            # 调试信息：检查类别分布
            if i_iter_val < 3:  # 只对前几个样本输出调试信息
                pred_classes = torch.unique(pred, return_counts=True)
                gt_classes = torch.unique(occ_ori, return_counts=True)
                print(f"Sample {i_iter_val} - Class distribution:")
                print(f"  GT classes: {gt_classes[0].cpu().numpy()}")
                print(f"  GT counts: {gt_classes[1].cpu().numpy()}")
                print(f"  Pred classes: {pred_classes[0].cpu().numpy()}")  
                print(f"  Pred counts: {pred_classes[1].cpu().numpy()}")
                print(f"  Logit stats: mean={logit.mean().item():.3f}, std={logit.std().item():.3f}")
                print(f"  Max logit per class: {logit.max(dim=-1)[0].mean().item():.3f}")
            
            # save npy
            np.save(f"{save_occ_root}/{i_iter_val}_gt_occ.npy", occ_ori.cpu().numpy())
            np.save(f"{save_occ_root}/{i_iter_val}_pred.npy", pred.cpu().numpy())
            # np.save(f"{save_occ_root}/{i_iter_val}_{scene_metas['scene_tokens'][0][0]}_gt_occ.npy", occ_ori.cpu().numpy())
            # np.save(f"{save_occ_root}/{i_iter_val}_{scene_metas['scene_tokens'][0][0]}_pred.npy", pred.cpu().numpy())
            
            CalMeanIou_sem._after_step(pred, occ_ori)

            target_occs_iou = deepcopy(occ_ori)
            target_occs_iou[target_occs_iou != 0] = 1
            target_occs_iou[target_occs_iou == 0] = 0
            pred_iou = deepcopy(pred)
            pred_iou[pred_iou!=0] = 1
            pred_iou[pred_iou==0] = 0
            
            CalMeanIou_vox._after_step(pred_iou, target_occs_iou)

            # occ_ori_noT = occ_ori.reshape(-1,200,200,16)
            # pred_noT = pred.reshape(-1,200,200,16)
            ae_feature_ori = ae_eval.forward_eval(occ_ori) #B*T,2048
            ae_feature_gen = ae_eval.forward_eval(pred)

            Cal_fid_mmd._after_step(ae_feature_ori,ae_feature_gen)
            # Use actual batch/time dims for TC metric
            Cal_TC._after_step(ae_feature_gen.reshape(pred.shape[0], pred.shape[1], -1))

            # if i_iter_val > 25:
            #     fid,mmd = Cal_fid_mmd._after_epoch()
            #     break
            # fid,mmd = Cal_fid_mmd._after_epoch()
            # print(fid,mmd)
            if vis:
                os.makedirs(vis_root,exist_ok=True)
                os.makedirs(zvis_root,exist_ok=True)

                pred = pred.squeeze().cpu().numpy()
                occ_ori = occ_ori.squeeze().cpu().numpy()
                
                for i in range(Tframe):
                    # 创建只有两个子图的布局：生成的和原始的occupancy
                    fig=plt.figure(figsize=(14,7))
                    
                    # 生成的occupancy
                    pred_i=pred[i]
                    fov_voxels,p_colors =draw_return(pred_i, 
                        None, # predict_pts,
                        [-40, -40, -1], 
                        [0.4] * 3, 
                        )
                    ax = fig.add_subplot(121, projection='3d') 
                    ax.scatter(fov_voxels[:, 0],fov_voxels[:, 1],fov_voxels[:, 2],c=p_colors,s=1)
                    ax.set_box_aspect([1,1,0.125])  
                    ax.view_init(elev=90, azim=-90)
                    ax.set_xlim(xmin = -40, xmax = 40)
                    ax.set_ylim(ymin = -40, ymax = 40)
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                    ax.set_zlabel('Z (m)')
                    ax.set_title(f"Generated Occupancy\nFrame {i+1}/{Tframe}")

                    # 原始的occupancy
                    occ_i=occ_ori[i]
                    fov_voxels,p_colors =draw_return(occ_i, 
                        None, # predict_pts,
                        [-40, -40, -1], 
                        [0.4] * 3, 
                        )
                    ax_ori = fig.add_subplot(122, projection='3d')
                    ax_ori.scatter(fov_voxels[:, 0],fov_voxels[:, 1],fov_voxels[:, 2],c=p_colors,s=1)
                    ax_ori.set_box_aspect([1,1,0.125])  
                    ax_ori.view_init(elev=90, azim=-90)
                    ax_ori.set_xlim(xmin = -40, xmax = 40)
                    ax_ori.set_ylim(ymin = -40, ymax = 40)
                    ax_ori.set_xlabel('X (m)')
                    ax_ori.set_ylabel('Y (m)')
                    ax_ori.set_zlabel('Z (m)')
                    ax_ori.set_title(f"Ground Truth Occupancy\nFrame {i+1}/{Tframe}")

                    fig.suptitle(f"Occupancy Prediction Results (Sample {i_iter_val})\nCFG Scale: {args.cfg_scale}, DDIM Inversion: {args.inversion}", fontsize=14)
                    fig.tight_layout()
                    fig_np=figure_to_array(fig)
                    images.append(fig_np)
                    plt.close(fig)

                    # 可选：保存latent可视化
                    z_ori_vis = z_ori.squeeze().cpu().numpy()
                    vis_matrix(z_ori_vis[0,i],zvis_root+f"{i_iter_val}_{i}_0.png")

                print(f"vis_root is {vis_root}")
                # 保存视频
                imageio.mimsave(vis_root + f"{args.inversion}_inversion_{iter_num}_{i_iter_val}_{lambda_np}_{args.cfg_scale}.mp4",images,'mp4',fps = 4)
                # 保存最后一帧的图片
                if len(images) > 0:
                    plt.figure(figsize=(12,6))
                    plt.imshow(images[-1])
                    plt.axis('off')
                    plt.title(f"Final Frame - cfg = {args.cfg_scale}")
                    plt.savefig(vis_root + f"{i_iter_val}_final_frame_cfg-{args.cfg_scale}.png", bbox_inches='tight')
                    plt.close()

        val_miou, avg_val_miou = CalMeanIou_sem._after_epoch() 
        val_iou, avg_val_iou = CalMeanIou_vox._after_epoch() 
        
        logger.info(f'Avg mIoU: %.2f%%' % (avg_val_miou))
        logger.info(f'Avg IoU: %.2f%%' % (avg_val_iou))

        val_TC = Cal_TC._after_epoch()
        logger.info(f'Avg TC: %.4f' % (val_TC))
        fid,mmd = Cal_fid_mmd._after_epoch()
        logger.info(f'FID: %.4f' % (fid))
        logger.info(f'MMD: %.6f' % (mmd))


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument('--inversion', action='store_true', default=False)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="xx")
    parser.add_argument("--image-size", type=int, choices=[256, 128], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dit-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=2.0)  # 匹配训练时的CFG scale
    parser.add_argument("--in-ch", type=int, default=4)
    # parser.add_argument("--log-every", type=int, default=500)
    # parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument("--vae-ckpt", type=str, default="")
    parser.add_argument("--vae-config", type=str, default="")
    parser.add_argument("--ae-ckpt", type=str, default="ckpts/AE_eval/epoch_196.pth")
    parser.add_argument("--lambda-noise-prior", type=float, default=0.05)
    # parser.add_argument('--vae-config', default='./config/train_vqvae_4.py')
    parser.add_argument("--num-sampling-steps", type=int, default=25)  # 匹配训练时采样步数
    parser.add_argument("--results-dir", type=str, default="outputs/eval_OccDiT_outpainting_concate")
    args = parser.parse_args()
    main(args)
    
    # Usage examples:
    # python eval_OccDiT_nuplan_filled_concat.py --ckpt="/path/to/checkpoint.pt" --vae_ckpt="/path/to/vae.pt" --vae_config="/path/to/config.py"
    # python eval_OccDiT_nuplan_filled_concat.py --vis --ckpt="/path/to/checkpoint.pt" --vae_ckpt="/path/to/vae.pt" --vae_config="/path/to/config.py"

    
