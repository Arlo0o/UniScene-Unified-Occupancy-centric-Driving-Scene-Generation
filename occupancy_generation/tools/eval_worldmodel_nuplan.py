# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import sys
from pathlib import Path
# 获取当前文件的父目录的父目录（上一级目录）
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
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
from utils.download import find_model

from diffusion.models import DiT_WorldModel
from diffusion import create_diffusion
from tqdm import tqdm
# from diffusers.models import AutoencoderKL

from diffusion.bev_cod import BEV_concat_net
import datetime
import shutil

from mmengine import Config
from mmengine.registry import MODELS

from dataset.dataload_util import CustomDataset_wm_continuous_pro

from dataset import get_nuScenes_label_name
from utils.metric_util import MeanIoU, multi_step_MeanIou,multi_step_fid_mmd


import imageio
import matplotlib.pyplot as plt
#################################################################################
#                                                   #
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


# find token in gts

#################################################################################
#                                  Eval Loop                                    #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    torch.set_grad_enabled(False)
    # Variables for monitoring/logging purposes:
    start_epoch = 0
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()



    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    rank=args.local_rank
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    # Tframe = 5
    T_pred = 6
    T_condition = 2
    use_bev_concat=True
    use_occ_meta=True
    use_vq=False
    # meta_num=4*Tframe + 12*(Tframe-1)
    meta_num= 4 + 12*(T_pred-1)
    # meta_num= 4 + 12*(Tframe-1)
    in_ch=4
    vis=args.vis
    ref_idx = 0
    scale_factor = 70

    bev_ch_use=[0,2,8,9,10,11,12,13,14,15,16,17]
    
    DiT_cfg={"depth":12, "in_channels":in_ch, "hidden_size":512,"use_label":False, "use_bev_concat":use_bev_concat,"bev_in_ch":1,"bev_out_ch":1,"use_meta":use_occ_meta, "bev_dropout_prob":0.1,"meta_num":meta_num,"direct_concat":True,"T_pred":T_pred,"T_condition":T_condition}
    # DiT_cfg={"depth":6, "in_channels":in_ch, "hidden_size":512,"use_label":False, "use_bev_concat":use_bev_concat,"bev_in_ch":1,"bev_out_ch":1,"use_meta":use_occ_meta, "bev_dropout_prob":0.1,"meta_num":meta_num,"direct_concat":True,"T_pred":T_pred,"T_condition":T_condition}
    # model = DiT_Occsora(**DiT_cfg).to(device)
    Dit_model = DiT_WorldModel(**DiT_cfg).to(device)
    

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    Dit_model.load_state_dict(state_dict,strict=True)
    Dit_model.eval()  # important!

    ckpt_path_list = ckpt_path.split('/')
    log_path='/'.join(ckpt_path_list[:-2])
    vis_root='/'.join(ckpt_path_list[:-2]) +'/vis_continuous/video/'
    zvis_root = '/'.join(ckpt_path_list[:-2]) +'/vis_continuous/zvis/'
    diffusion = create_diffusion(str(args.num_sampling_steps))  # default: 1000 steps, linear noise schedule
    iter_num = ckpt_path_list[-1].split('.')[0]
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    Dit_model = DDP(Dit_model.to(device), device_ids=[rank])
    # y_net =DDP(y_net.to(device), device_ids=[rank],find_unused_parameters=True)
    
    # logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
 
    
    imageset = "/lpai/volumes/ad-lmm-data-proc-bd-ga/hzhu/data/nuplan_pkls/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_val_noted_with_ego_stationary_dynamic_others.pkl"
    file1_path = './step2/train/Zmid_4_me/'
    bev_path = '/lpai/dataset/nuplan-bev/0-1-0/nuplan_bev/mini/val'
    gts_path = "/lpai/dataset/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_200_200_16/"
    npy_save_path = '/mnt/volumes/ad-lmm-data-proc-bd-ga/hzhu/code/occ_gen/results/wm_0430'
    os.makedirs(npy_save_path,exist_ok=True)

    dataset = CustomDataset_wm_continuous_pro(imageset,gts_path,file1_path,bev_path,bev_ch_use,meta_num=4,Tframe=T_condition+T_pred)


    p_time = T_pred 
    
    bz = int(args.global_batch_size // dist.get_world_size())
    # bz=1
    if vis:
        bz = 1
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=bz,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )  


    # Prepare models for training:
    # update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    Dit_model.eval()  # important! This enables embedding dropout for classifier-free guidance
    # ema.eval()  # EMA model should always be in eval mode

    cfg1 = Config.fromfile(args.vae_config)
    import model_vae
    my_VQVAE = MODELS.build(cfg1.model)
    my_VQVAE = my_VQVAE.to(device)

    resume_from = args.vae_ckpt
    vae_ckpt = torch.load(resume_from,map_location='cpu')
    my_VQVAE.load_state_dict(vae_ckpt['state_dict'], strict=True)

    # my_VQVAE = DDP(my_VQVAE.to(device), device_ids=[rank])
    # my_VQVAE = my_VQVAE.to(device)
    # # eval
    my_VQVAE.eval()


    from model_vae.VAE.AE_eval import Autoencoder_2D
    ae_eval = Autoencoder_2D(num_classes=18,expansion=4)

    ae_ckpt_path = args.ae_ckpt
    ae_ckpt = torch.load(ae_ckpt_path,map_location='cpu')
    ae_eval.load_state_dict(ae_ckpt['state_dict'], strict=True)
    ae_eval = ae_eval.to(device)
    ae_eval.eval()

    #logger
    from mmengine.logging import MMLogger
    if not vis:
        log_file = os.path.join(log_path, f'Dit_eval_ddim_continuous_mVAE_{iter_num}_{args.cfg_scale}.log')
        logger = MMLogger('genocc', log_file=log_file)
        MMLogger._instance_dict['genocc'] = logger
        logger.info(f'Cfg scale:{args.cfg_scale}')

    label_name = get_nuScenes_label_name(cfg1.label_mapping)
    unique_label = np.asarray(cfg1.unique_label)
    unique_label_str = [label_name[l] for l in unique_label]
    CalMeanIou_sem = multi_step_MeanIou(unique_label, cfg1.get('ignore_label', -100), unique_label_str, 'sem', times=p_time)
    CalMeanIou_sem.reset()

    CalMeanIou_vox = multi_step_MeanIou([1], cfg1.get('ignore_label', -100), ['occupied'], 'vox', times=p_time)
    CalMeanIou_vox.reset()

    Cal_fid_mmd = multi_step_fid_mmd()

    with torch.no_grad():
        for epoch in range(start_epoch, args.epochs):
            sampler.set_epoch(epoch)

            # for i_iter_val, (occ_ori,y,occ_meta, pose_meta) in enumerate(tqdm(loader)):
            # for i_iter_val, (x_all,y_all,occ_all,occ_meta_all,pose_meta_all) in enumerate(tqdm(loader)):
            for i_iter_val, (bev_all,occ_all,occ_meta_all,pose_meta_all) in enumerate(tqdm(loader)):
                occ_all = occ_all.to(device)
                with torch.no_grad():
                    vae_out = my_VQVAE(occ_all)
                mid_z = vae_out["middd"]

                x = mid_z[:,:,T_condition:]
                x_ref = mid_z[:,:,:T_condition]
                y = bev_all[:,T_condition:]
                
                x = x.to(device)
                y = y.to(device)
                x_ref = x_ref.to(device)


                occ_ori = occ_all[:,T_condition:]
                occ_ori = occ_ori.to(device)
                occ_meta = occ_meta_all[:,T_condition:]
                pose_meta = pose_meta_all[:,T_condition:]

                # x = x.to(device) * scale_factor
                x_ref = x_ref.to(device) * scale_factor

                # z = x.permute(0,2,1,3,4) 
                z_ref = x_ref
                # z_ref = x_ref.permute(0,2,1,3,4) 
                z = torch.randn((bz,4, T_pred, 50,50), device=device)# + 0.05* z_ori_s
                # z = z.permute(0,2,1,3,4)
                print(f"z shape: {z.shape}, x shape: {x.shape}, x_ref shape: {x_ref.shape}, z_ref shape: {z_ref.shape}")
                # z = z_ori_s


                z = torch.cat([z, z], 0)
                z_ref = torch.cat([z_ref, z_ref], 0)
                y_null = -torch.ones_like(y)
                y = torch.cat([y, y_null], 0)
                print(f"z_ref shape: {z_ref.shape}, z shape: {z.shape}, x shape: {x.shape}, y shape: {y.shape}")

                if use_occ_meta:
                    new_meta = torch.cat((occ_meta[:,ref_idx],pose_meta.reshape(-1,12*(T_pred-1))),dim=1)
                    # new_meta = torch.cat((occ_meta.reshape(-1,4*Tframe),pose_meta.reshape(-1,12*(Tframe-1))),dim=1)
                    # new_meta = torch.cat((occ_meta[:,ref_idx],pose_meta.reshape(-1,12*(Tframe-1))),dim=1)
                    new_meta = new_meta.to(device)
                    new_meta = torch.cat([new_meta, new_meta], 0)

                    # new_meta_null = torch.zeros_like(new_meta)
                    # new_meta = torch.cat([new_meta, new_meta_null], 0)
                    model_kwargs = dict(x_ref=z_ref,y=y, meta=new_meta, cfg_scale=args.cfg_scale)

                else:
                    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                
                # z_ori_inv = torch.cat([z_ori, z_ori], 0)
                # latent = z_ori_inv.clone().detach()
                # model_kwargs_inv = dict(y=y, meta=new_meta, cfg_scale=1)
                # for t in range(0,args.num_sampling_steps):
                #     t_inv = torch.full((z.shape[0],), t).to(device)
                #     # t_inv = torch.randint(0, 1, (z.shape[0],), device=device)
                #     ddim_inversion_samples = diffusion.ddim_reverse_sample(Dit_model.module.forward_with_cfg,latent,t_inv,model_kwargs=model_kwargs_inv)
                #     # c_latent,uc_latent = ddim_inversion_samples['sample'].chunk(2, dim=0)
                #     # latent = torch.cat([c_latent, c_latent], 0)
                #     # vis_matrix(latent[0,1,0].cpu().numpy(),zvis_root+f"{i_iter_val}_01_{t}.png")

                # samples,_ = latent.chunk(2, dim=0)
                # z = torch.cat([samples, samples], 0)

                

                samples = diffusion.ddim_sample_loop(
                    Dit_model.module.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device)
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                
                # samples = samples.permute(0,2,1,3,4)
                # samples = samples.reshape(-1,4,50,50) 
                
                # shapes=[torch.Size([200, 200]), torch.Size([100, 100])]
                samples = samples / scale_factor

                rec_shape=[bz,T_pred,200,200,16]
                if use_vq==False:
                    result=my_VQVAE.generate(samples,rec_shape)
                else:
                    result=my_VQVAE.generate_vq(samples,rec_shape)
                logit=result["logits"]
                pred = logit.argmax(dim=-1) #  200, 200, 16

                # save pred and occ_ori
                pred_save = pred.cpu().numpy()
                occ_ori_save = occ_ori.cpu().numpy()
                np.save(f'{npy_save_path}/pred_{i_iter_val}.npy', pred_save.astype(np.int8))
                np.save(f'{npy_save_path}/occ_ori_{i_iter_val}.npy', occ_ori_save.astype(np.int8))
                
                CalMeanIou_sem._after_step(pred, occ_ori)

                target_occs_iou = deepcopy(occ_ori)
                target_occs_iou[target_occs_iou != 17] = 1
                target_occs_iou[target_occs_iou == 17] = 0
                pred_iou = deepcopy(pred)
                pred_iou[pred_iou!=17] = 1
                pred_iou[pred_iou==17] = 0
                
                CalMeanIou_vox._after_step(pred_iou, target_occs_iou)

                # occ_ori_noT = occ_ori.reshape(-1,200,200,16)
                # pred_noT = pred.reshape(-1,200,200,16)
                ae_feature_ori = ae_eval.forward_eval(occ_ori) #B*T,2048
                ae_feature_gen = ae_eval.forward_eval(pred)

                Cal_fid_mmd._after_step(ae_feature_ori,ae_feature_gen)
                
            val_miou, avg_val_miou = CalMeanIou_sem._after_epoch() 
            val_iou, avg_val_iou = CalMeanIou_vox._after_epoch() 
            
            logger.info(f'Avg mIoU: %.2f%%' % (avg_val_miou))
            logger.info(f'Avg IoU: %.2f%%' % (avg_val_iou))

            if rank==0:
                fid,mmd = Cal_fid_mmd._after_epoch()
                logger.info(f'FID: %.2f' % (fid))
                logger.info(f'MMD: %.6f' % (mmd))


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=4)
    # parser.add_argument("--log-every", type=int, default=500)
    # parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--ckpt", type=str, default="/mnt/volumes/ad-lmm-data-proc-bd-ga/hzhu/code/occ_gen/ckpt/wm/0244400.pt")
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument("--local-rank", type=int,default=0)
    parser.add_argument("--vae_ckpt", type=str, default="./ckpt/3d_vae_200_10hz/epoch_61.pth")
    parser.add_argument("--vae_config", type=str, default="./ckpt/3d_vae_200_10hz/train_3dvae_nuplan_200.py")
    parser.add_argument("--ae_ckpt", type=str, default="ckpt/AE_eval/epoch_196.pth")
    # parser.add_argument('--vae-config', default='./config/train_vqvae_4.py')
    parser.add_argument("--num-sampling-steps", type=int, default=50) #1000
    args = parser.parse_args()
    main(args)
    
