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
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
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

from diffusion.models import DiT_2Frame, DiT_models, DiT_gjz, DiT_Occsora
from diffusion import create_diffusion
from tqdm import tqdm
# from diffusers.models import AutoencoderKL

from diffusion.bev_cod import BEV_concat_net
import datetime
import shutil

from mmengine import Config
from mmengine.registry import MODELS

from dataset.dataload_util import CustomDataset_nBEV1,CustomDataset_Tframe_12hz

from dataset import get_nuScenes_label_name
from utils.metric_util import MeanIoU, multi_step_MeanIou,multi_step_fid_mmd,multi_step_TemporalConsistency


# from pyvirtualdisplay import Display
# display = Display(visible=False, size=(2560, 1440))
# display.start()

# from mayavi import mlab
# import mayavi
# mlab.options.offscreen = True
# print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

# from vis_Dit_time import draw_return,figure_to_array
# from visualize_demo import draw
from sample import vis_matrix
# import imageio
# import matplotlib.pyplot as plt

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
    # dist.destroy_process_group()
    pass


class OpenSceneDataset(Dataset):
    def __init__(self, npy_file_paths, npz_file_paths, transform=None):
        """
        初始化数据集

        :param npy_file_paths: List of .npy file paths (e.g., the OpenScene data for occ)
        :param npz_file_paths: List of .npz file paths (e.g., the BEV data)
        :param transform: 可选的数据预处理（如有需要）
        """
        self.npy_file_paths = npy_file_paths
        self.npz_file_paths = npz_file_paths
        self.transform = transform

    def __len__(self):
        return len(self.npy_file_paths)

    def __getitem__(self, idx):
        """
        加载单个数据对
        """
        # 加载npy文件
        occ_data = np.load(self.npy_file_paths[idx])
        
        # 加载npz文件
        bev_data = np.load(self.npz_file_paths[idx])
        
        # 转为Tensor（根据需要，可能要做一些预处理）
        occ_data = torch.tensor(occ_data, dtype=torch.int32)
        bev_data = torch.tensor(bev_data['arr_0'], dtype=torch.int32)  # 这里假设 'arr_0' 是你需要的数据

        # 如果有transform操作，应用它
        if self.transform:
            occ_data = self.transform(occ_data)
            bev_data = self.transform(bev_data)

        return occ_data, bev_data

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
    # dist.init_process_group("nccl")
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    rank = 0  # Since we're not using distributed, set rank to 0
    # rank=args.local_rank
    device = rank % torch.cuda.device_count()
    seed = args.global_seed  # No need to adjust seed for distributed
    # seed = args.global_seed * dist.get_world_size() + rank

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}.")  # Removed world_size
    
    Tframe = 8
    use_bev_concat=True
    use_occ_meta=True
    use_vq=False
    use_inversion = args.use_inversion
    
    meta_num_mode = 3
    if meta_num_mode == 1:
        meta_num=4*Tframe + 12*(Tframe-1)
    elif meta_num_mode == 2:
        meta_num= 4 + 12*(Tframe-1)
    else:
        meta_num = 4
    in_ch=args.in_ch
    vis=args.vis
    ref_idx = 0
    scale_factor = 18

    # bev_ch_use=[0,2,8,9,10,11,12,13,14,15,16,17]
    bev_ch_use=[0,1,5,6,7,8,9,10] # openscene
    lambda_np = args.lambda_noise_prior
    
    DiT_cfg={"depth":12,"input_size":args.input_size, "patch_size": args.patch_size, "in_channels":in_ch, "hidden_size":512, "use_label":False, "use_bev_concat":use_bev_concat,"bev_in_ch":1,"bev_out_ch":1,"use_meta":use_occ_meta, "meta_num":meta_num,"direct_concat":True,"Tframe":Tframe,"dsr":args.dsr}
    # DiT_cfg={"depth":12, "in_channels":in_ch, "hidden_size":256, "use_label":False, "use_bev_concat":use_bev_concat,"bev_in_ch":1,"bev_out_ch":1,"use_meta":use_occ_meta, "meta_num":meta_num,"direct_concat":True,"Tframe":Tframe,"dsr":4}
    

    # DiT_cfg={"depth":6, "in_channels":in_ch, "hidden_size":256,"use_label":False, "use_bev_concat":use_bev_concat,"bev_in_ch":1,"bev_out_ch":1,"use_meta":use_occ_meta, "bev_dropout_prob":0.1,"meta_num":meta_num,"direct_concat":True,"Tframe":Tframe}
    Dit_model = DiT_2Frame(**DiT_cfg).to(device)
    # Dit_model = DiT_Occsora(**DiT_cfg).to(device)
    

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    Dit_model.load_state_dict(state_dict,strict=True)
    Dit_model.eval()  # important!

    ckpt_path_list = ckpt_path.split('/')
    log_path='/'.join(ckpt_path_list[:-2])
    vis_root='/'.join(ckpt_path_list[:-2]) +'/vis_continuous/video_mayavi/'
    zvis_root = '/'.join(ckpt_path_list[:-2]) +'/vis_continuous/zvis/'
    occ_save_root = '/code/code/Diff_occ/occ_gen/occ_gen/out/eval_dit_12hz_inversion'
    os.makedirs(zvis_root,exist_ok=True)
    os.makedirs(occ_save_root,exist_ok=True)
    # diffusion = create_diffusion(str(args.num_sampling_steps))  # default: 1000 steps, linear noise schedule
    diffusion = create_diffusion('ddim50')  # default: 1000 steps, linear noise schedule
    iter_num = ckpt_path_list[-1].split('.')[0]
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    # Dit_model = DDP(Dit_model.to(device), device_ids=[rank])
    # y_net =DDP(y_net.to(device), device_ids=[rank],find_unused_parameters=True)
    
    # logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
 
    
    
    # imageset = "data/nuscenes_infos_val_temporal_v3_scene.pkl"
    # file1_path = './step2/val/Zmid_4/'
    # file2_path = './step2/val/bevmap_4/'
    # gts_path = "data/nuscenes/gts"

   
    # dataset = CustomDataset_Tframe_continuous(imageset,gts_path,file2_path,bev_ch_use,meta_num=4,Tframe=Tframe)
    # dataset = CustomDataset_2frame_continuous(imageset,gts_path,file1_path,file2_path,bev_ch_use,meta_num=4,Tframe=Tframe)
    # imageset = "/gpfs/public-shared/fileset-groups/crosshair/zys/nuscenes/nuscenes_mmdet3d-12Hz/nuscenes_advanced_12Hz_infos_val.pkl"
    imageset = "/code/code/Diff_occ/occ_gen/occ_gen/data/nuscenes/nuscenes_interp_12Hz_infos_val.pkl"
    # occ_base_path = "/gpfs/public-shared/fileset-groups/crosshair/guojiazhe/occ_12hz/nuscene_quantized_200_200_16/quantized"
    # occ_base_path = "/gpfs/public-shared/fileset-groups/crosshair/guojiazhe/occ_12hz/nuscene_quantized_400_400_32/quantized"
    occ_base_path = "/data/longhun/3D/nuscenes/data/pyramid_occ/nuscene_quantized_400_400_32/quantized"

    # bev_path = "/gpfs/public-shared/fileset-groups/crosshair/guojiazhe/12hz_bevlayout_800_800"
    bev_path = "/code/code/Diff_occ/occ_gen/occ_gen/data/step2_12hz_200/val/bevmap_4"
    # dataset = CustomDataset_Tframe_12hz(imageset,occ_base_path,bev_path,bev_ch_use,meta_num=4,Tframe=32,use_clip=False)
    
    p_time = Tframe 
    
    # bz = int(args.global_batch_size // dist.get_world_size())
    # bz=1
    # if vis:
    #     bz = 1
    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=rank,
    #     shuffle=True,
    #     seed=args.global_seed
    # )
    # loader = DataLoader(
    #     dataset,
    #     batch_size=bz,
    #     shuffle=False,
    #     sampler=sampler,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True
    # )  


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


    # from model_vae.VAE.AE_eval import Autoencoder_2D
    # ae_eval = Autoencoder_2D(num_classes=18,expansion=4)

    # ae_ckpt_path = args.ae_ckpt
    # ae_ckpt = torch.load(ae_ckpt_path,map_location='cpu')
    # ae_eval.load_state_dict(ae_ckpt['state_dict'], strict=True)
    # ae_eval = ae_eval.to(device)
    # ae_eval.eval()

    #logger
    from mmengine.logging import MMLogger
    if not vis:
        log_file = os.path.join(log_path, f'Dit_eval_ddim_{use_inversion}_{iter_num}_{args.cfg_scale}_{lambda_np}.log')
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

    # Cal_fid_mmd = multi_step_fid_mmd()

    # Cal_TC = multi_step_TemporalConsistency("TC",times=p_time)
    # Cal_TC.reset()
    # Cal_TC_real = multi_step_TemporalConsistency("TC_real",times=p_time)
    # Cal_TC_real.reset()
    # ======openscene======

    openscene_occ = ["/code/code/Diff_occ/occ_gen/occ_gen/test_openscene/log-0002-scene-0012_1.npy"]
    openscene_bev = ["/code/code/Diff_occ/occ_gen/occ_gen/test_openscene/892ed622fe9e5fe9.npz"]
    dataset = OpenSceneDataset(openscene_occ,openscene_bev)
    # 使用DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for i_iter_val, (occ_ori,bev) in enumerate(tqdm(dataloader)):
            # 添加第二个维度为1， 也就是是1，1，200，200，16, 我需要重复8次， 也就是1，8，200，200，16
            occ_ori = occ_ori.unsqueeze(1)
            bev = bev.unsqueeze(1)
            occ_ori = occ_ori.repeat(1,8,1,1,1)
            bev = bev.repeat(1,8,1,1,1) # 1,8, 200, 200, 11
            bev = bev.permute(0,1,4,2,3) # 1, 8, 11, 200, 200

            occ_ori = occ_ori.to(device)
            y = bev.to(device)
            with torch.no_grad():
                z_ori=my_VQVAE.encode(occ_ori) *scale_factor # x: B C T H W
            B,C,T,H,W=z_ori.shape
            Bo,To,Ho,Wo,Do = occ_ori.shape  
            print(f"=> z_ori shape: {z_ori.shape}, occ_ori shape: {occ_ori.shape}, y shape: {y.shape}")
            z = torch.cat([z_ori, z_ori], 0)
            y_null = -torch.ones_like(y)
            y = torch.cat([y, y_null], 0)
            print(f"=> z shape: {z.shape}, y shape: {y.shape}")
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

            samples = diffusion.ddim_sample_loop(
                Dit_model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device)
            
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            
            rec_shape=[1,1,Ho,Wo,Do]
            result = my_VQVAE.generate(samples,rec_shape)
            logit=result["logits"]
            pred = logit.argmax(dim=-1) #  200, 200, 16

            print(pred.shape)


        # dataloader
    
    # # ======openscene======
    with torch.no_grad():
        for epoch in range(start_epoch, args.epochs):
            sampler.set_epoch(epoch)

            for i_iter_val, (occ_ori,y,occ_meta, pose_meta) in enumerate(tqdm(loader)):
                scene_idx = [2,7,14,21,26,36,40]
                if i_iter_val not in scene_idx and vis==True:
                    continue

                occ_ori = occ_ori.to(device)
                y = y.to(device)
                images=[]
                # n_time=z_ori.shape[1]-1
                
                with torch.no_grad():
                    z_ori=my_VQVAE.encode(occ_ori) *scale_factor # x: B C T H W
                # print(y_s.shape,occ_ori_s.shape,occ_meta_s.shape,pose_meta_s.shape)
                B,C,T,H,W=z_ori.shape
                Bo,To,Ho,Wo,Do = occ_ori.shape
                if use_inversion == False:
                    print("=>no inversion")
                    z_ori_s = z_ori[:,:,0].unsqueeze(2) #
                    z_ori_s = z_ori[:,:,ref_idx].unsqueeze(2) #
                    z_ori_s = z_ori_s.repeat(1, 1,Tframe, 1, 1)
                    z_ori_s = z_ori_s.to(device)
                    z = torch.randn((bz,in_ch, Tframe, H,W), device=device) + lambda_np* z_ori_s #z: [N,C,T,H,W]
                    p_name_save = "noise"
                else:
                    print("=>use inversion")
                    z = z_ori  
                    p_name_save = "inversion"

                z = torch.cat([z, z], 0)
                y_null = -torch.ones_like(y)
                y = torch.cat([y, y_null], 0)

                if use_occ_meta:
                    if meta_num_mode == 1:
                        new_meta = torch.cat((occ_meta.reshape(-1,4*Tframe),pose_meta.reshape(-1,12*(Tframe-1))),dim=1)
                    elif meta_num_mode == 2:
                        new_meta = torch.cat((occ_meta[:,ref_idx],pose_meta.reshape(-1,12*(Tframe-1))),dim=1)
                    else:
                        new_meta = occ_meta[:,ref_idx]
                    new_meta = new_meta.to(device)
                    new_meta = torch.cat([new_meta, new_meta], 0)

                    # new_meta_null = torch.zeros_like(new_meta)
                    # new_meta = torch.cat([new_meta, new_meta_null], 0)
                    model_kwargs = dict(y=y, meta=new_meta, cfg_scale=args.cfg_scale)

                else:
                    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                
                if use_inversion:
                    print("=>use inversion")
                    z_ori_inv = torch.cat([z_ori, z_ori], 0)  #ddim inversion
                    latent = z_ori_inv.clone().detach()
                    model_kwargs_inv = dict(y=y, meta=new_meta, cfg_scale=1)
                    for t in range(20,args.num_sampling_steps):
                        t_inv = torch.full((z.shape[0],), t).to(device)
                        # t_inv = torch.randint(0, 1, (z.shape[0],), device=device)
                        ddim_inversion_samples = diffusion.ddim_reverse_sample(Dit_model.module.forward_with_cfg,latent,t_inv,model_kwargs=model_kwargs_inv)
                        # c_latent,uc_latent = ddim_inversion_samples['sample'].chunk(2, dim=0)
                        # latent = torch.cat([c_latent, c_latent], 0)
                        latent = ddim_inversion_samples['sample']
                        vis_matrix(latent[0,1,0].cpu().numpy(),zvis_root+f"{i_iter_val}_01_{t}.png")
                    inversion_noise,_ = latent.chunk(2, dim=0)
                    z = torch.cat([inversion_noise, inversion_noise], 0)

                
                print(f"z type: {type(z)}, z shape: {z.shape}")
                samples = diffusion.ddim_sample_loop(
                    Dit_model.module.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device)
                

                # t_ = torch.tensor([999] * z.shape[0], device=device)
                # # t_ = t_.to(device)
                # one_step_sample = diffusion.ddim_sample(
                #     Dit_model.module.forward_with_cfg, z, t_, clip_denoised=False, model_kwargs=model_kwargs)
                # samples = one_step_sample["pred_xstart"]

                
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                
                # samples = samples.permute(0,2,1,3,4)
                # samples = samples.reshape(-1,4,50,50) 
                
                # shapes=[torch.Size([200, 200]), torch.Size([100, 100])]
                samples = samples / scale_factor

                rec_shape=[bz,Tframe,Ho,Wo,Do]
                if use_vq==False:
                    result=my_VQVAE.generate(samples,rec_shape)
                else:
                    result=my_VQVAE.generate_vq(samples,rec_shape)
                logit=result["logits"]
                pred = logit.argmax(dim=-1) #  200, 200, 16
                


                CalMeanIou_sem._after_step(pred, occ_ori)
                # save occ
                occ_ori_np = occ_ori.cpu().numpy()
                pred_np = pred.cpu().numpy()
                np.save(f"{occ_save_root}/occ_ori_{i_iter_val}.npy",occ_ori_np.astype(np.uint8))
                np.save(f"{occ_save_root}/pred_{i_iter_val}.npy",pred_np.astype(np.uint8))

                target_occs_iou = deepcopy(occ_ori)
                target_occs_iou[target_occs_iou != 0] = 1
                target_occs_iou[target_occs_iou == 0] = 0
                pred_iou = deepcopy(pred)
                pred_iou[pred_iou!=0] = 1
                pred_iou[pred_iou==0] = 0
                
                CalMeanIou_vox._after_step(pred_iou, target_occs_iou)
                print(f"=> done one iter")
                # occ_ori_noT = occ_ori.reshape(-1,200,200,16)
                # pred_noT = pred.reshape(-1,200,200,16)
                
                # ae_feature_ori = ae_eval.forward_eval(occ_ori) #B*T,2048
                # ae_feature_gen = ae_eval.forward_eval(pred)

                # Cal_fid_mmd._after_step(ae_feature_ori,ae_feature_gen)
                
                # Cal_TC._after_step(ae_feature_gen.reshape(bz,p_time,-1))

                # if i_iter_val > 25:
                #     fid,mmd = Cal_fid_mmd._after_epoch()
                #     break
                # fid,mmd = Cal_fid_mmd._after_epoch()
                # print(fid,mmd)
                # dist.barrier()
                
                # if vis:
                #     # bz = 1
                #     # dst_dir = os.path.join(vis_root,str(i_iter_val))
                #     os.makedirs(vis_root,exist_ok=True)
                #     os.makedirs(zvis_root,exist_ok=True)

                #     bevmaps = y[0].squeeze().cpu().numpy()
                #     pred = pred.squeeze().cpu().numpy()
                #     occ_ori = occ_ori.squeeze().cpu().numpy()
                    
                #     for i in range(Tframe):
                        
                #         pred_i=pred[i]
                        
                #         # save_path = vis_root
                #         # save_folder = os.path.join(save_path, '{}_assets'.format(token))
                #         # cat_save_file = os.path.join(save_path, '{}_cat_vis.png'.format(token))

                #         # draw(pred_i, 
                #         #     None, # predict_pts,
                #         #     [-50, -50, -5], 
                #         #     [0.25] * 3, 
                #         #     None, #  grid.squeeze(0).cpu().numpy(), 
                #         #     None,#  pt_label.squeeze(-1),
                #         #     vis_root,#recon_dir,
                #         #     None, # img_metas[0]['cam_positions'],
                #         #     None, # img_metas[0]['focal_positions'],
                #         #     timestamp=str(i_iter_val) + '_' + str(i),
                #         #     mode=0,
                #         #     sem=False)
                        
                #         # occ_i=occ_ori[i]
                #         # draw(occ_i, 
                #         #     None, # predict_pts,
                #         #     [-50, -50, -5], 
                #         #     [0.25] * 3, 
                #         #     None, #  grid.squeeze(0).cpu().numpy(), 
                #         #     None,#  pt_label.squeeze(-1),
                #         #     vis_root,#recon_dir,
                #         #     None, # img_metas[0]['cam_positions'],
                #         #     None, # img_metas[0]['focal_positions'],
                #         #     timestamp=str(i_iter_val) + '_' + str(i) +"_gt",
                #         #     mode=0,
                #         #     sem=False)
                        
                        
                #         fig=plt.figure(figsize=(15,6))
                #         pred_i=pred[i]
                #         fov_voxels,p_colors =draw_return(pred_i, 
                #             None, # predict_pts,
                #             [-50, -50, -5], 
                #             [0.25] * 3, 
                #             )
                #         ax = fig.add_subplot(int(f"131"),projection='3d') 
                #         ax.scatter(fov_voxels[:, 0],fov_voxels[:, 1],fov_voxels[:, 2],c=p_colors,s=1)
                #         ax.set_box_aspect([1,1,0.125])  
                #         ax.view_init(elev=90, azim=-90)
                #         ax.set_xlim(xmin = -50, xmax = 50)
                #         ax.set_ylim(ymin = -50, ymax = 50)
                #         ax.set_title(f"Occ Gen T = {i}")

                #         occ_i=occ_ori[i]
                #         fov_voxels,p_colors =draw_return(occ_i, 
                #             None, # predict_pts,
                #             [-50, -50, -5], 
                #             [0.25] * 3, 
                #             )
                #         ax_ori = fig.add_subplot(int(f"132"),projection='3d')
                #         ax_ori.scatter(fov_voxels[:, 0],fov_voxels[:, 1],fov_voxels[:, 2],c=p_colors,s=1)
                #         ax_ori.set_box_aspect([1,1,0.125])  
                #         ax_ori.view_init(elev=90, azim=-90)
                #         ax_ori.set_xlim(xmin = -50, xmax = 50)
                #         ax_ori.set_ylim(ymin = -50, ymax = 50)
                #         ax_ori.set_title(f"Occ Ori T = {i}")

                #         ax_bev = fig.add_subplot(int(f"133"))
                #         ax_bev.imshow(bevmaps[i],origin="lower",cmap='coolwarm')
                #         ax_bev.margins(x=0.8, y=0.8)
                #         ax_bev.set_title("BEV Layout")
                #         fig.suptitle(f" cfg = {args.cfg_scale}")
                #         fig.tight_layout()
                #         fig_np=figure_to_array(fig)
                #         images.append(fig_np)
                #         plt.close(fig)

                #         z_ori_vis = z_ori.squeeze().cpu().numpy()
                #         print(np.var(z_ori_vis))
                #         # vis_matrix(z_ori_vis[0,i],zvis_root+f"{i_iter_val}_{i}_0.png")
                        

                    
                #     imageio.mimsave(vis_root + f"{p_name_save}_{iter_num}_{i_iter_val}_{lambda_np}_{args.cfg_scale}.mp4",images,'mp4',fps=4)

            val_miou, avg_val_miou = CalMeanIou_sem._after_epoch() 
            val_iou, avg_val_iou = CalMeanIou_vox._after_epoch() 
            
            logger.info(f'Avg mIoU: %.2f%%' % (avg_val_miou))
            logger.info(f'Avg IoU: %.2f%%' % (avg_val_iou))

            # val_TC = Cal_TC._after_epoch()
            # logger.info(f'Avg TC: %.4f' % (val_TC))
            # if rank==0:
            #     fid,mmd = Cal_fid_mmd._after_epoch()
            #     logger.info(f'FID: %.4f' % (fid))
            #     logger.info(f'MMD: %.6f' % (mmd))


    


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 128], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=6)
    parser.add_argument("--in_ch", type=int, default=4)
    # parser.add_argument("--log-every", type=int, default=500)
    # parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--ckpt", type=str, default="/code/code/Diff_occ/occ_gen/occ_gen/results/2024-10-19/checkpoints/0320000.pt")
    parser.add_argument('--vis', action='store_true', default=False)
    # parser.add_argument("--local_rank", type=int,default=0)
    parser.add_argument("--input_size", type=int,default=100)
    parser.add_argument("--patch_size", type=int,default=4)
    parser.add_argument("--dsr", type=int,default=2)
    parser.add_argument("--vae_ckpt", type=str, default="/code/code/Diff_occ/occ_gen/occ_gen/ckpt/VAE/epoch_296.pth")
    parser.add_argument("--vae_config", type=str, default="/code/code/Diff_occ/occ_gen/occ_gen/ckpt/VAE/train_vae_4_DwT_L_me.py")
    parser.add_argument("--ae_ckpt", type=str, default="out/AE_eval/epoch_196.pth")
    parser.add_argument("--lambda_noise_prior", type=float, default=0)
    # parser.add_argument('--vae-config', default='./config/train_vqvae_4.py')
    parser.add_argument("--num-sampling-steps", type=int, default=50) #1000
    parser.add_argument("--use_inversion", action='store_true', default=False) #1000
    args = parser.parse_args()
    main(args)
    
    #python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 eval_continuous_mVAE_fid.py --ckpt="/data/WM/OccWorld/results/2024-09-08 22:27:31/checkpoints/0080000.pt"
    #python -m torch.distributed.launch --nnodes=1 --nproc_per_node=1 eval_continuous_mVAE_fid.py --vis --ckpt=

    