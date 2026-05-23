
import torch
from torch.utils.data import DataLoader, Dataset

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

from diffusion.models import DiT_models, DiT_multiframe
from diffusion import create_diffusion
from tqdm import tqdm
# from diffusers.models import AutoencoderKL

from diffusion.bev_cod import BEV_concat_net
import datetime
import shutil

from mmengine import Config
from mmengine.registry import MODELS

from visualize.visualize_demo_plt import draw,get_grid_coords
from dataset.dataload_util import CustomDataset_nBEV1_time
import matplotlib.pyplot as plt
import imageio
import matplotlib.animation as animation

def draw_return(
    voxels,          # semantic occupancy predictions
    pred_pts,        # lidarseg predictions
    vox_origin,
    voxel_size=0.2,  # voxel size in the real world
    grid=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    save_dir=None,
    cam_positions=None,
    focal_positions=None,
    timestamp=None,
    mode=0,
    sem=False,
):
    w, h, z = voxels.shape

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    if mode == 0:
        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    elif mode == 1:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        pred_pts = pred_pts[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, pred_pts.reshape(-1)]).T
    elif mode == 2:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        gt_label = pt_label[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, gt_label.reshape(-1)]).T
    else:
        raise NotImplementedError

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 17)
    ]
    print(len(fov_voxels))
    
    # import pdb; pdb.set_trace()
    # fig = plt.figure(figsize=(6,6))
    # ax = fig.add_subplot(111,projection='3d')

    voxel_size = sum(voxel_size) / 3
    colors = np.array(
        [
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 150, 245, 255],       # car                  blue
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255,   0,   0, 255],       # pedestrian           red
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red  # add
            [139, 137, 137, 255],
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
            # [  0, 255, 127, 255],       # ego car              dark cyan
            # [255,  99,  71, 255],       # ego car
            # [  0, 191, 255, 255]        # ego car
        ]
    ).astype(np.uint8)
    # print(fov_voxels[:, 3])
    p_colors=colors[fov_voxels[:, 3].astype(np.uint8)-1]/255
    
    return fov_voxels,p_colors

    # ax.scatter(fov_voxels[:, 0],fov_voxels[:, 1],fov_voxels[:, 2],c=p_colors,s=1)
    # # ax.set_box_aspect([10,10,1]) 
    # ax.set_box_aspect([1,1,0.125])  
    # ax.view_init(elev=90, azim=-90)

    # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.savefig(os.path.join(save_dir, f'vis_{timestamp}.png'),dpi=300)
   
def figure_to_array(myfig):
    """
    """
    myfig.canvas.draw()
    X = np.array(myfig.canvas.renderer.buffer_rgba())
    return X


#################################################################################
#                                  visualize by time                            #
#################################################################################

def main(args):
     # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # bev_ch_use=[0,2,5,6,8,9,10,11,12,13,14,15,16,17]
    bev_ch_use=[0,2,8,9,10,11,12,13,14,15,16,17]
    
    use_occ_meta=True
    meta_num = 4
    DiT_cfg={"depth":6, "in_channels":4, "hidden_size":512,"use_label":False, "use_bev_concat":True,"bev_in_ch":1,"bev_out_ch":1,"use_meta":use_occ_meta,"meta_num": meta_num,"direct_concat":True}
    # DiT_cfg={"depth":28, "in_channels":4, "hidden_size":512, "use_label":False, "use_bev_concat":True,"bev_in_ch":1,"bev_out_ch":1,"direct_concat":True}
    # DiT_cfg={"depth":28, "in_channels":4, "hidden_size":512, "use_label":False, "use_bev_concat":True,"bev_in_ch":1,"bev_out_ch":1,"use_meta":use_occ_meta,"direct_concat":True}
    Dit_model = DiT_multiframe(**DiT_cfg).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    Dit_model.load_state_dict(state_dict,strict=True)
    Dit_model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    

    ckpt_path_list = ckpt_path.split('/')
    log_path='/'.join(ckpt_path_list[:-2])
    vis_root='/'.join(ckpt_path_list[:-2]) +'/vis/'
    
    imageset = "data/nuscenes_infos_val_temporal_v3_scene.pkl"
    file1_path = './step2/val/Zmid_4/'
    file2_path = './step2/val/bevmap_4/'
    gts_path = "data/nuscenes/gts"
    bev_ch_use=[0,2,8,9,10,11,12,13,14,15,16,17]

    Tdataset = CustomDataset_nBEV1_time(imageset,gts_path, file1_path,file2_path,bev_ch_use,meta_num)

    loader = DataLoader(
        Tdataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )  

    cfg1 = Config.fromfile(args.vae_config)
    import model
    my_VQVAE = MODELS.build(cfg1.model)
    my_VQVAE = my_VQVAE.to(device)

    # resume_from="/data/WM/OccWorld/out/vqvae/epoch_200.pth"
    resume_from="/data/WM/OccWorld/out/vqvae_4/epoch_196.pth"
    vqvae_ckpt = torch.load(resume_from,map_location='cpu')
    my_VQVAE.load_state_dict(vqvae_ckpt['state_dict'], strict=True)

    my_VQVAE = my_VQVAE.to(device)
    # # eval
    my_VQVAE.eval()

    # from mmengine.logging import MMLogger
    # # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log_file = os.path.join(log_path, 'vis.log')
    # logger = MMLogger('genocc', log_file=log_file)
    # MMLogger._instance_dict['genocc'] = logger
    # logger.info(f'Config:')



    # label_name = get_nuScenes_label_name(cfg1.label_mapping)
    # unique_label = np.asarray(cfg1.unique_label)
    # unique_label_str = [label_name[l] for l in unique_label]
    # CalMeanIou_sem = multi_step_MeanIou(unique_label, cfg1.get('ignore_label', -100), unique_label_str, 'sem', times=bz)
    # CalMeanIou_sem.reset()
    use_meta_null= args.occ_meta_null
    use_design_c = False

    with torch.no_grad():
        for i_iter_val, (z_ori, y ,occ_ori,occ_meta) in enumerate(loader):
            if i_iter_val not in args.scene_idx:
                continue
            if i_iter_val > max(args.scene_idx):
                break

            occ_ori = occ_ori.to(device)
            y = y.to(device)
            # print(occ_ori.shape)
            # print(y.shape)

            bz=z_ori.shape[1] # time
            # z = torch.randn((bz,4, 50,50), device=device)
            # z = torch.cat([z, z], 0)

            z = torch.randn((1,4, 50,50), device=device)
            z = torch.repeat_interleave(z, repeats=2*bz, dim=0)
            # z = torch.cat([z]*2*bz, 0)

            y = y.squeeze(0)
            # print(y.shape)
            occ_ori = occ_ori.squeeze(0)
            occ_meta_ori = occ_meta.squeeze(0)

            y_null = -torch.ones_like(y)
            y = torch.cat([y, y_null], 0)
            # y = torch.tensor(y.clone().detach(), dtype=torch.float, device=device) 

            if use_occ_meta:
                # occ_meta = occ_meta.to(device)
                # occ_meta = torch.cat([occ_meta, occ_meta], 0)

                if meta_num==1:
                    # c_score = 1
                    # occ_meta = c_score*torch.ones([y.shape[0],1])
                    # cs_str = c_score
                    occ_meta = torch.cat([occ_meta_ori, occ_meta_ori], 0)
                    cs_str = "ori"
                elif meta_num==4:

                    occ_meta_null=torch.zeros_like(occ_meta_ori)

                    
                    c_score = [1,0.0,0.0,0.5]
                    cs_tensor = torch.tensor(c_score)
                    cs_str = f"all:{c_score[0]}-t:{c_score[1]}-m:{c_score[2]}-v:{c_score[3]}"

                    if use_design_c:
                        occ_meta_1 = cs_tensor.repeat(y_null.shape[0], 1)
                    else:
                        cs_str = 'ori'
                        occ_meta_1 = occ_meta_ori

                    if use_meta_null: 
                        occ_meta_2 = occ_meta_null
                        # occ_meta = torch.cat([occ_meta_1, occ_meta_null], 0)
                    else:
                        cs_str +="_wonull"
                        occ_meta_2 = occ_meta_1
                        # occ_meta = torch.cat([occ_meta_1, occ_meta_ori], 0)   
                    occ_meta = torch.cat([occ_meta_1, occ_meta_2], 0)
                # print(occ_meta.shape)

                occ_meta = occ_meta.to(device)
                model_kwargs = dict(y=y, meta=occ_meta, cfg_scale=args.cfg_scale)
            else:
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            # samples = diffusion.p_sample_loop(
            #     Dit_model.module.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)

            samples = diffusion.ddim_sample_loop(
                Dit_model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)

            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            # print(samples.shape)

            
            shapes=[torch.Size([200, 200]), torch.Size([100, 100])]
            rec_shape=[bz,1,200,200,16]
            result=my_VQVAE.generate(samples,shapes,rec_shape)
            # result=my_VQVAE.generate_vq(samples_array,shapes,rec_shape)
            logit=result["logits"]
            pred = logit.argmax(dim=-1).squeeze().cpu().numpy()  #  200, 200, 16
            occ_ori = occ_ori.squeeze().cpu().numpy()
            print(pred.shape)

            dst_dir = os.path.join(vis_root,str(i_iter_val))
            os.makedirs(dst_dir,exist_ok=True)

            # fig = plt.figure(figsize=(20,9))
            # n_col=5
            bevmaps = y.squeeze().cpu().numpy()
            images=[]
            for i in range(bz):
                pred_i=pred[i]
                fov_voxels,p_colors =draw_return(pred_i, 
                    None, # predict_pts,
                    [-40, -40, -1], 
                    [0.4] * 3, 
                    None, #  grid.squeeze(0).cpu().numpy(), 
                    None,#  pt_label.squeeze(-1),
                    dst_dir,#recon_dir,
                    None, # img_metas[0]['cam_positions'],
                    None, # img_metas[0]['focal_positions'],
                    timestamp=str(i),
                    mode=0,
                    sem=False)
                fig=plt.figure(figsize=(15,6))
                ax = fig.add_subplot(131,projection='3d')
                ax.scatter(fov_voxels[:, 0],fov_voxels[:, 1],fov_voxels[:, 2],c=p_colors,s=1)
                ax.set_box_aspect([1,1,0.125])  
                ax.view_init(elev=90, azim=-90)
                # ax.set_title(f"T = {i}")
                ax.set_xlim(xmin = -40, xmax = 40)
                ax.set_ylim(ymin = -40, ymax = 40)
                ax.set_title(f"Occ Gen {cs_str}")


                occ_i=occ_ori[i]
                fov_voxels,p_colors =draw_return(occ_i, 
                    None, # predict_pts,
                    [-40, -40, -1], 
                    [0.4] * 3, 
                    None, #  grid.squeeze(0).cpu().numpy(), 
                    None,#  pt_label.squeeze(-1),
                    dst_dir,#recon_dir,
                    None, # img_metas[0]['cam_positions'],
                    None, # img_metas[0]['focal_positions'],
                    timestamp=str(i),
                    mode=0,
                    sem=False)
                ax_ori = fig.add_subplot(132,projection='3d')
                ax_ori.scatter(fov_voxels[:, 0],fov_voxels[:, 1],fov_voxels[:, 2],c=p_colors,s=1)
                ax_ori.set_box_aspect([1,1,0.125])  
                ax_ori.view_init(elev=90, azim=-90)
                ax_ori.set_xlim(xmin = -40, xmax = 40)
                ax_ori.set_ylim(ymin = -40, ymax = 40)
                ax_ori.set_title(f"Occ Ori {occ_meta_ori[i]}")

                ax_bev = fig.add_subplot(133)
                ax_bev.imshow(bevmaps[i],origin="lower",cmap='coolwarm')
                ax_bev.margins(x=0.8, y=0.8)
                ax_bev.set_title("BEV Layout")
                fig.suptitle(f"T = {i} cfg = {args.cfg_scale}")
                fig.tight_layout()
                


                fig_np=figure_to_array(fig)
                images.append(fig_np)
                plt.close(fig)
            # imageio.mimsave(os.path.join(dst_dir, f'vis_{i_iter_val}.mp4'),images,'mp4',fps = 4)
            # occ_meta = c_score*torch.ones([y.shape[0],1]).to(device)

            imageio.mimsave(os.path.join(dst_dir, f'vis_{i_iter_val}_{cs_str}_{args.cfg_scale}.mp4'),images,'mp4',fps = 4)

            # imageio.mimsave(os.path.join(dst_dir, 'vis.gif'),images,duration=250,loop=0)

            #     ax = fig.add_subplot(2,n_col,i+1,projection='3d')
            #     ax.scatter(fov_voxels[:, 0],fov_voxels[:, 1],fov_voxels[:, 2],c=p_colors,s=1)
            #     ax.set_box_aspect([1,1,0.125])  
            #     ax.view_init(elev=90, azim=-90)
            #     ax.set_title(f"T = {i}")
            # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            # plt.savefig(os.path.join(dst_dir, f'all_zs6.png'),dpi=300)
            

            
            # break
            # print(pred.shape)
            
        #     occ_ori = torch.unsqueeze(occ_ori, dim=0)
        #     pred = pred.permute(1,0,2,3,4)
        #     CalMeanIou_sem._after_step(pred, occ_ori)
        # val_miou, avg_val_miou = CalMeanIou_sem._after_epoch()
        # logger.info(f'Avg mIoU: %.2f%%' % (avg_val_miou))

    # upload oss    
    # oss = "aws --endpoint-url=http://oss.i.brainpp.cn s3"
    # t_time = ckpt_path_list[-3].replace(" ","-")
    # print(t_time)
    # print(vis_root)
    # os.system(f"{oss} sync \"{vis_root}\" s3://guojiazhe/Diffusion/{t_time}/vis/")
    

    # logger.info("Done!")
    # cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data-path", type=str, default="path")
    # parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 128], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=32)#71 32
    parser.add_argument("--local_rank", type=int,default=0)
    parser.add_argument("--occ_meta_null", action='store_true', default=False)
    parser.add_argument('--vae-config', default='./config/train_vqvae_4.py')
    parser.add_argument("--num-sampling-steps", type=int, default=50) #1000
    parser.add_argument('--scene-idx', nargs='+', type=int, default=[7,16,18,19,89,101])
    args = parser.parse_args()
    main(args)
    # python vis_Dit_time.py --ckpt="/data/WM/OccWorld/results/2024-07-29 18:31:01/checkpoints/0230000.pt"