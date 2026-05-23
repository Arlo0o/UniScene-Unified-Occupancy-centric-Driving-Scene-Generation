from curses import meta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

import time, argparse, os.path as osp, os
import torch, numpy as np

import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmengine.registry import MODELS
import sys
sys.path.append('.')  # 将父目录添加到 sys.path 中

from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def pass_print(*args, **kwargs):
    pass

                

    

def main(args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # load config
    cfg = Config.fromfile(args.vae_config)
    cfg.work_dir = args.work_dir
    
    os.makedirs(args.work_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{cfg.get("data_type", "gts")}_visualize_autoreg_{timestamp}.log')
    logger = MMLogger('genocc', log_file=log_file)
    MMLogger._instance_dict['genocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model_vae
    my_model = MODELS.build(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    my_model = my_model.cuda()
    raw_model = my_model
    logger.info('done ddp model')
    from dataset import get_dataloader
    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
        rank=0)
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    epoch = 'last'
    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        epoch = ckpt['epoch']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    # eval
    my_model.eval()
    os.environ['eval'] = 'true'
    recon_dir = os.path.join(args.work_dir, args.dir_name+f'{cfg.get("data_type", "gts")}_autoreg', str(epoch))
    os.makedirs(recon_dir, exist_ok=True)
    dataset = cfg.val_dataset_config['type']
    recon_dir = os.path.join(recon_dir, dataset)

    save_root = "data/step2"

    
    with torch.no_grad():

        save_path=f"{save_root}/train"
        bev_save_path=f"{save_path}/bevmap_4"

        for i_iter_val, (input_occs, target_occs, bevmaps, metas) in enumerate(tqdm(train_dataset_loader)):
            print(input_occs.shape)
            print(target_occs.shape)
            print(bevmaps.shape)
            print(metas)
            break

        #     os.makedirs(bev_save_path, exist_ok=True)

        #     scene_len=len(metas[0]["scene_token"])
        #     bevmap=np.squeeze(bevmaps.cpu().numpy())

        #     #save code
        #     for i in range(scene_len):
        #         scene_token=metas[0]["scene_token"][i]
        #         bev_save_filepath=os.path.join(bev_save_path,f"{scene_token}")
        #         print(f"=> bevmap[i] shape = {bevmap[i].shape}")
        #         np.savez(bev_save_filepath, bevmap[i]) 

        # save_path=f"{save_root}/val"
        # bev_save_path=f"{save_path}/bevmap_4"

        # for i_iter_val, (input_occs, target_occs, metas, bevmaps) in enumerate( tqdm(val_dataset_loader)):

        #     os.makedirs(bev_save_path, exist_ok=True)

        #     scene_len=len(metas[0]["scene_token"])
        #     bevmap=np.squeeze(bevmaps.cpu().numpy())
        #     # only save the last frame
        #     scene_token=metas[0]["scene_token"][-1]
        #     temp_token=metas[0]["scene_token"][-2]
        #     bev_save_filepath=os.path.join(bev_save_path,f"{scene_token}")
        #     np.savez(bev_save_filepath, bevmap[-1]) 
        #     print(f"save bevmap to {bev_save_filepath}, and the last last frame token is {temp_token}, bevmap shape: {bevmap.shape}")    

            # #save code
            # for i in range(scene_len):
            #     scene_token=metas[0]["scene_token"][i]
            #     bev_save_filepath=os.path.join(bev_save_path,f"{scene_token}")
            #     np.savez(bev_save_filepath, bevmap[i]) 

            

            
            
            

if __name__ == '__main__':
    # Eval settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--vae-config', default='config/save_step2_me.py')
    parser.add_argument('--work-dir', type=str, default='./ckpt/VAE/')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--dir-name', type=str, default='vis')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-trials', type=int, default=10)
    args = parser.parse_args()
    
    ngpus = 1
    args.gpus = ngpus
    main(args)

