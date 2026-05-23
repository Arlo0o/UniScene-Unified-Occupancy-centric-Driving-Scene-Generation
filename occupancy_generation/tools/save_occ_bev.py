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

# from  map_visualizer import visualize_map
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.distributed as dist

def pass_print(*args, **kwargs):
    pass

def main(local_rank,args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # load config
    cfg = Config.fromfile(args.vae_config)
    cfg.work_dir = args.work_dir
    
    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", cfg.get("port", 29500))
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1

    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.vae_config)))
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
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
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
        dist=distributed)

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
    # recon_dir = os.path.join(args.work_dir, args.dir_name+f'{cfg.get("data_type", "gts")}_autoreg', str(epoch))
    # os.makedirs(recon_dir, exist_ok=True)
    # dataset = cfg.val_dataset_config['type']
    # recon_dir = os.path.join(recon_dir, dataset)

    save_root = "./step2"

    save_path=f"{save_root}/val"
    # bev_save_path=f"{save_path}/bevmap_4"
    Zmid_save_path=f"{save_path}/Zmid_4"

    if local_rank==0:
        # os.makedirs(bev_save_path, exist_ok=True)
        os.makedirs(Zmid_save_path, exist_ok=True)

    with torch.no_grad():
        for i_iter_val, (input_occs, target_occs, metas) in enumerate( tqdm(val_dataset_loader)) :
        # for i_iter_val, (input_occs, target_occs, metas, bevmaps) in enumerate( tqdm(val_dataset_loader)) :
            # if i_iter_val not in args.scene_idx:
            #     continue
            # if i_iter_val > max(args.scene_idx):
            #     break
            '''
            if i_iter_val < start_frame:
                continue'''

            scene_token_s=metas[0]["scene_token"][0]
            scene_token_e=metas[0]["scene_token"][-1]
            if os.path.exists(os.path.join(Zmid_save_path,f"{scene_token_s}.npz")) and os.path.exists(os.path.join(Zmid_save_path,f"{scene_token_e}.npz")):
                continue
            # bev_save_filepath_s=os.path.join(bev_save_path,f"{scene_token_s}.npz")
            # bev_save_filepath_e=os.path.join(bev_save_path,f"{scene_token_e}.npz")

            # if os.path.exists(bev_save_filepath_s) and os.path.exists(bev_save_filepath_e):
                # continue

            input_occs = input_occs.cuda()
            if args.onlyvqvae==True:
                result = my_model(x=input_occs, metas=metas)
                # input_occs = input_occs[0]
            else:    
                result=my_model(x=input_occs, metas=metas)
                # result = my_model.forward_autoreg_with_pose(
                #         x=input_occs, metas=metas, 
                #         start_frame=cfg.get('start_frame', 0),
                #         mid_frame=cfg.get('mid_frame', 5),
                #         end_frame=cfg.get('end_frame', 11))
                # input_occs = result['input_occs']
            
            # logits = result['logits']
            # n_frames = logits.shape[1]
            # print(n_frames,input_occs.shape[1])

            scene_len=len(metas[0]["scene_token"])
            middd=np.squeeze(result['middd'].cpu().numpy())
            middd = middd.transpose(1,0,2,3)
            # bevmap=np.squeeze(bevmaps.cpu().numpy())
            # print(scene_len)
            # print(bevmap.dtype)
            # print(middd.dtype)

            #save code
            for i in range(scene_len):
                scene_token=metas[0]["scene_token"][i]
                # bev_save_filepath=os.path.join(bev_save_path,f"{scene_token}")
                Zmid_save_filepath=os.path.join(Zmid_save_path,f"{scene_token}")
                # np.savez(bev_save_filepath, bevmap[i]) 
                np.savez(Zmid_save_filepath, middd=middd[i]) 
        
        save_path=f"{save_root}/train"
        # bev_save_path=f"{save_path}/bevmap_4"
        Zmid_save_path=f"{save_path}/Zmid_4"
        for i_iter_train, (input_occs, target_occs, metas) in enumerate( tqdm(train_dataset_loader)) :
        # for i_iter_val, (input_occs, target_occs, metas, bevmaps) in enumerate( tqdm(val_dataset_loader)) :
            # if i_iter_val not in args.scene_idx:
            #     continue
            # if i_iter_val > max(args.scene_idx):
            #     break
            '''
            if i_iter_val < start_frame:
                continue'''

            scene_token_s=metas[0]["scene_token"][0]
            scene_token_e=metas[0]["scene_token"][-1]
            # bev_save_filepath_s=os.path.join(bev_save_path,f"{scene_token_s}.npz")
            # bev_save_filepath_e=os.path.join(bev_save_path,f"{scene_token_e}.npz")

            # if os.path.exists(bev_save_filepath_s) and os.path.exists(bev_save_filepath_e):
                # continue

            input_occs = input_occs.cuda()
            if args.onlyvqvae==True:
                result = my_model.encoder(x=input_occs, metas=metas)
                # input_occs = input_occs[0]
            else:    
                result=my_model(x=input_occs, metas=metas)
            

            scene_len=len(metas[0]["scene_token"])
            middd=np.squeeze(result['middd'].cpu().numpy())
            middd = middd.transpose(1,0,2,3)
            # bevmap=np.squeeze(bevmaps.cpu().numpy())
            # print(scene_len)
            # print(bevmap.dtype)
            # print(middd.dtype)

            #save code
            for i in range(scene_len):
                scene_token=metas[0]["scene_token"][i]
                # bev_save_filepath=os.path.join(bev_save_path,f"{scene_token}")
                Zmid_save_filepath=os.path.join(Zmid_save_path,f"{scene_token}")
                # np.savez(bev_save_filepath, bevmap[i]) 
                np.savez(Zmid_save_filepath, middd=middd[i]) 
            

if __name__ == '__main__':
    # Eval settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--vae-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--dir-name', type=str, default='vis')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-trials', type=int, default=10)
    parser.add_argument('--frame-idx', nargs='+', type=int, default=[0, 10])
    parser.add_argument('--scene-idx', nargs='+', type=int, default=[2,7,16,18,19,87,89,96,101])
    parser.add_argument('--onlyvqvae', action='store_true') # add
    args = parser.parse_args()
    
    # ngpus = torch.cuda.device_count()
    ngpus=1
    args.gpus = ngpus
    # print(args)
    
    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)

    # python save_occ_bev_ddp.py --vae-config config/save_step2.py  --work-dir out/vqvae_4 --onlyvqvae

