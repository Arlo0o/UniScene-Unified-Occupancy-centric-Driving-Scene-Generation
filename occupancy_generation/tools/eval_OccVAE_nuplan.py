import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist
from copy import deepcopy
import sys
sys.path.append('.')

import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim import build_optim_wrapper
from mmengine.logging import MMLogger
from mmengine.utils import symlink
from mmengine.registry import MODELS
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler
from utils.load_save_util import revise_ckpt, revise_ckpt_1
import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level='WARNING')

def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
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
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('genocc', log_file=log_file)
    MMLogger._instance_dict['genocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model_vae
    from dataset import get_dataloader, get_nuScenes_label_name, get_val_dataloader
    from loss import OPENOCC_LOSS
    from utils.metric_util import MeanIoU, multi_step_MeanIou
    from utils.freeze_model import freeze_model

    my_model = MODELS.build(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if cfg.get('freeze_dict', False):
        logger.info(f'Freezing model according to freeze_dict:{cfg.freeze_dict}')
        freeze_model(my_model, cfg.freeze_dict)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params after freezed: {n_parameters}')
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

    val_dataset_loader = get_val_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        iter_resume=args.iter_resume)

    # get optimizer, loss, scheduler
    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    max_num_epochs = cfg.max_epochs


    # resume and load
    epoch = 0
    global_iter = 0
    last_iter = 0
    best_val_iou = [0]*cfg.get('return_len_', 10)
    best_val_miou = [0]*cfg.get('return_len_', 10)

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        # optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        global_iter = ckpt['global_iter']
        last_iter = ckpt['last_iter'] if 'last_iter' in ckpt else 0
        if 'best_val_iou' in ckpt:
            best_val_iou = ckpt['best_val_iou']
        if 'best_val_miou' in ckpt:
            best_val_miou = ckpt['best_val_miou']
            
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        if cfg.get('revise_ckpt', False):
            if cfg.revise_ckpt == 1:
                print('revise_ckpt')
                print(raw_model.load_state_dict(revise_ckpt(state_dict), strict=False))
            elif cfg.revise_ckpt == 2:
                print('revise_ckpt_1')
                print(raw_model.load_state_dict(revise_ckpt_1(state_dict), strict=False))
            elif cfg.revise_ckpt == 3:
                print('revise_ckpt_2')
                print(raw_model.vae.load_state_dict(state_dict, strict=False))
        else:
            print(raw_model.load_state_dict(state_dict, strict=False))
        
    # training
    print_freq = cfg.print_freq
    first_run = True
    grad_norm = 0
    
    label_name = get_nuScenes_label_name(cfg.label_mapping)
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [label_name[l] for l in unique_label]
    CalMeanIou_sem = multi_step_MeanIou(unique_label, cfg.get('ignore_label', -100), unique_label_str, 'sem', times=cfg.get('return_len_', 10))
    CalMeanIou_vox = multi_step_MeanIou([1], cfg.get('ignore_label', -100), ['occupied'], 'vox', times=cfg.get('return_len_', 10))
    logger.info('compiling model')
    my_model = torch.compile(my_model)
    logger.info('done compile model')
    best_plan_loss = 100000
    # while epoch < max_num_epochs:
    # with torch.no_grad():
    my_model.eval()
    os.environ['eval'] = 'true'
    val_loss_list = []
    CalMeanIou_sem.reset()
    CalMeanIou_vox.reset()
    plan_loss = 0
    save_dir = os.path.join(args.work_dir, 'save_occ')
    os.makedirs(save_dir, exist_ok=True)
    val_data_len = len(val_dataset_loader)
    logger.info(f'Val data length is {val_data_len}')
    
    with torch.no_grad():
        for i_iter_val, (input_occs, target_occs, metas) in enumerate(val_dataset_loader):
            if i_iter_val>10:
                break
            input_occs = input_occs.cuda()
            target_occs = target_occs.cuda()
            data_time_e = time.time()
            
            result_dict = my_model(x=input_occs, metas=metas)

            loss_input = {
                'inputs': input_occs,
                'target_occs': target_occs,
                # 'metas': metas
            }
            for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                loss_input.update({
                    loss_input_key: result_dict[loss_input_val]
                })
            loss, loss_dict = loss_func(loss_input)
            plan_loss += loss_dict.get('PlanRegLoss', 0)
            plan_loss += loss_dict.get('PlanRegLossLidar', 0)
            if result_dict.get('target_occs', None) is not None:
                target_occs = result_dict['target_occs']
            target_occs_iou = deepcopy(target_occs)
            print(result_dict['sem_pred'].shape)
            for i in range(20):
                if (result_dict['sem_pred'] == i).any():
                    print(f"label {i} in result_dict['sem_pred'] {bool((result_dict['sem_pred'] == i).any())} sum {int((result_dict['sem_pred'] == i).sum())}")

            target_occs_iou[target_occs_iou != 0] = 1
            target_occs_iou[target_occs_iou == 0] = 0

            result_dict['iou_pred_fix'] = deepcopy(result_dict['sem_pred'])
            result_dict['iou_pred_fix'][result_dict['iou_pred_fix'] != 0] = 1
            result_dict['iou_pred_fix'][result_dict['iou_pred_fix'] == 0] = 0


            
            CalMeanIou_sem._after_step(result_dict['sem_pred'], target_occs)
            CalMeanIou_vox._after_step(result_dict['iou_pred_fix'], target_occs_iou)
            # save and visualize
            # if i_iter_val % 100 == 0:
            np.save(os.path.join(save_dir, f'val_{i_iter_val}_reconsem.npy'), result_dict['sem_pred'].detach().cpu().numpy().astype(np.uint8))
            # np.save(os.path.join(save_dir, f'val_{i_iter_val}_reconocc.npy'), result_dict['iou_pred_fix'].detach().cpu().numpy().astype(np.uint8))
            np.save(os.path.join(save_dir, f'val_{i_iter_val}_input.npy'), input_occs.detach().cpu().numpy().astype(np.uint8))
            # np.save(os.path.join(save_dir, f'val_{i_iter_val}_output.npy'), target_occs.detach().cpu().numpy().astype(np.uint8))
            # np.save(os.path.join(save_dir, f'val_{i_iter_val}_targetocc.npy'), target_occs_iou.detach().cpu().numpy().astype(np.uint8))

            val_loss_list.append(loss.detach().cpu().numpy())
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Epoch %d Iter %5d / %d : Loss: %.3f (%.3f)'%(
                    epoch, i_iter_val, val_data_len, loss.item(), np.mean(val_loss_list)))
                detailed_loss = []
                for loss_name, loss_value in loss_dict.items():
                    detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                detailed_loss = ', '.join(detailed_loss)
                logger.info(detailed_loss)
    val_miou, _ = CalMeanIou_sem._after_epoch()
    val_iou, _ = CalMeanIou_vox._after_epoch()
    
    # del target_occs, input_occs
    plan_loss = plan_loss/len(val_dataset_loader)
    if plan_loss < best_plan_loss:
        best_plan_loss = plan_loss
    logger.info(f'PlanRegLoss is {plan_loss} while the best plan loss is {best_plan_loss}')
    #logger.info(f'PlanRegLoss is {plan_loss/len(val_dataset_loader)}')
    best_val_iou = [max(best_val_iou[i], val_iou[i]) for i in range(len(best_val_iou))]
    best_val_miou = [max(best_val_miou[i], val_miou[i]) for i in range(len(best_val_miou))]
    #logger.info(f'PlanRegLoss is {plan_loss/len(val_dataset_loader)}')
    logger.info(f'Current val iou is {val_iou} while the best val iou is {best_val_iou}')
    logger.info(f'Current val miou is {val_miou} while the best val miou is {best_val_miou}')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--vae-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./ckpt/VAE')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--iter-resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    # ngpus=1
    args.gpus = ngpus
    # print(args)
    
    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
