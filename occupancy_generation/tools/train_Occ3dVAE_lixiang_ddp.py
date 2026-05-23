import time, argparse, os.path as osp, os
import datetime
import torch, numpy as np
torch.cuda.empty_cache()
import torch.distributed as tdist
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy
import sys
import os
sys.path.append('.')  # 将父目录添加到 sys.path 中

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
from datetime import timedelta
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level='WARNING')

from torch.utils.tensorboard import SummaryWriter

def pass_print(*args, **kwargs):
    pass

def create_logger(logging_dir,rank):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0:  # real logger
        # 创建logger实例
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(f"{logging_dir}/log.log")
        file_handler.setLevel(logging.DEBUG)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                   datefmt='%Y-%m-%d %H:%M:%S')
        
        # 设置格式化器
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info("Logger initialized")
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    
    return logger



def main(args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # setup ddp
    
    if not tdist.is_initialized():
        tdist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.cuda.set_device(local_rank)
    print(f"=> Process {local_rank} using GPU: {torch.cuda.current_device()}")
    print(f"=> ddp initialized, world_size: {tdist.get_world_size()}")

    # load config
    cfg = Config.fromfile(args.vae_config)
    cfg.work_dir = args.work_dir

    # setup work dir
    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        ct_str=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") 
        experiment_dir = f'{args.work_dir}/{ct_str}'
        checkpoint_dir = f'{experiment_dir}/checkpoints'
        save_occ_dir = f'{experiment_dir}/save_occ'
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(save_occ_dir, exist_ok=True)
        cfg.dump(osp.join(experiment_dir, osp.basename(args.vae_config)))
        logger = create_logger(experiment_dir, local_rank)
        logger.info(f'Experiment dir: {experiment_dir}')
        logger.info(f'Config:\n{cfg.pretty_text}')
        # log_file = osp.join(experiment_dir, f'{ct_str}.log')
    else:
        logger = create_logger(None, local_rank)

    # build model
    import model_vae
    from dataset import get_dataloader, get_nuScenes_label_name
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
    # if distributed:

    my_model = DDP(my_model.cuda(), device_ids=[local_rank])
    raw_model = my_model.module
    logger.info('done ddp model')

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=True,
        rank=local_rank,
        )
    print(f"=> dataloader done")

    # get optimizer, loss, scheduler
    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    max_num_epochs = cfg.max_epochs
    if cfg.get('multisteplr', False):
        scheduler = MultiStepLRScheduler(
            optimizer,
            **cfg.multisteplr_config)
    else:
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=len(train_dataset_loader) * max_num_epochs,
            lr_min=1e-6,
            warmup_t=cfg.get('warmup_iters', 500),
            warmup_lr_init=1e-6,
            t_in_epochs=False)

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
        # epoch = ckpt['epoch']
        # global_iter = ckpt['global_iter']
        # last_iter = ckpt['last_iter'] if 'last_iter' in ckpt else 0
        if 'best_val_iou' in ckpt:
            best_val_iou = ckpt['best_val_iou']
        if 'best_val_miou' in ckpt:
            best_val_miou = ckpt['best_val_miou']
            
        if hasattr(train_dataset_loader.sampler, 'set_last_iter'):
            train_dataset_loader.sampler.set_last_iter(last_iter)
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
    # logger.info('compiling model')
    # my_model = torch.compile(my_model)
    # logger.info('done compile model')
    best_plan_loss = 100000

    # 初始化 Tensorboard
    if local_rank == 0:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        tb_log_dir = osp.join(experiment_dir, f'tb_logs_{timestamp}')
        tb_writer = SummaryWriter(log_dir=tb_log_dir)

    while epoch < max_num_epochs:
    # with torch.no_grad():
        my_model.train()
        os.environ['eval'] = 'false'
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_list = []
        time.sleep(1)
        data_time_s = time.time()
        time_s = time.time()
        print(f"=> training start {epoch} epoch, time: {time_s}")
        for i_iter, (input_occs, target_occs, metas) in enumerate(train_dataset_loader):

            if first_run:
                i_iter = i_iter + last_iter
            
            input_occs = input_occs.cuda()
            # target_occs = target_occs.cuda()
            data_time_e = time.time()

            result_dict = my_model(x=input_occs, metas=metas)

            loss_input = {
                'inputs': input_occs,
                'target_occs': target_occs,
                # 'target_occs': input_occs
                # 'metas': metas
            }
            
            for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                loss_input.update({
                    loss_input_key: result_dict[loss_input_val]})
            loss, loss_dict = loss_func(loss_input)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
            optimizer.step()

            loss_list.append(loss.detach().cpu().item())
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and local_rank == 0:
                if i_iter%1000==0:
                        np.save(os.path.join(save_occ_dir, f'{epoch}_{i_iter}_train_input_occs.npy'), input_occs.cpu().numpy().astype(np.int8))
                        np.save(os.path.join(save_occ_dir, f'{epoch}_{i_iter}_train_pred_occs.npy'), result_dict['sem_pred'].cpu().numpy().astype(np.int8))
                lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.3f, lr: %.7f, time: %.3f (%.3f)'%(
                    epoch, i_iter, len(train_dataset_loader), 
                    loss.item(), np.mean(loss_list), grad_norm, lr,
                    time_e - time_s, data_time_e - data_time_s))
                detailed_loss = []
                for loss_name, loss_value in loss_dict.items():
                    detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                detailed_loss = ', '.join(detailed_loss)
                logger.info(detailed_loss)
                
                # 记录训练 loss
                tb_writer.add_scalar('Train/Loss', loss.item(), global_iter)
                for loss_name, loss_value in loss_dict.items():
                    tb_writer.add_scalar(f'Train/{loss_name}', loss_value, global_iter)
                
                loss_list = []
            data_time_s = time.time()
            time_s = time.time()

        
        # save checkpoint
        # if local_rank == 0 and epoch % cfg.get('save_every_epochs', 1) == 0:
        ckpt_epoch =2
        if local_rank == 0 and epoch % ckpt_epoch == 0:
            dict_to_save = {
                'state_dict': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                # 'epoch': epoch + 1,
                # 'global_iter': global_iter,
            }
            save_file_name = os.path.join(os.path.abspath(checkpoint_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            logger.info(f'save checkpoint to {save_file_name}')
            dst_file = osp.join(checkpoint_dir, 'latest.pth')
            symlink(save_file_name, dst_file)

        epoch += 1
        first_run = False
        
        # eval
        # if epoch % cfg.get('eval_every_epochs', 1) != 0:
        val_epoch=200
        if epoch % val_epoch !=0:
            continue
        my_model.eval()
        os.environ['eval'] = 'true'
        val_loss_list = []
        CalMeanIou_sem.reset()
        CalMeanIou_vox.reset()
        plan_loss = 0
        
        with torch.no_grad():
            for i_iter_val, (input_occs, target_occs, metas) in enumerate(val_dataset_loader):
                
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
                target_occs_iou[target_occs_iou != 0] = 1
                target_occs_iou[target_occs_iou == 0] = 0
                
                CalMeanIou_sem._after_step(result_dict['sem_pred'], target_occs)
                CalMeanIou_vox._after_step(result_dict['iou_pred'], target_occs_iou)
                val_loss_list.append(loss.detach().cpu().numpy())
                if i_iter_val % print_freq == 0 and local_rank == 0:
                    if i_iter_val%1000==0:
                        np.save(os.path.join(save_occ_dir, f'{epoch}_{i_iter_val}_val_input_occs.npy'), input_occs.cpu().numpy().astype(np.int8))
                        np.save(os.path.join(save_occ_dir, f'{epoch}_{i_iter_val}_val_pred_occs.npy'), result_dict['sem_pred'].cpu().numpy().astype(np.int8))
                    logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.3f, lr: %.7f, time: %.3f (%.3f)'%(
                    epoch, i_iter, len(train_dataset_loader), 
                    loss.item(), np.mean(loss_list), grad_norm, lr,
                    time_e - time_s, data_time_e - data_time_s))
                    logger.info('[EVAL] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f)'%(
                        epoch, i_iter_val,len(val_dataset_loader), loss.item(), np.mean(val_loss_list)))
                    detailed_loss = []
                    for loss_name, loss_value in loss_dict.items():
                        detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                    detailed_loss = ', '.join(detailed_loss)
                    logger.info(detailed_loss)

                    # 记录验证 loss
                    tb_writer.add_scalar('Val/Loss', loss.item(), global_iter)
                    for loss_name, loss_value in loss_dict.items():
                        tb_writer.add_scalar(f'Val/{loss_name}', loss_value, global_iter)
        val_miou, _ = CalMeanIou_sem._after_epoch()
        val_iou, _ = CalMeanIou_vox._after_epoch()
        
        # 记录 iou 和 miou
        if local_rank == 0:
            for i, (iou_val, miou_val) in enumerate(zip(val_iou, val_miou)):
                tb_writer.add_scalar(f'Val/IoU_{i}', iou_val, global_iter)
                tb_writer.add_scalar(f'Val/mIoU_{i}', miou_val, global_iter)

        del target_occs, input_occs
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
    parser.add_argument('--vae-config', default='config/train_3dvae_nuplan_200.py')
    parser.add_argument('--work-dir', type=str, default='./out/3dVAE_5')
    parser.add_argument('--resume-from', type=str, default='')
    # parser.add_argument('--iter-resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    
    
    # 获取GPU数量
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(f"=> args.gpus: {args}")

    # if ngpus > 1:
    #     torch.multiprocessing.spawn(main, args=(args,), nprocs=ngpus)
    # else:
    main(args)
    
