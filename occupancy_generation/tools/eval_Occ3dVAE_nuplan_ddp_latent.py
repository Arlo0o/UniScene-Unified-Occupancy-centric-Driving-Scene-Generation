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
from tqdm import tqdm

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
        # ct_str=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") 
        # experiment_dir = f'{args.work_dir}/{ct_str}'
        experiment_dir = f'{args.work_dir}'
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
    from dataset import get_val_dataloader, get_nuScenes_label_name
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

    # 若处于debug，打乱验证集便于快速观察
    if args.debug:
        cfg.val_loader['shuffle'] = True

    val_dataset_loader = get_val_dataloader(
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
    # 打印频率可被命令行覆写
    print_freq = cfg.print_freq
    if getattr(args, 'print_freq', -1) and args.print_freq > 0:
        print_freq = args.print_freq
    first_run = True
    grad_norm = 0
    
    label_name = get_nuScenes_label_name(cfg.label_mapping)
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [label_name[l] for l in unique_label]
    CalMeanIou_sem = multi_step_MeanIou(unique_label, cfg.get('ignore_label', -100), unique_label_str, 'sem', times=cfg.get('return_len_', 5))
    CalMeanIou_vox = multi_step_MeanIou([1], cfg.get('ignore_label', -100), ['occupied'], 'vox', times=cfg.get('return_len_', 5))
    # logger.info('compiling model')
    # my_model = torch.compile(my_model)
    # logger.info('done compile model')
    best_plan_loss = 100000

    # 初始化 Tensorboard
    if local_rank == 0:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        tb_log_dir = osp.join(experiment_dir, f'tb_logs_{timestamp}')
        tb_writer = SummaryWriter(log_dir=tb_log_dir)


    my_model.eval()
    os.environ['eval'] = 'true'
    val_loss_list = []
    CalMeanIou_sem.reset()
    CalMeanIou_vox.reset()
    plan_loss = 0
    
    # 用于累计 latent 的全局统计
    lat_sum = None
    lat_sqsum = None
    lat_count = None

    max_iters = getattr(args, 'max_iters', -1)
    with torch.no_grad():
        for i_iter_val, (input_occs, target_occs, metas) in enumerate(tqdm(val_dataset_loader)):
            # 限制最大迭代数（优先使用命令行参数），debug模式下默认更小
            if max_iters is not None and max_iters > 0 and i_iter_val >= max_iters:
                break
            if args.debug and i_iter_val > 100:
                break

            input_occs = input_occs.cuda()
            target_occs = target_occs.cuda()
            data_time_e = time.time()
            
            result_dict = my_model(x=input_occs, metas=metas)
            # 统计 latent 空间尺度（支持 VAE 路径）
            z_latent = result_dict.get('latent', None)
            z_mu = result_dict.get('z_mu', None)
            z_sigma = result_dict.get('z_sigma', None)
            if z_latent is not None:
                # 按样本维归约，得到每个 batch 的 sum/sqsum/count，并累加到局部统计量
                with torch.no_grad():
                    if lat_sum is None:
                        lat_sum = torch.zeros(1, device=z_latent.device, dtype=torch.float64)
                        lat_sqsum = torch.zeros(1, device=z_latent.device, dtype=torch.float64)
                        lat_count = torch.zeros(1, device=z_latent.device, dtype=torch.float64)
                    batch_sum = z_latent.to(torch.float64).sum()
                    batch_sqsum = (z_latent.to(torch.float64) ** 2).sum()
                    batch_count = torch.tensor([z_latent.numel()], device=z_latent.device, dtype=torch.float64)
                    lat_sum += batch_sum
                    lat_sqsum += batch_sqsum
                    lat_count += batch_count

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
            if i_iter_val % 1 == 0 and local_rank == 0 and getattr(args, 'save_occ', False):
                np.save(os.path.join(save_occ_dir, f'{metas[0]["scene_token"][0]}_gt.npy'), input_occs.cpu().numpy().astype(np.int8))
                np.save(os.path.join(save_occ_dir, f'{metas[0]["scene_token"][0]}_pred.npy'), result_dict['sem_pred'].cpu().numpy().astype(np.int8)[0])

                detailed_loss = []
                for loss_name, loss_value in loss_dict.items():
                    detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                detailed_loss = ', '.join(detailed_loss)
                logger.info(detailed_loss)

                # 记录验证 loss
                tb_writer.add_scalar('Val/Loss', loss.item(), global_iter)
                for loss_name, loss_value in loss_dict.items():
                    tb_writer.add_scalar(f'Val/{loss_name}', loss_value, global_iter)
                # 可视化一小段 latent 的直方图（避免过大开销）
                if z_latent is not None:
                    z_sample = z_latent.flatten()
                    cap = getattr(args, 'latent_hist_max', 100000)
                    if z_sample.numel() > cap:
                        z_sample = z_sample[:cap]
                    tb_writer.add_histogram('Latent/z', z_sample.detach().cpu().numpy(), global_iter, bins='auto')
    # 分布式聚合 latent 统计并计算全局均值/标准差
    if lat_sum is not None:
        tdist.all_reduce(lat_sum, op=tdist.ReduceOp.SUM)
        tdist.all_reduce(lat_sqsum, op=tdist.ReduceOp.SUM)
        tdist.all_reduce(lat_count, op=tdist.ReduceOp.SUM)
        lat_mean = (lat_sum / lat_count).item()
        lat_var = (lat_sqsum / lat_count - (lat_mean ** 2))
        lat_std = torch.sqrt(torch.clamp(lat_var, min=0.0)).item()
        if local_rank == 0:
            # 写入 TensorBoard 与保存到磁盘
            tb_writer.add_scalar('Latent/mean', lat_mean, global_iter)
            tb_writer.add_scalar('Latent/std', lat_std, global_iter)
            with open(os.path.join(experiment_dir, 'latent_stats.txt'), 'w') as f:
                f.write(f'mean: {lat_mean}\nstd: {lat_std}\ncount: {int(lat_count.item())}\n')
            logger.info(f'Latent stats -> mean: {lat_mean:.6f}, std: {lat_std:.6f}, count: {int(lat_count.item())}')

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
    # 新增常用调试与行为控制参数
    parser.add_argument('--debug', action='store_true', default=False, help='启用debug，打乱验证集并限制迭代数')
    parser.add_argument('--save-occ', action='store_true', default=False, help='保存输入与预测的occ npy')
    parser.add_argument('--max-iters', dest='max_iters', type=int, default=-1, help='最多迭代多少batch，-1表示不限')
    parser.add_argument('--print-freq', dest='print_freq', type=int, default=-1, help='打印与保存频率，覆盖cfg.print_freq')
    parser.add_argument('--latent-hist-max', dest='latent_hist_max', type=int, default=100000, help='latent直方图的最大采样点数')
    args = parser.parse_args()
    
    
    # 获取GPU数量
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(f"=> args.gpus: {args}")

    # if ngpus > 1:
    #     torch.multiprocessing.spawn(main, args=(args,), nprocs=ngpus)
    # else:
    main(args)
    
