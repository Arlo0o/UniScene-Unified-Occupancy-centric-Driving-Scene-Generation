from typing import Optional
import pickle
import time

import os
import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

def get_max_cuda_memory(device: Optional[torch.device] = None) -> int:
    """Returns the maximum GPU memory occupied by tensors in megabytes (MB) for
    a given device. By default, this returns the peak allocated memory since
    the beginning of this program.

    Args:
        device (torch.device, optional): selected device. Returns
            statistic for the current device, given by
            :func:`~torch.cuda.current_device`, if ``device`` is None.
            Defaults to None.

    Returns:
        int: The maximum GPU memory occupied by tensors in megabytes
        for a given device.
    """
    mem = torch.cuda.max_memory_allocated(device=device)
    mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                          dtype=torch.int,
                          device=device)
    torch.cuda.reset_peak_memory_stats()
    return int(mem_mb.item())

def batch_index_to_list(item, batch_size):
    ret = []
    for bs in range(batch_size):
        batch_mask = item[:, 0] == bs
        ret.append(item[batch_mask][:, 1:])
    return ret

def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None, result_tag=""):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / (f'final_result_{result_tag}' if result_tag != "" else 'final_result') / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    class_names = dataset.class_names

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()
        memory_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()
    print(model)
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False) and i > 10:#i > 100:
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'
            memory_meter.update(get_max_cuda_memory())
            disp_dict['memory'] = f'{memory_meter.val:.2f}({memory_meter.avg:.2f})'

        #if i == 500:
        if i == 100:
            logger.info(f'Avg Inference Time (ms): {infer_time_meter.avg:.2f}')
            logger.info(f'Avg Memory (MB): {memory_meter.avg:.2f}')
            logger.info(f'Avg Memory (GB): {memory_meter.avg/1024:.2f}')
            exit()

        #torch.cuda.empty_cache()
        #gpu_info = os.popen('nvidia-smi').read()
        #print(gpu_info)
        

        # pc_out = pred_dicts['pc_out']
        # if 'gt_pts' in pred_dicts:
        #     pc_gt = pred_dicts['gt_pts']
        # else:
        #     pc_gt = batch_dict['points']
        #     pc_gt = batch_index_to_list(pc_gt, batch_size=batch_dict['batch_size'])
        # chamfer_distance = dataset.update_chamfer_distance(pc_out, pc_gt, frame_ids=batch_dict['frame_id'], save_path=final_output_dir if args.save_to_file else None, save_type=args.save_type if hasattr(args, 'save_type') else 'npy')
        

        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()
    
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    ret_dict = {}

    if not dist_test:
        result_str, result_dict = dataset.evaluation()
    else:
        rank, world_size = common_utils.get_dist_info()
        result_str, result_dict = dataset.evaluation(dist_test=True, world_size=world_size, rank=rank, tmpdir=result_dir / 'tmpdir')

    if cfg.LOCAL_RANK != 0:
        return {}
    
    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
