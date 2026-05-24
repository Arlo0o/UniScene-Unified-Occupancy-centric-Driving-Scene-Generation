import os
import pickle
import time
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from uniscenev2_lidar.models import load_data_to_gpu
from uniscenev2_lidar.utils import common_utils
from uniscenev2_lidar.datasets.nuscenes_occ.eval_utils.jsd import JensenShannonDivergence
from uniscenev2_lidar.datasets.nuscenes_occ.eval_utils.mmd_gpu import MaximumMeanDiscrepancy
from uniscenev2_lidar.datasets.nuscenes_occ.eval_utils.voxelize import _voxelize_gpu

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
    # class_names = dataset.class_names

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

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

    # JSD and MMD metric
    JSD_SHAPE = [1, 100, 100]
    metric_jsd = JensenShannonDivergence(JSD_SHAPE).cuda()
    metric_mmd = MaximumMeanDiscrepancy().cuda()

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

        if getattr(args, 'infer_time', False) and i > 100:
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'


        pc_out = pred_dicts['pc_out']
        if 'gt_pts' in pred_dicts:
            pc_gt = pred_dicts['gt_pts']
        else:
            pc_gt = batch_dict['points']
            pc_gt = batch_index_to_list(pc_gt, batch_size=batch_dict['batch_size'])
        chamfer_distance = dataset.update_chamfer_distance(pc_out, 
                                                            pc_gt, 
                                                            frame_ids=batch_dict['frame_id'], 
                                                            save_path=final_output_dir if args.save_to_file else None, 
                                                            save_type=args.save_type if hasattr(args, 'save_type') else 'npy',
                                                            lidar_wise=True
                                                            )

        ########################################### JSD & MMD ###############################################
        batch_pts1, batch_pts2 = pc_gt, pc_out
        coors = []
        for bs_idx, coor in enumerate(batch_pts1):
            coor_pad = F.pad(coor[:, :3], (1, 0), mode='constant', value=bs_idx)
            coors.append(coor_pad)
        batch_pts1 = torch.cat(coors, dim=0)

        coors = []
        for bs_idx, coor in enumerate(batch_pts2):
            coor_pad = F.pad(coor[:, :3], (1, 0), mode='constant', value=bs_idx)
            coors.append(coor_pad)
        batch_pts2 = torch.cat(coors, dim=0)

        # bxyz
        pts1, pts2 = batch_pts1, batch_pts2
        dis1 = torch.linalg.norm(pts1[:, 1:4], dim=-1)
        dis_mask1 = (dis1 > 3.0) & (dis1 < 50.0)
        pts1 = pts1[dis_mask1]
        
        dis2 = torch.linalg.norm(pts2[:, 1:4], dim=-1)
        dis_mask2 = (dis2 > 3.0) & (dis2 < 50.0)
        pts2 = pts2[dis_mask2]

        EVAL_SPATIAL_RANGE = [-51.2, 51.2, -51.2, 51.2, -5.0, 3.0]
        EVAL_VOXEL_SIZE = [0.15625, 0.15625, 0.2]
        batch_voxelized1 = _voxelize_gpu(pts1, EVAL_SPATIAL_RANGE, EVAL_VOXEL_SIZE)
        batch_voxelized2 = _voxelize_gpu(pts2, EVAL_SPATIAL_RANGE, EVAL_VOXEL_SIZE, rotations=0, flip_vert=False)

        for voxelized1, voxelized2 in zip(batch_voxelized1, batch_voxelized2):
            data_map = {
                "lidar": voxelized1.unsqueeze(0),
                "sample": voxelized2.unsqueeze(0)
            }

            metric_jsd.update(data_map)
            metric_mmd.update(data_map)

        #####################################################################################################

        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()
    
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    ret_dict = {}

    # gather jsd & mmd results
    final_jsd = metric_jsd.compute()
    final_mmd = metric_mmd.compute()

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

    mmd_science = "{:e}".format(final_mmd)
    result_str = f'\nJSD: {final_jsd}\nMMD: {mmd_science}'
    ret_dict.update({'jsd': final_jsd, 'mmd': final_mmd})
    logger.info(result_str)

    # 废弃：先保存再eval，现在直接合并到上面了，一遍出结果一边eval
    # logger.info('**************** Start computing JSD & MMD...*****************')
    # result = subprocess.run(['python', 'uniscenev2_lidar/datasets/nuscenes_occ/eval_utils/eval_mmd_jsd_gpu_batch.py', str(os.path.join(final_output_dir, 'gt')), str(os.path.join(final_output_dir, 'pred'))], capture_output=True, text=True)
    # logger.info(result.stdout)


    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
