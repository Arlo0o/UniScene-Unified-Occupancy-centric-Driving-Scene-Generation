import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import torch
from eval_utils import eval_utils_occ2lidar as eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

CFG = "tools/cfgs/nuscenes_occ_models/occ2lidar_sparseunet_renderv2_s_priorsampler_p54_r200_intenw10_raydropw02_prerays.yaml"
CKPT = "checkpoints/occ2lidar.pth"
BATCH_SIZE = 2  # reduce if out of memory
WORK_DIR = f"exps/uniscenes-occ2lidar"


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("-c", "--cfg_file", type=str, default=CFG, help="specify the config for training")
    parser.add_argument("-w", "--work_dir", type=str, default=WORK_DIR, required=False)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, required=False, help="batch size for training")
    parser.add_argument("--workers", type=int, default=4, help="number of workers for dataloader")
    parser.add_argument("--extra_tag", type=str, default="default", help="extra tag for this experiment")
    parser.add_argument("--ckpt", type=str, default=CKPT, help="checkpoint to start from")
    parser.add_argument("--pretrained_model", type=str, default=None, help="pretrained_model")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"], default="none")
    parser.add_argument("--tcp_port", type=int, default=18888, help="tcp port for distrbuted training")
    parser.add_argument("--local_rank", type=int, default=None, help="local rank for distributed training")
    parser.add_argument(
        "--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER, help="set extra config keys if needed"
    )

    parser.add_argument("--max_waiting_mins", type=int, default=30, help="max waiting minutes")
    parser.add_argument("--start_epoch", type=int, default=0, help="")
    parser.add_argument("--eval_tag", type=str, default="default", help="eval tag for this experiment")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="specify a ckpt directory to be evaluated if needed")
    parser.add_argument("--save_to_file", action="store_true", default=False, help="")
    parser.add_argument("--result_tag", type=str, default="")
    parser.add_argument("--infer_time", action="store_true", default=False, help="calculate inference latency")

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, dist_test=False, result_tag=""):
    # load checkpoint
    if args.ckpt is not None:
        model.load_params_from_file(
            filename=args.ckpt, logger=logger, to_cpu=dist_test, pre_trained_path=args.pretrained_model
        )
    else:
        logger.warning("!!! no checkpoint loaded !!!")
    model.cuda()

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, logger, dist_test=dist_test, result_dir=eval_output_dir, result_tag=result_tag
    )


def main():
    args, cfg = parse_config()

    if args.infer_time:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if args.launcher == "none":
        dist_test = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, "init_dist_%s" % args.launcher)(
            args.tcp_port, args.local_rank, backend="nccl"
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, "Batch size should match the number of gpus"
        args.batch_size = args.batch_size // total_gpus

    # output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / "eval"
    eval_output_dir = eval_output_dir / cfg.DATA_CONFIG.DATA_SPLIT["test"]

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ("log_eval_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info("**********************Start logging**********************")
    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ.keys() else "ALL"
    logger.info("CUDA_VISIBLE_DEVICES=%s" % gpu_list)

    if dist_test:
        logger.info("total_batch_size: %d" % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    log_config_to_file(cfg, logger=logger)

    args.ckpt_dir if args.ckpt_dir is not None else output_dir / "ckpt"

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    with torch.no_grad():
        eval_single_ckpt(
            model, test_loader, args, eval_output_dir, logger, dist_test=dist_test, result_tag=args.result_tag
        )


if __name__ == "__main__":
    main()
