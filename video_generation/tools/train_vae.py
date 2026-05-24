from IPython import embed
import os
import sys
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录的路径
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(root_dir)
import random
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pprint import pformat
from einops import rearrange, repeat
import torch
import torch.distributed as dist
import wandb
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm
from torch.nn import functional as F
from uniscenev2_video.acceleration.checkpoint import set_grad_checkpoint
from uniscenev2_video.acceleration.parallel_states import get_data_parallel_group, get_sequence_parallel_group
from uniscenev2_video.datasets.dataloader import prepare_dataloader
# from torchvision.io import write_video
import logging
from mmcv.parallel import DataContainer
from uniscenev2_video.registry import DATASETS, MODELS, SCHEDULERS, build_module
from uniscenev2_video.utils.ckpt_utils import load, model_gathering, model_sharding, record_model_param_shape, save, prepare_ckpt, RandomStateManager
from uniscenev2_video.utils.config_utils import define_experiment_workspace, parse_configs, save_training_config
from uniscenev2_video.utils.lr_scheduler import LinearWarmupLR, MultiStepWithLinearWarmupLR
from uniscenev2_video.utils.misc import (
    Timer,
    all_reduce_mean,
    create_logger,
    create_tensorboard_writer,
    format_numel_str,
    get_model_numel,
    requires_grad,
    to_torch_dtype,
    is_main_process,
    move_to,
    collate_bboxes_to_maxlen,
    add_box_latent,
)
import colossalai
import math
import imageio

from uniscenev2_video.utils.train_utils import MaskGenerator, create_colossalai_plugin, update_ema, default, sp_vae, run_validation, save_validation_vae, run_validation_vae
from uniscenev2_video.models.vae.losses import *



def write_video(save_path,img_list,fps=10):
    img_numpy_list = img_list.numpy()
    videoWriter = imageio.get_writer(save_path, fps=fps)
    for idx in range(len(img_numpy_list)):
        videoWriter.append_data(img_numpy_list[idx])
    videoWriter.close()


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=True)
    record_time = cfg.get("record_time", False)
    num_camera = cfg.get("num_camera", 1)
    use_extr_loss = cfg.get("use_extr_loss", False)
    f_frames = cfg.get("num_frames", 44)
    uc_keys = cfg.get("uc_keys", [])
    wo_clip = cfg.get("wo_clip", False)
    diff_type = cfg.get("diff_type", 'unet')
    ae_stride_t = cfg.get("n_cond_frames", 4)
    n_cond_frames = cfg.get("n_cond_frames", 4)
    fps = cfg.get("fps", 10)
    scale_factor = cfg.get("scale_factor", 1)
    overlap = cfg.get("overlap", 0)
    is_video_mask = cfg.get("is_video_mask", True)
    round_num = cfg.get("round_num", 1)
    time_downsample = cfg.get("time_downsample",4)
    verbose_mode = cfg.get("verbose_mode", False)
    # freeze_base_model = cfg.get("freeze_base_model",False)


    if cfg.get("mask_ratios", None) is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    
    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # == colossalai init distributed training ==
    # NOTE: A very large timeout is set to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(cfg.get("seed", 1024))
    torch.cuda.manual_seed_all(cfg.get("seed", 1024))
    coordinator = DistCoordinator()
    coordinator._local_rank = int(coordinator._local_rank)
    device = get_current_device()

    # == init exp_dir ==
    exp_name, exp_dir = define_experiment_workspace(cfg)
    coordinator.block_all()

    if coordinator.is_master():
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)
    coordinator.block_all()

    # == init logger, tensorboard & wandb ==
    logger = create_logger(exp_dir)
    logger.info("Experiment directory created at %s", exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    logger.info(f"ColossalAI version: {colossalai.__version__}")

    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(project="World Model", name=exp_name, config=cfg.to_dict(), dir="./outputs/wandb")
    

    # == init ColossalAI booster ==
    plugin = create_colossalai_plugin(
        plugin=cfg.get("plugin", "zero2"),
        dtype=cfg_dtype,
        grad_clip=cfg.get("grad_clip", 0),
        sp_size=cfg.get("sp_size", 1),
        reduce_bucket_size_in_m=cfg.get("reduce_bucket_size_in_m", 20),
        overlap_allgather=cfg.get("overlap_allgather", False),
        verbose=verbose_mode,
    )

    booster = Booster(plugin=plugin)
    torch.set_num_threads(1)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )

    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )

    num_steps_per_epoch = len(dataloader)
    val_dataset = build_module(cfg.val_dataset, DATASETS)
    validation_cfg = cfg.get("val",None)
    if validation_cfg is not None:
        if len(cfg.val.validation_index) < get_data_parallel_group().size():
            if isinstance(cfg.val.validation_index[0], int):
                cfg.val.validation_index += random.sample(
                    list(set(range(len(val_dataset))) - set(cfg.val.validation_index)),
                    min(get_data_parallel_group().size(), 32) - len(cfg.val.validation_index),
                )
                # for larger than 32, add them one-by-one.
                if get_data_parallel_group().size() > 32:
                    while len(cfg.val.validation_index) < get_data_parallel_group().size():
                        cfg.val.validation_index += random.sample(
                            list(set(range(len(val_dataset)))
                                 - set(cfg.val.validation_index)), 1,
                        )
            else:
                while len(cfg.val.validation_index) < get_data_parallel_group().size():
                    new_key = val_dataset.rand_another_key()
                    if new_key not in cfg.val.validation_index:
                        cfg.val.validation_index.append(new_key)

        val_dataset = torch.utils.data.Subset(val_dataset, cfg.val.validation_index)
    else:
        raise NotImplementedError()

    logger.info("Val Dataset contains %s samples.", len(val_dataset))
    dataloader_args['shuffle'] = False
    dataloader_args['dataset'] = val_dataset
    dataloader_args['batch_size'] = cfg.val.get("batch_size", 1)
    dataloader_args['num_workers'] = cfg.val.get("num_workers", 2)
    val_dataloader, val_sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )


    
    def collate_data_container_fn(batch, *, collate_fn_map=None):
        return batch
    # add datacontainer handler
    torch.utils.data._utils.collate.default_collate_fn_map.update({
        DataContainer: collate_data_container_fn
    })



    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    # text_encoder = build_module(cfg.get("text_encoder", None), MODELS, device=device, dtype=dtype)
    # if text_encoder is not None:
    #     text_encoder_output_dim = text_encoder.output_dim
    #     text_encoder_model_max_length = text_encoder.model_max_length
    # else:
    #     text_encoder_output_dim = cfg.get("text_encoder_output_dim", 4096)
    #     text_encoder_model_max_length = cfg.get("text_encoder_model_max_length", 300)

    model = (build_module(cfg.get("vae", None), MODELS).to(device, dtype).train() ) 
    
    
    VAE_Losses = VAELoss(
        logvar_init=cfg.get("logvar_init", 0.0),
        sigloss_weight=cfg.get("sigloss_weight" ),
        kl_loss_weight=cfg.get("kl_loss_weight" ),
        nll_weights = cfg.get("nll_weights" ),
        device=device,
        dtype=dtype, )
    # if vae is not None:
    #     vae = vae.to(device, dtype).eval()
    # vae_out_channels = cfg.get("vae_out_channels", 4)



    # # == build diffusion model ==
    # model = (
    #     build_module(
    #         cfg.model,
    #         MODELS
    #     )
    #     .to(device, dtype)
    #     .train()  ## 
    # )
    # model.prepare_text_embedding(text_encoder)

    # partial load pretrain (e.g., image pretrain)
    if cfg.get("partial_load", None) and not cfg.get("load", None):
        load_dir = cfg.partial_load
        if os.path.isdir(load_dir):
            from glob import glob
            weight = {}
            for path in glob(os.path.join(load_dir, "model/pytorch_model-*")):
                weight.update(torch.load(path, map_location="cpu"))
        else:
            weight = torch.load(load_dir, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(weight, strict=False)
        logger.info(f"[partial load] Missing keys: {missing_keys}")
        logger.info(f"[partial load] Unexpected keys: {unexpected_keys}")
        del weight, missing_keys, unexpected_keys
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "[Diffusion] Trainable model params: %s, Fix: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel - model_numel_trainable),
        format_numel_str(model_numel),
    )

    # == build ema for diffusion model ==
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)
    ema.eval()
    update_ema(ema, model, decay=0, sharded=False)


    # == setup loss function, build scheduler ==
    # scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # == setup optimizer ==
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        adamw_mode=True,
        lr=cfg.get("lr", 1e-5),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("adam_eps", 1e-8),
    )

    warmup_steps = cfg.get("warmup_steps", None)
    milestones_lr = cfg.get("milestones_lr", None)

    if warmup_steps is None:
        lr_scheduler = None
    else:
        if milestones_lr is None:
            lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=warmup_steps)
        else:
            lr_scheduler = MultiStepWithLinearWarmupLR(
                optimizer, milestones_lr=milestones_lr, warmup_steps=warmup_steps)

    # == additional preparation ==
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)
    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    logger.info("Preparing for distributed training...")
    # == boosting ==
    # NOTE: we set dtype first to make initialization of model consistent with the dtype; then reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boosting model for distributed training")


    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    start_epoch = start_step = log_step = acc_step = 0
    drop_cond_ratio = cfg.get("drop_cond_ratio", 0.0)
    drop_cond_ratio_t = cfg.get("drop_cond_ratio_t", 0.4)
    running_loss = 0.0
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint")
        ret = load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=None if cfg.get("start_from_scratch", False) else optimizer,
            lr_scheduler=None if cfg.get("reset_lr", False) or cfg.get("start_from_scratch", False) else lr_scheduler,
            sampler=None if cfg.get("start_from_scratch", False) else sampler,
            local_master=coordinator.is_node_master(),
        )
        if not cfg.get("start_from_scratch", False):
            start_epoch, start_step = ret
            if cfg.get("reset_lr", False) and lr_scheduler:
                total_step = start_epoch * num_steps_per_epoch + start_step
                lr_scheduler.last_epoch = total_step
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)

    model_sharding(ema)

    with RandomStateManager(verbose=True):
        print(f"{torch.randn(3)} {torch.randn(3, device=get_current_device())} "
              f"on rank {dist.get_rank()} "
              f"dp_rank {dist.get_rank(get_data_parallel_group())}")

    # =======================================================
    # 5. training loop
    # =======================================================
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    coordinator.block_all()
    timers = {}
    timer_keys = [
        "move_data",
        "encode",
        "move_data2",
        "mask",
        "diffusion",
        "backward",
        "update_ema",
        "reduce_loss",
        "misc",
    ]


    single_lidar = cfg.get("single_lidar", False)
    for key in timer_keys:
        if record_time:
            timers[key] = Timer(key, coordinator=None)
        else:
            timers[key] = nullcontext()

    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)
        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:

            for step, batch in pbar:
                if verbose_mode:
                    logger.info(f"Dataloader returns data! step={step}")
                B, T, NC = batch["pixel_values"].shape[:3]
                if single_lidar:  T = 1
                logging.debug(f"bs = {B}; t = {T}; shape = {batch['pixel_values'].shape}")
                timer_list = []
                with timers["move_data"] as move_data_t:
                    x_image = batch.pop("pixel_values").to(device, dtype)
                    x_image = rearrange(x_image, "B T NC C ... -> (B NC) C T ...")  # BxNC, C, T, H, W
                    
                    x = batch.pop("lidar_values").to(device, dtype)
                    if single_lidar:
                        random_number = random.randint(0, 5)
                        x = x[:, :, random_number,  ...].unsqueeze(2)
                        
                    x = rearrange(x, "B T NC C ... -> (B NC) C T ...")  # BxNC, C, T, H, W

                    
                    y = batch.pop("captions")[0]  # B, just take first frame
                    cond_frame_x = repeat(x[:,:,0:1], "B C 1 H W ... -> B C (1 T) H W ...", T=T)
                    # B, T, NC, 3, 7
                    cams = batch.pop("camera_param").to(device, dtype)
                    
                    cams_int, cams_ext =  cams[:,:,:,:3,:3], cams[:,:,:,:3,3:]
                    # cams_int  =  batch['camera_param_raw']['int']  
                    if single_lidar:
                        cams_int, cams_ext = cams_int[:, :, random_number,  ...].unsqueeze(2), cams_ext[:, :, random_number,  ...].unsqueeze(2)


                    cams_aug = batch['camera_param_raw']['aug'].to(device, dtype)
                    temp_cams = cams_aug[:,:,:,:3,:3]@cams[:,:,:,:,:3]
                    temp_cams[:,:,:,:,2]+=cams_aug[:,:,:,:3,3]
                    cams[:,:,:,:,:3] = temp_cams
                    cams = rearrange(cams, "B T NC ... -> (B NC) T 1 ...")  # BxNC, T, 1, 3, 7
                    rel_pos = batch.pop("frame_emb").to(device, dtype)
                    rel_pos = repeat(rel_pos, "B T ... -> (B NC) T 1 ...", NC=NC)  # BxNC, T, 1, 4, 4
                if record_time:
                    timer_list.append(move_data_t)
 
                if verbose_mode:
                    logger.info(f"Start model forward step! step={step}")
                # == diffusion loss computation ==
                # with timers["diffusion"] as loss_t:
                #     loss_dict = scheduler.training_losses(model, x, model_args, mask=mask)
                ###################---------------------------------------------------------------------------##############
                model.train()
                z, posterior, x_rec = model(x)
                
                # == loss initialization ==
                loss_dict = {}
                vae_loss = torch.tensor(0.00, device=device, requires_grad=True)
                # == loss: real image reconstruction ==
                sigloss, weighted_nll_loss, weighted_kl_loss = VAE_Losses(x, x_rec, posterior)
                loss_dict["sigloss"] = sigloss 
                loss_dict["kl_loss"] = weighted_kl_loss 
                loss_dict["weighted_nll_loss"] = weighted_nll_loss 
                vae_loss = vae_loss + weighted_nll_loss + weighted_kl_loss + sigloss
                
                loss_dict["loss"] = vae_loss 
                print("\n", {k: v.item() for k, v in loss_dict.items()})
                    
                # if record_time:
                #     timer_list.append(loss_t)
                # NOTE: backward needs all_reduce, we sychronize here!
                coordinator.block_all()

                if verbose_mode:
                    logger.info(f"Start model backward step! step={step}, loss={loss_dict['loss']}")
                # == backward & update ==
                with timers["backward"] as backward_t:
                    loss = loss_dict["loss"].mean()
                    booster.backward(loss=loss, optimizer=optimizer)
                    if verbose_mode:
                        logger.info(f"Start model update step! step={step}")
                    optimizer.step()
                    optimizer.zero_grad()

                    # update learning rate
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                if record_time:
                    timer_list.append(backward_t)

                if verbose_mode:
                    logger.info(f"Start after step ops! step={step}")
                # == update EMA ==
                with timers["update_ema"] as ema_t:
                    update_ema(ema, model.module, optimizer=optimizer, decay=cfg.get("ema_decay", 0.9999))
                if record_time:
                    timer_list.append(ema_t)

                # == update log info ==
                with timers["reduce_loss"] as reduce_loss_t:
                    all_reduce_mean(loss)
                    running_loss += loss.item()
                    global_step = epoch * num_steps_per_epoch + step
                    log_step += 1
                    acc_step += 1
                if record_time:
                    timer_list.append(reduce_loss_t)

                if record_time:
                    misc_t = timers['misc'].__enter__()
                    timer_list.append(misc_t)
                # == logging ==
                if coordinator.is_master() and (global_step + 1) % cfg.get("log_every", 1) == 0:
                    avg_loss = running_loss / log_step
                    lr = optimizer.param_groups[0]["lr"]
                    # progress bar, use str to avoid conversion
                    pbar.set_postfix({"loss": avg_loss, "step": str(step), "global_step": str(global_step), "lr": lr})
                    # tensorboard
                    tb_writer.add_scalar("loss", loss.item(), global_step)
                    tb_writer.add_scalar("avg_loss", avg_loss, global_step)
                    tb_writer.add_scalar("lr", lr, global_step)

                    running_loss = 0.0
                    log_step = 0
                # == checkpoint saving ==
                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    if verbose_mode:
                        logger.info(f"Start to save ckpt! step={step}")
                    model_gathering(ema, ema_shape_dict)
                    save_dir = save(
                        booster,
                        exp_dir,
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        sampler=sampler,
                        epoch=epoch,
                        step=step + 1,
                        global_step=global_step + 1,
                        batch_size=cfg.get("batch_size", None),
                    )
                    if dist.get_rank() == 0:
                        model_sharding(ema)
                    logger.info(
                        "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                        epoch,
                        step + 1,
                        global_step + 1,
                        save_dir,
                    )
                    sub_dir_name = os.path.basename(save_dir)

             
                sample_every = cfg.get("sample_every", 0)

                if sample_every > 0 and (global_step + 1) % sample_every == 0:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    with RandomStateManager(verbose=False):
                        with torch.no_grad():
                            val_dir = run_validation_vae(
                                cfg.val,
                                model,
                                device,
                                dtype,
                                val_dataloader,
                                coordinator,
                                global_step + 1,
                                exp_dir,
                                cfg.mv_order_map,
                                cfg.t_order_map,
                            )
                    val_sampler.reset()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    sub_dir_name = os.path.basename(val_dir)
                    
                if record_time:
                    misc_t.__exit__(*sys.exc_info())
                    log_str = f"Rank {dist.get_rank()} | Epoch {epoch} | Step {step} | "
                    for timer in timer_list:
                        log_str += f"{timer.name}: {timer.elapsed_time:.3f}s | "
                    log_str += f"Total: {sum([t.elapsed_time for t in timer_list]):.3f}s"
                    logger.info(log_str)
        sampler.reset()
        start_step = 0

if __name__ == "__main__":
    main()
    
     

 