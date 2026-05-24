import math
import random
from collections import OrderedDict
from inspect import isfunction
import torch
import torch.distributed as dist
from einops import rearrange, repeat
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator, ProcessGroupMesh
from uniscenev2_video.acceleration.parallel_states import set_data_parallel_group, set_sequence_parallel_group, get_data_parallel_group
from uniscenev2_video.acceleration.parallel_states import get_data_parallel_group, get_sequence_parallel_group
from uniscenev2_video.acceleration.plugin import ZeroSeqParallelPlugin
from uniscenev2_video.acceleration.communications import gather_tensors
from .inference_utils import add_null_condition, concat_6_views_pt, enable_offload, concat_8_views_pt
from .misc import get_logger, move_to, warn_once, collate_bboxes_to_maxlen, add_box_latent
from IPython import embed
import os
import copy
import logging
from functools import partial
from uniscenev2_video.datasets.utils import save_sample
from uniscenev2_video.registry import SCHEDULERS, build_module
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np
import imageio
import matplotlib.cm as cm
from tqdm import tqdm
import matplotlib.pyplot as plt

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    else:
        return d() if isfunction(d) else d

def write_video(save_path,img_list,fps=10):
    img_numpy_list = img_list.numpy()
    videoWriter = imageio.get_writer(save_path, fps=fps)
    for idx in range(len(img_numpy_list)):
        videoWriter.append_data(img_numpy_list[idx])
    videoWriter.close()


# PALETTE = [
#     [120, 120, 240],  # 柔和的蓝紫色  
#     [128, 64, 128],   # 经典的道路紫色  
#     [135, 206, 235],  # 天蓝色  
#     [0, 102, 204],    # 深蓝色 
#     [255, 85, 85],    # 鲜明的红色  
#     [255, 165, 0],    # 橙色  
#     [0, 0, 0],        # 黑色  
#     [192, 192, 192],  # 灰色  
#     [169, 169, 169],  # 深灰色  
#     [50, 205, 50],    # 明亮的绿色  
#     [255, 222, 173],   # 浅杏色  
# ]

# def write_seg_video(save_path,x,fps=10,value_range=(-1, 1)):
#     x = x.permute(1, 2, 3, 0).to(torch.uint8)
#     img_numpy_list = x.cpu().numpy()
#     seg_map = np.zeros((img_numpy_list.shape[0],img_numpy_list.shape[1],img_numpy_list.shape[2],3))
#     for seg_id in np.unique(img_numpy_list):
#         seg_map[img_numpy_list[:,:,:,0]==seg_id,:] = PALETTE[seg_id-1]
#     videoWriter = imageio.get_writer(save_path, fps=fps)
#     for idx in range(len(seg_map)):
#         videoWriter.append_data(seg_map[idx])
#     videoWriter.close()

# 语义图颜色映射表（RGBA）
occ_colors_map = np.array([
    [255, 158, 0, 255],    # 1 vehicle
    [255, 99, 71, 255],    # 2 placeholder
    [255, 140, 0, 255],    # 3 placeholder
    [255, 69, 0, 255],     # 4 placeholder
    [233, 150, 70, 255],   # 5 czone_sign
    [220, 20, 60, 255],    # 6 bicycle
    [255, 61, 99, 255],    # 7 generic_object
    [0, 0, 230, 255],      # 8 pedestrian
    [47, 79, 79, 255],     # 9 traffic_cone
    [112, 128, 144, 255],  # 10 barrier
    [0, 207, 191, 255],    # 11 background_surface
    [255, 0, 255, 255],    # 12 drive surface
    [75, 0, 75, 255],      # 13 no drive surface
    [0, 175, 0, 255],      # 14 road bound line
    [255, 0, 0, 255],      # 15 road line
    [0, 175, 0, 255],      # 16 None
    [0, 0, 0, 255],        # 17 unknown
]).astype(np.uint8)

# 设置透明通道（alpha = 0）的类别（11~17）
for i in range(10, 17):
    occ_colors_map[i][3] = 0

# 加载 .npz 文件
def load_npz_file(file_path):
    data = np.load(file_path)
    key = list(data.keys())[0]
    return data[key]

# 创建带有透明背景的语义图（RGBA）
def create_rgba_mask(semantic, color_map):
    H, W = semantic.shape
    indices = np.where(semantic == 0, 16, semantic - 1)
    indices = np.clip(indices, 0, len(color_map) - 1).astype(np.int_)
    rgba_mask = color_map[indices]
    return rgba_mask
 
PALETTE = [
    [255, 158, 0],    # 1 vehicle
    [255, 99, 71],    # 2 placeholder
    [255, 140, 0],    # 3 placeholder
    [255, 69, 0],     # 4 placeholder
    [233, 150, 70],   # 5 czone_sign
    [220, 20, 60],    # 6 bicycle
    [255, 61, 99],    # 7 generic_object
    [0, 0, 230],      # 8 pedestrian
    [47, 79, 79],     # 9 traffic_cone
    [112, 128, 144],  # 10 barrier
    [0, 207, 191],    # 11 background_surface
    [255, 0, 255],    # 12 drive surface
    [75, 0, 75],      # 13 no drive surface
    [0, 175, 0],      # 14 road bound line
    [255, 0, 0],      # 15 road line
    [0, 175, 0],      # 16 None
    [0, 0, 0],        # 17 unknown
]

def write_seg_video(save_path, x, fps=10, value_range=(-1, 1)):
    # 将输入张量 x 转换为 NumPy 数组，并调整维度
    x = x.permute(1, 2, 3, 0).to(torch.uint8)  # 转换为 uint8 类型
    img_numpy_list = x.cpu().numpy()  # 转换为 NumPy 数组

    # 初始化 seg_map，确保数据类型为 uint8
    seg_map = np.zeros(
        (img_numpy_list.shape[0], img_numpy_list.shape[1], img_numpy_list.shape[2], 3),
        dtype=np.uint8
    )

    # 根据分割图的值填充颜色
    for seg_id in np.unique(img_numpy_list):
        if seg_id == 0:
            continue  # 忽略背景（假设 seg_id=0 表示背景）
        seg_map[img_numpy_list[:, :, :, 0] == seg_id] = PALETTE[seg_id - 1]

    # 使用 imageio 写入视频
    videoWriter = imageio.get_writer(save_path, fps=fps)
    for idx in range(len(seg_map)):
        videoWriter.append_data(seg_map[idx])
    videoWriter.close()
    

    
    
def depth_to_rgba(depth, cmap='viridis'):
    # 深度值有效性判断（仅基于 depth 值）
    depth_valid = (depth > 0) & (depth < 39)

    if np.any(depth_valid):
        vmin, vmax = depth[depth_valid].min(), depth[depth_valid].max()
        norm = plt.Normalize(vmin, vmax)
        cmap = cm.get_cmap(cmap)
        depth_normalized = cmap(norm(depth))
        depth_rgba = (depth_normalized * 255).astype(np.uint8)
    else:
        # 全部无效
        depth_rgba = np.zeros((depth.shape[0], depth.shape[1], 4), dtype=np.uint8)
        return depth_rgba

    # 构建 alpha 通道：仅基于 depth 有效性
    alpha = np.ones_like(depth, dtype=np.uint8) * 255
    alpha[~depth_valid] = 0

    # 将无效区域的 RGB 通道也置为 0
    depth_rgba[~depth_valid, :3] = 0  # RGB 通道设为 0
    depth_rgba[~depth_valid, 3] = 0  # 确保 alpha 也为 0（透明）

    return depth_rgba

def write_depth_video(save_path, x, fps=10):
    """
    将输入的深度视频保存为热力图视频。
    参数:
        save_path (str): 保存视频的路径。
        x (torch.Tensor): 深度视频张量，形状为 (T, H, W)，其中 T 是帧数，H 和 W 是高度和宽度。
        fps (int): 视频的帧率，默认为 10。
    """
    # 如果输入张量包含批次维度，移除批次维度
    if len(x.shape) == 4:  # 形状为 [B, T, H, W]
        x = x.squeeze(0)  # 移除批次维度，形状变为 [T, H, W]
    # 确保输入张量形状为 (T, H, W)
    if len(x.shape) != 3:
        raise ValueError(f"Expected input shape (T, H, W), but got {x.shape}")
    # 确保 save_path 包含有效扩展名
    if not save_path.endswith(".mp4"):
        save_path += "_depth.mp4"
    # 确保输入张量形状为 (T, H, W)
    if len(x.shape) != 3:
        raise ValueError(f"Expected input shape (T, H, W), but got {x.shape}")
    # 归一化深度值到 [0, 1]
    depth_min = x.min()
    depth_max = x.max()
    normalized_depth = ((x - depth_min) / (depth_max - depth_min) * 100).to(torch.uint8)
   
    # 创建视频写入器
    writer = imageio.get_writer(save_path, fps=fps)
    # 逐帧处理并写入
    for depth_frame in normalized_depth.cpu().numpy():
        # # 应用 JET 颜色映射
        # heatmap = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
        # # 转换颜色空间从 BGR 到 RGB
        # heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        # import pdb; pdb.set_trace()
        heatmap_rgb = depth_to_rgba(depth_frame)
        writer.append_data(heatmap_rgb)
    writer.close()
    
    
def create_colossalai_plugin(plugin, dtype, grad_clip, sp_size, reduce_bucket_size_in_m: int = 20, overlap_allgather=False, verbose=False):
    if plugin == "zero2":
        assert sp_size == 1, "Zero2 plugin does not support sequence parallelism"
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=dtype,
            initial_scale=2**16,
            max_norm=grad_clip,
            reduce_bucket_size_in_m=reduce_bucket_size_in_m,
            overlap_allgather=overlap_allgather,
            verbose=verbose,
        )
        dp_size = dist.get_world_size()
        DP_AXIS, SP_AXIS = 0, 1
        pg_mesh = ProcessGroupMesh(dp_size, sp_size)
        dp_group = pg_mesh.get_group_along_axis(DP_AXIS)
        sp_group = pg_mesh.get_group_along_axis(SP_AXIS)
        set_data_parallel_group(dp_group)
        set_sequence_parallel_group(sp_group)
    elif plugin == "zero2-seq":
        assert sp_size > 1, "Zero2-seq plugin requires sequence parallelism"
        plugin = ZeroSeqParallelPlugin(
            sp_size=sp_size,
            stage=2,
            precision=dtype,
            initial_scale=2**16,
            max_norm=grad_clip,
            reduce_bucket_size_in_m=reduce_bucket_size_in_m,
            overlap_allgather=overlap_allgather,
            verbose=verbose,
        )
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {plugin}")
    return plugin


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
) -> None:
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed":
            continue
        if not param.requires_grad:
            continue
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        else:
            if param.data.dtype != torch.float32:
                param_id = id(param)
                # master_param = optimizer._param_store.working_to_master_param[param_id]
                master_param = optimizer.working_to_master_param[param_id]
                param_data = master_param.data
            else:
                param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


class MaskGenerator:
    def __init__(self, mask_ratios):
        valid_mask_names = [
            "identity",
            "quarter_random",
            "quarter_head",
            "quarter_tail",
            "quarter_head_tail",
            "image_random",
            "image_head",
            "image_tail",
            "image_head_tail",
            "random",
            "intepolate",
        ]
        assert all(
            mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ), f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}"
        assert all(
            mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}"
        assert all(
            mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}"
        # sum of mask_ratios should be 1
        if "identity" not in mask_ratios:
            mask_ratios["identity"] = 1.0 - sum(mask_ratios.values())
        assert math.isclose(
            sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}"
        get_logger().info("mask ratios: %s", mask_ratios)
        self.mask_ratios = mask_ratios

    def get_mask(self, x, scales=4):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break

        num_frames = x.shape[2]
        # Hardcoded condition_frames
        condition_frames_max = num_frames // scales
        # condition_frames_max = num_frames

        mask = torch.ones(num_frames, dtype=torch.bool, device=x.device)
        if num_frames <= 1:
            return mask

        if mask_name == "quarter_random":
            random_size = random.randint(1, condition_frames_max)
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "image_random":
            random_size = 1
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "quarter_head":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
        elif mask_name == "image_head":
            random_size = 1
            mask[:random_size] = 0
        elif mask_name == "quarter_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[-random_size:] = 0
        elif mask_name == "image_tail":
            random_size = 1
            mask[-random_size:] = 0
        elif mask_name == "quarter_head_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "image_head_tail":
            random_size = 1
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "intepolate":
            random_start = random.randint(0, 1)
            mask[random_start::2] = 0
        elif mask_name == "random":
            mask_ratio = random.uniform(0.1, 0.9)
            mask = torch.rand(num_frames, device=x.device) > mask_ratio
            # if mask is all False, set the last frame to True
            if not mask.any():
                mask[-1] = 1

        return mask

    def get_masks(self, x):
        masks = []
        for _ in range(len(x)):
            mask = self.get_mask(x)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        return masks


def sp_vae(x, vae_func, sp_group: dist.ProcessGroup):
    """use sp_group to scatter vae encode

    Args:
        x (torch.Tensor): (B NC) C T ... or B C T ...
        vae (nn.Module): vae model
        dp_group (dist.ProcessGroup): _description_
    """
    group_size = dist.get_world_size(sp_group)
    local_rank = dist.get_rank(sp_group)
    B = x.shape[0]

    copy_size = group_size
    while copy_size < B:
        copy_size += group_size
    per_rank_bs = copy_size // group_size

    if per_rank_bs >= B:
        warn_once(
            f"x shape {x.shape} with {group_size} ranks does not fit dp_encode "
            f"fallback to the normal one."
        )
        return vae_func(x)

    if copy_size > B:
        x_copy_num = math.ceil(copy_size / B)
        x_temp = torch.cat([x for _ in range(x_copy_num)])[:copy_size]
        warn_once(f"Pad B={B} to {x_temp.shape}")
    elif copy_size < B:
        raise RuntimeError(f"{x.shape} got copy_size={copy_size}")
    else:
        x_temp = x

    local_x = x_temp[local_rank * per_rank_bs:(local_rank + 1) * per_rank_bs]
    assert local_x.shape[0] == per_rank_bs
    del x_temp
    local_latent = vae_func(local_x)

    global_latent = [torch.empty_like(local_latent) for _ in range(group_size)]
    dist.all_gather(global_latent, local_latent, group=sp_group)
    dist.barrier(sp_group)
    del local_latent
    global_latent = torch.cat(global_latent, dim=0)[:B]
    return global_latent


@torch.no_grad()
def run_validation(val_cfg, text_encoder, vae, model, device, dtype,
                   val_loader: torch.utils.data.DataLoader,
                   coordinator: DistCoordinator, global_step: int,
                   exp_dir: str, mv_order_map, t_order_map, bbox_mode=None, 
                    nuplan = False,num_cond_frame=3, cfg=None,use_file_name=False ):
    video_save_dir = os.path.join(exp_dir, f"validation-global_step{global_step}")
    if use_file_name:
        video_save_dir = os.path.join(exp_dir, f"save_with_filename")
        
    if 1:#coordinator.is_master():
        os.makedirs(video_save_dir, exist_ok=True)
    verbose = val_cfg.get("verbose", 1)
    num_sample = val_cfg.get("num_sample", 1 )
    save_fps = val_cfg.save_fps

    val_cfg.cpu_offload = val_cfg.get("cpu_offload", False)
    if val_cfg.cpu_offload:
        raise NotImplementedError()
        text_encoder.t5.model.to("cpu")
        model.to("cpu")
        vae.to("cpu")
        text_encoder.t5.model, model, vae, last_hook = enable_offload(
            text_encoder.t5.model, model, vae, device)

    validation_scheduler = build_module(val_cfg.scheduler, SCHEDULERS)
    text_encoder.y_embedder = model.module.y_embedder  # hack for classifier-free guidance
    model.eval()

    
    for i, batch in enumerate(tqdm(val_loader, desc="Validation Progress")):
        if i%1==0:
        # for i, batch in enumerate(val_loader):
            first_frame_l1_name  = batch['meta_data']['metas'][0][0]._data['filename'][0]
            save_file_name = first_frame_l1_name.split("/")[-3] + "-" + first_frame_l1_name.split("/")[-2]+ "-" +first_frame_l1_name.split("/")[-1]
            # ['./dataset1/nuplan/sensor_blobs_train/2021.09.16.15.12.03_veh-42_01037_01434/CAM_L1/54aa3ac579e15741.jpg', './dataset1/nuplan/sensor_blobs_train/2021.09.16.15.12.03_veh-42_01037_01434/CAM_L0/70e15b7085f753bd.jpg', './dataset1/nuplan/sensor_blobs_train/2021.09.16.15.12.03_veh-42_01037_01434/CAM_F0/92dee83e9d0556da.jpg', './dataset1/nuplan/sensor_blobs_train/2021.09.16.15.12.03_veh-42_01037_01434/CAM_R0/d5b23a2c8669506a.jpg', 
            # './dataset1/nuplan/sensor_blobs_train/2021.09.16.15.12.03_veh-42_01037_01434/CAM_R1/ed1a64e4ada35cd3.jpg', './dataset1/nuplan/sensor_blobs_train/2021.09.16.15.12.03_veh-42_01037_01434/CAM_R2/f090cae4679d5ff7.jpg', './dataset1/nuplan/sensor_blobs_train/2021.09.16.15.12.03_veh-42_01037_01434/CAM_B0/d55171b2855a5ed7.jpg', './dataset1/nuplan/sensor_blobs_train/2021.09.16.15.12.03_veh-42_01037_01434/CAM_L2/92f0ccb2fe275bdf.jpg']
            
            torch.cuda.empty_cache()
            generator = torch.Generator("cpu").manual_seed(val_cfg.seed)
            bl_generator = torch.Generator("cpu").manual_seed(val_cfg.seed)
            B, T, NC = batch["pixel_values"].shape[:3]
            latent_size = vae.get_latent_size((T, *batch["pixel_values"].shape[-2:]))
            
            if bbox_mode != None:
                bbox = batch.pop("bboxes_3d_data")
                bbox = [bbox_i.data for bbox_i in bbox]
                bbox = collate_bboxes_to_maxlen(bbox, device, dtype, NC, T)
            else: bbox = None
            pixel_values = batch["pixel_values"].to(device, dtype)
            pixel_values = rearrange(pixel_values, "B T NC C ... -> (B NC) C T ...")
            # cond_frame_x = repeat(pixel_values[:,:,0:1], "B C 1 H W ... -> B C (1 T) H W ...", T=T)
            cond_frame_x = torch.zeros_like(pixel_values)
            cond_frame_x[:,:,0:num_cond_frame]=pixel_values[:,:,0:num_cond_frame] 
            
            with torch.no_grad():
                # with RandomStateManager(verbose=verbose_mode):
                cond_frame_x = sp_vae(cond_frame_x, vae.encode,
                        get_sequence_parallel_group())
            
            # == prepare batch prompts ==
            y = batch.pop("captions")[0]  # B, just take first frame
            
            
            if cfg.model.with_depth and cfg.model.with_seg:
                seg_map = batch.pop('semantic_map').to(device, dtype)
                seg_map = rearrange(seg_map, "B T NC C ... -> (B NC) C T ...")  # BxNC, C, T, H, W
                depth_map = batch.pop('depth_map').to(device, dtype)
                depth_map = rearrange(depth_map, "B T NC C ... -> (B NC) C T ...")  # BxNC, C, T, H, W
                
            
            # # B, T, NC, 3, 7
            # cams = batch.pop("camera_param").to(device, dtype)
            # cams_aug = batch['camera_param_raw']['aug'].to(device, dtype)
            # temp_cams = cams_aug[:,:,:,:3,:3]@cams[:,:,:,:,:3]
            # temp_cams[:,:,:,:,2]+=cams_aug[:,:,:,:3,3]
            # cams[:,:,:,:,:3] = temp_cams
            # cams = rearrange(cams, "B T NC ... -> (B NC) T 1 ...")  # BxNC, T, 1, 3, 7
            # rel_pos = batch.pop("frame_emb").to(device, dtype)
            # rel_pos = repeat(rel_pos, "B T ... -> (B NC) T 1 ...", NC=NC)  # BxNC, T, 1, 4, 4
            # with torch.no_grad():
            #     cam_K = cams[:,:,0,:,:3].clone()
            #     ego_to_world = rel_pos[:,:,0].clone()
            #     cam_ext = torch.zeros_like(ego_to_world)
            #     cam_ext[:,:,0,0] = 1
            #     cam_ext[:,:,1,1] = 1
            #     cam_ext[:,:,2,2] = 1
            #     cam_ext[:,:,3,3] = 1
            #     cam_ext[:,:,:3] = cams[:,:,0,:,3:]
            #     c2w = ego_to_world @ cam_ext
            #     H, W = batch["pixel_values"].shape[-2:]
                # plucker_embed = model.module.ray_condition(cam_K, c2w, H, W, device=cam_K.device)
                # plucker_embed = rearrange(plucker_embed, "(B NC) C T ... -> B NC T C ...", NC=NC)

            # == model input format ==
            model_args = {}
            if cfg.model.with_depth and cfg.model.with_seg:
                    model_args['seg_map'] = seg_map
                    model_args['depth_map'] = depth_map
                    
            # model_args["plucker_embed"] = plucker_embed
            model_args["cond_frame_x"] = cond_frame_x
            model_args["bbox"] = bbox
            # model_args["cams"] = cams
            # model_args["rel_pos"] = rel_pos
            model_args["fps"] = batch.pop('fps')
            model_args["height"] = batch.pop("height")
            model_args["width"] = batch.pop("width")
            model_args["num_frames"] = batch.pop("num_frames")
            model_args = move_to(model_args, device=device, dtype=dtype)
            # no need to move these
            model_args["mv_order_map"] = mv_order_map
            model_args["t_order_map"] = t_order_map

            _fpss = gather_tensors(model_args['fps'], pg=get_data_parallel_group())
            for ns in range(num_sample):

                z = torch.randn(
                    len(y)*NC, vae.out_channels, *latent_size, generator=generator,
                ).to(device=device, dtype=dtype)
                
                if bbox is not None:
                    # null set values to all zeros, this should be safe
                    bbox = add_box_latent(bbox, B, NC, T, 
                        partial(model.module.sample_box_latent, generator=bl_generator))
                    # overwrite!
                    new_bbox = {}
                    for k, v in bbox.items():
                        new_bbox[k] = rearrange(v, "B T NC ... -> (B NC) T ...")  # BxNC, T, len, 3, 7
                    model_args["bbox"] = move_to(new_bbox, device=device, dtype=dtype)


                # == add null condition ==
                # y is handled by scheduler.sample
                _model_args = add_null_condition(
                    copy.deepcopy(model_args),
                    # model.module.camera_embedder.uncond_cam.to(device),
                    prepend=False,
                )
    
                # == inference ==
                with torch.no_grad():
                    samples = validation_scheduler.sample(
                        model,
                        text_encoder,
                        z=z,
                        prompts=y,
                        device=device,
                        additional_args=_model_args,
                        progress=verbose >= 2 and 1,#coordinator.is_master(),
                        mask=None,
                    )
                # samples = rearrange(samples, "B (C NC) T ... -> (B NC) C T ...", NC=NC)
                samples = vae.decode(samples.to(dtype), num_frames=T)
                samples = rearrange(samples, "(B NC) C T ... -> B NC C T ...", NC=NC)
                
                if val_cfg.cpu_offload:
                    last_hook.offload()
                vid_samples = []
                for sample in samples:
                    if nuplan:
                        vid_samples.append(
                            concat_8_views_pt(sample, oneline=False)
                        )
                    else:
                        vid_samples.append(
                            concat_6_views_pt(sample, oneline=False)
                        )
                samples = torch.stack(vid_samples, dim=0)  # B, C, T, ...
                del z, vid_samples, _model_args
                torch.cuda.empty_cache()

                # gather sample from all processes
                coordinator.block_all()
                _samples = gather_tensors(samples, pg=get_data_parallel_group())
                # == save samples ==
                if 1:#coordinator.is_master():
                    video_clips = []
                    fpss = []
                    for sample, fps in zip(_samples, _fpss):  # list of B, C, T ...
                        video_clips += [s.cpu() for s in sample]  # list of C, T ...
                        fpss += [int(_fps) for _fps in fps]
                    for idx, video in enumerate(video_clips):
                        save_path = os.path.join(video_save_dir, f"sample-{i + idx:04d}" )
                        if use_file_name:
                            save_path = os.path.join(video_save_dir, f"sample-{save_file_name}")
                        save_path = save_sample(
                            video,
                            fps=save_fps if save_fps else fpss[idx],
                            save_path=save_path,
                            high_quality=True,
                            verbose=verbose >= 2,
                        )
                del samples, _samples
                coordinator.block_all()

            # save_gt
            x = batch.pop("pixel_values").to(device, dtype)
            # x = rearrange(x, "B T NC C ... -> B NC C T ...")  # BxNC, C, T, H, W
            x = rearrange(x, "B T NC C ... -> (B NC) C T ...")  # BxNC, C, T, H, W
            with torch.no_grad():
                # with RandomStateManager(verbose=verbose_mode):
                x = sp_vae(x, vae.encode,
                        get_sequence_parallel_group())

                x = sp_vae(x, vae.decode,
                        get_sequence_parallel_group())

                x = rearrange(x, "(B NC) C T ... -> B NC C T ...",NC=NC)  # BxNC, C, T, H, W
                
                
            torch.cuda.empty_cache()
            _samples = gather_tensors(x, pg=get_data_parallel_group())
            if 1:#coordinator.is_master():
                samples = []
                fpss = []
                for sample, fps in zip(_samples, _fpss):
                    samples += [s.cpu() for s in sample]
                    fpss += [int(_fps) for _fps in fps]
                for idx, sample in enumerate(samples):
                    if nuplan:
                        vid_sample = concat_8_views_pt(sample, oneline=False)
                    else:
                        vid_sample = concat_6_views_pt(sample, oneline=False)

                    save_path = os.path.join(video_save_dir, f"gt-{i + idx:04d}")
                    if use_file_name:
                        save_path = os.path.join(video_save_dir, f"gt-{save_file_name}")
                    save_path = save_sample(
                        vid_sample,
                        fps=save_fps if save_fps else fpss[idx],
                        save_path=save_path,
                        high_quality=True,
                        verbose=verbose >= 2,
                    )
                
            del _samples
            torch.cuda.synchronize()
            coordinator.block_all()

            torch.cuda.empty_cache()
            # import pdb; pdb.set_trace()
            if cfg.model.with_depth and cfg.model.with_seg:
                torch.cuda.empty_cache()
                logging.info("start gather seg and depth ...")
                depth_map = rearrange(depth_map, "(B NC) C T ... -> B NC C T ...",NC=NC)
                _depth = gather_tensors(depth_map, pg=get_data_parallel_group())
                seg_map = rearrange(seg_map, "(B NC) C T ... -> B NC C T ...",NC=NC)
                _seg = gather_tensors(seg_map, pg=get_data_parallel_group())
                logging.info("end gather gt ...")
                if 1:#coordinator.is_master():
                    # embed()
                    # exit()
                    dep=[]
                    fpss = []
                    seg = []
                    for depths, semantic, fps in zip(_depth, _seg, _fpss):
                        dep += [s.cpu() for s in depths]
                        fpss += [int(_fps) for _fps in fps]
                        seg += [se.cpu() for se in semantic]
                    idx = 0
                    for idx, sample in enumerate(dep):
                        vid_sample_depth = concat_8_views_pt(dep[idx], oneline=False)
                        
                        depth_path = str(os.path.join(video_save_dir, f"dep-{ i + idx:04d}.mp4") )
                        seg_path = str(os.path.join(video_save_dir, f"seg-{ i + idx:04d}.mp4") ) 
                        if use_file_name:
                            depth_path = str(os.path.join(video_save_dir, f"dep-{save_file_name}.mp4") )
                            seg_path = str(os.path.join(video_save_dir, f"seg-{save_file_name}.mp4") )
                        write_depth_video(
                            save_path= depth_path ,
                            x=vid_sample_depth,
                            fps=save_fps if save_fps else fpss[idx],
                        )
                        vid_sample_seg = concat_8_views_pt(seg[idx], oneline=False)
                        write_seg_video(save_path=seg_path  , x=vid_sample_seg,  fps=save_fps if save_fps else fpss[idx])              
                torch.cuda.synchronize()
                coordinator.block_all()
                torch.cuda.empty_cache()



    if val_cfg.cpu_offload:
        # TODO: need to remove hooks
        raise NotImplementedError()
    torch.cuda.empty_cache()
    return video_save_dir





#################################################---lidarvae----##################################################################
@torch.no_grad()
def run_validation_vae(val_cfg, model, device, dtype,
                   val_loader: torch.utils.data.DataLoader,
                   coordinator: DistCoordinator, global_step: int,
                   exp_dir: str, mv_order_map, t_order_map):
    video_save_dir = os.path.join(exp_dir, f"validation-global_step{global_step}")
    if 1:#coordinator.is_master():
        os.makedirs(video_save_dir, exist_ok=True)
    verbose = val_cfg.get("verbose", 1)
    num_sample = val_cfg.get("num_sample", 2)
    save_fps = val_cfg.save_fps
    model.eval()
    for i, batch in enumerate(val_loader):
        torch.cuda.empty_cache()
        # == inference ==
        with torch.no_grad():
            B, T, NC = batch["pixel_values"].shape[:3]
            # x = batch.pop("pixel_values").to(device, dtype)
            x = batch.pop("lidar_values").to(device, dtype)
            
            x = rearrange(x, "B T NC C ... -> (B NC) C T ...")  # BxNC, C, T, H, W

            z, posterior, x_rec = model(x) 
            x_rec = x_rec.unsqueeze(0).permute(0, 3, 1, 2, 4, 5)
            x = x.unsqueeze(0).permute(0, 3, 1, 2, 4, 5)    ## B T V C H W [1, 17, 6, 3, 112, 200]
            # print( "\n  LIDAR GT:----->", x.shape, x.max().item(), x.min().item(),  "LIDAR REC:----->",  x_rec.max().item(), x_rec.min().item(), )
            
            torch.cuda.empty_cache()
            # gather sample from all processes
            # coordinator.block_all()
            
            # _samples = gather_tensors(samples, pg=get_data_parallel_group())
            # logging.info("end gather sample ...")
            # == save samples ==
            if 1:#coordinator.is_master():
                if torch.distributed.get_rank() == 0:
                    save_validation_vae( all_idx=global_step, curr_idx = i, rec = x_rec.float().cpu(), real = x.float().cpu(), video_save_dir = video_save_dir, 
                                    coordinator=coordinator,verbose=verbose, save_fps= batch.pop('fps')[0]  ) 
            # coordinator.block_all()
                                
    return video_save_dir



def tensor_to_ply(tensor, filename):
    import open3d as o3d
    """
    将形状为 (N, 3) 的torch.tensor保存为PLY文件。
    参数:
    - tensor: 形状为 (N, 3) 的torch.tensor，包含点云数据。
    - filename: 要保存的PLY文件名。
    """
    # 如果tensor是在GPU上，移动到CPU
    points = tensor.cpu().numpy()
    # 创建open3d点云对象
    pcd = o3d.geometry.PointCloud()
    # 设置点云坐标
    pcd.points = o3d.utility.Vector3dVector(points)
    # 保存为PLY文件
    o3d.io.write_point_cloud(filename, pcd)




@torch.no_grad()
def combine_point_cloud(depth_image, intrinsic_matrix, extrinsic_matrix, num_views = 6):
    # 合并所有视角的点云
    all_lidar_points = []
    for i in range(num_views):
        lidar_points = depth2points(depth_image[i].numpy(), intrinsic_matrix[i].numpy(), extrinsic_matrix[i].numpy() )
        all_lidar_points.append( torch.from_numpy(lidar_points).T )
    # 将所有点云合并成一个张量
    merged_lidar_points = torch.cat(all_lidar_points, dim=0)
    return  merged_lidar_points


def img2cam(intr_martix,depth):
    y_index, x_index = np.nonzero(depth)
    ones = np.ones(len(x_index))
    pix_coords = np.stack([x_index, y_index, ones], axis=0)
    normalize_points = np.dot(np.linalg.inv(intr_martix), pix_coords)
    points = (normalize_points * depth[y_index, x_index])
    return points
def  depth2points(final_lidar_map, cam_intrinsic, cam_extrinsic):
    # final_mask = 1-(final_lidar_map > 0).astype(np.uint8)
    cam_point = img2cam(cam_intrinsic[:3,:3],final_lidar_map )
    cam_point_homo = np.concatenate([cam_point,np.ones((1,cam_point.shape[1]))],axis=0)
    # sensor2lidar = np.eye(4)
    # sensor2lidar[:3,:3] = cam_info['sensor2lidar_rotation']
    # sensor2lidar[:3, 3] = cam_info['sensor2lidar_translation']
    lidar_points = np.dot(cam_extrinsic,cam_point_homo)[:3,:]
    return  lidar_points



@torch.no_grad()
def save_validation_vae( all_idx=None, rec=None, real=None, video_save_dir=None, 
                        coordinator=None, verbose=None, save_fps=None, curr_idx=None, 
                        single_lidar=False, _cams=None, w_scale =None,  h_scale=None ):
    with torch.no_grad():
        if 1:#coordinator.is_master():
            os.makedirs(video_save_dir, exist_ok=True)
            # save_gt
            # x = batch.pop("pixel_values").to(device, dtype)
            real = rearrange(real, "B T NC C ... -> B NC C T ...")  # B NC C T, H, W
            torch.cuda.empty_cache()
            if 0:
                real_samples = gather_tensors(real, pg=get_data_parallel_group())
            else: real_samples =  [real]
            # coordinator.block_all()
            if 1:#coordinator.is_master():
                samples = []
                for real_sample  in  real_samples :
                    samples += [s.cpu() for s in real_sample]
                for idx, sample in enumerate(samples):
                    if curr_idx!=None: idx = curr_idx
                    if single_lidar==False:
                        vid_sample = concat_6_views_pt(sample, oneline=False)
                    else: vid_sample =  sample.squeeze()
                    if torch.distributed.get_rank() == 0:
                        _ = save_sample(
                            vid_sample,
                            fps=save_fps  ,
                            save_path=os.path.join(video_save_dir, f"gt_color_{all_idx:04d}_{idx:04d}"),
                            high_quality=True,
                            verbose=verbose >= 2,
                            apply_color = True,
                            normalize=True,
                            # force_video = True,
                        )
                        if single_lidar==False:
                            vid_sample = concat_6_views_pt(sample, oneline=False)
                        else: vid_sample =  sample.squeeze()
                        _ = save_sample(
                            vid_sample,
                            fps=save_fps ,
                            save_path=os.path.join(video_save_dir, f"gt_{all_idx:04d}_{idx:04d}"),
                            high_quality=True,
                            verbose=verbose >= 2,
                            normalize=True,
                        )
                        
                        if single_lidar or _cams!=None  or  w_scale!=None  or  h_scale!=None :
                            real = ((real+1.0)/2.0)*120.0  # B NC C T, H, W
                            real = real[0 , :, 0, 0 ]  #  B NC, T, C, H, W
                            camera_int =  _cams[0][0,0]  # B, T, NC, 3, 7
                            camera_ext =  _cams[1][0,0]
                            # print(real.max(), real.min(), real.shape , camera_int.shape, camera_ext.shape, )
                            # if   w_scale!=None  or  h_scale!=None  :
                            #     camera_int[:, 0,0] *= w_scale
                            #     camera_int[:, 0,2] *= w_scale
                            #     camera_int[:, 1,1] *= h_scale
                            #     camera_int[:, 1,2] *= h_scale
                            # 调用函数将深度图转换为三维点
                            lidar_points = combine_point_cloud( real.float().cpu(), camera_int.float().cpu(), camera_ext.float().cpu(), num_views = 1 if single_lidar else 6 )
                            tensor_to_ply( lidar_points, os.path.join(video_save_dir, f"gt_{all_idx:04d}_{idx:04d}_lidar.ply")   ) 
                            

                    del  real_samples
                    torch.cuda.synchronize()
                    # coordinator.block_all()
            
            
            rec = rearrange(rec, "B T NC C ... -> B NC C T ...")  # BxNC, C, T, H, W
            torch.cuda.empty_cache()
            if 0:
                _samples = gather_tensors(rec, pg=get_data_parallel_group())
            else: _samples =  [rec]
            # coordinator.block_all()
            if 1:#coordinator.is_master():
                samples = []
                for sample in _samples:
                    samples += [s.cpu() for s in sample]
                for idx, sample in enumerate(samples):
                    if curr_idx!=None: idx = curr_idx
                    if single_lidar==False:
                        vid_sample = concat_6_views_pt(sample, oneline=False)
                    else: vid_sample =  sample.squeeze() 
                    if torch.distributed.get_rank() == 0:
                        _ = save_sample(
                            vid_sample,
                            fps=save_fps ,
                            save_path=os.path.join(video_save_dir, f"rec_color_{all_idx:04d}_{idx:04d}"),
                            high_quality=True,
                            verbose=verbose >= 2,
                            apply_color = True,
                            normalize=True,
                            # force_video = True,
                        )
                        
                        if single_lidar or _cams!=None  or  w_scale!=None  or  h_scale!=None :
                            rec = ((rec+1.0)/2.0)*120.0  # B NC C T, H, W
                            rec = rec[0 , :, 0, 0 ]  #  B NC, T, C, H, W
                            lidar_points = combine_point_cloud( rec.float().cpu(), camera_int.float().cpu(), camera_ext.float().cpu() , num_views = 1 if single_lidar else 6 )
                            tensor_to_ply( lidar_points, os.path.join(video_save_dir, f"rec_{all_idx:04d}_{idx:04d}_lidar.ply")   ) 
                            
                            
                    if single_lidar==False:
                        vid_sample = concat_6_views_pt(sample, oneline=False)
                    else: vid_sample =  sample.squeeze() 
                    _ = save_sample(
                        vid_sample,
                        fps=save_fps ,
                        save_path=os.path.join(video_save_dir, f"rec_{all_idx:04d}_{idx:04d}"),
                        high_quality=True,
                        verbose=verbose >= 2,
                        normalize=True,
                    )
                    del   _samples
                    torch.cuda.synchronize()
                    # coordinator.block_all()
        return video_save_dir