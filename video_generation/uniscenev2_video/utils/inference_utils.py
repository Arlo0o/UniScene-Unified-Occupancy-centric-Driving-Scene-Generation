import json
import os
import re
from typing import Tuple
from PIL import Image
from copy import deepcopy
import torch
from einops import repeat, rearrange
from uniscenev2_video.datasets import IMG_FPS

def concat_6_views_pt(imgs, oneline=False):
    if oneline:
        imgs = rearrange(imgs, "NC C T H W -> C T H (NC W)")
    else:
        imgs_up = rearrange(imgs[:3], "NC C T H W -> C T H (NC W)")
        imgs_down = rearrange(imgs[3:], "NC C T H W -> C T H (NC W)")
        imgs = torch.cat([imgs_up, imgs_down], dim=2)
    return imgs
def concat_8_views_pt(imgs, oneline=False):
    if oneline:
        imgs = rearrange(imgs, "NC C T H W -> C T H (NC W)")
    else:
        imgs_up = rearrange(imgs[:4], "NC C T H W -> C T H (NC W)")
        imgs_down = rearrange(imgs[4:], "NC C T H W -> C T H (NC W)")
        imgs = torch.cat([imgs_up, imgs_down], dim=2)
    return imgs

def enable_offload(encoder_model, model, vae, cuda_device):
    from accelerate.big_modeling import cpu_offload_with_hook
    encoder_model, hook1 = cpu_offload_with_hook(encoder_model, cuda_device)
    model, hook2 = cpu_offload_with_hook(model, cuda_device, hook1)
    vae, hook3 = cpu_offload_with_hook(vae, cuda_device, hook2)
    return encoder_model, model, vae, hook3


def add_null_condition(_model_args, prepend=False):
    # will not change the original dict
    unchanged_keys = ["mv_order_map", "t_order_map", "height", "width", "num_frames", "fps"]
    handled_keys = []
    model_args = {}
    if _model_args['bbox'] is not None:
        if "bbox" in _model_args:
            handled_keys.append("bbox")
            _bbox = _model_args['bbox']
            bbox = {}
            for k in _bbox.keys():
                null_item = torch.zeros_like(_bbox[k])
                if prepend:
                    bbox[k] = torch.cat([null_item, _bbox[k]], dim=0)
                else:
                    bbox[k] = torch.cat([_bbox[k], null_item], dim=0)
            model_args['bbox'] = bbox
    if "cams" in _model_args:
        handled_keys.append("cams")
        cams = _model_args['cams']  # BxNC, T, 1, 3, 7
        null_cams = torch.zeros_like(cams)
        # BNC, T, L = null_cams.shape[:3]
        # null_cams = null_cams.reshape(-1, 3, 7)
        # null_cams[:] = uncond_cam[None]
        # null_cams = null_cams.reshape(BNC, T, L, 3, 7)
        if prepend:
            model_args['cams'] = torch.cat([null_cams, cams], dim=0)
        else:
            model_args['cams'] = torch.cat([cams, null_cams], dim=0)

    if "rel_pos" in _model_args:
        handled_keys.append("rel_pos")
        rel_pos = _model_args['rel_pos']  # BxNC, T, 1, 4, 4
        null_rel_pos = torch.zeros_like(rel_pos)
        # BNC, T, L = null_rel_pos.shape[:3]
        # null_rel_pos = null_rel_pos.reshape(-1, 3, 4)
        # null_rel_pos = null_rel_pos.reshape(BNC, T, L, 3, 4)
        if prepend:
            model_args['rel_pos'] = torch.cat([null_rel_pos, rel_pos], dim=0)
        else:
            model_args['rel_pos'] = torch.cat([rel_pos, null_rel_pos], dim=0)

    if "plucker_embed" in _model_args:
        handled_keys.append("plucker_embed")
        plucker_embed = _model_args['plucker_embed']  # BxNC, T, 1, 4, 4
        null_plucker_embed = torch.zeros_like(plucker_embed)
        if prepend:
            model_args['plucker_embed'] = torch.cat([null_plucker_embed, plucker_embed], dim=0)
        else:
            model_args['plucker_embed'] = torch.cat([plucker_embed, null_plucker_embed], dim=0)


    if "seg_map" in _model_args:
        handled_keys.append("seg_map")
        plucker_embed = _model_args['seg_map']  # BxNC, T, 1, 4, 4
        null_plucker_embed = torch.zeros_like(plucker_embed)
        if prepend:
            model_args['seg_map'] = torch.cat([null_plucker_embed, plucker_embed], dim=0)
        else:
            model_args['seg_map'] = torch.cat([plucker_embed, null_plucker_embed], dim=0)



    if "depth_map" in _model_args:
        handled_keys.append("depth_map")
        plucker_embed = _model_args['depth_map']  # BxNC, T, 1, 4, 4
        null_plucker_embed = torch.zeros_like(plucker_embed)
        if prepend:
            model_args['depth_map'] = torch.cat([null_plucker_embed, plucker_embed], dim=0)
        else:
            model_args['depth_map'] = torch.cat([plucker_embed, null_plucker_embed], dim=0)
            
            
    if "occ" in _model_args:
        handled_keys.append("occ")
        plucker_embed = _model_args['occ']  # BxNC, T, 1, 4, 4
        null_plucker_embed = torch.zeros_like(plucker_embed)
        if prepend:
            model_args['occ'] = torch.cat([null_plucker_embed, plucker_embed], dim=0)
        else:
            model_args['occ'] = torch.cat([plucker_embed, null_plucker_embed], dim=0)

            
    if "cond_frame_x" in _model_args:
        handled_keys.append("cond_frame_x")
        # cond_frame_x = _model_args["cond_frame_x"]
        model_args['cond_frame_x'] = _model_args["cond_frame_x"] #torch.cat([cond_frame_x, cond_frame_x], dim=0)


    for k in _model_args.keys():
        # if  _model_args['bbox'] == None and k=='bbox' : continue
        if _model_args[k] == None:
            print(f"handle key={k}")
            continue
        if k in handled_keys:
            continue
        elif k in unchanged_keys:
            model_args[k] = _model_args[k]
        else:
            # print(f"handle key={k}")
            model_args[k] = repeat(_model_args[k], "b ... -> (2 b) ...")
    return model_args