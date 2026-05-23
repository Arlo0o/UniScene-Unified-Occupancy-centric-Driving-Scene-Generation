import os
import re
import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torchvision.io import write_video
from torchvision.utils import save_image
from copy import deepcopy
from . import video_transforms
from uniscenev2.mmdet_plugin.core.bbox import LiDARInstance3DBoxes

IMG_FPS = 120
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

regex = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def is_img(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in IMG_EXTENSIONS


def is_vid(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in VID_EXTENSIONS


def is_url(url):
    return re.match(regex, url) is not None


def read_file(input_path):
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {input_path}")


def download_url(input_path):
    output_dir = "cache"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, base_name)
    img_data = requests.get(input_path).content
    with open(output_path, "wb") as handler:
        handler.write(img_data)
    print(f"URL {input_path} downloaded to {output_path}")
    return output_path


def temporal_random_crop(vframes, num_frames, frame_interval):
    temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
    total_frames = len(vframes)
    start_frame_ind, end_frame_ind = temporal_sample(total_frames)
    assert (
        end_frame_ind - start_frame_ind >= num_frames
    ), f"Not enough frames to sample, {end_frame_ind} - {start_frame_ind} < {num_frames}"
    frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, num_frames, dtype=int)
    video = vframes[frame_indice]
    return video


def get_transforms_video(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "image_size must be square for center crop"
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                # video_transforms.RandomHorizontalFlipVideo(),
                video_transforms.UCFCenterCropVideo(image_size[0]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop":
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.ResizeCrop(image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform_video


def get_transforms_image(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "Image size must be square for center crop"
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size[0])),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop":
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: resize_crop_to_fill(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform


def read_image_from_path(path, transform=None, transform_name="center", num_frames=1, image_size=(256, 256)):
    image = pil_loader(path)
    if transform is None:
        transform = get_transforms_image(image_size=image_size, name=transform_name)
    image = transform(image)
    video = image.unsqueeze(0).repeat(num_frames, 1, 1, 1)
    video = video.permute(1, 0, 2, 3)
    return video


def read_video_from_path(path, transform=None, transform_name="center", image_size=(256, 256)):
    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
    if transform is None:
        transform = get_transforms_video(image_size=image_size, name=transform_name)
    video = transform(vframes)  # T C H W
    video = video.permute(1, 0, 2, 3)
    return video


def read_from_path(path, image_size, transform_name="center"):
    if is_url(path):
        path = download_url(path)
    ext = os.path.splitext(path)[-1].lower()
    if ext.lower() in VID_EXTENSIONS:
        return read_video_from_path(path, image_size=image_size, transform_name=transform_name)
    else:
        assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
        return read_image_from_path(path, image_size=image_size, transform_name=transform_name)


def save_sample(x, save_path=None, fps=8, normalize=True, value_range=(-1, 1), force_video=False, high_quality=False, verbose=True, with_postfix=True, force_image=False, save_per_n_frame=-1, apply_color=False):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    x_ini = x
    assert x.ndim == 4, f"Input dim is {x.ndim}/{x.shape}"
    x = x.to("cpu")
    if with_postfix:
        save_path += f"_{x.shape[-2]}x{x.shape[-1]}"

    if not force_video and x.shape[1] == 1:  # T = 1: save as image
        if not is_img(save_path):
            save_path += ".png"
        x = x.squeeze(1)
        save_image([x], save_path, normalize=normalize, value_range=value_range)
       
    else:
        if with_postfix:
            save_path += f"_f{x.shape[1]}_fps{fps}.mp4"
        elif not is_vid(save_path):
            save_path += ".mp4"
        
        if normalize:
            low, high = value_range
            x.clamp_(min=low, max=high)
            x.sub_(low).div_(max(high - low, 1e-5))
        
    
        x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to(torch.uint8) ## T H W C
        
        if apply_color:
            imgList= []
            for xi in x.numpy():
                xi = (( (xi-xi.min()) /xi.max())*255).astype(np.uint8)
                mask = xi == 0
                xi = cv2.applyColorMap(xi, cv2.COLORMAP_JET)
                xi = cv2.convertScaleAbs(xi, alpha=1.5, beta=0)
                xi[mask] = 0
                imgList.append(xi)
            
        else:
            imgList = [ xi for xi in x.numpy() ]
        if force_image:
            os.makedirs(save_path)
            for xi, _x in enumerate(imgList):
                _save_path = os.path.join(save_path, f"f{xi:05d}.png")
                Image.fromarray(_x).save(_save_path)
        elif high_quality:
            from moviepy.editor import ImageSequenceClip
            if save_per_n_frame > 0 and len(imgList) > save_per_n_frame:
                single_value = len(imgList) % 2
                for i in range(0, len(imgList) - single_value, save_per_n_frame):
                    if i == 0:
                        _save_path = f"_f0-{save_per_n_frame + 1}".join(os.path.splitext(save_path))
                        vid_len = save_per_n_frame + single_value
                    else:
                        vid_len = save_per_n_frame
                        _save_path = f"_f{i + 1}-{i + 1 + vid_len}".join(os.path.splitext(save_path))
                        i += single_value
                    if len(imgList[i:i+vid_len]) < vid_len:
                        logging.warning(f"{len(imgList)} will stop at frame {i}.")
                        break
                    clip = ImageSequenceClip(imgList[i:i+vid_len], fps=fps)
                    clip.write_videofile(
                        _save_path, verbose=verbose, bitrate="4M",
                        logger='bar' if verbose else None)
                    clip.close()
            else:
                clip = ImageSequenceClip(imgList, fps=fps)
                clip.write_videofile(
                    save_path, verbose=verbose, bitrate="4M",
                    logger='bar' if verbose else None)
                clip.close()
        else:
            if torch.distributed.get_rank() == 0:
                write_video(save_path, x, fps=fps, video_codec="h264")
    if verbose:
        print(f"Saved to {save_path}")
    del x, x_ini
    return save_path


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def resize_crop_to_fill(pil_image, image_size):
    w, h = pil_image.size  # PIL is (W, H)
    th, tw = image_size
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = 0
        j = int(round((sw - tw) / 2.0))
    else:
        sh, sw = round(h * rw), tw
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = int(round((sh - th) / 2.0))
        j = 0
    arr = np.array(image)
    assert i + th <= arr.shape[0] and j + tw <= arr.shape[1]
    return Image.fromarray(arr[i : i + th, j : j + tw])

def box_center_shift(bboxes: LiDARInstance3DBoxes, new_center):
    raw_data = bboxes.tensor.numpy()
    new_bboxes = LiDARInstance3DBoxes(
        raw_data, box_dim=raw_data.shape[-1], origin=new_center)
    return new_bboxes

def trans_boxes_to_view(bboxes, transform, aug_matrix=None, proj=True):
    """2d projection with given transformation.

    Args:
        bboxes (LiDARInstance3DBoxes): bboxes
        transform (np.array): 4x4 matrix
        aug_matrix (np.array, optional): 4x4 matrix. Defaults to None.

    Returns:
        np.array: (N, 8, 3) normlized, where z = 1 or -1
    """
    if len(bboxes) == 0:
        return None

    bboxes_trans = box_center_shift(bboxes, (0.5, 0.5, 0.5))
    trans = transform
    if aug_matrix is not None:
        aug = aug_matrix
        trans = aug @ trans
    corners = bboxes_trans.corners
    num_bboxes = corners.shape[0]

    coords = np.concatenate(
        [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
    )
    trans = deepcopy(trans).reshape(4, 4)
    coords = coords @ trans.T

    coords = coords.reshape(-1, 4)
    # we do not filter > 0, need to keep sign of z
    if proj:
        z = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= z
        coords[:, 1] /= z
        coords[:, 2] /= np.abs(coords[:, 2])

    coords = coords[..., :3].reshape(-1, 8, 3)
    return coords


def trans_boxes_to_views(bboxes, transforms, aug_matrixes=None, proj=True):
    """This is a wrapper to perform projection on different `transforms`.

    Args:
        bboxes (LiDARInstance3DBoxes): bboxes
        transforms (List[np.arrray]): each is 4x4.
        aug_matrixes (List[np.array], optional): each is 4x4. Defaults to None.

    Returns:
        List[np.array]: each is Nx8x3, where z always equals to 1 or -1
    """
    if len(bboxes) == 0:
        return None

    coords = []
    for idx in range(len(transforms)):
        if aug_matrixes is not None:
            aug_matrix = aug_matrixes[idx]
        else:
            aug_matrix = None
        coords.append(
            trans_boxes_to_view(bboxes, transforms[idx], aug_matrix, proj))
    return coords
