import cv2
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np


def apply_depth_colormap(gray, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    Input:
        gray: gray image, tensor/numpy, (H, W)
    Output:
        depth: (3, H, W), tensor
    """
    if type(gray) is not np.ndarray:
        gray = gray.detach().cpu().numpy().astype(np.float32)
    gray = gray.squeeze()
    assert len(gray.shape) == 2
    x = np.nan_to_num(gray)  # change nan to 0
    if minmax is None:
        mi = np.min(x)  # get minimum positive value
        ma = np.max(x)
    else:
        mi, ma = minmax
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    # TODO
    x = 1 - x  # reverse the colormap
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(cv2.applyColorMap(x, cmap))
    x = T.ToTensor()(x)  # (3, H, W)
    return x


def apply_semantic_colormap(semantic):
    """
    Input:
        semantic: semantic image, tensor/numpy, (N, H, W)
    Output:
        depth: (3, H, W), tensor
    """

    color_id = np.zeros((20, 3), dtype=np.uint8)
    color_id[0, :] = [255, 120, 50]
    color_id[1, :] = [255, 192, 203]
    color_id[2, :] = [255, 255, 0]
    color_id[3, :] = [0, 150, 245]
    color_id[4, :] = [0, 255, 255]
    color_id[5, :] = [255, 127, 0]
    color_id[6, :] = [255, 0, 0]
    color_id[7, :] = [255, 240, 150]
    color_id[8, :] = [135, 60, 0]
    color_id[9, :] = [160, 32, 240]
    color_id[10, :] = [255, 0, 255]
    color_id[11, :] = [139, 137, 137]
    color_id[12, :] = [75, 0, 75]
    color_id[13, :] = [150, 240, 80]
    color_id[14, :] = [230, 230, 250]
    color_id[15, :] = [0, 175, 0]
    color_id[16, :] = [0, 255, 127]
    color_id[17, :] = [222, 155, 161]
    color_id[18, :] = [140, 62, 69]
    color_id[19, :] = [227, 164, 30]

    if semantic.shape[0] != 1:
        semantic = torch.max(semantic, dim=0)[1].squeeze()
    else:
        semantic = semantic.squeeze()

    x = torch.zeros((3, semantic.shape[0], semantic.shape[1]), dtype=torch.float)
    for i in range(20):
        x[0][semantic == i] = color_id[i][0]
        x[1][semantic == i] = color_id[i][1]
        x[2][semantic == i] = color_id[i][2]

    return x / 255.0
