from typing import Tuple

import numpy as np
import torch
import gzip
from numba import njit

__all__ = ["voxelize", "voxelize_with_intensity", "voxelize_with_value"]

#DIVIDE By 255.0

def unscale_field(field, n_min, n_max):

    field = (field * (n_max - n_min)) + n_min

    return field
    

def scale_field(field, n_min, n_max):
    field = (field - n_min) / (n_max - n_min)
    field = np.clip(field, 0, 1.0)

    return field

def load_from_zip_file(filename: str):
    f = gzip.GzipFile(filename, "r")
    data = np.load(f)
    f.close()

    return data

def voxelize_with_value(
    coords: np.ndarray,
    intensity: np.ndarray,
    spatial_range: Tuple[float, float, float, float, float, float],
    voxel_size: Tuple[float, float, float],
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    spatial_range = np.array(spatial_range, dtype=coords.dtype)
    voxel_size = np.array(voxel_size, dtype=coords.dtype)

    coords = coords[:,:3]

    # Get coordinate min and max
    coords_min, coords_max = spatial_range[[0, 2, 4]], spatial_range[[1, 3, 5]]

    # Quantize coordinates
    coords = ((coords - coords_min) / voxel_size).astype(np.int32)
    coords, idx = np.unique(coords, axis=0, return_index=True)
    intensity = intensity[idx]

    # Create volume
    volume_size = np.ceil((coords_max - coords_min) / voxel_size).astype(np.int32)
    volume = np.zeros(volume_size.tolist(), dtype=dtype)
    intensity_volume = np.zeros(volume_size.tolist(), dtype=dtype)

    # Remove points outside the volume
    mask = np.all((coords[:] >= 0) & (coords[:] < volume_size[:]), axis=1)
    coords = coords[mask]
    intensity = intensity[mask]

    # Fill volume
    volume[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    intensity_volume[coords[:, 0], coords[:, 1], coords[:, 2]] = intensity
    #intensity_volume = intensity_volume.astype(np.float32) / 255.0
    return volume, intensity_volume


def voxelize_with_intensity(
    coords: np.ndarray,
    intensity: np.ndarray,
    spatial_range: Tuple[float, float, float, float, float, float],
    voxel_size: Tuple[float, float, float],
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    volume, intensity_volume = voxelize_with_value(coords, intensity, spatial_range, voxel_size, dtype)
    return volume, intensity_volume / 255.0

def voxelize(
    coords: np.ndarray,
    spatial_range: Tuple[float, float, float, float, float, float],
    voxel_size: Tuple[float, float, float],
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    spatial_range = np.array(spatial_range, dtype=coords.dtype)
    voxel_size = np.array(voxel_size, dtype=coords.dtype)

    # Get coordinate min and max
    coords_min, coords_max = spatial_range[[0, 2, 4]], spatial_range[[1, 3, 5]]

    # Quantize coordinates
    coords = ((coords - coords_min) / voxel_size).astype(np.int32)
    coords = np.unique(coords, axis=0)

    # Create volume
    volume_size = np.ceil((coords_max - coords_min) / voxel_size).astype(np.int32)
    volume = np.zeros(volume_size.tolist(), dtype=dtype)

    # Remove points outside the volume
    mask = np.all((coords[:] >= 0) & (coords[:] < volume_size[:]), axis=1)
    coords = coords[mask]

    # Fill volume
    volume[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return volume

#From Cross view transformers


def encode(x):
    """
    (h, w, c) np.uint8 {0, 255}
    """
    n = x.shape[2]

    # assert n < 16
    assert x.ndim == 3
    assert x.dtype == np.uint8
    assert all(x in [0, 255] for x in np.unique(x))

    shift = np.arange(n, dtype=np.int32)[None, None]

    binary = (x > 0)
    binary = (binary << shift).sum(-1)
    binary = binary.astype(np.int32)

    return binary


def decode(img, n):
    """
    returns (h, w, n) np.int32 {0, 1}
    """
    shift = np.arange(n, dtype=np.int32)[None, None]

    x = np.array(img)[..., None]
    x = (x >> shift) & 1

    return x


