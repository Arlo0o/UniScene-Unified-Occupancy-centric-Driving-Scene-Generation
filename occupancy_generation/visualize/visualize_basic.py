import numpy as np


SEMANTIC_COLORS = np.array(
    [
        [255, 120, 50, 255],
        [255, 192, 203, 255],
        [255, 255, 0, 255],
        [0, 150, 245, 255],
        [0, 255, 255, 255],
        [255, 127, 0, 255],
        [255, 0, 0, 255],
        [255, 240, 150, 255],
        [135, 60, 0, 255],
        [160, 32, 240, 255],
        [255, 0, 255, 255],
        [139, 137, 137, 255],
        [75, 0, 75, 255],
        [150, 240, 80, 255],
        [230, 230, 250, 255],
        [0, 175, 0, 255],
    ],
    dtype=np.float32,
) / 255.0


def _to_numpy(array):
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def get_grid_coords(dims, resolution):
    resolution = np.asarray(resolution, dtype=np.float32)
    if resolution.ndim == 0:
        resolution = np.repeat(resolution, 3)

    x = np.arange(dims[0], dtype=np.float32)
    y = np.arange(dims[1], dtype=np.float32)
    z = np.arange(dims[2], dtype=np.float32)
    coords = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    return coords * resolution + resolution / 2.0


def draw_return(
    voxels,
    pred_pts=None,
    vox_origin=(0.0, 0.0, 0.0),
    voxel_size=0.2,
    grid=None,
    pt_label=None,
    save_dir=None,
    cam_positions=None,
    focal_positions=None,
    timestamp=None,
    mode=0,
    sem=False,
):
    voxels = _to_numpy(voxels)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    if voxel_size.ndim == 0:
        voxel_size = np.repeat(voxel_size, 3)

    grid_coords = get_grid_coords(voxels.shape[:3], voxel_size)
    grid_coords += np.asarray(vox_origin, dtype=np.float32).reshape(1, 3)

    if mode == 0:
        labels = voxels.reshape(-1)
    elif mode in (1, 2):
        if grid is None:
            raise ValueError("grid must be provided when mode is 1 or 2")
        grid = _to_numpy(grid).astype(np.int64)
        flat_index = grid[:, 0] * voxels.shape[1] * voxels.shape[2] + grid[:, 1] * voxels.shape[2] + grid[:, 2]
        flat_index, point_index = np.unique(flat_index, return_index=True)
        grid_coords = grid_coords[flat_index]
        source_labels = pred_pts if mode == 1 else pt_label
        labels = _to_numpy(source_labels)[point_index].reshape(-1)
    else:
        raise NotImplementedError(f"Unsupported visualization mode: {mode}")

    occupied = (labels > 0) & (labels < 17)
    fov_voxels = np.concatenate([grid_coords[occupied], labels[occupied, None]], axis=1)
    color_ids = np.clip(fov_voxels[:, 3].astype(np.int64) - 1, 0, len(SEMANTIC_COLORS) - 1)
    return fov_voxels, SEMANTIC_COLORS[color_ids]


def figure_to_array(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.renderer.buffer_rgba())
