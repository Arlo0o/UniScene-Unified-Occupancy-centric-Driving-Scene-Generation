import torch


def _voxelize_gpu(batch_pts, spatial_range, voxel_size, rotations=None, flip_vert=False):
    # spatial_range = SPATIAL_RANGE
    # voxel_size = VOXEL_SIZE
    dtype = torch.float32
    spatial_range = torch.tensor(spatial_range, dtype=batch_pts.dtype).cuda()
    voxel_size = torch.tensor(voxel_size, dtype=batch_pts.dtype).cuda()

    # Get coordinate min and max
    coords_min, coords_max = spatial_range[[0, 2, 4]], spatial_range[[1, 3, 5]]

    # Quantize coordinates
    batch_size = int(batch_pts[-1, 0]) + 1
    results = []
    for bs in range(batch_size):
        batch_mask = batch_pts[:, 0].int()==bs
        pts = batch_pts[batch_mask][:, 1:]
        coords = ((pts - coords_min) / voxel_size).to(torch.int32)
        coords = torch.unique(coords, dim=0)

        # Create volume
        volume_size = torch.ceil((coords_max - coords_min) / voxel_size).to(torch.int32)
        volume = torch.zeros(volume_size.tolist(), dtype=dtype, device=coords.device)

        # Remove points outside the volume
        mask = torch.all((coords[:] >= 0) & (coords[:] < volume_size[:]), dim=1)
        coords = coords[mask].long()

        # Fill volume
        volume[coords[:, 0], coords[:, 1], coords[:, 2]] = 1

        voxelized = volume
        # voxelized = voxelize(pts, SPATIAL_RANGE, VOXEL_SIZE)
        voxelized = voxelized.permute((2, 0, 1))
        # voxelized = np.transpose(voxelized, (2, 0, 1))

        if(rotations is not None):
            voxelized = torch.rot90(voxelized, k=rotations, dims=(1,2)).clone()

        if(flip_vert):
            voxelized = torch.flip(voxelized, dims=2).clone()

        results.append(voxelized)
    return results