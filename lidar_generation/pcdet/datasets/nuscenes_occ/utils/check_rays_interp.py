import numpy as np
import torch
import math

def _intersect_with_aabb(
    rays_o, rays_d, aabb
):
    """Returns collection of valid rays within a specified near/far bounding box along with a mask
    specifying which rays are valid

    Args:
        rays_o: (num_rays, 3) ray origins, scaled
        rays_d: (num_rays, 3) ray directions
        aabb: (6, ) This is [min point (x,y,z), max point (x,y,z)], scaled
    """
    # avoid divide by zero
    dir_fraction = 1.0 / (rays_d + 1e-6)

    # x
    t1 = (aabb[0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
    t2 = (aabb[3] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
    # y
    t3 = (aabb[1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
    t4 = (aabb[4] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
    # z
    t5 = (aabb[2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
    t6 = (aabb[5] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]

    nears = torch.max(
        torch.cat([torch.minimum(t1, t2), torch.minimum(t3, t4), torch.minimum(t5, t6)], dim=1), dim=1
    ).values
    fars = torch.min(
        torch.cat([torch.maximum(t1, t2), torch.maximum(t3, t4), torch.maximum(t5, t6)], dim=1), dim=1
    ).values

    # clamp to near plane
    nears = torch.clamp(nears, min=0.0)
    assert torch.all(nears < fars), "not collide with scene box"
    fars = torch.maximum(fars, nears + 1e-6)

    return nears, fars

def check_rays_interp(grid, rays, Xmin, Ymin, Zmin, Xmax, Ymax, Zmax, vx, vy, vz, W, H, D):
    rays_o = torch.zeros_like(rays)
    nears, fars = _intersect_with_aabb(rays_o, rays, [Xmin, Ymin, Zmin, Xmax, Ymax, Zmax])
    num_rays = rays.shape[0]
    num_samples = int(2*math.ceil(math.sqrt(max(Xmax**2+Ymax**2+Zmax**2, Xmin**2+Ymin**2+Zmin**2))))
    bins = torch.linspace(0.0, 1.0, num_samples + 1).to(grid.device)
    bins.expand(size=(num_rays, -1))
    ts = bins * fars.unsqueeze(-1)
    sample_pts = rays_o[:, None, :] + ts[:, :, None] * rays[:, None, :]
    pc_range = torch.tensor([Xmin, Ymin, Zmin, Xmax, Ymax, Zmax], dtype=torch.float32, device=grid.device)
    sample_pts = (sample_pts - pc_range[:3]) / (pc_range[3:] - pc_range[:3])
    sample_pts = sample_pts * 2 - 1
    sample_pts = sample_pts.flip(-1)
    res = torch.nn.functional.grid_sample(grid[None, None, :, :, :].to(torch.float32), sample_pts[None, None, :, :, :].to(torch.float32), mode='nearest', align_corners=False).squeeze()
    return res.any(-1)


if __name__ == '__main__':
    # Example usage:
    # Define grid dimensions and voxel sizes
    W, H, D = 200, 200, 16  # Grid size in voxels
    vx, vy, vz = 0.5, 0.5, 0.5  # Voxel sizes in meters

    # Define grid boundaries
    Xmin, Ymin, Zmin = -50.0, -50.0, -5.0
    Xmax, Ymax, Zmax = 50.0, 50.0, 3.0

    # Create occupancy grid
    # grid = np.zeros((W, H, D), dtype=np.uint8)
    # Assume some occupied voxels
    grid = np.load('/code/OpenPCDet/bd00c003806e418d88c4315d17eb09b7.npy').astype(bool)
    # grid = np.random.randint(0, 2, (W, H, D)).astype(np.bool_)
    
    # Define rays
    rays = np.random.rand(100000, 3)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    rays_o = np.zeros_like(rays)
    
    check_rays_interp(torch.from_numpy(grid).cuda(), torch.from_numpy(rays).cuda(), Xmin, Ymin, Zmin, Xmax, Ymax, Zmax, vx, vy, vz, W, H, D)