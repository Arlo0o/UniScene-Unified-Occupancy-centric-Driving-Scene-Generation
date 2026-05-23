import numpy as np
from numba import njit, prange

@njit
def ray_voxel_intersection(grid, ray_direction, Xmin, Ymin, Zmin, vx, vy, vz, W, H, D):
    dx, dy, dz = ray_direction
    o_x, o_y, o_z = 0.0, 0.0, 0.0  # Ray origin at (0, 0, 0)

    # Compute initial voxel index for the ray origin
    x = int(np.floor((o_x - Xmin) / vx))
    y = int(np.floor((o_y - Ymin) / vy))
    z = int(np.floor((o_z - Zmin) / vz))

    # Initialize step and tDelta for x, y, z
    stepX = 1 if dx > 0 else -1 if dx < 0 else 0
    stepY = 1 if dy > 0 else -1 if dy < 0 else 0
    stepZ = 1 if dz > 0 else -1 if dz < 0 else 0

    tDeltaX = abs(vx / dx) if dx != 0 else np.inf
    tDeltaY = abs(vy / dy) if dy != 0 else np.inf
    tDeltaZ = abs(vz / dz) if dz != 0 else np.inf

    # Calculate initial tMax for each axis
    if dx > 0:
        voxel_boundary_x = Xmin + (x + 1) * vx
        tMaxX = (voxel_boundary_x - o_x) / dx
    elif dx < 0:
        voxel_boundary_x = Xmin + x * vx
        tMaxX = (voxel_boundary_x - o_x) / dx
    else:
        tMaxX = np.inf

    if dy > 0:
        voxel_boundary_y = Ymin + (y + 1) * vy
        tMaxY = (voxel_boundary_y - o_y) / dy
    elif dy < 0:
        voxel_boundary_y = Ymin + y * vy
        tMaxY = (voxel_boundary_y - o_y) / dy
    else:
        tMaxY = np.inf

    if dz > 0:
        voxel_boundary_z = Zmin + (z + 1) * vz
        tMaxZ = (voxel_boundary_z - o_z) / dz
    elif dz < 0:
        voxel_boundary_z = Zmin + z * vz
        tMaxZ = (voxel_boundary_z - o_z) / dz
    else:
        tMaxZ = np.inf

    # Perform ray marching through the grid
    while (0 <= x < W) and (0 <= y < H) and (0 <= z < D):
        if grid[x, y, z]:
            return True  # Intersection with occupied voxel

        # Move to the next voxel boundary
        if tMaxX < tMaxY:
            if tMaxX < tMaxZ:
                x += stepX
                tMaxX += tDeltaX
            else:
                z += stepZ
                tMaxZ += tDeltaZ
        else:
            if tMaxY < tMaxZ:
                y += stepY
                tMaxY += tDeltaY
            else:
                z += stepZ
                tMaxZ += tDeltaZ

    return False  # No intersection found

@njit(parallel=True)
def check_rays_cpu(grid, rays, Xmin, Ymin, Zmin, vx, vy, vz, W, H, D):
    N = rays.shape[0]
    #W, H, D = grid.shape
    results = np.zeros(N, dtype=np.bool_)
    for i in prange(N):
        ray_direction = rays[i]
        results[i] = ray_voxel_intersection(
            grid, ray_direction, Xmin, Ymin, Zmin, vx, vy, vz, W, H, D
        )
    return results


if __name__ == '__main__':
    # Example usage:
    # Define grid dimensions and voxel sizes
    W, H, D = 200, 200, 16  # Grid size in voxels
    vx, vy, vz = 0.5, 0.5, 0.5  # Voxel sizes in meters

    # Define grid boundaries
    Xmin, Ymin, Zmin = -50, -50, -5.0
    Xmax, Ymax, Zmax = 50.0, 50.0, 3.0

    # Create occupancy grid
    # grid = np.zeros((D, H, W), dtype=np.uint8)
    # Assume some occupied voxels
    grid = np.random.randint(0, 2, (W, H, D)).astype(np.bool_)

    # Define rays
    rays = np.random.rand(40000, 3)

    # Check intersections
    intersections = check_rays_cpu(grid.flatten(), rays, Xmin, Ymin, Zmin, vx, vy, vz, W, H, D)
    print(intersections)
