#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)

__global__ void dda3d_kernel(
    const float* rays, const bool* grid, bool* intersections,
    float Xmin, float Ymin, float Zmin, float vx, float vy, float vz,
    int W, int H, int D, int num_rays) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    // 获取当前射线的方向
    float dx = rays[i * 3 + 0];
    float dy = rays[i * 3 + 1];
    float dz = rays[i * 3 + 2];

    // 计算射线起点在占用网格中的索引
    int x = (0 - Xmin) / vx;
    int y = (0 - Ymin) / vy;
    int z = (0 - Zmin) / vz;

    int stepX = (dx > 0) ? 1 : -1;
    int stepY = (dy > 0) ? 1 : -1;
    int stepZ = (dz > 0) ? 1 : -1;

    float tDeltaX = (dx != 0) ? abs(vx / dx) : INFINITY;
    float tDeltaY = (dy != 0) ? abs(vy / dy) : INFINITY;
    float tDeltaZ = (dz != 0) ? abs(vz / dz) : INFINITY;

    float tMaxX = (dx > 0) ? (Xmin + (x + 1) * vx) / dx : (Xmin + x * vx) / dx;
    float tMaxY = (dy > 0) ? (Ymin + (y + 1) * vy) / dy : (Ymin + y * vy) / dy;
    float tMaxZ = (dz > 0) ? (Zmin + (z + 1) * vz) / dz : (Zmin + z * vz) / dz;

    // 遍历射线经过的体素
    while ((0 <= x && x < W) && (0 <= y && y < H) && (0 <= z && z < D)) {
        // 注意，grid大小为[W, H, D]
        int grid_index = z + y * D + x * H * D;

        // 如果体素被占用，设置结果为 true
        if (grid[grid_index]) {
            intersections[i] = true;
            return;
        }

        // 更新步进逻辑，选择最小的 tMax 并更新相应的索引
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                x += stepX;
                tMaxX += tDeltaX;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                y += stepY;
                tMaxY += tDeltaY;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        }
    }
    intersections[i] = false;  // 如果没有相交，结果为 false
}


__global__ void raycast_kernel(
    const float* rays, const bool* grid, bool* intersections, int* hits,
    float Xmin, float Ymin, float Zmin, float vx, float vy, float vz,
    int W, int H, int D, int num_rays) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    // 获取当前射线的方向
    float dx = rays[i * 3 + 0];
    float dy = rays[i * 3 + 1];
    float dz = rays[i * 3 + 2];

    // 计算射线起点在占用网格中的索引
    int x = (0 - Xmin) / vx;
    int y = (0 - Ymin) / vy;
    int z = (0 - Zmin) / vz;

    int stepX = (dx > 0) ? 1 : -1;
    int stepY = (dy > 0) ? 1 : -1;
    int stepZ = (dz > 0) ? 1 : -1;

    float tDeltaX = (dx != 0) ? abs(vx / dx) : INFINITY;
    float tDeltaY = (dy != 0) ? abs(vy / dy) : INFINITY;
    float tDeltaZ = (dz != 0) ? abs(vz / dz) : INFINITY;

    float tMaxX = (dx > 0) ? (Xmin + (x + 1) * vx) / dx : (Xmin + x * vx) / dx;
    float tMaxY = (dy > 0) ? (Ymin + (y + 1) * vy) / dy : (Ymin + y * vy) / dy;
    float tMaxZ = (dz > 0) ? (Zmin + (z + 1) * vz) / dz : (Zmin + z * vz) / dz;

    // 遍历射线经过的体素
    while ((0 <= x && x < W) && (0 <= y && y < H) && (0 <= z && z < D)) {
        // 注意，grid大小为[W, H, D]
        int grid_index = z + y * D + x * H * D;

        // 如果体素被占用，设置结果为 true
        if (grid[grid_index]) {
            intersections[i] = true;
            hits[i * 3] = x;
            hits[i * 3 + 1] = y;
            hits[i * 3 + 2] = z;
            return;
        }

        // 更新步进逻辑，选择最小的 tMax 并更新相应的索引
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                x += stepX;
                tMaxX += tDeltaX;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                y += stepY;
                tMaxY += tDeltaY;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        }
    }
    intersections[i] = false;  // 如果没有相交，结果为 false
}

__global__ void raycast_kernel_wori(
    const float* rays, const bool* grid, bool* intersections, int* hits, const float* origins,
    float Xmin, float Ymin, float Zmin, float vx, float vy, float vz,
    int W, int H, int D, int num_rays) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    // 获取当前射线的方向
    float dx = rays[i * 3 + 0];
    float dy = rays[i * 3 + 1];
    float dz = rays[i * 3 + 2];

    // 获取当前射线的原点
    float ox = origins[i * 3 + 0];
    float oy = origins[i * 3 + 1];
    float oz = origins[i * 3 + 2];

    // 计算射线起点在占用网格中的索引
    int x = (ox - Xmin) / vx;
    int y = (oy - Ymin) / vy;
    int z = (oz - Zmin) / vz;

    int stepX = (dx > 0) ? 1 : -1;
    int stepY = (dy > 0) ? 1 : -1;
    int stepZ = (dz > 0) ? 1 : -1;

    float tDeltaX = (dx != 0) ? abs(vx / dx) : INFINITY;
    float tDeltaY = (dy != 0) ? abs(vy / dy) : INFINITY;
    float tDeltaZ = (dz != 0) ? abs(vz / dz) : INFINITY;

    float tMaxX = (dx > 0) ? (Xmin + (x + 1) * vx) / dx : (Xmin + x * vx) / dx;
    float tMaxY = (dy > 0) ? (Ymin + (y + 1) * vy) / dy : (Ymin + y * vy) / dy;
    float tMaxZ = (dz > 0) ? (Zmin + (z + 1) * vz) / dz : (Zmin + z * vz) / dz;

    // 遍历射线经过的体素
    while ((0 <= x && x < W) && (0 <= y && y < H) && (0 <= z && z < D)) {
        // 注意，grid大小为[W, H, D]
        int grid_index = z + y * D + x * H * D;

        // 如果体素被占用，设置结果为 true
        if (grid[grid_index]) {
            intersections[i] = true;
            hits[i * 3] = x;
            hits[i * 3 + 1] = y;
            hits[i * 3 + 2] = z;
            return;
        }

        // 更新步进逻辑，选择最小的 tMax 并更新相应的索引
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                x += stepX;
                tMaxX += tDeltaX;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                y += stepY;
                tMaxY += tDeltaY;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        }
    }
    intersections[i] = false;  // 如果没有相交，结果为 false
}


__device__ bool ray_box_intersection(
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    int x, int y, int z,
    float Xmin, float Ymin, float Zmin,
    float vx, float vy, float vz,
    float& t_hit
) {
    float cube_xmin = Xmin + x * vx;
    float cube_xmax = cube_xmin + vx;
    float cube_ymin = Ymin + y * vy;
    float cube_ymax = cube_ymin + vy;
    float cube_zmin = Zmin + z * vz;
    float cube_zmax = cube_zmin + vz;

    float inv_dx = 1.0 / (dx + 1e-8);
    float inv_dy = 1.0 / (dy + 1e-8);
    float inv_dz = 1.0 / (dz + 1e-8);

    float tx1 = (cube_xmin - ox) * inv_dx;
    float tx2 = (cube_xmax - ox) * inv_dx;
    float ty1 = (cube_ymin - oy) * inv_dy;
    float ty2 = (cube_ymax - oy) * inv_dy;
    float tz1 = (cube_zmin - oz) * inv_dz;
    float tz2 = (cube_zmax - oz) * inv_dz;

    float tmin = fmaxf(fmaxf(fminf(tx1, tx2), fminf(ty1, ty2)), fminf(tz1, tz2));
    float tmax = fminf(fminf(fmaxf(tx1, tx2), fmaxf(ty1, ty2)), fmaxf(tz1, tz2));

    if (tmax < 0 || tmin > tmax){
        return false;
    } 

    t_hit = tmin;
    return true;
}

__global__ void raycast_kernel_wori2(
    const float* rays, const bool* grid, bool* intersections, float* hits, const float* origins,
    float Xmin, float Ymin, float Zmin, float vx, float vy, float vz,
    int W, int H, int D, int num_rays) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    // 获取当前射线的方向
    float dx = rays[i * 3 + 0];
    float dy = rays[i * 3 + 1];
    float dz = rays[i * 3 + 2];

    // 获取当前射线的原点
    float ox = origins[i * 3 + 0];
    float oy = origins[i * 3 + 1];
    float oz = origins[i * 3 + 2];

    // 计算射线起点在占用网格中的索引
    int x = (ox - Xmin) / vx;
    int y = (oy - Ymin) / vy;
    int z = (oz - Zmin) / vz;

    int stepX = (dx > 0) ? 1 : -1;
    int stepY = (dy > 0) ? 1 : -1;
    int stepZ = (dz > 0) ? 1 : -1;

    float tDeltaX = (dx != 0) ? abs(vx / dx) : INFINITY;
    float tDeltaY = (dy != 0) ? abs(vy / dy) : INFINITY;
    float tDeltaZ = (dz != 0) ? abs(vz / dz) : INFINITY;

    float tMaxX = (dx > 0) ? (Xmin + (x + 1) * vx) / dx : (Xmin + x * vx) / dx;
    float tMaxY = (dy > 0) ? (Ymin + (y + 1) * vy) / dy : (Ymin + y * vy) / dy;
    float tMaxZ = (dz > 0) ? (Zmin + (z + 1) * vz) / dz : (Zmin + z * vz) / dz;

    // 遍历射线经过的体素
    while ((0 <= x && x < W) && (0 <= y && y < H) && (0 <= z && z < D)) {
        // 注意，grid大小为[W, H, D]
        int grid_index = z + y * D + x * H * D;

        // 如果体素被占用，设置结果为 true
        if (grid[grid_index]) {
            intersections[i] = true;
            float t_hit;
            if (ray_box_intersection(ox, oy, oz, dx, dy, dz, x, y, z, Xmin, Ymin, Zmin, vx, vy, vz, t_hit)) {
                hits[i] = t_hit;
            } else {
                hits[i] = 0.0f;
            }
            return;
        }

        // 更新步进逻辑，选择最小的 tMax 并更新相应的索引
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                x += stepX;
                tMaxX += tDeltaX;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                y += stepY;
                tMaxY += tDeltaY;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        }
    }
    intersections[i] = false;  // 如果没有相交，结果为 false
}

void dda3d_launcher(const float* rays, const bool* grid, bool* intersections,
    float Xmin, float Ymin, float Zmin, float vx, float vy, float vz,
    int W, int H, int D, int num_rays){
    dim3 blockSize(DIVUP(num_rays, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    dda3d_kernel<<<blockSize, threadSize>>>(rays, grid, intersections, Xmin, Ymin, Zmin, vx, vy, vz, W, H, D, num_rays);
}

void raycast_launcher(const float* rays, const bool* grid, bool* intersections, int* hits,
    float Xmin, float Ymin, float Zmin, float vx, float vy, float vz,
    int W, int H, int D, int num_rays){
    dim3 blockSize(DIVUP(num_rays, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    raycast_kernel<<<blockSize, threadSize>>>(rays, grid, intersections, hits, Xmin, Ymin, Zmin, vx, vy, vz, W, H, D, num_rays);
}

void raycast_launcher_wori(const float* rays, const bool* grid, bool* intersections, int* hits, const float* origins,
    float Xmin, float Ymin, float Zmin, float vx, float vy, float vz,
    int W, int H, int D, int num_rays){
    dim3 blockSize(DIVUP(num_rays, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    raycast_kernel_wori<<<blockSize, threadSize>>>(rays, grid, intersections, hits, origins, Xmin, Ymin, Zmin, vx, vy, vz, W, H, D, num_rays);
}

void raycast_launcher_wori_2(const float* rays, const bool* grid, bool* intersections, float* hits, const float* origins,
    float Xmin, float Ymin, float Zmin, float vx, float vy, float vz,
    int W, int H, int D, int num_rays){
    dim3 blockSize(DIVUP(num_rays, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    raycast_kernel_wori2<<<blockSize, threadSize>>>(rays, grid, intersections, hits, origins, Xmin, Ymin, Zmin, vx, vy, vz, W, H, D, num_rays);
}