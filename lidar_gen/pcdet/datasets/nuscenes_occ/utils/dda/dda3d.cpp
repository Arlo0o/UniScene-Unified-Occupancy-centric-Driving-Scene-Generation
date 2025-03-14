#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


void dda3d_launcher(const float* rays, const bool* grid, bool* intersections,
    float Xmin, float Ymin, float Zmin, float vx, float vy, float vz,
    int W, int H, int D, int num_rays);


void raycast_launcher(const float* rays, const bool* grid, bool* intersections, int* hits,
    float Xmin, float Ymin, float Zmin, float vx, float vy, float vz,
    int W, int H, int D, int num_rays);

void dda3d_gpu(at::Tensor rays, at::Tensor grid, at::Tensor intersections, float Xmin, float Ymin, float Zmin, float vx, float vy, float vz, int W, int H, int D, int num_rays){
    CHECK_INPUT(rays);
    CHECK_INPUT(grid);
    CHECK_INPUT(intersections);

    const float* rays_ptr = rays.data_ptr<float>();
    const bool* grid_ptr = grid.data_ptr<bool>();
    bool* intersections_ptr = intersections.data_ptr<bool>();
    dda3d_launcher(rays_ptr, grid_ptr, intersections_ptr, Xmin, Ymin, Zmin, vx, vy, vz, W, H, D, num_rays);
}

void raycast_gpu(at::Tensor rays, at::Tensor grid, at::Tensor intersections, at::Tensor hits, float Xmin, float Ymin, float Zmin, float vx, float vy, float vz, int W, int H, int D, int num_rays){
    CHECK_INPUT(rays);
    CHECK_INPUT(grid);
    CHECK_INPUT(intersections);
    CHECK_INPUT(hits);

    const float* rays_ptr = rays.data_ptr<float>();
    const bool* grid_ptr = grid.data_ptr<bool>();
    bool* intersections_ptr = intersections.data_ptr<bool>();
    int* hits_ptr = hits.data_ptr<int>();
    raycast_launcher(rays_ptr, grid_ptr, intersections_ptr, hits_ptr, Xmin, Ymin, Zmin, vx, vy, vz, W, H, D, num_rays);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dda3d_gpu", &dda3d_gpu, "dda3d (CUDA)");
  m.def("raycast_gpu", &raycast_gpu, "raycast (CUDA)");
}