#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void launch_smooth_sampler_forward_kernel(
    const torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &grid,
    int64_t padding_mode, bool align_corners, bool apply_smoothstep);

void launch_bool_sampler_kernel(
    const torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &grid,
    int64_t padding_mode, bool align_corners, bool apply_smoothstep);

void launch_smooth_sampler_backward_kernel(
    const torch::Tensor& grad_input, const torch::Tensor &grad_grid,
    const torch::Tensor& grad_output, const torch::Tensor& input,
    const torch::Tensor& grid, int64_t padding_mode, bool align_corners,
    bool apply_smoothstep, bool input_requires_grad);

void launch_smooth_sampler_backward_backward_kernel(
    const torch::Tensor& grad_input,
    const torch::Tensor& grad_grid,
    const torch::Tensor& grad_grad_out,
    const torch::Tensor& input,
    const torch::Tensor& grid,
    const torch::Tensor& grad_out_input,
    const torch::Tensor& grad_out_grid,
    const torch::Tensor& grad_output,
    int64_t padding_mode,
    const bool align_corners,
    const bool apply_smoothstep,
    const bool input_requires_grad);

torch::Tensor smooth_sampler_forward(torch::Tensor input, torch::Tensor grid,
                                     int64_t padding_mode, bool align_corners, bool apply_smoothstep) {
  CHECK_INPUT(input)
  CHECK_INPUT(grid)
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = torch::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]},
      input.options());
  launch_smooth_sampler_forward_kernel(
      output, input, grid, padding_mode, align_corners, apply_smoothstep);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> smooth_sampler_backward(torch::Tensor grad_output, torch::Tensor input,
                                                                 torch::Tensor grid, int64_t padding_mode, bool align_corners,
                                                                 bool apply_smoothstep, bool input_requires_grad) {
  CHECK_INPUT(grad_output)
  CHECK_INPUT(input)
  CHECK_INPUT(grid)
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  auto grad_input = torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grid = torch::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_smooth_sampler_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, padding_mode, align_corners, apply_smoothstep, input_requires_grad);
  return std::make_tuple(grad_input, grad_grid);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> smooth_sampler_backward_backward(torch::Tensor grad_out_input, torch::Tensor grad_out_grid,
                                                                          torch::Tensor input, torch::Tensor grid, torch::Tensor grad_output,
                                                                          int64_t padding_mode, bool align_corners,
                                                                          bool apply_smoothstep, bool input_requires_grad) {
  CHECK_INPUT(grad_out_input)
  CHECK_INPUT(grad_out_grid)
  CHECK_INPUT(input)
  CHECK_INPUT(grid)
  CHECK_INPUT(grad_output)
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  
  auto grad_input = torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grid = torch::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grad_out = torch::zeros_like(grad_output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_smooth_sampler_backward_backward_kernel(grad_input, grad_grid, grad_grad_out, input, grid,
                                                 grad_out_input, grad_out_grid, grad_output,
                                                 padding_mode, align_corners, apply_smoothstep, input_requires_grad);
  return std::make_tuple(grad_input, grad_grid, grad_grad_out);
}

torch::Tensor bool_sample(torch::Tensor input, torch::Tensor grid,
                                     int64_t padding_mode, bool align_corners, bool apply_smoothstep) {
  CHECK_INPUT(input)
  CHECK_INPUT(grid)
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = torch::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]},
      input.options());
  launch_bool_sampler_kernel(
      output, input, grid, padding_mode, align_corners, apply_smoothstep);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &smooth_sampler_forward, "Smooth sampler forward (CUDA)");
  m.def("backward", &smooth_sampler_backward, "Smooth sampler backward (CUDA)");
  m.def("backward_backward", &smooth_sampler_backward_backward, "Smooth sampler backward backward (CUDA)");
  m.def("bool_sample", &bool_sample, "bool sample (CUDA)");
}