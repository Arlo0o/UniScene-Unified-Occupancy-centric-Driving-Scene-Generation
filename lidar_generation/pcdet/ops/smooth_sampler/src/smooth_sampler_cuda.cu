#include <torch/extension.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>

using namespace at::cuda::detail;
using at::native::detail::GridSamplerPadding;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int64_t N, const int max_threads_per_block=CUDA_NUM_THREADS) {
  TORCH_INTERNAL_ASSERT(N > 0, "CUDA kernel launch blocks must be positive, but got N=", N);
  constexpr int64_t max_int = std::numeric_limits<int>::max();

  // Round up division for positive number that cannot cause integer overflow
  auto block_num = (N - 1) / max_threads_per_block + 1;
  TORCH_INTERNAL_ASSERT(block_num <= max_int, "Can't schedule too many blocks on CUDA device");

  return static_cast<int>(block_num);
}

template<typename scalar_t, typename index_t>
static __forceinline__ __device__
void safe_add_3d(scalar_t *data, int d, int h, int w,
                 int sD, int sH, int sW, int D, int H, int W,
                 scalar_t delta,
                 const index_t NC_offset,
                 const index_t memory_span) {
  if (at::native::within_bounds_3d(d, h, w, D, H, W)) {
    at::native::fastAtomicAdd(data,
                  NC_offset + d * sD + h * sH + w * sW,
                  memory_span,
                  delta,
                  true);
  }
}

// like at::native::safe_add_3d but without bound check
template<typename scalar_t, typename index_t>
static __forceinline__ __device__
void add_3d(scalar_t *data, int d, int h, int w,
            int sD, int sH, int sW,
            scalar_t delta,
            const index_t NC_offset_inp,
            const index_t memory_span) {
  at::native::fastAtomicAdd(data,
                NC_offset_inp + d * sD + h * sH + w * sW,
                memory_span,
                delta,
                true);
}

__device__ inline float smoothstep(float val) {
	return val * val * (3.0f - 2.0f * val);
}

__device__ inline float smoothstep_derivative(float val) {
	return 6 * val * (1.0f - val);
}

__device__ inline float smoothstep_2nd_derivative(float val) {
	return 6.0f - 12.0f * val;
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void smooth_sampler_kernel(
    const index_t nthreads,
    TensorInfo<scalar_t, index_t> input,
    TensorInfo<scalar_t, index_t> grid,
    TensorInfo<scalar_t, index_t> output,
    const GridSamplerPadding padding_mode,
    bool align_corners,
    bool apply_smoothstep) {

  index_t C = input.sizes[1];
  index_t inp_D = input.sizes[2];
  index_t inp_H = input.sizes[3];
  index_t inp_W = input.sizes[4];
  index_t out_D = grid.sizes[1];
  index_t out_H = grid.sizes[2];
  index_t out_W = grid.sizes[3];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sD = input.strides[2];
  index_t inp_sH = input.strides[3];
  index_t inp_sW = input.strides[4];
  index_t grid_sN = grid.strides[0];
  index_t grid_sD = grid.strides[1];
  index_t grid_sH = grid.strides[2];
  index_t grid_sW = grid.strides[3];
  index_t grid_sCoor = grid.strides[4];
  index_t out_sN = output.strides[0];
  index_t out_sC = output.strides[1];
  index_t out_sD = output.strides[2];
  index_t out_sH = output.strides[3];
  index_t out_sW = output.strides[4];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % out_W;
    const index_t h = (index / out_W) % out_H;
    const index_t d = (index / (out_H * out_W)) % out_D;
    const index_t n = index / (out_D * out_H * out_W);
    const index_t grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y, z co-ordinates from grid
    scalar_t ix = grid.data[grid_offset];
    scalar_t iy = grid.data[grid_offset + grid_sCoor];
    scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

    ix = at::native::grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
    iy = at::native::grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);
    iz = at::native::grid_sampler_compute_source_index(iz, inp_D, padding_mode, align_corners);

    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    index_t _ix = static_cast<index_t>(::floor(ix));
    index_t _iy = static_cast<index_t>(::floor(iy));
    index_t _iz = static_cast<index_t>(::floor(iz));
    index_t ix_ = _ix + 1;
    index_t iy_ = _iy + 1;
    index_t iz_ = _iz + 1;

    scalar_t pos_x_ = ix - _ix;
    scalar_t pos_y_ = iy - _iy;
    scalar_t pos_z_ = iz - _iz;

    if (apply_smoothstep) {
      pos_x_ = smoothstep(pos_x_);
      pos_y_ = smoothstep(pos_y_);
      pos_z_ = smoothstep(pos_z_);
    }

    scalar_t pos_x = 1.0f - pos_x_;
    scalar_t pos_y = 1.0f - pos_y_;
    scalar_t pos_z = 1.0f - pos_z_;

    // get surfaces to each neighbor:
    scalar_t tnw = pos_x  * pos_y  * pos_z;
    scalar_t tne = pos_x_ * pos_y  * pos_z;
    scalar_t tsw = pos_x  * pos_y_ * pos_z;
    scalar_t tse = pos_x_ * pos_y_ * pos_z;
    scalar_t bnw = pos_x  * pos_y  * pos_z_;
    scalar_t bne = pos_x_ * pos_y  * pos_z_;
    scalar_t bsw = pos_x  * pos_y_ * pos_z_;
    scalar_t bse = pos_x_ * pos_y_ * pos_z_;

    auto inp_ptr_NC = input.data + n * inp_sN;
    auto out_ptr_NCDHW = output.data + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
    for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
      *out_ptr_NCDHW = static_cast<scalar_t>(0);
      if (at::native::within_bounds_3d(_iz, _iy, _ix, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + _ix * inp_sW] * tnw;
      }
      if (at::native::within_bounds_3d(_iz, _iy, ix_, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + ix_ * inp_sW] * tne;
      }
      if (at::native::within_bounds_3d(_iz, iy_, _ix, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + _ix * inp_sW] * tsw;
      }
      if (at::native::within_bounds_3d(_iz, iy_, ix_, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + ix_ * inp_sW] * tse;
      }
      if (at::native::within_bounds_3d(iz_, _iy, _ix, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + _ix * inp_sW] * bnw;
      }
      if (at::native::within_bounds_3d(iz_, _iy, ix_, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + ix_ * inp_sW] * bne;
      }
      if (at::native::within_bounds_3d(iz_, iy_, _ix, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + _ix * inp_sW] * bsw;
      }
      if (at::native::within_bounds_3d(iz_, iy_, ix_, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + ix_ * inp_sW] * bse;
      }
    }
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void smooth_sampler_backward_kernel(
    const index_t nthreads,
    TensorInfo<scalar_t, index_t> grad_output,
    TensorInfo<scalar_t, index_t> input,
    TensorInfo<scalar_t, index_t> grid,
    TensorInfo<scalar_t, index_t> grad_input,  // initialized to zeros (or unused if input_requires_grad is false)
    TensorInfo<scalar_t, index_t> grad_grid,   // initialized to empty
    const GridSamplerPadding padding_mode,
    bool align_corners,
    bool apply_smoothstep,
    const index_t grad_input_memory_span,
    const bool input_requires_grad) {

  index_t C = input.sizes[1];
  index_t inp_D = input.sizes[2];
  index_t inp_H = input.sizes[3];
  index_t inp_W = input.sizes[4];
  index_t out_D = grid.sizes[1];
  index_t out_H = grid.sizes[2];
  index_t out_W = grid.sizes[3];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sD = input.strides[2];
  index_t inp_sH = input.strides[3];
  index_t inp_sW = input.strides[4];
  index_t grid_sN = grid.strides[0];
  index_t grid_sD = grid.strides[1];
  index_t grid_sH = grid.strides[2];
  index_t grid_sW = grid.strides[3];
  index_t grid_sCoor = grid.strides[4];
  index_t gOut_sN = grad_output.strides[0];
  index_t gOut_sC = grad_output.strides[1];
  index_t gOut_sD = grad_output.strides[2];
  index_t gOut_sH = grad_output.strides[3];
  index_t gOut_sW = grad_output.strides[4];
  // gInp_* (and NC_offset_inp below) are not really needed if input_requires_grad is false.
  int64_t gInp_sN = 0;
  int64_t gInp_sC = 0;
  int64_t gInp_sD = 0;
  int64_t gInp_sH = 0;
  int64_t gInp_sW = 0;
  if (input_requires_grad) {
    gInp_sN = grad_input.strides[0];
    gInp_sC = grad_input.strides[1];
    gInp_sD = grad_input.strides[2];
    gInp_sH = grad_input.strides[3];
    gInp_sW = grad_input.strides[4];
  }
  index_t gGrid_sW = grad_grid.strides[3];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % out_W;
    const index_t h = (index / out_W) % out_H;
    const index_t d = (index / (out_H * out_W)) % out_D;
    const index_t n = index / (out_D * out_H * out_W);
    const auto grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y, z co-ordinates from grid
    scalar_t ix = grid.data[grid_offset];
    scalar_t iy = grid.data[grid_offset + grid_sCoor];
    scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

    // multipliers for gradients on ix, iy, and iz
    scalar_t dL_dix_mult, dL_diy_mult, dL_diz_mult;
    ix = at::native::grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &dL_dix_mult);
    iy = at::native::grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &dL_diy_mult);
    iz = at::native::grid_sampler_compute_source_index_set_grad(iz, inp_D, padding_mode, align_corners, &dL_diz_mult);

    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    index_t _ix = static_cast<index_t>(::floor(ix));
    index_t _iy = static_cast<index_t>(::floor(iy));
    index_t _iz = static_cast<index_t>(::floor(iz));
    index_t ix_ = _ix + 1;
    index_t iy_ = _iy + 1;
    index_t iz_ = _iz + 1;

    scalar_t pos_x_ = ix - _ix;
    scalar_t pos_y_ = iy - _iy;
    scalar_t pos_z_ = iz - _iz;

    float pos_x_derivative = 1.0f;
    float pos_y_derivative = 1.0f;
    float pos_z_derivative = 1.0f;

    if (apply_smoothstep) {
      pos_x_derivative = smoothstep_derivative(pos_x_);
      pos_y_derivative = smoothstep_derivative(pos_y_);
      pos_z_derivative = smoothstep_derivative(pos_z_);
      pos_x_ = smoothstep(pos_x_);
      pos_y_ = smoothstep(pos_y_);
      pos_z_ = smoothstep(pos_z_);
    }

    scalar_t pos_x = 1.0f - pos_x_;
    scalar_t pos_y = 1.0f - pos_y_;
    scalar_t pos_z = 1.0f - pos_z_;

    // get surfaces to each neighbor:
    scalar_t tnw = pos_x  * pos_y  * pos_z;
    scalar_t tne = pos_x_ * pos_y  * pos_z;
    scalar_t tsw = pos_x  * pos_y_ * pos_z;
    scalar_t tse = pos_x_ * pos_y_ * pos_z;
    scalar_t bnw = pos_x  * pos_y  * pos_z_;
    scalar_t bne = pos_x_ * pos_y  * pos_z_;
    scalar_t bsw = pos_x  * pos_y_ * pos_z_;
    scalar_t bse = pos_x_ * pos_y_ * pos_z_;

    scalar_t dL_dix = static_cast<scalar_t>(0), dL_diy = static_cast<scalar_t>(0), dL_diz = static_cast<scalar_t>(0);
    scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
    index_t NC_offset_inp;
    if (input_requires_grad) {
      NC_offset_inp = n * gInp_sN;
    }
    scalar_t *inp_ptr_NC = input.data + n * inp_sN;
    // calculate bilinear weighted pixel value and set output pixel
    for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, NC_offset_inp += gInp_sC, inp_ptr_NC += inp_sC) {
      scalar_t gOut = *gOut_ptr_NCDHW;

      // calculate and set grad_input. See Note [Passing pointer and offset to at::native::fastAtomicAdd].
      if (input_requires_grad) {
        safe_add_3d(grad_input.data, _iz, _iy, _ix, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw * gOut,
                    NC_offset_inp, grad_input_memory_span);
        safe_add_3d(grad_input.data, _iz, _iy, ix_, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne * gOut,
                    NC_offset_inp, grad_input_memory_span);
        safe_add_3d(grad_input.data, _iz, iy_, _ix, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw * gOut,
                    NC_offset_inp, grad_input_memory_span);
        safe_add_3d(grad_input.data, _iz, iy_, ix_, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse * gOut,
                    NC_offset_inp, grad_input_memory_span);
        safe_add_3d(grad_input.data, iz_, _iy, _ix, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw * gOut,
                    NC_offset_inp, grad_input_memory_span);
        safe_add_3d(grad_input.data, iz_, _iy, ix_, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne * gOut,
                    NC_offset_inp, grad_input_memory_span);
        safe_add_3d(grad_input.data, iz_, iy_, _ix, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw * gOut,
                    NC_offset_inp, grad_input_memory_span);
        safe_add_3d(grad_input.data, iz_, iy_, ix_, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse * gOut,
                    NC_offset_inp, grad_input_memory_span);
      }
      // calculate grad_grid
      if (at::native::within_bounds_3d(_iz, _iy, _ix, inp_D, inp_H, inp_W)) {
        scalar_t tnw_val = inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + _ix * inp_sW];
        dL_dix -= tnw_val * (pos_y) * (pos_z) * gOut;
        dL_diy -= tnw_val * (pos_x) * (pos_z) * gOut;
        dL_diz -= tnw_val * (pos_x) * (pos_y) * gOut;
      }
      if (at::native::within_bounds_3d(_iz, _iy, ix_, inp_D, inp_H, inp_W)) {
        scalar_t tne_val = inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + ix_ * inp_sW];
        dL_dix += tne_val * (pos_y) * (pos_z) * gOut;
        dL_diy -= tne_val * (pos_x_) * (pos_z) * gOut;
        dL_diz -= tne_val * (pos_x_) * (pos_y) * gOut;
      }
      if (at::native::within_bounds_3d(_iz, iy_, _ix, inp_D, inp_H, inp_W)) {
        scalar_t tsw_val = inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + _ix * inp_sW];
        dL_dix -= tsw_val * (pos_y_) * (pos_z) * gOut;
        dL_diy += tsw_val * (pos_x) * (pos_z) * gOut;
        dL_diz -= tsw_val * (pos_x) * (pos_y_) * gOut;
      }
      if (at::native::within_bounds_3d(_iz, iy_, ix_, inp_D, inp_H, inp_W)) {
        scalar_t tse_val = inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + ix_ * inp_sW];
        dL_dix += tse_val * (pos_y_) * (pos_z) * gOut;
        dL_diy += tse_val * (pos_x_) * (pos_z) * gOut;
        dL_diz -= tse_val * (pos_x_) * (pos_y_) * gOut;
      }
      if (at::native::within_bounds_3d(iz_, _iy, _ix, inp_D, inp_H, inp_W)) {
        scalar_t bnw_val = inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + _ix * inp_sW];
        dL_dix -= bnw_val * (pos_y) * (pos_z_) * gOut;
        dL_diy -= bnw_val * (pos_x) * (pos_z_) * gOut;
        dL_diz += bnw_val * (pos_x) * (pos_y) * gOut;
      }
      if (at::native::within_bounds_3d(iz_, _iy, ix_, inp_D, inp_H, inp_W)) {
        scalar_t bne_val = inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + ix_ * inp_sW];
        dL_dix += bne_val * (pos_y) * (pos_z_) * gOut;
        dL_diy -= bne_val * (pos_x_) * (pos_z_) * gOut;
        dL_diz += bne_val * (pos_x_) * (pos_y) * gOut;
      }
      if (at::native::within_bounds_3d(iz_, iy_, _ix, inp_D, inp_H, inp_W)) {
        scalar_t bsw_val = inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + _ix * inp_sW];
        dL_dix -= bsw_val * (pos_y_) * (pos_z_) * gOut;
        dL_diy += bsw_val * (pos_x) * (pos_z_) * gOut;
        dL_diz += bsw_val * (pos_x) * (pos_y_) * gOut;
      }
      if (at::native::within_bounds_3d(iz_, iy_, ix_, inp_D, inp_H, inp_W)) {
        scalar_t bse_val = inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + ix_ * inp_sW];
        dL_dix += bse_val * (pos_y_) * (pos_z_) * gOut;
        dL_diy += bse_val * (pos_x_) * (pos_z_) * gOut;
        dL_diz += bse_val * (pos_x_) * (pos_y_) * gOut;
      }
    }

    // assuming grad_grid is contiguous
    // thus we can
    //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
    //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
    scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
    gGrid_ptr_NDHW[0] = dL_dix_mult * dL_dix * pos_x_derivative;
    gGrid_ptr_NDHW[1] = dL_diy_mult * dL_diy * pos_y_derivative;
    gGrid_ptr_NDHW[2] = dL_diz_mult * dL_diz * pos_z_derivative;
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void smooth_sampler_backward_backward_kernel(
    const index_t nthreads,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_input, // initialized to empty
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_grid, // initialized to zeros
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_grad_out, // initialized to zeros
    at::cuda::detail::TensorInfo<scalar_t, index_t> input,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grid,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_out_input,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_out_grid,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_output,
    const GridSamplerPadding padding_mode,
    bool align_corners,
    bool apply_smoothstep,
    bool input_requires_grad,
    const index_t grad_input_memory_span,
    const index_t grad_grad_out_memory_span) {
  index_t C = input.sizes[1];
  index_t inp_D = input.sizes[2];
  index_t inp_H = input.sizes[3];

  index_t inp_W = input.sizes[4];
  index_t out_D = grid.sizes[1];
  index_t out_H = grid.sizes[2];
  index_t out_W = grid.sizes[3];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sD = input.strides[2];
  index_t inp_sH = input.strides[3];
  index_t inp_sW = input.strides[4];
  index_t grid_sN = grid.strides[0];
  index_t grid_sD = grid.strides[1];
  index_t grid_sH = grid.strides[2];
  index_t grid_sW = grid.strides[3];
  index_t grid_sCoor = grid.strides[4];

  index_t gGrid_sW = grad_grid.strides[3];

  index_t gOut_sN = grad_output.strides[0];
  index_t gOut_sC = grad_output.strides[1];
  index_t gOut_sD = grad_output.strides[2];
  index_t gOut_sH = grad_output.strides[3];
  index_t gOut_sW = grad_output.strides[4];

  index_t gOutGrid_sW = grad_out_grid.strides[3];

  index_t gOutInput_sN = 0;
  index_t gOutInput_sC = 0;

  if (input_requires_grad) {
    gOutInput_sN = grad_out_input.strides[0];
    gOutInput_sC = grad_out_input.strides[1];
  }

  index_t gInp_sN = grad_input.strides[0];
  index_t gInp_sC = grad_input.strides[1];
  index_t gInp_sD = grad_input.strides[2];
  index_t gInp_sH = grad_input.strides[3];
  index_t gInp_sW = grad_input.strides[4];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % out_W;
    const index_t h = (index / out_W) % out_H;
    const index_t d = (index / (out_H * out_W)) % out_D;
    const index_t n = index / (out_D * out_H * out_W);
    const auto grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y, z co-ordinates from grid
    scalar_t ix = grid.data[grid_offset];
    scalar_t iy = grid.data[grid_offset + grid_sCoor];
    scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

    // multipliers for gradients on ix, iy, and iz
    scalar_t dL_dix_mult, dL_diy_mult, dL_diz_mult;
    ix = at::native::grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &dL_dix_mult);
    iy = at::native::grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &dL_diy_mult);
    iz = at::native::grid_sampler_compute_source_index_set_grad(iz, inp_D, padding_mode, align_corners, &dL_diz_mult);

    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    index_t _ix = static_cast<index_t>(::floor(ix));
    index_t _iy = static_cast<index_t>(::floor(iy));
    index_t _iz = static_cast<index_t>(::floor(iz));
    index_t ix_ = _ix + 1;
    index_t iy_ = _iy + 1;
    index_t iz_ = _iz + 1;

    scalar_t pos_x_ = ix - _ix;
    scalar_t pos_y_ = iy - _iy;
    scalar_t pos_z_ = iz - _iz;

    scalar_t pos_x_derivative_ = dL_dix_mult;
    scalar_t pos_y_derivative_ = dL_diy_mult;
    scalar_t pos_z_derivative_ = dL_diz_mult;

    scalar_t pos_x_2nd_derivative_ = 0.0f;
    scalar_t pos_y_2nd_derivative_ = 0.0f;
    scalar_t pos_z_2nd_derivative_ = 0.0f;

    scalar_t pos_x_2nd_derivative = 0.0f;
    scalar_t pos_y_2nd_derivative = 0.0f;
    scalar_t pos_z_2nd_derivative = 0.0f;

    if (apply_smoothstep) {
      pos_x_derivative_ *= smoothstep_derivative(pos_x_);
      pos_y_derivative_ *= smoothstep_derivative(pos_y_);
      pos_z_derivative_ *= smoothstep_derivative(pos_z_);

      pos_x_2nd_derivative_ = dL_dix_mult * dL_dix_mult * smoothstep_2nd_derivative(pos_x_);
      pos_y_2nd_derivative_ = dL_diy_mult * dL_diy_mult * smoothstep_2nd_derivative(pos_y_);
      pos_z_2nd_derivative_ = dL_diz_mult * dL_diz_mult * smoothstep_2nd_derivative(pos_z_);

      pos_x_2nd_derivative = -pos_x_2nd_derivative_;
      pos_y_2nd_derivative = -pos_y_2nd_derivative_;
      pos_z_2nd_derivative = -pos_z_2nd_derivative_;

      pos_x_ = smoothstep(pos_x_);
      pos_y_ = smoothstep(pos_y_);
      pos_z_ = smoothstep(pos_z_);
    }

    scalar_t pos_x = 1.0f - pos_x_;
    scalar_t pos_y = 1.0f - pos_y_;
    scalar_t pos_z = 1.0f - pos_z_;

    scalar_t pos_x_derivative = -pos_x_derivative_;
    scalar_t pos_y_derivative = -pos_y_derivative_;
    scalar_t pos_z_derivative = -pos_z_derivative_;

    index_t index_corners[2][3] = {{_ix, _iy, _iz},
                                   {ix_, iy_, iz_}};
    scalar_t pos_corners[2][9] = {{pos_x, pos_y, pos_z,
                                   pos_x_derivative, pos_y_derivative, pos_z_derivative,
                                   pos_x_2nd_derivative, pos_y_2nd_derivative, pos_z_2nd_derivative},
                                  {pos_x_, pos_y_, pos_z_,
                                   pos_x_derivative_, pos_y_derivative_, pos_z_derivative_,
                                   pos_x_2nd_derivative_, pos_y_2nd_derivative_, pos_z_2nd_derivative_}};
    scalar_t surface_coefficients[8] = {};
    scalar_t out_derivatives[8][12] = {};

    #pragma unroll
    for (int shift = 0; shift < 8; shift++) {
      int px = (shift >> 0) & 1;
      int py = (shift >> 1) & 1;
      int pz = (shift >> 2) & 1;

      surface_coefficients[shift] = pos_corners[px][0] * pos_corners[py][1] * pos_corners[pz][2];

      out_derivatives[shift][0] = pos_corners[py][1] * pos_corners[pz][2] * pos_corners[px][3]; // dOut_dx / surf_weight
      out_derivatives[shift][1] = pos_corners[py][1] * pos_corners[pz][2] * pos_corners[px][6]; // d2Out_dx2 / surf_weight
      out_derivatives[shift][2] = pos_corners[py][4] * pos_corners[pz][2] * pos_corners[px][3]; // d2Out_dxdy / surf_weight
      out_derivatives[shift][3] = pos_corners[py][1] * pos_corners[pz][5] * pos_corners[px][3]; // d2Out_dxdz / surf_weight

      out_derivatives[shift][4] = pos_corners[px][0] * pos_corners[pz][2] * pos_corners[py][4]; // dOut_dy / surf_weight
      out_derivatives[shift][5] = pos_corners[px][0] * pos_corners[pz][2] * pos_corners[py][7]; // d2Out_dy2 / surf_weight
      out_derivatives[shift][6] = pos_corners[px][3] * pos_corners[pz][2] * pos_corners[py][4]; // d2Out_dydx / surf_weight
      out_derivatives[shift][7] = pos_corners[px][0] * pos_corners[pz][5] * pos_corners[py][4]; // d2Out_dydz / surf_weight

      out_derivatives[shift][8] = pos_corners[px][0] * pos_corners[py][1] * pos_corners[pz][5]; // dOut_dz / surf_weight
      out_derivatives[shift][9] = pos_corners[px][0] * pos_corners[py][1] * pos_corners[pz][8]; // d2Out_dz2 / surf_weight
      out_derivatives[shift][10] = pos_corners[px][3] * pos_corners[py][1] * pos_corners[pz][5]; // d2Out_dzdx / surf_weight
      out_derivatives[shift][11] = pos_corners[px][0] * pos_corners[py][4] * pos_corners[pz][5]; // d2Out_dzdy / surf_weight
    }

    scalar_t d2L_dix2 = static_cast<scalar_t>(0), d2L_diy2 = static_cast<scalar_t>(0), d2L_diz2 = static_cast<scalar_t>(0);
    index_t offset_out_DHW = d * gOut_sD + h * gOut_sH + w * gOut_sW;
    scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + offset_out_DHW;
    index_t NC_offset_inp = n * gInp_sN;
    index_t NC_offset_out = n * gOut_sN;
    scalar_t *inp_ptr_NC = input.data + n * inp_sN;

    scalar_t *gOutInput_ptr_NC = NULL;

    if (input_requires_grad) {
      gOutInput_ptr_NC = grad_out_input.data + n * gOutInput_sN;
    }

    scalar_t *gOutGrid_ptr_NDHW = grad_out_grid.data + index * gOutGrid_sW;
    scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;

    for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, inp_ptr_NC += inp_sC, gOutInput_ptr_NC += gOutInput_sC, NC_offset_inp += gInp_sC, NC_offset_out += gOut_sC) {
      scalar_t gOut = *gOut_ptr_NCDHW;

      #pragma unroll
      for (int shift = 0; shift < 8; shift++) {
        int px = (shift >> 0) & 1;
        int py = (shift >> 1) & 1;
        int pz = (shift >> 2) & 1;

        index_t ix = index_corners[px][0];
        index_t iy = index_corners[py][1];
        index_t iz = index_corners[pz][2];

        // Slightly unprecise naming: in fact these are divided by surf_weight.
        scalar_t dOut_dx = out_derivatives[shift][0]; // E.g. variable "dOut_dx" is mathematically "dOut/dx * 1/surf_weight"
        scalar_t d2Out_dx2 = out_derivatives[shift][1];
        scalar_t d2Out_dxdy = out_derivatives[shift][2];
        scalar_t d2Out_dxdz = out_derivatives[shift][3];
        scalar_t dOut_dy = out_derivatives[shift][4];
        scalar_t d2Out_dy2 = out_derivatives[shift][5];
        scalar_t d2Out_dydx = out_derivatives[shift][6];
        scalar_t d2Out_dydz = out_derivatives[shift][7];
        scalar_t dOut_dz = out_derivatives[shift][8];
        scalar_t d2Out_dz2 = out_derivatives[shift][9];
        scalar_t d2Out_dzdx = out_derivatives[shift][10];
        scalar_t d2Out_dzdy = out_derivatives[shift][11];
        scalar_t surface_coeff = surface_coefficients[shift];

        if (at::native::within_bounds_3d(iz, iy, ix, inp_D, inp_H, inp_W)) {
          index_t inp_el = iz * inp_sD + iy * inp_sH + ix * inp_sW;
          scalar_t surf_weight = inp_ptr_NC[inp_el];

          scalar_t dL_dx = gOut * dOut_dx;
          scalar_t dL_dy = gOut * dOut_dy;
          scalar_t dL_dz = gOut * dOut_dz;

          scalar_t gOutGrid_x = gOutGrid_ptr_NDHW[0];
          scalar_t gOutGrid_y = gOutGrid_ptr_NDHW[1];
          scalar_t gOutGrid_z = gOutGrid_ptr_NDHW[2];

          scalar_t grad_grad_out_delta = surf_weight * (dOut_dx * gOutGrid_x
                                                       + dOut_dy * gOutGrid_y
                                                       + dOut_dz * gOutGrid_z);

          if (gOutInput_ptr_NC != NULL) {
            scalar_t gOutInput = gOutInput_ptr_NC[inp_el];
            grad_grad_out_delta += gOutInput * surface_coeff;
            d2L_dix2 += dL_dx * gOutInput;
            d2L_diy2 += dL_dy * gOutInput;
            d2L_diz2 += dL_dz * gOutInput;
          }

          at::native::fastAtomicAdd(grad_grad_out.data,
                                    NC_offset_out + offset_out_DHW,
                                    grad_grad_out_memory_span,
                                    grad_grad_out_delta,
                                    true);

          d2L_dix2 += surf_weight * gOut * (d2Out_dx2 * gOutGrid_x
                                            + d2Out_dxdy * gOutGrid_y
                                            + d2Out_dxdz * gOutGrid_z);
          d2L_diy2 += surf_weight * gOut * (d2Out_dydx * gOutGrid_x
                                            + d2Out_dy2 * gOutGrid_y
                                            + d2Out_dydz * gOutGrid_z);
          d2L_diz2 += surf_weight * gOut * (d2Out_dzdx * gOutGrid_x
                                            + d2Out_dzdy * gOutGrid_y
                                            + d2Out_dz2 * gOutGrid_z);

          add_3d(grad_input.data, iz, iy, ix, gInp_sD, gInp_sH, gInp_sW,
                dL_dx * gOutGrid_x + dL_dy * gOutGrid_y + dL_dz * gOutGrid_z,
                NC_offset_inp, grad_input_memory_span);
        }
      }
    }

    gGrid_ptr_NDHW[0] = d2L_dix2;
    gGrid_ptr_NDHW[1] = d2L_diy2;
    gGrid_ptr_NDHW[2] = d2L_diz2;
  }
}

void launch_smooth_sampler_forward_kernel(
    const torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &grid,
    int64_t padding_mode, bool align_corners, bool apply_smoothstep) {
  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  int64_t count = N * D * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "smooth_sampler_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(output)) {
        smooth_sampler_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<scalar_t, int>(input),
            getTensorInfo<scalar_t, int>(grid),
            getTensorInfo<scalar_t, int>(output),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            apply_smoothstep);
      } else {
        smooth_sampler_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<scalar_t, int64_t>(input),
            getTensorInfo<scalar_t, int64_t>(grid),
            getTensorInfo<scalar_t, int64_t>(output),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            apply_smoothstep);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}

void launch_smooth_sampler_backward_kernel(
    const torch::Tensor &grad_input, const torch::Tensor &grad_grid,
    const torch::Tensor& grad_output, const torch::Tensor& input,
    const torch::Tensor& grid, int64_t padding_mode,
    bool align_corners, bool apply_smoothstep, bool input_requires_grad) {
  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  int64_t count = N * D * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "smooth_sampler_backward_cuda", [&] {
      if (at::native::canUse32BitIndexMath(input) && at::native::canUse32BitIndexMath(grid) &&
          at::native::canUse32BitIndexMath(grad_output)) {
        smooth_sampler_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            at::cuda::detail::getTensorInfo<scalar_t, int>(grad_output),
            at::cuda::detail::getTensorInfo<scalar_t, int>(input),
            at::cuda::detail::getTensorInfo<scalar_t, int>(grid),
            input_requires_grad ? getTensorInfo<scalar_t, int>(grad_input) : TensorInfo<scalar_t, int>(),
            getTensorInfo<scalar_t, int>(grad_grid),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            apply_smoothstep,
            /*grad_input_memory_span =*/input_requires_grad ? static_cast<int>(grad_input.numel()) : 0,
            input_requires_grad);
      } else {
        smooth_sampler_backward_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_output),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(input),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grid),
            input_requires_grad ? at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_input) : at::cuda::detail::TensorInfo<scalar_t, int64_t>(),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_grid),
            static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
            align_corners,
            apply_smoothstep,
            /*grad_input_memory_span =*/input_requires_grad ? grad_input.numel() : 0,
            input_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}

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
    const bool input_requires_grad) {
  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  int64_t count = N * D * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "smooth_sampler_backward_backward_cuda", [&] {
      if (at::native::canUse32BitIndexMath(input) && at::native::canUse32BitIndexMath(grid) && at::native::canUse32BitIndexMath(grad_output)
          && at::native::canUse32BitIndexMath(grad_out_input) && at::native::canUse32BitIndexMath(grad_out_grid)) {
        smooth_sampler_backward_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            at::cuda::detail::getTensorInfo<scalar_t, int>(grad_input),
            at::cuda::detail::getTensorInfo<scalar_t, int>(grad_grid),
            at::cuda::detail::getTensorInfo<scalar_t, int>(grad_grad_out),
            at::cuda::detail::getTensorInfo<scalar_t, int>(input),
            at::cuda::detail::getTensorInfo<scalar_t, int>(grid),
            input_requires_grad ? at::cuda::detail::getTensorInfo<scalar_t, int>(grad_out_input) : at::cuda::detail::TensorInfo<scalar_t, int>(),
            at::cuda::detail::getTensorInfo<scalar_t, int>(grad_out_grid),
            at::cuda::detail::getTensorInfo<scalar_t, int>(grad_output),
            static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
            align_corners,
            apply_smoothstep,
            input_requires_grad,
            static_cast<int>(grad_input.numel()),
            static_cast<int>(grad_grad_out.numel()));
      } else {
        smooth_sampler_backward_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_input),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_grid),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_grad_out),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(input),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grid),
            input_requires_grad ? at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_out_input) : at::cuda::detail::TensorInfo<scalar_t, int64_t>(),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_out_grid),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_output),
            static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
            align_corners,
            apply_smoothstep,
            input_requires_grad,
            grad_input.numel(),
            grad_grad_out.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}


template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void bool_sampler_kernel(
    const index_t nthreads,
    TensorInfo<bool, index_t> input,
    TensorInfo<scalar_t, index_t> grid,
    TensorInfo<bool, index_t> output,
    const GridSamplerPadding padding_mode,
    bool align_corners,
    bool apply_smoothstep) {

  index_t C = input.sizes[1];
  index_t inp_D = input.sizes[2];
  index_t inp_H = input.sizes[3];
  index_t inp_W = input.sizes[4];
  index_t out_D = grid.sizes[1];
  index_t out_H = grid.sizes[2];
  index_t out_W = grid.sizes[3];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sD = input.strides[2];
  index_t inp_sH = input.strides[3];
  index_t inp_sW = input.strides[4];
  index_t grid_sN = grid.strides[0];
  index_t grid_sD = grid.strides[1];
  index_t grid_sH = grid.strides[2];
  index_t grid_sW = grid.strides[3];
  index_t grid_sCoor = grid.strides[4];
  index_t out_sN = output.strides[0];
  index_t out_sC = output.strides[1];
  index_t out_sD = output.strides[2];
  index_t out_sH = output.strides[3];
  index_t out_sW = output.strides[4];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % out_W;
    const index_t h = (index / out_W) % out_H;
    const index_t d = (index / (out_H * out_W)) % out_D;
    const index_t n = index / (out_D * out_H * out_W);
    const index_t grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y, z co-ordinates from grid
    scalar_t ix = grid.data[grid_offset];
    scalar_t iy = grid.data[grid_offset + grid_sCoor];
    scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

    ix = at::native::grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
    iy = at::native::grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);
    iz = at::native::grid_sampler_compute_source_index(iz, inp_D, padding_mode, align_corners);

    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    index_t _ix = static_cast<index_t>(::floor(ix));
    index_t _iy = static_cast<index_t>(::floor(iy));
    index_t _iz = static_cast<index_t>(::floor(iz));
    index_t ix_ = _ix + 1;
    index_t iy_ = _iy + 1;
    index_t iz_ = _iz + 1;

    scalar_t pos_x_ = ix - _ix;
    scalar_t pos_y_ = iy - _iy;
    scalar_t pos_z_ = iz - _iz;

    if (apply_smoothstep) {
      pos_x_ = smoothstep(pos_x_);
      pos_y_ = smoothstep(pos_y_);
      pos_z_ = smoothstep(pos_z_);
    }

    scalar_t pos_x = 1.0f - pos_x_;
    scalar_t pos_y = 1.0f - pos_y_;
    scalar_t pos_z = 1.0f - pos_z_;

    // get surfaces to each neighbor:
    // scalar_t tnw = pos_x  * pos_y  * pos_z;
    // scalar_t tne = pos_x_ * pos_y  * pos_z;
    // scalar_t tsw = pos_x  * pos_y_ * pos_z;
    // scalar_t tse = pos_x_ * pos_y_ * pos_z;
    // scalar_t bnw = pos_x  * pos_y  * pos_z_;
    // scalar_t bne = pos_x_ * pos_y  * pos_z_;
    // scalar_t bsw = pos_x  * pos_y_ * pos_z_;
    // scalar_t bse = pos_x_ * pos_y_ * pos_z_;

    auto inp_ptr_NC = input.data + n * inp_sN;
    auto out_ptr_NCDHW = output.data + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
    for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
      *out_ptr_NCDHW = inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + _ix * inp_sW] || inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + ix_ * inp_sW] ||
                       inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + _ix * inp_sW] || inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + ix_ * inp_sW] ||
                       inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + _ix * inp_sW] || inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + ix_ * inp_sW] ||
                       inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + _ix * inp_sW] || inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + ix_ * inp_sW];
      // if (at::native::within_bounds_3d(_iz, _iy, _ix, inp_D, inp_H, inp_W)) {
      //   *out_ptr_NCDHW += inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + _ix * inp_sW] * tnw;
      // }
      // if (at::native::within_bounds_3d(_iz, _iy, ix_, inp_D, inp_H, inp_W)) {
      //   *out_ptr_NCDHW += inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + ix_ * inp_sW] * tne;
      // }
      // if (at::native::within_bounds_3d(_iz, iy_, _ix, inp_D, inp_H, inp_W)) {
      //   *out_ptr_NCDHW += inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + _ix * inp_sW] * tsw;
      // }
      // if (at::native::within_bounds_3d(_iz, iy_, ix_, inp_D, inp_H, inp_W)) {
      //   *out_ptr_NCDHW += inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + ix_ * inp_sW] * tse;
      // }
      // if (at::native::within_bounds_3d(iz_, _iy, _ix, inp_D, inp_H, inp_W)) {
      //   *out_ptr_NCDHW += inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + _ix * inp_sW] * bnw;
      // }
      // if (at::native::within_bounds_3d(iz_, _iy, ix_, inp_D, inp_H, inp_W)) {
      //   *out_ptr_NCDHW += inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + ix_ * inp_sW] * bne;
      // }
      // if (at::native::within_bounds_3d(iz_, iy_, _ix, inp_D, inp_H, inp_W)) {
      //   *out_ptr_NCDHW += inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + _ix * inp_sW] * bsw;
      // }
      // if (at::native::within_bounds_3d(iz_, iy_, ix_, inp_D, inp_H, inp_W)) {
      //   *out_ptr_NCDHW += inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + ix_ * inp_sW] * bse;
      // }
    }
  }
}


void launch_bool_sampler_kernel(
    const torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &grid,
    int64_t padding_mode, bool align_corners, bool apply_smoothstep) {
  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  int64_t count = N * D * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grid.scalar_type(), "bool_sampler_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(output)) {
        bool_sampler_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<bool, int>(input),
            getTensorInfo<scalar_t, int>(grid),
            getTensorInfo<bool, int>(output),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            apply_smoothstep);
      } else {
        bool_sampler_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<bool, int64_t>(input),
            getTensorInfo<scalar_t, int64_t>(grid),
            getTensorInfo<bool, int64_t>(output),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            apply_smoothstep);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}