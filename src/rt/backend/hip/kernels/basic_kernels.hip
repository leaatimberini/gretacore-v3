#include "gcore/rt/hip/kernels/basic_kernels.hpp"

namespace gcore::rt::hip::kernels {

__global__ void fill_kernel(uint32_t *data, uint32_t value, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = value;
  }
}

void launch_fill(hipStream_t stream, uint32_t *data, uint32_t value, size_t n) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  fill_kernel<<<grid_size, block_size, 0, stream>>>(data, value, n);
}

__global__ void rmsnorm_naive_kernel(const float *x, const float *gamma,
                                     float *y, uint32_t rows, uint32_t cols,
                                     float eps) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows)
    return; // 1 thread per row (naive)

  uint32_t base = row * cols;

  float ms = 0.0f;
  for (uint32_t c = 0; c < cols; ++c) {
    float v = x[base + c];
    ms += v * v;
  }
  ms /= float(cols);
  float inv = rsqrtf(ms + eps);

  for (uint32_t c = 0; c < cols; ++c) {
    float v = x[base + c] * inv;
    y[base + c] = v * gamma[c];
  }
}

void launch_rmsnorm_naive(hipStream_t stream, const float *x,
                          const float *gamma, float *y, uint32_t rows,
                          uint32_t cols, float eps) {
  // Naive launch: 1 thread per row.
  // Very inefficient for large cols, but matches Vulkan reference logic.
  int block_size = 256;
  int grid_size = (rows + block_size - 1) / block_size;
  rmsnorm_naive_kernel<<<grid_size, block_size, 0, stream>>>(x, gamma, y, rows,
                                                             cols, eps);
}

__global__ void softmax_naive_kernel(const float *x, float *y, uint32_t rows,
                                     uint32_t cols) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows)
    return; // 1 thread per row

  uint32_t base = row * cols;

  float maxv = x[base + 0];
  for (uint32_t c = 1; c < cols; ++c) {
    float v = x[base + c];
    if (v > maxv)
      maxv = v;
  }

  float sum = 0.0f;
  for (uint32_t c = 0; c < cols; ++c) {
    float e = expf(x[base + c] - maxv);
    y[base + c] = e;
    sum += e;
  }

  float inv = 1.0f / sum;
  for (uint32_t c = 0; c < cols; ++c) {
    y[base + c] *= inv;
  }
}

void launch_softmax_naive(hipStream_t stream, const float *x, float *y,
                          uint32_t rows, uint32_t cols) {
  int block_size = 256;
  int grid_size = (rows + block_size - 1) / block_size;
  softmax_naive_kernel<<<grid_size, block_size, 0, stream>>>(x, y, rows, cols);
}

} // namespace gcore::rt::hip::kernels
