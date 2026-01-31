#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

namespace gcore::rt::hip::kernels {

void launch_fill(hipStream_t stream, uint32_t *data, uint32_t value, size_t n);

void launch_rmsnorm_naive(hipStream_t stream, const float *x,
                          const float *gamma, float *y, uint32_t rows,
                          uint32_t cols, float eps);

void launch_softmax_naive(hipStream_t stream, const float *x, float *y,
                          uint32_t rows, uint32_t cols);

} // namespace gcore::rt::hip::kernels
