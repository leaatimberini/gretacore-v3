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

void launch_add(hipStream_t stream, const float *a, const float *b, float *c,
                size_t n);

void launch_silu(hipStream_t stream, const float *x, float *y, size_t n);

void launch_mul(hipStream_t stream, const float *a, const float *b, float *c,
                size_t n);

void launch_embedding_lookup(hipStream_t stream, const int32_t *tokens,
                             const float *embeddings, float *output,
                             uint32_t seq_len, uint32_t dim);

} // namespace gcore::rt::hip::kernels
