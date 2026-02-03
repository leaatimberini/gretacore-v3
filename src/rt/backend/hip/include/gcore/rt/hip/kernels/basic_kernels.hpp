#pragma once

#include <cstddef>
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
                             uint32_t seq_len, uint32_t dim,
                             uint32_t vocab_size, bool row_major);

struct DebugStats {
  uint64_t nan_count;
  uint64_t inf_count;
  float min_val;
  float max_val;
  float max_abs;
  double sum;
  double sum_abs;
  uint64_t count;
};

// Debug instrumentation
void launch_debug_tensor_stats(hipStream_t stream, const char *label,
                               const float *d_data, uint32_t n);

void launch_debug_tensor_stats_ex(hipStream_t stream, const float *d_data,
                                  uint32_t n, DebugStats *h_out);

void launch_argmax(hipStream_t stream, const float *d_logits, uint32_t n,
                   int32_t *h_out);

} // namespace gcore::rt::hip::kernels
