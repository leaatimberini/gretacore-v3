#include "gcore/rt/hip/kernels/basic_kernels.hpp"
#include <iostream>
#include <cstring>
#include <cmath>

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

__global__ void rmsnorm_parallel_kernel(const float *__restrict__ x,
                                       const float *__restrict__ gamma,
                                       float *__restrict__ y, uint32_t rows,
                                       uint32_t cols, float eps) {
  uint32_t row = blockIdx.x;
  if (row >= rows)
    return;

  uint32_t tid = threadIdx.x;
  uint32_t bdim = blockDim.x;
  uint32_t base = row * cols;

  float ms = 0.0f;
  for (uint32_t c = tid; c < cols; c += bdim) {
    float v = x[base + c];
    ms += v * v;
  }

  __shared__ float s_ms[256];
  s_ms[tid] = ms;
  __syncthreads();

  for (uint32_t s = bdim / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_ms[tid] += s_ms[tid + s];
    }
    __syncthreads();
  }

  float inv = rsqrtf(s_ms[0] / float(cols) + eps);

  for (uint32_t c = tid; c < cols; c += bdim) {
    y[base + c] = (x[base + c] * inv) * gamma[c];
  }
}

void launch_rmsnorm_naive(hipStream_t stream, const float *x,
                          const float *gamma, float *y, uint32_t rows,
                          uint32_t cols, float eps) {
  dim3 block(256, 1, 1);
  dim3 grid(rows, 1, 1);
  rmsnorm_parallel_kernel<<<grid, block, 0, stream>>>(x, gamma, y, rows, cols,
                                                      eps);
}

__global__ void softmax_parallel_kernel(const float *__restrict__ x,
                                        float *__restrict__ y, uint32_t rows,
                                        uint32_t cols) {
  uint32_t row = blockIdx.x;
  if (row >= rows)
    return;

  uint32_t tid = threadIdx.x;
  uint32_t bdim = blockDim.x;
  uint32_t base = row * cols;

  // Max reduction
  float maxv = -1e38f;
  for (uint32_t c = tid; c < cols; c += bdim) {
    float v = x[base + c];
    if (v > maxv)
      maxv = v;
  }

  __shared__ float s_max[256];
  s_max[tid] = maxv;
  __syncthreads();
  for (uint32_t s = bdim / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
    }
    __syncthreads();
  }
  maxv = s_max[0];

  // Sum reduction
  float sum = 0.0f;
  for (uint32_t c = tid; c < cols; c += bdim) {
    float e = expf(x[base + c] - maxv);
    y[base + c] = e;
    sum += e;
  }

  __shared__ float s_sum[256];
  s_sum[tid] = sum;
  __syncthreads();
  for (uint32_t s = bdim / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_sum[tid] += s_sum[tid + s];
    }
    __syncthreads();
  }
  float inv_sum = 1.0f / (s_sum[0] + 1e-10f);

  for (uint32_t c = tid; c < cols; c += bdim) {
    y[base + c] *= inv_sum;
  }
}

void launch_softmax_naive(hipStream_t stream, const float *x, float *y,
                          uint32_t rows, uint32_t cols) {
  dim3 block(256, 1, 1);
  dim3 grid(rows, 1, 1);
  softmax_parallel_kernel<<<grid, block, 0, stream>>>(x, y, rows, cols);
}

__global__ void add_kernel(const float *a, const float *b, float *c, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

void launch_add(hipStream_t stream, const float *a, const float *b, float *c,
                size_t n) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  add_kernel<<<grid_size, block_size, 0, stream>>>(a, b, c, n);
}

__global__ void silu_kernel(const float *x, float *y, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float v = x[idx];
    y[idx] = v / (1.0f + expf(-v));
  }
}

void launch_silu(hipStream_t stream, const float *x, float *y, size_t n) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  silu_kernel<<<grid_size, block_size, 0, stream>>>(x, y, n);
}

__global__ void mul_kernel(const float *a, const float *b, float *c, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] * b[idx];
  }
}

void launch_mul(hipStream_t stream, const float *a, const float *b, float *c,
                size_t n) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  mul_kernel<<<grid_size, block_size, 0, stream>>>(a, b, c, n);
}

__global__ void embedding_lookup_kernel(const int32_t *tokens,
                                        const float *embeddings, float *output,
                                        uint32_t seq_len, uint32_t dim,
                                        uint32_t vocab_size,
                                        uint8_t layout_row_major) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total = seq_len * dim;
  if (idx >= total)
    return;

  uint32_t s = idx / dim;
  uint32_t d = idx % dim;
  int32_t token = tokens[s];

  if (token >= 0 && token < (int32_t)vocab_size) {
    if (layout_row_major) {
      output[idx] =
          embeddings[static_cast<uint32_t>(token) * dim + d];
    } else {
      output[idx] =
          embeddings[d * vocab_size + static_cast<uint32_t>(token)];
    }
  } else {
    output[idx] = 0.0f;
  }
}

void launch_embedding_lookup(hipStream_t stream, const int32_t *tokens,
                             const float *embeddings, float *output,
                             uint32_t seq_len, uint32_t dim,
                             uint32_t vocab_size, bool row_major) {
  uint32_t total = seq_len * dim;
  int block_size = 256;
  int grid_size = (total + block_size - 1) / block_size;
  embedding_lookup_kernel<<<grid_size, block_size, 0, stream>>>(
      tokens, embeddings, output, seq_len, dim, vocab_size,
      row_major ? 1 : 0);
}

// Debug instrumentation
__global__ void debug_tensor_stats_kernel(const float *data, uint32_t n,
                                         DebugStats *out) {
  uint32_t tid = threadIdx.x;
  uint32_t gid = blockIdx.x * blockDim.x + tid;

  uint64_t nans = 0;
  uint64_t infs = 0;
  float max_abs = 0.0f;
  float min_v = 1e38f;
  float max_v = -1e38f;
  double sum_v = 0.0;
  double sum_a = 0.0;
  uint64_t count_v = 0;

  for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
    float v = data[i];
    if (isnan(v)) {
      nans++;
    } else if (isinf(v)) {
      infs++;
    } else {
      float av = fabsf(v);
      if (av > max_abs)
        max_abs = av;
      if (v < min_v)
        min_v = v;
      if (v > max_v)
        max_v = v;
      sum_v += (double)v;
      sum_a += (double)av;
      count_v++;
    }
  }

  __shared__ uint64_t s_nans[256];
  __shared__ uint64_t s_infs[256];
  __shared__ float s_max_abs[256];
  __shared__ float s_min[256];
  __shared__ float s_max[256];
  __shared__ double s_sum[256];
  __shared__ double s_sum_abs[256];
  __shared__ uint64_t s_count[256];

  s_nans[tid] = nans;
  s_infs[tid] = infs;
  s_max_abs[tid] = max_abs;
  s_min[tid] = min_v;
  s_max[tid] = max_v;
  s_sum[tid] = sum_v;
  s_sum_abs[tid] = sum_a;
  s_count[tid] = count_v;
  __syncthreads();

  for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_nans[tid] += s_nans[tid + s];
      s_infs[tid] += s_infs[tid + s];
      s_max_abs[tid] = fmaxf(s_max_abs[tid], s_max_abs[tid + s]);
      s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
      s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
      s_sum[tid] += s_sum[tid + s];
      s_sum_abs[tid] += s_sum_abs[tid + s];
      s_count[tid] += s_count[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd((unsigned long long *)&out->nan_count,
              (unsigned long long)s_nans[0]);
    atomicAdd((unsigned long long *)&out->inf_count,
              (unsigned long long)s_infs[0]);
    atomicMax((int *)reinterpret_cast<float *>(&out->max_abs),
              *reinterpret_cast<int *>(&s_max_abs[0]));
    atomicMin((int *)reinterpret_cast<float *>(&out->min_val),
              *reinterpret_cast<int *>(&s_min[0]));
    atomicMax((int *)reinterpret_cast<float *>(&out->max_val),
              *reinterpret_cast<int *>(&s_max[0]));
    atomicAdd(&out->sum, s_sum[0]);
    atomicAdd(&out->sum_abs, s_sum_abs[0]);
    atomicAdd((unsigned long long *)&out->count,
              (unsigned long long)s_count[0]);
  }
}

void launch_debug_tensor_stats_ex(hipStream_t stream, const float *d_data,
                                  uint32_t n, DebugStats *h_out) {
  DebugStats *d_stats;
  (void)hipMalloc(&d_stats, sizeof(DebugStats));
  (void)hipMemsetAsync(d_stats, 0, sizeof(DebugStats), stream);

  float init_min = 1e38f;
  float init_max = -1e38f;
  (void)hipMemcpyAsync(&d_stats->min_val, &init_min, sizeof(float),
                       hipMemcpyHostToDevice, stream);
  (void)hipMemcpyAsync(&d_stats->max_val, &init_max, sizeof(float),
                       hipMemcpyHostToDevice, stream);

  uint32_t block = 256;
  uint32_t grid = (n + block - 1) / block;
  if (grid > 1024) grid = 1024;

  debug_tensor_stats_kernel<<<grid, block, 0, stream>>>(d_data, n, d_stats);
  (void)hipMemcpyAsync(h_out, d_stats, sizeof(DebugStats), hipMemcpyDeviceToHost,
                       stream);
  (void)hipStreamSynchronize(stream);
  (void)hipFree(d_stats);
}

void launch_debug_tensor_stats(hipStream_t stream, const char *label,
                               const float *d_data, uint32_t n) {
  DebugStats h_stats;
  launch_debug_tensor_stats_ex(stream, d_data, n, &h_stats);

  printf("[STATS] %s: n=%u nan=%lu inf=%lu min=%.4f max=%.4f abs_max=%.4f "
         "avg=%.4f\n",
         label, n, (unsigned long)h_stats.nan_count, (unsigned long)h_stats.inf_count, 
         h_stats.min_val, h_stats.max_val, h_stats.max_abs,
         h_stats.count > 0 ? (float)(h_stats.sum / h_stats.count) : 0.0f);
}

__global__ void argmax_kernel(const float *logits, uint32_t n, int32_t *out) {
  uint32_t tid = threadIdx.x;
  float max_val = -1e38f;
  int32_t max_idx = -1;

  for (uint32_t i = tid; i < n; i += blockDim.x) {
    float v = logits[i];
    if (v > max_val) {
      max_val = v;
      max_idx = (int32_t)i;
    }
  }

  __shared__ float s_max[256];
  __shared__ int32_t s_idx[256];

  s_max[tid] = max_val;
  s_idx[tid] = max_idx;
  __syncthreads();

  for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (s_max[tid + s] > s_max[tid]) {
        s_max[tid] = s_max[tid + s];
        s_idx[tid] = s_idx[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    *out = s_idx[0];
  }
}

void launch_argmax(hipStream_t stream, const float *d_logits, uint32_t n,
                   int32_t *h_out) {
  int32_t *d_out;
  (void)hipMalloc(&d_out, sizeof(int32_t));
  argmax_kernel<<<1, 256, 0, stream>>>(d_logits, n, d_out);
  (void)hipMemcpyAsync(h_out, d_out, sizeof(int32_t), hipMemcpyDeviceToHost, stream);
  (void)hipStreamSynchronize(stream);
  (void)hipFree(d_out);
}

} // namespace gcore::rt::hip::kernels
