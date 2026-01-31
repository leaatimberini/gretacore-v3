#include "gcore/rt/hip/kernels/attention_kernels.hpp"
#include <cmath>

namespace gcore::rt::hip::kernels {

__global__ void rope_kernel(float *x, uint32_t seq_len, uint32_t num_heads, 
                            uint32_t head_dim, float base) {
  // We want to process each (pos, head, i) where i in [0, head_dim/2)
  // Total workload = seq_len * num_heads * (head_dim / 2)
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total_pairs = seq_len * num_heads * (head_dim / 2);
  
  if (idx >= total_pairs) return;

  uint32_t half_dim = head_dim / 2;
  uint32_t pair_idx = idx % half_dim;
  uint32_t head_idx = (idx / half_dim) % num_heads;
  uint32_t pos = idx / (half_dim * num_heads);

  // theta = pos * base^(-2i/dim)
  float theta = (float)pos * powf(base, -2.0f * (float)pair_idx / (float)head_dim);
  float cos_val = cosf(theta);
  float sin_val = sinf(theta);

  uint32_t base_idx = pos * (num_heads * head_dim) + head_idx * head_dim + (2 * pair_idx);
  
  float v0 = x[base_idx];
  float v1 = x[base_idx + 1];

  x[base_idx] = v0 * cos_val - v1 * sin_val;
  x[base_idx + 1] = v0 * sin_val + v1 * cos_val;
}

void launch_rope(hipStream_t stream, float *x, uint32_t seq_len, 
                 uint32_t num_heads, uint32_t head_dim, float base) {
  uint32_t total_pairs = seq_len * num_heads * (head_dim / 2);
  int block_size = 256;
  int grid_size = (total_pairs + block_size - 1) / block_size;
  
  rope_kernel<<<grid_size, block_size, 0, stream>>>(x, seq_len, num_heads, head_dim, base);
}

__global__ void causal_mask_kernel(float *data, uint32_t seq_len, float mask_val) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= seq_len) return;

  uint32_t base = row * seq_len;
  for (uint32_t col = row + 1; col < seq_len; ++col) {
    data[base + col] = mask_val;
  }
}

void launch_causal_mask(hipStream_t stream, float *data, uint32_t seq_len, float mask_val) {
  int block_size = 256;
  int grid_size = (seq_len + block_size - 1) / block_size;
  
  causal_mask_kernel<<<grid_size, block_size, 0, stream>>>(data, seq_len, mask_val);
}

__global__ void kv_update_kernel(float *cache_k, float *cache_v, const float *new_k,
                               const float *new_v, uint32_t pos,
                               uint32_t max_seq_len, uint32_t num_heads,
                               uint32_t head_dim) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total = num_heads * head_dim;

  if (idx >= total) return;

  uint32_t head_idx = idx / head_dim;
  uint32_t feat_idx = idx % head_dim;

  uint32_t src_off = head_idx * head_dim + feat_idx;
  uint32_t dst_off = head_idx * (max_seq_len * head_dim) + pos * head_dim + feat_idx;

  cache_k[dst_off] = new_k[src_off];
  cache_v[dst_off] = new_v[src_off];
}

void launch_kv_update(hipStream_t stream, float *cache_k, float *cache_v,
                      const float *new_k, const float *new_v, uint32_t pos,
                      uint32_t max_seq_len, uint32_t num_heads,
                      uint32_t head_dim) {
  uint32_t total = num_heads * head_dim;
  uint32_t threads = 256;
  uint32_t blocks = (total + threads - 1) / threads;

  kv_update_kernel<<<blocks, threads, 0, stream>>>(
      cache_k, cache_v, new_k, new_v, pos, max_seq_len, num_heads, head_dim);
}

} // namespace gcore::rt::hip::kernels
