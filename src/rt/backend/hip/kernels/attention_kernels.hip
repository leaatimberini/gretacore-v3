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

// =============================================================================
// FlashAttention v2 - Tiled Attention with Online Softmax
// =============================================================================
// Key insight: Compute attention block-by-block without materializing N×N matrix
// Memory: O(N) instead of O(N²)
// Algorithm:
//   For each query block Bq:
//     Initialize: O = 0, l = 0, m = -inf
//     For each key/value block Bk:
//       S = Bq @ Bk^T / sqrt(d)
//       m_new = max(m, rowmax(S))
//       P = exp(S - m_new)
//       l_new = exp(m - m_new) * l + rowsum(P)
//       O = exp(m - m_new) * O + P @ Bv
//       m = m_new, l = l_new
//     O = O / l
// =============================================================================

#define FLASH_BLOCK_SIZE 64  // Block size for Q, K, V tiles
#define FLASH_HEAD_DIM 128   // Must match model head_dim

// Simplified FlashAttention kernel for single-head, single-query (decode mode)
// Q: [1, head_dim], K_cache: [seq_len, head_dim], V_cache: [seq_len, head_dim]
// Output: [1, head_dim]
__global__ void flash_attention_decode_kernel(
    const float *__restrict__ Q,      // [num_heads, head_dim]
    const float *__restrict__ K,      // [num_heads, seq_len, head_dim]
    const float *__restrict__ V,      // [num_heads, seq_len, head_dim]
    float *__restrict__ O,            // [num_heads, head_dim]
    uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
    float scale) {

  // Each block handles one attention head
  uint32_t head = blockIdx.x;
  uint32_t tid = threadIdx.x;
  
  if (head >= num_heads) return;
  
  // Shared memory for partial results
  __shared__ float s_score[FLASH_BLOCK_SIZE];
  __shared__ float s_max;
  __shared__ float s_sum;
  __shared__ float s_output[FLASH_HEAD_DIM];
  
  // Initialize output
  if (tid < head_dim) {
    s_output[tid] = 0.0f;
  }
  if (tid == 0) {
    s_max = -INFINITY;
    s_sum = 0.0f;
  }
  __syncthreads();

  // Pointers for this head
  const float *q_ptr = Q + head * head_dim;
  const float *k_ptr = K + head * seq_len * head_dim;
  const float *v_ptr = V + head * seq_len * head_dim;
  float *o_ptr = O + head * head_dim;

  // Process K/V in blocks
  for (uint32_t k_start = 0; k_start < seq_len; k_start += FLASH_BLOCK_SIZE) {
    uint32_t k_end = min(k_start + FLASH_BLOCK_SIZE, seq_len);
    uint32_t block_len = k_end - k_start;
    
    // Compute attention scores for this block: S = Q @ K[k_start:k_end]^T
    if (tid < block_len) {
      uint32_t k_idx = k_start + tid;
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_dim; ++d) {
        dot += q_ptr[d] * k_ptr[k_idx * head_dim + d];
      }
      s_score[tid] = dot * scale;
    } else if (tid < FLASH_BLOCK_SIZE) {
      s_score[tid] = -INFINITY;
    }
    __syncthreads();
    
    // Find block max (parallel reduction)
    float block_max = s_score[tid < block_len ? tid : 0];
    for (uint32_t stride = FLASH_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        block_max = fmaxf(block_max, s_score[tid + stride]);
      }
      __syncthreads();
      if (tid < stride) {
        s_score[tid] = block_max;
      }
      __syncthreads();
    }
    float new_max = fmaxf(s_max, s_score[0]);
    __syncthreads();
    
    // Compute exp(score - new_max) and sum
    float my_exp = 0.0f;
    if (tid < block_len) {
      uint32_t k_idx = k_start + tid;
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_dim; ++d) {
        dot += q_ptr[d] * k_ptr[k_idx * head_dim + d];
      }
      my_exp = expf(dot * scale - new_max);
      s_score[tid] = my_exp;
    } else if (tid < FLASH_BLOCK_SIZE) {
      s_score[tid] = 0.0f;
    }
    __syncthreads();
    
    // Parallel sum reduction
    float block_sum = s_score[tid < FLASH_BLOCK_SIZE ? tid : 0];
    for (uint32_t stride = FLASH_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        block_sum += s_score[tid + stride];
      }
      __syncthreads();
      if (tid < stride) {
        s_score[tid] = block_sum;
      }
      __syncthreads();
    }
    
    // Update running statistics
    float correction = expf(s_max - new_max);
    if (tid == 0) {
      s_sum = correction * s_sum + s_score[0];
      s_max = new_max;
    }
    __syncthreads();
    
    // Update output: O = correction * O + P @ V[k_start:k_end]
    if (tid < head_dim) {
      float o_update = correction * s_output[tid];
      for (uint32_t ki = 0; ki < block_len; ++ki) {
        uint32_t k_idx = k_start + ki;
        // Recompute exp for this ki
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) {
          dot += q_ptr[d] * k_ptr[k_idx * head_dim + d];
        }
        float score_exp = expf(dot * scale - s_max);
        o_update += score_exp * v_ptr[k_idx * head_dim + tid];
      }
      s_output[tid] = o_update;
    }
    __syncthreads();
  }

  // Final normalization: O = O / sum
  if (tid < head_dim) {
    o_ptr[tid] = s_output[tid] / s_sum;
  }
}

void launch_flash_attention_decode(hipStream_t stream, 
                                   const float *Q, const float *K, const float *V,
                                   float *O, uint32_t num_heads, uint32_t seq_len, 
                                   uint32_t head_dim, float scale) {
  
  // One block per head, threads handle parallel work within head
  dim3 grid(num_heads);
  dim3 block(max(FLASH_BLOCK_SIZE, (int)head_dim));
  
  flash_attention_decode_kernel<<<grid, block, 0, stream>>>(
      Q, K, V, O, num_heads, seq_len, head_dim, scale);
}

// FlashAttention for prefill (multiple queries)
// Uses tiled matrix multiplication with online softmax
__global__ void flash_attention_prefill_kernel(
    const float *__restrict__ Q,      // [seq_len, num_heads, head_dim]
    const float *__restrict__ K,      // [seq_len, num_heads, head_dim]
    const float *__restrict__ V,      // [seq_len, num_heads, head_dim]
    float *__restrict__ O,            // [seq_len, num_heads, head_dim]
    uint32_t seq_len, uint32_t num_heads, uint32_t head_dim,
    float scale, bool causal) {

  // Grid: (num_heads, ceil(seq_len/BLOCK_SIZE))
  // Each block computes BLOCK_SIZE rows of output for one head
  
  uint32_t head = blockIdx.x;
  uint32_t q_block = blockIdx.y;
  uint32_t tid = threadIdx.x;
  
  uint32_t q_start = q_block * FLASH_BLOCK_SIZE;
  if (q_start >= seq_len) return;
  uint32_t q_end = min(q_start + FLASH_BLOCK_SIZE, seq_len);
  
  // Each thread handles one query position
  uint32_t q_idx = q_start + tid;
  bool valid_q = (q_idx < q_end);
  
  // Initialize running stats for this query
  float m = -INFINITY;  // Running max
  float l = 0.0f;       // Running sum of exp
  float o[FLASH_HEAD_DIM / 4] = {0};  // Partial output (assumes head_dim <= 128)
  
  // Process K/V in blocks
  uint32_t max_k = causal ? (q_idx + 1) : seq_len;
  
  for (uint32_t k_start = 0; k_start < max_k; k_start += FLASH_BLOCK_SIZE) {
    uint32_t k_end = min(k_start + FLASH_BLOCK_SIZE, max_k);
    
    // For each key in block
    for (uint32_t k_idx = k_start; k_idx < k_end; ++k_idx) {
      if (!valid_q) continue;
      
      // Compute Q[q_idx] @ K[k_idx]^T
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_dim; ++d) {
        uint32_t q_off = q_idx * num_heads * head_dim + head * head_dim + d;
        uint32_t k_off = k_idx * num_heads * head_dim + head * head_dim + d;
        dot += Q[q_off] * K[k_off];
      }
      float s = dot * scale;
      
      // Online softmax update
      float m_new = fmaxf(m, s);
      float p = expf(s - m_new);
      float correction = expf(m - m_new);
      l = correction * l + p;
      
      // Update output with V contribution
      for (uint32_t d = 0; d < head_dim && d < FLASH_HEAD_DIM / 4; ++d) {
        uint32_t v_off = k_idx * num_heads * head_dim + head * head_dim + d;
        o[d] = correction * o[d] + p * V[v_off];
      }
      m = m_new;
    }
  }
  
  // Final normalization and write output
  if (valid_q) {
    for (uint32_t d = 0; d < head_dim && d < FLASH_HEAD_DIM / 4; ++d) {
      uint32_t o_off = q_idx * num_heads * head_dim + head * head_dim + d;
      O[o_off] = o[d] / l;
    }
  }
}

void launch_flash_attention_prefill(hipStream_t stream,
                                    const float *Q, const float *K, const float *V,
                                    float *O, uint32_t seq_len, uint32_t num_heads, 
                                    uint32_t head_dim, float scale, bool causal) {
  
  dim3 grid(num_heads, (seq_len + FLASH_BLOCK_SIZE - 1) / FLASH_BLOCK_SIZE);
  dim3 block(FLASH_BLOCK_SIZE);
  
  flash_attention_prefill_kernel<<<grid, block, 0, stream>>>(
      Q, K, V, O, seq_len, num_heads, head_dim, scale, causal);
}

} // namespace gcore::rt::hip::kernels
