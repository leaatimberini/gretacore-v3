#include "gcore/rt/hip/kernels/attention_kernels.hpp"
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static inline bool attn_align_trace_enabled() {
  const char *v = std::getenv("GRETA_TRACE_ATTN_ALIGN");
  if (v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y'))
    return true;
  const char *v2 = std::getenv("GRETA_TRACE_ATTN_DECODE_VERIFY");
  return v2 && (v2[0] == '1' || v2[0] == 'y' || v2[0] == 'Y');
}

static inline void attn_log_align(const char *tag, const void *ptr,
                                  size_t align) {
  if (!ptr)
    return;
  if (reinterpret_cast<uintptr_t>(ptr) % align != 0) {
    std::fprintf(stderr,
                 "[ATTN_ALIGN] %s ptr=%p not %zub aligned\n", tag, ptr,
                 align);
  }
}

namespace gcore::rt::hip::kernels {

__device__ void rope_logic(float *x, uint32_t seq_len, uint32_t num_heads,
                           uint32_t head_dim, float base, uint32_t pos_offset) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t half_dim = head_dim / 2;
  uint32_t total_pairs = seq_len * num_heads * half_dim;
  
  if (idx >= total_pairs) return;

  uint32_t pair_idx = idx % half_dim;
  uint32_t head_idx = (idx / half_dim) % num_heads;
  uint32_t s_batch = idx / (half_dim * num_heads);
  uint32_t pos = pos_offset + s_batch;

  float theta = (float)pos * powf(base, -2.0f * (float)pair_idx / (float)head_dim);
  float cos_val = cosf(theta);
  float sin_val = sinf(theta);

  uint32_t base_idx = s_batch * (num_heads * head_dim) + head_idx * head_dim + pair_idx;
  
  float v0 = x[base_idx];
  float v1 = x[base_idx + half_dim];

  x[base_idx] = v0 * cos_val - v1 * sin_val;
  x[base_idx + half_dim] = v0 * sin_val + v1 * cos_val;
}

__global__ void rope_kernel(float *x, uint32_t seq_len, uint32_t num_heads, 
                            uint32_t head_dim, float base, uint32_t pos_offset) {
  rope_logic(x, seq_len, num_heads, head_dim, base, pos_offset);
}

__global__ void rope_kernel_p(float *x, uint32_t seq_len, uint32_t num_heads,
                               uint32_t head_dim, float base,
                               const uint32_t *pos_ptr) {
  rope_logic(x, seq_len, num_heads, head_dim, base, *pos_ptr);
}

void launch_rope(hipStream_t stream, float *x, uint32_t seq_len, 
                 uint32_t num_heads, uint32_t head_dim, float base, uint32_t pos_offset) {
  uint32_t half_dim = head_dim / 2;
  uint32_t total_pairs = seq_len * num_heads * half_dim;
  int block_size = 256;
  int grid_size = (total_pairs + block_size - 1) / block_size;
  
  rope_kernel<<<grid_size, block_size, 0, stream>>>(x, seq_len, num_heads, head_dim, base, pos_offset);
}

void launch_rope(hipStream_t stream, float *x, uint32_t seq_len,
                 uint32_t num_heads, uint32_t head_dim, float base,
                 const uint32_t *d_pos) {
  uint32_t half_dim = head_dim / 2;
  uint32_t total_pairs = seq_len * num_heads * half_dim;
  int block_size = 256;
  int grid_size = (total_pairs + block_size - 1) / block_size;

  rope_kernel_p<<<grid_size, block_size, 0, stream>>>(x, seq_len, num_heads, head_dim, base, d_pos);
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

__device__ void kv_update_logic(float *cache_k, float *cache_v, const float *new_k,
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

__global__ void kv_update_kernel(float *cache_k, float *cache_v, const float *new_k,
                               const float *new_v, uint32_t pos,
                               uint32_t max_seq_len, uint32_t num_heads,
                               uint32_t head_dim) {
  kv_update_logic(cache_k, cache_v, new_k, new_v, pos, max_seq_len, num_heads, head_dim);
}

__global__ void kv_update_kernel_p(float *cache_k, float *cache_v, const float *new_k,
                                 const float *new_v, const uint32_t *pos_ptr,
                                 uint32_t max_seq_len, uint32_t num_heads,
                                 uint32_t head_dim) {
  kv_update_logic(cache_k, cache_v, new_k, new_v, *pos_ptr, max_seq_len, num_heads, head_dim);
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

void launch_kv_update(hipStream_t stream, float *cache_k, float *cache_v,
                      const float *new_k, const float *new_v, 
                      const uint32_t *d_pos, uint32_t max_seq_len,
                      uint32_t num_heads, uint32_t head_dim) {
  uint32_t total = num_heads * head_dim;
  uint32_t threads = 256;
  uint32_t blocks = (total + threads - 1) / threads;

  kv_update_kernel_p<<<blocks, threads, 0, stream>>>(
      cache_k, cache_v, new_k, new_v, d_pos, max_seq_len, num_heads, head_dim);
}

#define FLASH_BLOCK_SIZE 64
#define FLASH_HEAD_DIM 128

__device__ __forceinline__ float round_fp16_device(float x) {
  return __half2float(__float2half_rn(x));
}

__device__ void flash_attention_decode_logic(
    const float *__restrict__ Q,      // [num_heads, head_dim]
    const float *__restrict__ K,      // [num_heads_kv, max_seq, head_dim]
    const float *__restrict__ V,      // [num_heads_kv, max_seq, head_dim]
    float *__restrict__ O,            // [num_heads, head_dim]
    uint32_t num_heads, uint32_t num_heads_kv, uint32_t seq_len,
    uint32_t max_seq_len, uint32_t head_dim, float scale, int accum_mode) {

  uint32_t head = blockIdx.x;
  uint32_t tid = threadIdx.x;
  
  if (head >= num_heads) return;
  if (num_heads_kv == 0) return;
  uint32_t group = num_heads / num_heads_kv;
  uint32_t kv_head = (group > 0) ? (head / group) : 0;
  if (kv_head >= num_heads_kv) return;
  
  __shared__ float s_score[FLASH_BLOCK_SIZE];
  __shared__ float s_max;
  __shared__ float s_sum;
  __shared__ float s_output[FLASH_HEAD_DIM];
  
  if (tid < head_dim) s_output[tid] = 0.0f;
  if (tid == 0) { s_max = -INFINITY; s_sum = 0.0f; }
  __syncthreads();

  const float *q_ptr = Q + head * head_dim;
  const float *k_ptr = K + kv_head * max_seq_len * head_dim;
  const float *v_ptr = V + kv_head * max_seq_len * head_dim;
  float *o_ptr = O + head * head_dim;

  for (uint32_t k_start = 0; k_start < seq_len; k_start += FLASH_BLOCK_SIZE) {
    uint32_t k_end = min(k_start + FLASH_BLOCK_SIZE, seq_len);
    uint32_t block_len = k_end - k_start;
    
    if (tid < block_len) {
      uint32_t k_idx = k_start + tid;
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_dim; ++d) {
        float prod = q_ptr[d] * k_ptr[k_idx * head_dim + d];
        if (accum_mode == 1) {
          prod = round_fp16_device(prod);
          dot = round_fp16_device(dot + prod);
        } else {
          dot = fmaf(q_ptr[d], k_ptr[k_idx * head_dim + d], dot);
        }
      }
      float s = dot * scale;
      s_score[tid] = (accum_mode == 1) ? round_fp16_device(s) : s;
    } else if (tid < FLASH_BLOCK_SIZE) {
      s_score[tid] = -INFINITY;
    }
    __syncthreads();
    
    for (uint32_t stride = FLASH_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
      if (tid < stride) s_score[tid] = fmaxf(s_score[tid], s_score[tid + stride]);
      __syncthreads();
    }
    float new_max = fmaxf(s_max, s_score[0]);
    __syncthreads();
    
    if (tid < block_len) {
      uint32_t k_idx = k_start + tid;
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_dim; ++d) {
        float prod = q_ptr[d] * k_ptr[k_idx * head_dim + d];
        if (accum_mode == 1) {
          prod = round_fp16_device(prod);
          dot = round_fp16_device(dot + prod);
        } else {
          dot = fmaf(q_ptr[d], k_ptr[k_idx * head_dim + d], dot);
        }
      }
      float s = expf(dot * scale - new_max);
      s_score[tid] = (accum_mode == 1) ? round_fp16_device(s) : s;
    } else if (tid < FLASH_BLOCK_SIZE) {
      s_score[tid] = 0.0f;
    }
    __syncthreads();
    
    for (uint32_t stride = FLASH_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
      if (tid < stride) s_score[tid] += s_score[tid + stride];
      __syncthreads();
    }
    
    float correction = expf(s_max - new_max);
    if (tid == 0) {
      s_sum = correction * s_sum + s_score[0];
    }
    __syncthreads();
    
    if (tid == 0) s_max = new_max;
    __syncthreads();

    if (tid < block_len) {
      uint32_t k_idx = k_start + tid;
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_dim; ++d) {
        float prod = q_ptr[d] * k_ptr[k_idx * head_dim + d];
        if (accum_mode == 1) {
          prod = round_fp16_device(prod);
          dot = round_fp16_device(dot + prod);
        } else {
          dot = fmaf(q_ptr[d], k_ptr[k_idx * head_dim + d], dot);
        }
      }
      float s = expf(dot * scale - s_max);
      s_score[tid] = (accum_mode == 1) ? round_fp16_device(s) : s;
    } else if (tid < FLASH_BLOCK_SIZE) {
      s_score[tid] = 0.0f;
    }
    __syncthreads();
    
    if (tid < head_dim) {
      float o_update = correction * s_output[tid];
      for (uint32_t ki = 0; ki < block_len; ++ki) {
        float term = s_score[ki] * v_ptr[(k_start + ki) * head_dim + tid];
        if (accum_mode == 1) {
          term = round_fp16_device(term);
          o_update = round_fp16_device(o_update + term);
        } else {
          o_update = fmaf(s_score[ki], v_ptr[(k_start + ki) * head_dim + tid],
                          o_update);
        }
      }
      s_output[tid] = o_update;
    }
    __syncthreads();
  }

  if (tid < head_dim) o_ptr[tid] = s_output[tid] / s_sum;
}

__global__ void flash_attention_decode_kernel(
    const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
    float *__restrict__ O, uint32_t num_heads, uint32_t num_heads_kv,
    uint32_t seq_len, uint32_t max_seq_len, uint32_t head_dim, float scale,
    int accum_mode) {
  flash_attention_decode_logic(Q, K, V, O, num_heads, num_heads_kv, seq_len,
                               max_seq_len, head_dim, scale, accum_mode);
}

__global__ void flash_attention_decode_kernel_p(
    const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
    float *__restrict__ O, uint32_t num_heads, uint32_t num_heads_kv,
    const uint32_t *pos_ptr, uint32_t max_seq_len, uint32_t head_dim,
    float scale, int accum_mode) {
  flash_attention_decode_logic(Q, K, V, O, num_heads, num_heads_kv,
                               *pos_ptr + 1,
                               max_seq_len, head_dim, scale, accum_mode);
}

void launch_flash_attention_decode(hipStream_t stream, 
                                   const float *Q, const float *K, const float *V,
                                   float *O, uint32_t num_heads,
                                   uint32_t num_heads_kv, uint32_t seq_len,
                                   uint32_t max_seq_len, uint32_t head_dim,
                                   float scale, int accum_mode) {
  if (head_dim > FLASH_HEAD_DIM) {
    if (attn_align_trace_enabled()) {
      std::fprintf(stderr,
                   "[ATTN_DECODE] head_dim=%u exceeds FLASH_HEAD_DIM=%u\n",
                   head_dim, FLASH_HEAD_DIM);
    }
    return;
  }
  if (attn_align_trace_enabled()) {
    attn_log_align("Q", Q, 16);
    attn_log_align("K", K, 16);
    attn_log_align("V", V, 16);
    attn_log_align("O", O, 16);
    attn_log_align("Q", Q, 32);
    attn_log_align("K", K, 32);
    attn_log_align("V", V, 32);
    attn_log_align("O", O, 32);
  }
  dim3 grid(num_heads);
  dim3 block(max(FLASH_BLOCK_SIZE, (int)head_dim));
  flash_attention_decode_kernel<<<grid, block, 0, stream>>>(
      Q, K, V, O, num_heads, num_heads_kv, seq_len, max_seq_len, head_dim,
      scale, accum_mode);
}

void launch_flash_attention_decode(hipStream_t stream, 
                                   const float *Q, const float *K, const float *V,
                                   float *O, uint32_t num_heads,
                                   uint32_t num_heads_kv,
                                   const uint32_t *d_pos,
                                   uint32_t max_seq_len, uint32_t head_dim,
                                   float scale, int accum_mode) {
  if (head_dim > FLASH_HEAD_DIM) {
    if (attn_align_trace_enabled()) {
      std::fprintf(stderr,
                   "[ATTN_DECODE] head_dim=%u exceeds FLASH_HEAD_DIM=%u\n",
                   head_dim, FLASH_HEAD_DIM);
    }
    return;
  }
  if (attn_align_trace_enabled()) {
    attn_log_align("Q", Q, 16);
    attn_log_align("K", K, 16);
    attn_log_align("V", V, 16);
    attn_log_align("O", O, 16);
    attn_log_align("Q", Q, 32);
    attn_log_align("K", K, 32);
    attn_log_align("V", V, 32);
    attn_log_align("O", O, 32);
  }
  dim3 grid(num_heads);
  dim3 block(max(FLASH_BLOCK_SIZE, (int)head_dim));
  flash_attention_decode_kernel_p<<<grid, block, 0, stream>>>(
      Q, K, V, O, num_heads, num_heads_kv, d_pos, max_seq_len, head_dim,
      scale, accum_mode);
}

__global__ void attn_softmax_trace_kernel(
    const float *__restrict__ Q, const float *__restrict__ K_cache,
    uint32_t num_heads, uint32_t num_heads_kv, uint32_t head_dim,
    uint32_t seq_len, uint32_t max_seq_len, uint32_t head,
    uint32_t window_start, uint32_t window_len, float scale,
    float *__restrict__ qk_out, float *__restrict__ softmax_out,
    float *__restrict__ stats_out) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  if (!Q || !K_cache || !qk_out || !softmax_out || !stats_out)
    return;
  if (num_heads == 0 || head_dim == 0 || seq_len == 0)
    return;
  if (num_heads_kv == 0)
    return;

  const uint32_t group = num_heads / num_heads_kv;
  const uint32_t kv_head = (group > 0) ? (head / group) : 0;
  if (head >= num_heads || kv_head >= num_heads_kv)
    return;

  const float *q_ptr = Q + head * head_dim;
  const float *k_ptr = K_cache + kv_head * max_seq_len * head_dim;

  float max_qk = -INFINITY;
  float second_qk = -INFINITY;
  for (uint32_t t = 0; t < seq_len; ++t) {
    const float *k_t = k_ptr + t * head_dim;
    float dot = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot = fmaf(q_ptr[d], k_t[d], dot);
    }
    float qk = dot * scale;
    if (qk > max_qk) {
      second_qk = max_qk;
      max_qk = qk;
    } else if (qk > second_qk) {
      second_qk = qk;
    }
  }

  double sumexp = 0.0;
  double sumexp_log = 0.0;
  for (uint32_t t = 0; t < seq_len; ++t) {
    const float *k_t = k_ptr + t * head_dim;
    float dot = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot = fmaf(q_ptr[d], k_t[d], dot);
    }
    float qk = dot * scale;
    float e = expf(qk - max_qk);
    sumexp += static_cast<double>(e);
    sumexp_log += static_cast<double>(e) * static_cast<double>(qk - max_qk);
  }

  float sumexp_f = static_cast<float>(sumexp);
  float inv_sum = (sumexp > 0.0) ? static_cast<float>(1.0 / sumexp) : 0.0f;
  float top1_prob = inv_sum;
  float top2_prob =
      (sumexp > 0.0 && isfinite(second_qk))
          ? expf(second_qk - max_qk) * inv_sum
          : 0.0f;
  float entropy =
      (sumexp > 0.0)
          ? static_cast<float>(log(sumexp) - sumexp_log / sumexp)
          : 0.0f;

  for (uint32_t i = 0; i < window_len; ++i) {
    uint32_t t = window_start + i;
    if (t >= seq_len) {
      qk_out[i] = 0.0f;
      softmax_out[i] = 0.0f;
      continue;
    }
    const float *k_t = k_ptr + t * head_dim;
    float dot = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot = fmaf(q_ptr[d], k_t[d], dot);
    }
    float qk = dot * scale;
    qk_out[i] = qk;
    softmax_out[i] = (sumexp > 0.0) ? expf(qk - max_qk) * inv_sum : 0.0f;
  }

  stats_out[0] = max_qk;
  stats_out[1] = sumexp_f;
  stats_out[2] = top1_prob;
  stats_out[3] = top2_prob;
  stats_out[4] = entropy;
}

void launch_attn_softmax_trace(hipStream_t stream, const float *Q,
                               const float *K_cache, uint32_t num_heads,
                               uint32_t num_heads_kv, uint32_t head_dim,
                               uint32_t seq_len, uint32_t max_seq_len,
                               uint32_t head, uint32_t window_start,
                               uint32_t window_len, float scale,
                               float *qk_out, float *softmax_out,
                               float *stats_out) {
  dim3 grid(1);
  dim3 block(1);
  attn_softmax_trace_kernel<<<grid, block, 0, stream>>>(
      Q, K_cache, num_heads, num_heads_kv, head_dim, seq_len, max_seq_len, head,
      window_start, window_len, scale, qk_out, softmax_out, stats_out);
}

__global__ void attn_vacc_vsample_kernel(
    const float *__restrict__ V_cache, uint32_t num_heads,
    uint32_t num_heads_kv, uint32_t head_dim, uint32_t seq_len,
    uint32_t max_seq_len, uint32_t head, uint32_t window_start,
    uint32_t window_len, uint32_t dims_sample, float *__restrict__ v_row_out,
    float *__restrict__ v_col_out) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  if (!V_cache || !v_row_out || !v_col_out)
    return;
  if (num_heads == 0 || num_heads_kv == 0 || head_dim == 0 || seq_len == 0)
    return;
  if (head >= num_heads)
    return;
  if (dims_sample == 0 || window_len == 0)
    return;

  const uint32_t group = num_heads / num_heads_kv;
  const uint32_t kv_head = (group > 0) ? (head / group) : 0;
  if (kv_head >= num_heads_kv)
    return;

  const float *v_ptr = V_cache + kv_head * max_seq_len * head_dim;

  for (uint32_t i = 0; i < window_len; ++i) {
    uint32_t t = window_start + i;
    for (uint32_t d = 0; d < dims_sample; ++d) {
      float row_val = 0.0f;
      float col_val = 0.0f;
      if (t < seq_len) {
        row_val = v_ptr[t * head_dim + d];
        col_val = v_ptr[d * max_seq_len + t];
      }
      v_row_out[i * dims_sample + d] = row_val;
      v_col_out[i * dims_sample + d] = col_val;
    }
  }
}

void launch_attn_vacc_vsample(hipStream_t stream, const float *V_cache,
                              uint32_t num_heads, uint32_t num_heads_kv,
                              uint32_t head_dim, uint32_t seq_len,
                              uint32_t max_seq_len, uint32_t head,
                              uint32_t window_start, uint32_t window_len,
                              uint32_t dims_sample, float *v_row_out,
                              float *v_col_out) {
  dim3 grid(1);
  dim3 block(1);
  attn_vacc_vsample_kernel<<<grid, block, 0, stream>>>(
      V_cache, num_heads, num_heads_kv, head_dim, seq_len, max_seq_len, head,
      window_start, window_len, dims_sample, v_row_out, v_col_out);
}



__global__ void flash_attention_prefill_kernel(
    const float *__restrict__ Q,      // [seq_len, num_heads, head_dim]
    const float *__restrict__ K,      // [seq_len, num_heads_kv, head_dim]
    const float *__restrict__ V,      // [seq_len, num_heads_kv, head_dim]
    float *__restrict__ O,            // [seq_len, num_heads, head_dim]
    uint32_t seq_len, uint32_t num_heads, uint32_t num_heads_kv,
    uint32_t head_dim, float scale, bool causal) {

  uint32_t head = blockIdx.x;
  uint32_t q_block = blockIdx.y;
  uint32_t tid = threadIdx.x;
  if (num_heads_kv == 0) return;
  uint32_t group = num_heads / num_heads_kv;
  uint32_t kv_head = (group > 0) ? (head / group) : 0;
  if (kv_head >= num_heads_kv) return;
  
  uint32_t q_start = q_block * FLASH_BLOCK_SIZE;
  if (q_start >= seq_len) return;
  uint32_t q_end = min(q_start + FLASH_BLOCK_SIZE, seq_len);
  
  uint32_t q_idx = q_start + tid;
  bool valid_q = (q_idx < q_end);
  
  float m = -INFINITY;
  float l = 0.0f;
  float o[FLASH_HEAD_DIM] = {0}; // Fixed buffer size
  
  uint32_t max_k = causal ? (q_idx + 1) : seq_len;
  
  for (uint32_t k_start = 0; k_start < max_k; k_start += FLASH_BLOCK_SIZE) {
    uint32_t k_end = min(k_start + FLASH_BLOCK_SIZE, max_k);
    
    for (uint32_t k_idx = k_start; k_idx < k_end; ++k_idx) {
      if (!valid_q) continue;
      
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_dim; ++d) {
        uint32_t q_off = q_idx * num_heads * head_dim + head * head_dim + d;
        uint32_t k_off =
            k_idx * num_heads_kv * head_dim + kv_head * head_dim + d;
        dot += Q[q_off] * K[k_off];
      }
      float s = dot * scale;
      
      float m_new = fmaxf(m, s);
      float p = expf(s - m_new);
      float correction = expf(m - m_new);
      l = correction * l + p;
      
      for (uint32_t d = 0; d < head_dim; ++d) {
        uint32_t v_off =
            k_idx * num_heads_kv * head_dim + kv_head * head_dim + d;
        o[d] = correction * o[d] + p * V[v_off];
      }
      m = m_new;
    }
  }
  
  if (valid_q) {
    for (uint32_t d = 0; d < head_dim; ++d) {
      uint32_t o_off = q_idx * num_heads * head_dim + head * head_dim + d;
      O[o_off] = o[d] / l;
    }
  }
}

void launch_flash_attention_prefill(hipStream_t stream,
                                    const float *Q, const float *K, const float *V,
                                    float *O, uint32_t seq_len, uint32_t num_heads,
                                    uint32_t num_heads_kv, uint32_t head_dim,
                                    float scale, bool causal) {
  dim3 grid(num_heads, (seq_len + FLASH_BLOCK_SIZE - 1) / FLASH_BLOCK_SIZE);
  dim3 block(FLASH_BLOCK_SIZE);
  flash_attention_prefill_kernel<<<grid, block, 0, stream>>>(
      Q, K, V, O, seq_len, num_heads, num_heads_kv, head_dim, scale, causal);
}

} // namespace gcore::rt::hip::kernels
