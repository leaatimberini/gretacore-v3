#include "gcore/rt/hip/kernels/fused_attention_kernels.hpp"
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

static inline void attn_log_align(const char *tag, const void *ptr, size_t align) {
    if (!ptr) return;
    if (reinterpret_cast<uintptr_t>(ptr) % align != 0) {
        std::fprintf(stderr, "[ATTN_ALIGN] %s ptr=%p not %zub aligned\n", tag, ptr, align);
    }
}

namespace gcore::rt::hip::kernels {

#define FLASH_BLOCK_SIZE 64
#define FLASH_HEAD_DIM 128

__device__ void apply_rope_single(float& v0, float& v1, int pair_idx, int head_dim, int pos, float base) {
    float theta = (float)pos * powf(base, -2.0f * (float)pair_idx / (float)head_dim);
    float cos_val = cosf(theta);
    float sin_val = sinf(theta);
    float old_v0 = v0;
    v0 = old_v0 * cos_val - v1 * sin_val;
    v1 = old_v0 * sin_val + v1 * cos_val;
}

__device__ void fused_rope_kv_update_decode_logic(
    float* q, float* k, float* v,
    float* cache_k, float* cache_v,
    uint32_t pos, uint32_t max_seq_len,
    uint32_t num_heads, uint32_t head_dim, float rope_base) {

    uint32_t head = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (head >= num_heads || tid >= head_dim) return;

    uint32_t feat_idx = tid;
    uint32_t head_off = head * head_dim;
    
    // 1. V Update: Direct Copy
    float val_v = v[head_off + feat_idx];
    uint32_t dst_off_v = head * (max_seq_len * head_dim) + pos * head_dim + feat_idx;
    cache_v[dst_off_v] = val_v;

    // 2. K Update: RoPE + Write
    if (feat_idx < head_dim / 2) {
        float k0 = k[head_off + feat_idx];
        float k1 = k[head_off + feat_idx + head_dim / 2];
        
        apply_rope_single(k0, k1, feat_idx, head_dim, (int)pos, rope_base);
        
        uint32_t dst_off_k = head * (max_seq_len * head_dim) + pos * head_dim + feat_idx;
        cache_k[dst_off_k] = k0;
        cache_k[dst_off_k + head_dim / 2] = k1;
    }
}

__global__ void fused_rope_kv_update_decode_kernel(
    float* q, float* k, float* v,
    float* cache_k, float* cache_v,
    uint32_t pos, uint32_t max_seq_len,
    uint32_t num_heads, uint32_t head_dim, float rope_base) {
    fused_rope_kv_update_decode_logic(q, k, v, cache_k, cache_v, pos, max_seq_len, num_heads, head_dim, rope_base);
}

__global__ void fused_rope_kv_update_decode_kernel_p(
    float* q, float* k, float* v,
    float* cache_k, float* cache_v,
    const uint32_t* pos_ptr, uint32_t max_seq_len,
    uint32_t num_heads, uint32_t head_dim, float rope_base) {
    fused_rope_kv_update_decode_logic(q, k, v, cache_k, cache_v, *pos_ptr, max_seq_len, num_heads, head_dim, rope_base);
}

void launch_fused_rope_kv_update_decode(hipStream_t stream,
                                       float* q, float* k, float* v,
                                       float* cache_k, float* cache_v,
                                       const uint32_t* d_pos, uint32_t max_seq_len,
                                       uint32_t num_heads, uint32_t head_dim, float rope_base) {
    if (attn_align_trace_enabled()) {
        attn_log_align("Q", q, 16);
        attn_log_align("K", k, 16);
        attn_log_align("V", v, 16);
        attn_log_align("K_cache", cache_k, 16);
        attn_log_align("V_cache", cache_v, 16);
        attn_log_align("Q", q, 32);
        attn_log_align("K", k, 32);
        attn_log_align("V", v, 32);
        attn_log_align("K_cache", cache_k, 32);
        attn_log_align("V_cache", cache_v, 32);
        if (head_dim > FLASH_HEAD_DIM) {
            std::fprintf(stderr,
                         "[ATTN_DECODE] head_dim=%u exceeds FLASH_HEAD_DIM=%u\n",
                         head_dim, FLASH_HEAD_DIM);
        }
    }
    dim3 grid(num_heads);
    dim3 block(head_dim);
    fused_rope_kv_update_decode_kernel_p<<<grid, block, 0, stream>>>(
        q, k, v, cache_k, cache_v, d_pos, max_seq_len, num_heads, head_dim, rope_base);
}

__device__ void flash_attention_decode_fused_rope_logic(
    const float *__restrict__ Q,      // non-rope
    const float *__restrict__ K_cache, 
    const float *__restrict__ V_cache,
    float *__restrict__ O,
    uint32_t num_heads, uint32_t seq_len, uint32_t max_seq_len, 
    uint32_t head_dim, float scale, float rope_base) {

    uint32_t head = blockIdx.x;
    uint32_t tid = threadIdx.x;
    
    __shared__ float s_q[FLASH_HEAD_DIM];
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_score[FLASH_BLOCK_SIZE];
    __shared__ float s_output[FLASH_HEAD_DIM];

    // Load Q and apply RoPE in shared memory
    if (tid < head_dim) {
        s_q[tid] = Q[head * head_dim + tid];
    }
    __syncthreads();

    if (tid < head_dim / 2) {
        apply_rope_single(s_q[tid], s_q[tid + head_dim / 2], tid, head_dim, seq_len - 1, rope_base);
    }
    __syncthreads();

    if (tid < head_dim) s_output[tid] = 0.0f;
    if (tid == 0) { s_max = -INFINITY; s_sum = 0.0f; }
    __syncthreads();

    const float *k_ptr = K_cache + head * max_seq_len * head_dim;
    const float *v_ptr = V_cache + head * max_seq_len * head_dim;
    float *o_ptr = O + head * head_dim;

    for (uint32_t k_start = 0; k_start < seq_len; k_start += FLASH_BLOCK_SIZE) {
        uint32_t k_end = min(k_start + FLASH_BLOCK_SIZE, seq_len);
        uint32_t block_len = k_end - k_start;
        
        if (tid < block_len) {
            uint32_t k_idx = k_start + tid;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += s_q[d] * k_ptr[k_idx * head_dim + d];
            }
            s_score[tid] = dot * scale;
        } else if (tid < FLASH_BLOCK_SIZE) {
            s_score[tid] = -INFINITY;
        }
        __syncthreads();
        
        float local_max = s_score[tid];
        for (int offset = FLASH_BLOCK_SIZE/2; offset > 0; offset /= 2) {
            float other = __shfl_xor(local_max, offset);
            local_max = fmaxf(local_max, other);
        }
        float new_max = fmaxf(s_max, local_max);

        float exp_score = 0.0f;
        if (tid < block_len) {
            exp_score = expf(s_score[tid] - new_max);
            s_score[tid] = exp_score;
        } else {
            s_score[tid] = 0.0f;
        }
        __syncthreads();

        float local_sum = s_score[tid];
        for (int offset = FLASH_BLOCK_SIZE/2; offset > 0; offset /= 2) {
            local_sum += __shfl_xor(local_sum, offset);
        }
        
        float correction = expf(s_max - new_max);
        if (tid == 0) {
            s_sum = correction * s_sum + local_sum;
            s_max = new_max;
        }
        __syncthreads();

        if (tid < head_dim) {
            float o_acc = correction * s_output[tid];
            for (uint32_t ki = 0; ki < block_len; ++ki) {
                o_acc += s_score[ki] * v_ptr[(k_start + ki) * head_dim + tid];
            }
            s_output[tid] = o_acc;
        }
        __syncthreads();
    }

    if (tid < head_dim) o_ptr[tid] = s_output[tid] / s_sum;
}

__global__ void flash_attention_decode_fused_rope_kernel(
    const float *__restrict__ Q, const float *__restrict__ K_cache, const float *__restrict__ V_cache,
    float *__restrict__ O, uint32_t num_heads, uint32_t seq_len, uint32_t max_seq_len, 
    uint32_t head_dim, float scale, float rope_base) {
    flash_attention_decode_fused_rope_logic(Q, K_cache, V_cache, O, num_heads, seq_len, max_seq_len, head_dim, scale, rope_base);
}

__global__ void flash_attention_decode_fused_rope_kernel_p(
    const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
    float *__restrict__ O, uint32_t num_heads, const uint32_t *pos_ptr, uint32_t max_seq_len, 
    uint32_t head_dim, float scale, float rope_base) {
    flash_attention_decode_fused_rope_logic(Q, K, V, O, num_heads, *pos_ptr + 1, max_seq_len, head_dim, scale, rope_base);
}

void launch_flash_attention_decode_fused_rope(hipStream_t stream, 
                                              const float *Q, const float *K, const float *V,
                                              float *O, uint32_t num_heads, const uint32_t *d_pos,
                                              uint32_t max_seq_len, uint32_t head_dim, float scale,
                                              float rope_base) {
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
    flash_attention_decode_fused_rope_kernel_p<<<grid, block, 0, stream>>>(
        Q, K, V, O, num_heads, d_pos, max_seq_len, head_dim, scale, rope_base);
}

} // namespace gcore::rt::hip::kernels
