#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>

namespace gcore::rt::hip::kernels {

void launch_fused_rope_kv_update_decode(hipStream_t stream, float *q, float *k,
                                        float *v, float *cache_k,
                                        float *cache_v, const uint32_t *d_pos,
                                        uint32_t max_seq_len,
                                        uint32_t num_heads, uint32_t head_dim,
                                        float rope_base);

void launch_flash_attention_decode_fused_rope(
    hipStream_t stream,
    const float *Q, // non-rope
    const float *K, // in cache (already rope'd)
    const float *V, // in cache
    float *O, uint32_t num_heads, const uint32_t *d_pos, uint32_t max_seq_len,
    uint32_t head_dim, float scale, float rope_base);

} // namespace gcore::rt::hip::kernels
