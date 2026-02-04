#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

namespace gcore::rt::hip::kernels {

/**
 * @brief Apply Rotary Position Embeddings (RoPE).
 *
 * @param stream HIP stream.
 * @param x Input/Output tensor (seq_len, num_heads, head_dim).
 * @param seq_len Sequence length.
 * @param num_heads Number of attention heads.
 * @param head_dim Dimension of each head (must be even).
 * @param base Frequency base (e.g., 10000.0).
 */
void launch_rope(hipStream_t stream, float *x, uint32_t seq_len,
                 uint32_t num_heads, uint32_t head_dim, float base,
                 uint32_t pos_offset);

void launch_rope(hipStream_t stream, float *x, uint32_t seq_len,
                 uint32_t num_heads, uint32_t head_dim, float base,
                 const uint32_t *d_pos);

/**
 * @brief Apply Causal Masking (inplace).
 * Sets values where col > row to a large negative number.
 *
 * @param stream HIP stream.
 * @param data Matrix of shape (seq_len, seq_len).
 * @param seq_len Sequence length.
 * @param mask_val Value to use for masked elements (usually -1e9f).
 */
void launch_causal_mask(hipStream_t stream, float *data, uint32_t seq_len,
                        float mask_val);

/**
 * @brief Update KV-Cache with new token projections.
 *
 * @param stream HIP stream.
 * @param cache_k Global Key cache [num_heads, max_seq_len, head_dim].
 * @param cache_v Global Value cache [num_heads, max_seq_len, head_dim].
 * @param new_k New Key projections [num_heads, head_dim].
 * @param new_v New Value projections [num_heads, head_dim].
 * @param pos Current sequence position.
 * @param max_seq_len Maximum sequence length (cache capacity).
 * @param num_heads Number of heads.
 * @param head_dim Dimension of each head.
 */
void launch_kv_update(hipStream_t stream, float *cache_k, float *cache_v,
                      const float *new_k, const float *new_v, uint32_t pos,
                      uint32_t max_seq_len, uint32_t num_heads,
                      uint32_t head_dim);

void launch_kv_update(hipStream_t stream, float *cache_k, float *cache_v,
                      const float *new_k, const float *new_v,
                      const uint32_t *d_pos, uint32_t max_seq_len,
                      uint32_t num_heads, uint32_t head_dim);

/**
 * @brief FlashAttention v2 for decode mode (single query against KV cache).
 *
 * Computes attention with O(N) memory using online softmax.
 *
 * @param stream HIP stream.
 * @param Q Query tensor [num_heads, head_dim].
 * @param K Key cache [num_heads, seq_len, head_dim].
 * @param V Value cache [num_heads, seq_len, head_dim].
 * @param O Output tensor [num_heads, head_dim].
 * @param num_heads Number of attention heads.
 * @param seq_len Current sequence length in cache.
 * @param head_dim Dimension of each head.
 * @param scale Attention scale factor (1/sqrt(head_dim)).
 */
void launch_flash_attention_decode(hipStream_t stream, const float *Q,
                                   const float *K, const float *V, float *O,
                                   uint32_t num_heads, uint32_t num_heads_kv,
                                   uint32_t seq_len, uint32_t max_seq_len,
                                   uint32_t head_dim, float scale,
                                   int accum_mode = 0);

void launch_flash_attention_decode(hipStream_t stream, const float *Q,
                                   const float *K, const float *V, float *O,
                                   uint32_t num_heads, uint32_t num_heads_kv,
                                   const uint32_t *d_pos,
                                   uint32_t max_seq_len, uint32_t head_dim,
                                   float scale, int accum_mode = 0);

void launch_attn_softmax_trace(hipStream_t stream, const float *Q,
                               const float *K_cache, uint32_t num_heads,
                               uint32_t num_heads_kv, uint32_t head_dim,
                               uint32_t seq_len, uint32_t max_seq_len,
                               uint32_t head, uint32_t window_start,
                               uint32_t window_len, float scale,
                               float *qk_out, float *softmax_out,
                               float *stats_out);

void launch_attn_vacc_vsample(hipStream_t stream, const float *V_cache,
                              uint32_t num_heads, uint32_t num_heads_kv,
                              uint32_t head_dim, uint32_t seq_len,
                              uint32_t max_seq_len, uint32_t head,
                              uint32_t window_start, uint32_t window_len,
                              uint32_t dims_sample, float *v_row_out,
                              float *v_col_out);

/**
 * @brief FlashAttention v2 for prefill mode (multiple queries).
 *
 * Computes attention with O(N) memory using online softmax.
 * Supports causal masking for autoregressive models.
 *
 * @param stream HIP stream.
 * @param Q Query tensor [seq_len, num_heads, head_dim].
 * @param K Key tensor [seq_len, num_heads, head_dim].
 * @param V Value tensor [seq_len, num_heads, head_dim].
 * @param O Output tensor [seq_len, num_heads, head_dim].
 * @param seq_len Sequence length.
 * @param num_heads Number of attention heads.
 * @param head_dim Dimension of each head.
 * @param scale Attention scale factor (1/sqrt(head_dim)).
 * @param causal Whether to apply causal masking.
 */
void launch_flash_attention_prefill(hipStream_t stream, const float *Q,
                                    const float *K, const float *V, float *O,
                                    uint32_t seq_len, uint32_t num_heads,
                                    uint32_t num_heads_kv, uint32_t head_dim,
                                    float scale, bool causal);

} // namespace gcore::rt::hip::kernels
