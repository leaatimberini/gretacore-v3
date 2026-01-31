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
                 uint32_t num_heads, uint32_t head_dim, float base);

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

} // namespace gcore::rt::hip::kernels
