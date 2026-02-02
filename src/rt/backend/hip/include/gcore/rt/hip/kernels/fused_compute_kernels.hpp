#pragma once

#include <cstdint>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace gcore::rt::hip::kernels {

/**
 * Fused RMSNorm + QKV Projections (3x GEMV)
 * [Q, K, V] = Linear_QKV(RMSNorm(X, gamma))
 *
 * @param stream HIP stream
 * @param x Input activations [D]
 * @param gamma RMSNorm weights [D]
 * @param w_q Q weights [D, D]
 * @param w_k K weights [D, D]
 * @param w_v V weights [D, D]
 * @param q_out Output Q [D]
 * @param k_out Output K [D]
 * @param v_out Output V [D]
 * @param D Hidden dimension
 * @param eps RMSNorm epsilon
 */
void launch_fused_rmsnorm_qkv_gemv_f16(hipStream_t stream, const float *x,
                                       const float *gamma, const __half *w_q,
                                       const __half *w_k, const __half *w_v,
                                       float *q_out, float *k_out, float *v_out,
                                       uint32_t D, float eps);

/**
 * Fused FFN Front-end (GEMV-based)
 * Y = SiLU(x * W1) * (x * W3)
 *
 * @param stream HIP stream
 * @param x Input activations [D]
 * @param w1 Gate weights [H, D]
 * @param w3 Up weights [H, D]
 * @param y Output [H]
 * @param D Hidden dimension
 * @param H Intermediate dimension (e.g. 11008)
 */
void launch_fused_ffn_front_f16(hipStream_t stream, const float *x,
                                const __half *w1, const __half *w3, float *y,
                                uint32_t D, uint32_t H);

} // namespace gcore::rt::hip::kernels
