#pragma once

#include <cstdint>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace gcore::rt::hip::kernels {

// Standard Tiled SGEMM (C = A * B) - FP32
// A is M x K, B is K x N, C is M x N
// lda, ldb, ldc are strides
void launch_gemm_tiled_f32(hipStream_t stream, const float *a, const float *b,
                           float *c, uint32_t M, uint32_t N, uint32_t K,
                           uint32_t lda, uint32_t ldb, uint32_t ldc);

// Matrix Core GEMM (MFMA) - FP32
// Uses __builtin_amdgcn_mfma_f32_16x16x4f32
void launch_gemm_mfma_f32(hipStream_t stream, const float *a, const float *b,
                          float *c, uint32_t M, uint32_t N, uint32_t K,
                          uint32_t lda, uint32_t ldb, uint32_t ldc);

// Tiled HGEMM (C = A * B) - FP16
// All inputs/outputs in half precision, accumulator in FP32
void launch_gemm_tiled_f16(hipStream_t stream, const __half *a, const __half *b,
                           __half *c, uint32_t M, uint32_t N, uint32_t K,
                           uint32_t lda, uint32_t ldb, uint32_t ldc);

// Mixed Precision GEMM - FP32 activations, FP16 weights, FP32 output
// Allows 2x memory savings on weights while keeping activation precision
void launch_gemm_mixed_f16f32(hipStream_t stream, const float *a,
                              const __half *b, float *c, uint32_t M, uint32_t N,
                              uint32_t K, uint32_t lda, uint32_t ldb,
                              uint32_t ldc);

void launch_lm_head_gemv(hipStream_t stream, const float *x, const __half *w,
                         float *y, uint32_t M, uint32_t N, uint32_t K);

// Matrix Core Mixed Precision GEMM (MFMA)
// Uses __builtin_amdgcn_mfma_f32_16x16x16f16
void launch_gemm_mfma_mixed_f16f32(hipStream_t stream, const float *a,
                                   const __half *b, float *c, uint32_t M,
                                   uint32_t N, uint32_t K, uint32_t lda,
                                   uint32_t ldb, uint32_t ldc);

// INT8 Weight-Only GEMM (Mixed Precision: FP32 acts, INT8 weights, FP32 accum)
void launch_gemm_mfma_int8_wt_fp32_acc32(hipStream_t stream, const float *a,
                                         const int8_t *b, float *c,
                                         const float *scales, uint32_t M,
                                         uint32_t N, uint32_t K, uint32_t lda,
                                         uint32_t ldb, uint32_t ldc,
                                         uint32_t group_size);

// INT4 Weight-Only GEMM
void launch_gemm_mfma_int4_wt_fp32_acc32(
    hipStream_t stream, const void *a, const int8_t *b, float *c,
    const float *scales, const float *head_scales, uint32_t M, uint32_t N,
    uint32_t K, uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t group_size,
    uint32_t head_dim, bool is_fp16_a, bool force_gemv);

} // namespace gcore::rt::hip::kernels
