#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

namespace gcore::rt::hip::kernels {

// Standard Tiled SGEMM (C = A * B)
// A is M x K, B is K x N, C is M x N
// lda, ldb, ldc are strides
void launch_gemm_tiled_f32(hipStream_t stream, const float *a, const float *b,
                           float *c, uint32_t M, uint32_t N, uint32_t K,
                           uint32_t lda, uint32_t ldb, uint32_t ldc);

// Matrix Core GEMM (MFMA)
// Uses __builtin_amdgcn_mfma_f32_16x16x4f32
void launch_gemm_mfma_f32(hipStream_t stream, const float *a, const float *b,
                          float *c, uint32_t M, uint32_t N, uint32_t K,
                          uint32_t lda, uint32_t ldb, uint32_t ldc);

} // namespace gcore::rt::hip::kernels
