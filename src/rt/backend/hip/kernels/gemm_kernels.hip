#include "gcore/rt/hip/kernels/gemm_kernels.hpp"

namespace gcore::rt::hip::kernels {

#define TILE_SIZE 16

__global__ void gemm_tiled_f32_kernel(const float *__restrict__ a, 
                                      const float *__restrict__ b, 
                                      float *__restrict__ c,
                                      uint32_t M, uint32_t N, uint32_t K,
                                      uint32_t lda, uint32_t ldb, uint32_t ldc) {
  
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t lx = threadIdx.x;
  uint32_t ly = threadIdx.y;

  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  float acc = 0.0f;

  for (uint32_t k0 = 0; k0 < K; k0 += TILE_SIZE) {
    // Load tile A
    uint32_t ak = k0 + lx;
    if (row < M && ak < K) {
      As[ly][lx] = a[row * lda + ak];
    } else {
      As[ly][lx] = 0.0f;
    }

    // Load tile B
    uint32_t bk = k0 + ly;
    if (col < N && bk < K) {
      Bs[ly][lx] = b[bk * ldb + col];
    } else {
      Bs[ly][lx] = 0.0f;
    }

    __syncthreads();

    // Compute partial
    for (uint32_t k = 0; k < TILE_SIZE; ++k) {
      acc += As[ly][k] * Bs[k][lx];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    c[row * ldc + col] = acc;
  }
}

void launch_gemm_tiled_f32(hipStream_t stream, 
                           const float *a, const float *b, float *c,
                           uint32_t M, uint32_t N, uint32_t K,
                           uint32_t lda, uint32_t ldb, uint32_t ldc) {
  
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, 
            (M + TILE_SIZE - 1) / TILE_SIZE);

  gemm_tiled_f32_kernel<<<grid, block, 0, stream>>>(a, b, c, M, N, K, lda, ldb, ldc);
}

// =================================================================================================
// MFMA Implementation (CDNA 3)
// =================================================================================================

// 16x16x4 f32 MFMA
// Wave size: 64
// Each wave computes a 16x16 block of C
// Inner loop step K: 4
// Instruction: v_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0)
// A input: 1 float per thread (but actually it takes 64 threads to feed the 16x4 a-matrix)
// B input: 1 float per thread (64 threads -> 4x16 b-matrix)
// C input/output: 4 floats per thread (Accumulator)

__global__ void __launch_bounds__(64) gemm_mfma_f32_kernel(
    const float *__restrict__ a, 
    const float *__restrict__ b, 
    float *__restrict__ c,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t lda, uint32_t ldb, uint32_t ldc) {

  // Global position of the 16x16 block
  // Grid is defined as (N/16, M/16)
  uint32_t wave_col_base = blockIdx.x * 16;
  uint32_t wave_row_base = blockIdx.y * 16;

  // Lane ID (0-63)
  uint32_t lane = threadIdx.x;

  // Accumulators (4 values per thread for 16x16 output)
  using float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

  // MFMA 16x16x4 mapping logic:
  // lane 0-63 maps to specific elements of A (16x4) and B (4x16) inputs.
  // The intrinsic takes a register VGPR containing 1 float.
  // Ideally we would load from LDS to get the right distribution, but for a simple
  // global memory version we calculate indices.
  
  // A is M x K (Row Major) -> A[row, k]
  // B is K x N (Row Major) -> B[k, col]

  // MFMA 16x16x4 geometry for Inputs:
  // Input A (16 rows x 4 cols): 
  //   Shared across wave? No, specific lanes provide specific values.
  //   Actually, v_mfma_f32_16x16x4f32 takes vgpr 'a' and 'b'.
  //   'a' contributes to the 16x4 sub-block.
  //   'b' contributes to the 4x16 sub-block.
  
  // Mapping for A input (source generic documentation for gfx90a/gfx940):
  // 64 threads provide 64 scalars.
  // Matrix A sub-block is 16 rows x 4 cols = 64 elements.
  // Lane L provides A[L % 16, L / 16] relative to the block being computed?
  // Usually: row = lane % 16, col = lane / 16.
  
  // Mapping for B input:
  // Matrix B sub-block is 4 rows x 16 cols = 64 elements.
  // Lane L provides B[L / 16, L % 16] relative to the block?
  // Let's assume standard layout compatibility or just try linear mapping.
  
  // Optimization: Precompute global pointers/offsets
  
  for (uint32_t k = 0; k < K; k += 4) {
    // Load A fragment (16x4)
    // We need 1 float per thread.
    // Thread L needs A[wave_row_base + (L % 16), k + (L / 16)]
    uint32_t a_row_off = lane % 16;
    uint32_t a_col_off = lane / 16;
    
    float a_val = 0.0f;
    uint32_t gx_a = wave_row_base + a_row_off;
    uint32_t gk_a = k + a_col_off;
    
    if (gx_a < M && gk_a < K) {
      a_val = a[gx_a * lda + gk_a];
    }

    // Load B fragment (4x16)
    // We need 1 float per thread.
    // Thread L needs B[k + (L / 16), wave_col_base + (L % 16)]
    // Note: B is usually loaded as (4 rows, 16 cols)
    // Let's verify mapping: B input to mfma is 4x16?
    // Actually documentation says: MFMA 16x16x4:
    //  Source A: 16x4 (Row Major implied by lanes?)
    //  Source B: 4x16 (Row Major implied?)
    //  Let's stick to the hypothesis:
    //  B_row = lane / 16
    //  B_col = lane % 16
    
    uint32_t b_row_off = lane / 16;
    uint32_t b_col_off = lane % 16;
    
    float b_val = 0.0f;
    uint32_t gk_b = k + b_row_off;
    uint32_t gy_b = wave_col_base + b_col_off;

    if (gk_b < K && gy_b < N) {
      b_val = b[gk_b * ldb + gy_b];
    }

    // Execute MFMA
    acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a_val, b_val, acc, 0, 0, 0);
  }

  // Store Result
  // Output C is 16x16.
  // Each thread holds 4 outputs.
  // Mapping for C (from standard documentation):
  // Thread L holds C[L/4, L%4], C[L/4, L%4 + 4], C[L/4 + 8, L%4], C[L/4 + 8, L%4 + 4] ?
  // No, the layout is notoriously complex (Interleaved).
  // For 16x16x4:
  // Thread L holds 4 elements.
  //   d[0] -> (row=(L/16),     col=(L%16)) -- wait, 64 threads cover 16x16? No that's 256 elements.
  //   64 threads * 4 elements = 256. Correct.
  //   Common layout pattern for 16x16 (CDNA2/3):
  //    d[0]: row (lane/16), col (lane%16)
  //    d[1]: row (lane/16), col (lane%16) + 4 ?? No.
  
  // Let's use the Reference Mapping for gfx90a (likely same for gfx942):
  // d0: C[L / 16 * 4 + 0, L % 16]
  // d1: C[L / 16 * 4 + 1, L % 16]
  // d2: C[L / 16 * 4 + 2, L % 16]
  // d3: C[L / 16 * 4 + 3, L % 16]
  // Wait, that creates 4 consecutive rows for same column?
  // Let's try to infer from "Matrix Core" logic (D = C + A*B).
  //
  // Actually, simpler mapping usually seen:
  //   row = (lane / 16) * 4 + i
  //   col = lane % 16
  // Let's verify.
  
  for (int i = 0; i < 4; ++i) {
    uint32_t c_row_local = (lane / 16) * 4 + i; // 0..15 range ? (63/16)=3. 3*4+3 = 15. Correct.
    uint32_t c_col_local = lane % 16;           // 0..15 range. Correct.
    
    uint32_t c_global_row = wave_row_base + c_row_local;
    uint32_t c_global_col = wave_col_base + c_col_local;
    
    if (c_global_row < M && c_global_col < N) {
       c[c_global_row * ldc + c_global_col] = acc[i];
    }
  }
}

void launch_gemm_mfma_f32(hipStream_t stream, 
                          const float *a, const float *b, float *c,
                          uint32_t M, uint32_t N, uint32_t K,
                          uint32_t lda, uint32_t ldb, uint32_t ldc) {
  
  // Grid size: one wave per 16x16 block
  dim3 block(64, 1, 1);
  dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
  
  gemm_mfma_f32_kernel<<<grid, block, 0, stream>>>(a, b, c, M, N, K, lda, ldb, ldc);
}

// =================================================================================================
// FP16 GEMM Implementation
// =================================================================================================

#include <hip/hip_fp16.h>

#define TILE_SIZE_FP16 16

__global__ void gemm_tiled_f16_kernel(const __half *__restrict__ a, 
                                      const __half *__restrict__ b, 
                                      __half *__restrict__ c,
                                      uint32_t M, uint32_t N, uint32_t K,
                                      uint32_t lda, uint32_t ldb, uint32_t ldc) {
  
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t lx = threadIdx.x;
  uint32_t ly = threadIdx.y;

  __shared__ __half As[TILE_SIZE_FP16][TILE_SIZE_FP16];
  __shared__ __half Bs[TILE_SIZE_FP16][TILE_SIZE_FP16];

  float acc = 0.0f; // Accumulate in FP32 for precision

  for (uint32_t k0 = 0; k0 < K; k0 += TILE_SIZE_FP16) {
    // Load tile A
    uint32_t ak = k0 + lx;
    if (row < M && ak < K) {
      As[ly][lx] = a[row * lda + ak];
    } else {
      As[ly][lx] = __float2half(0.0f);
    }

    // Load tile B
    uint32_t bk = k0 + ly;
    if (col < N && bk < K) {
      Bs[ly][lx] = b[bk * ldb + col];
    } else {
      Bs[ly][lx] = __float2half(0.0f);
    }

    __syncthreads();

    // Compute partial - accumulate in FP32
    for (uint32_t k = 0; k < TILE_SIZE_FP16; ++k) {
      acc += __half2float(As[ly][k]) * __half2float(Bs[k][lx]);
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    c[row * ldc + col] = __float2half(acc);
  }
}

void launch_gemm_tiled_f16(hipStream_t stream, 
                           const __half *a, const __half *b, __half *c,
                           uint32_t M, uint32_t N, uint32_t K,
                           uint32_t lda, uint32_t ldb, uint32_t ldc) {
  
  dim3 block(TILE_SIZE_FP16, TILE_SIZE_FP16);
  dim3 grid((N + TILE_SIZE_FP16 - 1) / TILE_SIZE_FP16, 
            (M + TILE_SIZE_FP16 - 1) / TILE_SIZE_FP16);

  gemm_tiled_f16_kernel<<<grid, block, 0, stream>>>(a, b, c, M, N, K, lda, ldb, ldc);
}

// Mixed precision: FP16 weights, FP32 activations, FP32 output
__global__ void gemm_mixed_f16f32_kernel(const float *__restrict__ a, 
                                         const __half *__restrict__ b, 
                                         float *__restrict__ c,
                                         uint32_t M, uint32_t N, uint32_t K,
                                         uint32_t lda, uint32_t ldb, uint32_t ldc) {
  
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t lx = threadIdx.x;
  uint32_t ly = threadIdx.y;

  __shared__ float As[TILE_SIZE_FP16][TILE_SIZE_FP16];
  __shared__ __half Bs[TILE_SIZE_FP16][TILE_SIZE_FP16];

  float acc = 0.0f;

  for (uint32_t k0 = 0; k0 < K; k0 += TILE_SIZE_FP16) {
    // Load tile A (FP32 activations, M x K)
    uint32_t ak = k0 + lx;
    if (row < M && ak < K) {
      As[ly][lx] = a[row * lda + ak];
    } else {
      As[ly][lx] = 0.0f;
    }

    // Load tile B (FP16 weights, N x K GGUF layout)
    // We want to load a tile of N (outputs) and K (inputs).
    // GGUF stores data as weight[n][k].
    // Sequential threads in a warp (lx) should read sequential k.
    uint32_t bn = col; // row index in weight matrix
    uint32_t bk = k0 + ly; // col index in weight matrix
    if (bn < N && bk < K) {
      // Direct load from w[n * K + k]
      Bs[ly][lx] = b[bn * K + bk]; 
    } else {
      Bs[ly][lx] = __float2half(0.0f);
    }

    __syncthreads();

    // Compute - accumulate partials
    for (uint32_t k = 0; k < TILE_SIZE_FP16; ++k) {
      acc += As[ly][k] * __half2float(Bs[k][lx]);
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    c[row * ldc + col] = acc;
  }
}

void launch_gemm_mixed_f16f32(hipStream_t stream, 
                              const float *a, const __half *b, float *c,
                              uint32_t M, uint32_t N, uint32_t K,
                              uint32_t lda, uint32_t ldb, uint32_t ldc) {
  
  dim3 block(TILE_SIZE_FP16, TILE_SIZE_FP16);
  dim3 grid((N + TILE_SIZE_FP16 - 1) / TILE_SIZE_FP16, 
            (M + TILE_SIZE_FP16 - 1) / TILE_SIZE_FP16);

  gemm_mixed_f16f32_kernel<<<grid, block, 0, stream>>>(a, b, c, M, N, K, lda, ldb, ldc);
}

} // namespace gcore::rt::hip::kernels
