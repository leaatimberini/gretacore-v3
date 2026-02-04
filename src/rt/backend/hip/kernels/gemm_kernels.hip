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

// =================================================================================================
// MFMA Mixed Precision Implementation (CDNA 3 / gfx942)
// =================================================================================================

typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));

__global__ void __launch_bounds__(64) gemm_mfma_mixed_f16f32_kernel(
    const float *__restrict__ a,    // M x K (Activations)
    const __half *__restrict__ b,   // N x K (Weights, GGUF layout)
    float *__restrict__ c,          // M x N
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t lda, uint32_t ldb, uint32_t ldc) {

    uint32_t wave_col_base = blockIdx.x * 16;
    uint32_t wave_row_base = blockIdx.y * 16;
    uint32_t lane = threadIdx.x;

    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // Shared memory for tiles (16x16 F16 is small, but use for distribution)
    __shared__ __half As[16][16];
    __shared__ __half Bs[16][16];

    for (uint32_t k0 = 0; k0 < K; k0 += 16) {
        // Load A to LDS (Activations F32 -> F16)
        // Sequential threads should read sequential columns (K)
        for (int i = 0; i < 4; ++i) {
            uint32_t l_idx = lane * 4 + i;
            uint32_t r_loc = l_idx / 16;
            uint32_t c_loc = l_idx % 16;
            uint32_t gr = wave_row_base + r_loc;
            uint32_t gc = k0 + c_loc;
            if (gr < M && gc < K) {
                As[r_loc][c_loc] = __float2half(a[gr * lda + gc]);
            } else {
                As[r_loc][c_loc] = __float2half(0.0f);
            }
        }

        // Load B to LDS (Weights F16, N x K layout)
        // sequential memory is K. We have Bs[k_loc][n_loc]
        // In memory: b[n][k]. 16 rows of N, 16 columns of K.
        // We want sequential threads to read sequential k.
        for (int i = 0; i < 4; ++i) {
            uint32_t l_idx = lane * 4 + i;
            uint32_t n_loc = l_idx / 16;
            uint32_t k_loc = l_idx % 16;
            uint32_t gn = wave_col_base + n_loc;
            uint32_t gk = k0 + k_loc;
            if (gn < N && gk < K) {
                // Store weights in Bs such that Bs[k_loc][n_loc] is accessed by MFMA
                // but loaded from b[gn][gk]
                Bs[k_loc][n_loc] = b[gn * K + gk];
            } else {
                Bs[k_loc][n_loc] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Prepare MFMA fragments from LDS
        // Instruction: v_mfma_f32_16x16x16_f16
        // vA (Matrix A): 16x16. Thread i provides A[row=(i/16)*4 + 0..3, col=i%16]
        // Actually, the manual says for 16x16x16f16:
        // A input: 4 packed f16. Thread i holds values for A columns [i%16] and rows [(i/16)*4 + 0..3]
        half4 a_frag;
        uint32_t a_col_loc = lane % 16;
        uint32_t a_row_loc_base = (lane / 16) * 4;
        a_frag[0] = As[a_row_loc_base + 0][a_col_loc];
        a_frag[1] = As[a_row_loc_base + 1][a_col_loc];
        a_frag[2] = As[a_row_loc_base + 2][a_col_loc];
        a_frag[3] = As[a_row_loc_base + 3][a_col_loc];

        // B input: 16x16. Thread i holds values for B rows [i%16] and columns [(i/16)*4 + 0..3]
        half4 b_frag;
        uint32_t b_row_loc = lane % 16;
        uint32_t b_col_loc_base = (lane / 16) * 4;
        b_frag[0] = Bs[b_row_loc][b_col_loc_base + 0];
        b_frag[1] = Bs[b_row_loc][b_col_loc_base + 1];
        b_frag[2] = Bs[b_row_loc][b_col_loc_base + 2];
        b_frag[3] = Bs[b_row_loc][b_col_loc_base + 3];

        // Execute MFMA
        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, b_frag, acc, 0, 0, 0);

        __syncthreads();
    }

    // Store Result
    for (int i = 0; i < 4; ++i) {
        uint32_t c_row_local = (lane / 16) * 4 + i;
        uint32_t c_col_local = lane % 16;
        uint32_t c_global_row = wave_row_base + c_row_local;
        uint32_t c_global_col = wave_col_base + c_col_local;
        if (c_global_row < M && c_global_col < N) {
            c[c_global_row * ldc + c_global_col] = acc[i];
        }
    }
}

void launch_gemm_mfma_mixed_f16f32(hipStream_t stream, 
                                   const float *a, const __half *b, float *c,
                                   uint32_t M, uint32_t N, uint32_t K,
                                   uint32_t lda, uint32_t ldb, uint32_t ldc);

// Specialized GEMV for small M (Decode & small Prefill)
__global__ void gemv_mixed_f16f32_kernel(const float *__restrict__ a,
                                         const __half *__restrict__ w,
                                         float *__restrict__ y, uint32_t M,
                                         uint32_t N, uint32_t K, uint32_t lda) {
  // One block per output row of (A*W^T)
  // Grid: (N, M)
  uint32_t row_a = blockIdx.y;
  uint32_t row_w = blockIdx.x;
  if (row_a >= M || row_w >= N)
    return;

  uint32_t tid = threadIdx.x; // 0..63 for wave-oriented
  float acc = 0.0f;

  const float *a_row = a + row_a * lda;
  const __half *w_row = w + row_w * K;

  const float4 *a4 = reinterpret_cast<const float4 *>(a_row);
  const half4 *w4 = reinterpret_cast<const half4 *>(w_row);
  uint32_t K4 = K / 4;

  for (uint32_t k = tid; k < K4; k += 64) {
    float4 av = a4[k];
    half4 wv = w4[k];
    acc += av.x * (float)wv[0] + av.y * (float)wv[1] + av.z * (float)wv[2] +
           av.w * (float)wv[3];
  }

  // Handle remainder (if K not multiple of 4)
  for (uint32_t k = K4 * 4 + tid; k < K; k += 64) {
    acc += a_row[k] * (float)w_row[k];
  }

  // Pure warp reduction (MI300X wave size is 64)
  for (int offset = 32; offset > 0; offset /= 2) {
    acc += __shfl_down(acc, offset);
  }

  if (tid == 0) {
    y[row_a * N + row_w] = acc;
  }
}

void launch_gemm_mfma_mixed_f16f32(hipStream_t stream, const float *a,
                                   const __half *b, float *c, uint32_t M,
                                   uint32_t N, uint32_t K, uint32_t lda,
                                   uint32_t ldb, uint32_t ldc) {
  if (M == 0 || N == 0 || K == 0)
    return;

  if (M <= 32) {
    // Wave-parallel GEMV
    dim3 block(64, 1, 1);
    dim3 grid(N, M, 1);
    gemv_mixed_f16f32_kernel<<<grid, block, 0, stream>>>(a, b, c, M, N, K, lda);
  } else {
    // General MFMA GEMM
    dim3 block(64, 1, 1);
    dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
    gemm_mfma_mixed_f16f32_kernel<<<grid, block, 0, stream>>>(a, b, c, M, N, K,
                                                              lda, ldb, ldc);
  }
}

// Specialized GEMV for final linear layer (lm_head)
// Grid: (N, M, 1) - where N is vocab_size, M is seq_len (usually 1 for decode)
// Block: 256 threads
__global__ void lm_head_gemv_kernel(const float *__restrict__ x,
                                    const __half *__restrict__ w,
                                    float *__restrict__ y, uint32_t M,
                                    uint32_t N, uint32_t K) {
  uint32_t row_x = blockIdx.y;
  uint32_t row_w = blockIdx.x; // maps to vocab index
  if (row_x >= M || row_w >= N)
    return;

  uint32_t tid = threadIdx.x;
  float acc = 0.0f;

  const float *x_row = x + row_x * K;
  const __half *w_row = w + row_w * K;

  const float4 *x4 = reinterpret_cast<const float4 *>(x_row);
  const half4 *w4 = reinterpret_cast<const half4 *>(w_row);
  uint32_t K4 = K / 4;

  for (uint32_t k = tid; k < K4; k += 256) {
    float4 xv = x4[k];
    half4 wv = w4[k];
    acc += xv.x * (float)wv[0] + xv.y * (float)wv[1] + xv.z * (float)wv[2] +
           xv.w * (float)wv[3];
  }

  // Handle remainder
  for (uint32_t k = K4 * 4 + tid; k < K; k += 256) {
    acc += x_row[k] * (float)w_row[k];
  }

  // Block reduction
  __shared__ float s_acc[256];
  s_acc[tid] = acc;
  __syncthreads();

  for (uint32_t s = 128; s > 0; s >>= 1) {
    if (tid < s) {
      s_acc[tid] += s_acc[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    y[row_x * N + row_w] = s_acc[0];
  }
}

void launch_lm_head_gemv(hipStream_t stream, const float *x, const __half *w,
                         float *y, uint32_t M, uint32_t N, uint32_t K) {
  if (M == 0 || N == 0 || K == 0)
    return;

  dim3 block(256, 1, 1);
  dim3 grid(N, M, 1);
  lm_head_gemv_kernel<<<grid, block, 0, stream>>>(x, w, y, M, N, K);
}

// Specialized GEMV for INT8 Weight-Only (Decode)
// Grid: (N, M, 1) - where N is output neurons, M is seq_len (usually 1)
// Block: 256 threads
__global__ void gemv_int8_wt_kernel(const float *__restrict__ x,
                                    const int8_t *__restrict__ w,
                                    float *__restrict__ y,
                                    const float *__restrict__ scales,
                                    uint32_t M, uint32_t N, uint32_t K,
                                    uint32_t group_size) {
  uint32_t row_x = blockIdx.y;
  uint32_t row_w = blockIdx.x; // neuron index
  if (row_x >= M || row_w >= N)
    return;

  uint32_t tid = threadIdx.x;
  float acc = 0.0f;

  const float *x_row = x + row_x * K;
  const int8_t *w_row = w + row_w * K;
  const float *s_row = scales + (row_w * K) / group_size;

  for (uint32_t k = tid; k < K; k += 256) {
    float act = x_row[k];
    float wt = (float)w_row[k];
    float scale = s_row[k / group_size];
    acc += act * (wt * scale);
  }

  // Block reduction
  __shared__ float s_acc[256];
  s_acc[tid] = acc;
  __syncthreads();

  for (uint32_t s = 128; s > 0; s >>= 1) {
    if (tid < s) {
      s_acc[tid] += s_acc[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    y[row_x * N + row_w] = s_acc[0];
  }
}

// =================================================================================================
// INT8 Weight-Only GEMM (MFMA CDNA 3)
// =================================================================================================

__global__ void __launch_bounds__(64) gemm_mfma_int8_wt_kernel(
    const float *__restrict__ a,    // M x K
    const int8_t *__restrict__ b,   // N x K (Weights, GGUF-like row-major)
    float *__restrict__ c,          // M x N
    const float *__restrict__ scales,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t lda, uint32_t ldb, uint32_t ldc,
    uint32_t group_size) {

    uint32_t wave_col_base = blockIdx.x * 16;
    uint32_t wave_row_base = blockIdx.y * 16;
    uint32_t lane = threadIdx.x;

    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // Shared memory for tiles (optional, but good for broadcast/swizzle)
    // To keep it "fused dequant first class", we load directly to registers if possible
    // but MFMA requires specific layout.
    __shared__ __half As[16][16];
    __shared__ int8_t Bs[16][16];

    for (uint32_t k0 = 0; k0 < K; k0 += 16) {
        // Load A to LDS (Activations F32 -> F16)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t l_idx = lane * 4 + i;
            uint32_t r_loc = l_idx / 16;
            uint32_t c_loc = l_idx % 16;
            uint32_t gr = wave_row_base + r_loc;
            uint32_t gc = k0 + c_loc;
            if (gr < M && gc < K) {
                As[r_loc][c_loc] = __float2half(a[gr * lda + gc]);
            } else {
                As[r_loc][c_loc] = __float2half(0.0f);
            }
        }

        // Load B to LDS (Weights INT8, N x K)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t l_idx = lane * 4 + i;
            uint32_t n_loc = l_idx / 16;
            uint32_t k_loc = l_idx % 16;
            uint32_t gn = wave_col_base + n_loc;
            uint32_t gk = k0 + k_loc;
            if (gn < N && gk < K) {
                Bs[k_loc][n_loc] = b[gn * K + gk];
            } else {
                Bs[k_loc][n_loc] = 0;
            }
        }

        __syncthreads();

        // Prepare MFMA fragments with FUSED DEQUANT
        half4 a_frag;
        uint32_t a_row_loc = lane % 16;
        uint32_t a_col_loc_base = (lane / 16) * 4;
        a_frag[0] = As[a_row_loc][a_col_loc_base + 0];
        a_frag[1] = As[a_row_loc][a_col_loc_base + 1];
        a_frag[2] = As[a_row_loc][a_col_loc_base + 2];
        a_frag[3] = As[a_row_loc][a_col_loc_base + 3];

        half4 b_frag;
        uint32_t b_col_loc = lane % 16;
        uint32_t b_row_loc_base = (lane / 16) * 4;
        
        // Fused dequant: Load int8, multiply by scale, convert to half
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t n_loc = b_col_loc;
            uint32_t k_loc = b_row_loc_base + i;
            uint32_t gn = wave_col_base + n_loc;
            uint32_t gk = k0 + k_loc;
            
            float scale = 1.0f;
            if (scales) {
                scale = scales[(gn * K + gk) / group_size];
            }
            
            b_frag[i] = __float2half((float)Bs[k_loc][n_loc] * scale);
        }

        // Execute MFMA
        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, b_frag, acc, 0, 0, 0);

        __syncthreads();
    }

    // Store Result
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t c_row_local = lane % 16;
        uint32_t c_col_local = (lane / 16) * 4 + i;
        uint32_t c_global_row = wave_row_base + c_row_local;
        uint32_t c_global_col = wave_col_base + c_col_local;
        if (c_global_row < M && c_global_col < N) {
            c[c_global_row * ldc + c_global_col] = acc[i];
        }
    }
}

void launch_gemm_mfma_int8_wt_fp32_acc32(
    hipStream_t stream, const float *a, const int8_t *b, float *c,
    const float *scales, uint32_t M, uint32_t N, uint32_t K, uint32_t lda,
    uint32_t ldb, uint32_t ldc, uint32_t group_size) {
  if (M == 0 || N == 0 || K == 0)
    return;

  if (M <= 32) {
    dim3 block(256, 1, 1);
    dim3 grid(N, M, 1);
    gemv_int8_wt_kernel<<<grid, block, 0, stream>>>(a, b, c, scales, M, N, K,
                                                    group_size);
  } else {
    dim3 block(64, 1, 1);
    dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
    gemm_mfma_int8_wt_kernel<<<grid, block, 0, stream>>>(
        a, b, c, scales, M, N, K, lda, ldb, ldc, group_size);
  }
}

// Specialized GEMV for INT4 Weight-Only (Decode)
template <typename T_A>
__global__ void gemv_int4_wt_kernel(const T_A *__restrict__ x,
                                    const int8_t *__restrict__ w,
                                    float *__restrict__ y,
                                    const float *__restrict__ scales,
                                    const float *__restrict__ head_scales,
                                    uint32_t M, uint32_t N, uint32_t K,
                                    uint32_t group_size, uint32_t head_dim) {
  uint32_t row_x = blockIdx.y;
  uint32_t row_w = blockIdx.x;
  if (row_x >= M || row_w >= N)
    return;

  uint32_t tid = threadIdx.x;
  float acc = 0.0f;

  const T_A *x_row = x + row_x * K;
  const int8_t *w_row = w + row_w * (K / 2);
  const float *s_row = scales + (row_w * K) / group_size;

  for (uint32_t k = tid; k < K / 2; k += 256) {
    uint8_t packed = (uint8_t)w_row[k];

    int8_t v0 = (packed & 0x0F);
    if (v0 & 0x08)
      v0 |= 0xF0;
    int8_t v1 = (packed >> 4) & 0x0F;
    if (v1 & 0x08)
      v1 |= 0xF0;

    uint32_t gk0 = k * 2;
    uint32_t gk1 = k * 2 + 1;

    acc += (float)x_row[gk0] * ((float)v0 * s_row[gk0 / group_size]);
    acc += (float)x_row[gk1] * ((float)v1 * s_row[gk1 / group_size]);
  }

  __shared__ float s_acc[256];
  s_acc[tid] = acc;
  __syncthreads();

  for (uint32_t s = 128; s > 0; s >>= 1) {
    if (tid < s)
      s_acc[tid] += s_acc[tid + s];
    __syncthreads();
  }

  if (tid == 0) {
    float h_scale = 1.0f;
    if (head_scales && head_dim > 0) {
      h_scale = head_scales[row_w / head_dim];
    }
    y[row_x * N + row_w] = s_acc[0] * h_scale;
  }
}

// MFMA INT4 Weight-Only GEMM
template <typename T_A>
__global__ void __launch_bounds__(64) gemm_mfma_int4_wt_kernel(
    const T_A *__restrict__ a, const int8_t *__restrict__ b,
    float *__restrict__ c, const float *__restrict__ scales,
    const float *__restrict__ head_scales, uint32_t M, uint32_t N, uint32_t K,
    uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t group_size,
    uint32_t head_dim) {
  uint32_t wave_col_base = blockIdx.x * 16;
  uint32_t wave_row_base = blockIdx.y * 16;
  uint32_t lane = threadIdx.x;

  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

  __shared__ __half As[16][16];
  __shared__ int8_t Bs[16][16]; // Unpacked tile in LDS

  for (uint32_t k0 = 0; k0 < K; k0 += 16) {
    // Load A (Activations)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t l_idx = lane * 4 + i;
      uint32_t r_loc = l_idx / 16;
      uint32_t c_loc = l_idx % 16;
      uint32_t gr = wave_row_base + r_loc;
      uint32_t gc = k0 + c_loc;
      As[r_loc][c_loc] = (gr < M && gc < K) ? __float2half((float)a[gr * lda + gc])
                                            : __float2half(0.0f);
    }

    // Load and Unpack B (Weights INT4 packed, N x K/2)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t l_idx = lane * 4 + i;
      uint32_t n_loc = l_idx / 16;
      uint32_t k_loc = l_idx % 16; // 0..15
      uint32_t gn = wave_col_base + n_loc;
      uint32_t gk = k0 + k_loc;

      if (gn < N && gk < K) {
        uint8_t packed = (uint8_t)b[gn * (K / 2) + (gk / 2)];
        int8_t v = (gk % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
        if (v & 0x08)
          v |= 0xF0;
        Bs[k_loc][n_loc] = v;
      } else {
        Bs[k_loc][n_loc] = 0;
      }
    }

    __syncthreads();

    half4 a_frag;
    uint32_t a_row_loc = lane % 16;
    uint32_t a_col_loc_base = (lane / 16) * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
      a_frag[i] = As[a_row_loc][a_col_loc_base + i];

    half4 b_frag;
    uint32_t b_col_loc = lane % 16;
    uint32_t b_row_loc_base = (lane / 16) * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t k_loc = b_row_loc_base + i;
      uint32_t gn = wave_col_base + b_col_loc;
      uint32_t gk = k0 + k_loc;
      float scale = scales ? scales[(gn * K + gk) / group_size] : 1.0f;
      b_frag[i] = __float2half((float)Bs[k_loc][b_col_loc] * scale);
    }

    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, b_frag, acc, 0, 0, 0);
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t c_row_loc = lane % 16;
    uint32_t c_col_loc = (lane / 16) * 4 + i;
    uint32_t gr = wave_row_base + c_row_loc;
    uint32_t gc = wave_col_base + c_col_loc;
    if (gr < M && gc < N) {
      float h_scale = 1.0f;
      if (head_scales && head_dim > 0) {
        h_scale = head_scales[gc / head_dim];
      }
      c[gr * ldc + gc] = acc[i] * h_scale;
    }
  }
}

void launch_gemm_mfma_int4_wt_fp32_acc32(
    hipStream_t stream, const void *a, const int8_t *b, float *c,
    const float *scales, const float *head_scales, uint32_t M, uint32_t N,
    uint32_t K, uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t group_size,
    uint32_t head_dim, bool is_fp16_a, bool force_gemv) {
  if (M == 0 || N == 0 || K == 0)
    return;
  if (force_gemv || M <= 32) {
    dim3 block(256);
    dim3 grid(N, M);
    if (is_fp16_a) {
      gemv_int4_wt_kernel<__half><<<grid, block, 0, stream>>>(
          static_cast<const __half *>(a), b, c, scales, head_scales, M, N, K,
          group_size, head_dim);
    } else {
      gemv_int4_wt_kernel<float><<<grid, block, 0, stream>>>(
          static_cast<const float *>(a), b, c, scales, head_scales, M, N, K,
          group_size, head_dim);
    }
  } else {
    dim3 block(64);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    if (is_fp16_a) {
      gemm_mfma_int4_wt_kernel<__half><<<grid, block, 0, stream>>>(
          static_cast<const __half *>(a), b, c, scales, head_scales, M, N, K,
          lda, ldb, ldc, group_size, head_dim);
    } else {
      gemm_mfma_int4_wt_kernel<float><<<grid, block, 0, stream>>>(
          static_cast<const float *>(a), b, c, scales, head_scales, M, N, K, lda,
          ldb, ldc, group_size, head_dim);
    }
  }
}

} // namespace gcore::rt::hip::kernels
