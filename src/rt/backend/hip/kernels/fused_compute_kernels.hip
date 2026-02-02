#include "gcore/rt/hip/kernels/fused_compute_kernels.hpp"
#include <hip/hip_runtime.h>

namespace gcore::rt::hip::kernels {

/**
 * Fused RMSNorm + GEMV (FP32 Activations, FP16 Weights)
 * 
 * Algorithm:
 * 1. Compute RMS of X (Hidden Dim D)
 * 2. Multiply by (gamma / RMS) and dot product with Weights W
 * 3. Store result to Y [N]
 * 
 * Optimization for Llama-2-7B (D=4096):
 * - Block Size: 256
 * - Each block handles one or more output neurons i in N.
 * - RMS is shared across the entire block.
 */

#include <hip/hip_fp16.h>

__global__ void __launch_bounds__(256)
fused_rmsnorm_qkv_gemv_kernel_f16(const float* __restrict__ x,
                                  const float* __restrict__ gamma,
                                  const __half* __restrict__ w_q,
                                  const __half* __restrict__ w_k,
                                  const __half* __restrict__ w_v,
                                  float* __restrict__ q_out,
                                  float* __restrict__ k_out,
                                  float* __restrict__ v_out,
                                  uint32_t D,
                                  float eps) {
    
    const int WAVE_SIZE = 64;
    const int NUM_WAVES = 256 / WAVE_SIZE;
    
    // Total D = 4096. float[4096] = 16KB. Fits in 64KB LDS.
    extern __shared__ float s_norm_x[]; // Size: float[D]
    __shared__ float s_rms_val;
    __shared__ float s_wave_results[NUM_WAVES];

    // 1. Calculate RMS of input vector x (Block-parallel)
    float ss = 0.0f;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    uint32_t D4 = D / 4;

    for (uint32_t i = threadIdx.x; i < D4; i += blockDim.x) {
        float4 val = x4[i];
        ss += val.x * val.x;
        ss += val.y * val.y;
        ss += val.z * val.z;
        ss += val.w * val.w;
    }

    // Warp reduction
    for (int offset = 32; offset > 0; offset /= 2) {
        ss += __shfl_down(ss, offset);
    }
    
    int wave_id = threadIdx.x / WAVE_SIZE;
    int lane_id = threadIdx.x % WAVE_SIZE;

    if (lane_id == 0) s_wave_results[wave_id] = ss;
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_ss = 0;
        for (int i = 0; i < NUM_WAVES; ++i) block_ss += s_wave_results[i];
        s_rms_val = rsqrtf((block_ss / D) + eps);
    }
    __syncthreads();

    float inv_rms = s_rms_val;

    // 2. Normalize and Cache in LDS
    // This eliminates the math from the heavy GEMV loop
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    float4* sn4 = reinterpret_cast<float4*>(s_norm_x);
    for (uint32_t i = threadIdx.x; i < D4; i += blockDim.x) {
        float4 xv = x4[i];
        float4 gv = g4[i];
        float4 res;
        res.x = xv.x * inv_rms * gv.x;
        res.y = xv.y * inv_rms * gv.y;
        res.z = xv.z * inv_rms * gv.z;
        res.w = xv.w * inv_rms * gv.w;
        sn4[i] = res;
    }
    __syncthreads();

    // 3. QKV Projections
    uint32_t global_wave_id = (blockIdx.x * NUM_WAVES) + wave_id;
    if (global_wave_id >= 3 * D) return;

    uint32_t matrix_id = global_wave_id / D;
    uint32_t row_id = global_wave_id % D;
    
    const __half* weights = (matrix_id == 0) ? w_q : (matrix_id == 1 ? w_k : w_v);
    float* output = (matrix_id == 0) ? q_out : (matrix_id == 1 ? k_out : v_out);
    
    const float4* w4 = reinterpret_cast<const float4*>(weights + row_id * D);
    const float4* sn4_in = reinterpret_cast<const float4*>(s_norm_x);
    
    uint32_t D8 = D / 8;
    float wave_acc = 0.0f;
    
    for (uint32_t j = lane_id; j < D8; j += WAVE_SIZE) {
        float4 w_vec = w4[j];
        float4 nx0 = sn4_in[j * 2];
        float4 nx1 = sn4_in[j * 2 + 1];

        const __half* wh = reinterpret_cast<const __half*>(&w_vec);
        wave_acc += nx0.x * __half2float(wh[0]);
        wave_acc += nx0.y * __half2float(wh[1]);
        wave_acc += nx0.z * __half2float(wh[2]);
        wave_acc += nx0.w * __half2float(wh[3]);

        wave_acc += nx1.x * __half2float(wh[4]);
        wave_acc += nx1.y * __half2float(wh[5]);
        wave_acc += nx1.z * __half2float(wh[6]);
        wave_acc += nx1.w * __half2float(wh[7]);
    }

    // Wave reduction
    for (int offset = 32; offset > 0; offset /= 2) {
        wave_acc += __shfl_down(wave_acc, offset);
    }

    if (lane_id == 0) {
        output[row_id] = wave_acc;
    }
}

void launch_fused_rmsnorm_qkv_gemv_f16(hipStream_t stream, 
                                       const float* x, 
                                       const float* gamma, 
                                       const __half* w_q, 
                                       const __half* w_k, 
                                       const __half* w_v,
                                       float* q_out, 
                                       float* k_out, 
                                       float* v_out,
                                       uint32_t D, 
                                       float eps) {
    
    const int WAVE_SIZE = 64;
    const int THREADS = 256;
    const int WAVES_PER_BLOCK = THREADS / WAVE_SIZE;
    uint32_t total_outputs = 3 * D;
    uint32_t blocks = (total_outputs + WAVES_PER_BLOCK - 1) / WAVES_PER_BLOCK;

    size_t lds_size = D * sizeof(float);

    hipLaunchKernelGGL(fused_rmsnorm_qkv_gemv_kernel_f16,
                       dim3(blocks), dim3(THREADS), lds_size, stream,
                       x, gamma, w_q, w_k, w_v, q_out, k_out, v_out, D, eps);
}

__device__ inline float fast_silu(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ inline float dot2(half2 a, half2 b, float c) {
    return __builtin_amdgcn_fdot2(a, b, c, false);
}

__global__ void __launch_bounds__(256)
fused_ffn_front_gemv_kernel_f16(const float* __restrict__ x,
                                const __half* __restrict__ w1,
                                const __half* __restrict__ w3,
                                float* __restrict__ y,
                                uint32_t D,
                                uint32_t H) {
    
    const int WAVE_SIZE = 64;
    const int THREADS_PER_BLOCK = 256;
    const int NUM_WAVES = THREADS_PER_BLOCK / WAVE_SIZE;
    
    __shared__ half2 s_xh2[2048]; 

    uint32_t D2 = D / 2;
    for (uint32_t i = threadIdx.x; i < D2; i += THREADS_PER_BLOCK) {
        s_xh2[i] = __floats2half2_rn(x[i*2], x[i*2 + 1]);
    }
    __syncthreads();

    int wave_id = threadIdx.x / WAVE_SIZE;
    int lane_id = threadIdx.x % WAVE_SIZE;

    uint32_t h_idx = blockIdx.x * NUM_WAVES + wave_id;
    if (h_idx >= H) return;

    const uint4* w1_u4 = reinterpret_cast<const uint4*>(w1 + h_idx * D);
    const uint4* w3_u4 = reinterpret_cast<const uint4*>(w3 + h_idx * D);
    
    uint32_t D8 = D / 8;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    // Loop 1: Stream W1 (Gate) - 8x Unroll
    #pragma unroll 8
    for (uint32_t j = lane_id; j < D8; j += WAVE_SIZE) {
        uint4 w = w1_u4[j];
        uint32_t l = j * 4;
        gate_acc = dot2(s_xh2[l],   ((half2*)&w)[0], gate_acc);
        gate_acc = dot2(s_xh2[l+1], ((half2*)&w)[1], gate_acc);
        gate_acc = dot2(s_xh2[l+2], ((half2*)&w)[2], gate_acc);
        gate_acc = dot2(s_xh2[l+3], ((half2*)&w)[3], gate_acc);
    }

    // Loop 2: Stream W3 (Up) - 8x Unroll
    #pragma unroll 8
    for (uint32_t j = lane_id; j < D8; j += WAVE_SIZE) {
        uint4 w = w3_u4[j];
        uint32_t l = j * 4;
        up_acc = dot2(s_xh2[l],   ((half2*)&w)[0], up_acc);
        up_acc = dot2(s_xh2[l+1], ((half2*)&w)[1], up_acc);
        up_acc = dot2(s_xh2[l+2], ((half2*)&w)[2], up_acc);
        up_acc = dot2(s_xh2[l+3], ((half2*)&w)[3], up_acc);
    }

    // Wave reduction
    for (int offset = 32; offset > 0; offset /= 2) {
        gate_acc += __shfl_down(gate_acc, offset);
        up_acc += __shfl_down(up_acc, offset);
    }

    if (lane_id == 0) {
        y[h_idx] = fast_silu(gate_acc) * up_acc;
    }
}

void launch_fused_ffn_front_f16(hipStream_t stream, const float *x,
                                const __half *w1, const __half *w3, float *y,
                                uint32_t D, uint32_t H) {
    const int WAVE_SIZE = 64;
    const int THREADS = 256;
    const int WAVES_PER_BLOCK = THREADS / WAVE_SIZE;
    uint32_t blocks = (H + WAVES_PER_BLOCK - 1) / WAVES_PER_BLOCK;

    hipLaunchKernelGGL(fused_ffn_front_gemv_kernel_f16, dim3(blocks),
                       dim3(THREADS), 0, stream, x, w1, w3, y, D, H);
}

} // namespace gcore::rt::hip::kernels
