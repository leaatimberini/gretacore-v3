#pragma once
#include "gcore/rt/greta_runtime.hpp"
#include <cstdint>

/**
 * GRETA CORE - Compute Core (L1)
 */

namespace gcore::compute {

using namespace gcore::rt;

class GretaCompute {
public:
  static GretaResult gemm(GretaStream *stream, GretaMemory *A, GretaMemory *B,
                          GretaMemory *C, uint32_t M, uint32_t N, uint32_t K,
                          bool transpose_A = false, bool transpose_B = false,
                          GretaDataType accum_type = GretaDataType::FP32);

  static GretaResult
  attention_decode(GretaStream *stream, GretaMemory *Q, GretaMemory *K_cache,
                   GretaMemory *V_cache, GretaMemory *d_pos, GretaMemory *O,
                   uint32_t num_heads, uint32_t num_heads_kv,
                   uint32_t head_dim, uint32_t seq_len, uint32_t max_seq_len,
                   float scale, float rope_base);

  static GretaResult rmsnorm(GretaStream *stream, GretaMemory *input,
                             GretaMemory *weight, GretaMemory *output,
                             uint32_t dim, float eps);
};

class GretaFused {
public:
  static GretaResult rmsnorm_qkv(GretaStream *stream, GretaMemory *x,
                                 GretaMemory *gamma, GretaMemory *w_q,
                                 GretaMemory *w_k, GretaMemory *w_v,
                                 GretaMemory *q_out, GretaMemory *k_out,
                                 GretaMemory *v_out, uint32_t D, float eps);
};

} // namespace gcore::compute
