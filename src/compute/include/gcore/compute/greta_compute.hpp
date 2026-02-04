#pragma once
#include "gcore/rt/greta_runtime.hpp"
#include <cstdint>
#include <string>

/**
 * GRETA CORE - Compute Core (L1)
 */

namespace gcore::compute {

using namespace gcore::rt;


struct GemmAuditInfo {
  std::string op_label;
  std::string route;
  std::string force_route;
  std::string force_route_decode;
  std::string quant_mode;
  std::string layout_used;
  std::string layout_assumed;
  std::string layout_actual;
  int type_a = 0;
  int type_b = 0;
  int accum_type = 0;
  uint32_t m = 0;
  uint32_t n = 0;
  uint32_t k = 0;
  uint32_t lda = 0;
  uint32_t ldb = 0;
  uint32_t ldc = 0;
  uintptr_t a_ptr = 0;
  uintptr_t b_ptr_base = 0;
  uintptr_t b_ptr_effective = 0;
  uintptr_t c_ptr = 0;
  bool perhead_enabled = false;
  uintptr_t scales_ptr = 0;
  uint64_t scales_hash = 0;
  uintptr_t head_scales_ptr = 0;
  uint64_t head_scales_hash = 0;
};

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

  static void set_op_label(const char *label);
  static GemmAuditInfo get_last_gemm_audit();
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
