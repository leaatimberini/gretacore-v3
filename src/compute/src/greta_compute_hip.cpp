#include "gcore/compute/greta_compute.hpp"
#include "gcore/rt/hip/greta_runtime_hip.hpp"
#include "gcore/rt/hip/kernels/attention_kernels.hpp"
#include "gcore/rt/hip/kernels/fused_attention_kernels.hpp"
#include "gcore/rt/hip/kernels/gemm_kernels.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

namespace gcore::compute {

using namespace gcore::rt::hip;
using namespace gcore::rt::hip::kernels;

GretaResult GretaCompute::gemm(GretaStream *stream, GretaMemory *A,
                               GretaMemory *B, GretaMemory *C, uint32_t M,
                               uint32_t N, uint32_t K, bool transpose_A,
                               bool transpose_B, GretaDataType accum_type) {
  auto *s = static_cast<GretaStreamHip *>(stream);

  uint32_t lda = K;
  uint32_t ldb = N;
  uint32_t ldc = N;

  // 1. Auditoría y Control de Entorno
  const char *force_gemm = std::getenv("GRETA_GEMM_FORCE");
  const char *profile_blocks = std::getenv("GRETA_PROFILE_BLOCKS");

  // 1. Auditoría y Control de Entorno
  const uint32_t GEMM_MFMA_THRESHOLD = 32;
  bool use_mfma = (M > GEMM_MFMA_THRESHOLD);
  std::string reason = use_mfma ? "M > threshold" : "M <= threshold";

  if (force_gemm) {
    std::string force_str(force_gemm);
    if (force_str == "MFMA") {
      use_mfma = true;
      reason = "Forced by GRETA_GEMM_FORCE=MFMA";
    } else if (force_str == "VALU") {
      use_mfma = false;
      reason = "Forced by GRETA_GEMM_FORCE=VALU";
    }
  }

  GretaDataType type_A = A->data_type();
  GretaDataType type_B = B->data_type();

  if (profile_blocks && std::string(profile_blocks) == "1") {
    printf(
        "[GRETA_L1_AUDIT] GEMM (M=%u, N=%u, K=%u) | Threshold=%u | Route=%s | "
        "Reason=M %s threshold | Types(A=%d, B=%d, Acc=%d)\n",
        M, N, K, GEMM_MFMA_THRESHOLD, use_mfma ? "MFMA" : "VALU",
        use_mfma ? ">" : "<=", (int)type_A, (int)type_B, (int)accum_type);

    // Assert de consistencia (solo en modo perfilado)
    if (use_mfma && M <= GEMM_MFMA_THRESHOLD && !force_gemm) {
      fprintf(stderr,
              "[GRETA_ERROR] Inconsistency: Route=MFMA but M <= threshold\n");
    }
  }

  // 2. Despacho de Kernel
  if (type_B == GretaDataType::INT8) {
    auto qinfo = B->quant_info();
    launch_gemm_mfma_int8_wt_fp32_acc32(
        s->handle(), static_cast<const float *>(A->data()),
        static_cast<const int8_t *>(B->data()), static_cast<float *>(C->data()),
        static_cast<const float *>(qinfo.scales), M, N, K, lda, ldb, ldc,
        qinfo.group_size);
  } else if (type_B == GretaDataType::INT4) {
    auto qinfo = B->quant_info();
    launch_gemm_mfma_int4_wt_fp32_acc32(
        s->handle(), A->data(), static_cast<const int8_t *>(B->data()),
        static_cast<float *>(C->data()),
        static_cast<const float *>(qinfo.scales), M, N, K, lda, ldb, ldc,
        qinfo.group_size, A->data_type() == GretaDataType::FP16);
  } else if (!use_mfma) {
    launch_gemm_mixed_f16f32(s->handle(), static_cast<const float *>(A->data()),
                             static_cast<const __half *>(B->data()),
                             static_cast<float *>(C->data()), M, N, K, lda, ldb,
                             ldc);
  } else {
    launch_gemm_mfma_mixed_f16f32(
        s->handle(), static_cast<const float *>(A->data()),
        static_cast<const __half *>(B->data()), static_cast<float *>(C->data()),
        M, N, K, lda, ldb, ldc);
  }

  return GretaResult::SUCCESS;
}

GretaResult GretaCompute::attention_decode(
    GretaStream *stream, GretaMemory *Q, GretaMemory *K_cache,
    GretaMemory *V_cache, GretaMemory *d_pos, GretaMemory *O,
    uint32_t num_heads, uint32_t head_dim, uint32_t seq_len,
    uint32_t max_seq_len, float scale, float rope_base) {
  auto *s = static_cast<GretaStreamHip *>(stream);

  // Auditoría
  const char *profile_blocks = std::getenv("GRETA_PROFILE_BLOCKS");
  if (profile_blocks && std::string(profile_blocks) == "1") {
    printf("[GRETA_L1_AUDIT] ATTN Decode | Heads: %u | HeadDim: %u | MaxSeq: "
           "%u | Scale: %.4f\n",
           num_heads, head_dim, max_seq_len, scale);
  }

  // Wrap Task 4.3 logic: FlashAttention Decode with Shared-Memory RoPE
  launch_flash_attention_decode_fused_rope(
      s->handle(), static_cast<const float *>(Q->data()),
      static_cast<const float *>(K_cache->data()),
      static_cast<const float *>(V_cache->data()),
      static_cast<float *>(O->data()), num_heads,
      static_cast<const uint32_t *>(d_pos->data()), max_seq_len, head_dim,
      scale, rope_base);

  return GretaResult::SUCCESS;
}

GretaResult GretaCompute::rmsnorm(GretaStream *stream, GretaMemory *input,
                                  GretaMemory *weight, GretaMemory *output,
                                  uint32_t dim, float eps) {
  // Placeholder for RMSNorm L1 (would call basic_kernels.hip)
  return GretaResult::SUCCESS;
}

} // namespace gcore::compute
