#include "gcore/compute/greta_compute.hpp"
#include "gcore/rt/hip/greta_runtime_hip.hpp"
#include "gcore/rt/hip/kernels/attention_kernels.hpp"
#include "gcore/rt/hip/kernels/fused_attention_kernels.hpp"
#include "gcore/rt/hip/kernels/gemm_kernels.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static bool env_flag(const char *k) {
  const char *v = std::getenv(k);
  return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

static bool prefill_force_wq_row() {
  return env_flag("GRETA_PREFILL_FORCE_WQ_ROW");
}

static bool prefill_force_wk_row() {
  return env_flag("GRETA_PREFILL_FORCE_WK_ROW");
}

static bool prefill_force_wv_row() {
  const char *v = std::getenv("GRETA_PREFILL_FORCE_WV_LAYOUT");
  if (v && *v) {
    if (std::strcmp(v, "row") == 0 || std::strcmp(v, "ROW") == 0)
      return true;
    if (std::strcmp(v, "col") == 0 || std::strcmp(v, "COL") == 0 ||
        std::strcmp(v, "auto") == 0 || std::strcmp(v, "AUTO") == 0)
      return false;
  }
  return env_flag("GRETA_PREFILL_FORCE_WV_ROW");
}

static const char *wo_layout_force_env() {
  const char *v = std::getenv("GRETA_WO_LAYOUT_FORCE");
  if (!v || !*v)
    return nullptr;
  return v;
}

static const char *prefill_qkv_layout_env() {
  const char *v = std::getenv("GRETA_PREFILL_QKV_LAYOUT");
  if (!v || !*v)
    return nullptr;
  return v;
}

static bool prefill_force_wo_row() {
  const char *layout = prefill_qkv_layout_env();
  if (layout && (std::strcmp(layout, "row") == 0 ||
                 std::strcmp(layout, "ROW") == 0)) {
    return true;
  }
  return false;
}

static bool trace_lmhead_enabled() {
  static const bool enabled =
      env_flag("GRETA_TRACE_PREFILL_DECODE_DELTA") ||
      env_flag("GRETA_TRACE_RMS_VERIFY") ||
      env_flag("GRETA_TRACE_LMHEAD_CPU_PROBE");
  return enabled;
}

static std::string &current_op_label() {
  static std::string label;
  return label;
}

static bool is_lm_head_label(const std::string &label) {
  return label == "lm_head" || label == "lm_head_prefill" ||
         label == "lm_head_decode";
}

static bool is_lm_head_decode_label(const std::string &label) {
  return label == "lm_head_decode";
}

static bool is_attn_decode_label(const std::string &label) {
  return label.rfind("attn_", 0) == 0 &&
         label.find("_decode") != std::string::npos;
}

static bool is_attn_q_prefill_label(const std::string &label) {
  return label == "attn_q_prefill";
}

static bool is_attn_k_prefill_label(const std::string &label) {
  return label == "attn_k_prefill";
}

static bool is_attn_v_prefill_label(const std::string &label) {
  return label == "attn_v_prefill";
}

static bool is_attn_o_label(const std::string &label) {
  return label == "attn_o_prefill" || label == "attn_o_decode";
}

static bool is_attn_o_prefill_label(const std::string &label) {
  return label == "attn_o_prefill";
}

static gcore::compute::GemmAuditInfo &last_gemm_audit() {
  static gcore::compute::GemmAuditInfo info;
  return info;
}

static uint64_t hash_f32(const float *p, size_t n) {
  const size_t count = (n < 64) ? n : 64;
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < count; ++i) {
    uint32_t v;
    std::memcpy(&v, &p[i], sizeof(uint32_t));
    h ^= static_cast<uint64_t>(v);
    h *= 1099511628211ull;
  }
  return h;
}

static std::string dtype_name(gcore::rt::GretaDataType t) {
  switch (t) {
  case gcore::rt::GretaDataType::FP32:
    return "FP32";
  case gcore::rt::GretaDataType::FP16:
    return "FP16";
  case gcore::rt::GretaDataType::BF16:
    return "BF16";
  case gcore::rt::GretaDataType::INT8:
    return "INT8";
  case gcore::rt::GretaDataType::INT4:
    return "INT4";
  case gcore::rt::GretaDataType::Q4_K:
    return "Q4_K";
  default:
    return "UNKNOWN";
  }
}

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
  const char *env_perhead = std::getenv("GRETA_PERHEAD_QKV");
  bool perhead_enabled =
      (env_perhead == nullptr || std::string(env_perhead) == "1");

  // 1. Auditoría y Control de Entorno
  const uint32_t GEMM_MFMA_THRESHOLD = 32;
  bool use_mfma = (M > GEMM_MFMA_THRESHOLD);
  std::string reason = use_mfma ? "M > threshold" : "M <= threshold";

  const std::string op_label = current_op_label();
  const bool is_lm_head = is_lm_head_label(op_label);
  const bool is_lm_head_decode = is_lm_head_decode_label(op_label);

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

  const char *attn_force = std::getenv("GRETA_FORCE_ATTN_DECODE_MATMUL");
  if (attn_force && is_attn_decode_label(op_label)) {
    std::string force_str(attn_force);
    if (force_str == "mfma" || force_str == "MFMA") {
      use_mfma = true;
      reason = "Forced by GRETA_FORCE_ATTN_DECODE_MATMUL=mfma";
    } else if (force_str == "valu" || force_str == "VALU") {
      use_mfma = false;
      reason = "Forced by GRETA_FORCE_ATTN_DECODE_MATMUL=valu";
    }
  }
  std::string lmhead_force_route = "auto";
  std::string lmhead_force_route_decode = "auto";
  const char *lmhead_force = std::getenv("GRETA_LMHEAD_FORCE_ROUTE");
  if (lmhead_force && is_lm_head) {
    std::string force_str(lmhead_force);
    if (force_str == "mfma" || force_str == "MFMA") {
      use_mfma = true;
      reason = "Forced by GRETA_LMHEAD_FORCE_ROUTE=mfma";
      lmhead_force_route = "mfma";
    } else if (force_str == "valu" || force_str == "VALU") {
      use_mfma = false;
      reason = "Forced by GRETA_LMHEAD_FORCE_ROUTE=valu";
      lmhead_force_route = "valu";
    } else {
      lmhead_force_route = "auto";
    }
  }
  const char *lmhead_force_decode = std::getenv("GRETA_LMHEAD_FORCE_ROUTE_DECODE");
  if (lmhead_force_decode && is_lm_head_decode) {
    std::string force_str(lmhead_force_decode);
    if (force_str == "mfma" || force_str == "MFMA") {
      use_mfma = true;
      reason = "Forced by GRETA_LMHEAD_FORCE_ROUTE_DECODE=mfma";
      lmhead_force_route_decode = "mfma";
    } else if (force_str == "valu" || force_str == "VALU") {
      use_mfma = false;
      reason = "Forced by GRETA_LMHEAD_FORCE_ROUTE_DECODE=valu";
      lmhead_force_route_decode = "valu";
    } else {
      lmhead_force_route_decode = "auto";
    }
  }
  if (is_lm_head && lmhead_force_route == "auto" &&
      (!is_lm_head_decode || lmhead_force_route_decode == "auto")) {
    use_mfma = false;
    reason = "LM head MFMA disabled (B3.16)";
  }



  GretaDataType type_A = A->data_type();
  GretaDataType type_B = B->data_type();

  const bool trace_audit = trace_lmhead_enabled() && is_lm_head;
  if (trace_audit) {
    auto &info = last_gemm_audit();
    info.op_label = op_label;
    info.route = use_mfma ? "MFMA" : "VALU";
    info.force_route = lmhead_force_route;
    info.force_route_decode = lmhead_force_route_decode;
    info.quant_mode = dtype_name(type_B);
    info.layout_used = "KxN row_major";
    info.layout_assumed = "KxN row_major";
    info.layout_actual = "KxN row_major";
    info.type_a = static_cast<int>(type_A);
    info.type_b = static_cast<int>(type_B);
    info.accum_type = static_cast<int>(accum_type);
    info.m = M;
    info.n = N;
    info.k = K;
    info.lda = lda;
    info.ldb = ldb;
    info.ldc = ldc;
    info.a_ptr = reinterpret_cast<uintptr_t>(A->data());
    info.b_ptr_base = reinterpret_cast<uintptr_t>(B->data());
    info.b_ptr_effective = reinterpret_cast<uintptr_t>(B->data());
    info.c_ptr = reinterpret_cast<uintptr_t>(C->data());
    info.perhead_enabled = perhead_enabled;
    info.scales_ptr = 0;
    info.scales_hash = 0;
    info.head_scales_ptr = 0;
    info.head_scales_hash = 0;
    if ((type_B == GretaDataType::INT4 || type_B == GretaDataType::INT8)) {
      auto qinfo = B->quant_info();
      info.scales_ptr = reinterpret_cast<uintptr_t>(qinfo.scales);
      info.head_scales_ptr = reinterpret_cast<uintptr_t>(qinfo.head_scales);
      if (qinfo.scales) {
        float tmp[64];
        if (hipMemcpy(tmp, qinfo.scales, sizeof(tmp), hipMemcpyDeviceToHost) == hipSuccess) {
          info.scales_hash = hash_f32(tmp, 64);
        }
      }
      if (qinfo.head_scales && qinfo.num_heads > 0) {
        const size_t count = qinfo.num_heads < 64 ? qinfo.num_heads : 64;
        std::vector<float> tmp(count);
        if (hipMemcpy(tmp.data(), qinfo.head_scales, count * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess) {
          info.head_scales_hash = hash_f32(tmp.data(), count);
        }
      }
    }
  }

  if (profile_blocks && std::string(profile_blocks) == "1") {
    printf(
        "[GRETA_L1_AUDIT] GEMM (M=%u, N=%u, K=%u) | Threshold=%u | Route=%s | "
        "Reason=M %s threshold | Types(A=%d, B=%d, Acc=%d) | Heads=%u | "
        "PH=%s\n",
        M, N, K, GEMM_MFMA_THRESHOLD, use_mfma ? "MFMA" : "VALU",
        use_mfma ? ">" : "<=", (int)type_A, (int)type_B, (int)accum_type,
        B->quant_info().num_heads, perhead_enabled ? "ON" : "OFF");

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
    uint32_t num_heads = qinfo.num_heads;
    uint32_t head_dim = (num_heads > 0) ? (N / num_heads) : 0;
    bool force_gemv = false;
    if (is_attn_q_prefill_label(op_label)) {
      if (prefill_force_wq_row())
        force_gemv = true;
      const char *layout = prefill_qkv_layout_env();
      if (layout && (std::strcmp(layout, "row") == 0 ||
                     std::strcmp(layout, "ROW") == 0)) {
        force_gemv = true;
      }
    }
    if (is_attn_k_prefill_label(op_label)) {
      if (prefill_force_wk_row())
        force_gemv = true;
      const char *layout = prefill_qkv_layout_env();
      if (layout && (std::strcmp(layout, "row") == 0 ||
                     std::strcmp(layout, "ROW") == 0)) {
        force_gemv = true;
      }
    }
    if (is_attn_v_prefill_label(op_label)) {
      if (prefill_force_wv_row())
        force_gemv = true;
    }
    if (is_attn_o_prefill_label(op_label)) {
      if (prefill_force_wo_row())
        force_gemv = true;
    }
    if (is_attn_o_label(op_label)) {
      const char *layout = wo_layout_force_env();
      if (layout && (std::strcmp(layout, "row") == 0 ||
                     std::strcmp(layout, "ROW") == 0)) {
        force_gemv = true;
      } else if (layout && (std::strcmp(layout, "col") == 0 ||
                            std::strcmp(layout, "COL") == 0)) {
        force_gemv = false;
      }
    }
    launch_gemm_mfma_int4_wt_fp32_acc32(
        s->handle(), A->data(), static_cast<const int8_t *>(B->data()),
        static_cast<float *>(C->data()),
        static_cast<const float *>(qinfo.scales),
        perhead_enabled ? static_cast<const float *>(qinfo.head_scales)
                        : nullptr,
        M, N, K, lda, ldb, ldc, qinfo.group_size, head_dim,
        A->data_type() == GretaDataType::FP16, force_gemv);
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
    uint32_t num_heads, uint32_t num_heads_kv, uint32_t head_dim,
    uint32_t seq_len, uint32_t max_seq_len, float scale, float rope_base) {
  auto *s = static_cast<GretaStreamHip *>(stream);

  // Auditoría
  const char *profile_blocks = std::getenv("GRETA_PROFILE_BLOCKS");
  if (profile_blocks && std::string(profile_blocks) == "1") {
    printf("[GRETA_L1_AUDIT] ATTN Decode | Heads: %u | KV Heads: %u | "
           "HeadDim: %u | MaxSeq: %u | Scale: %.4f\n",
           num_heads, num_heads_kv, head_dim, max_seq_len, scale);
  }

  (void)rope_base;
  int accum_mode = 0;
  const char *accum_env = std::getenv("GRETA_ATTN_ACCUM");
  if (accum_env) {
    std::string mode(accum_env);
    if (mode == "fp16" || mode == "FP16") {
      accum_mode = 1;
    } else if (mode == "fp32" || mode == "FP32") {
      accum_mode = 0;
    }
  }
  // Non-fused decode path (RoPE applied in scheduler)
  launch_flash_attention_decode(
      s->handle(), static_cast<const float *>(Q->data()),
      static_cast<const float *>(K_cache->data()),
      static_cast<const float *>(V_cache->data()),
      static_cast<float *>(O->data()), num_heads, num_heads_kv,
      static_cast<const uint32_t *>(d_pos->data()), max_seq_len, head_dim,
      scale, accum_mode);

  return GretaResult::SUCCESS;
}

GretaResult GretaCompute::rmsnorm(GretaStream *stream, GretaMemory *input,
                                  GretaMemory *weight, GretaMemory *output,
                                  uint32_t dim, float eps) {
  // Placeholder for RMSNorm L1 (would call basic_kernels.hip)
  return GretaResult::SUCCESS;
}

void GretaCompute::set_op_label(const char *label) {
  if (label) {
    current_op_label() = label;
  } else {
    current_op_label().clear();
  }
}

GemmAuditInfo GretaCompute::get_last_gemm_audit() {
  return last_gemm_audit();
}

} // namespace gcore::compute
