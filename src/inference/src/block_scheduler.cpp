#include "gcore/inference/block_scheduler.hpp"
#include "gcore/compute/greta_compute.hpp"
#include "gcore/inference/stage_trace.hpp"
#include "gcore/inference/weight_loader.hpp"
#include "gcore/rt/greta_runtime.hpp"
#include "gcore/rt/hip/greta_runtime_hip.hpp"
#include "gcore/rt/hip/kernels/attention_kernels.hpp"
#include "gcore/rt/hip/kernels/basic_kernels.hpp"
#include "gcore/rt/hip/kernels/fused_attention_kernels.hpp"
#include "gcore/rt/hip/kernels/fused_compute_kernels.hpp"
#include "gcore/rt/hip/kernels/gemm_kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <hip/hip_fp16.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__GNUC__)
#define GRETA_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define GRETA_UNLIKELY(x) (x)
#endif

#define TRACE_ON(layer)                                                        \
  (GRETA_UNLIKELY(tracer_.should_trace_layer((int)(layer), trace_step_)))

#define PROFILE_ON() (GRETA_UNLIKELY(tracer_.profile_enabled()))

namespace gcore::inference {

static bool env_flag(const char *k) {
  const char *v = std::getenv(k);
  return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

struct F32Stats {
  float min = 0.0f;
  float max = 0.0f;
  float mean = 0.0f;
  int nan = 0;
  int inf = 0;
};

enum class AttnTracePoint : uint32_t {
  Q = 1u << 0,
  K = 1u << 1,
  V = 1u << 2,
  ATTN_OUT = 1u << 3,
  X_OUT = 1u << 4,
};

static uint32_t attn_point_mask_from_list(const char *v) {
  if (!v || !*v)
    return static_cast<uint32_t>(AttnTracePoint::Q) |
           static_cast<uint32_t>(AttnTracePoint::K) |
           static_cast<uint32_t>(AttnTracePoint::V) |
           static_cast<uint32_t>(AttnTracePoint::ATTN_OUT) |
           static_cast<uint32_t>(AttnTracePoint::X_OUT);
  std::string s(v);
  uint32_t mask = 0;
  size_t start = 0;
  while (start < s.size()) {
    size_t end = s.find(',', start);
    std::string token = s.substr(
        start, (end == std::string::npos) ? s.size() - start : end - start);
    if (token == "q")
      mask |= static_cast<uint32_t>(AttnTracePoint::Q);
    else if (token == "k")
      mask |= static_cast<uint32_t>(AttnTracePoint::K);
    else if (token == "v")
      mask |= static_cast<uint32_t>(AttnTracePoint::V);
    else if (token == "attn_out")
      mask |= static_cast<uint32_t>(AttnTracePoint::ATTN_OUT);
    else if (token == "x_out")
      mask |= static_cast<uint32_t>(AttnTracePoint::X_OUT);
    if (end == std::string::npos)
      break;
    start = end + 1;
  }
  if (mask == 0) {
    mask = static_cast<uint32_t>(AttnTracePoint::Q) |
           static_cast<uint32_t>(AttnTracePoint::K) |
           static_cast<uint32_t>(AttnTracePoint::V) |
           static_cast<uint32_t>(AttnTracePoint::ATTN_OUT) |
           static_cast<uint32_t>(AttnTracePoint::X_OUT);
  }
  return mask;
}

static std::vector<int> parse_layers(const char *v) {
  std::vector<int> layers;
  if (!v || !*v)
    return layers;
  std::string s(v);
  if (s == "all" || s == "ALL" || s == "*")
    return layers;
  size_t start = 0;
  while (start < s.size()) {
    size_t end = s.find(',', start);
    std::string token = s.substr(
        start, (end == std::string::npos) ? s.size() - start : end - start);
    if (!token.empty()) {
      char *e = nullptr;
      int val = std::strtol(token.c_str(), &e, 10);
      if (e != token.c_str())
        layers.push_back(val);
    }
    if (end == std::string::npos)
      break;
    start = end + 1;
  }
  return layers;
}

static bool attn_trace_layer_selected(size_t layer_idx, size_t num_layers) {
  const char *layers_env = std::getenv("GRETA_TRACE_ATTN_LAYERS");
  std::vector<int> layers = parse_layers(layers_env);
  if (layers.empty()) {
    const int last = num_layers > 0 ? static_cast<int>(num_layers - 1) : 0;
    const int defaults[] = {0, 1, 2, last};
    for (int l : defaults) {
      if (l == static_cast<int>(layer_idx))
        return true;
    }
    return false;
  }
  for (int l : layers) {
    if (l == static_cast<int>(layer_idx))
      return true;
  }
  return false;
}

static const char *attn_trace_out_path() {
  const char *out = std::getenv("GRETA_TRACE_ATTN_DECODE_OUT");
  if (out && *out)
    return out;
  return std::getenv("GRETA_TRACE_PREFILL_DECODE_OUT");
}

static const char *attn_shadow_out_path() {
  const char *out = std::getenv("GRETA_ATTN_DECODE_MFMA_SHADOW_OUT");
  if (out && *out)
    return out;
  return attn_trace_out_path();
}

static bool trace_attn_decode_verify_enabled() {
  return env_flag("GRETA_TRACE_ATTN_DECODE_VERIFY");
}

static bool attn_decode_ref_enabled() {
  return env_flag("GRETA_ATTN_DECODE_REF");
}

static bool trace_attn_ref_enabled() {
  return env_flag("GRETA_TRACE_ATTN_REF");
}

static const char *attn_ref_out_path() {
  const char *out = std::getenv("GRETA_TRACE_ATTN_REF_OUT");
  if (out && *out)
    return out;
  return nullptr;
}

static bool trace_attn_softmax_enabled() {
  return env_flag("GRETA_TRACE_ATTN_SOFTMAX");
}

static const char *attn_softmax_out_path() {
  const char *out = std::getenv("GRETA_TRACE_ATTN_OUT");
  if (out && *out)
    return out;
  return nullptr;
}

static bool attn_softmax_layer_selected(size_t layer_idx, size_t num_layers) {
  const char *v = std::getenv("GRETA_TRACE_ATTN_LAYER");
  if (v && *v) {
    char *e = nullptr;
    int val = std::strtol(v, &e, 10);
    if (e != v)
      return static_cast<int>(layer_idx) == val;
  }
  return attn_trace_layer_selected(layer_idx, num_layers);
}

static uint32_t attn_softmax_head() {
  const char *v = std::getenv("GRETA_TRACE_ATTN_HEAD");
  if (!v || !*v)
    return 0;
  char *e = nullptr;
  long val = std::strtol(v, &e, 10);
  if (e == v || val < 0)
    return 0;
  return static_cast<uint32_t>(val);
}

static uint32_t attn_softmax_window() {
  const char *v = std::getenv("GRETA_TRACE_ATTN_KEYS_WINDOW");
  if (!v || !*v)
    return 64;
  char *e = nullptr;
  long val = std::strtol(v, &e, 10);
  if (e == v || val <= 0)
    return 64;
  return static_cast<uint32_t>(val);
}

static bool trace_attn_vacc_enabled() {
  return env_flag("GRETA_TRACE_ATTN_VACC");
}

static const char *attn_vacc_out_path() {
  const char *out = std::getenv("GRETA_TRACE_ATTN_OUT");
  if (out && *out)
    return out;
  return nullptr;
}

static bool trace_attn_l0_pipe_enabled() {
  return env_flag("GRETA_TRACE_ATTN_L0_PIPE");
}

static bool trace_attn_l0_norm_enabled() {
  return env_flag("GRETA_TRACE_ATTN_L0_NORM");
}

static bool trace_qkv_w_verify_enabled() {
  return env_flag("GRETA_TRACE_QKV_W_VERIFY");
}

static bool trace_wo_w_verify_enabled() {
  return env_flag("GRETA_TRACE_WO_W_VERIFY");
}

static bool trace_post_wo_enabled() { return env_flag("GRETA_TRACE_POST_WO"); }

static const char *post_wo_out_path() {
  const char *out = std::getenv("GRETA_TRACE_POST_WO_OUT");
  if (out && *out)
    return out;
  return nullptr;
}

static uint32_t post_wo_sample() {
  const char *v = std::getenv("GRETA_TRACE_POST_WO_SAMPLE");
  if (!v || !*v)
    return 1024;
  char *e = nullptr;
  long val = std::strtol(v, &e, 10);
  if (e == v || val <= 0)
    return 1024;
  return static_cast<uint32_t>(val);
}

static bool post_wo_trace_all_layers() {
  static bool cached = false;
  static bool trace_all = false;
  if (!cached) {
    const char *v = std::getenv("GRETA_TRACE_POST_WO_LAYERS");
    if (v && *v) {
      std::string s(v);
      if (s == "all" || s == "ALL" || s == "*")
        trace_all = true;
    }
    cached = true;
  }
  return trace_all;
}

static std::vector<int> post_wo_layers() {
  static bool cached = false;
  static std::vector<int> layers;
  if (!cached) {
    const char *v = std::getenv("GRETA_TRACE_POST_WO_LAYERS");
    if (!v || !*v) {
      layers.push_back(0);
    } else {
      std::string s(v);
      size_t start = 0;
      while (start < s.size()) {
        size_t end = s.find(',', start);
        std::string token = s.substr(
            start, (end == std::string::npos) ? s.size() - start : end - start);
        if (!token.empty()) {
          char *e = nullptr;
          long val = std::strtol(token.c_str(), &e, 10);
          if (e != token.c_str())
            layers.push_back(static_cast<int>(val));
        }
        if (end == std::string::npos)
          break;
        start = end + 1;
      }
      if (layers.empty() && !post_wo_trace_all_layers()) {
        layers.push_back(0);
      }
    }
    cached = true;
  }
  return layers;
}

static bool post_wo_layer_selected(size_t layer_idx, size_t num_layers) {
  if (post_wo_trace_all_layers())
    return true;
  const auto layers = post_wo_layers();
  if (layers.empty())
    return true;
  for (int layer : layers) {
    if (layer < 0 && static_cast<size_t>(-layer) == num_layers)
      return true;
    if (static_cast<size_t>(layer) == layer_idx)
      return true;
  }
  return false;
}

static bool post_wo_phase_enabled(const char *phase) {
  const char *v = std::getenv("GRETA_TRACE_POST_WO_PHASES");
  if (!v || !*v)
    return true;
  std::string s(v);
  size_t start = 0;
  while (start < s.size()) {
    size_t end = s.find(',', start);
    std::string token = s.substr(
        start, (end == std::string::npos) ? s.size() - start : end - start);
    if (!token.empty() && token == phase)
      return true;
    if (end == std::string::npos)
      break;
    start = end + 1;
  }
  return false;
}

static bool trace_rmsnorm_enabled() { return env_flag("GRETA_TRACE_RMSNORM"); }

static const char *rmsnorm_out_path() {
  const char *out = std::getenv("GRETA_TRACE_RMSNORM_OUT");
  if (out && *out)
    return out;
  return nullptr;
}

static uint32_t rmsnorm_sample() {
  const char *v = std::getenv("GRETA_TRACE_RMSNORM_SAMPLE");
  if (!v || !*v)
    return 1024;
  char *e = nullptr;
  long val = std::strtol(v, &e, 10);
  if (e == v || val <= 0)
    return 1024;
  return static_cast<uint32_t>(val);
}

static bool rmsnorm_trace_all_layers() {
  static bool cached = false;
  static bool trace_all = false;
  if (!cached) {
    const char *v = std::getenv("GRETA_TRACE_RMSNORM_LAYERS");
    if (v && *v) {
      std::string s(v);
      if (s == "all" || s == "ALL" || s == "*")
        trace_all = true;
    }
    cached = true;
  }
  return trace_all;
}

static std::vector<int> rmsnorm_layers() {
  static bool cached = false;
  static std::vector<int> layers;
  if (!cached) {
    const char *v = std::getenv("GRETA_TRACE_RMSNORM_LAYERS");
    if (!v || !*v) {
      layers.push_back(0);
    } else {
      std::string s(v);
      size_t start = 0;
      while (start < s.size()) {
        size_t end = s.find(',', start);
        std::string token = s.substr(
            start, (end == std::string::npos) ? s.size() - start : end - start);
        if (!token.empty()) {
          char *e = nullptr;
          long val = std::strtol(token.c_str(), &e, 10);
          if (e != token.c_str())
            layers.push_back(static_cast<int>(val));
        }
        if (end == std::string::npos)
          break;
        start = end + 1;
      }
      if (layers.empty() && !rmsnorm_trace_all_layers()) {
        layers.push_back(0);
      }
    }
    cached = true;
  }
  return layers;
}

static bool rmsnorm_layer_selected(size_t layer_idx, size_t num_layers) {
  if (rmsnorm_trace_all_layers())
    return true;
  const auto layers = rmsnorm_layers();
  if (layers.empty())
    return true;
  for (int layer : layers) {
    if (layer < 0 && static_cast<size_t>(-layer) == num_layers)
      return true;
    if (static_cast<size_t>(layer) == layer_idx)
      return true;
  }
  return false;
}

static bool rmsnorm_phase_enabled(const char *phase) {
  const char *v = std::getenv("GRETA_TRACE_RMSNORM_PHASES");
  if (!v || !*v)
    return true;
  std::string s(v);
  size_t start = 0;
  while (start < s.size()) {
    size_t end = s.find(',', start);
    std::string token = s.substr(
        start, (end == std::string::npos) ? s.size() - start : end - start);
    if (!token.empty() && token == phase)
      return true;
    if (end == std::string::npos)
      break;
    start = end + 1;
  }
  return false;
}

static const char *attn_l0_pipe_out_path() {
  const char *out = std::getenv("GRETA_TRACE_ATTN_L0_PIPE_OUT");
  if (out && *out)
    return out;
  return nullptr;
}

static const char *qkv_force_route_env() {
  const char *v = std::getenv("GRETA_QKV_FORCE_ROUTE");
  if (!v || !*v)
    return nullptr;
  if (std::strcmp(v, "auto") == 0 || std::strcmp(v, "AUTO") == 0)
    return nullptr;
  return v;
}

static bool qkv_force_gemm_enabled() {
  return env_flag("GRETA_QKV_FORCE_GEMM");
}

static std::string to_route_label(const char *v) {
  if (!v || !*v)
    return "auto";
  std::string s(v);
  if (s == "mfma" || s == "MFMA")
    return "MFMA";
  if (s == "valu" || s == "VALU")
    return "VALU";
  return s;
}

static std::string qkv_route_used(uint32_t m, bool is_decode_step,
                                  bool use_fused, const char *force_route) {
  if (use_fused)
    return "FUSED_GEMV";

  if (force_route && *force_route)
    return to_route_label(force_route);

  const char *force_gemm = std::getenv("GRETA_GEMM_FORCE");
  if (force_gemm && *force_gemm)
    return to_route_label(force_gemm);

  if (is_decode_step) {
    const char *attn_force = std::getenv("GRETA_FORCE_ATTN_DECODE_MATMUL");
    if (attn_force && *attn_force)
      return to_route_label(attn_force);
  }

  const uint32_t GEMM_MFMA_THRESHOLD = 32;
  return (m > GEMM_MFMA_THRESHOLD) ? "MFMA" : "VALU";
}

static bool trace_v_addr_enabled() { return env_flag("GRETA_TRACE_V_ADDR"); }

static const char *v_addr_out_path() {
  const char *out = std::getenv("GRETA_TRACE_V_ADDR_OUT");
  if (out && *out)
    return out;
  return nullptr;
}

static uint32_t attn_vacc_dims_sample() {
  const char *v = std::getenv("GRETA_TRACE_ATTN_DIMS_SAMPLE");
  if (!v || !*v)
    return 16;
  char *e = nullptr;
  long val = std::strtol(v, &e, 10);
  if (e == v || val <= 0)
    return 16;
  return static_cast<uint32_t>(val);
}

static bool attn_mfma_shadow_enabled() {
  return env_flag("GRETA_ATTN_DECODE_MFMA_SHADOW");
}

static void mae_max_f32(const float *a, const float *b, size_t n, double *mae,
                        float *max_diff) {
  if (!mae || !max_diff)
    return;
  if (!a || !b || n == 0) {
    *mae = 0.0;
    *max_diff = 0.0f;
    return;
  }
  double sum = 0.0;
  float maxd = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float d = std::abs(a[i] - b[i]);
    sum += static_cast<double>(d);
    if (d > maxd)
      maxd = d;
  }
  *mae = sum / static_cast<double>(n);
  *max_diff = maxd;
}

static float fp16_to_fp32_local(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0) {
    if (mant == 0)
      return sign ? -0.0f : 0.0f;
    exp = 1;
    while ((mant & 0x400) == 0) {
      mant <<= 1;
      exp--;
    }
    mant &= ~0x400;
  } else if (exp == 31) {
    return sign ? -INFINITY : INFINITY;
  }
  uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  float result;
  std::memcpy(&result, &f, 4);
  return result;
}

static const char *dtype_label(gcore::rt::GretaDataType type) {
  switch (type) {
  case gcore::rt::GretaDataType::INT4:
    return "INT4";
  case gcore::rt::GretaDataType::INT8:
    return "INT8";
  case gcore::rt::GretaDataType::FP16:
    return "FP16";
  case gcore::rt::GretaDataType::FP32:
    return "FP32";
  default:
    return "UNKNOWN";
  }
}

struct QkvWeightHostCache {
  bool ready = false;
  gcore::rt::GretaDataType type = gcore::rt::GretaDataType::FP32;
  size_t bytes = 0;
  size_t elems = 0;
  uint32_t group_size = 0;
  std::vector<uint8_t> raw;
  std::vector<float> scales;
  std::vector<float> head_scales;
};

static bool ensure_qkv_weight_cache(const gcore::rt::hip::Buffer &w,
                                    const gcore::rt::hip::Buffer &scales,
                                    const gcore::rt::hip::Buffer &head_scales,
                                    size_t elems, QkvWeightHostCache *cache) {
  if (!cache)
    return false;
  if (cache->ready && cache->bytes == w.size() && cache->elems == elems &&
      cache->type == w.data_type())
    return true;

  cache->ready = false;
  cache->type = w.data_type();
  cache->bytes = w.size();
  cache->elems = elems;
  cache->group_size = w.quant_info().group_size;
  cache->raw.assign(cache->bytes, 0);
  if (cache->bytes > 0 &&
      !w.copy_to_host(cache->raw.data(), cache->bytes, nullptr)) {
    return false;
  }

  cache->scales.clear();
  if (scales.size() > 0) {
    size_t count = scales.size() / sizeof(float);
    cache->scales.assign(count, 0.0f);
    if (!scales.copy_to_host(cache->scales.data(), scales.size(), nullptr)) {
      return false;
    }
  }

  cache->head_scales.clear();
  if (head_scales.size() > 0) {
    size_t count = head_scales.size() / sizeof(float);
    cache->head_scales.assign(count, 0.0f);
    if (!head_scales.copy_to_host(cache->head_scales.data(), head_scales.size(),
                                  nullptr)) {
      return false;
    }
  }

  cache->ready = true;
  return true;
}

static float read_weight_value(const QkvWeightHostCache &cache, size_t idx) {
  if (idx >= cache.elems)
    return 0.0f;
  if (cache.type == gcore::rt::GretaDataType::INT4) {
    size_t byte_idx = idx / 2;
    uint8_t packed = cache.raw[byte_idx];
    int8_t v = (idx % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
    if (v & 0x08)
      v |= 0xF0;
    float scale = 1.0f;
    if (!cache.scales.empty() && cache.group_size > 0) {
      scale = cache.scales[idx / cache.group_size];
    }
    return static_cast<float>(v) * scale;
  }
  if (cache.type == gcore::rt::GretaDataType::INT8) {
    const int8_t *w = reinterpret_cast<const int8_t *>(cache.raw.data());
    float scale = 1.0f;
    if (!cache.scales.empty() && cache.group_size > 0) {
      scale = cache.scales[idx / cache.group_size];
    }
    return static_cast<float>(w[idx]) * scale;
  }
  if (cache.type == gcore::rt::GretaDataType::FP16) {
    const uint16_t *w = reinterpret_cast<const uint16_t *>(cache.raw.data());
    return fp16_to_fp32_local(w[idx]);
  }
  if (cache.type == gcore::rt::GretaDataType::FP32) {
    const float *w = reinterpret_cast<const float *>(cache.raw.data());
    return w[idx];
  }
  return 0.0f;
}

struct EnvOverride {
  std::string key;
  std::string prev;
  bool had_prev = false;
  EnvOverride(const char *k, const char *v) : key(k) {
    const char *p = std::getenv(k);
    if (p) {
      had_prev = true;
      prev = p;
    }
    if (v)
      setenv(k, v, 1);
  }
  ~EnvOverride() {
    if (had_prev) {
      setenv(key.c_str(), prev.c_str(), 1);
    } else {
      unsetenv(key.c_str());
    }
  }
};

static double mae_f32(const float *a, const float *b, size_t n) {
  if (n == 0)
    return 0.0;
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
  }
  return sum / static_cast<double>(n);
}

enum class AttnAccumMode { Fp32 = 0, Fp16 = 1 };

static AttnAccumMode attn_accum_mode() {
  const char *v = std::getenv("GRETA_ATTN_ACCUM");
  if (!v)
    return AttnAccumMode::Fp32;
  std::string s(v);
  if (s == "fp16" || s == "FP16")
    return AttnAccumMode::Fp16;
  return AttnAccumMode::Fp32;
}

static inline float round_fp16_host(float x) {
  return __half2float(__float2half_rn(x));
}

static void compute_attention_ref_fp32(const float *q, const float *k_cache,
                                       const float *v_cache, uint32_t num_heads,
                                       uint32_t num_heads_kv, uint32_t head_dim,
                                       uint32_t seq_len, uint32_t max_seq_len,
                                       float scale, std::vector<float> &out) {
  if (!q || !k_cache || !v_cache || num_heads == 0 || head_dim == 0)
    return;
  out.assign(static_cast<size_t>(num_heads) * head_dim, 0.0f);
  const uint32_t group = (num_heads_kv > 0) ? (num_heads / num_heads_kv) : 0;
  for (uint32_t h = 0; h < num_heads; ++h) {
    const uint32_t kv_head = (group > 0) ? (h / group) : 0;
    const float *q_ptr = q + h * head_dim;
    const float *k_ptr = k_cache + kv_head * max_seq_len * head_dim;
    const float *v_ptr = v_cache + kv_head * max_seq_len * head_dim;

    std::vector<float> scores(seq_len, 0.0f);
    float max_score = -INFINITY;
    for (uint32_t t = 0; t < seq_len; ++t) {
      const float *k_t = k_ptr + t * head_dim;
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_dim; ++d) {
        dot += q_ptr[d] * k_t[d];
      }
      float s = dot * scale;
      scores[t] = s;
      if (s > max_score)
        max_score = s;
    }
    double sum = 0.0;
    for (uint32_t t = 0; t < seq_len; ++t) {
      scores[t] = std::exp(scores[t] - max_score);
      sum += scores[t];
    }
    const double inv_sum = sum > 0.0 ? (1.0 / sum) : 0.0;

    float *o_ptr = out.data() + h * head_dim;
    for (uint32_t d = 0; d < head_dim; ++d) {
      double acc = 0.0;
      for (uint32_t t = 0; t < seq_len; ++t) {
        const float *v_t = v_ptr + t * head_dim;
        acc += static_cast<double>(scores[t]) * static_cast<double>(v_t[d]);
      }
      o_ptr[d] = static_cast<float>(acc * inv_sum);
    }
  }
}

static void
compute_attention_ref_fp16_accum(const float *q, const float *k_cache,
                                 const float *v_cache, uint32_t num_heads,
                                 uint32_t num_heads_kv, uint32_t head_dim,
                                 uint32_t seq_len, uint32_t max_seq_len,
                                 float scale, std::vector<float> &out) {
  if (!q || !k_cache || !v_cache || num_heads == 0 || head_dim == 0)
    return;
  out.assign(static_cast<size_t>(num_heads) * head_dim, 0.0f);
  const uint32_t group = (num_heads_kv > 0) ? (num_heads / num_heads_kv) : 0;
  for (uint32_t h = 0; h < num_heads; ++h) {
    const uint32_t kv_head = (group > 0) ? (h / group) : 0;
    const float *q_ptr = q + h * head_dim;
    const float *k_ptr = k_cache + kv_head * max_seq_len * head_dim;
    const float *v_ptr = v_cache + kv_head * max_seq_len * head_dim;

    std::vector<float> scores(seq_len, 0.0f);
    float max_score = -INFINITY;
    for (uint32_t t = 0; t < seq_len; ++t) {
      const float *k_t = k_ptr + t * head_dim;
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_dim; ++d) {
        float prod = round_fp16_host(q_ptr[d] * k_t[d]);
        dot = round_fp16_host(dot + prod);
      }
      float s = round_fp16_host(dot * scale);
      scores[t] = s;
      if (s > max_score)
        max_score = s;
    }
    double sum = 0.0;
    for (uint32_t t = 0; t < seq_len; ++t) {
      float e = std::exp(scores[t] - max_score);
      scores[t] = round_fp16_host(e);
      sum += scores[t];
    }
    const double inv_sum = sum > 0.0 ? (1.0 / sum) : 0.0;

    float *o_ptr = out.data() + h * head_dim;
    for (uint32_t d = 0; d < head_dim; ++d) {
      float acc = 0.0f;
      for (uint32_t t = 0; t < seq_len; ++t) {
        const float *v_t = v_ptr + t * head_dim;
        float term = round_fp16_host(scores[t] * v_t[d]);
        acc = round_fp16_host(acc + term);
      }
      o_ptr[d] = round_fp16_host(acc * static_cast<float>(inv_sum));
    }
  }
}

static void compute_attention_ref_fp64(const float *q, const float *k_cache,
                                       const float *v_cache, uint32_t num_heads,
                                       uint32_t num_heads_kv, uint32_t head_dim,
                                       uint32_t seq_len, uint32_t max_seq_len,
                                       double scale, std::vector<float> &out) {
  if (!q || !k_cache || !v_cache || num_heads == 0 || head_dim == 0)
    return;
  out.assign(static_cast<size_t>(num_heads) * head_dim, 0.0f);
  const uint32_t group = (num_heads_kv > 0) ? (num_heads / num_heads_kv) : 0;
  for (uint32_t h = 0; h < num_heads; ++h) {
    const uint32_t kv_head = (group > 0) ? (h / group) : 0;
    const float *q_ptr = q + h * head_dim;
    const float *k_ptr = k_cache + kv_head * max_seq_len * head_dim;
    const float *v_ptr = v_cache + kv_head * max_seq_len * head_dim;

    std::vector<double> scores(seq_len, 0.0);
    double max_score = -INFINITY;
    for (uint32_t t = 0; t < seq_len; ++t) {
      const float *k_t = k_ptr + t * head_dim;
      double dot = 0.0;
      for (uint32_t d = 0; d < head_dim; ++d) {
        dot += static_cast<double>(q_ptr[d]) * static_cast<double>(k_t[d]);
      }
      double s = dot * scale;
      scores[t] = s;
      if (s > max_score)
        max_score = s;
    }
    double sum = 0.0;
    for (uint32_t t = 0; t < seq_len; ++t) {
      scores[t] = std::exp(scores[t] - max_score);
      sum += scores[t];
    }
    const double inv_sum = sum > 0.0 ? (1.0 / sum) : 0.0;

    float *o_ptr = out.data() + h * head_dim;
    for (uint32_t d = 0; d < head_dim; ++d) {
      double acc = 0.0;
      for (uint32_t t = 0; t < seq_len; ++t) {
        const float *v_t = v_ptr + t * head_dim;
        acc += scores[t] * static_cast<double>(v_t[d]);
      }
      o_ptr[d] = static_cast<float>(acc * inv_sum);
    }
  }
}

static std::vector<double> per_head_mae_f32(const float *a, const float *b,
                                            uint32_t num_heads,
                                            uint32_t head_dim) {
  std::vector<double> out;
  if (!a || !b || num_heads == 0 || head_dim == 0)
    return out;
  out.resize(num_heads, 0.0);
  const size_t head_size = head_dim;
  for (uint32_t h = 0; h < num_heads; ++h) {
    double sum = 0.0;
    const float *pa = a + h * head_size;
    const float *pb = b + h * head_size;
    for (uint32_t d = 0; d < head_dim; ++d) {
      sum += std::abs(static_cast<double>(pa[d]) - static_cast<double>(pb[d]));
    }
    out[h] = sum / static_cast<double>(head_dim);
  }
  return out;
}

static void stats_hash_kv_subset(const float *base, uint32_t num_heads_kv,
                                 uint32_t head_dim, uint32_t max_seq_len,
                                 uint32_t seq_len, F32Stats *stats,
                                 uint64_t *hash) {
  if (!base || !stats || !hash || num_heads_kv == 0 || head_dim == 0 ||
      seq_len == 0) {
    return;
  }
  stats->min = std::numeric_limits<float>::infinity();
  stats->max = -std::numeric_limits<float>::infinity();
  stats->mean = 0.0f;
  stats->nan = 0;
  stats->inf = 0;
  double sum = 0.0;
  size_t count = 0;
  uint64_t h = 1469598103934665603ull;
  size_t hash_count = 0;
  for (uint32_t kv = 0; kv < num_heads_kv; ++kv) {
    const float *head_base = base + kv * max_seq_len * head_dim;
    for (uint32_t t = 0; t < seq_len; ++t) {
      const float *p = head_base + t * head_dim;
      for (uint32_t d = 0; d < head_dim; ++d) {
        float v = p[d];
        if (std::isnan(v)) {
          stats->nan++;
          continue;
        }
        if (std::isinf(v)) {
          stats->inf++;
          continue;
        }
        if (v < stats->min)
          stats->min = v;
        if (v > stats->max)
          stats->max = v;
        sum += v;
        count++;
        if (hash_count < 256) {
          uint32_t vv;
          std::memcpy(&vv, &v, sizeof(uint32_t));
          h ^= static_cast<uint64_t>(vv);
          h *= 1099511628211ull;
          hash_count++;
        }
      }
    }
  }
  stats->mean = count > 0 ? static_cast<float>(sum / count) : 0.0f;
  *hash = h;
}

static F32Stats stats_f32(const float *p, size_t n) {
  F32Stats s{};
  if (n == 0 || !p)
    return s;
  s.min = p[0];
  s.max = p[0];
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    float v = p[i];
    if (std::isnan(v)) {
      s.nan++;
      continue;
    }
    if (std::isinf(v)) {
      s.inf++;
      continue;
    }
    if (v < s.min)
      s.min = v;
    if (v > s.max)
      s.max = v;
    sum += v;
  }
  s.mean = (n > 0) ? static_cast<float>(sum / n) : 0.0f;
  return s;
}

static uint64_t hash_f32(const float *p, size_t n) {
  const size_t count = (n < 256) ? n : 256;
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < count; ++i) {
    uint32_t v;
    std::memcpy(&v, &p[i], sizeof(uint32_t));
    h ^= static_cast<uint64_t>(v);
    h *= 1099511628211ull;
  }
  return h;
}

static void append_line(const char *path, const std::string &line) {
  if (!path || !*path)
    return;
  std::ofstream f(path, std::ios::out | std::ios::app);
  if (!f.is_open())
    return;
  f << line << "\n";
}

static void post_wo_trace_tensor(const char *point, const char *phase,
                                 const char *prompt_id, size_t layer,
                                 uint32_t step, uint32_t pos_id,
                                 uint32_t seq_len, uint32_t tokens_total,
                                 const float *base, size_t stride_elems,
                                 size_t token_index, size_t alloc_bytes,
                                 hipStream_t stream) {
  if (!trace_post_wo_enabled())
    return;
  const char *out = post_wo_out_path();
  if (!out || !*out || !point || !phase)
    return;
  if (!post_wo_phase_enabled(phase))
    return;
  if (!base || stride_elems == 0)
    return;

  const size_t offset_elems = token_index * stride_elems;
  const float *ptr = base + offset_elems;
  const uint32_t sample_n =
      std::min<uint32_t>(post_wo_sample(), static_cast<uint32_t>(stride_elems));
  std::vector<float> host(sample_n, 0.0f);
  if (sample_n > 0) {
    hipMemcpyAsync(host.data(), ptr, sample_n * sizeof(float),
                   hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
  }
  const F32Stats stats = stats_f32(host.data(), host.size());
  const uint64_t hash = hash_f32(host.data(), host.size());

  std::ostringstream oss;
  oss << "{\"event\":\"post_wo_trace\"";
  if (prompt_id && *prompt_id)
    oss << ",\"prompt_id\":\"" << prompt_id << "\"";
  oss << ",\"phase\":\"" << phase << "\""
      << ",\"point\":\"" << point << "\""
      << ",\"layer\":" << layer << ",\"step\":" << step
      << ",\"pos_id\":" << pos_id << ",\"seq_len\":" << seq_len
      << ",\"tokens_total\":" << tokens_total
      << ",\"token_index\":" << token_index
      << ",\"ptr\":" << reinterpret_cast<uintptr_t>(ptr)
      << ",\"base_ptr\":" << reinterpret_cast<uintptr_t>(base)
      << ",\"offset_bytes\":" << (offset_elems * sizeof(float))
      << ",\"alloc_bytes\":" << alloc_bytes << ",\"sample_n\":" << sample_n
      << ",\"hash\":" << hash << ",\"min\":" << stats.min
      << ",\"max\":" << stats.max << ",\"mean\":" << stats.mean
      << ",\"nan\":" << stats.nan << ",\"inf\":" << stats.inf
      << ",\"sample\":[";
  for (size_t i = 0; i < host.size(); ++i) {
    if (i)
      oss << ",";
    oss << host[i];
  }
  oss << "]}";
  append_line(out, oss.str());
}

static void trace_rmsnorm(
    const char *phase, const char *prompt_id, size_t layer, size_t num_layers,
    uint32_t step, uint32_t pos_id, uint32_t seq_len, uint32_t tokens_total,
    const float *input, const float *output, size_t stride_elems,
    size_t token_index, size_t input_alloc_bytes, size_t output_alloc_bytes,
    const gcore::rt::hip::Buffer &weight, float eps, hipStream_t stream) {
  if (!trace_rmsnorm_enabled())
    return;
  const char *out = rmsnorm_out_path();
  if (!out || !*out || !phase)
    return;
  if (!rmsnorm_phase_enabled(phase))
    return;
  if (!rmsnorm_layer_selected(layer, num_layers))
    return;
  if (!input || !output || stride_elems == 0)
    return;

  const size_t offset_elems = token_index * stride_elems;
  const float *in_ptr = input + offset_elems;
  const float *out_ptr = output + offset_elems;
  const uint32_t sample_n =
      std::min<uint32_t>(rmsnorm_sample(), static_cast<uint32_t>(stride_elems));
  std::vector<float> in_host(sample_n, 0.0f);
  std::vector<float> out_host(sample_n, 0.0f);
  std::vector<float> w_host(sample_n, 0.0f);
  if (sample_n > 0) {
    hipMemcpyAsync(in_host.data(), in_ptr, sample_n * sizeof(float),
                   hipMemcpyDeviceToHost, stream);
    hipMemcpyAsync(out_host.data(), out_ptr, sample_n * sizeof(float),
                   hipMemcpyDeviceToHost, stream);
    if (weight.data()) {
      hipMemcpyAsync(w_host.data(), weight.data(), sample_n * sizeof(float),
                     hipMemcpyDeviceToHost, stream);
    }
    hipStreamSynchronize(stream);
  }

  const F32Stats instats = stats_f32(in_host.data(), in_host.size());
  const F32Stats outstats = stats_f32(out_host.data(), out_host.size());
  const F32Stats wstats = stats_f32(w_host.data(), w_host.size());
  const uint64_t inhash = hash_f32(in_host.data(), in_host.size());
  const uint64_t outhash = hash_f32(out_host.data(), out_host.size());
  const uint64_t whash = hash_f32(w_host.data(), w_host.size());

  double sumsq = 0.0;
  for (float v : in_host) {
    sumsq += static_cast<double>(v) * static_cast<double>(v);
  }
  double mean_sq = (in_host.empty()) ? 0.0 : (sumsq / in_host.size());
  double inv_rms =
      (mean_sq + eps) > 0.0 ? (1.0 / std::sqrt(mean_sq + eps)) : 0.0;

  std::ostringstream oss;
  oss << "{\"event\":\"rmsnorm_trace\"";
  if (prompt_id && *prompt_id)
    oss << ",\"prompt_id\":\"" << prompt_id << "\"";
  oss << ",\"phase\":\"" << phase << "\""
      << ",\"layer\":" << layer << ",\"step\":" << step
      << ",\"pos_id\":" << pos_id << ",\"seq_len\":" << seq_len
      << ",\"tokens_total\":" << tokens_total
      << ",\"token_index\":" << token_index << ",\"eps\":" << eps
      << ",\"sumsq\":" << mean_sq << ",\"inv_rms\":" << inv_rms
      << ",\"input_hash\":" << inhash << ",\"input_min\":" << instats.min
      << ",\"input_max\":" << instats.max << ",\"input_mean\":" << instats.mean
      << ",\"input_nan\":" << instats.nan << ",\"input_inf\":" << instats.inf
      << ",\"output_hash\":" << outhash << ",\"output_min\":" << outstats.min
      << ",\"output_max\":" << outstats.max
      << ",\"output_mean\":" << outstats.mean
      << ",\"output_nan\":" << outstats.nan
      << ",\"output_inf\":" << outstats.inf << ",\"weight_hash\":" << whash
      << ",\"weight_min\":" << wstats.min << ",\"weight_max\":" << wstats.max
      << ",\"weight_mean\":" << wstats.mean
      << ",\"input_ptr\":" << reinterpret_cast<uintptr_t>(in_ptr)
      << ",\"output_ptr\":" << reinterpret_cast<uintptr_t>(out_ptr)
      << ",\"weight_ptr\":" << reinterpret_cast<uintptr_t>(weight.data())
      << ",\"input_offset_bytes\":" << (offset_elems * sizeof(float))
      << ",\"output_offset_bytes\":" << (offset_elems * sizeof(float))
      << ",\"input_alloc_bytes\":" << input_alloc_bytes
      << ",\"output_alloc_bytes\":" << output_alloc_bytes
      << ",\"weight_bytes\":" << weight.size()
      << ",\"stride_bytes\":" << (stride_elems * sizeof(float))
      << ",\"input_dtype\":\"" << dtype_label(gcore::rt::GretaDataType::FP32)
      << "\""
      << ",\"output_dtype\":\"" << dtype_label(gcore::rt::GretaDataType::FP32)
      << "\""
      << ",\"weight_dtype\":\"" << dtype_label(weight.data_type()) << "\""
      << ",\"kernel\":\"rmsnorm_naive\""
      << ",\"sample_n\":" << sample_n << ",\"input_sample\":[";
  for (size_t i = 0; i < in_host.size(); ++i) {
    if (i)
      oss << ",";
    oss << in_host[i];
  }
  oss << "],\"output_sample\":[";
  for (size_t i = 0; i < out_host.size(); ++i) {
    if (i)
      oss << ",";
    oss << out_host[i];
  }
  oss << "],\"weight_sample\":[";
  for (size_t i = 0; i < w_host.size(); ++i) {
    if (i)
      oss << ",";
    oss << w_host[i];
  }
  oss << "]}";
  append_line(out, oss.str());
}

static const char *layer_delta_out_path() {
  const char *out = std::getenv("GRETA_TRACE_LAYER_DELTA_OUT");
  if (out && *out)
    return out;
  return std::getenv("GRETA_TRACE_PREFILL_DECODE_OUT");
}

class GretaMemoryView final : public gcore::rt::GretaMemory {
public:
  GretaMemoryView(gcore::rt::GretaMemory *base, size_t offset_bytes)
      : base_(base), offset_bytes_(offset_bytes) {}

  void *data() override {
    return static_cast<char *>(base_->data()) + offset_bytes_;
  }
  const void *data() const override {
    return static_cast<const char *>(base_->data()) + offset_bytes_;
  }
  size_t size() const override {
    size_t base_size = base_->size();
    return offset_bytes_ <= base_size ? (base_size - offset_bytes_) : 0;
  }

  gcore::rt::GretaDataType data_type() const override {
    return base_->data_type();
  }
  gcore::rt::GretaQuantInfo quant_info() const override {
    return base_->quant_info();
  }

  bool copy_from_host(const void *src, size_t size) override {
    hipError_t res = hipMemcpy(data(), src, size, hipMemcpyHostToDevice);
    return res == hipSuccess;
  }
  bool copy_to_host(void *dst, size_t size) const override {
    hipError_t res = hipMemcpy(dst, data(), size, hipMemcpyDeviceToHost);
    return res == hipSuccess;
  }

private:
  gcore::rt::GretaMemory *base_;
  size_t offset_bytes_;
};

static bool trace_kernel_sync_enabled() {
  static const bool enabled = env_flag("GRETA_TRACE_READOUT") ||
                              env_flag("GRETA_TRACE_PREFILL_DECODE") ||
                              env_flag("GRETA_TRACE_LANDSCAPE");
  return enabled;
}

static bool trace_hip_sync(const char *name, std::string *err) {
  if (!trace_kernel_sync_enabled())
    return true;
  hipError_t res = hipDeviceSynchronize();
  if (res != hipSuccess) {
    if (err)
      *err = std::string(name) +
             " hipDeviceSynchronize failed: " + hipGetErrorString(res);
    return false;
  }
  return true;
}

static bool trace_hip_check_and_sync(const char *name, std::string *err) {
  if (!trace_kernel_sync_enabled())
    return true;
  hipError_t err_code = hipGetLastError();
  if (err_code != hipSuccess) {
    if (err)
      *err = std::string(name) +
             " hipGetLastError: " + hipGetErrorString(err_code);
    return false;
  }
  return trace_hip_sync(name, err);
}

static bool trace_embed_verify_enabled() {
  static const bool enabled = env_flag("GRETA_TRACE_EMBED_VERIFY");
  return enabled;
}

static bool embed_layout_row_major() {
  static bool cached = false;
  static bool row_major = true;
  if (cached)
    return row_major;
  cached = true;
  const char *v = std::getenv("GRETA_EMBED_LAYOUT");
  if (!v)
    return row_major;
  std::string val(v);
  if (val == "col" || val == "COL" || val == "col_major") {
    row_major = false;
  } else {
    row_major = true;
  }
  return row_major;
}

static bool trace_embed_verify_once(const int32_t *tokens, size_t seq_len,
                                    uint32_t dim, uint32_t vocab_size,
                                    const gcore::rt::hip::Buffer &token_embd,
                                    const gcore::rt::hip::Buffer &x,
                                    bool layout_row_major, std::string *err) {
  static bool done = false;
  if (!trace_embed_verify_enabled() || done)
    return true;
  done = true;

  if (seq_len == 0 || dim == 0 || vocab_size == 0)
    return true;

  size_t s = seq_len - 1;
  int32_t token = tokens[s];
  if (token < 0 || token >= static_cast<int32_t>(vocab_size)) {
    std::cout << "[GRETA_TRACE_EMBED_VERIFY] token out of range: " << token
              << std::endl;
    return true;
  }

  if (!trace_hip_sync("Embedding Verify", err))
    return false;

  std::vector<float> out(dim);
  size_t out_offset = s * static_cast<size_t>(dim) * sizeof(float);
  if (!x.copy_to_host_offset(out.data(), out_offset,
                             static_cast<size_t>(dim) * sizeof(float), err))
    return false;

  std::vector<float> row(dim);
  size_t row_offset = static_cast<size_t>(token) * dim * sizeof(float);
  if (!token_embd.copy_to_host_offset(row.data(), row_offset,
                                      static_cast<size_t>(dim) * sizeof(float),
                                      err))
    return false;

  std::vector<float> col(dim);
  for (uint32_t d = 0; d < dim; ++d) {
    size_t col_offset =
        (static_cast<size_t>(d) * vocab_size + static_cast<size_t>(token)) *
        sizeof(float);
    if (!token_embd.copy_to_host_offset(&col[d], col_offset, sizeof(float),
                                        err))
      return false;
  }

  double mae_row = 0.0;
  double mae_col = 0.0;
  double max_row = 0.0;
  double max_col = 0.0;
  for (uint32_t d = 0; d < dim; ++d) {
    double dr = std::fabs(static_cast<double>(out[d]) - row[d]);
    double dc = std::fabs(static_cast<double>(out[d]) - col[d]);
    mae_row += dr;
    mae_col += dc;
    if (dr > max_row)
      max_row = dr;
    if (dc > max_col)
      max_col = dc;
  }
  mae_row /= static_cast<double>(dim);
  mae_col /= static_cast<double>(dim);

  const char *layout_probe_best =
      (mae_row <= mae_col) ? "row_major_match" : "col_major_match";
  const char *layout_used = layout_row_major ? "row" : "col";
  std::cout << "[GRETA_TRACE_EMBED_VERIFY] token=" << token << " seq_idx=" << s
            << " mae_row=" << mae_row << " mae_col=" << mae_col
            << " max_row=" << max_row << " max_col=" << max_col
            << " layout_used=" << layout_used
            << " layout_probe_best=" << layout_probe_best << std::endl;

  return true;
}

BlockScheduler::BlockScheduler() = default;

BlockScheduler::~BlockScheduler() {
  if (stream_ != nullptr) {
    delete stream_;
    stream_ = nullptr;
  }
}

bool BlockScheduler::init(const ModelConfig &config, std::string *err) {
  config_ = config;
  blocks_.resize(config_.num_layers);

  std::cout << "[GRETA_SCHED] Creating stream..." << std::endl;
  stream_ = gcore::rt::GretaContext::instance().create_stream();
  if (!stream_) {
    *err = "Failed to create GRETA stream";
    return false;
  }
  std::cout << "[GRETA_SCHED] Stream created successfully" << std::endl;

  tracer_.init_from_env();
  layer_tracer_.init_from_env(config_);

  initialized_ = true;
  return true;
}

bool BlockScheduler::allocate_weights(std::string *err) {
  if (!initialized_) {
    *err = "BlockScheduler not initialized";
    return false;
  }

  using Usage = gcore::rt::hip::BufferUsage;
  const size_t D = config_.dim;
  const size_t H = config_.hidden_dim;

  const char *use_int8 = std::getenv("GRETA_INT8_WEIGHTS");
  bool int8_mode = (use_int8 && std::string(use_int8) == "1");

  for (size_t i = 0; i < config_.num_layers; ++i) {
    auto &b = blocks_[i];
    if (int8_mode) {
      b.wq.allocate(D * D, Usage::DeviceOnly, gcore::rt::GretaDataType::INT8,
                    err);
      b.wk.allocate(D * D, Usage::DeviceOnly, gcore::rt::GretaDataType::INT8,
                    err);
      b.wv.allocate(D * D, Usage::DeviceOnly, gcore::rt::GretaDataType::INT8,
                    err);
      b.wo.allocate(D * D, Usage::DeviceOnly, gcore::rt::GretaDataType::INT8,
                    err);
      b.w1.allocate(D * H, Usage::DeviceOnly, gcore::rt::GretaDataType::INT8,
                    err);
      b.w2.allocate(H * D, Usage::DeviceOnly, gcore::rt::GretaDataType::INT8,
                    err);
      b.w3.allocate(D * H, Usage::DeviceOnly, gcore::rt::GretaDataType::INT8,
                    err);

      size_t nb = (D * D + 31) / 32;
      b.s_wq.allocate(nb * 4, Usage::DeviceOnly, gcore::rt::GretaDataType::FP32,
                      err);
      b.s_wk.allocate(nb * 4, Usage::DeviceOnly, gcore::rt::GretaDataType::FP32,
                      err);
      b.s_wv.allocate(nb * 4, Usage::DeviceOnly, gcore::rt::GretaDataType::FP32,
                      err);
      b.s_wo.allocate(nb * 4, Usage::DeviceOnly, gcore::rt::GretaDataType::FP32,
                      err);

      size_t nbh = (D * H + 31) / 32;
      b.s_w1.allocate(nbh * 4, Usage::DeviceOnly,
                      gcore::rt::GretaDataType::FP32, err);
      b.s_w2.allocate(nbh * 4, Usage::DeviceOnly,
                      gcore::rt::GretaDataType::FP32, err);
      b.s_w3.allocate(nbh * 4, Usage::DeviceOnly,
                      gcore::rt::GretaDataType::FP32, err);

      // Per-head scales for Attention (QKV)
      uint32_t num_heads = config_.num_heads;
      b.sh_wq.allocate(num_heads * 4, Usage::DeviceOnly,
                       gcore::rt::GretaDataType::FP32, err);
      b.sh_wk.allocate(num_heads * 4, Usage::DeviceOnly,
                       gcore::rt::GretaDataType::FP32, err);
      b.sh_wv.allocate(num_heads * 4, Usage::DeviceOnly,
                       gcore::rt::GretaDataType::FP32, err);
      b.sh_wo.allocate(num_heads * 4, Usage::DeviceOnly,
                       gcore::rt::GretaDataType::FP32, err);
    } else {
      b.wq.allocate(D * D * 2, Usage::DeviceOnly,
                    gcore::rt::GretaDataType::FP16, err);
      b.wk.allocate(D * D * 2, Usage::DeviceOnly,
                    gcore::rt::GretaDataType::FP16, err);
      b.wv.allocate(D * D * 2, Usage::DeviceOnly,
                    gcore::rt::GretaDataType::FP16, err);
      b.wo.allocate(D * D * 2, Usage::DeviceOnly,
                    gcore::rt::GretaDataType::FP16, err);
      b.w1.allocate(D * H * 2, Usage::DeviceOnly,
                    gcore::rt::GretaDataType::FP16, err);
      b.w2.allocate(H * D * 2, Usage::DeviceOnly,
                    gcore::rt::GretaDataType::FP16, err);
      b.w3.allocate(D * H * 2, Usage::DeviceOnly,
                    gcore::rt::GretaDataType::FP16, err);
    }
    b.attn_norm.allocate(D * 4, Usage::DeviceOnly,
                         gcore::rt::GretaDataType::FP32, err);
    b.ffn_norm.allocate(D * 4, Usage::DeviceOnly,
                        gcore::rt::GretaDataType::FP32, err);
  }

  token_embd_.allocate(config_.vocab_size * D * 4, Usage::DeviceOnly,
                       gcore::rt::GretaDataType::FP32, err);
  output_norm_.allocate(D * 4, Usage::DeviceOnly,
                        gcore::rt::GretaDataType::FP32, err);
  output_weight_.allocate(config_.vocab_size * D * 2, Usage::DeviceOnly,
                          gcore::rt::GretaDataType::FP16, err);

  return true;
}

bool BlockScheduler::allocate_activations(size_t batch_size, size_t max_seq_len,
                                          std::string *err) {
  if (!initialized_) {
    *err = "BlockScheduler not initialized";
    return false;
  }

  config_.max_seq_len = max_seq_len;

  using Usage = gcore::rt::hip::BufferUsage;
  const size_t D = config_.dim;
  const size_t H = config_.hidden_dim;
  const size_t L = config_.num_layers;
  const size_t heads_q = config_.num_heads;
  const size_t heads_kv =
      config_.num_heads_kv > 0 ? config_.num_heads_kv : config_.num_heads;
  const size_t head_dim = config_.head_dim;
  const size_t kv_dim = heads_kv * head_dim;

  size_t hidden_size = batch_size * max_seq_len * D * sizeof(float);
  activations_.x.allocate(hidden_size, Usage::DeviceOnly,
                          gcore::rt::GretaDataType::FP32, err);
  activations_.norm_out.allocate(hidden_size, Usage::DeviceOnly,
                                 gcore::rt::GretaDataType::FP32, err);
  activations_.q.allocate(hidden_size, Usage::DeviceOnly,
                          gcore::rt::GretaDataType::FP32, err);
  const size_t kv_hidden_size =
      batch_size * max_seq_len * kv_dim * sizeof(float);
  activations_.k.allocate(kv_hidden_size, Usage::DeviceOnly,
                          gcore::rt::GretaDataType::FP32, err);
  activations_.v.allocate(kv_hidden_size, Usage::DeviceOnly,
                          gcore::rt::GretaDataType::FP32, err);
  activations_.attn_out.allocate(hidden_size, Usage::DeviceOnly,
                                 gcore::rt::GretaDataType::FP32, err);

  size_t mlp_size = batch_size * max_seq_len * H * sizeof(float);
  activations_.mlp_gate.allocate(mlp_size, Usage::DeviceOnly,
                                 gcore::rt::GretaDataType::FP32, err);
  activations_.mlp_up.allocate(mlp_size, Usage::DeviceOnly,
                               gcore::rt::GretaDataType::FP32, err);
  activations_.mlp_out.allocate(hidden_size, Usage::DeviceOnly,
                                gcore::rt::GretaDataType::FP32, err);

  size_t kv_size = L * max_seq_len * heads_kv * head_dim * sizeof(float);
  activations_.kv_cache_k.allocate(kv_size, Usage::DeviceOnly,
                                   gcore::rt::GretaDataType::FP32, err);
  activations_.kv_cache_v.allocate(kv_size, Usage::DeviceOnly,
                                   gcore::rt::GretaDataType::FP32, err);

  size_t tokens_size = batch_size * max_seq_len * sizeof(int32_t);
  activations_.tokens.allocate(tokens_size, Usage::DeviceOnly,
                               gcore::rt::GretaDataType::FP16, err);

  activations_.d_pos.allocate(sizeof(uint32_t), Usage::DeviceOnly,
                              gcore::rt::GretaDataType::FP16, err);

  size_t logits_size =
      batch_size * max_seq_len * config_.vocab_size * sizeof(float);
  logits_.allocate(logits_size, Usage::DeviceOnly,
                   gcore::rt::GretaDataType::FP32, err);

  return true;
}

bool BlockScheduler::load_weights(WeightLoader &loader, std::string *err) {
  const char *use_int8 = std::getenv("GRETA_INT8_WEIGHTS");
  const char *use_int4 = std::getenv("GRETA_INT4_WEIGHTS");
  bool int8_mode = (use_int8 && std::string(use_int8) == "1");
  bool int4_mode = (use_int4 && std::string(use_int4) == "1");

  std::cout << "[GRETA_SCHED] Starting weight load (INT8: "
            << (int8_mode ? "ON" : "OFF")
            << ", INT4: " << (int4_mode ? "ON" : "OFF") << ")" << std::endl;

  for (size_t i = 0; i < config_.num_layers; ++i) {
    if (i % 8 == 0)
      std::cout << "[GRETA_SCHED] Loading layer " << i << "/"
                << config_.num_layers << "..." << std::endl;
    std::string prefix = "blk." + std::to_string(i) + ".";
    auto &b = blocks_[i];
    if (!loader.load_tensor(prefix + "attn_norm.weight", b.attn_norm, err))
      return false;
    if (!loader.load_tensor(prefix + "ffn_norm.weight", b.ffn_norm, err))
      return false;

    if (int4_mode) {
      if (!loader.load_tensor_int4(prefix + "attn_q.weight", b.wq, b.s_wq,
                                   b.sh_wq, err))
        return false;
      if (!loader.load_tensor_int4(prefix + "attn_k.weight", b.wk, b.s_wk,
                                   b.sh_wk, err))
        return false;
      if (!loader.load_tensor_int4(prefix + "attn_v.weight", b.wv, b.s_wv,
                                   b.sh_wv, err))
        return false;
      if (!loader.load_tensor_int4(prefix + "attn_output.weight", b.wo, b.s_wo,
                                   b.sh_wo, err))
        return false;
      if (!loader.load_tensor_int4(prefix + "ffn_gate.weight", b.w1, b.s_w1,
                                   b.sh_wo,
                                   err)) // sh_wo reused for FFN (dummy)
        return false;
      if (!loader.load_tensor_int4(prefix + "ffn_down.weight", b.w2, b.s_w2,
                                   b.sh_wo, err))
        return false;
      if (!loader.load_tensor_int4(prefix + "ffn_up.weight", b.w3, b.s_w3,
                                   b.sh_wo, err))
        return false;
    } else if (int8_mode) {
      if (!loader.load_tensor_int8(prefix + "attn_q.weight", b.wq, b.s_wq, err))
        return false;
      if (!loader.load_tensor_int8(prefix + "attn_k.weight", b.wk, b.s_wk, err))
        return false;
      if (!loader.load_tensor_int8(prefix + "attn_v.weight", b.wv, b.s_wv, err))
        return false;
      if (!loader.load_tensor_int8(prefix + "attn_output.weight", b.wo, b.s_wo,
                                   err))
        return false;
      if (!loader.load_tensor_int8(prefix + "ffn_gate.weight", b.w1, b.s_w1,
                                   err))
        return false;
      if (!loader.load_tensor_int8(prefix + "ffn_down.weight", b.w2, b.s_w2,
                                   err))
        return false;
      if (!loader.load_tensor_int8(prefix + "ffn_up.weight", b.w3, b.s_w3, err))
        return false;
    } else {
      if (!loader.load_tensor_fp16(prefix + "attn_q.weight", b.wq, err))
        return false;
      if (!loader.load_tensor_fp16(prefix + "attn_k.weight", b.wk, err))
        return false;
      if (!loader.load_tensor_fp16(prefix + "attn_v.weight", b.wv, err))
        return false;
      if (!loader.load_tensor_fp16(prefix + "attn_output.weight", b.wo, err))
        return false;
      if (!loader.load_tensor_fp16(prefix + "ffn_gate.weight", b.w1, err))
        return false;
      if (!loader.load_tensor_fp16(prefix + "ffn_down.weight", b.w2, err))
        return false;
      if (!loader.load_tensor_fp16(prefix + "ffn_up.weight", b.w3, err))
        return false;
    }
  }

  if (!loader.load_tensor("token_embd.weight", token_embd_, err))
    return false;
  if (!loader.load_tensor("output_norm.weight", output_norm_, err))
    return false;
  if (!loader.load_tensor_fp16("output.weight", output_weight_, err))
    return false;
  return true;
}

#define CHECK_HIP_KERNEL(cmd, name)                                            \
  do {                                                                         \
    cmd;                                                                       \
    hipError_t err_code = hipGetLastError();                                   \
    if (err_code != hipSuccess) {                                              \
      if (err)                                                                 \
        *err = std::string(name) +                                             \
               " launch failed: " + hipGetErrorString(err_code);               \
      return false;                                                            \
    }                                                                          \
    if (!trace_hip_sync(name, err))                                            \
      return false;                                                            \
  } while (0)

#define CHECK_GRETA(cmd, name)                                                 \
  do {                                                                         \
    if ((cmd) != gcore::rt::GretaResult::SUCCESS) {                            \
      if (err)                                                                 \
        *err = std::string(name) + " failed";                                  \
      return false;                                                            \
    }                                                                          \
    if (!trace_hip_check_and_sync(name, err))                                  \
      return false;                                                            \
  } while (0)

bool BlockScheduler::execute_layer(size_t layer_idx, size_t seq_start,
                                   size_t seq_len, const int32_t *tokens,
                                   std::string *err) {
  auto &b = blocks_[layer_idx];
  uint32_t D = static_cast<uint32_t>(config_.dim);
  uint32_t Hq = static_cast<uint32_t>(config_.num_heads);
  uint32_t Hkv = static_cast<uint32_t>(
      config_.num_heads_kv > 0 ? config_.num_heads_kv : config_.num_heads);
  uint32_t Dh = D / Hq;
  uint32_t hidden_dim = static_cast<uint32_t>(config_.hidden_dim);
  uint32_t S = static_cast<uint32_t>(seq_len);
  const bool is_decode_step = (S == 1 && seq_start > 0);

  const uint32_t n_x = S * D;
  const uint32_t n_mlp = S * hidden_dim;
  const uint32_t n_kv = S * Hkv * Dh;
  const uint32_t kv_dim = Hkv * Dh;

  using namespace gcore::rt::hip::kernels;
  float *x = static_cast<float *>(activations_.x.data());
  float *norm_out = static_cast<float *>(activations_.norm_out.data());
  float *q = static_cast<float *>(activations_.q.data());
  float *k = static_cast<float *>(activations_.k.data());
  float *v = static_cast<float *>(activations_.v.data());
  float *attn_out = static_cast<float *>(activations_.attn_out.data());
  float *mlp_gate = static_cast<float *>(activations_.mlp_gate.data());
  float *mlp_up = static_cast<float *>(activations_.mlp_up.data());
  float *mlp_out = static_cast<float *>(activations_.mlp_out.data());

  const float *attn_norm = static_cast<const float *>(b.attn_norm.data());
  const float *ffn_norm = static_cast<const float *>(b.ffn_norm.data());

  size_t offset = (size_t)layer_idx * (size_t)config_.max_seq_len *
                  (size_t)Hkv * (size_t)Dh;
  float *cache_k =
      static_cast<float *>(activations_.kv_cache_k.data()) + offset;
  float *cache_v =
      static_cast<float *>(activations_.kv_cache_v.data()) + offset;
  const uint32_t *d_pos =
      static_cast<const uint32_t *>(activations_.d_pos.data());

  hipStream_t hip_stream =
      static_cast<gcore::rt::hip::GretaStreamHip *>(stream_)->handle();

  const bool trace_layer = GRETA_UNLIKELY(layer_tracer_.enabled());
  const bool stage_enabled = stage_trace_enabled();
  const bool post_wo_enabled = trace_post_wo_enabled();
  const bool rmsnorm_enabled = trace_rmsnorm_enabled();
  const bool debug_input = stage_trace_debug_input();
  const char *stage_phase = nullptr;
  if (stage_enabled || post_wo_enabled || rmsnorm_enabled) {
    if (seq_len > 1 && trace_step_ == 0) {
      stage_phase = "prefill_last";
    } else if (seq_len == 1 && trace_step_ == 1) {
      stage_phase = "decode0";
    }
  }
  const bool stage_layer =
      stage_enabled && stage_phase &&
      stage_trace_layer_selected(layer_idx, config_.num_layers) &&
      stage_trace_phase_enabled(stage_phase);
  const bool post_wo_layer =
      trace_post_wo_enabled() && stage_phase &&
      post_wo_layer_selected(layer_idx, config_.num_layers) &&
      post_wo_phase_enabled(stage_phase) && post_wo_out_path();
  const uint32_t stage_tokens_total =
      static_cast<uint32_t>(seq_start + seq_len);
  const uint32_t stage_token_index =
      seq_len > 0 ? static_cast<uint32_t>(seq_len - 1) : 0;
  const uint32_t stage_pos_id =
      static_cast<uint32_t>(seq_start + stage_token_index);
  const char *stage_prompt_id = std::getenv("GRETA_TRACE_PROMPT_ID");

  if (stage_layer) {
    const size_t x_stride_elems = D;
    const size_t x_offset_bytes =
        static_cast<size_t>(stage_token_index) * x_stride_elems * sizeof(float);
    const uint32_t prompt_tokens = (seq_start == 0)
                                       ? static_cast<uint32_t>(seq_len)
                                       : static_cast<uint32_t>(seq_start);
    const char *src_kind =
        (seq_len > 1) ? "prefill_hidden_buffer" : "decode0_hidden_buffer";
    const char *route_label =
        (seq_len > 1) ? "EMBED_LOOKUP_PREFILL" : "EMBED_LOOKUP_DECODE";
    if (debug_input && stage_phase &&
        std::strcmp(stage_phase, "decode0") == 0) {
      src_kind = "decode0_input_override";
      route_label = "DEBUG_INPUT_INJECT";
    }

    StageInputMeta input_meta{};
    input_meta.src_kind = src_kind;
    input_meta.token_index_used = stage_token_index;
    input_meta.offset_bytes = x_offset_bytes;
    input_meta.ptr = reinterpret_cast<uintptr_t>(x);
    input_meta.alloc_bytes = activations_.x.size();
    input_meta.prompt_tokens = prompt_tokens;
    input_meta.kv_pos = stage_pos_id;
    input_meta.decode_step = static_cast<uint32_t>(trace_step_);
    input_meta.token_id = (tokens && S > 0)
                              ? static_cast<uint32_t>(tokens[stage_token_index])
                              : 0;
    input_meta.route = route_label;

    stage_trace_tensor("x_in", stage_phase, stage_prompt_id, layer_idx,
                       static_cast<uint32_t>(trace_step_), stage_pos_id,
                       static_cast<uint32_t>(seq_len), stage_tokens_total, x, D,
                       stage_token_index, hip_stream, &input_meta);
  }
  if (post_wo_layer) {
    post_wo_trace_tensor("x_in", stage_phase, stage_prompt_id, layer_idx,
                         static_cast<uint32_t>(trace_step_), stage_pos_id,
                         static_cast<uint32_t>(seq_len), stage_tokens_total, x,
                         D, stage_token_index, activations_.x.size(),
                         hip_stream);
  }
  if (trace_layer) {
    layer_tracer_.trace_tensor("x", trace_step_, static_cast<int>(layer_idx),
                               hip_stream, x, n_x);
  }

  gcore::rt::GretaEvent *start = nullptr, *stop = nullptr;
  if (PROFILE_ON()) {
    start = gcore::rt::GretaContext::instance().create_event();
    stop = gcore::rt::GretaContext::instance().create_event();
    start->record(stream_);
  }

  bool profile_attn = (std::getenv("GRETA_PROFILE_ATTN") != nullptr);
  gcore::rt::GretaEvent *ev_q_start = nullptr, *ev_q_end = nullptr;
  gcore::rt::GretaEvent *ev_k_start = nullptr, *ev_k_end = nullptr;
  gcore::rt::GretaEvent *ev_v_start = nullptr, *ev_v_end = nullptr;
  gcore::rt::GretaEvent *ev_rope_start = nullptr, *ev_rope_end = nullptr;
  gcore::rt::GretaEvent *ev_kv_start = nullptr, *ev_kv_end = nullptr;
  gcore::rt::GretaEvent *ev_core_start = nullptr, *ev_core_end = nullptr;

  if (profile_attn) {
    auto &ctx = gcore::rt::GretaContext::instance();
    ev_q_start = ctx.create_event();
    ev_q_end = ctx.create_event();
    ev_k_start = ctx.create_event();
    ev_k_end = ctx.create_event();
    ev_v_start = ctx.create_event();
    ev_v_end = ctx.create_event();
    ev_rope_start = ctx.create_event();
    ev_rope_end = ctx.create_event();
    ev_kv_start = ctx.create_event();
    ev_kv_end = ctx.create_event();
    ev_core_start = ctx.create_event();
    ev_core_end = ctx.create_event();
  }

  const char *use_fused_env = std::getenv("GRETA_USE_FUSED_RMSNORM");
  bool use_fused = (use_fused_env && std::string(use_fused_env) == "1") &&
                   (S == 1) && (Hkv == Hq);
  const char *qkv_force_route = qkv_force_route_env();
  const bool qkv_force_gemm = qkv_force_gemm_enabled();
  if (is_decode_step && qkv_force_gemm) {
    use_fused = false;
  }
  if (layer_idx == 0 && trace_qkv_w_verify_enabled()) {
    use_fused = false;
  }

  std::string q_route_used = "unknown";
  std::string k_route_used = "unknown";
  std::string v_route_used = "unknown";

  if (use_fused) {
    CHECK_HIP_KERNEL(launch_fused_rmsnorm_qkv_gemv_f16(
                         hip_stream, x, attn_norm,
                         static_cast<const __half *>(b.wq.data()),
                         static_cast<const __half *>(b.wk.data()),
                         static_cast<const __half *>(b.wv.data()), q, k, v, D,
                         config_.rms_eps),
                     "Fused RMSNorm+QKV");
    q_route_used = "FUSED_GEMV";
    k_route_used = "FUSED_GEMV";
    v_route_used = "FUSED_GEMV";

    if (trace_layer) {
      layer_tracer_.trace_tensor("q", trace_step_, static_cast<int>(layer_idx),
                                 hip_stream, q, n_x);
      layer_tracer_.trace_tensor("k", trace_step_, static_cast<int>(layer_idx),
                                 hip_stream, k, n_kv);
      layer_tracer_.trace_tensor("v", trace_step_, static_cast<int>(layer_idx),
                                 hip_stream, v, n_kv);
    }
  } else {
    CHECK_HIP_KERNEL(launch_rmsnorm_naive(hip_stream, x, attn_norm, norm_out, S,
                                          D, config_.rms_eps),
                     "RMSNorm (Attn)");

    if (trace_layer) {
      layer_tracer_.trace_tensor("norm_out", trace_step_,
                                 static_cast<int>(layer_idx), hip_stream,
                                 norm_out, n_x);
    }

    if (TRACE_ON(layer_idx)) {
      launch_debug_tensor_stats(hip_stream, "L.attn_rmsnorm.norm_out", norm_out,
                                n_x);
    }

    if (profile_attn)
      ev_q_start->record(stream_);
    gcore::compute::GretaCompute::set_op_label(
        is_decode_step ? "attn_q_decode" : "attn_q_prefill");
    if (is_decode_step && qkv_force_route) {
      const std::string forced = to_route_label(qkv_force_route);
      EnvOverride guard("GRETA_GEMM_FORCE", forced.c_str());
      CHECK_GRETA(
          gcore::compute::GretaCompute::gemm(stream_, &activations_.norm_out,
                                             &b.wq, &activations_.q, S, D, D),
          "GEMM Q");
    } else {
      CHECK_GRETA(
          gcore::compute::GretaCompute::gemm(stream_, &activations_.norm_out,
                                             &b.wq, &activations_.q, S, D, D),
          "GEMM Q");
    }
    q_route_used =
        qkv_route_used(S, is_decode_step, use_fused, qkv_force_route);
    gcore::compute::GretaCompute::set_op_label(nullptr);
    if (profile_attn)
      ev_q_end->record(stream_);

    if (profile_attn)
      ev_k_start->record(stream_);
    gcore::compute::GretaCompute::set_op_label(
        is_decode_step ? "attn_k_decode" : "attn_k_prefill");
    if (is_decode_step && qkv_force_route) {
      const std::string forced = to_route_label(qkv_force_route);
      EnvOverride guard("GRETA_GEMM_FORCE", forced.c_str());
      CHECK_GRETA(gcore::compute::GretaCompute::gemm(
                      stream_, &activations_.norm_out, &b.wk, &activations_.k,
                      S, kv_dim, D),
                  "GEMM K");
    } else {
      CHECK_GRETA(gcore::compute::GretaCompute::gemm(
                      stream_, &activations_.norm_out, &b.wk, &activations_.k,
                      S, kv_dim, D),
                  "GEMM K");
    }
    k_route_used =
        qkv_route_used(S, is_decode_step, use_fused, qkv_force_route);
    gcore::compute::GretaCompute::set_op_label(nullptr);
    if (profile_attn)
      ev_k_end->record(stream_);

    if (profile_attn)
      ev_v_start->record(stream_);
    gcore::compute::GretaCompute::set_op_label(
        is_decode_step ? "attn_v_decode" : "attn_v_prefill");
    if (is_decode_step && qkv_force_route) {
      const std::string forced = to_route_label(qkv_force_route);
      EnvOverride guard("GRETA_GEMM_FORCE", forced.c_str());
      CHECK_GRETA(gcore::compute::GretaCompute::gemm(
                      stream_, &activations_.norm_out, &b.wv, &activations_.v,
                      S, kv_dim, D),
                  "GEMM V");
    } else {
      CHECK_GRETA(gcore::compute::GretaCompute::gemm(
                      stream_, &activations_.norm_out, &b.wv, &activations_.v,
                      S, kv_dim, D),
                  "GEMM V");
    }
    v_route_used =
        qkv_route_used(S, is_decode_step, use_fused, qkv_force_route);
    gcore::compute::GretaCompute::set_op_label(nullptr);
    if (profile_attn)
      ev_v_end->record(stream_);

    if (trace_layer) {
      layer_tracer_.trace_tensor("q", trace_step_, static_cast<int>(layer_idx),
                                 hip_stream, q, n_x);
      layer_tracer_.trace_tensor("k", trace_step_, static_cast<int>(layer_idx),
                                 hip_stream, k, n_kv);
      layer_tracer_.trace_tensor("v", trace_step_, static_cast<int>(layer_idx),
                                 hip_stream, v, n_kv);
    }
  }

  if (TRACE_ON(layer_idx)) {
    launch_debug_tensor_stats(hip_stream, "L.q_proj.q", q, n_x);
    launch_debug_tensor_stats(hip_stream, "L.k_proj.k", k, n_kv);
    launch_debug_tensor_stats(hip_stream, "L.v_proj.v", v, n_kv);
  }

  uint32_t pos = static_cast<uint32_t>(seq_start);
  const char *use_fused_attn_env = std::getenv("GRETA_USE_FUSED_ATTENTION");
  bool use_fused_attn =
      (use_fused_attn_env && std::string(use_fused_attn_env) == "1") &&
      (S == 1) && (Hkv == Hq);
  if (S == 1) {
    const char *force_attn = std::getenv("GRETA_FORCE_ATTN_DECODE_KERNEL");
    if (force_attn) {
      std::string force_str(force_attn);
      if (force_str == "manual" || force_str == "MANUAL") {
        use_fused_attn = false;
      } else if (force_str == "fused" || force_str == "FUSED") {
        use_fused_attn = true;
      }
    }
  }

  if (profile_attn)
    ev_rope_start->record(stream_);

  if (use_fused_attn) {
    CHECK_HIP_KERNEL(launch_fused_rope_kv_update_decode(
                         hip_stream, q, k, v, cache_k, cache_v, d_pos,
                         config_.max_seq_len, Hkv, Dh, config_.rope_base),
                     "Fused RoPE+KV Update");
    CHECK_HIP_KERNEL(
        launch_rope(hip_stream, q, S, Hq, Dh, config_.rope_base, d_pos),
        "RoPE Q (Fused KV)");
  } else {
    if (S == 1) {
      CHECK_HIP_KERNEL(
          launch_rope(hip_stream, q, S, Hq, Dh, config_.rope_base, d_pos),
          "RoPE Q");
      CHECK_HIP_KERNEL(
          launch_rope(hip_stream, k, S, Hkv, Dh, config_.rope_base, d_pos),
          "RoPE K");
    } else {
      CHECK_HIP_KERNEL(
          launch_rope(hip_stream, q, S, Hq, Dh, config_.rope_base, pos),
          "RoPE Q");
      CHECK_HIP_KERNEL(
          launch_rope(hip_stream, k, S, Hkv, Dh, config_.rope_base, pos),
          "RoPE K");
    }
  }
  if (profile_attn)
    ev_rope_end->record(stream_);

  if (profile_attn)
    ev_kv_start->record(stream_);
  if (!use_fused_attn) {
    if (S == 1) {
      CHECK_HIP_KERNEL(launch_kv_update(hip_stream, cache_k, cache_v, k, v,
                                        d_pos, config_.max_seq_len, Hkv, Dh),
                       "KV Update");
    } else {
      for (uint32_t s = 0; s < S; ++s) {
        CHECK_HIP_KERNEL(launch_kv_update(hip_stream, cache_k, cache_v,
                                          k + s * kv_dim, v + s * kv_dim,
                                          pos + s, config_.max_seq_len, Hkv,
                                          Dh),
                         "KV Update");
      }
    }
  }
  if (profile_attn)
    ev_kv_end->record(stream_);

  float scale = 1.0f / sqrtf(static_cast<float>(Dh));
  if (profile_attn)
    ev_core_start->record(stream_);

  if (S == 1) {
    const int accum_mode = (attn_accum_mode() == AttnAccumMode::Fp16) ? 1 : 0;
    CHECK_HIP_KERNEL(launch_flash_attention_decode(
                         hip_stream, q, cache_k, cache_v, attn_out, Hq, Hkv,
                         d_pos, static_cast<uint32_t>(config_.max_seq_len), Dh,
                         scale, accum_mode),
                     "Attention Core (Decode)");
  } else {
    CHECK_HIP_KERNEL(launch_flash_attention_prefill(hip_stream, q, k, v,
                                                    attn_out, S, Hq, Hkv, Dh,
                                                    scale, true),
                     "Flash Attention Prefill");
  }

  if (profile_attn)
    ev_core_end->record(stream_);

  if (profile_attn && layer_idx == 0) {
    stream_->synchronize();
    float q_ms = ev_q_end->elapsed_time_since(ev_q_start);
    float k_ms = ev_k_end->elapsed_time_since(ev_k_start);
    float v_ms = ev_v_end->elapsed_time_since(ev_v_start);
    float rope_ms = ev_rope_end->elapsed_time_since(ev_rope_start);
    float kv_ms = ev_kv_end->elapsed_time_since(ev_kv_start);
    float core_ms = ev_core_end->elapsed_time_since(ev_core_start);

    printf(
        "[ATTN_PROF] Q:%.3f K:%.3f V:%.3f RoPE:%.3f KV:%.3f Core:%.3f (ms)\n",
        q_ms, k_ms, v_ms, rope_ms, kv_ms, core_ms);

    delete ev_q_start;
    delete ev_q_end;
    delete ev_k_start;
    delete ev_k_end;
    delete ev_v_start;
    delete ev_v_end;
    delete ev_rope_start;
    delete ev_rope_end;
    delete ev_kv_start;
    delete ev_kv_end;
    delete ev_core_start;
    delete ev_core_end;
  }

  if (attn_mfma_shadow_enabled() && seq_len == 1 && trace_step_ == 1 &&
      attn_trace_layer_selected(layer_idx, config_.num_layers)) {
    const char *out = attn_shadow_out_path();
    if (out && *out) {
      using Usage = gcore::rt::hip::BufferUsage;
      const size_t q_bytes = D * sizeof(float);
      const size_t kv_bytes = kv_dim * sizeof(float);
      const size_t kv_layer_stride_elems =
          static_cast<size_t>(config_.max_seq_len) * Hkv * Dh;
      const size_t kv_layer_stride_bytes =
          kv_layer_stride_elems * sizeof(float);
      const size_t kv_layer_offset_elems =
          static_cast<size_t>(layer_idx) * kv_layer_stride_elems;
      const float *k_cache_layer =
          static_cast<const float *>(activations_.kv_cache_k.data()) +
          kv_layer_offset_elems;
      const float *v_cache_layer =
          static_cast<const float *>(activations_.kv_cache_v.data()) +
          kv_layer_offset_elems;

      gcore::rt::hip::Buffer q_shadow;
      gcore::rt::hip::Buffer k_shadow;
      gcore::rt::hip::Buffer v_shadow;
      gcore::rt::hip::Buffer attn_out_mfma;
      gcore::rt::hip::Buffer attn_out_valu;
      gcore::rt::hip::Buffer kv_shadow_k;
      gcore::rt::hip::Buffer kv_shadow_v;

      std::string shadow_err;
      bool alloc_ok =
          q_shadow.allocate(q_bytes, Usage::DeviceOnly,
                            gcore::rt::GretaDataType::FP32, &shadow_err) &&
          k_shadow.allocate(kv_bytes, Usage::DeviceOnly,
                            gcore::rt::GretaDataType::FP32, &shadow_err) &&
          v_shadow.allocate(kv_bytes, Usage::DeviceOnly,
                            gcore::rt::GretaDataType::FP32, &shadow_err) &&
          attn_out_mfma.allocate(q_bytes, Usage::DeviceOnly,
                                 gcore::rt::GretaDataType::FP32, &shadow_err) &&
          attn_out_valu.allocate(q_bytes, Usage::DeviceOnly,
                                 gcore::rt::GretaDataType::FP32, &shadow_err) &&
          kv_shadow_k.allocate(kv_layer_stride_bytes, Usage::DeviceOnly,
                               gcore::rt::GretaDataType::FP32, &shadow_err) &&
          kv_shadow_v.allocate(kv_layer_stride_bytes, Usage::DeviceOnly,
                               gcore::rt::GretaDataType::FP32, &shadow_err);

      if (!alloc_ok) {
        std::cerr << "[ATTN_SHADOW] alloc failed: " << shadow_err << "\n";
      } else {
        auto run_route = [&](const char *route,
                             gcore::rt::hip::Buffer &attn_out_buf,
                             uint64_t &hash_out, F32Stats &stats_out) -> bool {
          EnvOverride guard("GRETA_FORCE_ATTN_DECODE_MATMUL", route);

          gcore::compute::GretaCompute::set_op_label("attn_q_decode");
          if (gcore::compute::GretaCompute::gemm(
                  stream_, &activations_.norm_out, &b.wq, &q_shadow, S, D, D) !=
              gcore::compute::GretaResult::SUCCESS) {
            gcore::compute::GretaCompute::set_op_label(nullptr);
            return false;
          }
          gcore::compute::GretaCompute::set_op_label("attn_k_decode");
          if (gcore::compute::GretaCompute::gemm(
                  stream_, &activations_.norm_out, &b.wk, &k_shadow, S, kv_dim,
                  D) != gcore::compute::GretaResult::SUCCESS) {
            gcore::compute::GretaCompute::set_op_label(nullptr);
            return false;
          }
          gcore::compute::GretaCompute::set_op_label("attn_v_decode");
          if (gcore::compute::GretaCompute::gemm(
                  stream_, &activations_.norm_out, &b.wv, &v_shadow, S, kv_dim,
                  D) != gcore::compute::GretaResult::SUCCESS) {
            gcore::compute::GretaCompute::set_op_label(nullptr);
            return false;
          }
          gcore::compute::GretaCompute::set_op_label(nullptr);

          if (use_fused_attn) {
            CHECK_HIP_KERNEL(
                launch_fused_rope_kv_update_decode(
                    hip_stream, static_cast<float *>(q_shadow.data()),
                    static_cast<float *>(k_shadow.data()),
                    static_cast<float *>(v_shadow.data()),
                    static_cast<float *>(kv_shadow_k.data()),
                    static_cast<float *>(kv_shadow_v.data()), d_pos,
                    config_.max_seq_len, Hkv, Dh, config_.rope_base),
                "Fused RoPE+KV Update (Shadow)");
            CHECK_HIP_KERNEL(launch_rope(hip_stream,
                                         static_cast<float *>(q_shadow.data()),
                                         S, Hq, Dh, config_.rope_base, d_pos),
                             "RoPE Q (Shadow)");
          } else {
            CHECK_HIP_KERNEL(launch_rope(hip_stream,
                                         static_cast<float *>(q_shadow.data()),
                                         S, Hq, Dh, config_.rope_base, d_pos),
                             "RoPE Q (Shadow)");
            CHECK_HIP_KERNEL(launch_rope(hip_stream,
                                         static_cast<float *>(k_shadow.data()),
                                         S, Hkv, Dh, config_.rope_base, d_pos),
                             "RoPE K (Shadow)");
            CHECK_HIP_KERNEL(
                launch_kv_update(hip_stream,
                                 static_cast<float *>(kv_shadow_k.data()),
                                 static_cast<float *>(kv_shadow_v.data()),
                                 static_cast<float *>(k_shadow.data()),
                                 static_cast<float *>(v_shadow.data()), d_pos,
                                 config_.max_seq_len, Hkv, Dh),
                "KV Update (Shadow)");
          }

          CHECK_GRETA(gcore::compute::GretaCompute::attention_decode(
                          stream_, &q_shadow, &kv_shadow_k, &kv_shadow_v,
                          &activations_.d_pos, &attn_out_buf, Hq, Hkv, Dh, S,
                          config_.max_seq_len, scale, config_.rope_base),
                      "Attention Core (Shadow)");

          std::vector<float> host(D);
          hipMemcpy(host.data(), attn_out_buf.data(), q_bytes,
                    hipMemcpyDeviceToHost);
          stats_out = stats_f32(host.data(), host.size());
          hash_out = hash_f32(host.data(), host.size());
          return true;
        };

        uint64_t hash_mfma = 0;
        uint64_t hash_valu = 0;
        F32Stats stats_mfma{};
        F32Stats stats_valu{};
        bool ok_mfma = false;
        bool ok_valu = false;

        if (hipMemcpy(kv_shadow_k.data(), k_cache_layer, kv_layer_stride_bytes,
                      hipMemcpyDeviceToDevice) == hipSuccess &&
            hipMemcpy(kv_shadow_v.data(), v_cache_layer, kv_layer_stride_bytes,
                      hipMemcpyDeviceToDevice) == hipSuccess) {
          ok_mfma = run_route("mfma", attn_out_mfma, hash_mfma, stats_mfma);
        }

        if (hipMemcpy(kv_shadow_k.data(), k_cache_layer, kv_layer_stride_bytes,
                      hipMemcpyDeviceToDevice) == hipSuccess &&
            hipMemcpy(kv_shadow_v.data(), v_cache_layer, kv_layer_stride_bytes,
                      hipMemcpyDeviceToDevice) == hipSuccess) {
          ok_valu = run_route("valu", attn_out_valu, hash_valu, stats_valu);
        }

        std::vector<float> mfma_host;
        std::vector<float> valu_host;
        double mae = 0.0;
        float max_diff = 0.0f;
        if (ok_mfma && ok_valu) {
          mfma_host.resize(D);
          valu_host.resize(D);
          hipMemcpy(mfma_host.data(), attn_out_mfma.data(), q_bytes,
                    hipMemcpyDeviceToHost);
          hipMemcpy(valu_host.data(), attn_out_valu.data(), q_bytes,
                    hipMemcpyDeviceToHost);
          mae_max_f32(mfma_host.data(), valu_host.data(), mfma_host.size(),
                      &mae, &max_diff);
        }

        std::ofstream ofs(out, std::ios::app);
        if (ofs) {
          ofs << "{\"phase\":\"decode0_shadow\",\"step\":" << trace_step_
              << ",\"layer\":" << layer_idx
              << ",\"mfma_ok\":" << (ok_mfma ? "true" : "false")
              << ",\"valu_ok\":" << (ok_valu ? "true" : "false")
              << ",\"attn_out_mfma_hash\":" << hash_mfma
              << ",\"attn_out_valu_hash\":" << hash_valu
              << ",\"attn_out_mae\":" << mae
              << ",\"attn_out_max_diff\":" << max_diff << ",\"head_dim\":" << Dh
              << ",\"kv_heads\":" << Hkv << ",\"seq_len\":" << (seq_start + 1)
              << ",\"use_fused_attn\":" << (use_fused_attn ? "true" : "false")
              << "}\n";
        }
      }
    }
  }

  if ((trace_attn_l0_pipe_enabled() || trace_attn_l0_norm_enabled() ||
       trace_qkv_w_verify_enabled()) &&
      layer_idx == 0) {
    const char *out = attn_l0_pipe_out_path();
    const char *phase = nullptr;
    if (seq_len > 1 && trace_step_ == 0) {
      phase = "prefill_last";
    } else if (seq_len == 1 && trace_step_ == 1) {
      phase = "decode0";
    }
    if (out && *out && phase) {
      const bool qkv_w_verify = trace_qkv_w_verify_enabled();
      const uint32_t head = 0;
      const uint32_t max_seq_len = static_cast<uint32_t>(config_.max_seq_len);
      uint32_t seq_len_used = (seq_len == 1)
                                  ? static_cast<uint32_t>(seq_start + 1)
                                  : static_cast<uint32_t>(seq_len);
      if (seq_len_used > max_seq_len)
        seq_len_used = max_seq_len;
      uint32_t pos_id_used =
          (seq_len == 1) ? static_cast<uint32_t>(seq_start)
                         : static_cast<uint32_t>(seq_start + seq_len - 1);
      const uint32_t token_index_used =
          (seq_len == 1) ? 0 : static_cast<uint32_t>(seq_len - 1);
      if (seq_len_used > 0 && pos_id_used >= seq_len_used) {
        pos_id_used = seq_len_used - 1;
      }
      const uint32_t group = (Hkv > 0) ? (Hq / Hkv) : 0;
      const uint32_t kv_head = (group > 0 && head < Hq) ? (head / group) : 0;
      if (kv_head < Hkv && Dh > 0 && seq_len_used > 0) {
        const float *q_token = q + static_cast<size_t>(token_index_used) * D;
        const float *q_head = q_token + head * Dh;

        const bool want_norm = trace_attn_l0_norm_enabled() || qkv_w_verify;
        const uint32_t norm_sample =
            want_norm
                ? (qkv_w_verify ? D
                                : std::min<uint32_t>(stage_trace_sample(), D))
                : 0;
        std::vector<float> norm_in_host;
        std::vector<float> norm_out_host;
        bool norm_out_valid = false;

        const size_t kv_head_stride_elems =
            static_cast<size_t>(max_seq_len) * Dh;
        const size_t kv_layer_stride_elems =
            static_cast<size_t>(max_seq_len) * Hkv * Dh;
        const size_t kv_head_stride_bytes =
            kv_head_stride_elems * sizeof(float);
        const size_t kv_layer_stride_bytes =
            kv_layer_stride_elems * sizeof(float);
        const size_t kv_pos_stride_bytes =
            static_cast<size_t>(Dh) * sizeof(float);

        const float *k_head_base =
            cache_k + static_cast<size_t>(kv_head) * kv_head_stride_elems;
        const float *v_head_base =
            cache_v + static_cast<size_t>(kv_head) * kv_head_stride_elems;

        std::vector<float> q_host(Dh);
        std::vector<float> k_host(seq_len_used * Dh);
        std::vector<float> v_host(seq_len_used * Dh);
        std::vector<float> attn_host(Dh);

        if (want_norm && norm_sample > 0) {
          norm_in_host.assign(norm_sample, 0.0f);
          const float *x_token = x + static_cast<size_t>(token_index_used) * D;
          hipMemcpyAsync(norm_in_host.data(), x_token,
                         norm_sample * sizeof(float), hipMemcpyDeviceToHost,
                         hip_stream);
          if (!use_fused) {
            norm_out_valid = true;
            norm_out_host.assign(norm_sample, 0.0f);
            const float *norm_token =
                norm_out + static_cast<size_t>(token_index_used) * D;
            hipMemcpyAsync(norm_out_host.data(), norm_token,
                           norm_sample * sizeof(float), hipMemcpyDeviceToHost,
                           hip_stream);
          }
        }

        hipMemcpyAsync(q_host.data(), q_head, Dh * sizeof(float),
                       hipMemcpyDeviceToHost, hip_stream);
        hipMemcpyAsync(k_host.data(), k_head_base,
                       static_cast<size_t>(seq_len_used) * Dh * sizeof(float),
                       hipMemcpyDeviceToHost, hip_stream);
        hipMemcpyAsync(v_host.data(), v_head_base,
                       static_cast<size_t>(seq_len_used) * Dh * sizeof(float),
                       hipMemcpyDeviceToHost, hip_stream);

        const float *attn_token =
            attn_out + static_cast<size_t>(token_index_used) * D;
        const float *attn_head = attn_token + head * Dh;
        hipMemcpyAsync(attn_host.data(), attn_head, Dh * sizeof(float),
                       hipMemcpyDeviceToHost, hip_stream);
        hipStreamSynchronize(hip_stream);

        const float *k_vec =
            k_host.data() + static_cast<size_t>(pos_id_used) * Dh;
        const float *v_vec =
            v_host.data() + static_cast<size_t>(pos_id_used) * Dh;

        const F32Stats q_stats = stats_f32(q_host.data(), q_host.size());
        const F32Stats k_stats = stats_f32(k_vec, Dh);
        const F32Stats v_stats = stats_f32(v_vec, Dh);
        const F32Stats attn_stats =
            stats_f32(attn_host.data(), attn_host.size());

        const uint64_t q_hash = hash_f32(q_host.data(), q_host.size());
        const uint64_t k_hash = hash_f32(k_vec, Dh);
        const uint64_t v_hash = hash_f32(v_vec, Dh);
        const uint64_t attn_hash = hash_f32(attn_host.data(), attn_host.size());

        F32Stats norm_in_stats{};
        F32Stats norm_out_stats{};
        uint64_t norm_in_hash = 0;
        uint64_t norm_out_hash = 0;
        if (!norm_in_host.empty()) {
          norm_in_stats = stats_f32(norm_in_host.data(), norm_in_host.size());
          norm_in_hash = hash_f32(norm_in_host.data(), norm_in_host.size());
        }
        if (norm_out_valid && !norm_out_host.empty()) {
          norm_out_stats =
              stats_f32(norm_out_host.data(), norm_out_host.size());
          norm_out_hash = hash_f32(norm_out_host.data(), norm_out_host.size());
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
        std::vector<float> qk_row(seq_len_used, 0.0f);
        std::vector<float> soft_row(seq_len_used, 0.0f);
        double max_qk = -INFINITY;
        for (uint32_t t = 0; t < seq_len_used; ++t) {
          const float *k_t = k_host.data() + static_cast<size_t>(t) * Dh;
          double dot = 0.0;
          for (uint32_t d = 0; d < Dh; ++d) {
            dot += static_cast<double>(q_host[d]) * static_cast<double>(k_t[d]);
          }
          const double qk = dot * static_cast<double>(scale);
          qk_row[t] = static_cast<float>(qk);
          if (qk > max_qk)
            max_qk = qk;
        }
        double sumexp = 0.0;
        for (uint32_t t = 0; t < seq_len_used; ++t) {
          sumexp += std::exp(static_cast<double>(qk_row[t]) - max_qk);
        }
        const double inv_sum = sumexp > 0.0 ? (1.0 / sumexp) : 0.0;
        for (uint32_t t = 0; t < seq_len_used; ++t) {
          soft_row[t] = static_cast<float>(
              std::exp(static_cast<double>(qk_row[t]) - max_qk) * inv_sum);
        }

        const F32Stats qk_stats = stats_f32(qk_row.data(), qk_row.size());
        const F32Stats soft_stats = stats_f32(soft_row.data(), soft_row.size());
        const uint64_t qk_hash = hash_f32(qk_row.data(), qk_row.size());
        const uint64_t soft_hash = hash_f32(soft_row.data(), soft_row.size());

        std::vector<double> pv_accum(Dh, 0.0);
        for (uint32_t t = 0; t < seq_len_used; ++t) {
          const double p = static_cast<double>(soft_row[t]);
          const float *v_t = v_host.data() + static_cast<size_t>(t) * Dh;
          for (uint32_t d = 0; d < Dh; ++d) {
            pv_accum[d] += p * static_cast<double>(v_t[d]);
          }
        }
        std::vector<float> pv_row(Dh, 0.0f);
        for (uint32_t d = 0; d < Dh; ++d) {
          pv_row[d] = static_cast<float>(pv_accum[d]);
        }

        const F32Stats pv_stats = stats_f32(pv_row.data(), pv_row.size());
        const uint64_t pv_hash = hash_f32(pv_row.data(), pv_row.size());

        double pv_attn_mae = 0.0;
        float pv_attn_max = 0.0f;
        mae_max_f32(pv_row.data(), attn_host.data(), pv_row.size(),
                    &pv_attn_mae, &pv_attn_max);

        std::string q_weight_layout =
            qkv_w_verify ? std::string("unavailable") : std::string("disabled");
        double q_weight_mae_row = 0.0;
        double q_weight_mae_col = 0.0;
        float q_weight_max_row = 0.0f;
        float q_weight_max_col = 0.0f;
        uint32_t q_weight_sample = 0;
        std::string k_weight_layout =
            qkv_w_verify ? std::string("unavailable") : std::string("disabled");
        double k_weight_mae_row = 0.0;
        double k_weight_mae_col = 0.0;
        float k_weight_max_row = 0.0f;
        float k_weight_max_col = 0.0f;
        uint32_t k_weight_sample = 0;
        std::string v_weight_layout =
            qkv_w_verify ? std::string("unavailable") : std::string("disabled");
        double v_weight_mae_row = 0.0;
        double v_weight_mae_col = 0.0;
        float v_weight_max_row = 0.0f;
        float v_weight_max_col = 0.0f;
        uint32_t v_weight_sample = 0;
        if (qkv_w_verify && norm_out_valid &&
            norm_out_host.size() >= static_cast<size_t>(D)) {
          q_weight_sample = std::min<uint32_t>(attn_vacc_dims_sample(),
                                               static_cast<uint32_t>(Dh));
          if (q_weight_sample > 0 && Dh > 0) {
            static QkvWeightHostCache wq_cache;
            const size_t wq_elems = static_cast<size_t>(D) * D;
            if (ensure_qkv_weight_cache(b.wq, b.s_wq, b.sh_wq, wq_elems,
                                        &wq_cache)) {
              std::vector<float> q_row(static_cast<size_t>(Dh), 0.0f);
              std::vector<float> q_col(static_cast<size_t>(Dh), 0.0f);
              const float *x_vec = norm_out_host.data();
              for (uint32_t j = 0; j < Dh; ++j) {
                double sum_row = 0.0;
                double sum_col = 0.0;
                for (uint32_t i = 0; i < D; ++i) {
                  const float x_i = x_vec[i];
                  const size_t idx_row = static_cast<size_t>(j) * D + i;
                  const size_t idx_col = static_cast<size_t>(i) * D + j;
                  sum_row +=
                      static_cast<double>(x_i) *
                      static_cast<double>(read_weight_value(wq_cache, idx_row));
                  sum_col +=
                      static_cast<double>(x_i) *
                      static_cast<double>(read_weight_value(wq_cache, idx_col));
                }
                float h_scale = 1.0f;
                if (!wq_cache.head_scales.empty() && Dh > 0) {
                  const size_t h_idx = j / Dh;
                  if (h_idx < wq_cache.head_scales.size()) {
                    h_scale = wq_cache.head_scales[h_idx];
                  }
                }
                q_row[j] = static_cast<float>(sum_row * h_scale);
                q_col[j] = static_cast<float>(sum_col * h_scale);
              }
              if (Dh % 2 == 0) {
                for (uint32_t pair = 0; pair < Dh / 2; ++pair) {
                  const float base = static_cast<float>(config_.rope_base);
                  const float theta =
                      static_cast<float>(pos_id_used) *
                      std::pow(base, -2.0f * (static_cast<float>(pair) /
                                              static_cast<float>(Dh)));
                  const float cos_val = std::cos(theta);
                  const float sin_val = std::sin(theta);

                  float v0 = q_row[pair];
                  float v1 = q_row[pair + Dh / 2];
                  float out0 = v0 * cos_val - v1 * sin_val;
                  float out1 = v0 * sin_val + v1 * cos_val;
                  q_row[pair] = out0;
                  q_row[pair + Dh / 2] = out1;

                  v0 = q_col[pair];
                  v1 = q_col[pair + Dh / 2];
                  out0 = v0 * cos_val - v1 * sin_val;
                  out1 = v0 * sin_val + v1 * cos_val;
                  q_col[pair] = out0;
                  q_col[pair + Dh / 2] = out1;
                }
              }
              mae_max_f32(q_row.data(), q_host.data(), q_weight_sample,
                          &q_weight_mae_row, &q_weight_max_row);
              mae_max_f32(q_col.data(), q_host.data(), q_weight_sample,
                          &q_weight_mae_col, &q_weight_max_col);
              q_weight_layout =
                  (q_weight_mae_row <= q_weight_mae_col) ? "row" : "col";
            } else {
              q_weight_layout = "error";
            }
          }

          if (q_weight_sample > 0 && Dh > 0 && kv_dim > 0) {
            k_weight_sample = q_weight_sample;
            static QkvWeightHostCache wk_cache;
            const size_t wk_elems = static_cast<size_t>(D) * kv_dim;
            if (ensure_qkv_weight_cache(b.wk, b.s_wk, b.sh_wk, wk_elems,
                                        &wk_cache)) {
              std::vector<float> k_row(static_cast<size_t>(Dh), 0.0f);
              std::vector<float> k_col(static_cast<size_t>(Dh), 0.0f);
              const float *x_vec = norm_out_host.data();
              const uint32_t row_base = kv_head * Dh;
              for (uint32_t j = 0; j < Dh; ++j) {
                const uint32_t row = row_base + j;
                double sum_row = 0.0;
                double sum_col = 0.0;
                for (uint32_t i = 0; i < D; ++i) {
                  const float x_i = x_vec[i];
                  const size_t idx_row = static_cast<size_t>(row) * D + i;
                  const size_t idx_col = static_cast<size_t>(i) * kv_dim + row;
                  sum_row +=
                      static_cast<double>(x_i) *
                      static_cast<double>(read_weight_value(wk_cache, idx_row));
                  sum_col +=
                      static_cast<double>(x_i) *
                      static_cast<double>(read_weight_value(wk_cache, idx_col));
                }
                float h_scale = 1.0f;
                if (!wk_cache.head_scales.empty() && Dh > 0) {
                  const size_t h_idx = row / Dh;
                  if (h_idx < wk_cache.head_scales.size()) {
                    h_scale = wk_cache.head_scales[h_idx];
                  }
                }
                k_row[j] = static_cast<float>(sum_row * h_scale);
                k_col[j] = static_cast<float>(sum_col * h_scale);
              }
              if (Dh % 2 == 0) {
                for (uint32_t pair = 0; pair < Dh / 2; ++pair) {
                  const float base = static_cast<float>(config_.rope_base);
                  const float theta =
                      static_cast<float>(pos_id_used) *
                      std::pow(base, -2.0f * (static_cast<float>(pair) /
                                              static_cast<float>(Dh)));
                  const float cos_val = std::cos(theta);
                  const float sin_val = std::sin(theta);

                  float v0 = k_row[pair];
                  float v1 = k_row[pair + Dh / 2];
                  float out0 = v0 * cos_val - v1 * sin_val;
                  float out1 = v0 * sin_val + v1 * cos_val;
                  k_row[pair] = out0;
                  k_row[pair + Dh / 2] = out1;

                  v0 = k_col[pair];
                  v1 = k_col[pair + Dh / 2];
                  out0 = v0 * cos_val - v1 * sin_val;
                  out1 = v0 * sin_val + v1 * cos_val;
                  k_col[pair] = out0;
                  k_col[pair + Dh / 2] = out1;
                }
              }
              mae_max_f32(k_row.data(), k_vec, k_weight_sample,
                          &k_weight_mae_row, &k_weight_max_row);
              mae_max_f32(k_col.data(), k_vec, k_weight_sample,
                          &k_weight_mae_col, &k_weight_max_col);
              k_weight_layout =
                  (k_weight_mae_row <= k_weight_mae_col) ? "row" : "col";
            } else {
              k_weight_layout = "error";
            }
          }

          if (q_weight_sample > 0 && Dh > 0 && kv_dim > 0) {
            v_weight_sample = q_weight_sample;
            static QkvWeightHostCache wv_cache;
            const size_t wv_elems = static_cast<size_t>(D) * kv_dim;
            if (ensure_qkv_weight_cache(b.wv, b.s_wv, b.sh_wv, wv_elems,
                                        &wv_cache)) {
              std::vector<float> v_row(static_cast<size_t>(Dh), 0.0f);
              std::vector<float> v_col(static_cast<size_t>(Dh), 0.0f);
              const float *x_vec = norm_out_host.data();
              const uint32_t row_base = kv_head * Dh;
              for (uint32_t j = 0; j < Dh; ++j) {
                const uint32_t row = row_base + j;
                double sum_row = 0.0;
                double sum_col = 0.0;
                for (uint32_t i = 0; i < D; ++i) {
                  const float x_i = x_vec[i];
                  const size_t idx_row = static_cast<size_t>(row) * D + i;
                  const size_t idx_col = static_cast<size_t>(i) * kv_dim + row;
                  sum_row +=
                      static_cast<double>(x_i) *
                      static_cast<double>(read_weight_value(wv_cache, idx_row));
                  sum_col +=
                      static_cast<double>(x_i) *
                      static_cast<double>(read_weight_value(wv_cache, idx_col));
                }
                float h_scale = 1.0f;
                if (!wv_cache.head_scales.empty() && Dh > 0) {
                  const size_t h_idx = row / Dh;
                  if (h_idx < wv_cache.head_scales.size()) {
                    h_scale = wv_cache.head_scales[h_idx];
                  }
                }
                v_row[j] = static_cast<float>(sum_row * h_scale);
                v_col[j] = static_cast<float>(sum_col * h_scale);
              }
              mae_max_f32(v_row.data(), v_vec, v_weight_sample,
                          &v_weight_mae_row, &v_weight_max_row);
              mae_max_f32(v_col.data(), v_vec, v_weight_sample,
                          &v_weight_mae_col, &v_weight_max_col);
              v_weight_layout =
                  (v_weight_mae_row <= v_weight_mae_col) ? "row" : "col";
            } else {
              v_weight_layout = "error";
            }
          }
        }

        auto append_span = [&](const char *name, const float *data, size_t n) {
          std::ostringstream tmp;
          tmp << ",\"" << name << "\":[";
          for (size_t i = 0; i < n; ++i) {
            if (i)
              tmp << ",";
            tmp << data[i];
          }
          tmp << "]";
          return tmp.str();
        };

        const char *prompt_id = std::getenv("GRETA_TRACE_PROMPT_ID");
        std::ostringstream oss;
        oss << "{\"event\":\"attn_l0_pipe\"";
        if (prompt_id && *prompt_id)
          oss << ",\"prompt_id\":\"" << prompt_id << "\"";
        oss << ",\"phase\":\"" << phase << "\""
            << ",\"layer\":" << layer_idx << ",\"head\":" << head
            << ",\"seq_len\":" << seq_len
            << ",\"seq_len_used\":" << seq_len_used
            << ",\"tokens_total\":" << (seq_start + seq_len)
            << ",\"pos_id\":" << pos_id_used << ",\"kv_pos\":" << pos_id_used
            << ",\"token_index\":" << token_index_used
            << ",\"scale_used\":" << scale << ",\"q_route_used\":\""
            << q_route_used << "\""
            << ",\"k_route_used\":\"" << k_route_used << "\""
            << ",\"v_route_used\":\"" << v_route_used << "\""
            << ",\"qkv_force_route\":\""
            << (qkv_force_route ? qkv_force_route : "auto") << "\""
            << ",\"qkv_force_gemm\":" << (qkv_force_gemm ? "true" : "false")
            << ",\"attn_norm_in_hash\":" << norm_in_hash
            << ",\"attn_norm_in_min\":" << norm_in_stats.min
            << ",\"attn_norm_in_max\":" << norm_in_stats.max
            << ",\"attn_norm_in_mean\":" << norm_in_stats.mean
            << ",\"attn_norm_out_valid\":"
            << (norm_out_valid ? "true" : "false")
            << ",\"attn_norm_out_hash\":" << norm_out_hash
            << ",\"attn_norm_out_min\":" << norm_out_stats.min
            << ",\"attn_norm_out_max\":" << norm_out_stats.max
            << ",\"attn_norm_out_mean\":" << norm_out_stats.mean
            << ",\"attn_norm_sample_n\":" << norm_sample
            << ",\"q_hash\":" << q_hash << ",\"q_min\":" << q_stats.min
            << ",\"q_max\":" << q_stats.max << ",\"q_mean\":" << q_stats.mean
            << ",\"k_hash\":" << k_hash << ",\"k_min\":" << k_stats.min
            << ",\"k_max\":" << k_stats.max << ",\"k_mean\":" << k_stats.mean
            << ",\"v_hash\":" << v_hash << ",\"v_min\":" << v_stats.min
            << ",\"v_max\":" << v_stats.max << ",\"v_mean\":" << v_stats.mean
            << ",\"qk_hash\":" << qk_hash << ",\"qk_min\":" << qk_stats.min
            << ",\"qk_max\":" << qk_stats.max
            << ",\"qk_mean\":" << qk_stats.mean
            << ",\"softmax_hash\":" << soft_hash
            << ",\"softmax_min\":" << soft_stats.min
            << ",\"softmax_max\":" << soft_stats.max
            << ",\"softmax_mean\":" << soft_stats.mean
            << ",\"pv_hash\":" << pv_hash << ",\"pv_min\":" << pv_stats.min
            << ",\"pv_max\":" << pv_stats.max
            << ",\"pv_mean\":" << pv_stats.mean
            << ",\"attn_out_hash\":" << attn_hash
            << ",\"attn_out_min\":" << attn_stats.min
            << ",\"attn_out_max\":" << attn_stats.max
            << ",\"attn_out_mean\":" << attn_stats.mean
            << ",\"pv_attn_mae\":" << pv_attn_mae
            << ",\"pv_attn_max_diff\":" << pv_attn_max << ",\"q_weight\":\"Q\""
            << ",\"q_weight_layout_best\":\"" << q_weight_layout << "\""
            << ",\"q_weight_mae_row\":" << q_weight_mae_row
            << ",\"q_weight_mae_col\":" << q_weight_mae_col
            << ",\"q_weight_max_row\":" << q_weight_max_row
            << ",\"q_weight_max_col\":" << q_weight_max_col
            << ",\"q_weight_dtype\":\"" << dtype_label(b.wq.data_type()) << "\""
            << ",\"q_weight_quant_mode\":\"" << dtype_label(b.wq.data_type())
            << "\""
            << ",\"q_weight_stride_used\":" << D
            << ",\"q_weight_head_dim\":" << Dh
            << ",\"q_weight_kv_heads\":" << Hkv
            << ",\"q_weight_sample_dims\":" << q_weight_sample
            << ",\"k_weight\":\"K\""
            << ",\"k_weight_layout_best\":\"" << k_weight_layout << "\""
            << ",\"k_weight_mae_row\":" << k_weight_mae_row
            << ",\"k_weight_mae_col\":" << k_weight_mae_col
            << ",\"k_weight_max_row\":" << k_weight_max_row
            << ",\"k_weight_max_col\":" << k_weight_max_col
            << ",\"k_weight_dtype\":\"" << dtype_label(b.wk.data_type()) << "\""
            << ",\"k_weight_quant_mode\":\"" << dtype_label(b.wk.data_type())
            << "\""
            << ",\"k_weight_stride_used\":" << D
            << ",\"k_weight_head_dim\":" << Dh
            << ",\"k_weight_kv_heads\":" << Hkv
            << ",\"k_weight_sample_dims\":" << k_weight_sample
            << ",\"v_weight\":\"V\""
            << ",\"v_weight_layout_best\":\"" << v_weight_layout << "\""
            << ",\"v_weight_mae_row\":" << v_weight_mae_row
            << ",\"v_weight_mae_col\":" << v_weight_mae_col
            << ",\"v_weight_max_row\":" << v_weight_max_row
            << ",\"v_weight_max_col\":" << v_weight_max_col
            << ",\"v_weight_dtype\":\"" << dtype_label(b.wv.data_type()) << "\""
            << ",\"v_weight_quant_mode\":\"" << dtype_label(b.wv.data_type())
            << "\""
            << ",\"v_weight_stride_used\":" << D
            << ",\"v_weight_head_dim\":" << Dh
            << ",\"v_weight_kv_heads\":" << Hkv
            << ",\"v_weight_sample_dims\":" << v_weight_sample
            << ",\"k_layout_used\":\"row_major\""
            << ",\"v_layout_used\":\"row_major\""
            << ",\"q_ptr\":" << reinterpret_cast<uintptr_t>(q)
            << ",\"q_offset_bytes\":"
            << (static_cast<size_t>(token_index_used) * D +
                static_cast<size_t>(head) * Dh) *
                   sizeof(float)
            << ",\"k_base_ptr\":" << reinterpret_cast<uintptr_t>(cache_k)
            << ",\"v_base_ptr\":" << reinterpret_cast<uintptr_t>(cache_v)
            << ",\"k_head_base_ptr\":"
            << reinterpret_cast<uintptr_t>(k_head_base)
            << ",\"v_head_base_ptr\":"
            << reinterpret_cast<uintptr_t>(v_head_base) << ",\"k_pos_ptr\":"
            << reinterpret_cast<uintptr_t>(
                   k_head_base + static_cast<size_t>(pos_id_used) * Dh)
            << ",\"v_pos_ptr\":"
            << reinterpret_cast<uintptr_t>(
                   v_head_base + static_cast<size_t>(pos_id_used) * Dh)
            << ",\"kv_layer_stride_bytes\":" << kv_layer_stride_bytes
            << ",\"kv_head_stride_bytes\":" << kv_head_stride_bytes
            << ",\"kv_pos_stride_bytes\":" << kv_pos_stride_bytes;

        if (!norm_in_host.empty()) {
          oss << append_span("attn_norm_in", norm_in_host.data(),
                             norm_in_host.size());
        } else {
          oss << ",\"attn_norm_in\":[]";
        }
        if (norm_out_valid && !norm_out_host.empty()) {
          oss << append_span("attn_norm_out", norm_out_host.data(),
                             norm_out_host.size());
        } else {
          oss << ",\"attn_norm_out\":[]";
        }
        oss << append_span("q", q_host.data(), q_host.size());
        oss << append_span("k", k_vec, Dh);
        oss << append_span("v", v_vec, Dh);
        oss << append_span("qk", qk_row.data(), qk_row.size());
        oss << append_span("softmax", soft_row.data(), soft_row.size());
        oss << append_span("pv", pv_row.data(), pv_row.size());
        oss << append_span("attn_out", attn_host.data(), attn_host.size());
        oss << "}";
        append_line(out, oss.str());
      }
    }
  }

  if (trace_layer) {
    layer_tracer_.trace_tensor("attn_out", trace_step_,
                               static_cast<int>(layer_idx), hip_stream,
                               attn_out, n_x);
  }

  if (TRACE_ON(layer_idx)) {
    launch_debug_tensor_stats(hip_stream, "L.attn.attn_out", attn_out, n_x);
  }

  if (stage_layer) {
    stage_trace_tensor("attn_out", stage_phase, stage_prompt_id, layer_idx,
                       static_cast<uint32_t>(trace_step_), stage_pos_id,
                       static_cast<uint32_t>(seq_len), stage_tokens_total,
                       attn_out, D, stage_token_index, hip_stream);
  }
  if (post_wo_layer) {
    post_wo_trace_tensor("attn_out", stage_phase, stage_prompt_id, layer_idx,
                         static_cast<uint32_t>(trace_step_), stage_pos_id,
                         static_cast<uint32_t>(seq_len), stage_tokens_total,
                         attn_out, D, stage_token_index,
                         activations_.attn_out.size(), hip_stream);
  }

  gcore::compute::GretaCompute::set_op_label(is_decode_step ? "attn_o_decode"
                                                            : "attn_o_prefill");
  CHECK_GRETA(
      gcore::compute::GretaCompute::gemm(stream_, &activations_.attn_out, &b.wo,
                                         &activations_.mlp_out, S, D, D),
      "GEMM O");
  gcore::compute::GretaCompute::set_op_label(nullptr);

  if (stage_layer) {
    stage_trace_tensor("wo_out", stage_phase, stage_prompt_id, layer_idx,
                       static_cast<uint32_t>(trace_step_), stage_pos_id,
                       static_cast<uint32_t>(seq_len), stage_tokens_total,
                       mlp_out, D, stage_token_index, hip_stream);
  }
  if (post_wo_layer) {
    post_wo_trace_tensor("wo_out", stage_phase, stage_prompt_id, layer_idx,
                         static_cast<uint32_t>(trace_step_), stage_pos_id,
                         static_cast<uint32_t>(seq_len), stage_tokens_total,
                         mlp_out, D, stage_token_index,
                         activations_.mlp_out.size(), hip_stream);
  }

  if (stage_layer && trace_wo_w_verify_enabled()) {
    const char *out = stage_trace_out_path();
    if (out && *out) {
      const uint32_t wo_weight_sample =
          std::min<uint32_t>(attn_vacc_dims_sample(), static_cast<uint32_t>(D));
      std::string wo_weight_layout = "disabled";
      double wo_weight_mae_row = 0.0;
      double wo_weight_mae_col = 0.0;
      float wo_weight_max_row = 0.0f;
      float wo_weight_max_col = 0.0f;

      std::vector<float> attn_out_host;
      std::vector<float> wo_out_host;
      if (wo_weight_sample > 0) {
        attn_out_host.resize(D);
        wo_out_host.resize(D);
        const float *attn_vec =
            attn_out + static_cast<size_t>(stage_token_index) * D;
        const float *wo_vec =
            mlp_out + static_cast<size_t>(stage_token_index) * D;
        hipError_t err_a = hipMemcpy(attn_out_host.data(), attn_vec,
                                     D * sizeof(float), hipMemcpyDeviceToHost);
        hipError_t err_b = hipMemcpy(wo_out_host.data(), wo_vec,
                                     D * sizeof(float), hipMemcpyDeviceToHost);
        if (err_a != hipSuccess || err_b != hipSuccess) {
          attn_out_host.clear();
          wo_out_host.clear();
        }
      }

      if (!attn_out_host.empty() && !wo_out_host.empty() &&
          wo_weight_sample > 0) {
        static QkvWeightHostCache wo_cache;
        const size_t wo_elems = static_cast<size_t>(D) * D;
        if (ensure_qkv_weight_cache(b.wo, b.s_wo, b.sh_wo, wo_elems,
                                    &wo_cache)) {
          std::vector<float> wo_row(static_cast<size_t>(wo_weight_sample),
                                    0.0f);
          std::vector<float> wo_col(static_cast<size_t>(wo_weight_sample),
                                    0.0f);
          for (uint32_t j = 0; j < wo_weight_sample; ++j) {
            double sum_row = 0.0;
            double sum_col = 0.0;
            for (uint32_t i = 0; i < D; ++i) {
              const float x_i = attn_out_host[i];
              const size_t idx_row = static_cast<size_t>(j) * D + i;
              const size_t idx_col = static_cast<size_t>(i) * D + j;
              sum_row +=
                  static_cast<double>(x_i) *
                  static_cast<double>(read_weight_value(wo_cache, idx_row));
              sum_col +=
                  static_cast<double>(x_i) *
                  static_cast<double>(read_weight_value(wo_cache, idx_col));
            }
            float h_scale = 1.0f;
            if (!wo_cache.head_scales.empty() && Dh > 0) {
              const size_t h_idx = j / Dh;
              if (h_idx < wo_cache.head_scales.size()) {
                h_scale = wo_cache.head_scales[h_idx];
              }
            }
            wo_row[j] = static_cast<float>(sum_row * h_scale);
            wo_col[j] = static_cast<float>(sum_col * h_scale);
          }
          mae_max_f32(wo_row.data(), wo_out_host.data(), wo_weight_sample,
                      &wo_weight_mae_row, &wo_weight_max_row);
          mae_max_f32(wo_col.data(), wo_out_host.data(), wo_weight_sample,
                      &wo_weight_mae_col, &wo_weight_max_col);
          wo_weight_layout =
              (wo_weight_mae_row <= wo_weight_mae_col) ? "row" : "col";
        } else {
          wo_weight_layout = "error";
        }
      }

      const char *wo_layout_env = std::getenv("GRETA_WO_LAYOUT_FORCE");
      const char *wo_layout_used =
          (wo_layout_env && *wo_layout_env) ? wo_layout_env : "auto";
      std::ostringstream oss;
      oss << "{\"event\":\"wo_verify\"";
      if (stage_prompt_id && *stage_prompt_id)
        oss << ",\"prompt_id\":\"" << stage_prompt_id << "\"";
      oss << ",\"phase\":\"" << stage_phase << "\""
          << ",\"layer\":" << layer_idx << ",\"head\":0"
          << ",\"seq_len\":" << seq_len << ",\"pos_id\":" << stage_pos_id
          << ",\"token_index\":" << stage_token_index
          << ",\"sample_n\":" << wo_weight_sample << ",\"wo_layout_best\":\""
          << wo_weight_layout << "\""
          << ",\"wo_layout_used\":\"" << wo_layout_used << "\""
          << ",\"wo_mae_row\":" << wo_weight_mae_row
          << ",\"wo_mae_col\":" << wo_weight_mae_col
          << ",\"wo_max_row\":" << wo_weight_max_row
          << ",\"wo_max_col\":" << wo_weight_max_col
          << ",\"wo_weight_dtype\":\"" << dtype_label(b.wo.data_type()) << "\""
          << ",\"wo_weight_quant_mode\":\"" << dtype_label(b.wo.data_type())
          << "\""
          << ",\"wo_weight_stride_used\":" << D
          << ",\"wo_weight_head_dim\":" << Dh
          << ",\"wo_weight_kv_heads\":" << Hkv
          << ",\"wo_w_ptr\":" << reinterpret_cast<uintptr_t>(b.wo.data())
          << ",\"wo_w_bytes\":" << b.wo.size() << "}";
      append_line(out, oss.str());
    }
  }

  CHECK_HIP_KERNEL(launch_add(hip_stream, x, mlp_out, x, S * D),
                   "Residual (Attn)");

  if (stage_layer) {
    stage_trace_tensor("x_after_attn", stage_phase, stage_prompt_id, layer_idx,
                       static_cast<uint32_t>(trace_step_), stage_pos_id,
                       static_cast<uint32_t>(seq_len), stage_tokens_total, x, D,
                       stage_token_index, hip_stream);
  }
  if (post_wo_layer) {
    post_wo_trace_tensor("x_after_attn", stage_phase, stage_prompt_id,
                         layer_idx, static_cast<uint32_t>(trace_step_),
                         stage_pos_id, static_cast<uint32_t>(seq_len),
                         stage_tokens_total, x, D, stage_token_index,
                         activations_.x.size(), hip_stream);
  }

  CHECK_HIP_KERNEL(
      launch_rmsnorm_naive(hip_stream, x,
                           static_cast<const float *>(b.ffn_norm.data()),
                           norm_out, S, D, config_.rms_eps),
      "RMSNorm (FFN)");

  if (trace_rmsnorm_enabled() && stage_phase &&
      rmsnorm_phase_enabled(stage_phase) &&
      rmsnorm_layer_selected(layer_idx, config_.num_layers) &&
      rmsnorm_out_path()) {
    trace_rmsnorm(stage_phase, stage_prompt_id, layer_idx, config_.num_layers,
                  static_cast<uint32_t>(trace_step_), stage_pos_id,
                  static_cast<uint32_t>(seq_len), stage_tokens_total, x,
                  norm_out, D, stage_token_index, activations_.x.size(),
                  activations_.norm_out.size(), b.ffn_norm, config_.rms_eps,
                  hip_stream);
  }

  if (stage_layer) {
    stage_trace_tensor("ffn_norm", stage_phase, stage_prompt_id, layer_idx,
                       static_cast<uint32_t>(trace_step_), stage_pos_id,
                       static_cast<uint32_t>(seq_len), stage_tokens_total,
                       norm_out, D, stage_token_index, hip_stream);
  }
  if (post_wo_layer) {
    post_wo_trace_tensor("ffn_norm", stage_phase, stage_prompt_id, layer_idx,
                         static_cast<uint32_t>(trace_step_), stage_pos_id,
                         static_cast<uint32_t>(seq_len), stage_tokens_total,
                         norm_out, D, stage_token_index,
                         activations_.norm_out.size(), hip_stream);
  }

  if (trace_layer) {
    layer_tracer_.trace_tensor("ffn_norm", trace_step_,
                               static_cast<int>(layer_idx), hip_stream,
                               norm_out, n_x);
  }

  if (TRACE_ON(layer_idx)) {
    launch_debug_tensor_stats(hip_stream, "L.ffn_rmsnorm.norm_out", norm_out,
                              n_x);
  }

  const char *use_fused_ffn_env = std::getenv("GRETA_USE_FUSED_FFN");
  bool use_fused_ffn =
      (use_fused_ffn_env && std::string(use_fused_ffn_env) == "1") && (S == 1);

  if (use_fused_ffn) {
    CHECK_HIP_KERNEL(
        launch_fused_ffn_front_f16(
            hip_stream, norm_out, static_cast<const __half *>(b.w1.data()),
            static_cast<const __half *>(b.w3.data()), mlp_gate, D, hidden_dim),
        "Fused FFN Front");
  } else {
    CHECK_GRETA(gcore::compute::GretaCompute::gemm(
                    stream_, &activations_.norm_out, &b.w1,
                    &activations_.mlp_gate, S, hidden_dim, D),
                "GEMM W1");
    CHECK_GRETA(gcore::compute::GretaCompute::gemm(
                    stream_, &activations_.norm_out, &b.w3,
                    &activations_.mlp_up, S, hidden_dim, D),
                "GEMM W3");

    if (trace_layer) {
      layer_tracer_.trace_tensor("mlp_gate", trace_step_,
                                 static_cast<int>(layer_idx), hip_stream,
                                 mlp_gate, n_mlp);
      layer_tracer_.trace_tensor("mlp_up", trace_step_,
                                 static_cast<int>(layer_idx), hip_stream,
                                 mlp_up, n_mlp);
    }

    if (TRACE_ON(layer_idx)) {
      launch_debug_tensor_stats(hip_stream, "L.ffn.gate.nonfused", mlp_gate,
                                n_mlp);
      launch_debug_tensor_stats(hip_stream, "L.ffn.up.nonfused", mlp_up, n_mlp);
    }

    CHECK_HIP_KERNEL(
        launch_silu(hip_stream, mlp_gate, mlp_gate, S * hidden_dim), "SiLU");
    CHECK_HIP_KERNEL(
        launch_mul(hip_stream, mlp_gate, mlp_up, mlp_gate, S * hidden_dim),
        "Mul");
  }

  if (TRACE_ON(layer_idx)) {
    launch_debug_tensor_stats(hip_stream, "L.ffn.gate_after_mul", mlp_gate,
                              n_mlp);
  }

  CHECK_GRETA(gcore::compute::GretaCompute::gemm(
                  stream_, &activations_.mlp_gate, &b.w2, &activations_.mlp_out,
                  S, D, hidden_dim),
              "GEMM W2");

  if (trace_layer) {
    layer_tracer_.trace_tensor("mlp_out", trace_step_,
                               static_cast<int>(layer_idx), hip_stream, mlp_out,
                               n_x);
  }

  if (TRACE_ON(layer_idx)) {
    launch_debug_tensor_stats(hip_stream, "L.ffn.out", mlp_out, n_x);
  }

  if (stage_layer) {
    stage_trace_tensor("mlp_out", stage_phase, stage_prompt_id, layer_idx,
                       static_cast<uint32_t>(trace_step_), stage_pos_id,
                       static_cast<uint32_t>(seq_len), stage_tokens_total,
                       mlp_out, D, stage_token_index, hip_stream);
  }
  if (post_wo_layer) {
    post_wo_trace_tensor("mlp_out", stage_phase, stage_prompt_id, layer_idx,
                         static_cast<uint32_t>(trace_step_), stage_pos_id,
                         static_cast<uint32_t>(seq_len), stage_tokens_total,
                         mlp_out, D, stage_token_index,
                         activations_.mlp_out.size(), hip_stream);
  }

  CHECK_HIP_KERNEL(launch_add(hip_stream, x, mlp_out, x, S * D),
                   "Residual (FFN)");

  if (stage_layer) {
    stage_trace_tensor("x_after_mlp", stage_phase, stage_prompt_id, layer_idx,
                       static_cast<uint32_t>(trace_step_), stage_pos_id,
                       static_cast<uint32_t>(seq_len), stage_tokens_total, x, D,
                       stage_token_index, hip_stream);
    if (stage_trace_point_enabled("x_out")) {
      stage_trace_tensor("x_out", stage_phase, stage_prompt_id, layer_idx,
                         static_cast<uint32_t>(trace_step_), stage_pos_id,
                         static_cast<uint32_t>(seq_len), stage_tokens_total, x,
                         D, stage_token_index, hip_stream);
    }
  }
  if (post_wo_layer) {
    post_wo_trace_tensor("x_after_mlp", stage_phase, stage_prompt_id, layer_idx,
                         static_cast<uint32_t>(trace_step_), stage_pos_id,
                         static_cast<uint32_t>(seq_len), stage_tokens_total, x,
                         D, stage_token_index, activations_.x.size(),
                         hip_stream);
    post_wo_trace_tensor("x_out", stage_phase, stage_prompt_id, layer_idx,
                         static_cast<uint32_t>(trace_step_), stage_pos_id,
                         static_cast<uint32_t>(seq_len), stage_tokens_total, x,
                         D, stage_token_index, activations_.x.size(),
                         hip_stream);
  }

  if (PROFILE_ON()) {
    stop->record(stream_);
    stream_->synchronize();
    float ms = stop->elapsed_time_since(start);
    printf("[PROFILE] Layer %zu total: %.3f ms\n", layer_idx, ms);
    delete start;
    delete stop;
  }
  if (trace_layer) {
    layer_tracer_.trace_tensor("x_out", trace_step_,
                               static_cast<int>(layer_idx), hip_stream, x, n_x);
  }

  if (PROFILE_ON()) {
    stop->record(stream_);
    stream_->synchronize();
    float ms = stop->elapsed_time_since(start);
    printf("[PROFILE] Layer %zu total: %.3f ms\n", layer_idx, ms);
    delete start;
    delete stop;
  }

  if (TRACE_ON(layer_idx)) {
    launch_debug_tensor_stats(hip_stream, "L.residual.x", x, n_x);
  }

  const bool trace_attn_verify = trace_attn_decode_verify_enabled();
  const bool trace_attn_ref = trace_attn_ref_enabled();
  const bool trace_attn_softmax = trace_attn_softmax_enabled();
  const bool trace_attn_vacc = trace_attn_vacc_enabled();
  const bool trace_v_addr = trace_v_addr_enabled();
  const bool trace_attn_layer =
      attn_trace_layer_selected(layer_idx, config_.num_layers);
  const bool trace_softmax_layer =
      attn_softmax_layer_selected(layer_idx, config_.num_layers);
  const bool trace_vacc_layer =
      attn_softmax_layer_selected(layer_idx, config_.num_layers);
  const bool trace_v_addr_layer =
      attn_softmax_layer_selected(layer_idx, config_.num_layers);
  if ((trace_attn_verify || trace_attn_ref || trace_attn_softmax ||
       trace_attn_vacc || trace_v_addr) &&
      seq_len == 1 && trace_step_ == 1 &&
      (trace_attn_layer || trace_softmax_layer || trace_vacc_layer ||
       trace_v_addr_layer)) {
    const char *out = attn_trace_out_path();
    const char *out_ref = attn_ref_out_path();
    const char *out_softmax = attn_softmax_out_path();
    const char *out_vacc = attn_vacc_out_path();
    const char *out_v_addr = v_addr_out_path();
    if ((trace_attn_verify && out && *out) || trace_attn_ref ||
        trace_attn_softmax || trace_attn_vacc ||
        (trace_v_addr && out_v_addr && *out_v_addr)) {
      const uint32_t point_mask =
          attn_point_mask_from_list(std::getenv("GRETA_TRACE_ATTN_POINTS"));
      const uint32_t seq_len_used = static_cast<uint32_t>(seq_start + 1);
      const uint32_t pos_id_used = static_cast<uint32_t>(seq_start);
      const uint32_t max_seq_len = static_cast<uint32_t>(config_.max_seq_len);

      std::vector<float> q_host;
      std::vector<float> attn_out_host;
      std::vector<float> x_out_host;
      if (trace_attn_softmax || trace_attn_vacc ||
          (point_mask & static_cast<uint32_t>(AttnTracePoint::Q))) {
        q_host.resize(D);
        hipMemcpy(q_host.data(), q, D * sizeof(float), hipMemcpyDeviceToHost);
      }
      if (trace_attn_vacc ||
          (point_mask & static_cast<uint32_t>(AttnTracePoint::ATTN_OUT))) {
        attn_out_host.resize(D);
        hipMemcpy(attn_out_host.data(), attn_out, D * sizeof(float),
                  hipMemcpyDeviceToHost);
      }
      if (point_mask & static_cast<uint32_t>(AttnTracePoint::X_OUT)) {
        x_out_host.resize(D);
        hipMemcpy(x_out_host.data(), x, D * sizeof(float),
                  hipMemcpyDeviceToHost);
      }

      const size_t kv_layer_stride_elems =
          static_cast<size_t>(config_.max_seq_len) * Hkv * Dh;
      const size_t kv_layer_stride_bytes =
          kv_layer_stride_elems * sizeof(float);
      const size_t kv_layer_offset_elems =
          static_cast<size_t>(layer_idx) * kv_layer_stride_elems;
      const float *k_cache_layer =
          static_cast<const float *>(activations_.kv_cache_k.data()) +
          kv_layer_offset_elems;
      const float *v_cache_layer =
          static_cast<const float *>(activations_.kv_cache_v.data()) +
          kv_layer_offset_elems;

      const uint32_t trace_head = attn_softmax_head();
      const uint32_t group = (Hkv > 0) ? (Hq / Hkv) : 0;
      const uint32_t trace_kv_head =
          (group > 0 && trace_head < Hq) ? (trace_head / group) : 0;
      const size_t kv_head_stride_elems =
          static_cast<size_t>(config_.max_seq_len) * Dh;
      const size_t kv_head_stride_bytes = kv_head_stride_elems * sizeof(float);

      std::vector<float> k_cache_host;
      std::vector<float> v_cache_host;
      bool kv_head_only = false;
      bool need_kv_full =
          attn_decode_ref_enabled() || trace_attn_ref || trace_attn_verify ||
          (point_mask & static_cast<uint32_t>(AttnTracePoint::K)) ||
          (point_mask & static_cast<uint32_t>(AttnTracePoint::V));
      bool need_kv_head = trace_attn_softmax || trace_attn_vacc || trace_v_addr;
      if (need_kv_full) {
        k_cache_host.resize(kv_layer_stride_elems);
        v_cache_host.resize(kv_layer_stride_elems);
        hipMemcpy(k_cache_host.data(), k_cache_layer, kv_layer_stride_bytes,
                  hipMemcpyDeviceToHost);
        hipMemcpy(v_cache_host.data(), v_cache_layer, kv_layer_stride_bytes,
                  hipMemcpyDeviceToHost);
      } else if (need_kv_head && trace_kv_head < Hkv) {
        kv_head_only = true;
        k_cache_host.resize(kv_head_stride_elems);
        v_cache_host.resize(kv_head_stride_elems);
        const float *k_head_dev =
            k_cache_layer + trace_kv_head * kv_head_stride_elems;
        const float *v_head_dev =
            v_cache_layer + trace_kv_head * kv_head_stride_elems;
        hipMemcpy(k_cache_host.data(), k_head_dev, kv_head_stride_bytes,
                  hipMemcpyDeviceToHost);
        hipMemcpy(v_cache_host.data(), v_head_dev, kv_head_stride_bytes,
                  hipMemcpyDeviceToHost);
      }

      F32Stats q_stats{};
      uint64_t q_hash = 0;
      if (!q_host.empty()) {
        q_stats = stats_f32(q_host.data(), q_host.size());
        q_hash = hash_f32(q_host.data(), q_host.size());
      }

      F32Stats k_stats{};
      F32Stats v_stats{};
      uint64_t k_hash = 0;
      uint64_t v_hash = 0;
      if (!k_cache_host.empty()) {
        if (kv_head_only) {
          k_stats = stats_f32(k_cache_host.data(), k_cache_host.size());
          k_hash = hash_f32(k_cache_host.data(), k_cache_host.size());
        } else {
          stats_hash_kv_subset(k_cache_host.data(), Hkv, Dh, max_seq_len,
                               seq_len_used, &k_stats, &k_hash);
        }
      }
      if (!v_cache_host.empty()) {
        if (kv_head_only) {
          v_stats = stats_f32(v_cache_host.data(), v_cache_host.size());
          v_hash = hash_f32(v_cache_host.data(), v_cache_host.size());
        } else {
          stats_hash_kv_subset(v_cache_host.data(), Hkv, Dh, max_seq_len,
                               seq_len_used, &v_stats, &v_hash);
        }
      }

      F32Stats attn_stats{};
      uint64_t attn_hash = 0;
      if (!attn_out_host.empty()) {
        attn_stats = stats_f32(attn_out_host.data(), attn_out_host.size());
        attn_hash = hash_f32(attn_out_host.data(), attn_out_host.size());
      }

      F32Stats x_stats{};
      uint64_t x_hash = 0;
      if (!x_out_host.empty()) {
        x_stats = stats_f32(x_out_host.data(), x_out_host.size());
        x_hash = hash_f32(x_out_host.data(), x_out_host.size());
      }

      std::vector<float> attn_ref;
      uint64_t attn_ref_hash = 0;
      double attn_mae = 0.0;
      if (attn_decode_ref_enabled() && !q_host.empty() &&
          !k_cache_host.empty() && !v_cache_host.empty() &&
          !attn_out_host.empty()) {
        const float scale = 1.0f / sqrtf(static_cast<float>(Dh));
        compute_attention_ref_fp32(q_host.data(), k_cache_host.data(),
                                   v_cache_host.data(), Hq, Hkv, Dh,
                                   seq_len_used, max_seq_len, scale, attn_ref);
        if (!attn_ref.empty()) {
          attn_ref_hash = hash_f32(attn_ref.data(), attn_ref.size());
          attn_mae = mae_f32(attn_out_host.data(), attn_ref.data(),
                             attn_out_host.size());
        }
      }

      const uint32_t kv_head = 0;
      const size_t kv_pos = pos_id_used;
      const size_t k_read_offset_elems =
          kv_head * kv_head_stride_elems + kv_pos * Dh;
      const size_t v_read_offset_elems = k_read_offset_elems;
      const size_t k_read_offset_bytes = k_read_offset_elems * sizeof(float);
      const size_t v_read_offset_bytes = v_read_offset_elems * sizeof(float);
      bool kv_invariant_ok = true;
      std::string kv_error;
      if (env_flag("GRETA_TRACE_KV_INVARIANTS")) {
        if (kv_pos != pos_id_used) {
          kv_invariant_ok = false;
          kv_error = "kv_pos_mismatch";
        } else if (kv_pos >= max_seq_len) {
          kv_invariant_ok = false;
          kv_error = "kv_pos_oob";
        } else if (k_read_offset_bytes + Dh * sizeof(float) >
                   kv_layer_stride_bytes) {
          kv_invariant_ok = false;
          kv_error = "k_offset_oob";
        } else if (v_read_offset_bytes + Dh * sizeof(float) >
                   kv_layer_stride_bytes) {
          kv_invariant_ok = false;
          kv_error = "v_offset_oob";
        }
        if (activations_.kv_cache_k.data() == activations_.kv_cache_v.data()) {
          kv_invariant_ok = false;
          kv_error = kv_error.empty() ? "kv_overlap" : kv_error;
        }
      }

      if (trace_attn_verify && trace_attn_layer && out && *out) {
        std::ostringstream oss;
        oss << "{\"event\":\"attn_decode_verify\""
            << ",\"phase\":\"decode0\""
            << ",\"layer\":" << layer_idx
            << ",\"seq_len_used\":" << seq_len_used
            << ",\"pos_id_used\":" << pos_id_used << ",\"kernel_path\":\""
            << (use_fused_attn ? "fused" : "manual") << "\""
            << ",\"matmul_route\":\""
            << (std::getenv("GRETA_FORCE_ATTN_DECODE_MATMUL")
                    ? std::getenv("GRETA_FORCE_ATTN_DECODE_MATMUL")
                    : "auto")
            << "\""
            << ",\"num_heads\":" << Hq << ",\"num_heads_kv\":" << Hkv
            << ",\"head_dim\":" << Dh << ",\"q_hash\":" << q_hash
            << ",\"q_min\":" << q_stats.min << ",\"q_max\":" << q_stats.max
            << ",\"q_mean\":" << q_stats.mean << ",\"k_hash\":" << k_hash
            << ",\"k_min\":" << k_stats.min << ",\"k_max\":" << k_stats.max
            << ",\"k_mean\":" << k_stats.mean << ",\"v_hash\":" << v_hash
            << ",\"v_min\":" << v_stats.min << ",\"v_max\":" << v_stats.max
            << ",\"v_mean\":" << v_stats.mean
            << ",\"attn_out_hash\":" << attn_hash
            << ",\"attn_out_min\":" << attn_stats.min
            << ",\"attn_out_max\":" << attn_stats.max
            << ",\"attn_out_mean\":" << attn_stats.mean
            << ",\"attn_out_ref_hash\":" << attn_ref_hash
            << ",\"attn_out_mae\":" << attn_mae << ",\"x_out_hash\":" << x_hash
            << ",\"x_out_min\":" << x_stats.min
            << ",\"x_out_max\":" << x_stats.max
            << ",\"x_out_mean\":" << x_stats.mean << ",\"kv_base_ptr_k\":"
            << reinterpret_cast<uintptr_t>(activations_.kv_cache_k.data())
            << ",\"kv_base_ptr_v\":"
            << reinterpret_cast<uintptr_t>(activations_.kv_cache_v.data())
            << ",\"kv_layer_stride_bytes\":" << kv_layer_stride_bytes
            << ",\"kv_pos\":" << kv_pos
            << ",\"k_read_offset_bytes\":" << k_read_offset_bytes
            << ",\"v_read_offset_bytes\":" << v_read_offset_bytes
            << ",\"k_write_offset_bytes\":" << k_read_offset_bytes
            << ",\"v_write_offset_bytes\":" << v_read_offset_bytes
            << ",\"kv_invariant_ok\":" << (kv_invariant_ok ? "true" : "false")
            << ",\"kv_error\":\"" << kv_error << "\""
            << "}";
        append_line(out, oss.str());
      }

      if (trace_attn_ref && trace_attn_layer && out_ref && *out_ref &&
          !q_host.empty() && !k_cache_host.empty() && !v_cache_host.empty() &&
          !attn_out_host.empty()) {
        const double scale_d = 1.0 / std::sqrt(static_cast<double>(Dh));
        std::vector<float> attn_ref_hp;
        compute_attention_ref_fp64(
            q_host.data(), k_cache_host.data(), v_cache_host.data(), Hq, Hkv,
            Dh, seq_len_used, max_seq_len, scale_d, attn_ref_hp);

        std::vector<float> attn_ref_accum;
        const AttnAccumMode mode = attn_accum_mode();
        if (mode == AttnAccumMode::Fp16) {
          compute_attention_ref_fp16_accum(
              q_host.data(), k_cache_host.data(), v_cache_host.data(), Hq, Hkv,
              Dh, seq_len_used, max_seq_len,
              1.0f / std::sqrt(static_cast<float>(Dh)), attn_ref_accum);
        } else {
          compute_attention_ref_fp32(
              q_host.data(), k_cache_host.data(), v_cache_host.data(), Hq, Hkv,
              Dh, seq_len_used, max_seq_len,
              1.0f / std::sqrt(static_cast<float>(Dh)), attn_ref_accum);
        }

        double ref_mae = 0.0;
        float ref_max_diff = 0.0f;
        double accum_mae = 0.0;
        float accum_max_diff = 0.0f;
        uint64_t ref_hp_hash = 0;
        uint64_t ref_accum_hash = 0;

        if (!attn_ref_hp.empty()) {
          ref_hp_hash = hash_f32(attn_ref_hp.data(), attn_ref_hp.size());
          mae_max_f32(attn_out_host.data(), attn_ref_hp.data(),
                      attn_out_host.size(), &ref_mae, &ref_max_diff);
        }
        if (!attn_ref_accum.empty() && !attn_ref_hp.empty()) {
          ref_accum_hash =
              hash_f32(attn_ref_accum.data(), attn_ref_accum.size());
          mae_max_f32(attn_ref_accum.data(), attn_ref_hp.data(),
                      attn_ref_accum.size(), &accum_mae, &accum_max_diff);
        }

        const std::vector<double> per_head =
            per_head_mae_f32(attn_out_host.data(), attn_ref_hp.data(), Hq, Dh);

        std::ostringstream oss;
        oss << "{\"event\":\"attn_precision_ref\""
            << ",\"phase\":\"decode0\""
            << ",\"layer\":" << layer_idx
            << ",\"seq_len_used\":" << seq_len_used
            << ",\"pos_id_used\":" << pos_id_used << ",\"attn_accum_mode\":\""
            << (mode == AttnAccumMode::Fp16 ? "fp16" : "fp32") << "\""
            << ",\"attn_out_hash\":" << attn_hash
            << ",\"attn_ref_hp_hash\":" << ref_hp_hash
            << ",\"attn_ref_accum_hash\":" << ref_accum_hash
            << ",\"attn_ref_mae\":" << ref_mae
            << ",\"attn_ref_max_diff\":" << ref_max_diff
            << ",\"attn_accum_mae\":" << accum_mae
            << ",\"attn_accum_max_diff\":" << accum_max_diff
            << ",\"per_head_mae\":[";
        for (size_t i = 0; i < per_head.size(); ++i) {
          if (i)
            oss << ",";
          oss << per_head[i];
        }
        oss << "]";
        oss << "}";
        append_line(out_ref, oss.str());
      }

      bool softmax_window_ok = false;
      uint32_t window_start = 0;
      uint32_t window_len = 0;
      uint32_t seq_len_trace = 0;
      std::vector<float> qk_gpu;
      std::vector<float> softmax_gpu;
      std::vector<float> stats_gpu;
      std::vector<double> qk_cpu;
      std::vector<double> softmax_cpu;
      std::vector<double> qk_full;
      std::vector<float> q_sample;
      std::vector<float> k_sample;
      double max_qk = -INFINITY;
      double sumexp = 0.0;
      double inv_sum = 0.0;
      double top1_prob = 0.0;
      double top2_prob = 0.0;
      double entropy = 0.0;
      double qk_mae = 0.0;
      double qk_max_diff = 0.0;
      double soft_mae = 0.0;
      double soft_max_diff = 0.0;
      uint64_t q_head_hash = 0;
      uint64_t k_head_hash = 0;
      bool qk_full_ok = false;

      const bool need_softmax_window =
          ((trace_attn_softmax && out_softmax && *out_softmax) ||
           (trace_attn_vacc && out_vacc && *out_vacc)) &&
          attn_softmax_layer_selected(layer_idx, config_.num_layers);

      const uint32_t head = attn_softmax_head();
      const float *q_head = (!q_host.empty() && head < Hq)
                                ? (q_host.data() + head * Dh)
                                : nullptr;
      const float *k_head = nullptr;
      if (!k_cache_host.empty() && trace_kv_head < Hkv) {
        k_head = kv_head_only ? k_cache_host.data()
                              : k_cache_host.data() +
                                    static_cast<size_t>(trace_kv_head) *
                                        config_.max_seq_len * Dh;
      }

      if (need_softmax_window && q_head && k_head) {
        const uint32_t window = attn_softmax_window();
        const uint32_t pos_id = pos_id_used;
        seq_len_trace = seq_len_used;
        window_start = (seq_len_trace > window) ? (seq_len_trace - window) : 0;
        window_len =
            seq_len_trace > window_start ? (seq_len_trace - window_start) : 0;
        if (window_len > 0) {
          using Usage = gcore::rt::hip::BufferUsage;
          gcore::rt::hip::Buffer qk_dev;
          gcore::rt::hip::Buffer softmax_dev;
          gcore::rt::hip::Buffer stats_dev;
          std::string err;
          bool alloc_ok =
              qk_dev.allocate(window_len * sizeof(float), Usage::DeviceOnly,
                              gcore::rt::GretaDataType::FP32, &err) &&
              softmax_dev.allocate(window_len * sizeof(float),
                                   Usage::DeviceOnly,
                                   gcore::rt::GretaDataType::FP32, &err) &&
              stats_dev.allocate(5 * sizeof(float), Usage::DeviceOnly,
                                 gcore::rt::GretaDataType::FP32, &err);
          if (!alloc_ok) {
            std::cerr << "[ATTN_SOFTMAX] alloc failed: " << err << "\n";
          } else {
            const size_t kv_layer_stride_elems =
                static_cast<size_t>(config_.max_seq_len) * Hkv * Dh;
            const size_t kv_layer_offset_elems =
                static_cast<size_t>(layer_idx) * kv_layer_stride_elems;
            const float *k_cache_layer =
                static_cast<const float *>(activations_.kv_cache_k.data()) +
                kv_layer_offset_elems;

            const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));

            gcore::rt::hip::kernels::launch_attn_softmax_trace(
                hip_stream, q, k_cache_layer, Hq, Hkv, Dh, seq_len_trace,
                static_cast<uint32_t>(config_.max_seq_len), head, window_start,
                window_len, scale, static_cast<float *>(qk_dev.data()),
                static_cast<float *>(softmax_dev.data()),
                static_cast<float *>(stats_dev.data()));

            hipStreamSynchronize(hip_stream);

            qk_gpu.assign(window_len, 0.0f);
            softmax_gpu.assign(window_len, 0.0f);
            stats_gpu.assign(5, 0.0f);
            hipMemcpy(qk_gpu.data(), qk_dev.data(), window_len * sizeof(float),
                      hipMemcpyDeviceToHost);
            hipMemcpy(softmax_gpu.data(), softmax_dev.data(),
                      window_len * sizeof(float), hipMemcpyDeviceToHost);
            hipMemcpy(stats_gpu.data(), stats_dev.data(),
                      stats_gpu.size() * sizeof(float), hipMemcpyDeviceToHost);

            q_head_hash = hash_f32(q_head, Dh);
            const float *k_pos = k_head + pos_id * Dh;
            k_head_hash = hash_f32(k_pos, Dh);

            const uint32_t sample_n = std::min<uint32_t>(8, Dh);
            q_sample.assign(q_head, q_head + sample_n);
            k_sample.assign(k_pos, k_pos + sample_n);

            qk_full.assign(seq_len_trace, 0.0);
            double second_qk = -INFINITY;
            for (uint32_t t = 0; t < seq_len_trace; ++t) {
              const float *k_t = k_head + t * Dh;
              double dot = 0.0;
              for (uint32_t d = 0; d < Dh; ++d) {
                dot += static_cast<double>(q_head[d]) *
                       static_cast<double>(k_t[d]);
              }
              double qk = dot * static_cast<double>(scale);
              qk_full[t] = qk;
              if (qk > max_qk) {
                second_qk = max_qk;
                max_qk = qk;
              } else if (qk > second_qk) {
                second_qk = qk;
              }
            }

            double sumexp_log = 0.0;
            for (uint32_t t = 0; t < seq_len_trace; ++t) {
              double e = std::exp(qk_full[t] - max_qk);
              sumexp += e;
              sumexp_log += e * (qk_full[t] - max_qk);
            }
            inv_sum = sumexp > 0.0 ? (1.0 / sumexp) : 0.0;
            top1_prob = inv_sum;
            top2_prob = (sumexp > 0.0 && std::isfinite(second_qk))
                            ? std::exp(second_qk - max_qk) * inv_sum
                            : 0.0;
            entropy =
                (sumexp > 0.0) ? (std::log(sumexp) - sumexp_log / sumexp) : 0.0;

            qk_cpu.assign(window_len, 0.0);
            softmax_cpu.assign(window_len, 0.0);
            for (uint32_t i = 0; i < window_len; ++i) {
              uint32_t t = window_start + i;
              if (t >= seq_len_trace)
                break;
              double qk = qk_full[t];
              qk_cpu[i] = qk;
              double p = (sumexp > 0.0) ? std::exp(qk - max_qk) * inv_sum : 0.0;
              softmax_cpu[i] = p;
            }

            for (uint32_t i = 0; i < window_len; ++i) {
              double dq = std::abs(static_cast<double>(qk_gpu[i]) - qk_cpu[i]);
              qk_mae += dq;
              if (dq > qk_max_diff)
                qk_max_diff = dq;
              double ds = std::abs(static_cast<double>(softmax_gpu[i]) -
                                   softmax_cpu[i]);
              soft_mae += ds;
              if (ds > soft_max_diff)
                soft_max_diff = ds;
            }
            qk_mae = (window_len > 0) ? (qk_mae / window_len) : 0.0;
            soft_mae = (window_len > 0) ? (soft_mae / window_len) : 0.0;
            softmax_window_ok = true;
            qk_full_ok = true;
          }
        }
      }

      if (trace_attn_softmax && out_softmax && *out_softmax &&
          softmax_window_ok) {
        const char *prompt_id = std::getenv("GRETA_TRACE_PROMPT_ID");
        std::ostringstream oss;
        oss << "{\"event\":\"attn_softmax_trace\"";
        if (prompt_id && *prompt_id) {
          oss << ",\"prompt_id\":\"" << prompt_id << "\"";
        }
        oss << ",\"phase\":\"decode0\""
            << ",\"layer\":" << layer_idx << ",\"head\":" << head
            << ",\"pos_id\":" << pos_id_used << ",\"seq_len\":" << seq_len_used
            << ",\"kv_heads\":" << Hkv << ",\"head_dim\":" << Dh
            << ",\"scale_used\":" << (1.0f / std::sqrt(static_cast<float>(Dh)))
            << ",\"q_hash\":" << q_head_hash << ",\"k_hash\":" << k_head_hash
            << ",\"q_sample\":[";
        for (size_t i = 0; i < q_sample.size(); ++i) {
          if (i)
            oss << ",";
          oss << q_sample[i];
        }
        oss << "],\"k_sample\":[";
        for (size_t i = 0; i < k_sample.size(); ++i) {
          if (i)
            oss << ",";
          oss << k_sample[i];
        }
        oss << "],\"qk_window_gpu\":[";
        for (size_t i = 0; i < qk_gpu.size(); ++i) {
          if (i)
            oss << ",";
          oss << qk_gpu[i];
        }
        oss << "],\"qk_window_cpu_fp64\":[";
        for (size_t i = 0; i < qk_cpu.size(); ++i) {
          if (i)
            oss << ",";
          oss << qk_cpu[i];
        }
        oss << "],\"qk_mae\":" << qk_mae << ",\"qk_max_diff\":" << qk_max_diff
            << ",\"softmax_gpu_stats\":{"
            << "\"max\":" << stats_gpu[0] << ",\"sumexp\":" << stats_gpu[1]
            << ",\"top1_prob\":" << stats_gpu[2]
            << ",\"top2_prob\":" << stats_gpu[3]
            << ",\"entropy\":" << stats_gpu[4] << "}"
            << ",\"softmax_cpu_stats\":{"
            << "\"max\":" << max_qk << ",\"sumexp\":" << sumexp
            << ",\"top1_prob\":" << top1_prob << ",\"top2_prob\":" << top2_prob
            << ",\"entropy\":" << entropy << "}"
            << ",\"softmax_window_gpu\":[";
        for (size_t i = 0; i < softmax_gpu.size(); ++i) {
          if (i)
            oss << ",";
          oss << softmax_gpu[i];
        }
        oss << "],\"softmax_window_cpu\":[";
        for (size_t i = 0; i < softmax_cpu.size(); ++i) {
          if (i)
            oss << ",";
          oss << softmax_cpu[i];
        }
        oss << "],\"softmax_mae\":" << soft_mae
            << ",\"softmax_max_diff\":" << soft_max_diff << "}";
        append_line(out_softmax, oss.str());
      }

      if (trace_attn_vacc && out_vacc && *out_vacc && softmax_window_ok &&
          !attn_out_host.empty() && head < Hq) {
        const uint32_t dims_sample = std::min(attn_vacc_dims_sample(), Dh);
        if (dims_sample > 0 && window_len > 0) {
          using Usage = gcore::rt::hip::BufferUsage;
          gcore::rt::hip::Buffer v_row_dev;
          gcore::rt::hip::Buffer v_col_dev;
          std::string err;
          const size_t v_bytes =
              static_cast<size_t>(window_len) * dims_sample * sizeof(float);
          bool alloc_ok =
              v_row_dev.allocate(v_bytes, Usage::DeviceOnly,
                                 gcore::rt::GretaDataType::FP32, &err) &&
              v_col_dev.allocate(v_bytes, Usage::DeviceOnly,
                                 gcore::rt::GretaDataType::FP32, &err);
          if (!alloc_ok) {
            std::cerr << "[ATTN_VACC] alloc failed: " << err << "\n";
          } else {
            const size_t kv_layer_stride_elems =
                static_cast<size_t>(config_.max_seq_len) * Hkv * Dh;
            const size_t kv_layer_offset_elems =
                static_cast<size_t>(layer_idx) * kv_layer_stride_elems;
            const float *v_cache_layer =
                static_cast<const float *>(activations_.kv_cache_v.data()) +
                kv_layer_offset_elems;

            gcore::rt::hip::kernels::launch_attn_vacc_vsample(
                hip_stream, v_cache_layer, Hq, Hkv, Dh, seq_len_used,
                static_cast<uint32_t>(config_.max_seq_len), head, window_start,
                window_len, dims_sample, static_cast<float *>(v_row_dev.data()),
                static_cast<float *>(v_col_dev.data()));
            hipStreamSynchronize(hip_stream);

            std::vector<float> v_row(window_len * dims_sample, 0.0f);
            std::vector<float> v_col(window_len * dims_sample, 0.0f);
            hipMemcpy(v_row.data(), v_row_dev.data(), v_bytes,
                      hipMemcpyDeviceToHost);
            hipMemcpy(v_col.data(), v_col_dev.data(), v_bytes,
                      hipMemcpyDeviceToHost);

            std::vector<float> pv_gpu_sample(dims_sample, 0.0f);
            const float *attn_head =
                attn_out_host.data() + static_cast<size_t>(head) * Dh;
            for (uint32_t d = 0; d < dims_sample; ++d) {
              pv_gpu_sample[d] = attn_head[d];
            }

            std::vector<double> pv_row(dims_sample, 0.0);
            std::vector<double> pv_col(dims_sample, 0.0);
            const float *v_head_host = nullptr;
            if (!v_cache_host.empty() && trace_kv_head < Hkv) {
              v_head_host = kv_head_only
                                ? v_cache_host.data()
                                : v_cache_host.data() +
                                      static_cast<size_t>(trace_kv_head) *
                                          kv_head_stride_elems;
            }
            bool pv_full = false;
            if (qk_full_ok && v_head_host && inv_sum > 0.0 &&
                seq_len_trace > 0) {
              pv_full = true;
              for (uint32_t t = 0; t < seq_len_trace; ++t) {
                double p = std::exp(qk_full[t] - max_qk) * inv_sum;
                const float *v_row_t =
                    v_head_host + static_cast<size_t>(t) * Dh;
                for (uint32_t d = 0; d < dims_sample; ++d) {
                  const float v_r = v_row_t[d];
                  const float v_c = v_head_host[d * max_seq_len + t];
                  pv_row[d] += p * static_cast<double>(v_r);
                  pv_col[d] += p * static_cast<double>(v_c);
                }
              }
            } else {
              for (uint32_t i = 0; i < window_len; ++i) {
                double p = (i < softmax_cpu.size()) ? softmax_cpu[i] : 0.0;
                for (uint32_t d = 0; d < dims_sample; ++d) {
                  const float v_r = v_row[i * dims_sample + d];
                  const float v_c = v_col[i * dims_sample + d];
                  pv_row[d] += p * static_cast<double>(v_r);
                  pv_col[d] += p * static_cast<double>(v_c);
                }
              }
            }

            double v_mae_row = 0.0;
            double v_mae_col = 0.0;
            double v_max_row = 0.0;
            double v_max_col = 0.0;
            for (uint32_t d = 0; d < dims_sample; ++d) {
              double dr =
                  std::abs(static_cast<double>(pv_gpu_sample[d]) - pv_row[d]);
              double dc =
                  std::abs(static_cast<double>(pv_gpu_sample[d]) - pv_col[d]);
              v_mae_row += dr;
              v_mae_col += dc;
              if (dr > v_max_row)
                v_max_row = dr;
              if (dc > v_max_col)
                v_max_col = dc;
            }
            v_mae_row = dims_sample > 0 ? (v_mae_row / dims_sample) : 0.0;
            v_mae_col = dims_sample > 0 ? (v_mae_col / dims_sample) : 0.0;

            const bool row_best = v_mae_row <= v_mae_col;
            const char *layout_best = row_best ? "row" : "col";
            const std::vector<double> &pv_best = row_best ? pv_row : pv_col;

            double pv_mae = 0.0;
            double pv_max_diff = 0.0;
            for (uint32_t d = 0; d < dims_sample; ++d) {
              double diff =
                  std::abs(static_cast<double>(pv_gpu_sample[d]) - pv_best[d]);
              pv_mae += diff;
              if (diff > pv_max_diff)
                pv_max_diff = diff;
            }
            pv_mae = dims_sample > 0 ? (pv_mae / dims_sample) : 0.0;

            const char *prompt_id = std::getenv("GRETA_TRACE_PROMPT_ID");
            std::ostringstream oss;
            oss << "{\"event\":\"attn_vacc_trace\"";
            if (prompt_id && *prompt_id) {
              oss << ",\"prompt_id\":\"" << prompt_id << "\"";
            }
            oss << ",\"phase\":\"decode0\""
                << ",\"layer\":" << layer_idx << ",\"head\":" << head
                << ",\"pos_id\":" << pos_id_used
                << ",\"seq_len\":" << seq_len_used << ",\"t\":" << pos_id_used
                << ",\"window_len\":" << window_len << ",\"head_dim\":" << Dh
                << ",\"dims_sample\":" << dims_sample << ",\"kv_heads\":" << Hkv
                << ",\"scale_used\":"
                << (1.0f / std::sqrt(static_cast<float>(Dh)))
                << ",\"p_window_gpu\":[";
            for (size_t i = 0; i < softmax_gpu.size(); ++i) {
              if (i)
                oss << ",";
              oss << softmax_gpu[i];
            }
            oss << "],\"v_window_gpu_sample\":[";
            for (uint32_t i = 0; i < window_len; ++i) {
              if (i)
                oss << ",";
              oss << "[";
              for (uint32_t d = 0; d < dims_sample; ++d) {
                if (d)
                  oss << ",";
                oss << v_row[i * dims_sample + d];
              }
              oss << "]";
            }
            oss << "],\"v_layout_probe\":{"
                << "\"v_layout_best\":\"" << layout_best << "\""
                << ",\"v_mae_row\":" << v_mae_row
                << ",\"v_mae_col\":" << v_mae_col
                << ",\"v_max_row\":" << v_max_row
                << ",\"v_max_col\":" << v_max_col << "}"
                << ",\"pv_cpu_fp64_sample\":[";
            for (uint32_t d = 0; d < dims_sample; ++d) {
              if (d)
                oss << ",";
              oss << pv_best[d];
            }
            oss << "],\"pv_gpu_sample\":[";
            for (uint32_t d = 0; d < dims_sample; ++d) {
              if (d)
                oss << ",";
              oss << pv_gpu_sample[d];
            }
            oss << "],\"pv_mae\":" << pv_mae
                << ",\"pv_max_diff\":" << pv_max_diff << ",\"pv_scope\":\""
                << (pv_full ? "full" : "window") << "\""
                << ",\"attn_out_scope\":\"per_head_concat\""
                << ",\"attn_out_gpu_sample\":[";
            for (uint32_t d = 0; d < dims_sample; ++d) {
              if (d)
                oss << ",";
              oss << pv_gpu_sample[d];
            }
            oss << "],\"attn_out_cpu_fp64_sample\":[";
            for (uint32_t d = 0; d < dims_sample; ++d) {
              if (d)
                oss << ",";
              oss << pv_best[d];
            }
            oss << "],\"attn_out_mae\":" << pv_mae
                << ",\"attn_out_max_diff\":" << pv_max_diff << "}";
            append_line(out_vacc, oss.str());
          }
        }
      }

      if (trace_v_addr && out_v_addr && *out_v_addr && trace_kv_head < Hkv) {
        const uint32_t dims_sample = std::min(attn_vacc_dims_sample(), Dh);
        const uint32_t pos_cur = pos_id_used;
        const uint32_t pos_prev = (pos_cur > 0) ? (pos_cur - 1) : pos_cur;
        const uint32_t pos_next =
            (pos_cur + 1 < max_seq_len) ? (pos_cur + 1) : pos_cur;
        const size_t kv_pos_stride_elems = Dh;
        const size_t kv_pos_stride_bytes = kv_pos_stride_elems * sizeof(float);

        const float *k_head_host = nullptr;
        const float *v_head_host = nullptr;
        if (!k_cache_host.empty() && !v_cache_host.empty()) {
          if (kv_head_only) {
            k_head_host = k_cache_host.data();
            v_head_host = v_cache_host.data();
          } else {
            k_head_host =
                k_cache_host.data() +
                static_cast<size_t>(trace_kv_head) * kv_head_stride_elems;
            v_head_host =
                v_cache_host.data() +
                static_cast<size_t>(trace_kv_head) * kv_head_stride_elems;
          }
        }

        auto row_at = [&](const float *base, uint32_t pos,
                          uint32_t d) -> float {
          if (!base)
            return 0.0f;
          return base[static_cast<size_t>(pos) * Dh + d];
        };
        auto col_at = [&](const float *base, uint32_t pos,
                          uint32_t d) -> float {
          if (!base)
            return 0.0f;
          return base[static_cast<size_t>(d) * max_seq_len + pos];
        };

        std::vector<float> k_cur_sample(dims_sample, 0.0f);
        std::vector<float> v_cur_sample(dims_sample, 0.0f);
        if (dims_sample > 0) {
          hipMemcpy(k_cur_sample.data(),
                    k + static_cast<size_t>(trace_kv_head) * Dh,
                    dims_sample * sizeof(float), hipMemcpyDeviceToHost);
          hipMemcpy(v_cur_sample.data(),
                    v + static_cast<size_t>(trace_kv_head) * Dh,
                    dims_sample * sizeof(float), hipMemcpyDeviceToHost);
        }

        std::vector<float> k_cache_row_pos(dims_sample, 0.0f);
        std::vector<float> k_cache_row_prev(dims_sample, 0.0f);
        std::vector<float> k_cache_row_next(dims_sample, 0.0f);
        std::vector<float> v_cache_row_pos(dims_sample, 0.0f);
        std::vector<float> v_cache_row_prev(dims_sample, 0.0f);
        std::vector<float> v_cache_row_next(dims_sample, 0.0f);
        std::vector<float> v_cache_col_pos(dims_sample, 0.0f);
        for (uint32_t d = 0; d < dims_sample; ++d) {
          k_cache_row_pos[d] = row_at(k_head_host, pos_cur, d);
          k_cache_row_prev[d] = row_at(k_head_host, pos_prev, d);
          k_cache_row_next[d] = row_at(k_head_host, pos_next, d);
          v_cache_row_pos[d] = row_at(v_head_host, pos_cur, d);
          v_cache_row_prev[d] = row_at(v_head_host, pos_prev, d);
          v_cache_row_next[d] = row_at(v_head_host, pos_next, d);
          v_cache_col_pos[d] = col_at(v_head_host, pos_cur, d);
        }

        auto mae_vec = [&](const std::vector<float> &a,
                           const std::vector<float> &b) -> double {
          if (a.size() != b.size() || a.empty())
            return 0.0;
          double sum = 0.0;
          for (size_t i = 0; i < a.size(); ++i) {
            sum +=
                std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
          }
          return sum / static_cast<double>(a.size());
        };

        const double mae_k_pos = mae_vec(k_cur_sample, k_cache_row_pos);
        const double mae_k_prev = mae_vec(k_cur_sample, k_cache_row_prev);
        const double mae_k_next = mae_vec(k_cur_sample, k_cache_row_next);
        const double mae_v_pos = mae_vec(v_cur_sample, v_cache_row_pos);
        const double mae_v_prev = mae_vec(v_cur_sample, v_cache_row_prev);
        const double mae_v_next = mae_vec(v_cur_sample, v_cache_row_next);
        const double mae_v_col = mae_vec(v_cur_sample, v_cache_col_pos);

        const uintptr_t k_base_ptr =
            reinterpret_cast<uintptr_t>(activations_.kv_cache_k.data());
        const uintptr_t v_base_ptr =
            reinterpret_cast<uintptr_t>(activations_.kv_cache_v.data());
        const uintptr_t k_layer_ptr =
            k_base_ptr + kv_layer_offset_elems * sizeof(float);
        const uintptr_t v_layer_ptr =
            v_base_ptr + kv_layer_offset_elems * sizeof(float);
        const uintptr_t k_head_ptr =
            k_layer_ptr +
            static_cast<uintptr_t>(trace_kv_head) * kv_head_stride_bytes;
        const uintptr_t v_head_ptr =
            v_layer_ptr +
            static_cast<uintptr_t>(trace_kv_head) * kv_head_stride_bytes;
        const uintptr_t k_pos_ptr =
            k_head_ptr + static_cast<uintptr_t>(pos_cur) * kv_pos_stride_bytes;
        const uintptr_t v_pos_ptr =
            v_head_ptr + static_cast<uintptr_t>(pos_cur) * kv_pos_stride_bytes;

        const char *prompt_id = std::getenv("GRETA_TRACE_PROMPT_ID");
        std::ostringstream oss;
        oss << "{\"event\":\"v_addr_trace\"";
        if (prompt_id && *prompt_id) {
          oss << ",\"prompt_id\":\"" << prompt_id << "\"";
        }
        oss << ",\"phase\":\"decode0\""
            << ",\"layer\":" << layer_idx << ",\"head\":" << head
            << ",\"kv_head\":" << trace_kv_head << ",\"pos_id\":" << pos_id_used
            << ",\"seq_len\":" << seq_len_used
            << ",\"tokens_total\":" << seq_len_used
            << ",\"kv_pos_used\":" << pos_cur << ",\"kv_pos_prev\":" << pos_prev
            << ",\"kv_pos_next\":" << pos_next
            << ",\"kv_layer_stride_bytes\":" << kv_layer_stride_bytes
            << ",\"kv_head_stride_bytes\":" << kv_head_stride_bytes
            << ",\"kv_pos_stride_bytes\":" << kv_pos_stride_bytes
            << ",\"k_base_ptr\":" << k_base_ptr
            << ",\"k_layer_ptr\":" << k_layer_ptr
            << ",\"k_head_ptr\":" << k_head_ptr
            << ",\"k_pos_ptr\":" << k_pos_ptr
            << ",\"v_base_ptr\":" << v_base_ptr
            << ",\"v_layer_ptr\":" << v_layer_ptr
            << ",\"v_head_ptr\":" << v_head_ptr
            << ",\"v_pos_ptr\":" << v_pos_ptr << ",\"k_elem_ptrs\":[";
        for (uint32_t d = 0; d < std::min<uint32_t>(4, Dh); ++d) {
          if (d)
            oss << ",";
          oss << (k_pos_ptr + static_cast<uintptr_t>(d) * sizeof(float));
        }
        oss << "],\"v_elem_ptrs\":[";
        for (uint32_t d = 0; d < std::min<uint32_t>(4, Dh); ++d) {
          if (d)
            oss << ",";
          oss << (v_pos_ptr + static_cast<uintptr_t>(d) * sizeof(float));
        }
        oss << "],\"k_cur_sample\":[";
        for (uint32_t d = 0; d < k_cur_sample.size(); ++d) {
          if (d)
            oss << ",";
          oss << k_cur_sample[d];
        }
        oss << "],\"v_cur_sample\":[";
        for (uint32_t d = 0; d < v_cur_sample.size(); ++d) {
          if (d)
            oss << ",";
          oss << v_cur_sample[d];
        }
        oss << "],\"k_cache_row_pos\":[";
        for (uint32_t d = 0; d < k_cache_row_pos.size(); ++d) {
          if (d)
            oss << ",";
          oss << k_cache_row_pos[d];
        }
        oss << "],\"k_cache_row_prev\":[";
        for (uint32_t d = 0; d < k_cache_row_prev.size(); ++d) {
          if (d)
            oss << ",";
          oss << k_cache_row_prev[d];
        }
        oss << "],\"k_cache_row_next\":[";
        for (uint32_t d = 0; d < k_cache_row_next.size(); ++d) {
          if (d)
            oss << ",";
          oss << k_cache_row_next[d];
        }
        oss << "],\"v_cache_row_pos\":[";
        for (uint32_t d = 0; d < v_cache_row_pos.size(); ++d) {
          if (d)
            oss << ",";
          oss << v_cache_row_pos[d];
        }
        oss << "],\"v_cache_row_prev\":[";
        for (uint32_t d = 0; d < v_cache_row_prev.size(); ++d) {
          if (d)
            oss << ",";
          oss << v_cache_row_prev[d];
        }
        oss << "],\"v_cache_row_next\":[";
        for (uint32_t d = 0; d < v_cache_row_next.size(); ++d) {
          if (d)
            oss << ",";
          oss << v_cache_row_next[d];
        }
        oss << "],\"v_cache_col_pos\":[";
        for (uint32_t d = 0; d < v_cache_col_pos.size(); ++d) {
          if (d)
            oss << ",";
          oss << v_cache_col_pos[d];
        }
        oss << "],\"mae_k_pos\":" << mae_k_pos
            << ",\"mae_k_prev\":" << mae_k_prev
            << ",\"mae_k_next\":" << mae_k_next
            << ",\"mae_v_pos\":" << mae_v_pos
            << ",\"mae_v_prev\":" << mae_v_prev
            << ",\"mae_v_next\":" << mae_v_next
            << ",\"mae_v_col\":" << mae_v_col << "}";
        append_line(out_v_addr, oss.str());
      }
    }
  }

  if (env_flag("GRETA_TRACE_LAYER_DELTA") && seq_len == 1 &&
      (layer_idx == 0 || (layer_idx + 1) == config_.num_layers)) {
    const char *out = layer_delta_out_path();
    if (out && *out) {
      auto capture = [&](const float *d, uint32_t n, F32Stats *stats,
                         uint64_t *hash) -> bool {
        if (!d || n == 0)
          return false;
        std::vector<float> host(n);
        if (hipMemcpyAsync(host.data(), d, n * sizeof(float),
                           hipMemcpyDeviceToHost, hip_stream) != hipSuccess) {
          return false;
        }
        hipStreamSynchronize(hip_stream);
        *stats = stats_f32(host.data(), n);
        *hash = hash_f32(host.data(), n);
        return true;
      };

      const uint32_t n_vec = D;
      F32Stats attn_stats{};
      F32Stats mlp_stats{};
      F32Stats x_stats{};
      uint64_t attn_hash = 0;
      uint64_t mlp_hash = 0;
      uint64_t x_hash = 0;
      capture(attn_out, n_vec, &attn_stats, &attn_hash);
      capture(mlp_out, n_vec, &mlp_stats, &mlp_hash);
      capture(x, n_vec, &x_stats, &x_hash);

      const size_t tokens_total = seq_start + seq_len;
      std::ostringstream oss;
      oss << "{\"event\":\"layer_delta\""
          << ",\"step\":" << trace_step_ << ",\"layer\":" << layer_idx
          << ",\"tokens_total\":" << tokens_total << ",\"seq_len\":" << seq_len
          << ",\"attn_out_hash\":" << attn_hash
          << ",\"attn_out_min\":" << attn_stats.min
          << ",\"attn_out_max\":" << attn_stats.max
          << ",\"attn_out_mean\":" << attn_stats.mean
          << ",\"mlp_out_hash\":" << mlp_hash
          << ",\"mlp_out_min\":" << mlp_stats.min
          << ",\"mlp_out_max\":" << mlp_stats.max
          << ",\"mlp_out_mean\":" << mlp_stats.mean
          << ",\"x_out_hash\":" << x_hash << ",\"x_out_min\":" << x_stats.min
          << ",\"x_out_max\":" << x_stats.max
          << ",\"x_out_mean\":" << x_stats.mean << "}";
      append_line(out, oss.str());
    }
  }

  return true;
}

bool BlockScheduler::forward(const int32_t *tokens, size_t seq_start,
                             size_t seq_len, std::string *err) {
  using namespace gcore::rt::hip::kernels;
  if (seq_len == 0) {
    if (err)
      *err = "Embedding Lookup input has seq_len=0";
    return false;
  }
  if (seq_len > config_.max_seq_len) {
    if (err) {
      std::ostringstream oss;
      oss << "Embedding Lookup seq_len(" << seq_len
          << ") exceeds GRETA_MAX_SEQ_LEN(" << config_.max_seq_len << ")";
      *err = oss.str();
    }
    return false;
  }
  if (seq_start + seq_len > config_.max_seq_len) {
    if (err) {
      std::ostringstream oss;
      oss << "Embedding Lookup range overflow: seq_start=" << seq_start
          << ", seq_len=" << seq_len << ", max_seq_len=" << config_.max_seq_len;
      *err = oss.str();
    }
    return false;
  }
  uint32_t S = static_cast<uint32_t>(seq_len);
  uint32_t D = static_cast<uint32_t>(config_.dim);
  uint32_t V = static_cast<uint32_t>(config_.vocab_size);

  activations_.tokens.copy_to_device(tokens, S * sizeof(int32_t), err);

  float *x = static_cast<float *>(activations_.x.data());
  const float *embd_w = static_cast<const float *>(token_embd_.data());
  const int32_t *d_tokens =
      static_cast<const int32_t *>(activations_.tokens.data());
  const bool embed_row_major = embed_layout_row_major();

  hipStream_t hip_stream =
      static_cast<gcore::rt::hip::GretaStreamHip *>(stream_)->handle();

  const bool stage_trace_on = stage_trace_enabled();
  const char *stage_phase_fwd = nullptr;
  if (stage_trace_on) {
    if (S > 1 && trace_step_ == 0) {
      stage_phase_fwd = "prefill_last";
    } else if (S == 1 && trace_step_ == 1) {
      stage_phase_fwd = "decode0";
    }
  }

  // B3.59: Weight Hash (First 1KB)
  if (stage_trace_on && stage_phase_fwd &&
      stage_trace_point_enabled("embd_w_hash")) {
    const char *stage_prompt_id = std::getenv("GRETA_TRACE_PROMPT_ID");
    const float *w_ptr = reinterpret_cast<const float *>(token_embd_.data());
    // We trace only a small sample (256 floats = 1KB) of the weight table to
    // verify it's the same
    stage_trace_tensor("embd_w_hash", stage_phase_fwd, stage_prompt_id, 0,
                       static_cast<uint32_t>(trace_step_), 0, 1, 0, w_ptr, 256,
                       0, hip_stream);
  }

  CHECK_HIP_KERNEL(launch_embedding_lookup(hip_stream, d_tokens, embd_w, x, S,
                                           D, config_.vocab_size,
                                           embed_row_major),
                   "Embedding Lookup");

  if (stage_trace_on && stage_phase_fwd &&
      stage_trace_point_enabled("embed_out")) {
    const char *stage_prompt_id = std::getenv("GRETA_TRACE_PROMPT_ID");
    const uint32_t stage_tokens_total = static_cast<uint32_t>(seq_start + S);
    const uint32_t stage_token_index = S > 0 ? static_cast<uint32_t>(S - 1) : 0;
    const uint32_t stage_pos_id =
        static_cast<uint32_t>(seq_start + stage_token_index);
    const int32_t token_id_val = tokens[stage_token_index];

    StageInputMeta input_meta{};
    input_meta.src_kind =
        (S > 1) ? "EMBED_LOOKUP_PREFILL" : "EMBED_LOOKUP_DECODE";
    input_meta.token_index_used = stage_token_index;
    input_meta.offset_bytes =
        static_cast<size_t>(stage_token_index) * D * sizeof(float);
    input_meta.ptr = reinterpret_cast<uintptr_t>(x);
    input_meta.alloc_bytes = activations_.x.size();
    input_meta.prompt_tokens =
        (seq_start == 0) ? S : static_cast<uint32_t>(seq_start);
    input_meta.kv_pos = stage_pos_id;
    input_meta.decode_step = static_cast<uint32_t>(trace_step_);
    input_meta.token_id = static_cast<uint32_t>(token_id_val);
    input_meta.route = input_meta.src_kind;

    stage_trace_tensor("embed_out", stage_phase_fwd, stage_prompt_id, 0,
                       static_cast<uint32_t>(trace_step_), stage_pos_id, S,
                       stage_tokens_total, x, D, stage_token_index, hip_stream,
                       &input_meta);
  }
  if (!trace_embed_verify_once(tokens, seq_len, D, V, token_embd_,
                               activations_.x, embed_row_major, err))
    return false;

  uint32_t pos = static_cast<uint32_t>(seq_start);
  activations_.d_pos.copy_to_device(&pos, sizeof(uint32_t), err);

  const char *use_graph_env = std::getenv("GRETA_GRAPH");
  bool use_graph =
      (use_graph_env && std::string(use_graph_env) == "1") && (S == 1);

  if (use_graph && graph_captured_) {
    const char *profile_blocks = std::getenv("GRETA_PROFILE_BLOCKS");
    if (profile_blocks && std::string(profile_blocks) == "1") {
      printf("[GRETA_L0_AUDIT] Graph Launch (Decode Step)\n");
    }
    CHECK_GRETA(graph_->launch(stream_), "Graph Launch");
  } else {
    if (use_graph && !graph_captured_) {
      if (!graph_)
        graph_ = gcore::rt::GretaContext::instance().create_graph();
      graph_->capture_start(stream_);
    }

    for (size_t i = 0; i < config_.num_layers; ++i) {
      if (!execute_layer(i, seq_start, seq_len, tokens, err))
        return false;
    }

    float *norm_out = static_cast<float *>(activations_.norm_out.data());
    const float *onorm_w = static_cast<const float *>(output_norm_.data());

    CHECK_HIP_KERNEL(launch_rmsnorm_naive(hip_stream, x, onorm_w, norm_out, S,
                                          D, config_.rms_eps),
                     "Final RMSNorm");

    const bool stage_enabled = stage_trace_enabled();
    const bool post_wo_enabled = trace_post_wo_enabled();
    const char *stage_phase = nullptr;
    if (stage_enabled || post_wo_enabled) {
      if (seq_len > 1 && trace_step_ == 0) {
        stage_phase = "prefill_last";
      } else if (seq_len == 1 && trace_step_ == 1) {
        stage_phase = "decode0";
      }
    }
    if (stage_phase && stage_trace_phase_enabled(stage_phase)) {
      const uint32_t stage_tokens_total =
          static_cast<uint32_t>(seq_start + seq_len);
      const uint32_t stage_token_index =
          seq_len > 0 ? static_cast<uint32_t>(seq_len - 1) : 0;
      const uint32_t stage_pos_id =
          static_cast<uint32_t>(seq_start + stage_token_index);
      const char *stage_prompt_id = std::getenv("GRETA_TRACE_PROMPT_ID");
      const size_t stride_elems = D;
      const size_t final_layer = config_.num_layers;

      if (stage_trace_point_enabled("final_norm")) {
        stage_trace_tensor("final_norm", stage_phase, stage_prompt_id,
                           final_layer, static_cast<uint32_t>(trace_step_),
                           stage_pos_id, static_cast<uint32_t>(seq_len),
                           stage_tokens_total, norm_out, stride_elems,
                           stage_token_index, hip_stream);
      }
      if (stage_trace_point_enabled("final_rms")) {
        stage_trace_tensor("final_rms", stage_phase, stage_prompt_id,
                           final_layer, static_cast<uint32_t>(trace_step_),
                           stage_pos_id, static_cast<uint32_t>(seq_len),
                           stage_tokens_total, norm_out, stride_elems,
                           stage_token_index, hip_stream);
      }
      if (stage_trace_point_enabled("lm_head_in")) {
        stage_trace_tensor("lm_head_in", stage_phase, stage_prompt_id,
                           final_layer, static_cast<uint32_t>(trace_step_),
                           stage_pos_id, static_cast<uint32_t>(seq_len),
                           stage_tokens_total, norm_out, stride_elems,
                           stage_token_index, hip_stream);
      }
    }

    if (trace_post_wo_enabled() && post_wo_out_path() && stage_phase &&
        post_wo_phase_enabled(stage_phase)) {
      const uint32_t stage_tokens_total =
          static_cast<uint32_t>(seq_start + seq_len);
      const uint32_t stage_token_index =
          seq_len > 0 ? static_cast<uint32_t>(seq_len - 1) : 0;
      const uint32_t stage_pos_id =
          static_cast<uint32_t>(seq_start + stage_token_index);
      const char *stage_prompt_id = std::getenv("GRETA_TRACE_PROMPT_ID");
      const size_t stride_elems = D;
      const size_t final_layer = config_.num_layers;
      if (post_wo_layer_selected(final_layer, config_.num_layers)) {
        post_wo_trace_tensor("final_rms", stage_phase, stage_prompt_id,
                             final_layer, static_cast<uint32_t>(trace_step_),
                             stage_pos_id, static_cast<uint32_t>(seq_len),
                             stage_tokens_total, norm_out, stride_elems,
                             stage_token_index, activations_.norm_out.size(),
                             hip_stream);
        post_wo_trace_tensor("lm_head_in", stage_phase, stage_prompt_id,
                             final_layer, static_cast<uint32_t>(trace_step_),
                             stage_pos_id, static_cast<uint32_t>(seq_len),
                             stage_tokens_total, norm_out, stride_elems,
                             stage_token_index, activations_.norm_out.size(),
                             hip_stream);
      }
    }

    size_t logits_offset_bytes =
        static_cast<size_t>(seq_start) * static_cast<size_t>(V) * sizeof(float);
    size_t logits_bytes =
        static_cast<size_t>(S) * static_cast<size_t>(V) * sizeof(float);
    if (logits_offset_bytes + logits_bytes > logits_.size()) {
      if (err) {
        *err = "LM Head logits offset out of range: offset=" +
               std::to_string(logits_offset_bytes) +
               " bytes=" + std::to_string(logits_bytes) +
               " alloc=" + std::to_string(logits_.size());
      }
      return false;
    }
    GretaMemoryView logits_view(&logits_, logits_offset_bytes);
    const bool is_decode = (seq_len == 1 && seq_start > 0);
    const char *lm_head_label =
        is_decode ? "lm_head_decode" : "lm_head_prefill";
    gcore::compute::GretaCompute::set_op_label(lm_head_label);
    CHECK_GRETA(gcore::compute::GretaCompute::gemm(
                    stream_, &activations_.norm_out, &output_weight_,
                    &logits_view, S, V, D),
                "LM Head");
    gcore::compute::GretaCompute::set_op_label(nullptr);

    if (use_graph && !graph_captured_) {
      graph_->capture_end(stream_);
      CHECK_GRETA(graph_->instantiate(), "Graph Instantiate");
      graph_captured_ = true;
    }
  }

  stream_->synchronize();
  trace_step_++;
  return true;
}

gcore::rt::hip::Buffer &BlockScheduler::get_hidden_state() {
  return activations_.x;
}

gcore::rt::hip::Buffer &BlockScheduler::get_norm_out() {
  return activations_.norm_out;
}

const gcore::rt::hip::Buffer &BlockScheduler::get_norm_out() const {
  return activations_.norm_out;
}

const gcore::rt::hip::Buffer &BlockScheduler::get_logits() const {
  return logits_;
}

const gcore::rt::hip::Buffer &BlockScheduler::get_output_weight() const {
  return output_weight_;
}

int32_t BlockScheduler::sample_greedy_gpu(size_t logits_offset_bytes,
                                          std::string *err) {
  (void)err;
  int32_t top_id = 0;
  hipStream_t hip_stream =
      static_cast<gcore::rt::hip::GretaStreamHip *>(stream_)->handle();
  const float *logits_base = static_cast<const float *>(logits_.data());
  const size_t offset_elems = logits_offset_bytes / sizeof(float);
  rt::hip::kernels::launch_argmax(hip_stream, logits_base + offset_elems,
                                  config_.vocab_size, &top_id);
  return top_id;
}

} // namespace gcore::inference
