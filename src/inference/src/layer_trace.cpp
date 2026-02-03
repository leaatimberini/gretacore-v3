#include "gcore/inference/layer_trace.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <unordered_set>

namespace gcore::inference {

static bool env_flag(const char *k) {
  const char *v = std::getenv(k);
  return v && v[0] == 1;
}

struct F32Stats {
  float min = 0.0f;
  float max = 0.0f;
  float mean = 0.0f;
  int nan = 0;
  int inf = 0;
};

static F32Stats stats_f32(const float *p, size_t n) {
  F32Stats s{};
  if (n == 0)
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

static std::vector<int> parse_layers(const char *v) {
  std::vector<int> layers;
  if (!v || !*v)
    return layers;
  std::string s(v);
  size_t start = 0;
  while (start < s.size()) {
    size_t end = s.find(',', start);
    std::string token =
        s.substr(start, (end == std::string::npos) ? s.size() - start
                                                    : end - start);
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

static uint32_t point_mask_from_list(const char *v) {
  if (!v || !*v)
    return 0;
  std::string s(v);
  uint32_t mask = 0;
  size_t start = 0;
  while (start < s.size()) {
    size_t end = s.find(',', start);
    std::string t =
        s.substr(start, (end == std::string::npos) ? s.size() - start
                                                    : end - start);
    if (t == "x")
      mask |= static_cast<uint32_t>(LayerTracePoint::X);
    else if (t == "norm_out")
      mask |= static_cast<uint32_t>(LayerTracePoint::NORM_OUT);
    else if (t == "q")
      mask |= static_cast<uint32_t>(LayerTracePoint::Q);
    else if (t == "k")
      mask |= static_cast<uint32_t>(LayerTracePoint::K);
    else if (t == "v")
      mask |= static_cast<uint32_t>(LayerTracePoint::V);
    else if (t == "attn_out")
      mask |= static_cast<uint32_t>(LayerTracePoint::ATTN_OUT);
    else if (t == "ffn_norm")
      mask |= static_cast<uint32_t>(LayerTracePoint::FFN_NORM);
    else if (t == "mlp_out")
      mask |= static_cast<uint32_t>(LayerTracePoint::MLP_OUT);
    else if (t == "x_out")
      mask |= static_cast<uint32_t>(LayerTracePoint::X_OUT);
    else if (t == "mlp_gate")
      mask |= static_cast<uint32_t>(LayerTracePoint::MLP_GATE);
    else if (t == "mlp_up")
      mask |= static_cast<uint32_t>(LayerTracePoint::MLP_UP);

    if (end == std::string::npos)
      break;
    start = end + 1;
  }
  return mask;
}

static uint32_t point_mask_for_tag(const char *tag) {
  if (!tag)
    return 0;
  if (std::string(tag) == "x")
    return static_cast<uint32_t>(LayerTracePoint::X);
  if (std::string(tag) == "norm_out")
    return static_cast<uint32_t>(LayerTracePoint::NORM_OUT);
  if (std::string(tag) == "q")
    return static_cast<uint32_t>(LayerTracePoint::Q);
  if (std::string(tag) == "k")
    return static_cast<uint32_t>(LayerTracePoint::K);
  if (std::string(tag) == "v")
    return static_cast<uint32_t>(LayerTracePoint::V);
  if (std::string(tag) == "attn_out")
    return static_cast<uint32_t>(LayerTracePoint::ATTN_OUT);
  if (std::string(tag) == "ffn_norm")
    return static_cast<uint32_t>(LayerTracePoint::FFN_NORM);
  if (std::string(tag) == "mlp_out")
    return static_cast<uint32_t>(LayerTracePoint::MLP_OUT);
  if (std::string(tag) == "x_out")
    return static_cast<uint32_t>(LayerTracePoint::X_OUT);
  if (std::string(tag) == "mlp_gate")
    return static_cast<uint32_t>(LayerTracePoint::MLP_GATE);
  if (std::string(tag) == "mlp_up")
    return static_cast<uint32_t>(LayerTracePoint::MLP_UP);
  return 0;
}

void LayerTracer::init_from_env(const ModelConfig &config) {
  (void)config;
  cfg_.enabled = env_flag("GRETA_TRACE_LAYER");
  if (!cfg_.enabled)
    return;

  const char *out = std::getenv("GRETA_TRACE_LAYER_OUT");
  if (out && *out)
    cfg_.out_path = out;
  else
    cfg_.out_path = "";

  cfg_.layers = parse_layers(std::getenv("GRETA_TRACE_LAYER_LAYERS"));
  cfg_.points_mask =
      point_mask_from_list(std::getenv("GRETA_TRACE_LAYER_POINTS"));
  if (cfg_.points_mask == 0) {
    cfg_.points_mask = static_cast<uint32_t>(LayerTracePoint::X) |
                       static_cast<uint32_t>(LayerTracePoint::ATTN_OUT) |
                       static_cast<uint32_t>(LayerTracePoint::MLP_OUT) |
                       static_cast<uint32_t>(LayerTracePoint::X_OUT);
  }

  if (!cfg_.out_path.empty()) {
    out_.open(cfg_.out_path, std::ios::out | std::ios::trunc);
  }
}

bool LayerTracer::should_trace_layer(int layer) const {
  if (!cfg_.enabled)
    return false;
  if (cfg_.layers.empty())
    return true;
  for (int l : cfg_.layers) {
    if (l == layer)
      return true;
  }
  return false;
}

bool LayerTracer::point_enabled(const char *tag) const {
  uint32_t bit = point_mask_for_tag(tag);
  if (bit == 0)
    return false;
  return (cfg_.points_mask & bit) != 0;
}

void LayerTracer::trace_tensor(const char *tag, int step, int layer,
                               hipStream_t stream, const float *d,
                               uint32_t n) {
  if (!cfg_.enabled)
    return;
  if (!should_trace_layer(layer))
    return;
  if (!point_enabled(tag))
    return;
  if (n == 0)
    return;

  std::vector<float> host(n);
  if (hipMemcpyAsync(host.data(), d, n * sizeof(float), hipMemcpyDeviceToHost,
                     stream) != hipSuccess) {
    return;
  }
  hipStreamSynchronize(stream);

  F32Stats s = stats_f32(host.data(), n);
  uint64_t h = hash_f32(host.data(), n);

  std::ostringstream oss;
  oss << "{\"step\":" << step << ",\"layer\":" << layer
      << ",\"tag\":\"" << tag << "\""
      << ",\"n\":" << n << ",\"hash\":" << h
      << ",\"min\":" << s.min << ",\"max\":" << s.max
      << ",\"mean\":" << s.mean << ",\"nan\":" << s.nan
      << ",\"inf\":" << s.inf << "}";

  if (out_.is_open()) {
    out_ << oss.str() << "\n";
  } else {
    std::cout << oss.str() << "\n";
  }
}

void LayerTracer::trace_tensor_f16(const char *tag, int step, int layer,
                                   hipStream_t stream, const __half *d,
                                   uint32_t n) {
  if (!cfg_.enabled)
    return;
  if (!should_trace_layer(layer))
    return;
  if (!point_enabled(tag))
    return;
  if (n == 0)
    return;

  std::vector<__half> host_half(n);
  if (hipMemcpyAsync(host_half.data(), d, n * sizeof(__half),
                     hipMemcpyDeviceToHost, stream) != hipSuccess) {
    return;
  }
  hipStreamSynchronize(stream);

  std::vector<float> host(n);
  for (uint32_t i = 0; i < n; ++i) {
    host[i] = __half2float(host_half[i]);
  }

  F32Stats s = stats_f32(host.data(), n);
  uint64_t h = hash_f32(host.data(), n);

  std::ostringstream oss;
  oss << "{\"step\":" << step << ",\"layer\":" << layer
      << ",\"tag\":\"" << tag << "\""
      << ",\"n\":" << n << ",\"hash\":" << h
      << ",\"min\":" << s.min << ",\"max\":" << s.max
      << ",\"mean\":" << s.mean << ",\"nan\":" << s.nan
      << ",\"inf\":" << s.inf << "}";

  if (out_.is_open()) {
    out_ << oss.str() << "\n";
  } else {
    std::cout << oss.str() << "\n";
  }
}

void layer_trace_emit_step_header(int step, size_t pos_id, size_t seq_len,
                                  size_t tokens_total, int32_t token_in,
                                  int32_t token_out, const ModelConfig &cfg) {
  if (!env_flag("GRETA_TRACE_LAYER"))
    return;
  const char *out = std::getenv("GRETA_TRACE_LAYER_OUT");
  if (!out || !*out)
    return;

  const char *layers = std::getenv("GRETA_TRACE_LAYER_LAYERS");
  const char *points = std::getenv("GRETA_TRACE_LAYER_POINTS");

  std::ofstream f(out, std::ios::out | std::ios::app);
  if (!f.is_open())
    return;

  std::ostringstream oss;
  oss << "{\"type\":\"step_header\""
      << ",\"step\":" << step
      << ",\"pos_id\":" << pos_id
      << ",\"seq_len\":" << seq_len
      << ",\"tokens_total\":" << tokens_total
      << ",\"token_in\":" << token_in
      << ",\"token_out\":" << token_out
      << ",\"dim\":" << cfg.dim
      << ",\"heads\":" << cfg.num_heads
      << ",\"head_dim\":" << cfg.head_dim
      << ",\"layers\":\"" << (layers ? layers : "") << "\""
      << ",\"points\":\"" << (points ? points : "") << "\""
      << "}";
  f << oss.str() << "\n";
}

} // namespace gcore::inference
