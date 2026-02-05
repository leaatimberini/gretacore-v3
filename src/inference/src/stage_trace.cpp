#include "gcore/inference/stage_trace.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace gcore::inference {

struct F32Stats {
  float min = 0.0f;
  float max = 0.0f;
  float mean = 0.0f;
  float abs_sum = 0.0f;
  int nan = 0;
  int inf = 0;
  size_t nz_count = 0;
};

static std::vector<std::string> split_csv(const char *v) {
  std::vector<std::string> out;
  if (!v || !*v)
    return out;
  std::string s(v);
  size_t start = 0;
  while (start < s.size()) {
    size_t end = s.find(',', start);
    std::string token = s.substr(
        start, (end == std::string::npos) ? s.size() - start : end - start);
    if (!token.empty())
      out.push_back(token);
    if (end == std::string::npos)
      break;
    start = end + 1;
  }
  return out;
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
      long val = std::strtol(token.c_str(), &e, 10);
      if (e != token.c_str())
        layers.push_back(static_cast<int>(val));
    }
    if (end == std::string::npos)
      break;
    start = end + 1;
  }
  return layers;
}

static F32Stats stats_f32(const float *p, size_t n) {
  F32Stats s{};
  if (n == 0 || !p)
    return s;
  s.min = p[0];
  s.max = p[0];
  double sum = 0.0;
  double abs_sum = 0.0;
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
    abs_sum += std::abs(v);
    if (v != 0.0f)
      s.nz_count++;
  }
  s.mean = (n > 0) ? static_cast<float>(sum / n) : 0.0f;
  s.abs_sum = static_cast<float>(abs_sum);
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

StageTraceConfig stage_trace_config() {
  static StageTraceConfig cfg;
  static bool initialized = false;
  if (!initialized) {
    cfg.enabled = false;
    const char *on = std::getenv("GRETA_TRACE_STAGE");
    if (on && (on[0] == '1' || on[0] == 'y' || on[0] == 'Y'))
      cfg.enabled = true;
    cfg.out_path = std::getenv("GRETA_TRACE_STAGE_OUT");
    cfg.layers = parse_layers(std::getenv("GRETA_TRACE_STAGE_LAYERS"));
    cfg.points = split_csv(std::getenv("GRETA_TRACE_STAGE_POINTS"));
    cfg.phases = split_csv(std::getenv("GRETA_TRACE_STAGE_PHASES"));
    cfg.debug_input = false;
    const char *dbg = std::getenv("GRETA_TRACE_STAGE_DEBUG_INPUT");
    if (dbg && (dbg[0] == '1' || dbg[0] == 'y' || dbg[0] == 'Y'))
      cfg.debug_input = true;
    const char *s = std::getenv("GRETA_TRACE_STAGE_SAMPLE");
    if (s && *s) {
      char *e = nullptr;
      long val = std::strtol(s, &e, 10);
      if (e != s && val > 0)
        cfg.sample = static_cast<uint32_t>(val);
    }
    initialized = true;
  }
  return cfg;
}

bool stage_trace_enabled() { return stage_trace_config().enabled; }

bool stage_trace_layer_selected(size_t layer_idx, size_t num_layers) {
  const auto cfg = stage_trace_config();
  if (!cfg.enabled)
    return false;
  if (cfg.layers.empty())
    return true;
  for (int layer : cfg.layers) {
    if (layer < 0 && static_cast<size_t>(-layer) == num_layers)
      return true;
    if (static_cast<size_t>(layer) == layer_idx)
      return true;
  }
  return false;
}

bool stage_trace_point_enabled(const char *point) {
  const auto cfg = stage_trace_config();
  if (!cfg.enabled)
    return false;
  if (cfg.points.empty())
    return true;
  for (const auto &p : cfg.points) {
    if (p == point)
      return true;
  }
  return false;
}

bool stage_trace_phase_enabled(const char *phase) {
  const auto cfg = stage_trace_config();
  if (!cfg.enabled)
    return false;
  if (cfg.phases.empty())
    return true;
  for (const auto &p : cfg.phases) {
    if (p == phase)
      return true;
  }
  return false;
}

uint32_t stage_trace_sample() { return stage_trace_config().sample; }

const char *stage_trace_out_path() { return stage_trace_config().out_path; }

bool stage_trace_debug_input() { return stage_trace_config().debug_input; }

void stage_trace_tensor(const char *point, const char *phase,
                        const char *prompt_id, size_t layer, uint32_t step,
                        uint32_t pos_id, uint32_t seq_len,
                        uint32_t tokens_total, const float *base,
                        size_t stride_elems, size_t token_index,
                        hipStream_t stream, const StageInputMeta *input_meta) {
  const auto cfg = stage_trace_config();
  if (!cfg.enabled || !cfg.out_path || !*cfg.out_path || !point || !phase)
    return;
  if (!stage_trace_phase_enabled(phase))
    return;
  if (!stage_trace_point_enabled(point))
    return;
  if (!base || stride_elems == 0)
    return;

  size_t offset_elems = token_index * stride_elems;
  const float *ptr = base + offset_elems;
  const uint32_t sample_n =
      std::min<uint32_t>(cfg.sample, static_cast<uint32_t>(stride_elems));
  std::vector<float> host(sample_n, 0.0f);
  if (sample_n > 0) {
    hipMemcpyAsync(host.data(), ptr, sample_n * sizeof(float),
                   hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
  }
  const F32Stats stats = stats_f32(host.data(), host.size());
  const uint64_t hash = hash_f32(host.data(), host.size());

  uint32_t debug_token_id = 0;
  if (cfg.debug_input && phase && std::strcmp(phase, "decode0") == 0) {
    const char *tid_env = std::getenv("GRETA_TRACE_DEBUG_INPUT_TOKEN_ID");
    if (tid_env) {
      debug_token_id =
          static_cast<uint32_t>(std::strtoul(tid_env, nullptr, 10));
    }
  }

  std::ostringstream oss;
  oss << "{\"event\":\"stage_trace\"";
  if (prompt_id && *prompt_id)
    oss << ",\"prompt_id\":\"" << prompt_id << "\"";
  oss << ",\"phase\":\"" << phase << "\""
      << ",\"point\":\"" << point << "\""
      << ",\"layer\":" << layer << ",\"step\":" << step
      << ",\"pos_id\":" << pos_id << ",\"seq_len\":" << seq_len
      << ",\"tokens_total\":" << tokens_total
      << ",\"token_index\":" << token_index
      << ",\"stride_elems\":" << stride_elems << ",\"sample_n\":" << sample_n
      << ",\"ptr\":" << reinterpret_cast<uintptr_t>(base)
      << ",\"offset_bytes\":" << (offset_elems * sizeof(float))
      << ",\"hash\":" << hash << ",\"min\":" << stats.min
      << ",\"max\":" << stats.max << ",\"mean\":" << stats.mean
      << ",\"abs_sum\":" << stats.abs_sum << ",\"nan\":" << stats.nan
      << ",\"inf\":" << stats.inf << ",\"nz_count\":" << stats.nz_count
      << ",\"sample\":[";
  for (size_t i = 0; i < host.size(); ++i) {
    if (i)
      oss << ",";
    oss << host[i];
  }
  oss << "]";

  uint32_t final_token_id = (input_meta ? input_meta->token_id : 0);
  if (final_token_id == 0 && debug_token_id != 0)
    final_token_id = debug_token_id;

  if (input_meta) {
    const char *kind = input_meta->src_kind ? input_meta->src_kind : "";
    oss << ",\"src_kind\":\"" << kind << "\""
        << ",\"token_index_used\":" << input_meta->token_index_used
        << ",\"offset_bytes\":" << input_meta->offset_bytes
        << ",\"alloc_bytes\":" << input_meta->alloc_bytes
        << ",\"prompt_tokens\":" << input_meta->prompt_tokens
        << ",\"kv_pos\":" << input_meta->kv_pos
        << ",\"decode_step\":" << input_meta->decode_step
        << ",\"token_id\":" << final_token_id << ",\"route\":\""
        << (input_meta->route ? input_meta->route : "") << "\"";
  }

  oss << "}";
  append_line(cfg.out_path, oss.str());
}

void stage_trace_logits(const char *phase, const char *prompt_id, uint32_t step,
                        uint32_t pos_id, uint32_t seq_len,
                        uint32_t tokens_total, const StageLogitsStats &stats) {
  const auto cfg = stage_trace_config();
  if (!cfg.enabled || !cfg.out_path || !*cfg.out_path || !phase)
    return;
  if (!stage_trace_phase_enabled(phase))
    return;
  if (!stage_trace_point_enabled("logits"))
    return;

  std::ostringstream oss;
  oss << "{\"event\":\"stage_logits\"";
  if (prompt_id && *prompt_id)
    oss << ",\"prompt_id\":\"" << prompt_id << "\"";
  oss << ",\"phase\":\"" << phase << "\""
      << ",\"point\":\"logits\""
      << ",\"layer\":-1"
      << ",\"step\":" << step << ",\"pos_id\":" << pos_id
      << ",\"seq_len\":" << seq_len << ",\"tokens_total\":" << tokens_total
      << ",\"hash\":" << stats.hash << ",\"min\":" << stats.min
      << ",\"max\":" << stats.max << ",\"mean\":" << stats.mean
      << ",\"top1_id\":" << stats.top1_id
      << ",\"top1_logit\":" << stats.top1_logit
      << ",\"top2_id\":" << stats.top2_id
      << ",\"top2_logit\":" << stats.top2_logit << ",\"gap\":" << stats.gap
      << ",\"vocab\":" << stats.vocab << ",\"logits_ptr\":" << stats.logits_ptr
      << ",\"logits_offset_bytes\":" << stats.logits_offset_bytes << "}";
  append_line(cfg.out_path, oss.str());
}

} // namespace gcore::inference
