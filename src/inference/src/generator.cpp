#include "gcore/inference/generator.hpp"
#include "gcore/inference/block_scheduler.hpp"
#include "gcore/inference/tokenizer.hpp"
#include "gcore/inference/trace.hpp"
#include "gcore/inference/layer_trace.hpp"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>

namespace gcore::inference {


struct F32Stats {
  float min = 0.0f;
  float max = 0.0f;
  float mean = 0.0f;
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
    if (v < s.min)
      s.min = v;
    if (v > s.max)
      s.max = v;
    sum += v;
  }
  s.mean = static_cast<float>(sum / n);
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

static void log_readout(const char *path, const char *phase, int step,
                        size_t token_index, size_t hidden_offset_bytes,
                        uintptr_t hidden_ptr, const F32Stats &hstats,
                        uint64_t hhash, size_t logits_offset_bytes,
                        uintptr_t logits_ptr, const F32Stats &lstats,
                        uint64_t lhash, size_t vocab) {
  std::ostringstream oss;
  oss << "{\"phase\":\"" << phase << "\""
      << ",\"step\":" << step
      << ",\"token_index\":" << token_index
      << ",\"hidden_offset\":" << hidden_offset_bytes
      << ",\"hidden_ptr\":" << hidden_ptr
      << ",\"hidden_hash\":" << hhash
      << ",\"hidden_min\":" << hstats.min
      << ",\"hidden_max\":" << hstats.max
      << ",\"hidden_mean\":" << hstats.mean
      << ",\"logits_offset\":" << logits_offset_bytes
      << ",\"logits_ptr\":" << logits_ptr
      << ",\"logits_hash\":" << lhash
      << ",\"logits_min\":" << lstats.min
      << ",\"logits_max\":" << lstats.max
      << ",\"logits_mean\":" << lstats.mean
      << ",\"vocab\":" << vocab
      << "}";
  append_line(path, oss.str());
}

static void log_landscape(const char *path, int step,
                          const std::vector<float> &logits, int topk) {
  if (!path || !*path)
    return;
  std::vector<std::pair<float, int>> v;
  v.reserve(logits.size());
  for (size_t i = 0; i < logits.size(); ++i)
    v.push_back({logits[i], (int)i});
  std::sort(v.rbegin(), v.rend());
  const int k = std::min<int>(topk, (int)v.size());
  const float top1 = v[0].first;
  const float top2 = v[1].first;
  const float gap = top1 - top2;

  float max_logit = v[0].first;
  double sum = 0.0;
  for (int i = 0; i < k; ++i)
    sum += std::exp(v[i].first - max_logit);
  double entropy = 0.0;
  for (int i = 0; i < k; ++i) {
    double p = std::exp(v[i].first - max_logit) / sum;
    entropy += -p * std::log(p + 1e-12);
  }

  std::ostringstream oss;
  oss << "{\"step\":" << step
      << ",\"top1\":{\"id\":" << v[0].second << ",\"logit\":" << v[0].first
      << "},\"top2\":{\"id\":" << v[1].second << ",\"logit\":" << v[1].first
      << "},\"gap\":" << gap
      << ",\"entropy_topk\":" << entropy
      << ",\"top5\":[";
  for (int i = 0; i < 5 && i < (int)v.size(); ++i) {
    if (i)
      oss << ",";
    oss << "{\"id\":" << v[i].second << ",\"logit\":" << v[i].first << "}";
  }
  oss << "]}";
  append_line(path, oss.str());
}

static bool env_flag(const char *k) {
  const char *v = std::getenv(k);
  return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

static bool validate_trace_shapes(const ModelConfig &config, std::string *err) {
  auto fail = [&](const std::string &msg) {
    if (err)
      *err = msg;
    std::cerr << "[GRETA_TRACE_SHAPE] " << msg << std::endl;
    return false;
  };

  if (config.vocab_size == 0)
    return fail("vocab_size=0");
  if (config.dim == 0)
    return fail("dim=0");
  if (config.num_layers == 0)
    return fail("num_layers=0");
  if (config.num_heads == 0)
    return fail("num_heads=0");
  if (config.num_heads_kv == 0)
    return fail("num_heads_kv=0");
  if (config.head_dim == 0)
    return fail("head_dim=0");

  if (config.dim % config.num_heads != 0) {
    return fail("dim not divisible by num_heads: dim=" +
                std::to_string(config.dim) +
                " num_heads=" + std::to_string(config.num_heads));
  }
  const uint32_t expected_head_dim = config.dim / config.num_heads;
  if (config.head_dim != expected_head_dim) {
    return fail("head_dim mismatch: head_dim=" +
                std::to_string(config.head_dim) +
                " expected=" + std::to_string(expected_head_dim));
  }
  if (config.num_heads_kv > config.num_heads) {
    return fail("num_heads_kv greater than num_heads: num_heads_kv=" +
                std::to_string(config.num_heads_kv) +
                " num_heads=" + std::to_string(config.num_heads));
  }
  if (config.num_heads % config.num_heads_kv != 0) {
    return fail("num_heads not divisible by num_heads_kv: num_heads=" +
                std::to_string(config.num_heads) +
                " num_heads_kv=" + std::to_string(config.num_heads_kv));
  }

  return true;
}

static void log_d2h_trace(bool enabled, const char *tensor_name, int step,
                          int layer, const gcore::rt::hip::Buffer &buffer,
                          size_t offset_bytes, size_t size_bytes) {
  if (!enabled)
    return;
  std::ostringstream oss;
  oss << "[GRETA_TRACE_D2H]"
      << " tensor=" << (tensor_name ? tensor_name : "unknown")
      << " step=" << step
      << " layer=" << layer
      << " src_ptr=0x" << std::hex
      << reinterpret_cast<uintptr_t>(buffer.data()) << std::dec
      << " alloc_bytes=" << buffer.size()
      << " offset_bytes=" << offset_bytes
      << " size_bytes=" << size_bytes;
  std::cerr << oss.str() << std::endl;
}

Generator::Generator() = default;

Generator::~Generator() = default;

bool Generator::init(const ModelConfig &config, BlockScheduler *scheduler,
                     std::string *err) {
  config_ = config;
  scheduler_ = scheduler;

  // Initialize tokenizer
  tokenizer_ = std::make_unique<Tokenizer>();
  tokenizer_->set_vocabulary(config_.vocabulary);

  initialized_ = true;
  return true;
}

int32_t Generator::sample(const float *logits, size_t vocab_size,
                          const SamplingParams &params) {
  // Diagnostic: Check if logits are sane
  float min_l = logits[0], max_l = logits[0], sum_l = 0.0f;
  int nan_count = 0;
  for (size_t i = 0; i < vocab_size; ++i) {
    float v = logits[i];
    if (std::isnan(v))
      nan_count++;
    else {
      if (v < min_l)
        min_l = v;
      if (v > max_l)
        max_l = v;
      sum_l += v;
    }
  }

  static int sample_count = 0;
  if (sample_count++ < 3) { // Print only for first 3 tokens
    std::cout << "[SAMPLE DEBUG] Logits stats: min=" << min_l
              << " max=" << max_l << " avg=" << (sum_l / vocab_size)
              << " NaNs=" << nan_count << std::endl;
    // Print top 5 logits
    std::vector<std::pair<float, int>> top;
    for (int i = 0; i < (int)vocab_size; ++i)
      top.push_back({logits[i], i});
    std::sort(top.rbegin(), top.rend());
    std::cout << "  Top tokens: ";
    for (int i = 0; i < 5; ++i)
      std::cout << top[i].second << "(" << top[i].first << ") ";
    std::cout << std::endl;
  }

  if (params.greedy) {
    int32_t max_id = 0;
    float max_val = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
      if (logits[i] > max_val) {
        max_val = logits[i];
        max_id = static_cast<int32_t>(i);
      }
    }
    return max_id;
  }

  // Softmax with temperature
  std::vector<float> probs(vocab_size);
  float sum = 0.0f;
  float max_logit = -INFINITY;
  for (size_t i = 0; i < vocab_size; ++i) {
    if (logits[i] > max_logit)
      max_logit = logits[i];
  }

  for (size_t i = 0; i < vocab_size; ++i) {
    float p = std::exp((logits[i] - max_logit) / params.temperature);
    probs[i] = p;
    sum += p;
  }

  // Random sample
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, sum);
  float r = dis(gen);

  float cumulative = 0.0f;
  for (size_t i = 0; i < vocab_size; ++i) {
    cumulative += probs[i];
    if (r <= cumulative)
      return static_cast<int32_t>(i);
  }

  return 0;
}

std::vector<int32_t>
Generator::generate_tokens(const std::vector<int32_t> &prompt_tokens,
                           const SamplingParams &params, GenerationStats *stats,
                           std::string *err, AlignmentCallback align_callback) {
  if (!initialized_) {
    if (err)
      *err = "Generator not initialized";
    return {};
  }

  std::vector<int32_t> output = prompt_tokens;
  auto start = std::chrono::high_resolution_clock::now();
  auto first_token_time = start;
  bool first_token = true;

  std::vector<float> logits_host(config_.vocab_size);

  const bool trace_readout = env_flag("GRETA_TRACE_READOUT");
  const char *trace_readout_out = std::getenv("GRETA_TRACE_READOUT_OUT");
  const bool trace_prefill_decode = env_flag("GRETA_TRACE_PREFILL_DECODE");
  const char *trace_prefill_decode_out = std::getenv("GRETA_TRACE_PREFILL_DECODE_OUT");
  const bool trace_landscape = env_flag("GRETA_TRACE_LANDSCAPE");
  const char *trace_landscape_out = std::getenv("GRETA_TRACE_LANDSCAPE_OUT");
  const int landscape_topk = 64;
  const bool trace_any = trace_readout || trace_prefill_decode || trace_landscape;

  std::vector<float> hidden_host;
  if (trace_readout || trace_prefill_decode)
    hidden_host.resize(config_.dim);

  if (trace_any) {
    if (!validate_trace_shapes(config_, err)) {
      return output;
    }
  }

  // 1. Prefill: Process all prompt tokens at once
  if (!scheduler_->forward(prompt_tokens.data(), 0, prompt_tokens.size(),
                           err)) {
    return output;
  }

  // Sample first generated token from the last set of logits in the prefill
  size_t last_token_offset =
      (prompt_tokens.size() - 1) * config_.vocab_size * sizeof(float);
  const auto &logits_buf = scheduler_->get_logits();
  log_d2h_trace(trace_any, "logits", 0, -1, logits_buf, last_token_offset,
                config_.vocab_size * sizeof(float));
  if (!scheduler_->get_logits().copy_to_host_offset(
          logits_host.data(), last_token_offset,
          config_.vocab_size * sizeof(float), err)) {
    return output;
  }

  if (trace_readout || trace_prefill_decode) {
    const size_t token_index = prompt_tokens.size() > 0 ? (prompt_tokens.size() - 1) : 0;
    const size_t hidden_offset = token_index * config_.dim * sizeof(float);
    const auto &hidden_buf = scheduler_->get_hidden_state();
    log_d2h_trace(trace_any, "hidden", 0, -1, hidden_buf, hidden_offset,
                  config_.dim * sizeof(float));
    if (!scheduler_->get_hidden_state().copy_to_host_offset(
            hidden_host.data(), hidden_offset,
            config_.dim * sizeof(float), err)) {
      return output;
    }
    const F32Stats hstats = stats_f32(hidden_host.data(), config_.dim);
    const uint64_t hhash = hash_f32(hidden_host.data(), config_.dim);
    const F32Stats lstats = stats_f32(logits_host.data(), config_.vocab_size);
    const uint64_t lhash = hash_f32(logits_host.data(), config_.vocab_size);
    const uintptr_t hidden_ptr = reinterpret_cast<uintptr_t>(scheduler_->get_hidden_state().data());
    const uintptr_t logits_ptr = reinterpret_cast<uintptr_t>(scheduler_->get_logits().data());
    if (trace_readout && trace_readout_out) {
      log_readout(trace_readout_out, "prefill", 0, token_index, hidden_offset,
                  hidden_ptr, hstats, hhash, last_token_offset, logits_ptr,
                  lstats, lhash, config_.vocab_size);
    }
    if (trace_prefill_decode && trace_prefill_decode_out) {
      log_readout(trace_prefill_decode_out, "prefill", 0, token_index, hidden_offset,
                  hidden_ptr, hstats, hhash, last_token_offset, logits_ptr,
                  lstats, lhash, config_.vocab_size);
    }
  }
  if (trace_landscape && trace_landscape_out) {
    log_landscape(trace_landscape_out, 0, logits_host, landscape_topk);
  }

  int32_t next_token = sample(logits_host.data(), config_.vocab_size, params);
  output.push_back(next_token);

  if (env_flag("GRETA_TRACE_LAYER")) {
    const size_t pos_id = prompt_tokens.size() > 0 ? (prompt_tokens.size() - 1) : 0;
    const size_t seq_len = prompt_tokens.size();
    const size_t tokens_total = output.size();
    const int32_t token_in = prompt_tokens.empty() ? -1 : prompt_tokens.back();
    layer_trace_emit_step_header(0, pos_id, seq_len, tokens_total, token_in,
                                 next_token, config_);
  }

  if (align_callback) {
    AlignmentStep step;
    step.step = 0;
    step.token_id = next_token;
    step.logit = logits_host[next_token];
    step.logit_min = logits_host[0];
    step.logit_max = logits_host[0];
    double sum = 0;
    step.nan_count = 0;
    step.inf_count = 0;
    std::vector<std::pair<float, int>> top;
    for (size_t i = 0; i < config_.vocab_size; ++i) {
      float v = logits_host[i];
      if (std::isnan(v))
        step.nan_count++;
      else if (std::isinf(v))
        step.inf_count++;
      else {
        if (v < step.logit_min)
          step.logit_min = v;
        if (v > step.logit_max)
          step.logit_max = v;
        sum += v;
      }
      top.push_back({v, (int)i});
    }
    step.logit_mean = (float)(sum / config_.vocab_size);
    std::sort(top.rbegin(), top.rend());
    for (int i = 0; i < 10 && i < (int)config_.vocab_size; ++i) {
      step.topk_ids.push_back(top[i].second);
      step.topk_logits.push_back(top[i].first);
    }
    align_callback(step);
  }

  first_token_time = std::chrono::high_resolution_clock::now();
  first_token = false;

  // 2. Decode loop: Generate remaining tokens one-by-one
  for (int i = 1; i < params.max_tokens; ++i) {
    if (next_token == tokenizer_->eos_id())
      break;

    // Use current sequence length (output.size() - 1) as start position for the
    // new token
    int32_t last_token_id = output.back();
    if (!scheduler_->forward(&last_token_id, output.size() - 1, 1, err)) {
      break;
    }
    const bool need_logits_host = !params.greedy || align_callback || trace_readout || trace_landscape || trace_prefill_decode;
    if (params.greedy && !align_callback && !need_logits_host) {
      next_token = scheduler_->sample_greedy_gpu(err);
    } else {
      const auto &logits_buf = scheduler_->get_logits();
      log_d2h_trace(trace_any, "logits", i, -1, logits_buf, 0,
                    config_.vocab_size * sizeof(float));
      if (!scheduler_->get_logits().copy_to_host(
              logits_host.data(), config_.vocab_size * sizeof(float), err)) {
        break;
      }

      if (trace_readout || trace_prefill_decode) {
        const size_t token_index = output.size() - 1;
        const size_t hidden_offset = 0;
        const auto &hidden_buf = scheduler_->get_hidden_state();
        log_d2h_trace(trace_any, "hidden", i, -1, hidden_buf, hidden_offset,
                      config_.dim * sizeof(float));
        if (!scheduler_->get_hidden_state().copy_to_host_offset(
                hidden_host.data(), hidden_offset,
                config_.dim * sizeof(float), err)) {
          break;
        }
        const F32Stats hstats = stats_f32(hidden_host.data(), config_.dim);
        const uint64_t hhash = hash_f32(hidden_host.data(), config_.dim);
        const F32Stats lstats = stats_f32(logits_host.data(), config_.vocab_size);
        const uint64_t lhash = hash_f32(logits_host.data(), config_.vocab_size);
        const uintptr_t hidden_ptr = reinterpret_cast<uintptr_t>(scheduler_->get_hidden_state().data());
        const uintptr_t logits_ptr = reinterpret_cast<uintptr_t>(scheduler_->get_logits().data());
        if (trace_readout && trace_readout_out) {
          log_readout(trace_readout_out, "decode", i, token_index, hidden_offset,
                      hidden_ptr, hstats, hhash, 0, logits_ptr,
                      lstats, lhash, config_.vocab_size);
        }
        if (trace_prefill_decode && trace_prefill_decode_out) {
          log_readout(trace_prefill_decode_out, "decode", i, token_index, hidden_offset,
                      hidden_ptr, hstats, hhash, 0, logits_ptr,
                      lstats, lhash, config_.vocab_size);
        }
      }

      if (trace_landscape && trace_landscape_out) {
        log_landscape(trace_landscape_out, i, logits_host, landscape_topk);
      }

      next_token = sample(logits_host.data(), config_.vocab_size, params);

      if (align_callback) {
        AlignmentStep step;
        step.step = i;
        step.token_id = next_token;
        step.logit = logits_host[next_token];
        step.logit_min = logits_host[0];
        step.logit_max = logits_host[0];
        double sum = 0;
        step.nan_count = 0;
        step.inf_count = 0;
        std::vector<std::pair<float, int>> top;
        for (size_t j = 0; j < config_.vocab_size; ++j) {
          float v = logits_host[j];
          if (std::isnan(v))
            step.nan_count++;
          else if (std::isinf(v))
            step.inf_count++;
          else {
            if (v < step.logit_min)
              step.logit_min = v;
            if (v > step.logit_max)
              step.logit_max = v;
            sum += v;
          }
          top.push_back({v, (int)j});
        }
        step.logit_mean = (float)(sum / config_.vocab_size);
        std::sort(top.rbegin(), top.rend());
        for (int k = 0; k < 10 && k < (int)config_.vocab_size; ++k) {
          step.topk_ids.push_back(top[k].second);
          step.topk_logits.push_back(top[k].first);
        }
        align_callback(step);
      }
    }
    if (env_flag("GRETA_TRACE_LAYER")) {
      const size_t pos_id = output.size() - 1;
      const size_t seq_len = 1;
      const size_t tokens_total = output.size();
      const int32_t token_in = last_token_id;
      layer_trace_emit_step_header(i, pos_id, seq_len, tokens_total, token_in,
                                   next_token, config_);
    }

    output.push_back(next_token);
  }

  auto end = std::chrono::high_resolution_clock::now();
  if (stats) {
    stats->prompt_tokens = prompt_tokens.size();
    stats->generated_tokens = output.size() - prompt_tokens.size();
    stats->total_time_ms =
        std::chrono::duration<float, std::milli>(end - start).count();
    stats->time_to_first_token_ms =
        std::chrono::duration<float, std::milli>(first_token_time - start)
            .count();
    stats->tokens_per_second =
        stats->generated_tokens / (stats->total_time_ms / 1000.0f);
  }

  return output;
}

std::string Generator::generate(const std::string &prompt,
                                const SamplingParams &params,
                                GenerationStats *stats, TokenCallback callback,
                                AlignmentCallback align_callback) {
  auto prompt_tokens = tokenizer_->encode(prompt);

  std::string err;
  auto output_tokens =
      generate_tokens(prompt_tokens, params, stats, &err, align_callback);
  if (!err.empty()) {
    std::cerr << "Generation error: " << err << "\n";
  }

  std::vector<int32_t> generated(output_tokens.begin() + prompt_tokens.size(),
                                 output_tokens.end());

  if (callback) {
    for (auto id : generated) {
      callback(id, tokenizer_->decode_token(id));
    }
  }

  return tokenizer_->decode(generated);
}

} // namespace gcore::inference
