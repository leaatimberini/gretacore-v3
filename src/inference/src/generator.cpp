#include "gcore/inference/generator.hpp"
#include "gcore/inference/block_scheduler.hpp"
#include "gcore/inference/tokenizer.hpp"
#include "gcore/inference/trace.hpp"
#include "gcore/inference/layer_trace.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

namespace gcore::inference {

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

  // 1. Prefill: Process all prompt tokens at once
  if (!scheduler_->forward(prompt_tokens.data(), 0, prompt_tokens.size(),
                           err)) {
    return output;
  }

  // Sample first generated token from the last set of logits in the prefill
  size_t last_token_offset =
      (prompt_tokens.size() - 1) * config_.vocab_size * sizeof(float);
  if (!scheduler_->get_logits().copy_to_host_offset(
          logits_host.data(), last_token_offset,
          config_.vocab_size * sizeof(float), err)) {
    return output;
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

    if (params.greedy && !align_callback) {
      next_token = scheduler_->sample_greedy_gpu(err);
    } else {
      if (!scheduler_->get_logits().copy_to_host(
              logits_host.data(), config_.vocab_size * sizeof(float), err)) {
        break;
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
