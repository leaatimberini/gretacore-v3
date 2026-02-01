#include "gcore/inference/generator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

namespace gcore::inference {

Generator::Generator() = default;
Generator::~Generator() = default;

bool Generator::init(const ModelConfig &config, BlockScheduler *scheduler,
                     Tokenizer *tokenizer, std::string *err) {
  if (!scheduler || !tokenizer) {
    *err = "Scheduler and tokenizer must be provided";
    return false;
  }

  config_ = config;
  scheduler_ = scheduler;
  tokenizer_ = tokenizer;
  initialized_ = true;
  current_pos_ = 0;

  return true;
}

int32_t Generator::sample(const float *logits, size_t vocab_size,
                          const SamplingParams &params) {
  if (params.greedy) {
    // Argmax
    int32_t max_idx = 0;
    float max_val = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
      if (logits[i] > max_val) {
        max_val = logits[i];
        max_idx = static_cast<int32_t>(i);
      }
    }
    return max_idx;
  }

  // Temperature scaling
  std::vector<float> scaled(vocab_size);
  float temp = std::max(params.temperature, 1e-6f);
  for (size_t i = 0; i < vocab_size; ++i) {
    scaled[i] = logits[i] / temp;
  }

  // Softmax
  float max_logit = *std::max_element(scaled.begin(), scaled.end());
  float sum = 0.0f;
  for (auto &s : scaled) {
    s = std::exp(s - max_logit);
    sum += s;
  }
  for (auto &s : scaled) {
    s /= sum;
  }

  // Top-K filtering
  if (params.top_k > 0 && params.top_k < static_cast<int32_t>(vocab_size)) {
    std::vector<std::pair<float, int32_t>> probs_idx(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
      probs_idx[i] = {scaled[i], static_cast<int32_t>(i)};
    }
    std::partial_sort(probs_idx.begin(), probs_idx.begin() + params.top_k,
                      probs_idx.end(), std::greater<>());

    // Filter and renormalize
    float norm = 0.0f;
    for (int k = 0; k < params.top_k; ++k) {
      norm += probs_idx[k].first;
    }
    for (size_t i = 0; i < vocab_size; ++i) {
      scaled[i] = 0.0f;
    }
    for (int k = 0; k < params.top_k; ++k) {
      scaled[probs_idx[k].second] = probs_idx[k].first / norm;
    }
  }

  // Random sampling
  static std::mt19937 gen(params.seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(gen);
  float cumsum = 0.0f;
  for (size_t i = 0; i < vocab_size; ++i) {
    cumsum += scaled[i];
    if (cumsum >= r) {
      return static_cast<int32_t>(i);
    }
  }
  return static_cast<int32_t>(vocab_size - 1);
}

std::vector<int32_t>
Generator::generate_tokens(const std::vector<int32_t> &prompt_tokens,
                           const SamplingParams &params, GenerationStats *stats,
                           std::string *err) {

  std::vector<int32_t> output = prompt_tokens;
  auto start = std::chrono::high_resolution_clock::now();
  auto first_token_time = start;
  bool first_token = true;

  std::vector<float> logits_host(config_.vocab_size);

  std::vector<float> logits_host(config_.vocab_size);

  // 1. Prefill
  if (!scheduler_->forward(0, prompt_tokens.size(), err)) {
    return {};
  }

  // Sample first generated token from the last logits of the prefill
  // Logits buffer has [seq_len, vocab_size]. We need index [seq_len - 1].
  size_t last_token_offset =
      (prompt_tokens.size() - 1) * config_.vocab_size * sizeof(float);
  if (!scheduler_->get_logits().copy_to_host_offset(
          logits_host.data(), last_token_offset,
          config_.vocab_size * sizeof(float), err)) {
    return {};
  }

  int32_t next_token = sample(logits_host.data(), config_.vocab_size, params);
  output.push_back(next_token);

  first_token_time = std::chrono::high_resolution_clock::now();
  first_token = false;

  if (callback) {
    callback(next_token, tokenizer_->decode_token(next_token));
  }

  // 2. Decode loop (for remaining tokens)
  for (int i = 1; i < params.max_tokens; ++i) {
    if (next_token == tokenizer_->eos_id())
      break;

    // Forward pass for 1 token at the current position
    // (output.size() - 1) is the position of the last generated token
    if (!scheduler_->forward(output.size() - 1, 1, err)) {
      break;
    }

    // For S=1, logits are at the beginning of the buffer (if scheduler resets
    // offset)
    if (!scheduler_->get_logits().copy_to_host(
            logits_host.data(), config_.vocab_size * sizeof(float), err)) {
      break;
    }

    next_token = sample(logits_host.data(), config_.vocab_size, params);
    output.push_back(next_token);

    if (callback) {
      callback(next_token, tokenizer_->decode_token(next_token));
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  if (stats) {
    stats->prompt_tokens = prompt_tokens.size();
    stats->generated_tokens = output.size() - prompt_tokens.size();
    stats->total_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    stats->time_to_first_token_ms =
        std::chrono::duration<double, std::milli>(first_token_time - start)
            .count();
    if (stats->total_time_ms > 0) {
      stats->tokens_per_second =
          (stats->generated_tokens * 1000.0) / stats->total_time_ms;
    }
  }

  return output;
}

std::string Generator::generate(const std::string &prompt,
                                const SamplingParams &params,
                                GenerationStats *stats,
                                TokenCallback callback) {
  // Encode prompt
  auto prompt_tokens = tokenizer_->encode(prompt);

  // Generate tokens
  std::string err;
  auto output_tokens = generate_tokens(prompt_tokens, params, stats, &err);
  if (!err.empty()) {
    std::cerr << "Generation error: " << err << "\n";
  }

  // Decode output (skip prompt tokens)
  std::vector<int32_t> generated(output_tokens.begin() + prompt_tokens.size(),
                                 output_tokens.end());

  // Call callback for each generated token if provided
  if (callback) {
    for (auto token : generated) {
      callback(token, tokenizer_->decode_token(token));
    }
  }

  return tokenizer_->decode(generated);
}

} // namespace gcore::inference
