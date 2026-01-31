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
                           const SamplingParams &params,
                           GenerationStats *stats) {

  std::vector<int32_t> output = prompt_tokens;
  auto start = std::chrono::high_resolution_clock::now();
  auto first_token_time = start;
  bool first_token = true;

  // Dummy logits for testing (uniform distribution)
  std::vector<float> dummy_logits(config_.vocab_size,
                                  1.0f / config_.vocab_size);

  for (int i = 0; i < params.max_tokens; ++i) {
    // In real implementation:
    // 1. Run forward pass through scheduler
    // 2. Get logits from final layer
    // 3. Sample next token

    // For now, use dummy logits
    int32_t next_token =
        sample(dummy_logits.data(), config_.vocab_size, params);

    if (first_token) {
      first_token_time = std::chrono::high_resolution_clock::now();
      first_token = false;
    }

    output.push_back(next_token);

    // Check for EOS
    if (next_token == tokenizer_->eos_id()) {
      break;
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
  auto output_tokens = generate_tokens(prompt_tokens, params, stats);

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
