#include "gcore/inference/generator.hpp"
#include "gcore/inference/block_scheduler.hpp"
#include "gcore/inference/tokenizer.hpp"

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
  config_ = config;
  scheduler_ = scheduler;
  tokenizer_ = tokenizer;
  initialized_ = true;
  return true;
}

int32_t Generator::sample(const float *logits, size_t vocab_size,
                          const SamplingParams &params) {
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
                           std::string *err) {
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

    // After forward for S=1, we can just read from the beginning of the logits
    // buffer because scheduler currently doesn't preserve full logit history in
    // the output pointer.
    if (!scheduler_->get_logits().copy_to_host(
            logits_host.data(), config_.vocab_size * sizeof(float), err)) {
      break;
    }

    next_token = sample(logits_host.data(), config_.vocab_size, params);
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
                                GenerationStats *stats,
                                TokenCallback callback) {
  auto prompt_tokens = tokenizer_->encode(prompt);

  std::string err;
  auto output_tokens = generate_tokens(prompt_tokens, params, stats, &err);
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
