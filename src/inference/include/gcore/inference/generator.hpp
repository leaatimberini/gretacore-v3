#pragma once

#include "gcore/inference/block_scheduler.hpp"
#include "gcore/inference/model_config.hpp"
#include "gcore/inference/tokenizer.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace gcore::inference {

/// Sampling parameters for text generation.
struct SamplingParams {
  float temperature = 1.0f; // Temperature for softmax
  int32_t top_k = 50;       // Top-K sampling (0 = disabled)
  float top_p = 1.0f;       // Top-P nucleus sampling (1.0 = disabled)
  int32_t max_tokens = 128; // Maximum tokens to generate
  int32_t seed = 42;        // Random seed for reproducibility
  bool greedy = false;      // Use greedy decoding (argmax)
};

/// Statistics from generation.
struct GenerationStats {
  size_t prompt_tokens = 0;
  size_t generated_tokens = 0;
  double total_time_ms = 0.0;
  double tokens_per_second = 0.0;
  double time_to_first_token_ms = 0.0;
};

/// Callback for streaming tokens during generation.
using TokenCallback =
    std::function<void(int32_t token_id, const std::string &text)>;

/// Text Generator: Autoregressive inference loop.
class Generator {
public:
  Generator();
  ~Generator();

  /// Initialize with model configuration and pre-allocated scheduler.
  bool init(const ModelConfig &config, BlockScheduler *scheduler,
            Tokenizer *tokenizer, std::string *err);

  /// Generate text from a prompt.
  std::string generate(const std::string &prompt, const SamplingParams &params,
                       GenerationStats *stats = nullptr,
                       TokenCallback callback = nullptr);

  /// Generate tokens from an already-encoded prompt.
  std::vector<int32_t>
  generate_tokens(const std::vector<int32_t> &prompt_tokens,
                  const SamplingParams &params,
                  GenerationStats *stats = nullptr, std::string *err = nullptr);

  /// Sample next token from logits.
  int32_t sample(const float *logits, size_t vocab_size,
                 const SamplingParams &params);

private:
  ModelConfig config_;
  BlockScheduler *scheduler_ = nullptr;
  Tokenizer *tokenizer_ = nullptr;
  bool initialized_ = false;

  // Internal state
  size_t current_pos_ = 0;
};

} // namespace gcore::inference
