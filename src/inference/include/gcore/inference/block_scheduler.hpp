#pragma once

#include "gcore/inference/model_config.hpp"
#include "gcore/rt/hip/buffer.hpp"
#include <hip/hip_runtime.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace gcore::inference {

/// Forward declarations
class WeightLoader;

/// A single transformer block's buffers.
struct BlockBuffers {
  // Attention weights
  gcore::rt::hip::Buffer wq; // Query projection
  gcore::rt::hip::Buffer wk; // Key projection
  gcore::rt::hip::Buffer wv; // Value projection
  gcore::rt::hip::Buffer wo; // Output projection

  // MLP weights
  gcore::rt::hip::Buffer w1; // Gate projection
  gcore::rt::hip::Buffer w2; // Down projection
  gcore::rt::hip::Buffer w3; // Up projection

  // Norms
  gcore::rt::hip::Buffer attn_norm;
  gcore::rt::hip::Buffer ffn_norm;
};

/// Intermediate activation buffers (reused across layers).
struct ActivationBuffers {
  gcore::rt::hip::Buffer x;        // Current hidden state [B, S, D]
  gcore::rt::hip::Buffer residual; // For skip connections
  gcore::rt::hip::Buffer q;        // Query [B, S, H*Dh]
  gcore::rt::hip::Buffer k;        // Key [B, S, H*Dh]
  gcore::rt::hip::Buffer v;        // Value [B, S, H*Dh]
  gcore::rt::hip::Buffer attn_out; // Attention output
  gcore::rt::hip::Buffer mlp_gate; // MLP gate activation
  gcore::rt::hip::Buffer mlp_up;   // MLP up projection
  gcore::rt::hip::Buffer mlp_out;  // MLP output
  gcore::rt::hip::Buffer norm_out; // RMSNorm output

  // KV Cache (persistent across tokens)
  gcore::rt::hip::Buffer kv_cache_k; // [L, max_seq, H, Dh]
  gcore::rt::hip::Buffer kv_cache_v; // [L, max_seq, H, Dh]
  // Input tokens [B, S]
  gcore::rt::hip::Buffer tokens;
};

/// Block Scheduler: Manages execution of N transformer layers.
class BlockScheduler {
public:
  BlockScheduler();
  ~BlockScheduler();

  /// Initialize the scheduler with model configuration.
  bool init(const ModelConfig &config, std::string *err);

  /// Allocate all weight buffers for the model.
  bool allocate_weights(std::string *err);

  /// Allocate activation buffers for batch_size and max sequence length.
  bool allocate_activations(size_t batch_size, size_t max_seq_len,
                            std::string *err);

  /// Load weights from a WeightLoader into GPU buffers.
  bool load_weights(WeightLoader &loader, std::string *err);

  /// Execute a forward pass for a single layer.
  bool execute_layer(size_t layer_idx, size_t seq_start, size_t seq_len,
                     std::string *err);

  /// Execute forward pass through all layers.
  bool forward(const int32_t *tokens, size_t seq_start, size_t seq_len,
               std::string *err);

  /// Get the final hidden state buffer.
  gcore::rt::hip::Buffer &get_hidden_state();

  /// Get the final logits buffer.
  gcore::rt::hip::Buffer &get_logits();

  /// Get model configuration.
  const ModelConfig &config() const { return config_; }

  /// Get number of allocated layers.
  size_t num_layers() const { return blocks_.size(); }

private:
  ModelConfig config_;
  std::vector<BlockBuffers> blocks_;
  ActivationBuffers activations_;

  // Global weights (outside transformer blocks)
  gcore::rt::hip::Buffer token_embd_;
  gcore::rt::hip::Buffer output_norm_;
  gcore::rt::hip::Buffer output_weight_;

  // Final logits [B, S, vocab_size]
  gcore::rt::hip::Buffer logits_;

  hipStream_t stream_ = nullptr;
  bool initialized_ = false;
  size_t current_seq_pos_ = 0;
};

} // namespace gcore::inference
