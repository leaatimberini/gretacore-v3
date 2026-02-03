#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gcore::inference {

/// Model configuration for Llama-style architectures.
struct ModelConfig {
  uint32_t dim = 4096;         // Hidden dimension
  uint32_t num_heads = 32;     // Number of attention heads
  uint32_t num_heads_kv = 32;  // Number of KV heads
  uint32_t num_layers = 32;    // Number of transformer layers
  uint32_t vocab_size = 32000; // Vocabulary size
  uint32_t hidden_dim = 11008; // MLP hidden dimension (Llama uses 11008 for 7B)
  uint32_t head_dim = 128;     // dim / num_heads
  uint32_t max_seq_len = 2048; // Maximum sequence length
  float rope_base = 10000.0f;  // RoPE base frequency
  float rms_eps = 1e-5f;       // RMSNorm epsilon
  std::vector<std::string> vocabulary; // Global vocabulary

  /// Create a Llama-2-7B configuration.
  static ModelConfig llama2_7b() {
    ModelConfig cfg;
    cfg.dim = 4096;
    cfg.num_heads = 32;
    cfg.num_heads_kv = 32;
    cfg.num_layers = 32;
    cfg.vocab_size = 32000;
    cfg.hidden_dim = 11008;
    cfg.head_dim = 128;
    cfg.max_seq_len = 2048;
    return cfg;
  }

  /// Create a Llama-2-13B configuration.
  static ModelConfig llama2_13b() {
    ModelConfig cfg;
    cfg.dim = 5120;
    cfg.num_heads = 40;
    cfg.num_heads_kv = 40;
    cfg.num_layers = 40;
    cfg.vocab_size = 32000;
    cfg.hidden_dim = 13824;
    cfg.head_dim = 128;
    cfg.max_seq_len = 4096;
    return cfg;
  }

  /// Compute total parameter count (approximate).
  size_t param_count() const {
    // Embedding + Attention + MLP + Output
    size_t embed = vocab_size * dim;
    size_t attn_per_layer = 4 * dim * dim;       // Q, K, V, O
    size_t mlp_per_layer = 3 * dim * hidden_dim; // gate, up, down
    size_t output = dim * vocab_size;
    return embed + num_layers * (attn_per_layer + mlp_per_layer) + output;
  }
};

} // namespace gcore::inference
