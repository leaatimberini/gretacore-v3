#include "gcore/inference/block_scheduler.hpp"
#include "gcore/inference/weight_loader.hpp"

#include <iostream>

namespace gcore::inference {

BlockScheduler::BlockScheduler() = default;
BlockScheduler::~BlockScheduler() = default;

bool BlockScheduler::init(const ModelConfig &config, std::string *err) {
  config_ = config;
  blocks_.resize(config_.num_layers);
  initialized_ = true;
  return true;
}

bool BlockScheduler::allocate_weights(std::string *err) {
  if (!initialized_) {
    *err = "BlockScheduler not initialized";
    return false;
  }

  using Usage = gcore::rt::hip::BufferUsage;
  const size_t D = config_.dim;
  const size_t H = config_.hidden_dim;
  const size_t bytes_per_elem = sizeof(float); // FP32 for now

  for (size_t i = 0; i < config_.num_layers; ++i) {
    auto &b = blocks_[i];

    // Attention projections: [D, D]
    if (!b.wq.allocate(D * D * bytes_per_elem, Usage::DeviceOnly, err))
      return false;
    if (!b.wk.allocate(D * D * bytes_per_elem, Usage::DeviceOnly, err))
      return false;
    if (!b.wv.allocate(D * D * bytes_per_elem, Usage::DeviceOnly, err))
      return false;
    if (!b.wo.allocate(D * D * bytes_per_elem, Usage::DeviceOnly, err))
      return false;

    // MLP projections: [D, H] and [H, D]
    if (!b.w1.allocate(D * H * bytes_per_elem, Usage::DeviceOnly, err))
      return false;
    if (!b.w2.allocate(H * D * bytes_per_elem, Usage::DeviceOnly, err))
      return false;
    if (!b.w3.allocate(D * H * bytes_per_elem, Usage::DeviceOnly, err))
      return false;

    // Norms: [D]
    if (!b.attn_norm.allocate(D * bytes_per_elem, Usage::DeviceOnly, err))
      return false;
    if (!b.ffn_norm.allocate(D * bytes_per_elem, Usage::DeviceOnly, err))
      return false;
  }

  return true;
}

bool BlockScheduler::allocate_activations(size_t batch_size, size_t max_seq_len,
                                          std::string *err) {
  if (!initialized_) {
    *err = "BlockScheduler not initialized";
    return false;
  }

  using Usage = gcore::rt::hip::BufferUsage;
  const size_t D = config_.dim;
  const size_t H = config_.hidden_dim;
  const size_t L = config_.num_layers;
  const size_t heads = config_.num_heads;
  const size_t head_dim = config_.head_dim;
  const size_t bytes_per_elem = sizeof(float);

  // Hidden states: [B, S, D]
  size_t hidden_size = batch_size * max_seq_len * D * bytes_per_elem;
  if (!activations_.x.allocate(hidden_size, Usage::DeviceOnly, err))
    return false;
  if (!activations_.residual.allocate(hidden_size, Usage::DeviceOnly, err))
    return false;

  // QKV: [B, S, D]
  if (!activations_.q.allocate(hidden_size, Usage::DeviceOnly, err))
    return false;
  if (!activations_.k.allocate(hidden_size, Usage::DeviceOnly, err))
    return false;
  if (!activations_.v.allocate(hidden_size, Usage::DeviceOnly, err))
    return false;
  if (!activations_.attn_out.allocate(hidden_size, Usage::DeviceOnly, err))
    return false;

  // MLP intermediates: [B, S, H]
  size_t mlp_size = batch_size * max_seq_len * H * bytes_per_elem;
  if (!activations_.mlp_gate.allocate(mlp_size, Usage::DeviceOnly, err))
    return false;
  if (!activations_.mlp_up.allocate(mlp_size, Usage::DeviceOnly, err))
    return false;
  if (!activations_.mlp_out.allocate(hidden_size, Usage::DeviceOnly, err))
    return false;

  // Norm output
  if (!activations_.norm_out.allocate(hidden_size, Usage::DeviceOnly, err))
    return false;

  // KV Cache: [L, max_seq, heads, head_dim]
  size_t kv_size = L * max_seq_len * heads * head_dim * bytes_per_elem;
  if (!activations_.kv_cache_k.allocate(kv_size, Usage::DeviceOnly, err))
    return false;
  if (!activations_.kv_cache_v.allocate(kv_size, Usage::DeviceOnly, err))
    return false;

  return true;
}

bool BlockScheduler::load_weights(WeightLoader &loader, std::string *err) {
  // Load weights from loader into GPU buffers
  // This maps tensor names from the weight file to our buffer structure

  auto tensors = loader.list_tensors();
  std::cout << "Loading " << tensors.size() << " tensors from model file\n";

  // For now, just validate the tensor list exists
  // Full implementation would iterate and load each tensor
  // Example: loader.load_tensor("layers.0.attention.wq.weight", blocks_[0].wq,
  // err);

  return true;
}

bool BlockScheduler::execute_layer(size_t layer_idx, size_t seq_start,
                                   size_t seq_len, std::string *err) {
  if (layer_idx >= blocks_.size()) {
    *err = "Layer index out of range: " + std::to_string(layer_idx);
    return false;
  }

  // The actual execution would use the GraphRunner with HIP nodes:
  // 1. RMSNorm(x) -> norm_out
  // 2. GEMM(norm_out, Wq) -> q
  // 3. GEMM(norm_out, Wk) -> k
  // 4. GEMM(norm_out, Wv) -> v
  // 5. RoPE(q, k)
  // 6. KV-Cache update
  // 7. Attention(q, k, v) -> attn_out
  // 8. GEMM(attn_out, Wo) + residual -> x
  // 9. RMSNorm(x) -> norm_out
  // 10. GEMM(norm_out, W1) -> mlp_gate
  // 11. GEMM(norm_out, W3) -> mlp_up
  // 12. SiLU(mlp_gate) * mlp_up -> mlp_out
  // 13. GEMM(mlp_out, W2) + residual -> x

  // For now, we just log the execution
  // Real implementation connects to HIPGraphRunner

  return true;
}

bool BlockScheduler::forward(size_t seq_start, size_t seq_len,
                             std::string *err) {
  for (size_t i = 0; i < config_.num_layers; ++i) {
    if (!execute_layer(i, seq_start, seq_len, err)) {
      return false;
    }
  }
  current_seq_pos_ = seq_start + seq_len;
  return true;
}

gcore::rt::hip::Buffer &BlockScheduler::get_hidden_state() {
  return activations_.x;
}

} // namespace gcore::inference
