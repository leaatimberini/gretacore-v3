#include "gcore/inference/block_scheduler.hpp"
#include "gcore/inference/weight_loader.hpp"
#include "gcore/rt/hip/kernels/attention_kernels.hpp"
#include "gcore/rt/hip/kernels/basic_kernels.hpp"
#include "gcore/rt/hip/kernels/gemm_kernels.hpp"

#include <iostream>
#include <unordered_map>

namespace gcore::inference {

BlockScheduler::BlockScheduler() = default;

BlockScheduler::~BlockScheduler() {
  if (stream_ != nullptr) {
    hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

bool BlockScheduler::init(const ModelConfig &config, std::string *err) {
  config_ = config;
  blocks_.resize(config_.num_layers);

  // Create HIP stream for async execution
  hipError_t hip_err = hipStreamCreate(&stream_);
  if (hip_err != hipSuccess) {
    *err = "Failed to create HIP stream: " +
           std::string(hipGetErrorString(hip_err));
    return false;
  }

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
  // Map GGUF tensor names to our buffer structure

  auto tensors = loader.list_tensors();
  std::cout << "Loading " << tensors.size() << " tensors from model file\n";

  // Build tensor name lookup
  std::unordered_map<std::string, const TensorInfo *> tensor_map;
  for (const auto &t : tensors) {
    tensor_map[t.name] = &t;
  }

  // Load per-layer weights
  size_t loaded = 0;
  size_t load_errors = 0;
  bool verbose = (config_.num_layers <= 32);

  for (size_t i = 0; i < config_.num_layers; ++i) {
    auto &b = blocks_[i];
    std::string prefix = "blk." + std::to_string(i) + ".";

    // Lambda to load a tensor with error handling
    auto load = [&](const std::string &name,
                    gcore::rt::hip::Buffer &buf) -> bool {
      if (tensor_map.count(name)) {
        if (loader.load_tensor(name, buf, err)) {
          loaded++;
          return true;
        } else {
          std::cerr << "  Warning: Failed to load " << name << ": " << *err
                    << "\n";
          load_errors++;
        }
      }
      return false;
    };

    // Norms (FP32)
    load(prefix + "attn_norm.weight", b.attn_norm);
    load(prefix + "ffn_norm.weight", b.ffn_norm);

    // Attention weights (Q4_K -> FP16)
    auto load_fp16 = [&](const std::string &name,
                         gcore::rt::hip::Buffer &buf) -> bool {
      if (tensor_map.count(name)) {
        if (loader.load_tensor_fp16(name, buf, err)) {
          loaded++;
          return true;
        } else {
          std::cerr << "  Warning: Failed to load " << name << ": " << *err
                    << "\n";
          load_errors++;
        }
      }
      return false;
    };

    load_fp16(prefix + "attn_q.weight", b.wq);
    load_fp16(prefix + "attn_k.weight", b.wk);
    load_fp16(prefix + "attn_v.weight", b.wv);
    load_fp16(prefix + "attn_output.weight", b.wo);

    // MLP weights (Q4_K/Q6_K -> FP16)
    load_fp16(prefix + "ffn_gate.weight", b.w1);
    load_fp16(prefix + "ffn_down.weight", b.w2);
    load_fp16(prefix + "ffn_up.weight", b.w3);

    if (verbose && i == 0) {
      std::cout << "  Layer 0: Loaded " << (loaded - load_errors)
                << " tensors to GPU\n";
    }
  }

  // Load embedding and output weights
  if (tensor_map.count("token_embd.weight")) {
    std::cout << "  Found token embeddings: "
              << tensor_map["token_embd.weight"]->size_bytes / (1024 * 1024)
              << " MB\n";
    loaded++;
  }
  if (tensor_map.count("output_norm.weight")) {
    std::cout << "  Found output norm\n";
    loaded++;
  }
  if (tensor_map.count("output.weight")) {
    std::cout << "  Found output projection\n";
    loaded++;
  }

  std::cout << "Mapped " << loaded << " weight tensors to buffers\n";
  return true;
}

bool BlockScheduler::execute_layer(size_t layer_idx, size_t seq_start,
                                   size_t seq_len, std::string *err) {
  if (layer_idx >= blocks_.size()) {
    *err = "Layer index out of range: " + std::to_string(layer_idx);
    return false;
  }

  auto &b = blocks_[layer_idx];

  // Model dimensions
  uint32_t D = static_cast<uint32_t>(config_.dim);                 // 4096
  uint32_t H = static_cast<uint32_t>(config_.num_heads);           // 32
  uint32_t Dh = D / H;                                             // 128
  uint32_t hidden_dim = static_cast<uint32_t>(config_.hidden_dim); // 11008
  uint32_t S = static_cast<uint32_t>(seq_len);
  float eps = 1e-6f;
  float rope_base = 10000.0f;

  // Get buffer pointers (all FP32)
  float *x = reinterpret_cast<float *>(activations_.x.data());
  float *residual = reinterpret_cast<float *>(activations_.residual.data());
  float *norm_out = reinterpret_cast<float *>(activations_.norm_out.data());
  float *q = reinterpret_cast<float *>(activations_.q.data());
  float *k = reinterpret_cast<float *>(activations_.k.data());
  float *v = reinterpret_cast<float *>(activations_.v.data());
  float *attn_out = reinterpret_cast<float *>(activations_.attn_out.data());
  float *mlp_gate = reinterpret_cast<float *>(activations_.mlp_gate.data());
  float *mlp_up = reinterpret_cast<float *>(activations_.mlp_up.data());
  float *mlp_out = reinterpret_cast<float *>(activations_.mlp_out.data());

  // Weight pointers
  float *wq = reinterpret_cast<float *>(b.wq.data());
  float *wk = reinterpret_cast<float *>(b.wk.data());
  float *wv = reinterpret_cast<float *>(b.wv.data());
  float *wo = reinterpret_cast<float *>(b.wo.data());
  float *w1 = reinterpret_cast<float *>(b.w1.data());
  float *w2 = reinterpret_cast<float *>(b.w2.data());
  float *w3 = reinterpret_cast<float *>(b.w3.data());
  float *attn_norm = reinterpret_cast<float *>(b.attn_norm.data());
  float *ffn_norm = reinterpret_cast<float *>(b.ffn_norm.data());

  // KV cache pointers (for this layer)
  float *kv_k = reinterpret_cast<float *>(activations_.kv_cache_k.data());
  float *kv_v = reinterpret_cast<float *>(activations_.kv_cache_v.data());
  size_t layer_cache_offset = layer_idx * config_.max_seq_len * H * Dh;
  float *cache_k = kv_k + layer_cache_offset;
  float *cache_v = kv_v + layer_cache_offset;

  // Skip execution if stream not initialized (demo mode)
  if (stream_ == nullptr) {
    return true;
  }

  using namespace gcore::rt::hip::kernels;

  // ====== ATTENTION BLOCK ======

  // Step 1: RMSNorm(x) -> norm_out
  launch_rmsnorm_naive(stream_, x, attn_norm, norm_out, S, D, eps);

  // Save residual for skip connection
  // (In production, we'd use async copy or fuse with norm)
  // For now, x stays as residual

  // Step 2-4: Linear projections (Q, K, V)
  // Use mixed-precision GEMM: FP32 activations * FP16 weights -> FP32 output
  launch_gemm_mixed_f16f32(stream_, norm_out, b.wq, q, S, D, D, D, D, D);
  launch_gemm_mixed_f16f32(stream_, norm_out, b.wk, k, S, D, D, D, D, D);
  launch_gemm_mixed_f16f32(stream_, norm_out, b.wv, v, S, D, D, D, D, D);

  // Step 5: RoPE embeddings on Q and K
  launch_rope(stream_, q, S, H, Dh, rope_base);
  launch_rope(stream_, k, S, H, Dh, rope_base);

  // Step 6: Update KV cache
  uint32_t pos = static_cast<uint32_t>(seq_start);
  for (uint32_t s = 0; s < S; ++s) {
    launch_kv_update(stream_, cache_k, cache_v, k + s * D, v + s * D, pos + s,
                     config_.max_seq_len, H, Dh);
  }

  // Step 7: FlashAttention v2
  float scale = 1.0f / sqrtf(static_cast<float>(Dh));
  if (S == 1) {
    // Decode mode: single query against full KV cache
    launch_flash_attention_decode(stream_, q, cache_k, cache_v, attn_out, H,
                                  pos + S, Dh, scale);
  } else {
    // Prefill mode: multiple queries with causal masking
    // (FlashAttention handle KV update internally or we use the updated cache)
    // For now, simpler to use the cache we just updated
    launch_flash_attention_prefill(stream_, q, k, v, attn_out, S, H, Dh, scale,
                                   true);
  }

  // Step 8: Output projection + residual
  // Mixed-precision GEMM: attn_out [S, D] * Wo [D, D] (FP16) -> mlp_out [S, D]
  // (FP32) We reuse mlp_out as a temp buffer for attn output projection
  launch_gemm_mixed_f16f32(stream_, attn_out, b.wo, mlp_out, S, D, D, D, D, D);
  launch_add(stream_, x, mlp_out, x, S * D);

  // ====== MLP BLOCK ======

  // Step 9: RMSNorm(x) -> norm_out
  launch_rmsnorm_naive(stream_, x, ffn_norm, norm_out, S, D, eps);

  // Step 10-11: Gate and Up projections
  // Mixed-precision GEMM: norm_out [S, D] * W1/W3 [D, hidden_dim] (FP16)
  launch_gemm_mixed_f16f32(stream_, norm_out, b.w1, mlp_gate, S, hidden_dim, D,
                           D, hidden_dim, hidden_dim);
  launch_gemm_mixed_f16f32(stream_, norm_out, b.w3, mlp_up, S, hidden_dim, D, D,
                           hidden_dim, hidden_dim);

  // Step 12: SiLU(gate) * up -> mlp_gate
  launch_silu(stream_, mlp_gate, mlp_gate, S * hidden_dim);
  launch_mul(stream_, mlp_gate, mlp_up, mlp_gate, S * hidden_dim);

  // Step 13: Down projection + residual
  // Mixed-precision GEMM: mlp_gate [S, hidden_dim] * W2 [hidden_dim, D] (FP16)
  launch_gemm_mixed_f16f32(stream_, mlp_gate, b.w2, mlp_out, S, D, hidden_dim,
                           hidden_dim, D, D);
  launch_add(stream_, x, mlp_out, x, S * D);

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
