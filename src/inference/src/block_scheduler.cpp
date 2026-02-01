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

  for (size_t i = 0; i < config_.num_layers; ++i) {
    auto &b = blocks_[i];
    if (!b.wq.allocate(D * D * sizeof(__half), Usage::DeviceOnly, err))
      return false;
    if (!b.wk.allocate(D * D * sizeof(__half), Usage::DeviceOnly, err))
      return false;
    if (!b.wv.allocate(D * D * sizeof(__half), Usage::DeviceOnly, err))
      return false;
    if (!b.wo.allocate(D * D * sizeof(__half), Usage::DeviceOnly, err))
      return false;
    if (!b.w1.allocate(D * H * sizeof(__half), Usage::DeviceOnly, err))
      return false;
    if (!b.w2.allocate(H * D * sizeof(__half), Usage::DeviceOnly, err))
      return false;
    if (!b.w3.allocate(D * H * sizeof(__half), Usage::DeviceOnly, err))
      return false;
    if (!b.attn_norm.allocate(D * sizeof(float), Usage::DeviceOnly, err))
      return false;
    if (!b.ffn_norm.allocate(D * sizeof(float), Usage::DeviceOnly, err))
      return false;
  }

  if (!token_embd_.allocate(config_.vocab_size * D * sizeof(float),
                            Usage::DeviceOnly, err))
    return false;
  if (!output_norm_.allocate(D * sizeof(float), Usage::DeviceOnly, err))
    return false;
  if (!output_weight_.allocate(config_.vocab_size * D * sizeof(__half),
                               Usage::DeviceOnly, err))
    return false;

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

  size_t hidden_size = batch_size * max_seq_len * D * sizeof(float);
  if (!activations_.x.allocate(hidden_size, Usage::DeviceOnly, err))
    return false;
  if (!activations_.norm_out.allocate(hidden_size, Usage::DeviceOnly, err))
    return false;
  if (!activations_.q.allocate(hidden_size, Usage::DeviceOnly, err))
    return false;
  if (!activations_.k.allocate(hidden_size, Usage::DeviceOnly, err))
    return false;
  if (!activations_.v.allocate(hidden_size, Usage::DeviceOnly, err))
    return false;
  if (!activations_.attn_out.allocate(hidden_size, Usage::DeviceOnly, err))
    return false;

  size_t mlp_size = batch_size * max_seq_len * H * sizeof(float);
  if (!activations_.mlp_gate.allocate(mlp_size, Usage::DeviceOnly, err))
    return false;
  if (!activations_.mlp_up.allocate(mlp_size, Usage::DeviceOnly, err))
    return false;
  if (!activations_.mlp_out.allocate(hidden_size, Usage::DeviceOnly, err))
    return false;

  size_t kv_size = L * max_seq_len * heads * head_dim * sizeof(float);
  if (!activations_.kv_cache_k.allocate(kv_size, Usage::DeviceOnly, err))
    return false;
  if (!activations_.kv_cache_v.allocate(kv_size, Usage::DeviceOnly, err))
    return false;

  size_t tokens_size = batch_size * max_seq_len * sizeof(int32_t);
  if (!activations_.tokens.allocate(tokens_size, Usage::DeviceOnly, err))
    return false;

  size_t logits_size =
      batch_size * max_seq_len * config_.vocab_size * sizeof(float);
  if (!logits_.allocate(logits_size, Usage::DeviceOnly, err))
    return false;

  return true;
}

bool BlockScheduler::load_weights(WeightLoader &loader, std::string *err) {
  auto tensors = loader.list_tensors();
  std::unordered_map<std::string, const TensorInfo *> tensor_map;
  for (const auto &t : tensors)
    tensor_map[t.name] = &t;

  for (size_t i = 0; i < config_.num_layers; ++i) {
    std::string prefix = "blk." + std::to_string(i) + ".";
    auto &b = blocks_[i];
    loader.load_tensor(prefix + "attn_norm.weight", b.attn_norm, err);
    loader.load_tensor(prefix + "ffn_norm.weight", b.ffn_norm, err);
    loader.load_tensor_fp16(prefix + "attn_q.weight", b.wq, err);
    loader.load_tensor_fp16(prefix + "attn_k.weight", b.wk, err);
    loader.load_tensor_fp16(prefix + "attn_v.weight", b.wv, err);
    loader.load_tensor_fp16(prefix + "attn_output.weight", b.wo, err);
    loader.load_tensor_fp16(prefix + "ffn_gate.weight", b.w1, err);
    loader.load_tensor_fp16(prefix + "ffn_down.weight", b.w2, err);
    loader.load_tensor_fp16(prefix + "ffn_up.weight", b.w3, err);
  }

  loader.load_tensor("token_embd.weight", token_embd_, err);
  loader.load_tensor("output_norm.weight", output_norm_, err);
  loader.load_tensor_fp16("output.weight", output_weight_, err);
  std::cout << "Mapped tensors to buffers\n";
  return true;
}

#define CHECK_HIP_KERNEL(cmd, name)                                            \
  do {                                                                         \
    cmd;                                                                       \
    hipError_t err_code = hipStreamSynchronize(stream_);                       \
    if (err_code != hipSuccess) {                                              \
      std::cout << "[HIP KERNEL ERROR] " << name << " failed with "            \
                << hipGetErrorString(err_code) << std::endl;                   \
      if (err)                                                                 \
        *err = std::string(name) + " failed: " + hipGetErrorString(err_code);  \
      return false;                                                            \
    }                                                                          \
  } while (0)

bool BlockScheduler::execute_layer(size_t layer_idx, size_t seq_start,
                                   size_t seq_len, std::string *err) {
  if (layer_idx >= blocks_.size())
    return false;
  auto &b = blocks_[layer_idx];

  uint32_t D = static_cast<uint32_t>(config_.dim);
  uint32_t H = static_cast<uint32_t>(config_.num_heads);
  uint32_t Dh = D / H;
  uint32_t hidden_dim = static_cast<uint32_t>(config_.hidden_dim);
  uint32_t S = static_cast<uint32_t>(seq_len);
  float eps = config_.rms_eps;
  float rope_base = config_.rope_base;

  using namespace gcore::rt::hip::kernels;
  float *x = static_cast<float *>(activations_.x.data());
  float *norm_out = static_cast<float *>(activations_.norm_out.data());
  float *q = static_cast<float *>(activations_.q.data());
  float *k = static_cast<float *>(activations_.k.data());
  float *v = static_cast<float *>(activations_.v.data());
  float *attn_out = static_cast<float *>(activations_.attn_out.data());
  float *mlp_gate = static_cast<float *>(activations_.mlp_gate.data());
  float *mlp_up = static_cast<float *>(activations_.mlp_up.data());
  float *mlp_out = static_cast<float *>(activations_.mlp_out.data());

  const float *attn_norm = static_cast<const float *>(b.attn_norm.data());
  const float *ffn_norm = static_cast<const float *>(b.ffn_norm.data());
  const __half *wq = static_cast<const __half *>(b.wq.data());
  const __half *wk = static_cast<const __half *>(b.wk.data());
  const __half *wv = static_cast<const __half *>(b.wv.data());

  float *cache_k_base = static_cast<float *>(activations_.kv_cache_k.data());
  float *cache_v_base = static_cast<float *>(activations_.kv_cache_v.data());
  size_t offset =
      (size_t)layer_idx * (size_t)config_.max_seq_len * (size_t)H * (size_t)Dh;
  float *cache_k = cache_k_base + offset;
  float *cache_v = cache_v_base + offset;

  if (layer_idx % 8 == 0) {
    std::cout << "  [LAYER " << layer_idx << "] cache_k=" << (void *)cache_k
              << " base=" << (void *)cache_k_base << " off=" << offset
              << std::endl;
  }

  CHECK_HIP_KERNEL(
      launch_rmsnorm_naive(stream_, x, attn_norm, norm_out, S, D, eps),
      "RMSNorm (Attn)");
  CHECK_HIP_KERNEL(
      launch_gemm_mixed_f16f32(stream_, norm_out, wq, q, S, D, D, D, D, D),
      "GEMM Q");
  CHECK_HIP_KERNEL(
      launch_gemm_mixed_f16f32(stream_, norm_out, wk, k, S, D, D, D, D, D),
      "GEMM K");
  CHECK_HIP_KERNEL(
      launch_gemm_mixed_f16f32(stream_, norm_out, wv, v, S, D, D, D, D, D),
      "GEMM V");

  CHECK_HIP_KERNEL(launch_rope(stream_, q, S, H, Dh, rope_base), "RoPE Q");
  CHECK_HIP_KERNEL(launch_rope(stream_, k, S, H, Dh, rope_base), "RoPE K");

  uint32_t pos = static_cast<uint32_t>(seq_start);
  for (uint32_t s = 0; s < S; ++s) {
    CHECK_HIP_KERNEL(launch_kv_update(stream_, cache_k, cache_v, k + s * D,
                                      v + s * D, pos + s, config_.max_seq_len,
                                      H, Dh),
                     "KV Update");
  }

  float scale = 1.0f / sqrtf(static_cast<float>(Dh));
  if (S == 1) {
    CHECK_HIP_KERNEL(
        launch_flash_attention_decode(stream_, q, cache_k, cache_v, attn_out, H,
                                      pos + 1, config_.max_seq_len, Dh, scale),
        "FlashAttn Decode");
  } else {
    CHECK_HIP_KERNEL(launch_flash_attention_prefill(stream_, q, k, v, attn_out,
                                                    S, H, Dh, scale, true),
                     "FlashAttn Prefill");
  }

  const __half *wo = static_cast<const __half *>(b.wo.data());
  CHECK_HIP_KERNEL(launch_gemm_mixed_f16f32(stream_, attn_out, wo, mlp_out, S,
                                            D, D, D, D, D),
                   "GEMM O");
  CHECK_HIP_KERNEL(launch_add(stream_, x, mlp_out, x, S * D),
                   "Residual (Attn)");

  CHECK_HIP_KERNEL(
      launch_rmsnorm_naive(stream_, x, ffn_norm, norm_out, S, D, eps),
      "RMSNorm (FFN)");
  const __half *w1 = static_cast<const __half *>(b.w1.data());
  const __half *w2 = static_cast<const __half *>(b.w2.data());
  const __half *w3 = static_cast<const __half *>(b.w3.data());
  CHECK_HIP_KERNEL(launch_gemm_mixed_f16f32(stream_, norm_out, w1, mlp_gate, S,
                                            hidden_dim, D, D, hidden_dim,
                                            hidden_dim),
                   "GEMM W1");
  CHECK_HIP_KERNEL(launch_gemm_mixed_f16f32(stream_, norm_out, w3, mlp_up, S,
                                            hidden_dim, D, D, hidden_dim,
                                            hidden_dim),
                   "GEMM W3");

  CHECK_HIP_KERNEL(launch_silu(stream_, mlp_gate, mlp_gate, S * hidden_dim),
                   "SiLU");
  CHECK_HIP_KERNEL(
      launch_mul(stream_, mlp_gate, mlp_up, mlp_gate, S * hidden_dim), "Mul");

  CHECK_HIP_KERNEL(launch_gemm_mixed_f16f32(stream_, mlp_gate, w2, mlp_out, S,
                                            D, hidden_dim, hidden_dim, D, D),
                   "GEMM W2");
  CHECK_HIP_KERNEL(launch_add(stream_, x, mlp_out, x, S * D), "Residual (FFN)");

  return true;
}

bool BlockScheduler::forward(const int32_t *tokens, size_t seq_start,
                             size_t seq_len, std::string *err) {
  using namespace gcore::rt::hip::kernels;
  uint32_t S = static_cast<uint32_t>(seq_len);
  uint32_t D = static_cast<uint32_t>(config_.dim);
  uint32_t V = static_cast<uint32_t>(config_.vocab_size);

  std::cout << "[BLOCK_SCHEDULER] Forward Start: pos=" << seq_start
            << " len=" << seq_len << std::endl;

  if (!activations_.tokens.copy_to_device(tokens, S * sizeof(int32_t), err))
    return false;

  float *x = static_cast<float *>(activations_.x.data());
  const float *embd_w = static_cast<const float *>(token_embd_.data());
  const int32_t *d_tokens =
      static_cast<const int32_t *>(activations_.tokens.data());

  CHECK_HIP_KERNEL(launch_embedding_lookup(stream_, d_tokens, embd_w, x, S, D),
                   "Embedding Lookup");

  for (size_t i = 0; i < config_.num_layers; ++i) {
    if (!execute_layer(i, seq_start, seq_len, err))
      return false;
  }

  float *norm_out = static_cast<float *>(activations_.norm_out.data());
  float *logits = static_cast<float *>(logits_.data());
  const float *onorm_w = static_cast<const float *>(output_norm_.data());
  const __half *ow_w = static_cast<const __half *>(output_weight_.data());

  CHECK_HIP_KERNEL(launch_rmsnorm_naive(stream_, x, onorm_w, norm_out, S, D,
                                        config_.rms_eps),
                   "Final RMSNorm");
  CHECK_HIP_KERNEL(launch_gemm_mixed_f16f32(stream_, norm_out, ow_w, logits, S,
                                            V, D, D, V, V),
                   "Logits GEMM");

  hipError_t err_code = hipStreamSynchronize(stream_);
  if (err_code != hipSuccess)
    return false;
  std::cout << "[BLOCK_SCHEDULER] Forward Done." << std::endl;
  return true;
}

gcore::rt::hip::Buffer &BlockScheduler::get_hidden_state() {
  return activations_.x;
}
gcore::rt::hip::Buffer &BlockScheduler::get_logits() { return logits_; }

} // namespace gcore::inference
