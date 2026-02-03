#include "gcore/inference/block_scheduler.hpp"
#include "gcore/compute/greta_compute.hpp"
#include "gcore/inference/weight_loader.hpp"
#include "gcore/rt/greta_runtime.hpp"
#include "gcore/rt/hip/greta_runtime_hip.hpp"
#include "gcore/rt/hip/kernels/attention_kernels.hpp"
#include "gcore/rt/hip/kernels/basic_kernels.hpp"
#include "gcore/rt/hip/kernels/fused_attention_kernels.hpp"
#include "gcore/rt/hip/kernels/fused_compute_kernels.hpp"
#include "gcore/rt/hip/kernels/gemm_kernels.hpp"
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>

#if defined(__GNUC__)
#define GRETA_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define GRETA_UNLIKELY(x) (x)
#endif

#define TRACE_ON(layer)                                                        \
  (GRETA_UNLIKELY(tracer_.should_trace_layer((int)(layer), trace_step_)))

#define PROFILE_ON() (GRETA_UNLIKELY(tracer_.profile_enabled()))

namespace gcore::inference {

static bool env_flag(const char *k) {
  const char *v = std::getenv(k);
  return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

class GretaMemoryView final : public gcore::rt::GretaMemory {
public:
  GretaMemoryView(gcore::rt::GretaMemory *base, size_t offset_bytes)
      : base_(base), offset_bytes_(offset_bytes) {}

  void *data() override {
    return static_cast<char *>(base_->data()) + offset_bytes_;
  }
  const void *data() const override {
    return static_cast<const char *>(base_->data()) + offset_bytes_;
  }
  size_t size() const override {
    size_t base_size = base_->size();
    return offset_bytes_ <= base_size ? (base_size - offset_bytes_) : 0;
  }

  gcore::rt::GretaDataType data_type() const override {
    return base_->data_type();
  }
  gcore::rt::GretaQuantInfo quant_info() const override {
    return base_->quant_info();
  }

  bool copy_from_host(const void *src, size_t size) override {
    hipError_t res = hipMemcpy(data(), src, size, hipMemcpyHostToDevice);
    return res == hipSuccess;
  }
  bool copy_to_host(void *dst, size_t size) const override {
    hipError_t res = hipMemcpy(dst, data(), size, hipMemcpyDeviceToHost);
    return res == hipSuccess;
  }

private:
  gcore::rt::GretaMemory *base_;
  size_t offset_bytes_;
};

static bool trace_kernel_sync_enabled() {
  static const bool enabled =
      env_flag("GRETA_TRACE_READOUT") || env_flag("GRETA_TRACE_PREFILL_DECODE") ||
      env_flag("GRETA_TRACE_LANDSCAPE");
  return enabled;
}

static bool trace_hip_sync(const char *name, std::string *err) {
  if (!trace_kernel_sync_enabled())
    return true;
  hipError_t res = hipDeviceSynchronize();
  if (res != hipSuccess) {
    if (err)
      *err = std::string(name) +
             " hipDeviceSynchronize failed: " + hipGetErrorString(res);
    return false;
  }
  return true;
}

static bool trace_hip_check_and_sync(const char *name, std::string *err) {
  if (!trace_kernel_sync_enabled())
    return true;
  hipError_t err_code = hipGetLastError();
  if (err_code != hipSuccess) {
    if (err)
      *err =
          std::string(name) + " hipGetLastError: " + hipGetErrorString(err_code);
    return false;
  }
  return trace_hip_sync(name, err);
}

static bool trace_embed_verify_enabled() {
  static const bool enabled = env_flag("GRETA_TRACE_EMBED_VERIFY");
  return enabled;
}

static bool embed_layout_row_major() {
  static bool cached = false;
  static bool row_major = true;
  if (cached)
    return row_major;
  cached = true;
  const char *v = std::getenv("GRETA_EMBED_LAYOUT");
  if (!v)
    return row_major;
  std::string val(v);
  if (val == "col" || val == "COL" || val == "col_major") {
    row_major = false;
  } else {
    row_major = true;
  }
  return row_major;
}

static bool trace_embed_verify_once(const int32_t *tokens, size_t seq_len,
                                    uint32_t dim, uint32_t vocab_size,
                                    const gcore::rt::hip::Buffer &token_embd,
                                    const gcore::rt::hip::Buffer &x,
                                    bool layout_row_major, std::string *err) {
  static bool done = false;
  if (!trace_embed_verify_enabled() || done)
    return true;
  done = true;

  if (seq_len == 0 || dim == 0 || vocab_size == 0)
    return true;

  size_t s = seq_len - 1;
  int32_t token = tokens[s];
  if (token < 0 || token >= static_cast<int32_t>(vocab_size)) {
    std::cout << "[GRETA_TRACE_EMBED_VERIFY] token out of range: " << token
              << std::endl;
    return true;
  }

  if (!trace_hip_sync("Embedding Verify", err))
    return false;

  std::vector<float> out(dim);
  size_t out_offset = s * static_cast<size_t>(dim) * sizeof(float);
  if (!x.copy_to_host_offset(out.data(), out_offset,
                             static_cast<size_t>(dim) * sizeof(float), err))
    return false;

  std::vector<float> row(dim);
  size_t row_offset = static_cast<size_t>(token) * dim * sizeof(float);
  if (!token_embd.copy_to_host_offset(
          row.data(), row_offset, static_cast<size_t>(dim) * sizeof(float), err))
    return false;

  std::vector<float> col(dim);
  for (uint32_t d = 0; d < dim; ++d) {
    size_t col_offset =
        (static_cast<size_t>(d) * vocab_size + static_cast<size_t>(token)) *
        sizeof(float);
    if (!token_embd.copy_to_host_offset(&col[d], col_offset, sizeof(float), err))
      return false;
  }

  double mae_row = 0.0;
  double mae_col = 0.0;
  double max_row = 0.0;
  double max_col = 0.0;
  for (uint32_t d = 0; d < dim; ++d) {
    double dr = std::fabs(static_cast<double>(out[d]) - row[d]);
    double dc = std::fabs(static_cast<double>(out[d]) - col[d]);
    mae_row += dr;
    mae_col += dc;
    if (dr > max_row)
      max_row = dr;
    if (dc > max_col)
      max_col = dc;
  }
  mae_row /= static_cast<double>(dim);
  mae_col /= static_cast<double>(dim);

  const char *layout_probe_best =
      (mae_row <= mae_col) ? "row_major_match" : "col_major_match";
  const char *layout_used = layout_row_major ? "row" : "col";
  std::cout << "[GRETA_TRACE_EMBED_VERIFY] token=" << token
            << " seq_idx=" << s << " mae_row=" << mae_row
            << " mae_col=" << mae_col << " max_row=" << max_row
            << " max_col=" << max_col << " layout_used=" << layout_used
            << " layout_probe_best=" << layout_probe_best << std::endl;

  return true;
}

BlockScheduler::BlockScheduler() = default;

BlockScheduler::~BlockScheduler() {
  if (stream_ != nullptr) {
    delete stream_;
    stream_ = nullptr;
  }
}

bool BlockScheduler::init(const ModelConfig &config, std::string *err) {
  config_ = config;
  blocks_.resize(config_.num_layers);

  std::cout << "[GRETA_SCHED] Creating stream..." << std::endl;
  stream_ = gcore::rt::GretaContext::instance().create_stream();
  if (!stream_) {
    *err = "Failed to create GRETA stream";
    return false;
  }
  std::cout << "[GRETA_SCHED] Stream created successfully" << std::endl;

  tracer_.init_from_env();
  layer_tracer_.init_from_env(config_);

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

  const char *use_int8 = std::getenv("GRETA_INT8_WEIGHTS");
  bool int8_mode = (use_int8 && std::string(use_int8) == "1");

  for (size_t i = 0; i < config_.num_layers; ++i) {
    auto &b = blocks_[i];
    if (int8_mode) {
      b.wq.allocate(D * D, Usage::DeviceOnly, gcore::rt::GretaDataType::INT8,
                    err);
      b.wk.allocate(D * D, Usage::DeviceOnly, gcore::rt::GretaDataType::INT8,
                    err);
      b.wv.allocate(D * D, Usage::DeviceOnly, gcore::rt::GretaDataType::INT8,
                    err);
      b.wo.allocate(D * D, Usage::DeviceOnly, gcore::rt::GretaDataType::INT8,
                    err);
      b.w1.allocate(D * H, Usage::DeviceOnly, gcore::rt::GretaDataType::INT8,
                    err);
      b.w2.allocate(H * D, Usage::DeviceOnly, gcore::rt::GretaDataType::INT8,
                    err);
      b.w3.allocate(D * H, Usage::DeviceOnly, gcore::rt::GretaDataType::INT8,
                    err);

      size_t nb = (D * D + 31) / 32;
      b.s_wq.allocate(nb * 4, Usage::DeviceOnly, gcore::rt::GretaDataType::FP32,
                      err);
      b.s_wk.allocate(nb * 4, Usage::DeviceOnly, gcore::rt::GretaDataType::FP32,
                      err);
      b.s_wv.allocate(nb * 4, Usage::DeviceOnly, gcore::rt::GretaDataType::FP32,
                      err);
      b.s_wo.allocate(nb * 4, Usage::DeviceOnly, gcore::rt::GretaDataType::FP32,
                      err);

      size_t nbh = (D * H + 31) / 32;
      b.s_w1.allocate(nbh * 4, Usage::DeviceOnly,
                      gcore::rt::GretaDataType::FP32, err);
      b.s_w2.allocate(nbh * 4, Usage::DeviceOnly,
                      gcore::rt::GretaDataType::FP32, err);
      b.s_w3.allocate(nbh * 4, Usage::DeviceOnly,
                      gcore::rt::GretaDataType::FP32, err);

      // Per-head scales for Attention (QKV)
      uint32_t num_heads = config_.num_heads;
      b.sh_wq.allocate(num_heads * 4, Usage::DeviceOnly,
                       gcore::rt::GretaDataType::FP32, err);
      b.sh_wk.allocate(num_heads * 4, Usage::DeviceOnly,
                       gcore::rt::GretaDataType::FP32, err);
      b.sh_wv.allocate(num_heads * 4, Usage::DeviceOnly,
                       gcore::rt::GretaDataType::FP32, err);
      b.sh_wo.allocate(num_heads * 4, Usage::DeviceOnly,
                       gcore::rt::GretaDataType::FP32, err);
    } else {
      b.wq.allocate(D * D * 2, Usage::DeviceOnly,
                    gcore::rt::GretaDataType::FP16, err);
      b.wk.allocate(D * D * 2, Usage::DeviceOnly,
                    gcore::rt::GretaDataType::FP16, err);
      b.wv.allocate(D * D * 2, Usage::DeviceOnly,
                    gcore::rt::GretaDataType::FP16, err);
      b.wo.allocate(D * D * 2, Usage::DeviceOnly,
                    gcore::rt::GretaDataType::FP16, err);
      b.w1.allocate(D * H * 2, Usage::DeviceOnly,
                    gcore::rt::GretaDataType::FP16, err);
      b.w2.allocate(H * D * 2, Usage::DeviceOnly,
                    gcore::rt::GretaDataType::FP16, err);
      b.w3.allocate(D * H * 2, Usage::DeviceOnly,
                    gcore::rt::GretaDataType::FP16, err);
    }
    b.attn_norm.allocate(D * 4, Usage::DeviceOnly,
                         gcore::rt::GretaDataType::FP32, err);
    b.ffn_norm.allocate(D * 4, Usage::DeviceOnly,
                        gcore::rt::GretaDataType::FP32, err);
  }

  token_embd_.allocate(config_.vocab_size * D * 4, Usage::DeviceOnly,
                       gcore::rt::GretaDataType::FP32, err);
  output_norm_.allocate(D * 4, Usage::DeviceOnly,
                        gcore::rt::GretaDataType::FP32, err);
  output_weight_.allocate(config_.vocab_size * D * 2, Usage::DeviceOnly,
                          gcore::rt::GretaDataType::FP16, err);

  return true;
}

bool BlockScheduler::allocate_activations(size_t batch_size, size_t max_seq_len,
                                          std::string *err) {
  if (!initialized_) {
    *err = "BlockScheduler not initialized";
    return false;
  }

  config_.max_seq_len = max_seq_len;

  using Usage = gcore::rt::hip::BufferUsage;
  const size_t D = config_.dim;
  const size_t H = config_.hidden_dim;
  const size_t L = config_.num_layers;
  const size_t heads_q = config_.num_heads;
  const size_t heads_kv =
      config_.num_heads_kv > 0 ? config_.num_heads_kv : config_.num_heads;
  const size_t head_dim = config_.head_dim;
  const size_t kv_dim = heads_kv * head_dim;

  size_t hidden_size = batch_size * max_seq_len * D * sizeof(float);
  activations_.x.allocate(hidden_size, Usage::DeviceOnly,
                          gcore::rt::GretaDataType::FP32, err);
  activations_.norm_out.allocate(hidden_size, Usage::DeviceOnly,
                                 gcore::rt::GretaDataType::FP32, err);
  activations_.q.allocate(hidden_size, Usage::DeviceOnly,
                          gcore::rt::GretaDataType::FP32, err);
  const size_t kv_hidden_size =
      batch_size * max_seq_len * kv_dim * sizeof(float);
  activations_.k.allocate(kv_hidden_size, Usage::DeviceOnly,
                          gcore::rt::GretaDataType::FP32, err);
  activations_.v.allocate(kv_hidden_size, Usage::DeviceOnly,
                          gcore::rt::GretaDataType::FP32, err);
  activations_.attn_out.allocate(hidden_size, Usage::DeviceOnly,
                                 gcore::rt::GretaDataType::FP32, err);

  size_t mlp_size = batch_size * max_seq_len * H * sizeof(float);
  activations_.mlp_gate.allocate(mlp_size, Usage::DeviceOnly,
                                 gcore::rt::GretaDataType::FP32, err);
  activations_.mlp_up.allocate(mlp_size, Usage::DeviceOnly,
                               gcore::rt::GretaDataType::FP32, err);
  activations_.mlp_out.allocate(hidden_size, Usage::DeviceOnly,
                                gcore::rt::GretaDataType::FP32, err);

  size_t kv_size = L * max_seq_len * heads_kv * head_dim * sizeof(float);
  activations_.kv_cache_k.allocate(kv_size, Usage::DeviceOnly,
                                   gcore::rt::GretaDataType::FP32, err);
  activations_.kv_cache_v.allocate(kv_size, Usage::DeviceOnly,
                                   gcore::rt::GretaDataType::FP32, err);

  size_t tokens_size = batch_size * max_seq_len * sizeof(int32_t);
  activations_.tokens.allocate(tokens_size, Usage::DeviceOnly,
                               gcore::rt::GretaDataType::FP16, err);

  activations_.d_pos.allocate(sizeof(uint32_t), Usage::DeviceOnly,
                              gcore::rt::GretaDataType::FP16, err);

  size_t logits_size =
      batch_size * max_seq_len * config_.vocab_size * sizeof(float);
  logits_.allocate(logits_size, Usage::DeviceOnly,
                   gcore::rt::GretaDataType::FP32, err);

  return true;
}

bool BlockScheduler::load_weights(WeightLoader &loader, std::string *err) {
  const char *use_int8 = std::getenv("GRETA_INT8_WEIGHTS");
  const char *use_int4 = std::getenv("GRETA_INT4_WEIGHTS");
  bool int8_mode = (use_int8 && std::string(use_int8) == "1");
  bool int4_mode = (use_int4 && std::string(use_int4) == "1");

  std::cout << "[GRETA_SCHED] Starting weight load (INT8: "
            << (int8_mode ? "ON" : "OFF")
            << ", INT4: " << (int4_mode ? "ON" : "OFF") << ")" << std::endl;

  for (size_t i = 0; i < config_.num_layers; ++i) {
    if (i % 8 == 0)
      std::cout << "[GRETA_SCHED] Loading layer " << i << "/"
                << config_.num_layers << "..." << std::endl;
    std::string prefix = "blk." + std::to_string(i) + ".";
    auto &b = blocks_[i];
    if (!loader.load_tensor(prefix + "attn_norm.weight", b.attn_norm, err))
      return false;
    if (!loader.load_tensor(prefix + "ffn_norm.weight", b.ffn_norm, err))
      return false;

    if (int4_mode) {
      if (!loader.load_tensor_int4(prefix + "attn_q.weight", b.wq, b.s_wq,
                                   b.sh_wq, err))
        return false;
      if (!loader.load_tensor_int4(prefix + "attn_k.weight", b.wk, b.s_wk,
                                   b.sh_wk, err))
        return false;
      if (!loader.load_tensor_int4(prefix + "attn_v.weight", b.wv, b.s_wv,
                                   b.sh_wv, err))
        return false;
      if (!loader.load_tensor_int4(prefix + "attn_output.weight", b.wo, b.s_wo,
                                   b.sh_wo, err))
        return false;
      if (!loader.load_tensor_int4(prefix + "ffn_gate.weight", b.w1, b.s_w1,
                                   b.sh_wo,
                                   err)) // sh_wo reused for FFN (dummy)
        return false;
      if (!loader.load_tensor_int4(prefix + "ffn_down.weight", b.w2, b.s_w2,
                                   b.sh_wo, err))
        return false;
      if (!loader.load_tensor_int4(prefix + "ffn_up.weight", b.w3, b.s_w3,
                                   b.sh_wo, err))
        return false;
    } else if (int8_mode) {
      if (!loader.load_tensor_int8(prefix + "attn_q.weight", b.wq, b.s_wq, err))
        return false;
      if (!loader.load_tensor_int8(prefix + "attn_k.weight", b.wk, b.s_wk, err))
        return false;
      if (!loader.load_tensor_int8(prefix + "attn_v.weight", b.wv, b.s_wv, err))
        return false;
      if (!loader.load_tensor_int8(prefix + "attn_output.weight", b.wo, b.s_wo,
                                   err))
        return false;
      if (!loader.load_tensor_int8(prefix + "ffn_gate.weight", b.w1, b.s_w1,
                                   err))
        return false;
      if (!loader.load_tensor_int8(prefix + "ffn_down.weight", b.w2, b.s_w2,
                                   err))
        return false;
      if (!loader.load_tensor_int8(prefix + "ffn_up.weight", b.w3, b.s_w3, err))
        return false;
    } else {
      if (!loader.load_tensor_fp16(prefix + "attn_q.weight", b.wq, err))
        return false;
      if (!loader.load_tensor_fp16(prefix + "attn_k.weight", b.wk, err))
        return false;
      if (!loader.load_tensor_fp16(prefix + "attn_v.weight", b.wv, err))
        return false;
      if (!loader.load_tensor_fp16(prefix + "attn_output.weight", b.wo, err))
        return false;
      if (!loader.load_tensor_fp16(prefix + "ffn_gate.weight", b.w1, err))
        return false;
      if (!loader.load_tensor_fp16(prefix + "ffn_down.weight", b.w2, err))
        return false;
      if (!loader.load_tensor_fp16(prefix + "ffn_up.weight", b.w3, err))
        return false;
    }
  }

  if (!loader.load_tensor("token_embd.weight", token_embd_, err))
    return false;
  if (!loader.load_tensor("output_norm.weight", output_norm_, err))
    return false;
  if (!loader.load_tensor_fp16("output.weight", output_weight_, err))
    return false;
  return true;
}

#define CHECK_HIP_KERNEL(cmd, name)                                            \
  do {                                                                         \
    cmd;                                                                       \
    hipError_t err_code = hipGetLastError();                                   \
    if (err_code != hipSuccess) {                                              \
      if (err)                                                                 \
        *err = std::string(name) +                                             \
               " launch failed: " + hipGetErrorString(err_code);               \
      return false;                                                            \
    }                                                                          \
    if (!trace_hip_sync(name, err))                                            \
      return false;                                                            \
  } while (0)

#define CHECK_GRETA(cmd, name)                                                 \
  do {                                                                         \
    if ((cmd) != gcore::rt::GretaResult::SUCCESS) {                            \
      if (err)                                                                 \
        *err = std::string(name) + " failed";                                  \
      return false;                                                            \
    }                                                                          \
    if (!trace_hip_check_and_sync(name, err))                                  \
      return false;                                                            \
  } while (0)

bool BlockScheduler::execute_layer(size_t layer_idx, size_t seq_start,
                                   size_t seq_len, std::string *err) {
  auto &b = blocks_[layer_idx];
  uint32_t D = static_cast<uint32_t>(config_.dim);
  uint32_t Hq = static_cast<uint32_t>(config_.num_heads);
  uint32_t Hkv = static_cast<uint32_t>(
      config_.num_heads_kv > 0 ? config_.num_heads_kv : config_.num_heads);
  uint32_t Dh = D / Hq;
  uint32_t hidden_dim = static_cast<uint32_t>(config_.hidden_dim);
  uint32_t S = static_cast<uint32_t>(seq_len);

  const uint32_t n_x = S * D;
  const uint32_t n_mlp = S * hidden_dim;
  const uint32_t n_kv = S * Hkv * Dh;
  const uint32_t kv_dim = Hkv * Dh;

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

  size_t offset = (size_t)layer_idx * (size_t)config_.max_seq_len *
                  (size_t)Hkv * (size_t)Dh;
  float *cache_k =
      static_cast<float *>(activations_.kv_cache_k.data()) + offset;
  float *cache_v =
      static_cast<float *>(activations_.kv_cache_v.data()) + offset;
  const uint32_t *d_pos =
      static_cast<const uint32_t *>(activations_.d_pos.data());

  hipStream_t hip_stream =
      static_cast<gcore::rt::hip::GretaStreamHip *>(stream_)->handle();

  const bool trace_layer = GRETA_UNLIKELY(layer_tracer_.enabled());
  if (trace_layer) {
    layer_tracer_.trace_tensor("x", trace_step_, static_cast<int>(layer_idx),
                               hip_stream, x, n_x);
  }

  gcore::rt::GretaEvent *start = nullptr, *stop = nullptr;
  if (PROFILE_ON()) {
    start = gcore::rt::GretaContext::instance().create_event();
    stop = gcore::rt::GretaContext::instance().create_event();
    start->record(stream_);
  }

  bool profile_attn = (std::getenv("GRETA_PROFILE_ATTN") != nullptr);
  gcore::rt::GretaEvent *ev_q_start = nullptr, *ev_q_end = nullptr;
  gcore::rt::GretaEvent *ev_k_start = nullptr, *ev_k_end = nullptr;
  gcore::rt::GretaEvent *ev_v_start = nullptr, *ev_v_end = nullptr;
  gcore::rt::GretaEvent *ev_rope_start = nullptr, *ev_rope_end = nullptr;
  gcore::rt::GretaEvent *ev_kv_start = nullptr, *ev_kv_end = nullptr;
  gcore::rt::GretaEvent *ev_core_start = nullptr, *ev_core_end = nullptr;

  if (profile_attn) {
    auto &ctx = gcore::rt::GretaContext::instance();
    ev_q_start = ctx.create_event();
    ev_q_end = ctx.create_event();
    ev_k_start = ctx.create_event();
    ev_k_end = ctx.create_event();
    ev_v_start = ctx.create_event();
    ev_v_end = ctx.create_event();
    ev_rope_start = ctx.create_event();
    ev_rope_end = ctx.create_event();
    ev_kv_start = ctx.create_event();
    ev_kv_end = ctx.create_event();
    ev_core_start = ctx.create_event();
    ev_core_end = ctx.create_event();
  }

  const char *use_fused_env = std::getenv("GRETA_USE_FUSED_RMSNORM");
  bool use_fused =
      (use_fused_env && std::string(use_fused_env) == "1") && (S == 1) &&
      (Hkv == Hq);

  if (use_fused) {
    CHECK_HIP_KERNEL(launch_fused_rmsnorm_qkv_gemv_f16(
                         hip_stream, x, attn_norm,
                         static_cast<const __half *>(b.wq.data()),
                         static_cast<const __half *>(b.wk.data()),
                         static_cast<const __half *>(b.wv.data()), q, k, v, D,
                         config_.rms_eps),
                     "Fused RMSNorm+QKV");

    if (trace_layer) {
      layer_tracer_.trace_tensor("q", trace_step_,
                                 static_cast<int>(layer_idx), hip_stream,
                                 q, n_x);
      layer_tracer_.trace_tensor("k", trace_step_,
                                 static_cast<int>(layer_idx), hip_stream,
                                 k, n_kv);
      layer_tracer_.trace_tensor("v", trace_step_,
                                 static_cast<int>(layer_idx), hip_stream,
                                 v, n_kv);
    }
  } else {
    CHECK_HIP_KERNEL(launch_rmsnorm_naive(hip_stream, x, attn_norm, norm_out, S,
                                          D, config_.rms_eps),
                     "RMSNorm (Attn)");

    if (trace_layer) {
      layer_tracer_.trace_tensor("norm_out", trace_step_,
                                 static_cast<int>(layer_idx), hip_stream,
                                 norm_out, n_x);
    }

    if (TRACE_ON(layer_idx)) {
      launch_debug_tensor_stats(hip_stream, "L.attn_rmsnorm.norm_out", norm_out,
                                n_x);
    }

    if (profile_attn)
      ev_q_start->record(stream_);
    CHECK_GRETA(
        gcore::compute::GretaCompute::gemm(stream_, &activations_.norm_out,
                                           &b.wq, &activations_.q, S, D, D),
        "GEMM Q");
    if (profile_attn)
      ev_q_end->record(stream_);

    if (profile_attn)
      ev_k_start->record(stream_);
    CHECK_GRETA(
        gcore::compute::GretaCompute::gemm(stream_, &activations_.norm_out,
                                           &b.wk, &activations_.k, S, kv_dim,
                                           D),
        "GEMM K");
    if (profile_attn)
      ev_k_end->record(stream_);

    if (profile_attn)
      ev_v_start->record(stream_);
    CHECK_GRETA(
        gcore::compute::GretaCompute::gemm(stream_, &activations_.norm_out,
                                           &b.wv, &activations_.v, S, kv_dim,
                                           D),
        "GEMM V");
    if (profile_attn)
      ev_v_end->record(stream_);

    if (trace_layer) {
      layer_tracer_.trace_tensor("q", trace_step_,
                                 static_cast<int>(layer_idx), hip_stream,
                                 q, n_x);
      layer_tracer_.trace_tensor("k", trace_step_,
                                 static_cast<int>(layer_idx), hip_stream,
                                 k, n_kv);
      layer_tracer_.trace_tensor("v", trace_step_,
                                 static_cast<int>(layer_idx), hip_stream,
                                 v, n_kv);
    }
  }

  if (TRACE_ON(layer_idx)) {
    launch_debug_tensor_stats(hip_stream, "L.q_proj.q", q, n_x);
    launch_debug_tensor_stats(hip_stream, "L.k_proj.k", k, n_kv);
    launch_debug_tensor_stats(hip_stream, "L.v_proj.v", v, n_kv);
  }

  uint32_t pos = static_cast<uint32_t>(seq_start);
  const char *use_fused_attn_env = std::getenv("GRETA_USE_FUSED_ATTENTION");
  bool use_fused_attn =
      (use_fused_attn_env && std::string(use_fused_attn_env) == "1") &&
      (S == 1) && (Hkv == Hq);

  if (profile_attn)
    ev_rope_start->record(stream_);

  if (use_fused_attn) {
    CHECK_HIP_KERNEL(launch_fused_rope_kv_update_decode(
                         hip_stream, q, k, v, cache_k, cache_v, d_pos,
                         config_.max_seq_len, Hq, Dh, config_.rope_base),
                     "Fused RoPE+KV Update");
    CHECK_HIP_KERNEL(
        launch_rope(hip_stream, q, S, Hq, Dh, config_.rope_base, d_pos),
        "RoPE Q (Fused KV)");
  } else {
    if (S == 1) {
      CHECK_HIP_KERNEL(
          launch_rope(hip_stream, q, S, Hq, Dh, config_.rope_base, d_pos),
          "RoPE Q");
      CHECK_HIP_KERNEL(
          launch_rope(hip_stream, k, S, Hkv, Dh, config_.rope_base, d_pos),
          "RoPE K");
    } else {
      CHECK_HIP_KERNEL(
          launch_rope(hip_stream, q, S, Hq, Dh, config_.rope_base, pos),
          "RoPE Q");
      CHECK_HIP_KERNEL(
          launch_rope(hip_stream, k, S, Hkv, Dh, config_.rope_base, pos),
          "RoPE K");
    }
  }
  if (profile_attn)
    ev_rope_end->record(stream_);

  if (profile_attn)
    ev_kv_start->record(stream_);
  if (!use_fused_attn) {
    if (S == 1) {
      CHECK_HIP_KERNEL(launch_kv_update(hip_stream, cache_k, cache_v, k, v,
                                        d_pos, config_.max_seq_len, Hkv, Dh),
                       "KV Update");
    } else {
      for (uint32_t s = 0; s < S; ++s) {
        CHECK_HIP_KERNEL(launch_kv_update(hip_stream, cache_k, cache_v,
                                          k + s * kv_dim, v + s * kv_dim,
                                          pos + s, config_.max_seq_len, Hkv,
                                          Dh),
                         "KV Update");
      }
    }
  }
  if (profile_attn)
    ev_kv_end->record(stream_);

  float scale = 1.0f / sqrtf(static_cast<float>(Dh));
  if (profile_attn)
    ev_core_start->record(stream_);

  if (S == 1) {
    CHECK_GRETA(gcore::compute::GretaCompute::attention_decode(
                    stream_, &activations_.q, &activations_.kv_cache_k,
                    &activations_.kv_cache_v, &activations_.d_pos,
                    &activations_.attn_out, Hq, Hkv, Dh, S,
                    config_.max_seq_len, scale, config_.rope_base),
                "Attention Core");
  } else {
    CHECK_HIP_KERNEL(launch_flash_attention_prefill(hip_stream, q, k, v,
                                                    attn_out, S, Hq, Hkv, Dh,
                                                    scale, true),
                     "Flash Attention Prefill");
  }

  if (profile_attn)
    ev_core_end->record(stream_);

  if (profile_attn && layer_idx == 0) {
    stream_->synchronize();
    float q_ms = ev_q_end->elapsed_time_since(ev_q_start);
    float k_ms = ev_k_end->elapsed_time_since(ev_k_start);
    float v_ms = ev_v_end->elapsed_time_since(ev_v_start);
    float rope_ms = ev_rope_end->elapsed_time_since(ev_rope_start);
    float kv_ms = ev_kv_end->elapsed_time_since(ev_kv_start);
    float core_ms = ev_core_end->elapsed_time_since(ev_core_start);

    printf(
        "[ATTN_PROF] Q:%.3f K:%.3f V:%.3f RoPE:%.3f KV:%.3f Core:%.3f (ms)\n",
        q_ms, k_ms, v_ms, rope_ms, kv_ms, core_ms);

    delete ev_q_start;
    delete ev_q_end;
    delete ev_k_start;
    delete ev_k_end;
    delete ev_v_start;
    delete ev_v_end;
    delete ev_rope_start;
    delete ev_rope_end;
    delete ev_kv_start;
    delete ev_kv_end;
    delete ev_core_start;
    delete ev_core_end;
  }

  if (trace_layer) {
    layer_tracer_.trace_tensor("attn_out", trace_step_,
                               static_cast<int>(layer_idx), hip_stream,
                               attn_out, n_x);
  }

  if (TRACE_ON(layer_idx)) {
    launch_debug_tensor_stats(hip_stream, "L.attn.attn_out", attn_out, n_x);
  }

  CHECK_GRETA(
      gcore::compute::GretaCompute::gemm(stream_, &activations_.attn_out, &b.wo,
                                         &activations_.mlp_out, S, D, D),
      "GEMM O");
  CHECK_HIP_KERNEL(launch_add(hip_stream, x, mlp_out, x, S * D),
                   "Residual (Attn)");

  CHECK_HIP_KERNEL(
      launch_rmsnorm_naive(hip_stream, x,
                           static_cast<const float *>(b.ffn_norm.data()),
                           norm_out, S, D, config_.rms_eps),
      "RMSNorm (FFN)");

  if (trace_layer) {
    layer_tracer_.trace_tensor("ffn_norm", trace_step_,
                               static_cast<int>(layer_idx), hip_stream,
                               norm_out, n_x);
  }

  if (TRACE_ON(layer_idx)) {
    launch_debug_tensor_stats(hip_stream, "L.ffn_rmsnorm.norm_out", norm_out,
                              n_x);
  }

  const char *use_fused_ffn_env = std::getenv("GRETA_USE_FUSED_FFN");
  bool use_fused_ffn =
      (use_fused_ffn_env && std::string(use_fused_ffn_env) == "1") && (S == 1);

  if (use_fused_ffn) {
    CHECK_HIP_KERNEL(
        launch_fused_ffn_front_f16(
            hip_stream, norm_out, static_cast<const __half *>(b.w1.data()),
            static_cast<const __half *>(b.w3.data()), mlp_gate, D, hidden_dim),
        "Fused FFN Front");
  } else {
    CHECK_GRETA(gcore::compute::GretaCompute::gemm(
                    stream_, &activations_.norm_out, &b.w1,
                    &activations_.mlp_gate, S, hidden_dim, D),
                "GEMM W1");
    CHECK_GRETA(gcore::compute::GretaCompute::gemm(
                    stream_, &activations_.norm_out, &b.w3,
                    &activations_.mlp_up, S, hidden_dim, D),
                "GEMM W3");

    if (trace_layer) {
      layer_tracer_.trace_tensor("mlp_gate", trace_step_,
                                 static_cast<int>(layer_idx), hip_stream,
                                 mlp_gate, n_mlp);
      layer_tracer_.trace_tensor("mlp_up", trace_step_,
                                 static_cast<int>(layer_idx), hip_stream,
                                 mlp_up, n_mlp);
    }

    if (TRACE_ON(layer_idx)) {
      launch_debug_tensor_stats(hip_stream, "L.ffn.gate.nonfused", mlp_gate,
                                n_mlp);
      launch_debug_tensor_stats(hip_stream, "L.ffn.up.nonfused", mlp_up, n_mlp);
    }

    CHECK_HIP_KERNEL(
        launch_silu(hip_stream, mlp_gate, mlp_gate, S * hidden_dim), "SiLU");
    CHECK_HIP_KERNEL(
        launch_mul(hip_stream, mlp_gate, mlp_up, mlp_gate, S * hidden_dim),
        "Mul");
  }

  if (TRACE_ON(layer_idx)) {
    launch_debug_tensor_stats(hip_stream, "L.ffn.gate_after_mul", mlp_gate,
                              n_mlp);
  }

  CHECK_GRETA(gcore::compute::GretaCompute::gemm(
                  stream_, &activations_.mlp_gate, &b.w2, &activations_.mlp_out,
                  S, D, hidden_dim),
              "GEMM W2");

  if (trace_layer) {
    layer_tracer_.trace_tensor("mlp_out", trace_step_,
                               static_cast<int>(layer_idx), hip_stream,
                               mlp_out, n_x);
  }

  if (TRACE_ON(layer_idx)) {
    launch_debug_tensor_stats(hip_stream, "L.ffn.out", mlp_out, n_x);
  }

  CHECK_HIP_KERNEL(launch_add(hip_stream, x, mlp_out, x, S * D),
                   "Residual (FFN)");

  if (PROFILE_ON()) {
    stop->record(stream_);
    stream_->synchronize();
    float ms = stop->elapsed_time_since(start);
    printf("[PROFILE] Layer %zu total: %.3f ms\n", layer_idx, ms);
    delete start;
    delete stop;
  }
  if (trace_layer) {
    layer_tracer_.trace_tensor("x_out", trace_step_,
                               static_cast<int>(layer_idx), hip_stream,
                               x, n_x);
  }

  if (PROFILE_ON()) {
    stop->record(stream_);
    stream_->synchronize();
    float ms = stop->elapsed_time_since(start);
    printf("[PROFILE] Layer %zu total: %.3f ms\n", layer_idx, ms);
    delete start;
    delete stop;
  }

  if (TRACE_ON(layer_idx)) {
    launch_debug_tensor_stats(hip_stream, "L.residual.x", x, n_x);
  }

  return true;
}

bool BlockScheduler::forward(const int32_t *tokens, size_t seq_start,
                             size_t seq_len, std::string *err) {
  using namespace gcore::rt::hip::kernels;
  uint32_t S = static_cast<uint32_t>(seq_len);
  uint32_t D = static_cast<uint32_t>(config_.dim);
  uint32_t V = static_cast<uint32_t>(config_.vocab_size);

  activations_.tokens.copy_to_device(tokens, S * sizeof(int32_t), err);

  float *x = static_cast<float *>(activations_.x.data());
  const float *embd_w = static_cast<const float *>(token_embd_.data());
  const int32_t *d_tokens =
      static_cast<const int32_t *>(activations_.tokens.data());
  const bool embed_row_major = embed_layout_row_major();

  hipStream_t hip_stream =
      static_cast<gcore::rt::hip::GretaStreamHip *>(stream_)->handle();

  CHECK_HIP_KERNEL(launch_embedding_lookup(hip_stream, d_tokens, embd_w, x, S,
                                           D, config_.vocab_size,
                                           embed_row_major),
                   "Embedding Lookup");
  if (!trace_embed_verify_once(tokens, seq_len, D, V, token_embd_,
                               activations_.x, embed_row_major, err))
    return false;

  uint32_t pos = static_cast<uint32_t>(seq_start);
  activations_.d_pos.copy_to_device(&pos, sizeof(uint32_t), err);

  const char *use_graph_env = std::getenv("GRETA_GRAPH");
  bool use_graph =
      (use_graph_env && std::string(use_graph_env) == "1") && (S == 1);

  if (use_graph && graph_captured_) {
    const char *profile_blocks = std::getenv("GRETA_PROFILE_BLOCKS");
    if (profile_blocks && std::string(profile_blocks) == "1") {
      printf("[GRETA_L0_AUDIT] Graph Launch (Decode Step)\n");
    }
    CHECK_GRETA(graph_->launch(stream_), "Graph Launch");
  } else {
    if (use_graph && !graph_captured_) {
      if (!graph_)
        graph_ = gcore::rt::GretaContext::instance().create_graph();
      graph_->capture_start(stream_);
    }

    for (size_t i = 0; i < config_.num_layers; ++i) {
      if (!execute_layer(i, seq_start, seq_len, err))
        return false;
    }

    float *norm_out = static_cast<float *>(activations_.norm_out.data());
    const float *onorm_w = static_cast<const float *>(output_norm_.data());

    CHECK_HIP_KERNEL(launch_rmsnorm_naive(hip_stream, x, onorm_w, norm_out, S,
                                          D, config_.rms_eps),
                     "Final RMSNorm");

    size_t logits_offset_bytes =
        static_cast<size_t>(seq_start) * static_cast<size_t>(V) * sizeof(float);
    size_t logits_bytes =
        static_cast<size_t>(S) * static_cast<size_t>(V) * sizeof(float);
    if (logits_offset_bytes + logits_bytes > logits_.size()) {
      if (err) {
        *err = "LM Head logits offset out of range: offset=" +
               std::to_string(logits_offset_bytes) +
               " bytes=" + std::to_string(logits_bytes) +
               " alloc=" + std::to_string(logits_.size());
      }
      return false;
    }
    GretaMemoryView logits_view(&logits_, logits_offset_bytes);
    CHECK_GRETA(
        gcore::compute::GretaCompute::gemm(stream_, &activations_.norm_out,
                                           &output_weight_, &logits_view, S, V,
                                           D),
        "LM Head");

    if (use_graph && !graph_captured_) {
      graph_->capture_end(stream_);
      CHECK_GRETA(graph_->instantiate(), "Graph Instantiate");
      graph_captured_ = true;
    }
  }

  stream_->synchronize();
  trace_step_++;
  return true;
}

gcore::rt::hip::Buffer &BlockScheduler::get_hidden_state() {
  return activations_.x;
}

gcore::rt::hip::Buffer &BlockScheduler::get_norm_out() {
  return activations_.norm_out;
}

const gcore::rt::hip::Buffer &BlockScheduler::get_norm_out() const {
  return activations_.norm_out;
}

const gcore::rt::hip::Buffer &BlockScheduler::get_logits() const {
  return logits_;
}

int32_t BlockScheduler::sample_greedy_gpu(size_t logits_offset_bytes,
                                          std::string *err) {
  (void)err;
  int32_t top_id = 0;
  hipStream_t hip_stream =
      static_cast<gcore::rt::hip::GretaStreamHip *>(stream_)->handle();
  const float *logits_base = static_cast<const float *>(logits_.data());
  const size_t offset_elems = logits_offset_bytes / sizeof(float);
  rt::hip::kernels::launch_argmax(hip_stream, logits_base + offset_elems,
                                  config_.vocab_size, &top_id);
  return top_id;
}

} // namespace gcore::inference
