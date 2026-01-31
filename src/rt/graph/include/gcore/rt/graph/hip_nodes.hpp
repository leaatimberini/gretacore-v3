#pragma once

#include "gcore/rt/graph/hip_graph.hpp"
#include "gcore/rt/hip/kernels/attention_kernels.hpp"
#include "gcore/rt/hip/kernels/basic_kernels.hpp"
#include "gcore/rt/hip/kernels/gemm_kernels.hpp"

namespace gcore::rt::graph {

class HIPFillNode : public HIPGraphNode {
public:
  HIPFillNode(uint32_t *data, uint32_t value, size_t n)
      : data_(data), value_(value), n_(n) {}

  bool record(hipStream_t stream, std::string *err) override {
    gcore::rt::hip::kernels::launch_fill(stream, data_, value_, n_);
    return true;
  }

  const char *name() const override { return "HIPFillNode"; }

private:
  uint32_t *data_;
  uint32_t value_;
  size_t n_;
};

class HIPRMSNormNode : public HIPGraphNode {
public:
  HIPRMSNormNode(const float *x, const float *gamma, float *y, uint32_t rows,
                 uint32_t cols, float eps)
      : x_(x), gamma_(gamma), y_(y), rows_(rows), cols_(cols), eps_(eps) {}

  bool record(hipStream_t stream, std::string *err) override {
    gcore::rt::hip::kernels::launch_rmsnorm_naive(stream, x_, gamma_, y_, rows_,
                                                  cols_, eps_);
    return true;
  }

  const char *name() const override { return "HIPRMSNormNode"; }

private:
  const float *x_;
  const float *gamma_;
  float *y_;
  uint32_t rows_;
  uint32_t cols_;
  float eps_;
};

class HIPGemmNode : public HIPGraphNode {
public:
  HIPGemmNode(const float *a, const float *b, float *c, uint32_t M, uint32_t N,
              uint32_t K, uint32_t lda, uint32_t ldb, uint32_t ldc)
      : a_(a), b_(b), c_(c), M_(M), N_(N), K_(K), lda_(lda), ldb_(ldb),
        ldc_(ldc) {}

  bool record(hipStream_t stream, std::string *err) override {
    gcore::rt::hip::kernels::launch_gemm_tiled_f32(stream, a_, b_, c_, M_, N_,
                                                   K_, lda_, ldb_, ldc_);
    return true;
  }

  const char *name() const override { return "HIPGemmNode"; }

private:
  const float *a_;
  const float *b_;
  float *c_;
  uint32_t M_, N_, K_;
  uint32_t lda_, ldb_, ldc_;
};

class HIPRoPENode : public HIPGraphNode {
public:
  HIPRoPENode(float *x, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim,
              float base)
      : x_(x), seq_len_(seq_len), num_heads_(num_heads), head_dim_(head_dim),
        base_(base) {}

  bool record(hipStream_t stream, std::string *err) override {
    gcore::rt::hip::kernels::launch_rope(stream, x_, seq_len_, num_heads_,
                                         head_dim_, base_);
    return true;
  }

  const char *name() const override { return "HIPRoPENode"; }

private:
  float *x_;
  uint32_t seq_len_, num_heads_, head_dim_;
  float base_;
};

class HIPCausalMaskNode : public HIPGraphNode {
public:
  HIPCausalMaskNode(float *data, uint32_t seq_len, float mask_val)
      : data_(data), seq_len_(seq_len), mask_val_(mask_val) {}

  bool record(hipStream_t stream, std::string *err) override {
    gcore::rt::hip::kernels::launch_causal_mask(stream, data_, seq_len_,
                                                mask_val_);
    return true;
  }

  const char *name() const override { return "HIPCausalMaskNode"; }

private:
  float *data_;
  uint32_t seq_len_;
  float mask_val_;
};

class HIPKVUpdateNode : public HIPGraphNode {
public:
  HIPKVUpdateNode(float *cache_k, float *cache_v, const float *new_k,
                  const float *new_v, uint32_t pos, uint32_t max_seq_len,
                  uint32_t num_heads, uint32_t head_dim)
      : cache_k_(cache_k), cache_v_(cache_v), new_k_(new_k), new_v_(new_v),
        pos_(pos), max_seq_len_(max_seq_len), num_heads_(num_heads),
        head_dim_(head_dim) {}

  bool record(hipStream_t stream, std::string *err) override {
    gcore::rt::hip::kernels::launch_kv_update(
        stream, cache_k_, cache_v_, new_k_, new_v_, pos_, max_seq_len_,
        num_heads_, head_dim_);
    return true;
  }

  const char *name() const override { return "HIPKVUpdateNode"; }

private:
  float *cache_k_;
  float *cache_v_;
  const float *new_k_;
  const float *new_v_;
  uint32_t pos_, max_seq_len_, num_heads_, head_dim_;
};

class HIPSoftmaxNode : public HIPGraphNode {
public:
  HIPSoftmaxNode(const float *x, float *y, uint32_t rows, uint32_t cols)
      : x_(x), y_(y), rows_(rows), cols_(cols) {}

  bool record(hipStream_t stream, std::string *err) override {
    gcore::rt::hip::kernels::launch_softmax_naive(stream, x_, y_, rows_, cols_);
    return true;
  }

  const char *name() const override { return "HIPSoftmaxNode"; }

private:
  const float *x_;
  float *y_;
  uint32_t rows_, cols_;
};

class HIPAddNode : public HIPGraphNode {
public:
  HIPAddNode(const float *a, const float *b, float *c, size_t n)
      : a_(a), b_(b), c_(c), n_(n) {}

  bool record(hipStream_t stream, std::string *err) override {
    gcore::rt::hip::kernels::launch_add(stream, a_, b_, c_, n_);
    return true;
  }

  const char *name() const override { return "HIPAddNode"; }

private:
  const float *a_;
  const float *b_;
  float *c_;
  size_t n_;
};

class HIPSiLUNode : public HIPGraphNode {
public:
  HIPSiLUNode(const float *x, float *y, size_t n) : x_(x), y_(y), n_(n) {}

  bool record(hipStream_t stream, std::string *err) override {
    gcore::rt::hip::kernels::launch_silu(stream, x_, y_, n_);
    return true;
  }

  const char *name() const override { return "HIPSiLUNode"; }

private:
  const float *x_;
  float *y_;
  size_t n_;
};

} // namespace gcore::rt::graph
