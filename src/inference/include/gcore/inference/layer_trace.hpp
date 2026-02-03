#pragma once

#include "gcore/inference/model_config.hpp"
#include "gcore/inference/trace.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace gcore::inference {

enum class LayerTracePoint : uint32_t {
  X = 1u << 0,
  NORM_OUT = 1u << 1,
  Q = 1u << 2,
  K = 1u << 3,
  V = 1u << 4,
  ATTN_OUT = 1u << 5,
  FFN_NORM = 1u << 6,
  MLP_OUT = 1u << 7,
  X_OUT = 1u << 8,
  MLP_GATE = 1u << 9,
  MLP_UP = 1u << 10,
};

struct LayerTraceConfig {
  bool enabled = false;
  std::vector<int> layers;
  uint32_t points_mask = 0;
  std::string out_path;
};

class LayerTracer {
public:
  void init_from_env(const ModelConfig &config);
  bool enabled() const { return cfg_.enabled; }
  bool should_trace_layer(int layer) const;
  bool point_enabled(const char *tag) const;
  void trace_tensor(const char *tag, int step, int layer, hipStream_t stream,
                    const float *d, uint32_t n);
  void trace_tensor_f16(const char *tag, int step, int layer, hipStream_t stream,
                        const __half *d, uint32_t n);

  const LayerTraceConfig &cfg() const { return cfg_; }

private:
  LayerTraceConfig cfg_{};
  std::ofstream out_;
};

void layer_trace_emit_step_header(int step, size_t pos_id, size_t seq_len,
                                  size_t tokens_total, int32_t token_in,
                                  int32_t token_out, const ModelConfig &cfg);

} // namespace gcore::inference
