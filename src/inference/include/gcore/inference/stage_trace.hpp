#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

namespace gcore::inference {

struct StageTraceConfig {
  bool enabled = false;
  std::vector<int> layers;
  std::vector<std::string> points;
  std::vector<std::string> phases;
  uint32_t sample = 256;
  const char *out_path = nullptr;
  bool debug_input = false;
};

struct StageInputMeta {
  const char *src_kind = nullptr;
  uint32_t token_index_used = 0;
  size_t offset_bytes = 0;
  uintptr_t ptr = 0;
  size_t alloc_bytes = 0;
  uint32_t prompt_tokens = 0;
  uint32_t kv_pos = 0;
  uint32_t decode_step = 0;
  uint32_t token_id = 0;
  const char *route = nullptr;
};

struct StageLogitsStats {
  uint64_t hash = 0;
  float min = 0.0f;
  float max = 0.0f;
  float mean = 0.0f;
  int top1_id = -1;
  float top1_logit = 0.0f;
  int top2_id = -1;
  float top2_logit = 0.0f;
  float gap = 0.0f;
  uintptr_t logits_ptr = 0;
  size_t logits_offset_bytes = 0;
  size_t vocab = 0;
};

StageTraceConfig stage_trace_config();

bool stage_trace_enabled();
bool stage_trace_layer_selected(size_t layer_idx, size_t num_layers);
bool stage_trace_point_enabled(const char *point);
bool stage_trace_phase_enabled(const char *phase);
uint32_t stage_trace_sample();
const char *stage_trace_out_path();
bool stage_trace_debug_input();

void stage_trace_tensor(const char *point, const char *phase,
                        const char *prompt_id, size_t layer, uint32_t step,
                        uint32_t pos_id, uint32_t seq_len,
                        uint32_t tokens_total, const float *base,
                        size_t stride_elems, size_t token_index,
                        hipStream_t stream,
                        const StageInputMeta *input_meta = nullptr);

void stage_trace_logits(const char *phase, const char *prompt_id, uint32_t step,
                        uint32_t pos_id, uint32_t seq_len,
                        uint32_t tokens_total, const StageLogitsStats &stats);

} // namespace gcore::inference
