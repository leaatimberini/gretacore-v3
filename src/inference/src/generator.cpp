#include "gcore/inference/generator.hpp"
#include "gcore/compute/greta_compute.hpp"
#include "gcore/inference/block_scheduler.hpp"
#include "gcore/inference/layer_trace.hpp"
#include "gcore/inference/stage_trace.hpp"
#include "gcore/inference/tokenizer.hpp"
#include "gcore/inference/trace.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <hip/hip_fp16.h>
#include <iostream>
#include <random>
#include <sstream>

namespace gcore::inference {

struct F32Stats {
  float min = 0.0f;
  float max = 0.0f;
  float mean = 0.0f;
};

static F32Stats stats_f32(const float *p, size_t n) {
  F32Stats s{};
  if (n == 0)
    return s;
  s.min = p[0];
  s.max = p[0];
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    float v = p[i];
    if (v < s.min)
      s.min = v;
    if (v > s.max)
      s.max = v;
    sum += v;
  }
  s.mean = static_cast<float>(sum / n);
  return s;
}

static uint64_t hash_f32(const float *p, size_t n) {
  const size_t count = (n < 256) ? n : 256;
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < count; ++i) {
    uint32_t v;
    std::memcpy(&v, &p[i], sizeof(uint32_t));
    h ^= static_cast<uint64_t>(v);
    h *= 1099511628211ull;
  }
  return h;
}

struct Top2Logits {
  int top1_id = -1;
  float top1_logit = 0.0f;
  int top2_id = -1;
  float top2_logit = 0.0f;
};

static Top2Logits top2_logits(const float *p, size_t n) {
  Top2Logits out{};
  if (n == 0)
    return out;
  out.top1_id = 0;
  out.top1_logit = p[0];
  out.top2_id = 0;
  out.top2_logit = -1e38f;
  for (size_t i = 1; i < n; ++i) {
    float v = p[i];
    if (v > out.top1_logit) {
      out.top2_logit = out.top1_logit;
      out.top2_id = out.top1_id;
      out.top1_logit = v;
      out.top1_id = static_cast<int>(i);
    } else if (v > out.top2_logit) {
      out.top2_logit = v;
      out.top2_id = static_cast<int>(i);
    }
  }
  return out;
}

static float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0) {
    if (mant == 0)
      return sign ? -0.0f : 0.0f;
    exp = 1;
    while ((mant & 0x400) == 0) {
      mant <<= 1;
      exp--;
    }
    mant &= ~0x400;
  } else if (exp == 31) {
    return sign ? -INFINITY : INFINITY;
  }
  uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  float result;
  std::memcpy(&result, &f, 4);
  return result;
}

static double sumsq_f32(const float *p, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double v = static_cast<double>(p[i]);
    sum += v * v;
  }
  return sum;
}

static const char *dtype_name(gcore::rt::GretaDataType t) {
  switch (t) {
  case gcore::rt::GretaDataType::FP32:
    return "FP32";
  case gcore::rt::GretaDataType::FP16:
    return "FP16";
  case gcore::rt::GretaDataType::BF16:
    return "BF16";
  case gcore::rt::GretaDataType::INT8:
    return "INT8";
  case gcore::rt::GretaDataType::INT4:
    return "INT4";
  default:
    return "UNKNOWN";
  }
}

static std::vector<int> topk_ids(const float *logits, size_t n, int k) {
  std::vector<std::pair<float, int>> v;
  v.reserve(n);
  for (size_t i = 0; i < n; ++i)
    v.push_back({logits[i], static_cast<int>(i)});
  std::sort(v.rbegin(), v.rend());
  const int kk = std::min<int>(k, (int)v.size());
  std::vector<int> ids;
  ids.reserve(kk);
  for (int i = 0; i < kk; ++i)
    ids.push_back(v[i].second);
  return ids;
}

static bool cpu_probe_lm_head(const gcore::rt::hip::Buffer &weights,
                              const std::vector<float> &rms_vec, size_t dim,
                              const std::vector<int> &ids,
                              std::vector<float> &out_logits,
                              int &out_top1_id) {
  if (weights.data_type() != gcore::rt::GretaDataType::FP16)
    return false;
  if (rms_vec.size() != dim)
    return false;

  std::vector<uint16_t> row(dim);
  out_logits.clear();
  out_logits.reserve(ids.size());

  float best_logit = -1e38f;
  int best_id = -1;

  for (int id : ids) {
    size_t offset = static_cast<size_t>(id) * dim * sizeof(uint16_t);
    if (offset + dim * sizeof(uint16_t) > weights.size())
      continue;
    if (!weights.copy_to_host_offset(row.data(), offset, dim * sizeof(uint16_t),
                                     nullptr))
      continue;
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
      float w = fp16_to_fp32(row[i]);
      sum += static_cast<double>(rms_vec[i]) * static_cast<double>(w);
    }
    float logit = static_cast<float>(sum);
    out_logits.push_back(logit);
    if (logit > best_logit) {
      best_logit = logit;
      best_id = id;
    }
  }

  out_top1_id = best_id;
  return best_id >= 0;
}

static void append_line(const char *path, const std::string &line);

static bool trace_lmhead_w_verify_once(const gcore::rt::hip::Buffer &weights,
                                       const std::vector<float> &rms_host,
                                       const std::vector<float> &logits_host,
                                       const ModelConfig &config,
                                       const char *out_path, std::string *err) {
  static bool done = false;
  if (done)
    return true;
  if (!out_path || !*out_path)
    return true;
  done = true;

  if (weights.data_type() != gcore::rt::GretaDataType::FP16) {
    std::ostringstream oss;
    oss << "{\"event\":\"lmhead_w_verify\",\"status\":\"unsupported_dtype\","
           "\"dtype\":\""
        << dtype_name(weights.data_type()) << "\"}";
    append_line(out_path, oss.str());
    return true;
  }

  const std::vector<int> token_ids = {79, 96965, 12345};
  const size_t dim = config.dim;
  const size_t vocab = config.vocab_size;
  const size_t elem_size = sizeof(__half);
  const size_t total_bytes = weights.size();
  const size_t total_elems = elem_size > 0 ? (total_bytes / elem_size) : 0;

  auto copy_elem = [&](size_t offset_elems, __half *out) -> bool {
    const size_t offset_bytes = offset_elems * elem_size;
    if (offset_bytes + elem_size > total_bytes) {
      return false;
    }
    return weights.copy_to_host_offset(out, offset_bytes, elem_size, err);
  };

  for (int token_id : token_ids) {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab)
      continue;

    std::vector<__half> row_dim(dim);
    size_t row_offset = static_cast<size_t>(token_id) * dim * elem_size;
    if (row_offset + dim * elem_size > total_bytes) {
      std::ostringstream oss;
      oss << "{\"event\":\"lmhead_w_verify\",\"status\":\"row_oob\",\"token_"
             "id\":"
          << token_id << ",\"row_offset_bytes\":" << row_offset
          << ",\"total_bytes\":" << total_bytes << "}";
      append_line(out_path, oss.str());
      continue;
    }
    if (!weights.copy_to_host_offset(row_dim.data(), row_offset,
                                     dim * elem_size, err))
      return false;

    std::vector<__half> col_dim(dim);
    size_t max_d = 0;
    if (vocab > 0 && static_cast<size_t>(token_id) < vocab &&
        total_elems > static_cast<size_t>(token_id)) {
      max_d = (total_elems - static_cast<size_t>(token_id) - 1) / vocab;
    }
    const size_t d_limit = (max_d + 1 < dim) ? (max_d + 1) : dim;
    if (d_limit < dim) {
      std::ostringstream oss;
      oss << "{\"event\":\"lmhead_w_verify\",\"status\":\"col_truncated\","
             "\"token_id\":"
          << token_id << ",\"d_limit\":" << d_limit << ",\"dim\":" << dim
          << "}";
      append_line(out_path, oss.str());
    }
    for (size_t d = 0; d < d_limit; ++d) {
      size_t col_offset = (d * vocab + static_cast<size_t>(token_id));
      if (!copy_elem(col_offset, &col_dim[d])) {
        std::ostringstream oss;
        oss << "{\"event\":\"lmhead_w_verify\",\"status\":\"col_oob\",\"token_"
               "id\":"
            << token_id << ",\"d\":" << d
            << ",\"col_offset_elems\":" << col_offset
            << ",\"total_elems\":" << total_elems << "}";
        append_line(out_path, oss.str());
        return false;
      }
    }

    auto dot_half = [&](const std::vector<__half> &w) -> float {
      float sum = 0.0f;
      for (size_t d = 0; d < dim; ++d) {
        sum += fp16_to_fp32(w[d]) * rms_host[d];
      }
      return sum;
    };

    float row_logit = dot_half(row_dim);
    float col_logit = dot_half(col_dim);
    float gpu_logit = logits_host[static_cast<size_t>(token_id)];
    float abs_err_row = std::fabs(row_logit - gpu_logit);
    float abs_err_col = std::fabs(col_logit - gpu_logit);
    const char *best_layout =
        (abs_err_row <= abs_err_col) ? "row_major_match" : "col_major_match";

    std::vector<float> row_win;
    std::vector<float> col_win;
    row_win.reserve(16);
    col_win.reserve(16);
    for (size_t i = 0; i < 16 && i < dim; ++i) {
      row_win.push_back(fp16_to_fp32(row_dim[i]));
      col_win.push_back(fp16_to_fp32(col_dim[i]));
    }
    uint64_t row_hash = hash_f32(row_win.data(), row_win.size());
    uint64_t col_hash = hash_f32(col_win.data(), col_win.size());

    std::ostringstream oss;
    oss << "{\"event\":\"lmhead_w_verify\",\"token_id\":" << token_id
        << ",\"vocab\":" << vocab << ",\"dim\":" << dim
        << ",\"gpu_logit\":" << gpu_logit << ",\"row_logit\":" << row_logit
        << ",\"col_logit\":" << col_logit << ",\"abs_err_row\":" << abs_err_row
        << ",\"abs_err_col\":" << abs_err_col << ",\"best_layout\":\""
        << best_layout << "\""
        << ",\"row_hash\":" << row_hash << ",\"col_hash\":" << col_hash
        << ",\"row_window\":[";
    for (size_t i = 0; i < row_win.size(); ++i) {
      if (i)
        oss << ",";
      oss << row_win[i];
    }
    oss << "],\"col_window\":[";
    for (size_t i = 0; i < col_win.size(); ++i) {
      if (i)
        oss << ",";
      oss << col_win[i];
    }
    oss << "]}";
    append_line(out_path, oss.str());
  }

  return true;
}

static void append_line(const char *path, const std::string &line) {
  if (!path || !*path)
    return;
  std::ofstream f(path, std::ios::out | std::ios::app);
  if (!f.is_open())
    return;
  f << line << "\n";
}

struct ReadoutTrace {
  const char *phase = nullptr;
  const char *readout_buffer_kind = nullptr;
  const char *hidden_source_tag = nullptr;
  int step = 0;
  size_t tokens_total = 0;
  size_t seq_len = 0;
  size_t pos_id = 0;
  size_t token_index = 0;
  size_t used_index = 0;
  size_t logical_last_index = 0;
  size_t expected_last_index = 0;
  size_t hidden_token_index_used = 0;
  bool readout_mismatch = false;
  size_t hidden_stride_bytes = 0;
  size_t hidden_offset_bytes = 0;
  size_t hidden_alloc_bytes = 0;
  uintptr_t hidden_src_ptr = 0;
  F32Stats hidden_stats{};
  uint64_t hidden_hash = 0;
  uintptr_t rms_in_ptr = 0;
  uintptr_t rms_out_ptr = 0;
  uintptr_t lm_in_ptr = 0;
  size_t rms_offset_bytes = 0;
  F32Stats rms_stats{};
  uint64_t rms_hash = 0;
  size_t logits_offset_bytes = 0;
  uintptr_t logits_ptr = 0;
  F32Stats logits_stats{};
  uint64_t logits_hash = 0;
  int top1_id = -1;
  float top1_logit = 0.0f;
  int top2_id = -1;
  float top2_logit = 0.0f;
  float gap = 0.0f;
  size_t vocab = 0;
};

static void log_readout(const char *path, const ReadoutTrace &t) {
  std::ostringstream oss;
  oss << "{\"phase\":\"" << (t.phase ? t.phase : "") << "\""
      << ",\"readout_buffer_kind\":\""
      << (t.readout_buffer_kind ? t.readout_buffer_kind : "") << "\""
      << ",\"hidden_source_tag\":\""
      << (t.hidden_source_tag ? t.hidden_source_tag : "") << "\""
      << ",\"step\":" << t.step << ",\"tokens_total\":" << t.tokens_total
      << ",\"seq_len\":" << t.seq_len << ",\"pos_id\":" << t.pos_id
      << ",\"token_index\":" << t.token_index
      << ",\"used_index\":" << t.used_index
      << ",\"logical_last_index\":" << t.logical_last_index
      << ",\"expected_last_index\":" << t.expected_last_index
      << ",\"hidden_token_index_used\":" << t.hidden_token_index_used
      << ",\"readout_mismatch\":" << (t.readout_mismatch ? "true" : "false")
      << ",\"hidden_src_ptr\":" << t.hidden_src_ptr
      << ",\"hidden_alloc_bytes\":" << t.hidden_alloc_bytes
      << ",\"hidden_stride_bytes\":" << t.hidden_stride_bytes
      << ",\"hidden_offset_bytes\":" << t.hidden_offset_bytes
      << ",\"hidden_hash\":" << t.hidden_hash
      << ",\"hidden_min\":" << t.hidden_stats.min
      << ",\"hidden_max\":" << t.hidden_stats.max
      << ",\"hidden_mean\":" << t.hidden_stats.mean
      << ",\"rms_in_ptr\":" << t.rms_in_ptr
      << ",\"rms_out_ptr\":" << t.rms_out_ptr
      << ",\"lm_in_ptr\":" << t.lm_in_ptr
      << ",\"rms_offset_bytes\":" << t.rms_offset_bytes
      << ",\"rms_hash\":" << t.rms_hash << ",\"rms_min\":" << t.rms_stats.min
      << ",\"rms_max\":" << t.rms_stats.max
      << ",\"rms_mean\":" << t.rms_stats.mean
      << ",\"logits_offset_bytes\":" << t.logits_offset_bytes
      << ",\"logits_ptr\":" << t.logits_ptr
      << ",\"logits_hash\":" << t.logits_hash
      << ",\"logits_min\":" << t.logits_stats.min
      << ",\"logits_max\":" << t.logits_stats.max
      << ",\"logits_mean\":" << t.logits_stats.mean
      << ",\"top1_id\":" << t.top1_id << ",\"top1_logit\":" << t.top1_logit
      << ",\"top2_id\":" << t.top2_id << ",\"top2_logit\":" << t.top2_logit
      << ",\"gap\":" << t.gap << ",\"vocab\":" << t.vocab << "}";
  append_line(path, oss.str());
}

struct DeltaTrace {
  const char *phase = nullptr;
  int step = 0;
  size_t tokens_total = 0;
  size_t seq_len = 0;
  size_t pos_id = 0;
  uintptr_t hidden_ptr = 0;
  size_t hidden_offset_bytes = 0;
  size_t hidden_token_index_used = 0;
  uint64_t hidden_hash = 0;
  F32Stats hidden_stats{};
  uint64_t rms_hash = 0;
  F32Stats rms_stats{};
  double rms_sumsq = 0.0;
  float rms_eps = 0.0f;
  const char *rms_weight_dtype = nullptr;
  const char *rms_input_dtype = nullptr;
  uint64_t logits_hash = 0;
  int top1_id = -1;
  float top1_logit = 0.0f;
  int top2_id = -1;
  float top2_logit = 0.0f;
  float gap = 0.0f;
  std::string lm_head_route;
  std::string lm_head_force_route;
  std::string lm_head_force_route_decode;
  std::string lm_head_quant_mode;
  std::string lm_head_layout_used;
  std::string lm_head_layout_assumed;
  std::string lm_head_layout_actual;
  uint32_t lm_head_m = 0;
  uint32_t lm_head_n = 0;
  uint32_t lm_head_k = 0;
  uint32_t lm_head_vocab = 0;
  uint32_t lm_head_lda = 0;
  uint32_t lm_head_ldb = 0;
  uint32_t lm_head_ldc = 0;
  uintptr_t lm_head_a_ptr = 0;
  uintptr_t lm_head_b_ptr_base = 0;
  uintptr_t lm_head_b_ptr_effective = 0;
  uintptr_t lm_head_c_ptr = 0;
  int lm_head_dtype_a = 0;
  int lm_head_dtype_b = 0;
  int lm_head_accum_dtype = 0;
  bool lm_head_perhead_enabled = false;
  uintptr_t lm_head_scales_ptr = 0;
  uint64_t lm_head_scales_hash = 0;
  uintptr_t lm_head_head_scales_ptr = 0;
  uint64_t lm_head_head_scales_hash = 0;
  int cpu_probe_top1_id = -1;
  bool cpu_probe_agrees_gpu = false;
  int cpu_probe_prefill_top1 = -1;
  int cpu_probe_decode0_top1 = -1;
};

static void log_delta(const char *path, const DeltaTrace &t) {
  if (!path || !*path)
    return;
  std::ostringstream oss;
  oss << "{\"phase\":\"" << (t.phase ? t.phase : "") << "\""
      << ",\"step\":" << t.step << ",\"tokens_total\":" << t.tokens_total
      << ",\"seq_len\":" << t.seq_len << ",\"pos_id\":" << t.pos_id
      << ",\"hidden_ptr\":" << t.hidden_ptr
      << ",\"hidden_offset_bytes\":" << t.hidden_offset_bytes
      << ",\"hidden_token_index_used\":" << t.hidden_token_index_used
      << ",\"hidden_hash\":" << t.hidden_hash
      << ",\"hidden_min\":" << t.hidden_stats.min
      << ",\"hidden_max\":" << t.hidden_stats.max
      << ",\"hidden_mean\":" << t.hidden_stats.mean
      << ",\"rms_hash\":" << t.rms_hash << ",\"rms_min\":" << t.rms_stats.min
      << ",\"rms_max\":" << t.rms_stats.max
      << ",\"rms_mean\":" << t.rms_stats.mean
      << ",\"rms_sumsq\":" << t.rms_sumsq << ",\"rms_eps\":" << t.rms_eps
      << ",\"rms_weight_dtype\":\""
      << (t.rms_weight_dtype ? t.rms_weight_dtype : "") << "\""
      << ",\"rms_input_dtype\":\""
      << (t.rms_input_dtype ? t.rms_input_dtype : "") << "\""
      << ",\"logits_hash\":" << t.logits_hash << ",\"top1_id\":" << t.top1_id
      << ",\"top1_logit\":" << t.top1_logit << ",\"top2_id\":" << t.top2_id
      << ",\"top2_logit\":" << t.top2_logit << ",\"gap\":" << t.gap
      << ",\"lm_head_route\":\"" << t.lm_head_route << "\""
      << ",\"lm_head_force_route\":\"" << t.lm_head_force_route << "\""
      << ",\"lm_head_force_route_decode\":\"" << t.lm_head_force_route_decode
      << "\""
      << ",\"lm_head_quant_mode\":\"" << t.lm_head_quant_mode << "\""
      << ",\"lm_head_layout_used\":\"" << t.lm_head_layout_used << "\""
      << ",\"lm_head_layout_assumed\":\"" << t.lm_head_layout_assumed << "\""
      << ",\"lm_head_layout_actual\":\"" << t.lm_head_layout_actual << "\""
      << ",\"lm_head_m\":" << t.lm_head_m << ",\"lm_head_n\":" << t.lm_head_n
      << ",\"lm_head_k\":" << t.lm_head_k
      << ",\"lm_head_vocab\":" << t.lm_head_vocab
      << ",\"lm_head_lda\":" << t.lm_head_lda
      << ",\"lm_head_ldb\":" << t.lm_head_ldb
      << ",\"lm_head_ldc\":" << t.lm_head_ldc
      << ",\"lm_head_a_ptr\":" << t.lm_head_a_ptr
      << ",\"lm_head_b_ptr_base\":" << t.lm_head_b_ptr_base
      << ",\"lm_head_b_ptr_effective\":" << t.lm_head_b_ptr_effective
      << ",\"lm_head_c_ptr\":" << t.lm_head_c_ptr
      << ",\"lm_head_dtype_a\":" << t.lm_head_dtype_a
      << ",\"lm_head_dtype_b\":" << t.lm_head_dtype_b
      << ",\"lm_head_accum_dtype\":" << t.lm_head_accum_dtype
      << ",\"lm_head_perhead_enabled\":"
      << (t.lm_head_perhead_enabled ? "true" : "false")
      << ",\"lm_head_scales_ptr\":" << t.lm_head_scales_ptr
      << ",\"lm_head_scales_hash\":" << t.lm_head_scales_hash
      << ",\"lm_head_head_scales_ptr\":" << t.lm_head_head_scales_ptr
      << ",\"lm_head_head_scales_hash\":" << t.lm_head_head_scales_hash
      << ",\"cpu_probe_top1_id\":" << t.cpu_probe_top1_id
      << ",\"cpu_probe_agrees_gpu\":"
      << (t.cpu_probe_agrees_gpu ? "true" : "false")
      << ",\"cpu_probe_prefill_top1\":" << t.cpu_probe_prefill_top1
      << ",\"cpu_probe_decode0_top1\":" << t.cpu_probe_decode0_top1 << "}";
  append_line(path, oss.str());
}

struct HiddenEquivTrace {
  size_t prefill_tokens_total = 0;
  size_t prefill_seq_len = 0;
  size_t prefill_pos_id = 0;
  uint64_t prefill_hash = 0;
  F32Stats prefill_stats{};
  size_t decode_tokens_total = 0;
  size_t decode_seq_len = 0;
  size_t decode_pos_id = 0;
  uint64_t decode_hash = 0;
  F32Stats decode_stats{};
};

static void log_hidden_equiv(const char *path, const HiddenEquivTrace &t) {
  if (!path || !*path)
    return;
  std::ostringstream oss;
  oss << "{\"event\":\"hidden_equiv\""
      << ",\"prefill_tokens_total\":" << t.prefill_tokens_total
      << ",\"prefill_seq_len\":" << t.prefill_seq_len
      << ",\"prefill_pos_id\":" << t.prefill_pos_id
      << ",\"prefill_hash\":" << t.prefill_hash
      << ",\"prefill_min\":" << t.prefill_stats.min
      << ",\"prefill_max\":" << t.prefill_stats.max
      << ",\"prefill_mean\":" << t.prefill_stats.mean
      << ",\"decode_tokens_total\":" << t.decode_tokens_total
      << ",\"decode_seq_len\":" << t.decode_seq_len
      << ",\"decode_pos_id\":" << t.decode_pos_id
      << ",\"decode_hash\":" << t.decode_hash
      << ",\"decode_min\":" << t.decode_stats.min
      << ",\"decode_max\":" << t.decode_stats.max
      << ",\"decode_mean\":" << t.decode_stats.mean << "}";
  append_line(path, oss.str());
}
static void log_landscape(const char *path, int step,
                          const std::vector<float> &logits, int topk) {
  if (!path || !*path)
    return;
  std::vector<std::pair<float, int>> v;
  v.reserve(logits.size());
  for (size_t i = 0; i < logits.size(); ++i)
    v.push_back({logits[i], (int)i});
  std::sort(v.rbegin(), v.rend());
  const int k = std::min<int>(topk, (int)v.size());
  const float top1 = v[0].first;
  const float top2 = v[1].first;
  const float gap = top1 - top2;

  float max_logit = v[0].first;
  double sum = 0.0;
  for (int i = 0; i < k; ++i)
    sum += std::exp(v[i].first - max_logit);
  double entropy = 0.0;
  for (int i = 0; i < k; ++i) {
    double p = std::exp(v[i].first - max_logit) / sum;
    entropy += -p * std::log(p + 1e-12);
  }

  std::ostringstream oss;
  oss << "{\"step\":" << step << ",\"top1\":{\"id\":" << v[0].second
      << ",\"logit\":" << v[0].first << "},\"top2\":{\"id\":" << v[1].second
      << ",\"logit\":" << v[1].first << "},\"gap\":" << gap
      << ",\"entropy_topk\":" << entropy << ",\"top5\":[";
  for (int i = 0; i < 5 && i < (int)v.size(); ++i) {
    if (i)
      oss << ",";
    oss << "{\"id\":" << v[i].second << ",\"logit\":" << v[i].first << "}";
  }
  oss << "]}";
  append_line(path, oss.str());
}

static bool env_flag(const char *k) {
  const char *v = std::getenv(k);
  return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

static bool trace_post_wo_enabled() { return env_flag("GRETA_TRACE_POST_WO"); }

static const char *post_wo_out_path() {
  const char *out = std::getenv("GRETA_TRACE_POST_WO_OUT");
  if (out && *out)
    return out;
  return nullptr;
}

static bool validate_trace_shapes(const ModelConfig &config, std::string *err) {
  auto fail = [&](const std::string &msg) {
    if (err)
      *err = msg;
    std::cerr << "[GRETA_TRACE_SHAPE] " << msg << std::endl;
    return false;
  };

  if (config.vocab_size == 0)
    return fail("vocab_size=0");
  if (config.dim == 0)
    return fail("dim=0");
  if (config.num_layers == 0)
    return fail("num_layers=0");
  if (config.num_heads == 0)
    return fail("num_heads=0");
  if (config.num_heads_kv == 0)
    return fail("num_heads_kv=0");
  if (config.head_dim == 0)
    return fail("head_dim=0");

  if (config.dim % config.num_heads != 0) {
    return fail(
        "dim not divisible by num_heads: dim=" + std::to_string(config.dim) +
        " num_heads=" + std::to_string(config.num_heads));
  }
  const uint32_t expected_head_dim = config.dim / config.num_heads;
  if (config.head_dim != expected_head_dim) {
    return fail(
        "head_dim mismatch: head_dim=" + std::to_string(config.head_dim) +
        " expected=" + std::to_string(expected_head_dim));
  }
  if (config.num_heads_kv > config.num_heads) {
    return fail("num_heads_kv greater than num_heads: num_heads_kv=" +
                std::to_string(config.num_heads_kv) +
                " num_heads=" + std::to_string(config.num_heads));
  }
  if (config.num_heads % config.num_heads_kv != 0) {
    return fail("num_heads not divisible by num_heads_kv: num_heads=" +
                std::to_string(config.num_heads) +
                " num_heads_kv=" + std::to_string(config.num_heads_kv));
  }

  return true;
}

static void log_d2h_trace(bool enabled, const char *tensor_name, int step,
                          int layer, const gcore::rt::hip::Buffer &buffer,
                          size_t offset_bytes, size_t size_bytes) {
  if (!enabled)
    return;
  std::ostringstream oss;
  oss << "[GRETA_TRACE_D2H]"
      << " tensor=" << (tensor_name ? tensor_name : "unknown")
      << " step=" << step << " layer=" << layer << " src_ptr=0x" << std::hex
      << reinterpret_cast<uintptr_t>(buffer.data()) << std::dec
      << " alloc_bytes=" << buffer.size() << " offset_bytes=" << offset_bytes
      << " size_bytes=" << size_bytes;
  std::cerr << oss.str() << std::endl;
}

Generator::Generator() = default;

Generator::~Generator() = default;

bool Generator::init(const ModelConfig &config, BlockScheduler *scheduler,
                     std::string *err) {
  config_ = config;
  scheduler_ = scheduler;

  // Initialize tokenizer
  tokenizer_ = std::make_unique<Tokenizer>();
  tokenizer_->set_vocabulary(config_.vocabulary);

  initialized_ = true;
  return true;
}

int32_t Generator::sample(const float *logits, size_t vocab_size,
                          const SamplingParams &params) {
  // Diagnostic: Check if logits are sane
  float min_l = logits[0], max_l = logits[0], sum_l = 0.0f;
  int nan_count = 0;
  for (size_t i = 0; i < vocab_size; ++i) {
    float v = logits[i];
    if (std::isnan(v))
      nan_count++;
    else {
      if (v < min_l)
        min_l = v;
      if (v > max_l)
        max_l = v;
      sum_l += v;
    }
  }

  static int sample_count = 0;
  if (sample_count++ < 3) { // Print only for first 3 tokens
    std::cout << "[SAMPLE DEBUG] Logits stats: min=" << min_l
              << " max=" << max_l << " avg=" << (sum_l / vocab_size)
              << " NaNs=" << nan_count << std::endl;
    // Print top 5 logits
    std::vector<std::pair<float, int>> top;
    for (int i = 0; i < (int)vocab_size; ++i)
      top.push_back({logits[i], i});
    std::sort(top.rbegin(), top.rend());
    std::cout << "  Top tokens: ";
    for (int i = 0; i < 5; ++i)
      std::cout << top[i].second << "(" << top[i].first << ") ";
    std::cout << std::endl;
  }

  if (params.greedy) {
    int32_t max_id = 0;
    float max_val = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
      if (logits[i] > max_val) {
        max_val = logits[i];
        max_id = static_cast<int32_t>(i);
      }
    }
    return max_id;
  }

  // Softmax with temperature
  std::vector<float> probs(vocab_size);
  float sum = 0.0f;
  float max_logit = -INFINITY;
  for (size_t i = 0; i < vocab_size; ++i) {
    if (logits[i] > max_logit)
      max_logit = logits[i];
  }

  for (size_t i = 0; i < vocab_size; ++i) {
    float p = std::exp((logits[i] - max_logit) / params.temperature);
    probs[i] = p;
    sum += p;
  }

  // Random sample
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, sum);
  float r = dis(gen);

  float cumulative = 0.0f;
  for (size_t i = 0; i < vocab_size; ++i) {
    cumulative += probs[i];
    if (r <= cumulative)
      return static_cast<int32_t>(i);
  }

  return 0;
}

std::vector<int32_t>
Generator::generate_tokens(const std::vector<int32_t> &prompt_tokens,
                           const SamplingParams &params, GenerationStats *stats,
                           std::string *err, AlignmentCallback align_callback) {
  if (!initialized_) {
    if (err)
      *err = "Generator not initialized";
    return {};
  }

  std::vector<int32_t> output = prompt_tokens;
  auto start = std::chrono::high_resolution_clock::now();
  auto first_token_time = start;
  bool first_token = true;

  std::vector<float> logits_host(config_.vocab_size);

  const bool trace_readout = env_flag("GRETA_TRACE_READOUT");
  const char *trace_readout_out = std::getenv("GRETA_TRACE_READOUT_OUT");
  const bool trace_prefill_decode = env_flag("GRETA_TRACE_PREFILL_DECODE");
  const char *trace_prefill_decode_out =
      std::getenv("GRETA_TRACE_PREFILL_DECODE_OUT");
  const bool trace_delta = env_flag("GRETA_TRACE_PREFILL_DECODE_DELTA");
  const char *trace_delta_out = std::getenv("GRETA_TRACE_PREFILL_DECODE_OUT");
  const bool trace_hidden_equiv = env_flag("GRETA_TRACE_HIDDEN_EQUIV");
  const bool trace_rms_verify = env_flag("GRETA_TRACE_RMS_VERIFY");
  const bool trace_cpu_probe = env_flag("GRETA_TRACE_LMHEAD_CPU_PROBE");
  const bool trace_lmhead_w_verify = env_flag("GRETA_TRACE_LMHEAD_W_VERIFY");
  const char *trace_lmhead_w_out = std::getenv("GRETA_TRACE_LMHEAD_W_OUT");
  const bool trace_landscape = env_flag("GRETA_TRACE_LANDSCAPE");
  const char *trace_landscape_out = std::getenv("GRETA_TRACE_LANDSCAPE_OUT");
  const int landscape_topk = 64;
  const bool trace_stage = stage_trace_enabled();
  const bool trace_stage_debug_input = stage_trace_debug_input();
  const bool trace_post_wo = trace_post_wo_enabled();
  const bool trace_any = trace_readout || trace_prefill_decode ||
                         trace_landscape || trace_delta ||
                         trace_lmhead_w_verify || trace_hidden_equiv ||
                         trace_stage || trace_post_wo;

  std::vector<float> hidden_host;
  std::vector<float> rms_host;
  if (trace_readout || trace_prefill_decode || trace_delta ||
      trace_lmhead_w_verify || trace_hidden_equiv || trace_stage ||
      trace_post_wo) {
    hidden_host.resize(config_.dim);
    rms_host.resize(config_.dim);
  }

  DeltaTrace prefill_delta;
  bool prefill_delta_ready = false;
  bool prefill_delta_written = false;
  std::vector<float> prefill_rms_copy;
  std::vector<int> prefill_topk_ids;
  int cpu_prefill_top1 = -1;
  int cpu_decode_top1 = -1;

  if (trace_any) {
    if (!validate_trace_shapes(config_, err)) {
      return output;
    }
  }

  // 1. Prefill: Process all prompt tokens at once
  if (!scheduler_->forward(prompt_tokens.data(), 0, prompt_tokens.size(),
                           err)) {
    return output;
  }

  // Sample first generated token from the last set of logits in the prefill
  size_t last_token_offset = 0;
  if (!prompt_tokens.empty()) {
    last_token_offset =
        (prompt_tokens.size() - 1) * config_.vocab_size * sizeof(float);
  }
  const auto &logits_buf = scheduler_->get_logits();
  log_d2h_trace(trace_any, "logits", 0, -1, logits_buf, last_token_offset,
                config_.vocab_size * sizeof(float));
  if (!scheduler_->get_logits().copy_to_host_offset(
          logits_host.data(), last_token_offset,
          config_.vocab_size * sizeof(float), err)) {
    return output;
  }

  if (trace_readout || trace_prefill_decode || trace_delta ||
      trace_lmhead_w_verify || trace_stage || trace_post_wo) {
    const size_t tokens_total = prompt_tokens.size();
    const size_t token_index = tokens_total > 0 ? (tokens_total - 1) : 0;
    const size_t logical_last_index = tokens_total > 0 ? (tokens_total - 1) : 0;
    const size_t hidden_token_index_used = token_index;
    const size_t used_index = hidden_token_index_used;
    const size_t seq_len = tokens_total;
    const size_t pos_id = token_index;
    const size_t hidden_stride_bytes = config_.dim * sizeof(float);
    const size_t hidden_offset = hidden_token_index_used * hidden_stride_bytes;
    const auto &hidden_buf = scheduler_->get_hidden_state();
    const auto &rms_buf = scheduler_->get_norm_out();
    log_d2h_trace(trace_any, "hidden", 0, -1, hidden_buf, hidden_offset,
                  config_.dim * sizeof(float));
    if (!scheduler_->get_hidden_state().copy_to_host_offset(
            hidden_host.data(), hidden_offset, config_.dim * sizeof(float),
            err)) {
      return output;
    }
    const size_t rms_offset = hidden_token_index_used * hidden_stride_bytes;
    if (!rms_buf.copy_to_host_offset(rms_host.data(), rms_offset,
                                     config_.dim * sizeof(float), err)) {
      return output;
    }
    const F32Stats hstats = stats_f32(hidden_host.data(), config_.dim);
    const uint64_t hhash = hash_f32(hidden_host.data(), config_.dim);
    const F32Stats rstats = stats_f32(rms_host.data(), config_.dim);
    const uint64_t rhash = hash_f32(rms_host.data(), config_.dim);
    const F32Stats lstats = stats_f32(logits_host.data(), config_.vocab_size);
    const uint64_t lhash = hash_f32(logits_host.data(), config_.vocab_size);
    const Top2Logits top2 = top2_logits(logits_host.data(), config_.vocab_size);
    const float gap = top2.top1_logit - top2.top2_logit;
    const uintptr_t hidden_ptr = reinterpret_cast<uintptr_t>(hidden_buf.data());
    const uintptr_t rms_ptr = reinterpret_cast<uintptr_t>(rms_buf.data());
    const uintptr_t logits_ptr = reinterpret_cast<uintptr_t>(logits_buf.data());
    const char *stage_prompt_id = std::getenv("GRETA_TRACE_PROMPT_ID");

    if (trace_stage && stage_trace_phase_enabled("prefill_last")) {
      StageLogitsStats stage_stats{};
      stage_stats.hash = lhash;
      stage_stats.min = lstats.min;
      stage_stats.max = lstats.max;
      stage_stats.mean = lstats.mean;
      stage_stats.top1_id = top2.top1_id;
      stage_stats.top1_logit = top2.top1_logit;
      stage_stats.top2_id = top2.top2_id;
      stage_stats.top2_logit = top2.top2_logit;
      stage_stats.gap = gap;
      stage_stats.logits_ptr = logits_ptr;
      stage_stats.logits_offset_bytes = last_token_offset;
      stage_stats.vocab = config_.vocab_size;
      stage_trace_logits("prefill_last", stage_prompt_id, 0,
                         static_cast<uint32_t>(pos_id),
                         static_cast<uint32_t>(seq_len),
                         static_cast<uint32_t>(tokens_total), stage_stats);
    }
    if (trace_post_wo_enabled()) {
      const char *out = post_wo_out_path();
      if (out && *out) {
        std::ostringstream oss;
        oss << "{\"event\":\"post_wo_logits\"";
        if (stage_prompt_id && *stage_prompt_id)
          oss << ",\"prompt_id\":\"" << stage_prompt_id << "\"";
        oss << ",\"phase\":\"prefill_last\""
            << ",\"pos_id\":" << pos_id << ",\"seq_len\":" << seq_len
            << ",\"tokens_total\":" << tokens_total
            << ",\"token_index\":" << token_index
            << ",\"logits_hash\":" << lhash << ",\"logits_min\":" << lstats.min
            << ",\"logits_max\":" << lstats.max
            << ",\"logits_mean\":" << lstats.mean
            << ",\"top1_id\":" << top2.top1_id
            << ",\"top1_logit\":" << top2.top1_logit
            << ",\"top2_id\":" << top2.top2_id
            << ",\"top2_logit\":" << top2.top2_logit << ",\"gap\":" << gap
            << ",\"logits_ptr\":" << logits_ptr
            << ",\"logits_offset_bytes\":" << last_token_offset
            << ",\"vocab\":" << config_.vocab_size << "}";
        append_line(out, oss.str());
      }
    }

    const bool readout_is_single_token = false;
    const bool readout_mismatch =
        readout_is_single_token ? (logical_last_index !=
                                   (tokens_total > 0 ? (tokens_total - 1) : 0))
                                : (used_index != logical_last_index);

    ReadoutTrace trace{};
    trace.phase = "prefill_last";
    trace.readout_buffer_kind = "seq";
    trace.hidden_source_tag = "prefill_hidden_seq";
    trace.step = 0;
    trace.tokens_total = tokens_total;
    trace.seq_len = seq_len;
    trace.pos_id = pos_id;
    trace.token_index = token_index;
    trace.used_index = used_index;
    trace.logical_last_index = logical_last_index;
    trace.expected_last_index = logical_last_index;
    trace.hidden_token_index_used = hidden_token_index_used;
    trace.readout_mismatch = readout_mismatch;
    trace.hidden_stride_bytes = hidden_stride_bytes;
    trace.hidden_offset_bytes = hidden_offset;
    trace.hidden_alloc_bytes = hidden_buf.size();
    trace.hidden_src_ptr = hidden_ptr;
    trace.hidden_stats = hstats;
    trace.hidden_hash = hhash;
    trace.rms_in_ptr = hidden_ptr;
    trace.rms_out_ptr = rms_ptr;
    trace.lm_in_ptr = rms_ptr;
    trace.rms_offset_bytes = rms_offset;
    trace.rms_stats = rstats;
    trace.rms_hash = rhash;
    trace.logits_offset_bytes = last_token_offset;
    trace.logits_ptr = logits_ptr;
    trace.logits_stats = lstats;
    trace.logits_hash = lhash;
    trace.top1_id = top2.top1_id;
    trace.top1_logit = top2.top1_logit;
    trace.top2_id = top2.top2_id;
    trace.top2_logit = top2.top2_logit;
    trace.gap = gap;
    trace.vocab = config_.vocab_size;

    if (trace_readout && trace_readout_out) {
      log_readout(trace_readout_out, trace);
    }
    if (trace_prefill_decode && trace_prefill_decode_out) {
      log_readout(trace_prefill_decode_out, trace);
    }
    if ((trace_delta || trace_hidden_equiv) && trace_delta_out) {
      gcore::compute::GemmAuditInfo audit =
          gcore::compute::GretaCompute::get_last_gemm_audit();
      DeltaTrace delta{};
      delta.phase = "prefill_last";
      delta.step = 0;
      delta.tokens_total = tokens_total;
      delta.seq_len = seq_len;
      delta.pos_id = pos_id;
      delta.hidden_ptr = hidden_ptr;
      delta.hidden_offset_bytes = hidden_offset;
      delta.hidden_token_index_used = hidden_token_index_used;
      delta.hidden_hash = hhash;
      delta.hidden_stats = hstats;
      delta.rms_hash = rhash;
      delta.rms_stats = rstats;
      delta.rms_sumsq =
          trace_rms_verify ? sumsq_f32(rms_host.data(), config_.dim) : 0.0;
      delta.rms_eps = trace_rms_verify ? config_.rms_eps : 0.0f;
      delta.rms_weight_dtype = trace_rms_verify ? "unknown" : "";
      delta.rms_input_dtype =
          trace_rms_verify
              ? dtype_name(scheduler_->get_hidden_state().data_type())
              : "";
      delta.logits_hash = lhash;
      delta.top1_id = top2.top1_id;
      delta.top1_logit = top2.top1_logit;
      delta.top2_id = top2.top2_id;
      delta.top2_logit = top2.top2_logit;
      delta.gap = gap;
      delta.lm_head_route = audit.route;
      delta.lm_head_force_route = audit.force_route;
      delta.lm_head_force_route_decode = audit.force_route_decode;
      delta.lm_head_quant_mode = audit.quant_mode;
      delta.lm_head_layout_used = audit.layout_used;
      delta.lm_head_layout_assumed = audit.layout_assumed;
      delta.lm_head_layout_actual = audit.layout_actual;
      delta.lm_head_m = audit.m;
      delta.lm_head_n = audit.n;
      delta.lm_head_k = audit.k;
      delta.lm_head_vocab = audit.n;
      delta.lm_head_lda = audit.lda;
      delta.lm_head_ldb = audit.ldb;
      delta.lm_head_ldc = audit.ldc;
      delta.lm_head_a_ptr = audit.a_ptr;
      delta.lm_head_b_ptr_base = audit.b_ptr_base;
      delta.lm_head_b_ptr_effective = audit.b_ptr_effective;
      delta.lm_head_c_ptr = audit.c_ptr;
      delta.lm_head_dtype_a = audit.type_a;
      delta.lm_head_dtype_b = audit.type_b;
      delta.lm_head_accum_dtype = audit.accum_type;
      delta.lm_head_perhead_enabled = audit.perhead_enabled;
      delta.lm_head_scales_ptr = audit.scales_ptr;
      delta.lm_head_scales_hash = audit.scales_hash;
      delta.lm_head_head_scales_ptr = audit.head_scales_ptr;
      delta.lm_head_head_scales_hash = audit.head_scales_hash;

      if (trace_cpu_probe) {
        prefill_rms_copy = rms_host;
        prefill_topk_ids = topk_ids(logits_host.data(), config_.vocab_size, 10);
      }

      prefill_delta = delta;
      prefill_delta_ready = true;

      if (!trace_cpu_probe) {
        log_delta(trace_delta_out, prefill_delta);
        prefill_delta_written = true;
      }
    }
    if (trace_lmhead_w_verify && trace_lmhead_w_out) {
      const auto &lm_head_w = scheduler_->get_output_weight();
      if (!trace_lmhead_w_verify_once(lm_head_w, rms_host, logits_host, config_,
                                      trace_lmhead_w_out, err)) {
        return output;
      }
    }
  }

  if (trace_landscape && trace_landscape_out) {
    log_landscape(trace_landscape_out, 0, logits_host, landscape_topk);
  }

  int32_t next_token = sample(logits_host.data(), config_.vocab_size, params);
  output.push_back(next_token);

  if (env_flag("GRETA_TRACE_LAYER")) {
    const size_t pos_id =
        prompt_tokens.size() > 0 ? (prompt_tokens.size() - 1) : 0;
    const size_t seq_len = prompt_tokens.size();
    const size_t tokens_total = output.size();
    const int32_t token_in = prompt_tokens.empty() ? -1 : prompt_tokens.back();
    layer_trace_emit_step_header(0, pos_id, seq_len, tokens_total, token_in,
                                 next_token, config_);
  }

  if (align_callback) {
    AlignmentStep step;
    step.step = 0;
    step.token_id = next_token;
    step.logit = logits_host[next_token];
    step.logit_min = logits_host[0];
    step.logit_max = logits_host[0];
    double sum = 0;
    step.nan_count = 0;
    step.inf_count = 0;
    std::vector<std::pair<float, int>> top;
    for (size_t i = 0; i < config_.vocab_size; ++i) {
      float v = logits_host[i];
      if (std::isnan(v))
        step.nan_count++;
      else if (std::isinf(v))
        step.inf_count++;
      else {
        if (v < step.logit_min)
          step.logit_min = v;
        if (v > step.logit_max)
          step.logit_max = v;
        sum += v;
      }
      top.push_back({v, (int)i});
    }
    step.logit_mean = (float)(sum / config_.vocab_size);
    std::sort(top.rbegin(), top.rend());
    for (int i = 0; i < 10 && i < (int)config_.vocab_size; ++i) {
      step.topk_ids.push_back(top[i].second);
      step.topk_logits.push_back(top[i].first);
    }
    align_callback(step);
  }

  first_token_time = std::chrono::high_resolution_clock::now();
  first_token = false;

  // 2. Decode loop: Generate remaining tokens one-by-one
  for (int i = 1; i < params.max_tokens; ++i) {
    if (next_token == tokenizer_->eos_id())
      break;

    // Use current sequence length (output.size() - 1) as start position for the
    // new token
    int32_t last_token_id = output.back();
    size_t decode_seq_start = output.size() - 1;
    if (trace_stage_debug_input && i == 1 && !prompt_tokens.empty()) {
      last_token_id = prompt_tokens.back();
      decode_seq_start = prompt_tokens.size() - 1;
      setenv("GRETA_TRACE_DEBUG_INPUT_TOKEN_ID",
             std::to_string(last_token_id).c_str(), 1);
    }
    if (!scheduler_->forward(&last_token_id, decode_seq_start, 1, err)) {
      break;
    }
    const bool need_logits_host =
        !params.greedy || align_callback || trace_readout || trace_landscape ||
        trace_prefill_decode || trace_delta || trace_stage || trace_post_wo;
    const size_t decode_logits_offset =
        decode_seq_start * config_.vocab_size * sizeof(float);
    if (params.greedy && !align_callback && !need_logits_host) {
      next_token = scheduler_->sample_greedy_gpu(decode_logits_offset, err);
    } else {
      const auto &logits_buf = scheduler_->get_logits();
      log_d2h_trace(trace_any, "logits", i, -1, logits_buf,
                    decode_logits_offset, config_.vocab_size * sizeof(float));
      if (!scheduler_->get_logits().copy_to_host_offset(
              logits_host.data(), decode_logits_offset,
              config_.vocab_size * sizeof(float), err)) {
        break;
      }

      if (trace_readout || trace_prefill_decode || trace_delta ||
          trace_lmhead_w_verify || trace_stage || trace_post_wo) {
        const size_t tokens_total = output.size();
        const size_t token_index = tokens_total > 0 ? (tokens_total - 1) : 0;
        const size_t logical_last_index =
            tokens_total > 0 ? (tokens_total - 1) : 0;
        const size_t hidden_token_index_used = 0;
        const size_t used_index = hidden_token_index_used;
        const size_t seq_len = 1;
        const size_t pos_id = decode_seq_start;
        const size_t hidden_stride_bytes = config_.dim * sizeof(float);
        const size_t hidden_offset =
            hidden_token_index_used * hidden_stride_bytes;
        const auto &hidden_buf = scheduler_->get_hidden_state();
        const auto &rms_buf = scheduler_->get_norm_out();
        log_d2h_trace(trace_any, "hidden", i, -1, hidden_buf, hidden_offset,
                      config_.dim * sizeof(float));
        if (!scheduler_->get_hidden_state().copy_to_host_offset(
                hidden_host.data(), hidden_offset, config_.dim * sizeof(float),
                err)) {
          break;
        }
        const size_t rms_offset = hidden_token_index_used * hidden_stride_bytes;
        if (!rms_buf.copy_to_host_offset(rms_host.data(), rms_offset,
                                         config_.dim * sizeof(float), err)) {
          break;
        }
        const F32Stats hstats = stats_f32(hidden_host.data(), config_.dim);
        const uint64_t hhash = hash_f32(hidden_host.data(), config_.dim);
        const F32Stats rstats = stats_f32(rms_host.data(), config_.dim);
        const uint64_t rhash = hash_f32(rms_host.data(), config_.dim);
        const F32Stats lstats =
            stats_f32(logits_host.data(), config_.vocab_size);
        const uint64_t lhash = hash_f32(logits_host.data(), config_.vocab_size);
        const Top2Logits top2 =
            top2_logits(logits_host.data(), config_.vocab_size);
        const float gap = top2.top1_logit - top2.top2_logit;
        const uintptr_t hidden_ptr =
            reinterpret_cast<uintptr_t>(hidden_buf.data());
        const uintptr_t rms_ptr = reinterpret_cast<uintptr_t>(rms_buf.data());
        const uintptr_t logits_ptr =
            reinterpret_cast<uintptr_t>(logits_buf.data());
        const char *stage_prompt_id = std::getenv("GRETA_TRACE_PROMPT_ID");

        if (trace_stage && i == 1 && stage_trace_phase_enabled("decode0")) {
          StageLogitsStats stage_stats{};
          stage_stats.hash = lhash;
          stage_stats.min = lstats.min;
          stage_stats.max = lstats.max;
          stage_stats.mean = lstats.mean;
          stage_stats.top1_id = top2.top1_id;
          stage_stats.top1_logit = top2.top1_logit;
          stage_stats.top2_id = top2.top2_id;
          stage_stats.top2_logit = top2.top2_logit;
          stage_stats.gap = gap;
          stage_stats.logits_ptr = logits_ptr;
          stage_stats.logits_offset_bytes = decode_logits_offset;
          stage_stats.vocab = config_.vocab_size;
          stage_trace_logits(
              "decode0", stage_prompt_id, static_cast<uint32_t>(i),
              static_cast<uint32_t>(pos_id), static_cast<uint32_t>(seq_len),
              static_cast<uint32_t>(tokens_total), stage_stats);
        }
        if (trace_post_wo_enabled() && i == 1) {
          const char *out = post_wo_out_path();
          if (out && *out) {
            std::ostringstream oss;
            oss << "{\"event\":\"post_wo_logits\"";
            if (stage_prompt_id && *stage_prompt_id)
              oss << ",\"prompt_id\":\"" << stage_prompt_id << "\"";
            oss << ",\"phase\":\"decode0\""
                << ",\"pos_id\":" << pos_id << ",\"seq_len\":" << seq_len
                << ",\"tokens_total\":" << tokens_total
                << ",\"token_index\":" << hidden_token_index_used
                << ",\"logits_hash\":" << lhash
                << ",\"logits_min\":" << lstats.min
                << ",\"logits_max\":" << lstats.max
                << ",\"logits_mean\":" << lstats.mean
                << ",\"top1_id\":" << top2.top1_id
                << ",\"top1_logit\":" << top2.top1_logit
                << ",\"top2_id\":" << top2.top2_id
                << ",\"top2_logit\":" << top2.top2_logit << ",\"gap\":" << gap
                << ",\"logits_ptr\":" << logits_ptr
                << ",\"logits_offset_bytes\":" << decode_logits_offset
                << ",\"vocab\":" << config_.vocab_size << "}";
            append_line(out, oss.str());
          }
        }

        const bool readout_is_single_token = true;
        const bool readout_mismatch =
            readout_is_single_token
                ? (logical_last_index !=
                   (tokens_total > 0 ? (tokens_total - 1) : 0))
                : (used_index != logical_last_index);

        ReadoutTrace trace{};
        trace.phase = "decode";
        trace.readout_buffer_kind = "single_token";
        trace.hidden_source_tag = "decode_hidden_single_token";
        trace.step = i;
        trace.tokens_total = tokens_total;
        trace.seq_len = seq_len;
        trace.pos_id = pos_id;
        trace.token_index = token_index;
        trace.used_index = used_index;
        trace.logical_last_index = logical_last_index;
        trace.expected_last_index = logical_last_index;
        trace.hidden_token_index_used = hidden_token_index_used;
        trace.readout_mismatch = readout_mismatch;
        trace.hidden_stride_bytes = hidden_stride_bytes;
        trace.hidden_offset_bytes = hidden_offset;
        trace.hidden_alloc_bytes = hidden_buf.size();
        trace.hidden_src_ptr = hidden_ptr;
        trace.hidden_stats = hstats;
        trace.hidden_hash = hhash;
        trace.rms_in_ptr = hidden_ptr;
        trace.rms_out_ptr = rms_ptr;
        trace.lm_in_ptr = rms_ptr;
        trace.rms_offset_bytes = rms_offset;
        trace.rms_stats = rstats;
        trace.rms_hash = rhash;
        trace.logits_offset_bytes = decode_logits_offset;
        trace.logits_ptr = logits_ptr;
        trace.logits_stats = lstats;
        trace.logits_hash = lhash;
        trace.top1_id = top2.top1_id;
        trace.top1_logit = top2.top1_logit;
        trace.top2_id = top2.top2_id;
        trace.top2_logit = top2.top2_logit;
        trace.gap = gap;
        trace.vocab = config_.vocab_size;

        if (trace_readout && trace_readout_out) {
          log_readout(trace_readout_out, trace);
        }
        if (trace_prefill_decode && trace_prefill_decode_out) {
          log_readout(trace_prefill_decode_out, trace);
        }
        if (trace_delta && trace_delta_out && i == 1) {
          gcore::compute::GemmAuditInfo audit =
              gcore::compute::GretaCompute::get_last_gemm_audit();
          DeltaTrace delta{};
          delta.phase = "decode0";
          delta.step = i;
          delta.tokens_total = tokens_total;
          delta.seq_len = seq_len;
          delta.pos_id = pos_id;
          delta.hidden_ptr = hidden_ptr;
          delta.hidden_offset_bytes = hidden_offset;
          delta.hidden_token_index_used = hidden_token_index_used;
          delta.hidden_hash = hhash;
          delta.hidden_stats = hstats;
          delta.rms_hash = rhash;
          delta.rms_stats = rstats;
          delta.rms_sumsq =
              trace_rms_verify ? sumsq_f32(rms_host.data(), config_.dim) : 0.0;
          delta.rms_eps = trace_rms_verify ? config_.rms_eps : 0.0f;
          delta.rms_weight_dtype = trace_rms_verify ? "unknown" : "";
          delta.rms_input_dtype =
              trace_rms_verify
                  ? dtype_name(scheduler_->get_hidden_state().data_type())
                  : "";
          delta.logits_hash = lhash;
          delta.top1_id = top2.top1_id;
          delta.top1_logit = top2.top1_logit;
          delta.top2_id = top2.top2_id;
          delta.top2_logit = top2.top2_logit;
          delta.gap = gap;
          delta.lm_head_route = audit.route;
          delta.lm_head_force_route = audit.force_route;
          delta.lm_head_force_route_decode = audit.force_route_decode;
          delta.lm_head_quant_mode = audit.quant_mode;
          delta.lm_head_layout_used = audit.layout_used;
          delta.lm_head_layout_assumed = audit.layout_assumed;
          delta.lm_head_layout_actual = audit.layout_actual;
          delta.lm_head_m = audit.m;
          delta.lm_head_n = audit.n;
          delta.lm_head_k = audit.k;
          delta.lm_head_vocab = audit.n;
          delta.lm_head_lda = audit.lda;
          delta.lm_head_ldb = audit.ldb;
          delta.lm_head_ldc = audit.ldc;
          delta.lm_head_a_ptr = audit.a_ptr;
          delta.lm_head_b_ptr_base = audit.b_ptr_base;
          delta.lm_head_b_ptr_effective = audit.b_ptr_effective;
          delta.lm_head_c_ptr = audit.c_ptr;
          delta.lm_head_dtype_a = audit.type_a;
          delta.lm_head_dtype_b = audit.type_b;
          delta.lm_head_accum_dtype = audit.accum_type;
          delta.lm_head_perhead_enabled = audit.perhead_enabled;
          delta.lm_head_scales_ptr = audit.scales_ptr;
          delta.lm_head_scales_hash = audit.scales_hash;
          delta.lm_head_head_scales_ptr = audit.head_scales_ptr;
          delta.lm_head_head_scales_hash = audit.head_scales_hash;

          if (trace_cpu_probe && prefill_delta_ready) {
            std::vector<int> decode_topk_ids =
                topk_ids(logits_host.data(), config_.vocab_size, 10);
            std::vector<int> candidate_ids = prefill_topk_ids;
            for (int id : decode_topk_ids) {
              if (std::find(candidate_ids.begin(), candidate_ids.end(), id) ==
                  candidate_ids.end()) {
                candidate_ids.push_back(id);
              }
            }

            const auto &lm_head_w = scheduler_->get_output_weight();
            std::vector<float> tmp_logits;
            if (cpu_probe_lm_head(lm_head_w, prefill_rms_copy, config_.dim,
                                  candidate_ids, tmp_logits,
                                  cpu_prefill_top1)) {
              prefill_delta.cpu_probe_top1_id = cpu_prefill_top1;
              prefill_delta.cpu_probe_agrees_gpu =
                  (cpu_prefill_top1 == prefill_delta.top1_id);
            }
            if (cpu_probe_lm_head(lm_head_w, rms_host, config_.dim,
                                  candidate_ids, tmp_logits, cpu_decode_top1)) {
              delta.cpu_probe_top1_id = cpu_decode_top1;
              delta.cpu_probe_agrees_gpu = (cpu_decode_top1 == delta.top1_id);
            }
            prefill_delta.cpu_probe_prefill_top1 = cpu_prefill_top1;
            prefill_delta.cpu_probe_decode0_top1 = cpu_decode_top1;
            delta.cpu_probe_prefill_top1 = cpu_prefill_top1;
            delta.cpu_probe_decode0_top1 = cpu_decode_top1;
          }

          if (prefill_delta_ready && !prefill_delta_written) {
            log_delta(trace_delta_out, prefill_delta);
            prefill_delta_written = true;
          }
          log_delta(trace_delta_out, delta);
        }
        if (trace_hidden_equiv && trace_delta_out && i == 1 &&
            prefill_delta_ready) {
          HiddenEquivTrace eq{};
          eq.prefill_tokens_total = prefill_delta.tokens_total;
          eq.prefill_seq_len = prefill_delta.seq_len;
          eq.prefill_pos_id = prefill_delta.pos_id;
          eq.prefill_hash = prefill_delta.hidden_hash;
          eq.prefill_stats = prefill_delta.hidden_stats;
          eq.decode_tokens_total = tokens_total;
          eq.decode_seq_len = seq_len;
          eq.decode_pos_id = pos_id;
          eq.decode_hash = hhash;
          eq.decode_stats = hstats;
          log_hidden_equiv(trace_delta_out, eq);
        }
      }

      if (trace_landscape && trace_landscape_out) {
        log_landscape(trace_landscape_out, i, logits_host, landscape_topk);
      }

      next_token = sample(logits_host.data(), config_.vocab_size, params);

      if (align_callback) {
        AlignmentStep step;
        step.step = i;
        step.token_id = next_token;
        step.logit = logits_host[next_token];
        step.logit_min = logits_host[0];
        step.logit_max = logits_host[0];
        double sum = 0;
        step.nan_count = 0;
        step.inf_count = 0;
        std::vector<std::pair<float, int>> top;
        for (size_t j = 0; j < config_.vocab_size; ++j) {
          float v = logits_host[j];
          if (std::isnan(v))
            step.nan_count++;
          else if (std::isinf(v))
            step.inf_count++;
          else {
            if (v < step.logit_min)
              step.logit_min = v;
            if (v > step.logit_max)
              step.logit_max = v;
            sum += v;
          }
          top.push_back({v, (int)j});
        }
        step.logit_mean = (float)(sum / config_.vocab_size);
        std::sort(top.rbegin(), top.rend());
        for (int k = 0; k < 10 && k < (int)config_.vocab_size; ++k) {
          step.topk_ids.push_back(top[k].second);
          step.topk_logits.push_back(top[k].first);
        }
        align_callback(step);
      }
    }
    if (env_flag("GRETA_TRACE_LAYER")) {
      const size_t pos_id = output.size() - 1;
      const size_t seq_len = 1;
      const size_t tokens_total = output.size();
      const int32_t token_in = last_token_id;
      layer_trace_emit_step_header(i, pos_id, seq_len, tokens_total, token_in,
                                   next_token, config_);
    }

    output.push_back(next_token);
  }

  auto end = std::chrono::high_resolution_clock::now();
  if (stats) {
    stats->prompt_tokens = prompt_tokens.size();
    stats->generated_tokens = output.size() - prompt_tokens.size();
    stats->total_time_ms =
        std::chrono::duration<float, std::milli>(end - start).count();
    stats->time_to_first_token_ms =
        std::chrono::duration<float, std::milli>(first_token_time - start)
            .count();
    stats->tokens_per_second =
        stats->generated_tokens / (stats->total_time_ms / 1000.0f);
  }

  return output;
}

std::string Generator::generate(const std::string &prompt,
                                const SamplingParams &params,
                                GenerationStats *stats, TokenCallback callback,
                                AlignmentCallback align_callback) {
  auto prompt_tokens = tokenizer_->encode(prompt);

  std::string err;
  auto output_tokens =
      generate_tokens(prompt_tokens, params, stats, &err, align_callback);
  if (!err.empty()) {
    std::cerr << "Generation error: " << err << "\n";
  }

  std::vector<int32_t> generated(output_tokens.begin() + prompt_tokens.size(),
                                 output_tokens.end());

  if (callback) {
    for (auto id : generated) {
      callback(id, tokenizer_->decode_token(id));
    }
  }

  return tokenizer_->decode(generated);
}

} // namespace gcore::inference
