#pragma once

#include "gcore/rt/vk/backend.hpp"

#include <cstdint>
#include <string>

#include <vulkan/vulkan.h>

namespace gcore::rt::vk {

// GEMM variant IDs (must match autotune cache winners)
enum class GemmVariant {
  TiledF16Acc32,  // "tiled_f16acc32"  -> gemm_f16acc32_tiled.comp.spv
  TiledVec2,      // "tiled_vec2"      -> gemm_f16acc32_tiled_vec2.comp.spv
  TiledVec2_32x8, // "tiled_vec2_32x8" -> gemm_f16acc32_tiled_vec2_32x8.comp.spv
  TiledVec2_DB,   // "tiled_vec2_db"   -> gemm_f16acc32_tiled_vec2_db.comp.spv
  Subgroup32,     // "subgroup"        -> gemm_f16acc32_subgroup.comp.spv
};

enum class GemmPrecision {
  F16Acc32,
  F32,
};

struct GemmDispatchDesc {
  VkBuffer A = VK_NULL_HANDLE;
  VkBuffer B = VK_NULL_HANDLE;
  VkBuffer C = VK_NULL_HANDLE;

  uint32_t M = 0;
  uint32_t N = 0;
  uint32_t K = 0;

  uint32_t lda = 0;
  uint32_t ldb = 0;
  uint32_t ldc = 0;
};

// Runtime GEMM (FP16 inputs, FP32 accum/output) using Vulkan compute pipelines.
// Kernel selection policy:
//   1) GRETA_VK_AUTOTUNE_FORCE=<winner-name>  (optional)
//   2) cache ~/.cache/gretacore/vk_autotune.json (si existe)
//   3) fallback: tiled_vec2_32x8
//
// This class is intentionally thin: it delegates pipeline creation/dispatch
// to src/rt/backend/vulkan/kernels/gemm_f16acc32_runtime.{hpp,cpp}.
class GemmF16Acc32 {
public:
  GemmF16Acc32() = default;
  ~GemmF16Acc32();

  // Initialize with an already initialized Backend.
  // shader_dir:
  // - If empty: uses GRETA_VK_SHADER_DIR if set, else "./build"
  bool init(Backend *backend, std::string shader_dir, std::string *err);

  void shutdown();

  // Records commands into cmd (does NOT submit).
  // Requires:
  // - cmd in recording state
  // - buffers created with VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
  bool record_dispatch(VkCommandBuffer cmd, const GemmDispatchDesc &d,
                       std::string *err);

  // Debug/telemetry
  GemmVariant last_variant() const { return last_variant_; }
  std::string last_winner_name() const { return last_winner_name_; }

private:
  Backend *backend_ = nullptr;
  std::string shader_dir_;

  // Opaque runtime cache (pimpl to keep header stable)
  struct Impl;
  Impl *impl_ = nullptr;

  GemmVariant last_variant_ = GemmVariant::TiledVec2_32x8;
  std::string last_winner_name_;

  static const char *variant_name(GemmVariant v);
};

// Runtime GEMM F32 (FP32 inputs, FP32 accum/output).
class GemmF32 {
public:
  GemmF32() = default;
  ~GemmF32();

  bool init(Backend *backend, std::string shader_dir, std::string *err);
  void shutdown();

  bool record_dispatch(VkCommandBuffer cmd, const GemmDispatchDesc &d,
                       std::string *err);

private:
  Backend *backend_ = nullptr;
  std::string shader_dir_;

  struct Impl;
  Impl *impl_ = nullptr;
};

// Auto GEMM: chooses precision based on backend support.
class GemmAuto {
public:
  GemmAuto() = default;
  ~GemmAuto();

  // preferred selects desired precision, but may fallback.
  bool init(Backend *backend, std::string shader_dir, GemmPrecision preferred,
            std::string *err);
  void shutdown();

  bool record_dispatch(VkCommandBuffer cmd, const GemmDispatchDesc &d,
                       std::string *err);

  GemmPrecision active_precision() const { return active_precision_; }
  std::string fallback_reason() const { return fallback_reason_; }

private:
  Backend *backend_ = nullptr;
  GemmPrecision active_precision_ = GemmPrecision::F32;
  std::string fallback_reason_;

  GemmF16Acc32 f16_;
  GemmF32 f32_;
};

} // namespace gcore::rt::vk
