#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

namespace gcore::rt::vk {

// Debe matchear nombres de winners del autotune/cache
enum class GemmKernelId : uint32_t {
  tiled_f16acc32 = 0,
  tiled_vec2 = 1,
  tiled_vec2_32x8 = 2,
  tiled_vec2_db = 3,
  subgroup = 4, // requiere requiredSubgroupSize=32
};

std::optional<GemmKernelId> kernel_from_name(const std::string &name);
const char *kernel_name(GemmKernelId id);
const char *kernel_spv_filename(GemmKernelId id);

struct GemmRunArgs {
  uint32_t M = 0;
  uint32_t N = 0;
  uint32_t K = 0;
  uint32_t lda = 0;
  uint32_t ldb = 0;
  uint32_t ldc = 0;
};

struct GemmPushConstants {
  uint32_t M, N, K, lda, ldb, ldc; // 24 bytes
};

struct GemmDispatchParams {
  // Buffers storage (binding 0=A, 1=B, 2=C)
  VkBuffer A = VK_NULL_HANDLE;
  VkBuffer B = VK_NULL_HANDLE;
  VkBuffer C = VK_NULL_HANDLE;

  GemmRunArgs args;

  // Optional policy:
  // si gx/gy=0 por algún motivo, el runtime puede decidir fallback.
  // Hoy lo dejamos simple.
};

class GemmPipelineCache {
public:
  explicit GemmPipelineCache(VkDevice dev) : dev_(dev) {}
  ~GemmPipelineCache() { destroy(); }

  // shader_dir: directorio donde están los .spv (por ej: "./build")
  void set_shader_dir(std::string dir) { shader_dir_ = std::move(dir); }

  // Habilita o no subgroup size control (lo decide Backend al crear device)
  void set_subgroup_size_control(bool enabled) {
    subgroup_size_control_ = enabled;
  }

  bool subgroup_size_control_enabled() const { return subgroup_size_control_; }

  bool get_or_create(VkPhysicalDevice phys, GemmKernelId kid, std::string *err);

  VkDescriptorSetLayout dsl() const { return dsl_; }
  VkPipelineLayout pl() const { return pl_; }
  VkPipeline pipe(GemmKernelId kid) const;

  void destroy();

private:
  struct Entry {
    bool ready = false;
    VkPipeline pipe = VK_NULL_HANDLE;
  };

  VkDevice dev_ = VK_NULL_HANDLE;
  std::string shader_dir_ = "./build";
  bool subgroup_size_control_ = false;

  VkDescriptorSetLayout dsl_ = VK_NULL_HANDLE;
  VkPipelineLayout pl_ = VK_NULL_HANDLE;

  std::unordered_map<uint32_t, Entry> pipes_;

  bool ensure_common_layouts(std::string *err);
  bool create_entry(VkPhysicalDevice phys, GemmKernelId kid, Entry &e,
                    std::string *err);

  std::vector<uint32_t> read_spv_u32(const std::string &filename,
                                     std::string *err) const;
};

// 1) Resuelve kernel id usando:
//    - GRETA_VK_AUTOTUNE_FORCE=<name> (si seteado)
//    - cache ~/.cache/gretacore/vk_autotune.json (si existe)
//    - fallback (tiled_vec2_32x8)
std::optional<GemmKernelId>
resolve_gemm_kernel_id_autotune(uint32_t M, uint32_t N, uint32_t K);

bool probe_gemm_f16acc32(VkDevice dev, VkPhysicalDevice phys,
                         GemmPipelineCache &cache, GemmKernelId kid,
                         std::string *err);

// Ejecuta dispatch con kernel explícito (call site “bajo nivel”)
bool dispatch_gemm_f16acc32(VkDevice dev, VkPhysicalDevice phys,
                            VkCommandBuffer cmd, GemmPipelineCache &cache,
                            GemmKernelId kid, const GemmDispatchParams &p);

// Ejecuta dispatch con kernel “auto” (resuelve kid por autotune/cache/env)
bool dispatch_gemm_f16acc32_auto(VkDevice dev, VkPhysicalDevice phys,
                                 VkCommandBuffer cmd, GemmPipelineCache &cache,
                                 const GemmDispatchParams &p, std::string *err);

} // namespace gcore::rt::vk
