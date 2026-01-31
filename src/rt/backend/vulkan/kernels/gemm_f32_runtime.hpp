#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

namespace gcore::rt::vk {

enum class GemmF32KernelId : uint32_t {
  tiled = 0,
};

const char *gemm_f32_kernel_name(GemmF32KernelId id);
const char *gemm_f32_kernel_spv_filename(GemmF32KernelId id);

struct GemmF32RunArgs {
  uint32_t M = 0;
  uint32_t N = 0;
  uint32_t K = 0;
  uint32_t lda = 0;
  uint32_t ldb = 0;
  uint32_t ldc = 0;
};

struct GemmF32PushConstants {
  uint32_t M, N, K, lda, ldb, ldc; // 24 bytes
};

struct GemmF32DispatchParams {
  VkBuffer A = VK_NULL_HANDLE;
  VkBuffer B = VK_NULL_HANDLE;
  VkBuffer C = VK_NULL_HANDLE;

  GemmF32RunArgs args;
};

class GemmF32PipelineCache {
public:
  explicit GemmF32PipelineCache(VkDevice dev) : dev_(dev) {}
  ~GemmF32PipelineCache() { destroy(); }

  void set_shader_dir(std::string dir) { shader_dir_ = std::move(dir); }

  bool get_or_create(VkPhysicalDevice phys, GemmF32KernelId kid, std::string *err);

  VkDescriptorSetLayout dsl() const { return dsl_; }
  VkPipelineLayout pl() const { return pl_; }
  VkPipeline pipe(GemmF32KernelId kid) const;

  void destroy();

private:
  struct Entry {
    bool ready = false;
    VkPipeline pipe = VK_NULL_HANDLE;
  };

  VkDevice dev_ = VK_NULL_HANDLE;
  std::string shader_dir_ = "./build";

  VkDescriptorSetLayout dsl_ = VK_NULL_HANDLE;
  VkPipelineLayout pl_ = VK_NULL_HANDLE;

  std::unordered_map<uint32_t, Entry> pipes_;

  bool ensure_common_layouts(std::string *err);
  bool create_entry(VkPhysicalDevice phys, GemmF32KernelId kid, Entry &e,
                    std::string *err);

  std::vector<uint32_t> read_spv_u32(const std::string &filename,
                                     std::string *err) const;
};

bool probe_gemm_f32(VkDevice dev, VkPhysicalDevice phys, GemmF32PipelineCache &cache,
                    GemmF32KernelId kid, std::string *err);

bool dispatch_gemm_f32(VkDevice dev, VkPhysicalDevice phys, VkCommandBuffer cmd,
                       GemmF32PipelineCache &cache, GemmF32KernelId kid,
                       const GemmF32DispatchParams &p);

} // namespace gcore::rt::vk
