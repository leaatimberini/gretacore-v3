#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace gcore::rt::vk {

struct DeviceInfo {
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
  VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
  std::string name;
  std::string driver_name; // best-effort (may be empty)
};

class Backend {
public:
  Backend() = default;
  ~Backend();

  // Initialize Vulkan instance + pick device + create logical device + queue +
  // command pool.
  bool init(std::string *err);
  void shutdown();

  VkInstance instance() const { return instance_; }
  VkPhysicalDevice physical_device() const { return phys_; }
  VkDevice device() const { return device_; }
  VkQueue queue() const { return queue_; }
  VkCommandPool command_pool() const { return pool_; }
  uint32_t queue_family_index() const { return qfam_; }

  const DeviceInfo &device_info() const { return info_; }

  // True if VK_EXT_subgroup_size_control extension+feature was enabled on
  // VkDevice.
  bool subgroup_size_control_enabled() const {
    return subgroup_size_control_enabled_;
  }
  uint32_t min_subgroup_size() const { return min_subgroup_size_; }
  uint32_t max_subgroup_size() const { return max_subgroup_size_; }

  // True if shaderFloat16 + storageBuffer16BitAccess were enabled.
  bool fp16_enabled() const { return fp16_enabled_; }

  // True if device reports fp16 support (extensions+features), even if disabled.
  bool fp16_supported() const { return fp16_supported_; }

  // Reason for fp16 disabled or empty if enabled.
  const std::string &fp16_status_reason() const { return fp16_reason_; }

  // True if robustBufferAccess was enabled (core feature).
  bool robust_buffer_access_enabled() const {
    return robust_buffer_access_enabled_;
  }

  // True if device is in GPU safe-mode policy (allows dispatch with reduced
  // feature set).
  bool gpu_safe_mode() const { return gpu_safe_mode_; }

  // Reason for safe-mode or empty if not enabled.
  const std::string &safe_mode_reason() const { return safe_mode_reason_; }

  // True if device/driver is blacklisted for GPU dispatch.
  bool gpu_blacklisted() const { return gpu_blacklisted_; }

  // Reason for blacklist or empty if not blacklisted.
  const std::string &blacklist_reason() const { return blacklist_reason_; }

  // Print extended diagnostics (capabilities + safety decisions).
  void print_diagnostics(std::ostream &os) const;

  // Picks a physical device with compute queue (skips llvmpipe if possible).
  static bool pick_device(VkInstance inst, VkPhysicalDevice *out_phys,
                          uint32_t *out_qfam, std::string *err);

  // Simple empty submit for smoke/latency measurement.
  bool empty_submit(uint64_t *out_wait_ns, std::string *err);

private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice phys_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  uint32_t qfam_ = 0;

  DeviceInfo info_{};

  // Subgroup size control (if enabled)
  bool subgroup_size_control_enabled_ = false;
  uint32_t min_subgroup_size_ = 0;
  uint32_t max_subgroup_size_ = 0;

  // FP16 features (if enabled)
  bool fp16_enabled_ = false;
  bool fp16_supported_ = false;
  std::string fp16_reason_;

  // Core robustness
  bool robust_buffer_access_enabled_ = false;

  // GPU safe-mode policy
  bool gpu_safe_mode_ = false;
  std::string safe_mode_reason_;

  // GPU dispatch blacklist
  bool gpu_blacklisted_ = false;
  std::string blacklist_reason_;

  // Helpers
  static bool device_has_extension(VkPhysicalDevice phys, const char *ext_name);
  static std::string driver_name_best_effort(VkPhysicalDevice phys);
  static void probe_subgroup_size_control_props(VkPhysicalDevice phys,
                                                uint32_t &minS, uint32_t &maxS);
};

} // namespace gcore::rt::vk
