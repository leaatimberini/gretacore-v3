#include "gcore/rt/vk/backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

namespace gcore::rt::vk {

static bool contains_llvmpipe(const char *name) {
  if (!name)
    return false;
  std::string s = name;
  return s.find("llvmpipe") != std::string::npos;
}

static std::string getenv_str(const char *k) {
  const char *v = std::getenv(k);
  if (!v || !*v)
    return {};
  return std::string(v);
}

static bool env_true(const char *k) {
  std::string s = getenv_str(k);
  if (s.empty())
    return false;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return (char)std::tolower(c);
  });
  return (s == "1" || s == "true" || s == "yes" || s == "on");
}

Backend::~Backend() { shutdown(); }

bool Backend::device_has_extension(VkPhysicalDevice phys,
                                   const char *ext_name) {
  uint32_t n = 0;
  if (vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr) !=
      VK_SUCCESS)
    return false;
  std::vector<VkExtensionProperties> exts(n);
  if (n) {
    if (vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, exts.data()) !=
        VK_SUCCESS)
      return false;
  }
  for (auto &e : exts) {
    if (std::strcmp(e.extensionName, ext_name) == 0)
      return true;
  }
  return false;
}

std::string Backend::driver_name_best_effort(VkPhysicalDevice phys) {
  // VK_KHR_driver_properties is a device extension. If present, we can query
  // driverName.
  if (!device_has_extension(phys, "VK_KHR_driver_properties"))
    return {};

  VkPhysicalDeviceDriverPropertiesKHR drv{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 p2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  p2.pNext = &drv;
  vkGetPhysicalDeviceProperties2(phys, &p2);
  return std::string(drv.driverName);
}

void Backend::probe_subgroup_size_control_props(VkPhysicalDevice phys,
                                                uint32_t &minS,
                                                uint32_t &maxS) {
  minS = 0;
  maxS = 0;

  if (!device_has_extension(phys, "VK_EXT_subgroup_size_control"))
    return;

  VkPhysicalDeviceSubgroupSizeControlPropertiesEXT sgctrl{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT};
  VkPhysicalDeviceSubgroupProperties sg{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 p2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};

  p2.pNext = &sg;
  sg.pNext = &sgctrl;
  vkGetPhysicalDeviceProperties2(phys, &p2);

  minS = sgctrl.minSubgroupSize;
  maxS = sgctrl.maxSubgroupSize;
}

bool Backend::pick_device(VkInstance inst, VkPhysicalDevice *out_phys,
                          uint32_t *out_qfam, std::string *err) {
  uint32_t ndev = 0;
  VkResult r = vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
  if (r != VK_SUCCESS || ndev == 0) {
    if (err)
      *err = "No Vulkan physical devices found.";
    return false;
  }

  std::vector<VkPhysicalDevice> devs(ndev);
  r = vkEnumeratePhysicalDevices(inst, &ndev, devs.data());
  if (r != VK_SUCCESS) {
    if (err)
      *err = "vkEnumeratePhysicalDevices failed.";
    return false;
  }

  // Prefer non-llvmpipe device with compute queue.
  VkPhysicalDevice best_dev = VK_NULL_HANDLE;
  uint32_t best_qfam = 0;

  auto try_pick = [&](bool allow_llvmpipe) -> bool {
    for (auto d : devs) {
      VkPhysicalDeviceProperties props{};
      vkGetPhysicalDeviceProperties(d, &props);

      if (!allow_llvmpipe && contains_llvmpipe(props.deviceName))
        continue;

      uint32_t nq = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
      std::vector<VkQueueFamilyProperties> qprops(nq);
      vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qprops.data());

      for (uint32_t i = 0; i < nq; i++) {
        if (qprops[i].queueCount > 0 &&
            (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
          best_dev = d;
          best_qfam = i;
          return true;
        }
      }
    }
    return false;
  };

  if (!try_pick(false)) {
    if (!try_pick(true)) {
      if (err)
        *err = "No suitable Vulkan physical device with compute queue found.";
      return false;
    }
  }

  *out_phys = best_dev;
  *out_qfam = best_qfam;
  return true;
}

bool Backend::init(std::string *err) {
  shutdown();

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "gretacore_runtime";
  app.applicationVersion = 1;
  app.pEngineName = "gretacore";
  app.engineVersion = 1;
  app.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ci.pApplicationInfo = &app;

  VkResult r = vkCreateInstance(&ci, nullptr, &instance_);
  if (r != VK_SUCCESS) {
    if (err)
      *err = "vkCreateInstance failed.";
    shutdown();
    return false;
  }

  if (!pick_device(instance_, &phys_, &qfam_, err)) {
    shutdown();
    return false;
  }

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(phys_, &props);

  info_.vendor_id = props.vendorID;
  info_.device_id = props.deviceID;
  info_.type = props.deviceType;
  info_.name = std::string(props.deviceName);
  info_.driver_name = driver_name_best_effort(phys_);

  const bool is_radv = (info_.driver_name.find("radv") != std::string::npos);
  const bool is_gfx1103 = (info_.name.find("GFX1103") != std::string::npos);
  const bool is_radv_gfx1103 = is_radv && is_gfx1103;

  // Decide if we can enable subgroup size control (extension + feature)
  const bool has_sg_ext =
      device_has_extension(phys_, "VK_EXT_subgroup_size_control");
  uint32_t minS = 0, maxS = 0;
  probe_subgroup_size_control_props(phys_, minS, maxS);
  min_subgroup_size_ = minS;
  max_subgroup_size_ = maxS;

  // Check FP16-related extensions
  const bool has_f16_int8_ext =
      device_has_extension(phys_, "VK_KHR_shader_float16_int8");
  const bool has_16bit_storage_ext =
      device_has_extension(phys_, "VK_KHR_16bit_storage");

  // Core features (robustBufferAccess)
  VkPhysicalDeviceFeatures core_feats{};
  vkGetPhysicalDeviceFeatures(phys_, &core_feats);
  const bool can_enable_robust = (core_feats.robustBufferAccess == VK_TRUE);
  robust_buffer_access_enabled_ = can_enable_robust;

  // Query features (subgroup + fp16)
  VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sgctrlf{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
  VkPhysicalDeviceFloat16Int8FeaturesKHR f16i8{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR};
  VkPhysicalDevice16BitStorageFeatures storage16{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};

  VkPhysicalDeviceFeatures2 feats2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  feats2.pNext = &sgctrlf;
  sgctrlf.pNext = &f16i8;
  f16i8.pNext = &storage16;

  vkGetPhysicalDeviceFeatures2(phys_, &feats2);

  bool can_enable_sgctrl =
      has_sg_ext && (sgctrlf.subgroupSizeControl == VK_TRUE);

  bool can_enable_fp16 = has_f16_int8_ext && has_16bit_storage_ext &&
                         (f16i8.shaderFloat16 == VK_TRUE) &&
                         (storage16.storageBuffer16BitAccess == VK_TRUE);
  fp16_supported_ = can_enable_fp16;
  fp16_reason_.clear();
  if (!fp16_supported_) {
    std::ostringstream oss;
    oss << "fp16 not supported (";
    bool first = true;
    if (!has_f16_int8_ext) {
      oss << "missing VK_KHR_shader_float16_int8";
      first = false;
    }
    if (!has_16bit_storage_ext) {
      if (!first)
        oss << ", ";
      oss << "missing VK_KHR_16bit_storage";
      first = false;
    }
    if (has_f16_int8_ext && !f16i8.shaderFloat16) {
      if (!first)
        oss << ", ";
      oss << "feature shaderFloat16=false";
      first = false;
    }
    if (has_16bit_storage_ext && !storage16.storageBuffer16BitAccess) {
      if (!first)
        oss << ", ";
      oss << "feature storageBuffer16BitAccess=false";
    }
    oss << ")";
    fp16_reason_ = oss.str();
  }

  // GPU safe-mode policy (default ON for RADV GFX1103 unless disabled).
  gpu_safe_mode_ = false;
  safe_mode_reason_.clear();
  if (is_radv_gfx1103 && !env_true("GRETA_VK_DISABLE_SAFE_MODE")) {
    gpu_safe_mode_ = true;
    safe_mode_reason_ =
        "RADV GFX1103 safe-mode (FP32 only, robustBufferAccess if available)";
  }

  // Safety: disable FP16 on RADV GFX1103 unless explicitly forced.
  if (!env_true("GRETA_VK_FORCE_FP16")) {
    if (is_radv_gfx1103) {
      can_enable_fp16 = false;
      fp16_reason_ =
          "disabled by safety policy for RADV GFX1103 (set GRETA_VK_FORCE_FP16=1 to override)";
    }
  }

  // Optional: force-disable FP16 even if supported/enabled (safety override).
  const bool force_fp32_env = env_true("GRETA_VK_FORCE_FP32") ||
                              env_true("GRETA_VK_DISABLE_FP16");
  const bool force_fp32 = force_fp32_env || gpu_safe_mode_;
  if (force_fp32) {
    can_enable_fp16 = false;
    if (fp16_reason_.empty()) {
      fp16_reason_ = force_fp32_env
                         ? "fp16 forced off by GRETA_VK_FORCE_FP32=1"
                         : "fp16 disabled by safe-mode";
    } else {
      fp16_reason_ = (force_fp32_env ? "fp16 forced off by GRETA_VK_FORCE_FP32=1; "
                                     : "fp16 disabled by safe-mode; ") +
                     fp16_reason_;
    }
  }

  // GPU dispatch blacklist (avoid system hang on known bad combos)
  gpu_blacklisted_ = false;
  blacklist_reason_.clear();
  if (!env_true("GRETA_VK_ALLOW_UNSAFE") && !gpu_safe_mode_) {
    if (is_radv_gfx1103) {
      gpu_blacklisted_ = true;
      blacklist_reason_ =
          "RADV GFX1103 blacklisted (set GRETA_VK_ALLOW_UNSAFE=1 to override)";
    }
  }

  // Queue create
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qfam_;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;

  std::vector<const char *> enabledExts;

  if (can_enable_sgctrl) {
    enabledExts.push_back("VK_EXT_subgroup_size_control");
    // keep feature enabled in create
    sgctrlf.subgroupSizeControl = VK_TRUE;
    subgroup_size_control_enabled_ = true;
  } else {
    sgctrlf.subgroupSizeControl = VK_FALSE;
    subgroup_size_control_enabled_ = false;
  }

  if (can_enable_fp16) {
    enabledExts.push_back("VK_KHR_shader_float16_int8");
    enabledExts.push_back("VK_KHR_16bit_storage");
    f16i8.shaderFloat16 = VK_TRUE;
    storage16.storageBuffer16BitAccess = VK_TRUE;
    fp16_enabled_ = true;
    fp16_reason_.clear();
  } else {
    f16i8.shaderFloat16 = VK_FALSE;
    storage16.storageBuffer16BitAccess = VK_FALSE;
    fp16_enabled_ = false;
    if (fp16_reason_.empty())
      fp16_reason_ = "fp16 disabled";
  }

  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = (uint32_t)enabledExts.size();
  dci.ppEnabledExtensionNames =
      enabledExts.empty() ? nullptr : enabledExts.data();

  // If enabling any features, pass Features2 via pNext.
  VkPhysicalDeviceFeatures2 feats2_enable{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  feats2_enable.features = {};
  feats2_enable.features.robustBufferAccess =
      robust_buffer_access_enabled_ ? VK_TRUE : VK_FALSE;

  void *pnext = nullptr;
  if (fp16_enabled_) {
    storage16.pNext = pnext;
    pnext = &storage16;
    f16i8.pNext = pnext;
    pnext = &f16i8;
  } else {
    storage16.pNext = nullptr;
    f16i8.pNext = nullptr;
  }

  if (subgroup_size_control_enabled_) {
    sgctrlf.pNext = pnext;
    pnext = &sgctrlf;
  } else {
    sgctrlf.pNext = pnext;
  }

  const bool any_features =
      (robust_buffer_access_enabled_ || fp16_enabled_ ||
       subgroup_size_control_enabled_);
  if (any_features) {
    feats2_enable.pNext = pnext;
    dci.pNext = &feats2_enable;
  }

  r = vkCreateDevice(phys_, &dci, nullptr, &device_);
  if (r != VK_SUCCESS) {
    if (err) {
      std::ostringstream oss;
      oss << "vkCreateDevice failed. subgroupSizeControl enable attempted="
          << (can_enable_sgctrl ? "yes" : "no")
          << ", fp16 enable attempted=" << (can_enable_fp16 ? "yes" : "no");
      *err = oss.str();
    }
    shutdown();
    return false;
  }

  vkGetDeviceQueue(device_, qfam_, 0, &queue_);

  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.queueFamilyIndex = qfam_;
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  r = vkCreateCommandPool(device_, &pci, nullptr, &pool_);
  if (r != VK_SUCCESS) {
    if (err)
      *err = "vkCreateCommandPool failed.";
    shutdown();
    return false;
  }

  return true;
}

bool Backend::empty_submit(uint64_t *out_wait_ns, std::string *err) {
  if (device_ == VK_NULL_HANDLE || queue_ == VK_NULL_HANDLE ||
      pool_ == VK_NULL_HANDLE) {
    if (err)
      *err = "empty_submit: backend no inicializado";
    return false;
  }

  VkCommandBufferAllocateInfo cbai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = pool_;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(device_, &cbai, &cmd) != VK_SUCCESS) {
    if (err)
      *err = "empty_submit: vkAllocateCommandBuffers failed";
    return false;
  }

  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
    if (err)
      *err = "empty_submit: vkBeginCommandBuffer failed";
    return false;
  }

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    if (err)
      *err = "empty_submit: vkEndCommandBuffer failed";
    return false;
  }

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;

  auto t0 = std::chrono::steady_clock::now();
  VkResult r = vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
  if (r != VK_SUCCESS) {
    if (err)
      *err = "empty_submit: vkQueueSubmit failed";
    return false;
  }

  r = vkQueueWaitIdle(queue_);
  auto t1 = std::chrono::steady_clock::now();
  if (r != VK_SUCCESS) {
    if (err)
      *err = "empty_submit: vkQueueWaitIdle failed";
    return false;
  }

  if (out_wait_ns) {
    *out_wait_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
            .count());
  }

  return true;
}

void Backend::print_diagnostics(std::ostream &os) const {
  os << "Capabilities:\n";
  os << "  subgroup_size_control_enabled="
     << (subgroup_size_control_enabled_ ? "true" : "false") << "\n";
  os << "  subgroup_size_range=[" << min_subgroup_size_ << ", "
     << max_subgroup_size_ << "]\n";
  os << "  fp16_supported=" << (fp16_supported_ ? "true" : "false") << "\n";
  os << "  fp16_enabled=" << (fp16_enabled_ ? "true" : "false") << "\n";
  if (!fp16_enabled_ && !fp16_reason_.empty())
    os << "  fp16_reason=" << fp16_reason_ << "\n";
  os << "  robust_buffer_access_enabled="
     << (robust_buffer_access_enabled_ ? "true" : "false") << "\n";
  os << "Safety:\n";
  os << "  gpu_safe_mode=" << (gpu_safe_mode_ ? "true" : "false") << "\n";
  if (gpu_safe_mode_ && !safe_mode_reason_.empty())
    os << "  safe_mode_reason=" << safe_mode_reason_ << "\n";
  os << "  gpu_blacklisted=" << (gpu_blacklisted_ ? "true" : "false") << "\n";
  if (gpu_blacklisted_ && !blacklist_reason_.empty())
    os << "  blacklist_reason=" << blacklist_reason_ << "\n";
}

void Backend::shutdown() {
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);
  }

  if (pool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device_, pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
  }

  if (device_ != VK_NULL_HANDLE) {
    vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;
  }

  if (instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
  }

  phys_ = VK_NULL_HANDLE;
  queue_ = VK_NULL_HANDLE;
  qfam_ = 0;

  info_ = DeviceInfo{};
  subgroup_size_control_enabled_ = false;
  min_subgroup_size_ = 0;
  max_subgroup_size_ = 0;
  fp16_enabled_ = false;
  fp16_supported_ = false;
  fp16_reason_.clear();
  robust_buffer_access_enabled_ = false;
  gpu_safe_mode_ = false;
  safe_mode_reason_.clear();
  gpu_blacklisted_ = false;
  blacklist_reason_.clear();
}

} // namespace gcore::rt::vk
