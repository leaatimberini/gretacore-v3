#include <vulkan/vulkan.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint32_t> read_spv_u32(const std::filesystem::path &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    std::cerr << "No se pudo abrir SPIR-V: " << p.string() << "\n";
    std::cout << "STATUS=FAILED reason=\"shader_read_failed\"\n";
    std::exit(1);
  }
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), {});
  if (bytes.size() % 4 != 0) {
    std::cerr << "SPIR-V size invalido (no multiplo de 4): " << p.string()
              << "\n";
    std::cout << "STATUS=FAILED reason=\"shader_read_failed\"\n";
    std::exit(1);
  }
  std::vector<uint32_t> out(bytes.size() / 4);
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

// --- FP16 pack helpers (host side) ---
static uint16_t f32_to_f16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  uint32_t sign = (u >> 31) & 1;
  int32_t exp = int32_t((u >> 23) & 0xFF) - 127;
  uint32_t mant = u & 0x7FFFFF;

  if (exp > 15) {
    return uint16_t((sign << 15) | (0x1F << 10));
  }
  if (exp < -14) {
    if (exp < -24)
      return uint16_t(sign << 15);
    mant |= 0x800000;
    int shift = (-14 - exp);
    uint32_t m = mant >> (shift + 13);
    return uint16_t((sign << 15) | (m & 0x3FF));
  }
  uint16_t he = uint16_t(exp + 15);
  uint16_t hm = uint16_t(mant >> 13);
  return uint16_t((sign << 15) | (he << 10) | hm);
}

static uint32_t pack_half2(float lo, float hi) {
  uint16_t hlo = f32_to_f16(lo);
  uint16_t hhi = f32_to_f16(hi);
  return uint32_t(hlo) | (uint32_t(hhi) << 16);
}

static void vk_check(VkResult r, const char *msg) {
  if (r != VK_SUCCESS) {
    std::cerr << "Vulkan error: " << msg << " (VkResult=" << r << ")\n";
    std::cout << "STATUS=FAILED reason=\"vulkan_error\"\n";
    std::exit(1);
  }
}

static uint32_t find_mem_type(VkPhysicalDevice phy, uint32_t typeBits,
                              VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp{};
  vkGetPhysicalDeviceMemoryProperties(phy, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if ((typeBits & (1u << i)) &&
        ((mp.memoryTypes[i].propertyFlags & props) == props))
      return i;
  }
  return UINT32_MAX;
}

static bool has_extension(const std::vector<VkExtensionProperties> &exts,
                          const char *name) {
  for (const auto &e : exts) {
    if (std::string(e.extensionName) == std::string(name))
      return true;
  }
  return false;
}

static bool env_true(const char *k) {
  const char *v = std::getenv(k);
  if (!v || !*v)
    return false;
  std::string s(v);
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return (char)std::tolower(c);
  });
  return (s == "1" || s == "true" || s == "yes" || s == "on");
}

static std::string driver_name_best_effort(VkPhysicalDevice phy,
                                           const std::vector<VkExtensionProperties> &exts) {
  if (!has_extension(exts, "VK_KHR_driver_properties"))
    return {};
  VkPhysicalDeviceDriverPropertiesKHR drv{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 p2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  p2.pNext = &drv;
  vkGetPhysicalDeviceProperties2(phy, &p2);
  return std::string(drv.driverName);
}

struct Buf {
  VkBuffer buf{};
  VkDeviceMemory mem{};
  VkDeviceSize size{};
  void *mapped{};
};

static Buf make_buffer(VkDevice dev, VkPhysicalDevice phy, VkDeviceSize size,
                       VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
  Buf b{};
  b.size = size;

  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = usage;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vk_check(vkCreateBuffer(dev, &bi, nullptr, &b.buf), "vkCreateBuffer");

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(dev, b.buf, &req);

  uint32_t mt = find_mem_type(phy, req.memoryTypeBits, props);
  if (mt == UINT32_MAX)
    throw std::runtime_error("No se encontró tipo de memoria compatible");

  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = mt;
  vk_check(vkAllocateMemory(dev, &ai, nullptr, &b.mem), "vkAllocateMemory");
  vk_check(vkBindBufferMemory(dev, b.buf, b.mem, 0), "vkBindBufferMemory");

  if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    vk_check(vkMapMemory(dev, b.mem, 0, size, 0, &b.mapped), "vkMapMemory");
  }
  return b;
}

static void destroy_buffer(VkDevice dev, Buf &b) {
  if (b.mapped)
    vkUnmapMemory(dev, b.mem);
  if (b.buf)
    vkDestroyBuffer(dev, b.buf, nullptr);
  if (b.mem)
    vkFreeMemory(dev, b.mem, nullptr);
  b = {};
}

static void print_device(VkPhysicalDevice phy) {
  VkPhysicalDeviceProperties p{};
  vkGetPhysicalDeviceProperties(phy, &p);
  std::cout << "Selected device:\n";
  std::cout << "  vendor_id=0x" << std::hex << p.vendorID << std::dec << "\n";
  std::cout << "  device_id=0x" << std::hex << p.deviceID << std::dec << "\n";
  std::cout << "  device_type=" << p.deviceType << "\n";
  std::cout << "  name=" << p.deviceName << "\n";
}

struct Push {
  uint32_t M, N, K;
  uint32_t lda, ldb, ldc;
};

int main(int argc, char **argv) {
  uint32_t M = 1024, N = 1024, K = 1024;
  int iters = 30;
  int batch = 20;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&](const char *k) {
      if (i + 1 >= argc) {
        std::cerr << "Falta valor para " << k << "\n";
        std::exit(1);
      }
    };
    if (a == "--m") {
      need("--m");
      M = uint32_t(std::stoul(argv[++i]));
    } else if (a == "--n") {
      need("--n");
      N = uint32_t(std::stoul(argv[++i]));
    } else if (a == "--k") {
      need("--k");
      K = uint32_t(std::stoul(argv[++i]));
    } else if (a == "--iters") {
      need("--iters");
      iters = std::stoi(argv[++i]);
    } else if (a == "--batch") {
      need("--batch");
      batch = std::stoi(argv[++i]);
    }
  }

  std::cout << "GRETA CORE Runtime Bench: vk_gemm_f16acc32_subgroup_ts_bench\n";
  std::cout << "M=" << M << " N=" << N << " K=" << K << " iters=" << iters
            << " batch=" << batch << "\n";

  // --- Instance ---
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "gretacore";
  app.applicationVersion = 1;
  app.pEngineName = "gretacore";
  app.engineVersion = 1;
  app.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;

  VkInstance inst{};
  vk_check(vkCreateInstance(&ici, nullptr, &inst), "vkCreateInstance");

  // --- Physical device ---
  uint32_t pcount = 0;
  vk_check(vkEnumeratePhysicalDevices(inst, &pcount, nullptr),
           "vkEnumeratePhysicalDevices(count)");
  if (pcount == 0) {
    std::cerr << "No hay GPUs Vulkan.\n";
    std::cout << "STATUS=FAILED reason=\"no_gpu\"\n";
    return 1;
  }
  std::vector<VkPhysicalDevice> phys(pcount);
  vk_check(vkEnumeratePhysicalDevices(inst, &pcount, phys.data()),
           "vkEnumeratePhysicalDevices");

  VkPhysicalDevice phy = phys[0];
  for (auto d : phys) {
    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(d, &p);
    std::string name = p.deviceName;
    if (name.find("llvmpipe") == std::string::npos) {
      phy = d;
      break;
    }
  }
  print_device(phy);

  // --- Subgroup properties + size control props ---
  VkPhysicalDeviceSubgroupProperties sg{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceSubgroupSizeControlPropertiesEXT sgctrlp{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT};
  VkPhysicalDeviceProperties2 p2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  p2.pNext = &sg;
  sg.pNext = &sgctrlp;
  vkGetPhysicalDeviceProperties2(phy, &p2);

  std::cout << "SUBGROUP:\n";
  std::cout << "  reported_subgroupSize=" << sg.subgroupSize << "\n";
  std::cout << "  sizeControl(min,max)=(" << sgctrlp.minSubgroupSize << ","
            << sgctrlp.maxSubgroupSize << ")\n";
  std::cout << "  maxComputeWorkgroupSubgroups="
            << sgctrlp.maxComputeWorkgroupSubgroups << "\n";

  // --- Queue family ---
  uint32_t qcount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phy, &qcount, nullptr);
  std::vector<VkQueueFamilyProperties> qprops(qcount);
  vkGetPhysicalDeviceQueueFamilyProperties(phy, &qcount, qprops.data());

  uint32_t qfam = UINT32_MAX;
  for (uint32_t i = 0; i < qcount; i++) {
    if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      qfam = i;
      break;
    }
  }
  if (qfam == UINT32_MAX) {
    std::cerr << "No compute queue.\n";
    return 1;
  }

  // --- Device extensions ---
  uint32_t extCount = 0;
  vk_check(
      vkEnumerateDeviceExtensionProperties(phy, nullptr, &extCount, nullptr),
      "vkEnumerateDeviceExtensionProperties(count)");
  std::vector<VkExtensionProperties> devExts(extCount);
  vk_check(vkEnumerateDeviceExtensionProperties(phy, nullptr, &extCount,
                                                devExts.data()),
           "vkEnumerateDeviceExtensionProperties");

  VkPhysicalDeviceProperties props0{};
  vkGetPhysicalDeviceProperties(phy, &props0);
  const std::string device_name = props0.deviceName;
  const std::string driver_name = driver_name_best_effort(phy, devExts);
  std::cout << "Driver name: " << driver_name << "\n";

  const bool is_radv = (driver_name.find("radv") != std::string::npos);
  const bool is_gfx1103 = (device_name.find("GFX1103") != std::string::npos);
  const bool blacklisted =
      (!env_true("GRETA_VK_ALLOW_UNSAFE") && is_radv && is_gfx1103);
  std::cout << "Safety:\n";
  std::cout << "  gpu_blacklisted=" << (blacklisted ? "true" : "false") << "\n";
  if (blacklisted) {
    std::cout << "SKIPPED: GPU blacklisted (RADV GFX1103). "
                 "Set GRETA_VK_ALLOW_UNSAFE=1 to override.\n";
    std::cout << "STATUS=SKIPPED reason=\"gpu_blacklisted\"\n";
    return 0;
  }

  const bool has_f16_ext =
      has_extension(devExts, "VK_KHR_shader_float16_int8");
  const bool has_16bit_ext = has_extension(devExts, "VK_KHR_16bit_storage");

  VkPhysicalDeviceFloat16Int8FeaturesKHR f16i8{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR};
  VkPhysicalDevice16BitStorageFeatures storage16{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  VkPhysicalDeviceFeatures2 feats2_query{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  feats2_query.pNext = &f16i8;
  f16i8.pNext = &storage16;
  vkGetPhysicalDeviceFeatures2(phy, &feats2_query);

  const bool fp16_supported = has_f16_ext && has_16bit_ext &&
                              (f16i8.shaderFloat16 == VK_TRUE) &&
                              (storage16.storageBuffer16BitAccess == VK_TRUE);
  std::cout << "Capabilities:\n";
  std::cout << "  fp16_supported=" << (fp16_supported ? "true" : "false")
            << "\n";
  if (!fp16_supported) {
    std::cout << "SKIPPED: FP16 not supported on this device\n";
    std::cout << "STATUS=SKIPPED reason=\"fp16_not_supported\"\n";
    return 0;
  }

  const char *ext_subgroup_size_control = "VK_EXT_subgroup_size_control";
  if (!has_extension(devExts, ext_subgroup_size_control)) {
    std::cerr << "Falta extension requerida: VK_EXT_subgroup_size_control\n";
    return 1;
  }

  std::vector<const char *> enabledExts;
  enabledExts.push_back(ext_subgroup_size_control);
  enabledExts.push_back("VK_KHR_shader_float16_int8");
  enabledExts.push_back("VK_KHR_16bit_storage");

  // --- Enable feature subgroupSizeControl ---
  VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sgctrlf{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
  VkPhysicalDeviceFeatures2 feats2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  feats2.pNext = &sgctrlf;
  vkGetPhysicalDeviceFeatures2(phy, &feats2);

  if (!sgctrlf.subgroupSizeControl) {
    std::cerr << "Feature subgroupSizeControl no soportado (aunque extensión "
                 "esté).\n";
    return 1;
  }
  sgctrlf.subgroupSizeControl = VK_TRUE;
  f16i8.shaderFloat16 = VK_TRUE;
  storage16.storageBuffer16BitAccess = VK_TRUE;

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qfam;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;

  VkPhysicalDeviceFeatures feats{};
  vkGetPhysicalDeviceFeatures(phy, &feats);

  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.pEnabledFeatures = &feats;
  dci.enabledExtensionCount = uint32_t(enabledExts.size());
  dci.ppEnabledExtensionNames = enabledExts.data();
  VkPhysicalDeviceFeatures2 feats2_enable{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  void *pnext = nullptr;
  storage16.pNext = pnext;
  pnext = &storage16;
  f16i8.pNext = pnext;
  pnext = &f16i8;
  sgctrlf.pNext = pnext;
  pnext = &sgctrlf;
  feats2_enable.pNext = pnext;
  dci.pNext = &feats2_enable;

  VkDevice dev{};
  vk_check(vkCreateDevice(phy, &dci, nullptr, &dev), "vkCreateDevice");

  VkQueue q{};
  vkGetDeviceQueue(dev, qfam, 0, &q);

  // --- Command pool/buffer ---
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.queueFamilyIndex = qfam;
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandPool pool{};
  vk_check(vkCreateCommandPool(dev, &pci, nullptr, &pool),
           "vkCreateCommandPool");

  VkCommandBufferAllocateInfo cai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  VkCommandBuffer cmd{};
  vk_check(vkAllocateCommandBuffers(dev, &cai, &cmd),
           "vkAllocateCommandBuffers");

  // --- Timestamp query pool ---
  VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpi.queryCount = 2;
  VkQueryPool qpool{};
  vk_check(vkCreateQueryPool(dev, &qpi, nullptr, &qpool), "vkCreateQueryPool");

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(phy, &props);
  std::cout << "TIMESTAMP:\n";
  std::cout << "  timestampPeriod(ns_per_tick)=" << props.limits.timestampPeriod
            << "\n";

  // --- Buffers ---
  if (K % 2 != 0 || N % 2 != 0) {
    std::cerr << "Este bench requiere K y N pares para half2 packing.\n";
    return 1;
  }

  uint32_t A_u32 = M * (K / 2);
  uint32_t B_u32 = K * (N / 2);
  uint32_t C_f32 = M * N;

  Buf A = make_buffer(dev, phy, VkDeviceSize(A_u32) * 4,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  Buf B = make_buffer(dev, phy, VkDeviceSize(B_u32) * 4,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  Buf C = make_buffer(dev, phy, VkDeviceSize(C_f32) * 4,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Debug buffer (1 uint32)
  Buf Dbg = make_buffer(dev, phy, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  *reinterpret_cast<uint32_t *>(Dbg.mapped) = 0;

  // Init A/B/C
  {
    auto *pa = reinterpret_cast<uint32_t *>(A.mapped);
    auto *pb = reinterpret_cast<uint32_t *>(B.mapped);
    auto *pcC = reinterpret_cast<float *>(C.mapped);

    for (uint32_t r = 0; r < M; r++) {
      for (uint32_t kk = 0; kk < K; kk += 2) {
        float a0 = float(((r + kk) % 13) - 6) * 0.01f;
        float a1 = float(((r + kk + 1) % 13) - 6) * 0.01f;
        pa[r * (K / 2) + (kk / 2)] = pack_half2(a0, a1);
      }
    }
    for (uint32_t kk = 0; kk < K; kk++) {
      for (uint32_t c = 0; c < N; c += 2) {
        float b0 = float(((kk + c) % 17) - 8) * 0.01f;
        float b1 = float(((kk + c + 1) % 17) - 8) * 0.01f;
        pb[kk * (N / 2) + (c / 2)] = pack_half2(b0, b1);
      }
    }
    std::fill(pcC, pcC + C_f32, 0.0f);
  }

  // --- Descriptor set layout (A,B,C,Dbg) ---
  VkDescriptorSetLayoutBinding b0{};
  b0.binding = 0;
  b0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  b0.descriptorCount = 1;
  b0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding b1{};
  b1.binding = 1;
  b1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  b1.descriptorCount = 1;
  b1.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding b2{};
  b2.binding = 2;
  b2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  b2.descriptorCount = 1;
  b2.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding b3{};
  b3.binding = 3;
  b3.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  b3.descriptorCount = 1;
  b3.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding binds[4] = {b0, b1, b2, b3};
  VkDescriptorSetLayoutCreateInfo dlci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dlci.bindingCount = 4;
  dlci.pBindings = binds;

  VkDescriptorSetLayout dsl{};
  vk_check(vkCreateDescriptorSetLayout(dev, &dlci, nullptr, &dsl),
           "vkCreateDescriptorSetLayout");

  // --- Pipeline layout (push constants) ---
  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = sizeof(Push);

  VkPipelineLayoutCreateInfo plci{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;

  VkPipelineLayout pl{};
  vk_check(vkCreatePipelineLayout(dev, &plci, nullptr, &pl),
           "vkCreatePipelineLayout");

  // --- Descriptor pool/set ---
  VkDescriptorPoolSize dps{};
  dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dps.descriptorCount = 4;

  VkDescriptorPoolCreateInfo dpci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &dps;

  VkDescriptorPool dp{};
  vk_check(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp),
           "vkCreateDescriptorPool");

  VkDescriptorSetAllocateInfo dsai{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = dp;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;

  VkDescriptorSet ds{};
  vk_check(vkAllocateDescriptorSets(dev, &dsai, &ds),
           "vkAllocateDescriptorSets");

  VkDescriptorBufferInfo abi{A.buf, 0, A.size};
  VkDescriptorBufferInfo bbi{B.buf, 0, B.size};
  VkDescriptorBufferInfo cbi{C.buf, 0, C.size};
  VkDescriptorBufferInfo dbi{Dbg.buf, 0, Dbg.size};

  VkWriteDescriptorSet w[4]{};

  w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  w[0].dstSet = ds;
  w[0].dstBinding = 0;
  w[0].descriptorCount = 1;
  w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  w[0].pBufferInfo = &abi;

  w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  w[1].dstSet = ds;
  w[1].dstBinding = 1;
  w[1].descriptorCount = 1;
  w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  w[1].pBufferInfo = &bbi;

  w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  w[2].dstSet = ds;
  w[2].dstBinding = 2;
  w[2].descriptorCount = 1;
  w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  w[2].pBufferInfo = &cbi;

  w[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  w[3].dstSet = ds;
  w[3].dstBinding = 3;
  w[3].descriptorCount = 1;
  w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  w[3].pBufferInfo = &dbi;

  vkUpdateDescriptorSets(dev, 4, w, 0, nullptr);

  // --- Shader module ---
  std::filesystem::path exe_dir = std::filesystem::path(argv[0]).parent_path();
  std::filesystem::path spv_path = exe_dir / "gemm_f16acc32_subgroup.comp.spv";

  if (!std::filesystem::exists(spv_path)) {
    std::cerr << "No encuentro el SPIR-V esperado.\n";
    std::cerr << "  esperado: " << spv_path.string() << "\n";
    std::cerr
        << "  tip: re-compilá y asegurate que Ninja genere *.spv en build/\n";
    return 1;
  }

  auto spv = read_spv_u32(spv_path);

  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spv.size() * sizeof(uint32_t);
  smci.pCode = spv.data();

  VkShaderModule sm{};
  vk_check(vkCreateShaderModule(dev, &smci, nullptr, &sm),
           "vkCreateShaderModule");

  // --- Compute pipeline ---
  VkPipelineShaderStageCreateInfo stage{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = sm;
  stage.pName = "main";

  // Force subgroup size = 32 at pipeline creation
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT reqSG{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT};
  reqSG.requiredSubgroupSize = 32;

  stage.flags |=
      VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT;
  stage.pNext = &reqSG;

  VkComputePipelineCreateInfo cpci{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage = stage;
  cpci.layout = pl;

  VkPipeline pipe{};
  vk_check(
      vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe),
      "vkCreateComputePipelines");

  // Workgroup mapping: x = ceil(N/8), y = M
  uint32_t gx = (N + 7) / 8;
  uint32_t gy = M;

  Push push{M, N, K, K, N, N};

  auto run_once = [&]() -> double {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vk_check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");

    vkCmdResetQueryPool(cmd, qpool, 0, 2);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds,
                            0, nullptr);
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push),
                       &push);

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, 0);
    for (int b = 0; b < batch; b++) {
      vkCmdDispatch(cmd, gx, gy, 1);
    }
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, 1);

    vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vk_check(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
    vk_check(vkQueueWaitIdle(q), "vkQueueWaitIdle");

    uint64_t ts[2]{};
    vk_check(vkGetQueryPoolResults(
                 dev, qpool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
             "vkGetQueryPoolResults");

    double ticks = double(ts[1] - ts[0]);
    double ns = ticks * double(props.limits.timestampPeriod);
    return ns / 1.0e6; // ms
  };

  // Warmup
  (void)run_once();

  // Read subgroup size written by shader
  uint32_t reported = *reinterpret_cast<uint32_t *>(Dbg.mapped);
  std::cout << "SUBGROUP (shader reported gl_SubgroupSize)=" << reported
            << "\n";

  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; i++)
    samples.push_back(run_once());

  std::sort(samples.begin(), samples.end());
  auto p50 = samples[size_t(0.50 * (samples.size() - 1))];
  auto p99 = samples[size_t(0.99 * (samples.size() - 1))];
  double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                double(samples.size());

  double ops = 2.0 * double(M) * double(N) * double(K) * double(batch);
  double mean_tflops = (ops / (mean / 1000.0)) / 1.0e12;
  double p50_tflops = (ops / (p50 / 1000.0)) / 1.0e12;
  double p99_tflops = (ops / (p99 / 1000.0)) / 1.0e12;

  std::cout << "RESULT vk_gemm_f16acc32_subgroup_ts_bench (GPU-only kernel):\n";
  std::cout << "  iters=" << iters << " batch=" << batch << "\n";
  std::cout << "  kernel_mean_ms=" << mean << "  kernel_p50_ms=" << p50
            << "  kernel_p99_ms=" << p99 << "\n";
  std::cout << "  mean_TFLOPs=" << mean_tflops << "  p50_TFLOPs=" << p50_tflops
            << "  p99_TFLOPs=" << p99_tflops << "\n";
  std::cout << "STATUS=OK\n";

  vkDeviceWaitIdle(dev);
  vkDestroyPipeline(dev, pipe, nullptr);
  vkDestroyShaderModule(dev, sm, nullptr);
  vkDestroyQueryPool(dev, qpool, nullptr);
  vkDestroyDescriptorPool(dev, dp, nullptr);
  vkDestroyPipelineLayout(dev, pl, nullptr);
  vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
  destroy_buffer(dev, A);
  destroy_buffer(dev, B);
  destroy_buffer(dev, C);
  destroy_buffer(dev, Dbg);
  vkDestroyCommandPool(dev, pool, nullptr);
  vkDestroyDevice(dev, nullptr);
  vkDestroyInstance(inst, nullptr);

  return 0;
}
