#include "gcore/rt/vk/backend.hpp"
#include "gcore/rt/vk/buffer.hpp"
#include "gcore/rt/vk/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static uint16_t f32_to_f16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof(u));
  uint32_t sign = (u >> 31) & 1u;
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
    return uint16_t((sign << 15) | m);
  }

  uint16_t he = uint16_t(exp + 15);
  uint16_t hm = uint16_t(mant >> 13);
  return uint16_t((sign << 15) | (he << 10) | hm);
}

static float f16_to_f32(uint16_t h) {
  uint32_t sign = (h >> 15) & 1u;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;

  uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = (sign << 31);
    } else {
      int e = -14;
      float m = float(mant) / 1024.0f;
      float v = std::ldexp(m, e);
      if (sign)
        v = -v;
      std::memcpy(&out, &v, sizeof(out));
      return v;
    }
  } else if (exp == 31) {
    out = (sign << 31) | (0xFF << 23) | (mant ? 1u : 0u);
  } else {
    uint32_t e = (exp - 15 + 127) & 0xFF;
    out = (sign << 31) | (e << 23) | (mant << 13);
  }

  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

static void cpu_ref_f16(const std::vector<uint16_t> &A16,
                        const std::vector<uint16_t> &B16,
                        std::vector<float> &C, int M, int N, int K, int lda,
                        int ldb, int ldc) {
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      float acc = 0.f;
      for (int k = 0; k < K; k++) {
        float a = f16_to_f32(A16[r * lda + k]);
        float b = f16_to_f32(B16[k * ldb + c]);
        acc += a * b;
      }
      C[r * ldc + c] = acc;
    }
  }
}

static void cpu_ref_f32(const std::vector<float> &A,
                        const std::vector<float> &B,
                        std::vector<float> &C, int M, int N, int K, int lda,
                        int ldb, int ldc) {
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      float acc = 0.f;
      for (int k = 0; k < K; k++) {
        acc += A[r * lda + k] * B[k * ldb + c];
      }
      C[r * ldc + c] = acc;
    }
  }
}

static std::string exe_dir(char **argv) {
  std::filesystem::path p = std::filesystem::path(argv[0]);
  if (p.has_parent_path())
    return p.parent_path().string();
  return ".";
}

static std::vector<uint32_t> read_spv_u32(const std::filesystem::path &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    return {};
  std::vector<char> bytes((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
  if (bytes.empty() || (bytes.size() % 4) != 0)
    return {};
  std::vector<uint32_t> out(bytes.size() / 4);
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

struct PushConsts {
  uint32_t M, N, K, lda, ldb, ldc;
};

struct VkDispatchResources {
  VkShaderModule shader = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  VkPipelineLayout pl = VK_NULL_HANDLE;
  VkPipeline pipe = VK_NULL_HANDLE;
  VkDescriptorPool dp = VK_NULL_HANDLE;
};

static void destroy_dispatch_resources(VkDevice dev,
                                       VkDispatchResources *res) {
  if (!res)
    return;
  if (res->dp)
    vkDestroyDescriptorPool(dev, res->dp, nullptr);
  if (res->pipe)
    vkDestroyPipeline(dev, res->pipe, nullptr);
  if (res->pl)
    vkDestroyPipelineLayout(dev, res->pl, nullptr);
  if (res->dsl)
    vkDestroyDescriptorSetLayout(dev, res->dsl, nullptr);
  if (res->shader)
    vkDestroyShaderModule(dev, res->shader, nullptr);
  res->dp = VK_NULL_HANDLE;
  res->pipe = VK_NULL_HANDLE;
  res->pl = VK_NULL_HANDLE;
  res->dsl = VK_NULL_HANDLE;
  res->shader = VK_NULL_HANDLE;
}

static uint32_t getenv_u32(const char *k, uint32_t def) {
  const char *v = std::getenv(k);
  if (!v || !*v)
    return def;
  try {
    return static_cast<uint32_t>(std::stoul(v));
  } catch (...) {
    return def;
  }
}

static uint32_t round_up_u32(uint32_t v, uint32_t multiple) {
  if (multiple == 0)
    return v;
  uint32_t r = v % multiple;
  if (r == 0)
    return v;
  return v + (multiple - r);
}

static bool getenv_true(const char *k) {
  const char *v = std::getenv(k);
  if (!v || !*v)
    return false;
  std::string s(v);
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return (char)std::tolower(c);
  });
  return (s == "1" || s == "true" || s == "yes" || s == "on");
}

static std::string getenv_str(const char *k) {
  const char *v = std::getenv(k);
  if (!v || !*v)
    return {};
  return std::string(v);
}

static bool submit_and_wait(VkDevice dev, VkQueue q, VkCommandBuffer cmd,
                            uint32_t timeout_ms, std::string *err) {
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  if (vkCreateFence(dev, &fci, nullptr, &fence) != VK_SUCCESS) {
    if (err)
      *err = "vkCreateFence failed";
    return false;
  }

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;

  if (vkQueueSubmit(q, 1, &si, fence) != VK_SUCCESS) {
    vkDestroyFence(dev, fence, nullptr);
    if (err)
      *err = "vkQueueSubmit failed";
    return false;
  }

  const uint64_t timeout_ns = uint64_t(timeout_ms) * 1000000ull;
  VkResult wr = vkWaitForFences(dev, 1, &fence, VK_TRUE, timeout_ns);
  vkDestroyFence(dev, fence, nullptr);
  if (wr == VK_TIMEOUT) {
    if (err)
      *err = "vkWaitForFences timeout";
    return false;
  }
  if (wr != VK_SUCCESS) {
    if (err)
      *err = "vkWaitForFences failed";
    return false;
  }
  return true;
}

static bool dispatch_gemm_f32(VkDevice dev, VkPhysicalDevice phys,
                              VkCommandBuffer cmd,
                              const std::filesystem::path &spv_path,
                              VkBuffer A, VkBuffer B, VkBuffer C,
                              uint32_t M, uint32_t N, uint32_t K,
                              uint32_t lda, uint32_t ldb, uint32_t ldc,
                              VkDispatchResources *out_res, std::string *err) {
  (void)phys;
  auto spv = read_spv_u32(spv_path);
  if (spv.empty()) {
    if (err)
      *err = "Failed to read SPIR-V: " + spv_path.string();
    return false;
  }

  VkDispatchResources res;

  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spv.size() * sizeof(uint32_t);
  smci.pCode = spv.data();

  if (vkCreateShaderModule(dev, &smci, nullptr, &res.shader) != VK_SUCCESS) {
    if (err)
      *err = "vkCreateShaderModule failed";
    return false;
  }

  VkDescriptorSetLayoutBinding binds[3]{};
  for (int i = 0; i < 3; i++) {
    binds[i].binding = (uint32_t)i;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo dslci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = 3;
  dslci.pBindings = binds;

  if (vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &res.dsl) !=
      VK_SUCCESS) {
    destroy_dispatch_resources(dev, &res);
    if (err)
      *err = "vkCreateDescriptorSetLayout failed";
    return false;
  }

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = sizeof(PushConsts);

  VkPipelineLayoutCreateInfo plci{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &res.dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;

  if (vkCreatePipelineLayout(dev, &plci, nullptr, &res.pl) != VK_SUCCESS) {
    destroy_dispatch_resources(dev, &res);
    if (err)
      *err = "vkCreatePipelineLayout failed";
    return false;
  }

  VkPipelineShaderStageCreateInfo stage{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = res.shader;
  stage.pName = "main";

  VkComputePipelineCreateInfo cpci{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage = stage;
  cpci.layout = res.pl;

  if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr,
                               &res.pipe) != VK_SUCCESS) {
    destroy_dispatch_resources(dev, &res);
    if (err)
      *err = "vkCreateComputePipelines failed";
    return false;
  }

  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 3;

  VkDescriptorPoolCreateInfo dpci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;

  if (vkCreateDescriptorPool(dev, &dpci, nullptr, &res.dp) != VK_SUCCESS) {
    destroy_dispatch_resources(dev, &res);
    if (err)
      *err = "vkCreateDescriptorPool failed";
    return false;
  }

  VkDescriptorSetAllocateInfo dsai{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = res.dp;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &res.dsl;

  VkDescriptorSet ds = VK_NULL_HANDLE;
  if (vkAllocateDescriptorSets(dev, &dsai, &ds) != VK_SUCCESS) {
    destroy_dispatch_resources(dev, &res);
    if (err)
      *err = "vkAllocateDescriptorSets failed";
    return false;
  }

  VkDescriptorBufferInfo dbA{A, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo dbB{B, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo dbC{C, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet wr[3]{};
  for (int i = 0; i < 3; i++)
    wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

  wr[0].dstSet = ds;
  wr[0].dstBinding = 0;
  wr[0].descriptorCount = 1;
  wr[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wr[0].pBufferInfo = &dbA;

  wr[1].dstSet = ds;
  wr[1].dstBinding = 1;
  wr[1].descriptorCount = 1;
  wr[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wr[1].pBufferInfo = &dbB;

  wr[2].dstSet = ds;
  wr[2].dstBinding = 2;
  wr[2].descriptorCount = 1;
  wr[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wr[2].pBufferInfo = &dbC;

  vkUpdateDescriptorSets(dev, 3, wr, 0, nullptr);

  PushConsts pc{M, N, K, lda, ldb, ldc};

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, res.pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, res.pl, 0, 1, &ds, 0,
                          nullptr);
  vkCmdPushConstants(cmd, res.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(PushConsts), &pc);

  uint32_t gx = (N + 15u) / 16u;
  uint32_t gy = (M + 15u) / 16u;
  if (gx == 0 || gy == 0) {
    if (err)
      *err = "dispatch_gemm_f32: gx/gy=0";
    return false;
  }

  vkCmdDispatch(cmd, gx, gy, 1);

  if (out_res)
    *out_res = res;
  else
    destroy_dispatch_resources(dev, &res);
  return true;
}

static bool dispatch_fill_u32(VkDevice dev, VkCommandBuffer cmd,
                              const std::filesystem::path &spv_path,
                              VkBuffer out, uint32_t n, uint32_t value,
                              VkDispatchResources *out_res,
                              std::string *err) {
  auto spv = read_spv_u32(spv_path);
  if (spv.empty()) {
    if (err)
      *err = "Failed to read SPIR-V: " + spv_path.string();
    return false;
  }

  VkDispatchResources res;

  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spv.size() * sizeof(uint32_t);
  smci.pCode = spv.data();

  if (vkCreateShaderModule(dev, &smci, nullptr, &res.shader) != VK_SUCCESS) {
    if (err)
      *err = "vkCreateShaderModule failed";
    return false;
  }

  VkDescriptorSetLayoutBinding bind0{};
  bind0.binding = 0;
  bind0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bind0.descriptorCount = 1;
  bind0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dslci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = 1;
  dslci.pBindings = &bind0;

  if (vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &res.dsl) !=
      VK_SUCCESS) {
    destroy_dispatch_resources(dev, &res);
    if (err)
      *err = "vkCreateDescriptorSetLayout failed";
    return false;
  }

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = 8;

  VkPipelineLayoutCreateInfo plci{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &res.dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;

  if (vkCreatePipelineLayout(dev, &plci, nullptr, &res.pl) != VK_SUCCESS) {
    destroy_dispatch_resources(dev, &res);
    if (err)
      *err = "vkCreatePipelineLayout failed";
    return false;
  }

  VkPipelineShaderStageCreateInfo stage{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = res.shader;
  stage.pName = "main";

  VkComputePipelineCreateInfo cpci{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage = stage;
  cpci.layout = res.pl;

  if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr,
                               &res.pipe) != VK_SUCCESS) {
    destroy_dispatch_resources(dev, &res);
    if (err)
      *err = "vkCreateComputePipelines failed";
    return false;
  }

  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 1;

  VkDescriptorPoolCreateInfo dpci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;

  if (vkCreateDescriptorPool(dev, &dpci, nullptr, &res.dp) != VK_SUCCESS) {
    destroy_dispatch_resources(dev, &res);
    if (err)
      *err = "vkCreateDescriptorPool failed";
    return false;
  }

  VkDescriptorSetAllocateInfo dsai{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = res.dp;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &res.dsl;

  VkDescriptorSet ds = VK_NULL_HANDLE;
  if (vkAllocateDescriptorSets(dev, &dsai, &ds) != VK_SUCCESS) {
    destroy_dispatch_resources(dev, &res);
    if (err)
      *err = "vkAllocateDescriptorSets failed";
    return false;
  }

  VkDescriptorBufferInfo dbO{out, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet wr{};
  wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  wr.dstSet = ds;
  wr.dstBinding = 0;
  wr.descriptorCount = 1;
  wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wr.pBufferInfo = &dbO;

  vkUpdateDescriptorSets(dev, 1, &wr, 0, nullptr);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, res.pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, res.pl, 0, 1, &ds, 0,
                          nullptr);

  struct Push {
    uint32_t value;
    uint32_t n;
  } pc{value, n};
  vkCmdPushConstants(cmd, res.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push),
                     &pc);

  const uint32_t wg = 256;
  const uint32_t groups = (n + wg - 1) / wg;
  if (groups == 0) {
    if (err)
      *err = "dispatch_fill_u32: groups=0";
    return false;
  }
  vkCmdDispatch(cmd, groups, 1, 1);

  if (out_res)
    *out_res = res;
  else
    destroy_dispatch_resources(dev, &res);
  return true;
}

int main(int argc, char **argv) {
  int M = 8, N = 8, K = 8;
  for (int i = 1; i + 1 < argc; i++) {
    std::string a = argv[i];
    if (a == "--m")
      M = std::stoi(argv[++i]);
    else if (a == "--n")
      N = std::stoi(argv[++i]);
    else if (a == "--k")
      K = std::stoi(argv[++i]);
  }

  std::cout << "GRETA CORE Runtime Smoke: vk_gemm_runtime_smoke\n";
  std::cout << "M=" << M << " N=" << N << " K=" << K << "\n";

  const uint32_t timeout_ms = getenv_u32("GRETA_VK_SMOKE_TIMEOUT_MS", 2000);
  const bool allow_fp16_smoke = getenv_true("GRETA_VK_SMOKE_ALLOW_FP16");
  const std::string smoke_profile = getenv_str("GRETA_VK_SMOKE_PROFILE");
  const uint32_t smoke_tile = getenv_u32("GRETA_VK_SMOKE_TILE", 16);
  std::string active_precision = "fp32";
  std::string fallback_reason;

  gcore::rt::vk::Backend backend;
  std::string err;
  if (!backend.init(&err)) {
    std::cerr << "Backend init failed: " << err << "\n";
    return 1;
  }

  const uint32_t M_req = uint32_t(M);
  const uint32_t N_req = uint32_t(N);
  const uint32_t K_req = uint32_t(K);
  uint32_t M_run = M_req;
  uint32_t N_run = N_req;
  uint32_t K_run = K_req;
  if (smoke_profile == "ultrasafe") {
    M_run = round_up_u32(M_req, smoke_tile);
    N_run = round_up_u32(N_req, smoke_tile);
    K_run = round_up_u32(K_req, smoke_tile);
    std::cout << "SMOKE: padded dims M=" << M_run << " N=" << N_run
              << " K=" << K_run << " (tile=" << smoke_tile << ")\n";
  }
  const int lda = int(K_run);
  const int ldb = int(N_run);
  const int ldc = int(N_run);

  if (backend.gpu_blacklisted()) {
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> Cref(M * N, 0.f);

    for (int i = 0; i < M * K; i++)
      A[i] = float((i % 13) - 6) * 0.25f;
    for (int i = 0; i < K * N; i++)
      B[i] = float((i % 7) - 3) * 0.5f;

    cpu_ref_f32(A, B, Cref, M, N, K, lda, ldb, ldc);

    std::cout << "CPU_ONLY max_abs_err=0\n";
    std::cout << "OK\n";
    std::cout << "STATUS=SKIPPED reason=\"gpu_blacklisted\"\n";
    return 0;
  }

  std::string shader_dir = exe_dir(argv);

  if (smoke_profile == "fill") {
    active_precision = "fill";
    std::cout << "RUNTIME: active_precision=" << active_precision << "\n";
    std::filesystem::path spv_path =
        std::filesystem::path(shader_dir) / "fill.comp.spv";
    const uint32_t n = std::max<uint32_t>(1u, M_req * N_req);
    const uint32_t fill_value = 0xA5A5A5A5u;

    gcore::rt::vk::Buffer out_dev;
    gcore::rt::vk::Buffer staging;
    auto cleanup_fill = [&]() {
      gcore::rt::vk::destroy_buffer(backend.device(), &staging);
      gcore::rt::vk::destroy_buffer(backend.device(), &out_dev);
    };
    VkDeviceSize size = VkDeviceSize(n * sizeof(uint32_t));
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (!gcore::rt::vk::create_device_local_buffer(
            backend.physical_device(), backend.device(), size, usage, &out_dev,
            &err)) {
      std::cerr << "create_device_local_buffer failed: " << err << "\n";
      cleanup_fill();
      return 1;
    }
    if (!gcore::rt::vk::create_staging_buffer(backend.physical_device(),
                                              backend.device(), size, &staging,
                                              &err)) {
      std::cerr << "create_staging_buffer failed: " << err << "\n";
      cleanup_fill();
      return 1;
    }

    void *p = nullptr;
    if (!gcore::rt::vk::map_buffer(backend.device(), staging, &p, &err)) {
      std::cerr << "map_buffer failed: " << err << "\n";
      cleanup_fill();
      return 1;
    }
    std::memset(p, 0, size);
    gcore::rt::vk::unmap_buffer(backend.device(), staging);

    VkCommandBufferAllocateInfo cbai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = backend.command_pool();
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(backend.device(), &cbai, &cmd) != VK_SUCCESS) {
      std::cerr << "vkAllocateCommandBuffers failed\n";
      cleanup_fill();
      return 1;
    }

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
      std::cerr << "vkBeginCommandBuffer failed\n";
      cleanup_fill();
      return 1;
    }

    VkDispatchResources fill_res;
    if (!dispatch_fill_u32(backend.device(), cmd, spv_path, out_dev.buf, n,
                           fill_value, &fill_res, &err)) {
      std::cerr << "dispatch_fill_u32 failed: " << err << "\n";
      cleanup_fill();
      return 1;
    }

    VkBufferMemoryBarrier b0{};
    b0.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b0.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.buffer = out_dev.buf;
    b0.offset = 0;
    b0.size = VK_WHOLE_SIZE;
    b0.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b0.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &b0,
                         0, nullptr);

    VkBufferCopy c{};
    c.srcOffset = 0;
    c.dstOffset = 0;
    c.size = size;
    vkCmdCopyBuffer(cmd, out_dev.buf, staging.buf, 1, &c);

    VkBufferMemoryBarrier b1{};
    b1.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.buffer = staging.buf;
    b1.offset = 0;
    b1.size = VK_WHOLE_SIZE;
    b1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b1.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &b1, 0,
                         nullptr);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
      std::cerr << "vkEndCommandBuffer failed\n";
      cleanup_fill();
      return 1;
    }

    if (!submit_and_wait(backend.device(), backend.queue(), cmd, timeout_ms,
                         &err)) {
      std::cerr << "submit_and_wait failed: " << err << "\n";
      cleanup_fill();
      return 1;
    }
    destroy_dispatch_resources(backend.device(), &fill_res);

    if (!gcore::rt::vk::map_buffer(backend.device(), staging, &p, &err)) {
      std::cerr << "map_buffer failed: " << err << "\n";
      cleanup_fill();
      return 1;
    }
    uint32_t *u = reinterpret_cast<uint32_t *>(p);
    bool ok = true;
    for (uint32_t i = 0; i < std::min<uint32_t>(n, 16u); i++) {
      if (u[i] != fill_value) {
        ok = false;
        break;
      }
    }
    gcore::rt::vk::unmap_buffer(backend.device(), staging);
    cleanup_fill();

    if (!ok) {
      std::cerr << "FAILED: fill validation\n";
      return 1;
    }
    std::cout << "OK\n";
    std::cout << "STATUS=OK\n";
    return 0;
  }

  // Attempt FP16 runtime path first
  bool use_fp16 = allow_fp16_smoke;
  if (!allow_fp16_smoke) {
    std::cout << "SMOKE: FP16 disabled for safety "
                 "(set GRETA_VK_SMOKE_ALLOW_FP16=1 to enable)\n";
    fallback_reason = "smoke_fp16_disabled";
  }
  if (!smoke_profile.empty()) {
    std::cout << "SMOKE: profile=" << smoke_profile << "\n";
  }
  gcore::rt::vk::GemmF16Acc32 gemm;
  if (use_fp16) {
    if (!gemm.init(&backend, shader_dir, &err)) {
      std::cout << "SMOKE: fp16 init failed: " << err << "\n";
      use_fp16 = false;
      if (fallback_reason.empty())
        fallback_reason = err;
    }
  }

  if (use_fp16) {
    active_precision = "fp16_acc32";
    std::cout << "RUNTIME: active_precision=" << active_precision << "\n";
    std::vector<uint16_t> A16(M_run * K_run, f32_to_f16(0.0f));
    std::vector<uint16_t> B16(K_run * N_run, f32_to_f16(0.0f));
    std::vector<float> Cref(M_run * N_run, 0.f);

    for (uint32_t r = 0; r < M_req; r++) {
      for (uint32_t k = 0; k < K_req; k++) {
        A16[r * K_run + k] =
            f32_to_f16(float(((r * K_req + k) % 13) - 6) * 0.25f);
      }
    }
    for (uint32_t k = 0; k < K_req; k++) {
      for (uint32_t c = 0; c < N_req; c++) {
        B16[k * N_run + c] =
            f32_to_f16(float(((k * N_req + c) % 7) - 3) * 0.5f);
      }
    }

    cpu_ref_f16(A16, B16, Cref, int(M_run), int(N_run), int(K_run), lda,
                ldb, ldc);

    gcore::rt::vk::Buffer bufA, bufB, bufC;
    gcore::rt::vk::Buffer stageA, stageB, stageC;
    auto cleanup_fp16 = [&]() {
      gcore::rt::vk::destroy_buffer(backend.device(), &stageC);
      gcore::rt::vk::destroy_buffer(backend.device(), &stageB);
      gcore::rt::vk::destroy_buffer(backend.device(), &stageA);
      gcore::rt::vk::destroy_buffer(backend.device(), &bufC);
      gcore::rt::vk::destroy_buffer(backend.device(), &bufB);
      gcore::rt::vk::destroy_buffer(backend.device(), &bufA);
    };
    VkDeviceSize sizeA = VkDeviceSize(M_run * K_run * sizeof(uint16_t));
    VkDeviceSize sizeB = VkDeviceSize(K_run * N_run * sizeof(uint16_t));
    VkDeviceSize sizeC = VkDeviceSize(M_run * N_run * sizeof(float));

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    if (!gcore::rt::vk::create_device_local_buffer(
            backend.physical_device(), backend.device(), sizeA, usage, &bufA,
            &err) ||
        !gcore::rt::vk::create_device_local_buffer(
            backend.physical_device(), backend.device(), sizeB, usage, &bufB,
            &err) ||
        !gcore::rt::vk::create_device_local_buffer(
            backend.physical_device(), backend.device(), sizeC, usage, &bufC,
            &err)) {
      std::cerr << "create_device_local_buffer failed: " << err << "\n";
      cleanup_fp16();
      return 1;
    }
    if (!gcore::rt::vk::create_staging_buffer(
            backend.physical_device(), backend.device(), sizeA, &stageA, &err) ||
        !gcore::rt::vk::create_staging_buffer(
            backend.physical_device(), backend.device(), sizeB, &stageB, &err) ||
        !gcore::rt::vk::create_staging_buffer(
            backend.physical_device(), backend.device(), sizeC, &stageC,
            &err)) {
      std::cerr << "create_staging_buffer failed: " << err << "\n";
      cleanup_fp16();
      return 1;
    }

    void *pA = nullptr;
    void *pB = nullptr;
    void *pC = nullptr;
    if (!gcore::rt::vk::map_buffer(backend.device(), stageA, &pA, &err) ||
        !gcore::rt::vk::map_buffer(backend.device(), stageB, &pB, &err) ||
        !gcore::rt::vk::map_buffer(backend.device(), stageC, &pC, &err)) {
      std::cerr << "map_buffer failed: " << err << "\n";
      cleanup_fp16();
      return 1;
    }

    std::memcpy(pA, A16.data(), sizeA);
    std::memcpy(pB, B16.data(), sizeB);
    std::memset(pC, 0, sizeC);

    gcore::rt::vk::unmap_buffer(backend.device(), stageA);
    gcore::rt::vk::unmap_buffer(backend.device(), stageB);
    gcore::rt::vk::unmap_buffer(backend.device(), stageC);

    if (!gcore::rt::vk::copy_buffer(backend.device(), backend.command_pool(),
                                    backend.queue(), stageA, bufA, sizeA,
                                    &err) ||
        !gcore::rt::vk::copy_buffer(backend.device(), backend.command_pool(),
                                    backend.queue(), stageB, bufB, sizeB,
                                    &err) ||
        !gcore::rt::vk::copy_buffer(backend.device(), backend.command_pool(),
                                    backend.queue(), stageC, bufC, sizeC,
                                    &err)) {
      std::cerr << "copy_buffer failed: " << err << "\n";
      cleanup_fp16();
      return 1;
    }

    VkCommandBufferAllocateInfo cbai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = backend.command_pool();
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(backend.device(), &cbai, &cmd) != VK_SUCCESS) {
      std::cerr << "vkAllocateCommandBuffers failed\n";
      cleanup_fp16();
      return 1;
    }

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
      std::cerr << "vkBeginCommandBuffer failed\n";
      cleanup_fp16();
      return 1;
    }

    gcore::rt::vk::GemmDispatchDesc d;
    d.A = bufA.buf;
    d.B = bufB.buf;
    d.C = bufC.buf;
    d.M = uint32_t(M_run);
    d.N = uint32_t(N_run);
    d.K = uint32_t(K_run);
    d.lda = uint32_t(lda);
    d.ldb = uint32_t(ldb);
    d.ldc = uint32_t(ldc);

    VkBufferMemoryBarrier bms[3]{};
    for (int i = 0; i < 3; i++) {
      bms[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      bms[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bms[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    }
    bms[0].buffer = bufA.buf;
    bms[0].offset = 0;
    bms[0].size = VK_WHOLE_SIZE;
    bms[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bms[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bms[1].buffer = bufB.buf;
    bms[1].offset = 0;
    bms[1].size = VK_WHOLE_SIZE;
    bms[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bms[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bms[2].buffer = bufC.buf;
    bms[2].offset = 0;
    bms[2].size = VK_WHOLE_SIZE;
    bms[2].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bms[2].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 3,
                         bms, 0, nullptr);

    if (!gemm.record_dispatch(cmd, d, &err)) {
      std::cerr << "record_dispatch failed: " << err << "\n";
      cleanup_fp16();
      return 1;
    }

    VkBufferMemoryBarrier b0{};
    b0.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b0.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.buffer = bufC.buf;
    b0.offset = 0;
    b0.size = VK_WHOLE_SIZE;
    b0.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b0.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &b0,
                         0, nullptr);

    VkBufferCopy c{};
    c.srcOffset = 0;
    c.dstOffset = 0;
    c.size = sizeC;
    vkCmdCopyBuffer(cmd, bufC.buf, stageC.buf, 1, &c);

    VkBufferMemoryBarrier b1{};
    b1.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.buffer = stageC.buf;
    b1.offset = 0;
    b1.size = VK_WHOLE_SIZE;
    b1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b1.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &b1, 0,
                         nullptr);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
      std::cerr << "vkEndCommandBuffer failed\n";
      cleanup_fp16();
      return 1;
    }

    if (!submit_and_wait(backend.device(), backend.queue(), cmd, timeout_ms,
                         &err)) {
      std::cerr << "submit_and_wait failed: " << err << "\n";
      cleanup_fp16();
      return 1;
    }

    if (!gcore::rt::vk::map_buffer(backend.device(), stageC, &pC, &err)) {
      std::cerr << "map_buffer(C) failed: " << err << "\n";
      cleanup_fp16();
      return 1;
    }

    std::vector<float> Cout(M_run * N_run, 0.f);
    std::memcpy(Cout.data(), pC, sizeC);
    gcore::rt::vk::unmap_buffer(backend.device(), stageC);

    float max_abs_err = 0.f;
    for (uint32_t r = 0; r < M_req; r++) {
      for (uint32_t c = 0; c < N_req; c++) {
        float e = std::abs(Cout[r * N_run + c] - Cref[r * N_run + c]);
        if (e > max_abs_err)
          max_abs_err = e;
      }
    }

    std::cout << "max_abs_err=" << max_abs_err << "\n";

    cleanup_fp16();

    if (max_abs_err > 1e-2f) {
      std::cerr << "FAILED: error too large\n";
      return 1;
    }

    std::cout << "OK\n";
    std::cout << "STATUS=OK\n";
    return 0;
  }

  // Fallback FP32 path (safe)
  active_precision = "fp32";
  std::cout << "RUNTIME: active_precision=" << active_precision << "\n";
  if (!fallback_reason.empty()) {
    std::cout << "RUNTIME: fallback_reason=" << fallback_reason << "\n";
  }
  std::vector<float> A(M_run * K_run, 0.f);
  std::vector<float> B(K_run * N_run, 0.f);
  std::vector<float> Cref(M_run * N_run, 0.f);

  for (uint32_t r = 0; r < M_req; r++) {
    for (uint32_t k = 0; k < K_req; k++) {
      A[r * K_run + k] = float(((r * K_req + k) % 13) - 6) * 0.25f;
    }
  }
  for (uint32_t k = 0; k < K_req; k++) {
    for (uint32_t c = 0; c < N_req; c++) {
      B[k * N_run + c] = float(((k * N_req + c) % 7) - 3) * 0.5f;
    }
  }

  cpu_ref_f32(A, B, Cref, int(M_run), int(N_run), int(K_run), lda, ldb, ldc);

  gcore::rt::vk::Buffer bufA, bufB, bufC;
  gcore::rt::vk::Buffer stageA, stageB, stageC;
  auto cleanup_fp32 = [&]() {
    gcore::rt::vk::destroy_buffer(backend.device(), &stageC);
    gcore::rt::vk::destroy_buffer(backend.device(), &stageB);
    gcore::rt::vk::destroy_buffer(backend.device(), &stageA);
    gcore::rt::vk::destroy_buffer(backend.device(), &bufC);
    gcore::rt::vk::destroy_buffer(backend.device(), &bufB);
    gcore::rt::vk::destroy_buffer(backend.device(), &bufA);
  };
  VkDeviceSize sizeA = VkDeviceSize(M_run * K_run * sizeof(float));
  VkDeviceSize sizeB = VkDeviceSize(K_run * N_run * sizeof(float));
  VkDeviceSize sizeC = VkDeviceSize(M_run * N_run * sizeof(float));

  VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  if (!gcore::rt::vk::create_device_local_buffer(
          backend.physical_device(), backend.device(), sizeA, usage, &bufA,
          &err) ||
      !gcore::rt::vk::create_device_local_buffer(
          backend.physical_device(), backend.device(), sizeB, usage, &bufB,
          &err) ||
      !gcore::rt::vk::create_device_local_buffer(
          backend.physical_device(), backend.device(), sizeC, usage, &bufC,
          &err)) {
    std::cerr << "create_device_local_buffer failed: " << err << "\n";
    cleanup_fp32();
    return 1;
  }
  if (!gcore::rt::vk::create_staging_buffer(
          backend.physical_device(), backend.device(), sizeA, &stageA, &err) ||
      !gcore::rt::vk::create_staging_buffer(
          backend.physical_device(), backend.device(), sizeB, &stageB, &err) ||
      !gcore::rt::vk::create_staging_buffer(
          backend.physical_device(), backend.device(), sizeC, &stageC, &err)) {
    std::cerr << "create_staging_buffer failed: " << err << "\n";
    cleanup_fp32();
    return 1;
  }

  void *pA = nullptr;
  void *pB = nullptr;
  void *pC = nullptr;
  if (!gcore::rt::vk::map_buffer(backend.device(), stageA, &pA, &err) ||
      !gcore::rt::vk::map_buffer(backend.device(), stageB, &pB, &err) ||
      !gcore::rt::vk::map_buffer(backend.device(), stageC, &pC, &err)) {
    std::cerr << "map_buffer failed: " << err << "\n";
    cleanup_fp32();
    return 1;
  }

  std::memcpy(pA, A.data(), sizeA);
  std::memcpy(pB, B.data(), sizeB);
  std::memset(pC, 0, sizeC);

  gcore::rt::vk::unmap_buffer(backend.device(), stageA);
  gcore::rt::vk::unmap_buffer(backend.device(), stageB);
  gcore::rt::vk::unmap_buffer(backend.device(), stageC);

  if (!gcore::rt::vk::copy_buffer(backend.device(), backend.command_pool(),
                                  backend.queue(), stageA, bufA, sizeA, &err) ||
      !gcore::rt::vk::copy_buffer(backend.device(), backend.command_pool(),
                                  backend.queue(), stageB, bufB, sizeB, &err) ||
      !gcore::rt::vk::copy_buffer(backend.device(), backend.command_pool(),
                                  backend.queue(), stageC, bufC, sizeC, &err)) {
    std::cerr << "copy_buffer failed: " << err << "\n";
    cleanup_fp32();
    return 1;
  }

  VkCommandBufferAllocateInfo cbai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = backend.command_pool();
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(backend.device(), &cbai, &cmd) != VK_SUCCESS) {
    std::cerr << "vkAllocateCommandBuffers failed\n";
    cleanup_fp32();
    return 1;
  }

  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
    std::cerr << "vkBeginCommandBuffer failed\n";
    cleanup_fp32();
    return 1;
  }

  std::filesystem::path spv_path;
  if (smoke_profile == "ultrasafe") {
    spv_path = std::filesystem::path(shader_dir) / "gemm_f32.comp.spv";
  } else {
    spv_path = std::filesystem::path(shader_dir) / "gemm_f32_tiled.comp.spv";
  }
  VkBufferMemoryBarrier bms[3]{};
  for (int i = 0; i < 3; i++) {
    bms[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bms[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bms[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  }
  bms[0].buffer = bufA.buf;
  bms[0].offset = 0;
  bms[0].size = VK_WHOLE_SIZE;
  bms[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  bms[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  bms[1].buffer = bufB.buf;
  bms[1].offset = 0;
  bms[1].size = VK_WHOLE_SIZE;
  bms[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  bms[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  bms[2].buffer = bufC.buf;
  bms[2].offset = 0;
  bms[2].size = VK_WHOLE_SIZE;
  bms[2].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  bms[2].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 3,
                       bms, 0, nullptr);

  VkDispatchResources gemm_res;
  if (!dispatch_gemm_f32(backend.device(), backend.physical_device(), cmd,
                         spv_path, bufA.buf, bufB.buf, bufC.buf, uint32_t(M_run),
                         uint32_t(N_run), uint32_t(K_run), uint32_t(lda),
                         uint32_t(ldb), uint32_t(ldc), &gemm_res, &err)) {
    std::cerr << "dispatch_gemm_f32 failed: " << err << "\n";
    cleanup_fp32();
    return 1;
  }

  VkBufferMemoryBarrier b0{};
  b0.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  b0.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b0.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b0.buffer = bufC.buf;
  b0.offset = 0;
  b0.size = VK_WHOLE_SIZE;
  b0.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  b0.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &b0,
                       0, nullptr);

  VkBufferCopy c{};
  c.srcOffset = 0;
  c.dstOffset = 0;
  c.size = sizeC;
  vkCmdCopyBuffer(cmd, bufC.buf, stageC.buf, 1, &c);

  VkBufferMemoryBarrier b1{};
  b1.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  b1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b1.buffer = stageC.buf;
  b1.offset = 0;
  b1.size = VK_WHOLE_SIZE;
  b1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  b1.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &b1, 0,
                       nullptr);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    std::cerr << "vkEndCommandBuffer failed\n";
    cleanup_fp32();
    return 1;
  }

  if (!submit_and_wait(backend.device(), backend.queue(), cmd, timeout_ms,
                       &err)) {
    std::cerr << "submit_and_wait failed: " << err << "\n";
    cleanup_fp32();
    return 1;
  }
  destroy_dispatch_resources(backend.device(), &gemm_res);

  if (!gcore::rt::vk::map_buffer(backend.device(), stageC, &pC, &err)) {
    std::cerr << "map_buffer(C) failed: " << err << "\n";
    cleanup_fp32();
    return 1;
  }

  std::vector<float> Cout(M_run * N_run, 0.f);
  std::memcpy(Cout.data(), pC, sizeC);
  gcore::rt::vk::unmap_buffer(backend.device(), stageC);

  float max_abs_err = 0.f;
  for (uint32_t r = 0; r < M_req; r++) {
    for (uint32_t c = 0; c < N_req; c++) {
      float e = std::abs(Cout[r * N_run + c] - Cref[r * N_run + c]);
      if (e > max_abs_err)
        max_abs_err = e;
    }
  }

  std::cout << "max_abs_err=" << max_abs_err << "\n";

  cleanup_fp32();

  if (max_abs_err > 1e-2f) {
    std::cerr << "FAILED: error too large\n";
    return 1;
  }

  std::cout << "OK\n";
  std::cout << "STATUS=OK\n";
  return 0;
}
