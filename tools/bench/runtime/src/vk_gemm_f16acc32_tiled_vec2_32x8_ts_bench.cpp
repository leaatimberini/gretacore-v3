#include "gcore/rt/vk/backend.hpp"
#include "gcore/rt/vk/buffer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static int argi(int argc, char **argv, const char *key, int def) {
  for (int i = 1; i + 1 < argc; i++) {
    if (std::string(argv[i]) == key)
      return std::stoi(argv[i + 1]);
  }
  return def;
}

static std::vector<uint32_t> read_spv_u32(const std::filesystem::path &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    return {};
  std::vector<char> bytes((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
  if (bytes.size() % 4 != 0)
    return {};
  std::vector<uint32_t> out(bytes.size() / 4);
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

static std::string vk_err_str(VkResult r) {
  return "VkResult=" + std::to_string(static_cast<int>(r));
}

static uint16_t f32_to_f16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof(u));
  uint32_t sign = (u >> 31) & 1;
  int32_t exp = int32_t((u >> 23) & 0xFF) - 127;
  uint32_t mant = u & 0x7FFFFF;

  if (exp > 15)
    return uint16_t((sign << 15) | (0x1F << 10));
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
  uint32_t sign = (h >> 15) & 1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;

  if (exp == 0 && mant == 0) {
    uint32_t out = (sign << 31);
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
  }
  if (exp == 31) {
    uint32_t out = (sign << 31) | (0xFF << 23) | (mant ? 1u : 0u);
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
  }
  if (exp == 0) {
    int e = -14;
    float m = float(mant) / 1024.0f;
    float v = std::ldexp(m, e);
    if (sign)
      v = -v;
    return v;
  }

  uint32_t e = (exp - 15 + 127) & 0xFF;
  uint32_t out = (sign << 31) | (e << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

static void cpu_ref_8x8_f16(const uint16_t *A16, const uint16_t *B16, float *C,
                            int M, int N, int K, int lda, int ldb, int ldc) {
  int rm = std::min(M, 8);
  int cn = std::min(N, 8);
  for (int r = 0; r < rm; r++) {
    for (int c = 0; c < cn; c++) {
      float acc = 0.f;
      for (int k = 0; k < K; k++) {
        acc += f16_to_f32(A16[r * lda + k]) * f16_to_f32(B16[k * ldb + c]);
      }
      C[r * ldc + c] = acc;
    }
  }
}

int main(int argc, char **argv) {
  const int M = argi(argc, argv, "--m", 1024);
  const int N = argi(argc, argv, "--n", 1024);
  const int K = argi(argc, argv, "--k", 1024);
  const int iters = argi(argc, argv, "--iters", 30);
  const int batch = std::max(1, argi(argc, argv, "--batch", 50));

  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  std::cout << "GRETA CORE Runtime Bench: "
               "vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench\n";
  std::cout << "M=" << M << " N=" << N << " K=" << K << " iters=" << iters
            << " batch=" << batch << "\n";

  gcore::rt::vk::Backend b;
  std::string err;
  if (!b.init(&err)) {
    std::cerr << "INIT FAILED: " << err << "\n";
    std::cout << "STATUS=FAILED reason=\"init_failed\"\n";
    return 1;
  }

  VkDevice dev = b.device();
  VkPhysicalDevice phys = b.physical_device();
  VkQueue q = b.queue();
  VkCommandPool pool = b.command_pool();

  auto info = b.device_info();
  std::cout << "Selected device:\n";
  std::cout << "  vendor_id=0x" << std::hex << info.vendor_id << std::dec
            << "\n";
  std::cout << "  device_id=0x" << std::hex << info.device_id << std::dec
            << "\n";
  std::cout << "  device_type=" << info.type << "\n";
  std::cout << "  name=" << info.name << "\n";
  std::cout << "  driver_name=" << info.driver_name << "\n";
  b.print_diagnostics(std::cout);

  if (b.gpu_blacklisted()) {
    std::cout << "SKIPPED: GPU blacklisted: " << b.blacklist_reason() << "\n";
    std::cout << "STATUS=SKIPPED reason=\"gpu_blacklisted\"\n";
    return 0;
  }
  if (!b.fp16_enabled()) {
    std::cout << "SKIPPED: FP16 not enabled: " << b.fp16_status_reason() << "\n";
    std::cout << "STATUS=SKIPPED reason=\"fp16_not_enabled\"\n";
    return 0;
  }

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(phys, &props);
  const double timestampPeriodNs = double(props.limits.timestampPeriod);
  std::cout << "TIMESTAMP:\n";
  std::cout << "  timestampPeriod(ns_per_tick)=" << timestampPeriodNs << "\n";

  std::filesystem::path spv_path = std::filesystem::current_path() / "build" /
                                   "gemm_f16acc32_tiled_vec2_32x8.comp.spv";
  auto spv = read_spv_u32(spv_path);
  if (spv.empty()) {
    std::cerr << "Failed to read SPIR-V: " << spv_path.string() << "\n";
    std::cout << "STATUS=FAILED reason=\"shader_read_failed\"\n";
    return 2;
  }

  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spv.size() * sizeof(uint32_t);
  smci.pCode = spv.data();

  VkShaderModule shader = VK_NULL_HANDLE;
  VkResult r = vkCreateShaderModule(dev, &smci, nullptr, &shader);
  if (r != VK_SUCCESS) {
    std::cerr << "vkCreateShaderModule failed: " << vk_err_str(r) << "\n";
    std::cout << "STATUS=FAILED reason=\"shader_module_failed\"\n";
    return 3;
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

  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  r = vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);
  if (r != VK_SUCCESS) {
    std::cerr << "vkCreateDescriptorSetLayout failed: " << vk_err_str(r)
              << "\n";
    std::cout << "STATUS=FAILED reason=\"dsl_failed\"\n";
    return 4;
  }

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = 24;

  VkPipelineLayoutCreateInfo plci{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;

  VkPipelineLayout pl = VK_NULL_HANDLE;
  r = vkCreatePipelineLayout(dev, &plci, nullptr, &pl);
  if (r != VK_SUCCESS)
    return 5;

  VkPipelineShaderStageCreateInfo stage{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = shader;
  stage.pName = "main";

  VkComputePipelineCreateInfo cpci{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage = stage;
  cpci.layout = pl;

  VkPipeline pipe = VK_NULL_HANDLE;
  r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);
  if (r != VK_SUCCESS)
    return 6;

  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 3;

  VkDescriptorPoolCreateInfo dpci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;

  VkDescriptorPool dp = VK_NULL_HANDLE;
  r = vkCreateDescriptorPool(dev, &dpci, nullptr, &dp);
  if (r != VK_SUCCESS)
    return 7;

  VkDescriptorSetAllocateInfo dsai{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = dp;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;

  VkDescriptorSet ds = VK_NULL_HANDLE;
  r = vkAllocateDescriptorSets(dev, &dsai, &ds);
  if (r != VK_SUCCESS)
    return 8;

  const VkDeviceSize bytesA =
      VkDeviceSize(M) * VkDeviceSize(K) * sizeof(uint16_t);
  const VkDeviceSize bytesB =
      VkDeviceSize(K) * VkDeviceSize(N) * sizeof(uint16_t);
  const VkDeviceSize bytesC = VkDeviceSize(M) * VkDeviceSize(N) * sizeof(float);

  gcore::rt::vk::Buffer bufA{}, bufB{}, bufC{};
  gcore::rt::vk::Buffer stageA{}, stageB{}, stageC{};
  auto cleanup = [&]() {
    gcore::rt::vk::destroy_buffer(dev, &stageC);
    gcore::rt::vk::destroy_buffer(dev, &stageB);
    gcore::rt::vk::destroy_buffer(dev, &stageA);
    gcore::rt::vk::destroy_buffer(dev, &bufC);
    gcore::rt::vk::destroy_buffer(dev, &bufB);
    gcore::rt::vk::destroy_buffer(dev, &bufA);
  };
  auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  if (!gcore::rt::vk::create_device_local_buffer(phys, dev, bytesA, usage,
                                                 &bufA, &err) ||
      !gcore::rt::vk::create_device_local_buffer(phys, dev, bytesB, usage,
                                                 &bufB, &err) ||
      !gcore::rt::vk::create_device_local_buffer(phys, dev, bytesC, usage,
                                                 &bufC, &err) ||
      !gcore::rt::vk::create_staging_buffer(phys, dev, bytesA, &stageA, &err) ||
      !gcore::rt::vk::create_staging_buffer(phys, dev, bytesB, &stageB, &err) ||
      !gcore::rt::vk::create_staging_buffer(phys, dev, bytesC, &stageC, &err)) {
    cleanup();
    return 11;
  }

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  auto upload_fp16 = [&](gcore::rt::vk::Buffer &stage,
                         const gcore::rt::vk::Buffer &device,
                         VkDeviceSize bytes) {
    return gcore::rt::vk::stage_host_to_device(
        dev, pool, q, stage, device, bytes,
        [&](void *ptr, VkDeviceSize sz) {
          auto *dst = reinterpret_cast<uint16_t *>(ptr);
          const size_t n = static_cast<size_t>(sz / sizeof(uint16_t));
          for (size_t i = 0; i < n; i++)
            dst[i] = f32_to_f16(dist(rng));
        },
        &err);
  };
  auto zero_fp32 = [&](gcore::rt::vk::Buffer &stage,
                       const gcore::rt::vk::Buffer &device,
                       VkDeviceSize bytes) {
    return gcore::rt::vk::stage_host_to_device(
        dev, pool, q, stage, device, bytes,
        [](void *ptr, VkDeviceSize sz) {
          std::memset(ptr, 0, static_cast<size_t>(sz));
        },
        &err);
  };

  if (!upload_fp16(stageA, bufA, bytesA) ||
      !upload_fp16(stageB, bufB, bytesB) ||
      !zero_fp32(stageC, bufC, bytesC)) {
    cleanup();
    return 12;
  }

  VkDescriptorBufferInfo dbA{bufA.buf, 0, bytesA};
  VkDescriptorBufferInfo dbB{bufB.buf, 0, bytesB};
  VkDescriptorBufferInfo dbC{bufC.buf, 0, bytesC};

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

  VkCommandBufferAllocateInfo cbai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  r = vkAllocateCommandBuffers(dev, &cbai, &cmd);
  if (r != VK_SUCCESS)
    return 13;

  VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpci.queryCount = uint32_t(2 * batch);

  VkQueryPool qp = VK_NULL_HANDLE;
  r = vkCreateQueryPool(dev, &qpci, nullptr, &qp);
  if (r != VK_SUCCESS)
    return 14;

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};

  // Shader mapping:
  // - filas por WG = 8
  // - "hilos X" por WG = 16, cada hilo hace 2 cols => 32 cols por WG
  const uint32_t gx = (uint32_t(N) + 32u - 1u) / 32u;
  const uint32_t gy = (uint32_t(M) + 8u - 1u) / 8u;

  struct Push {
    uint32_t M, N, K, lda, ldb, ldc;
  } pc{(uint32_t)M,   (uint32_t)N,   (uint32_t)K,
       (uint32_t)lda, (uint32_t)ldb, (uint32_t)ldc};

  std::vector<double> kernel_ms;
  kernel_ms.reserve((size_t)iters);

  bool validation_ok = true;
  bool timing_ok = true;

  for (int it = 0; it < iters; it++) {
    if (vkResetCommandPool(dev, pool, 0) != VK_SUCCESS)
      break;

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
      break;

    vkCmdResetQueryPool(cmd, qp, 0, uint32_t(2 * batch));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds,
                            0, nullptr);
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push),
                       &pc);

    for (int j = 0; j < batch; j++) {
      uint32_t q0 = uint32_t(2 * j);
      uint32_t q1 = uint32_t(2 * j + 1);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qp, q0);
      vkCmdDispatch(cmd, gx, gy, 1);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qp, q1);
    }

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
      break;

    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(dev, &fci, nullptr, &fence) != VK_SUCCESS)
      break;

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    if (vkQueueSubmit(q, 1, &si, fence) != VK_SUCCESS) {
      vkDestroyFence(dev, fence, nullptr);
      break;
    }
    if (vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
      vkDestroyFence(dev, fence, nullptr);
      break;
    }
    vkDestroyFence(dev, fence, nullptr);

    std::vector<uint64_t> ts(2 * batch, 0);
    if (vkGetQueryPoolResults(
            dev, qp, 0, uint32_t(2 * batch), ts.size() * sizeof(uint64_t),
            ts.data(), sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
      break;

    double sum_ns = 0.0;
    for (int j = 0; j < batch; j++) {
      uint64_t a0 = ts[2 * j];
      uint64_t a1 = ts[2 * j + 1];
      if (a1 > a0)
        sum_ns += double(a1 - a0) * timestampPeriodNs;
    }
    kernel_ms.push_back(sum_ns / 1e6);
  }

  // Validation
  std::vector<float> Cref(M * N, 0.f);
  std::vector<uint16_t> hostA(size_t(M) * size_t(K));
  std::vector<uint16_t> hostB(size_t(K) * size_t(N));
  std::vector<float> hostC(size_t(M) * size_t(N));
  if (gcore::rt::vk::read_device_to_host(
          dev, pool, q, bufA, stageA, bytesA,
          [&](const void *ptr, VkDeviceSize sz) {
            std::memcpy(hostA.data(), ptr, static_cast<size_t>(sz));
          },
          &err) &&
      gcore::rt::vk::read_device_to_host(
          dev, pool, q, bufB, stageB, bytesB,
          [&](const void *ptr, VkDeviceSize sz) {
            std::memcpy(hostB.data(), ptr, static_cast<size_t>(sz));
          },
          &err) &&
      gcore::rt::vk::read_device_to_host(
          dev, pool, q, bufC, stageC, bytesC,
          [&](const void *ptr, VkDeviceSize sz) {
            std::memcpy(hostC.data(), ptr, static_cast<size_t>(sz));
          },
          &err)) {

    cpu_ref_8x8_f16(hostA.data(), hostB.data(), Cref.data(), M, N, K, lda, ldb,
                    ldc);

    bool ok = true;
    double max_abs = 0.0;
    for (int r0 = 0; r0 < std::min(M, 8); r0++) {
      for (int c0 = 0; c0 < std::min(N, 8); c0++) {
        double absd = std::abs(
            double(hostC[size_t(r0) * size_t(ldc) + size_t(c0)]) -
            double(Cref[size_t(r0) * size_t(ldc) + size_t(c0)]));
        max_abs = std::max(max_abs, absd);
        if (absd > 2e-1)
          ok = false;
      }
    }
    std::cout << "VALIDATION(8x8): " << (ok ? "OK" : "FAILED")
              << "  max_abs_err=" << max_abs << "\n";
    if (!ok)
      validation_ok = false;
  } else {
    std::cerr << "read_device_to_host for validation failed: " << err << "\n";
    validation_ok = false;
  }

  if (!kernel_ms.empty()) {
    auto km = kernel_ms;
    std::sort(km.begin(), km.end());
    double mean = 0;
    for (auto x : kernel_ms)
      mean += x;
    mean /= kernel_ms.size();
    double p50 = km[km.size() / 2];
    double p99 =
        km[(size_t)std::min<size_t>(km.size() - 1, (km.size() * 99) / 100)];

    double flops_per_dispatch = 2.0 * double(M) * double(N) * double(K);
    double flops_total = flops_per_dispatch * double(batch);

    auto tf = [&](double ms) {
      double s = ms / 1e3;
      return (flops_total / s) / 1e12;
    };

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "RESULT vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench (GPU-only "
                 "kernel):\n";
    std::cout << "  iters=" << kernel_ms.size() << " batch=" << batch << "\n";
    std::cout << "  kernel_mean_ms=" << mean << "  kernel_p50_ms=" << p50
              << "  kernel_p99_ms=" << p99 << "\n";
    std::cout << "  mean_TFLOPs=" << tf(mean) << "  p50_TFLOPs=" << tf(p50)
              << "  p99_TFLOPs=" << tf(p99) << "\n";
  } else {
    std::cerr << "No kernel timestamp samples collected.\n";
    timing_ok = false;
  }
  if (!timing_ok) {
    std::cout << "STATUS=FAILED reason=\"no_timing\"\n";
  } else if (!validation_ok) {
    std::cout << "STATUS=FAILED reason=\"validation_failed\"\n";
  } else {
    std::cout << "STATUS=OK\n";
  }

  cleanup();

  vkDestroyQueryPool(dev, qp, nullptr);
  vkDestroyDescriptorPool(dev, dp, nullptr);
  vkDestroyPipeline(dev, pipe, nullptr);
  vkDestroyPipelineLayout(dev, pl, nullptr);
  vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
  vkDestroyShaderModule(dev, shader, nullptr);
  if (!timing_ok)
    return 11;
  if (!validation_ok)
    return 12;
  return 0;
}
