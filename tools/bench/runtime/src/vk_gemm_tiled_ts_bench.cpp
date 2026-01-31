#include "gcore/rt/vk/backend.hpp"
#include "gcore/rt/vk/buffer.hpp"

#include <algorithm>
#include <chrono>
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

static void cpu_ref_8x8(const float *A, const float *B, float *C, int M, int N,
                        int K, int lda, int ldb, int ldc) {
  int rm = std::min(M, 8);
  int cn = std::min(N, 8);
  for (int r = 0; r < rm; r++) {
    for (int c = 0; c < cn; c++) {
      float acc = 0.f;
      for (int k = 0; k < K; k++)
        acc += A[r * lda + k] * B[k * ldb + c];
      C[r * ldc + c] = acc;
    }
  }
}

int main(int argc, char **argv) {
  const int M = argi(argc, argv, "--m", 1024);
  const int N = argi(argc, argv, "--n", 1024);
  const int K = argi(argc, argv, "--k", 1024);

  // iters = cuántas muestras querés reportar
  const int iters = argi(argc, argv, "--iters", 30);
  const bool compute_only = argi(argc, argv, "--compute-only", 0) != 0;

  // batch = cuántos dispatch seguidos dentro de 1 submit (reduce overhead)
  const int batch = std::max(1, argi(argc, argv, "--batch", 20));

  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  std::cout << "GRETA CORE Runtime Bench: vk_gemm_tiled_ts_bench\n";
  std::cout << "M=" << M << " N=" << N << " K=" << K << " iters=" << iters
            << " batch=" << batch << " compute_only=" << (compute_only ? 1 : 0)
            << "\n";

  gcore::rt::vk::Backend b;
  std::string err;
  if (!b.init(&err)) {
    std::cerr << "INIT FAILED: " << err << "\n";
    std::cout << "STATUS=FAILED reason=\"init_failed\"\n";
    return 1;
  }

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

  VkDevice dev = b.device();
  VkPhysicalDevice phys = b.physical_device();
  VkQueue q = b.queue();
  VkCommandPool pool = b.command_pool();

  // Timestamp capabilities
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(phys, &props);
  const double timestampPeriodNs = double(props.limits.timestampPeriod);
  std::cout << "TIMESTAMP:\n";
  std::cout << "  timestampPeriod(ns_per_tick)=" << timestampPeriodNs << "\n";

  // SPIR-V (tiled)
  std::filesystem::path spv_path =
      std::filesystem::current_path() / "build" / "gemm_f32_tiled.comp.spv";
  auto spv = read_spv_u32(spv_path);
  if (spv.empty()) {
    std::filesystem::path alt =
        std::filesystem::current_path() / "tools" / "bench" / "runtime" /
        "build" / "gemm_f32_tiled.comp.spv";
    spv = read_spv_u32(alt);
    if (!spv.empty())
      spv_path = alt;
  }
  if (spv.empty()) {
    std::cerr << "Failed to read SPIR-V: " << spv_path.string() << "\n";
    std::cout << "STATUS=FAILED reason=\"shader_read_failed\"\n";
    return 2;
  }

  VkShaderModuleCreateInfo smci{};
  smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smci.codeSize = spv.size() * sizeof(uint32_t);
  smci.pCode = spv.data();

  VkShaderModule shader = VK_NULL_HANDLE;
  VkResult r = vkCreateShaderModule(dev, &smci, nullptr, &shader);
  if (r != VK_SUCCESS) {
    std::cerr << "vkCreateShaderModule failed: " << vk_err_str(r) << "\n";
    std::cout << "STATUS=FAILED reason=\"shader_module_failed\"\n";
    return 3;
  }

  // Descriptors
  VkDescriptorSetLayoutBinding binds[3]{};
  for (int i = 0; i < 3; i++) {
    binds[i].binding = (uint32_t)i;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = 3;
  dslci.pBindings = binds;

  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  r = vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);
  if (r != VK_SUCCESS) {
    std::cerr << "vkCreateDescriptorSetLayout failed: " << vk_err_str(r)
              << "\n";
    vkDestroyShaderModule(dev, shader, nullptr);
    return 4;
  }

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = 24; // 6 u32

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;

  VkPipelineLayout pl = VK_NULL_HANDLE;
  r = vkCreatePipelineLayout(dev, &plci, nullptr, &pl);
  if (r != VK_SUCCESS) {
    std::cerr << "vkCreatePipelineLayout failed: " << vk_err_str(r) << "\n";
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyShaderModule(dev, shader, nullptr);
    return 5;
  }

  VkPipelineShaderStageCreateInfo stage{};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = shader;
  stage.pName = "main";

  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage = stage;
  cpci.layout = pl;

  VkPipeline pipe = VK_NULL_HANDLE;
  r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);
  if (r != VK_SUCCESS) {
    std::cerr << "vkCreateComputePipelines failed: " << vk_err_str(r) << "\n";
    return 6;
  }

  // Descriptor pool
  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 3;

  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;

  VkDescriptorPool dp = VK_NULL_HANDLE;
  r = vkCreateDescriptorPool(dev, &dpci, nullptr, &dp);
  if (r != VK_SUCCESS) {
    std::cerr << "vkCreateDescriptorPool failed: " << vk_err_str(r) << "\n";
    return 7;
  }

  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = dp;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;

  VkDescriptorSet ds = VK_NULL_HANDLE;
  r = vkAllocateDescriptorSets(dev, &dsai, &ds);
  if (r != VK_SUCCESS) {
    std::cerr << "vkAllocateDescriptorSets failed: " << vk_err_str(r) << "\n";
    return 8;
  }

  // Buffers
  const VkDeviceSize bytesA = VkDeviceSize(M) * VkDeviceSize(K) * sizeof(float);
  const VkDeviceSize bytesB = VkDeviceSize(K) * VkDeviceSize(N) * sizeof(float);
  const VkDeviceSize bytesC = VkDeviceSize(M) * VkDeviceSize(N) * sizeof(float);

  gcore::rt::vk::Buffer bufA{}, bufB{}, bufC{};
  gcore::rt::vk::Buffer stageA{}, stageB{}, stageC{};
  auto cleanup_buffers = [&]() {
    gcore::rt::vk::destroy_buffer(dev, &stageC);
    gcore::rt::vk::destroy_buffer(dev, &stageB);
    gcore::rt::vk::destroy_buffer(dev, &stageA);
    gcore::rt::vk::destroy_buffer(dev, &bufC);
    gcore::rt::vk::destroy_buffer(dev, &bufB);
    gcore::rt::vk::destroy_buffer(dev, &bufA);
  };
  auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> hostA(M * K);
  std::vector<float> hostB(K * N);
  for (int i = 0; i < M * K; i++)
    hostA[i] = dist(rng);
  for (int i = 0; i < K * N; i++)
    hostB[i] = dist(rng);

  if (!gcore::rt::vk::create_device_local_buffer(phys, dev, bytesA, usage,
                                                 &bufA, &err) ||
      !gcore::rt::vk::create_device_local_buffer(phys, dev, bytesB, usage,
                                                 &bufB, &err) ||
      !gcore::rt::vk::create_device_local_buffer(phys, dev, bytesC, usage,
                                                 &bufC, &err) ||
      !gcore::rt::vk::create_staging_buffer(phys, dev, bytesA, &stageA, &err) ||
      !gcore::rt::vk::create_staging_buffer(phys, dev, bytesB, &stageB, &err) ||
      !gcore::rt::vk::create_staging_buffer(phys, dev, bytesC, &stageC, &err)) {
    std::cerr << "failed to prepare GEMM buffers: " << err << "\n";
    cleanup_buffers();
    return 9;
  }

  auto upload = [&](gcore::rt::vk::Buffer &stage,
                    const gcore::rt::vk::Buffer &device, const float *src,
                    VkDeviceSize bytes) {
    return gcore::rt::vk::stage_host_to_device(
        dev, pool, q, stage, device, bytes,
        [&](void *ptr, VkDeviceSize sz) {
          std::memcpy(ptr, src, static_cast<size_t>(sz));
        },
        &err);
  };
  auto zero_upload = [&](gcore::rt::vk::Buffer &stage,
                         const gcore::rt::vk::Buffer &device,
                         VkDeviceSize bytes) {
    return gcore::rt::vk::stage_host_to_device(
        dev, pool, q, stage, device, bytes,
        [](void *ptr, VkDeviceSize sz) {
          std::memset(ptr, 0, static_cast<size_t>(sz));
        },
        &err);
  };

  if (!upload(stageA, bufA, hostA.data(), bytesA) ||
      !upload(stageB, bufB, hostB.data(), bytesB) ||
      !zero_upload(stageC, bufC, bytesC)) {
    std::cerr << "upload failed: " << err << "\n";
    cleanup_buffers();
    return 10;
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

  // Command buffer
  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  r = vkAllocateCommandBuffers(dev, &cbai, &cmd);
  if (r != VK_SUCCESS) {
    std::cerr << "vkAllocateCommandBuffers failed: " << vk_err_str(r) << "\n";
    cleanup_buffers();
    return 13;
  }

  // Query pool (timestamps): 2 per dispatch (start/end) * batch
  VkQueryPoolCreateInfo qpci{};
  qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpci.queryCount = uint32_t(2 * batch);

  VkQueryPool qp = VK_NULL_HANDLE;
  r = vkCreateQueryPool(dev, &qpci, nullptr, &qp);
  if (r != VK_SUCCESS) {
    std::cerr << "vkCreateQueryPool failed: " << vk_err_str(r) << "\n";
    cleanup_buffers();
    return 14;
  }

  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

  const uint32_t lx = 16, ly = 16;
  const uint32_t gx = (uint32_t(N) + lx - 1) / lx;
  const uint32_t gy = (uint32_t(M) + ly - 1) / ly;

  struct Push {
    uint32_t M, N, K, lda, ldb, ldc;
  } pc{(uint32_t)M,   (uint32_t)N,   (uint32_t)K,
       (uint32_t)lda, (uint32_t)ldb, (uint32_t)ldc};

  // Collect per-iter GPU times (ms)
  std::vector<double> kernel_ms;
  kernel_ms.reserve((size_t)iters);

  // Also collect per-iter total "submit->fence" to quantify overhead (ms)
  std::vector<double> total_ms;
  total_ms.reserve((size_t)iters);

  bool validation_ok = true;
  bool timing_ok = true;

  for (int it = 0; it < iters; it++) {
    VkResult rr = vkResetCommandPool(dev, pool, 0);
    if (rr != VK_SUCCESS) {
      std::cerr << "vkResetCommandPool: " << vk_err_str(rr) << "\n";
      break;
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    rr = vkBeginCommandBuffer(cmd, &bi);
    if (rr != VK_SUCCESS) {
      std::cerr << "vkBeginCommandBuffer: " << vk_err_str(rr) << "\n";
      break;
    }

    vkCmdResetQueryPool(cmd, qp, 0, uint32_t(2 * batch));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds,
                            0, nullptr);
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push),
                       &pc);

    for (int j = 0; j < batch; j++) {
      const uint32_t q0 = uint32_t(2 * j);
      const uint32_t q1 = uint32_t(2 * j + 1);

      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qp, q0);
      vkCmdDispatch(cmd, gx, gy, 1);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qp, q1);
    }

    if (!compute_only) {
      // Make timestamps visible to host readback (query results are in memory;
      // pipeline barrier is fine)
      VkMemoryBarrier mb{};
      mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0,
                           nullptr);
    }

    rr = vkEndCommandBuffer(cmd);
    if (rr != VK_SUCCESS) {
      std::cerr << "vkEndCommandBuffer: " << vk_err_str(rr) << "\n";
      break;
    }

    VkFence fence = VK_NULL_HANDLE;
    rr = vkCreateFence(dev, &fci, nullptr, &fence);
    if (rr != VK_SUCCESS) {
      std::cerr << "vkCreateFence: " << vk_err_str(rr) << "\n";
      break;
    }

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    auto t0 = std::chrono::steady_clock::now();
    rr = vkQueueSubmit(q, 1, &si, fence);
    if (rr != VK_SUCCESS) {
      std::cerr << "vkQueueSubmit: " << vk_err_str(rr) << "\n";
      vkDestroyFence(dev, fence, nullptr);
      break;
    }

    rr = vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
    auto t1 = std::chrono::steady_clock::now();
    vkDestroyFence(dev, fence, nullptr);
    if (rr != VK_SUCCESS) {
      std::cerr << "vkWaitForFences: " << vk_err_str(rr) << "\n";
      break;
    }

    double tot = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_ms.push_back(tot);

    // Read timestamps
    std::vector<uint64_t> ts(2 * batch, 0);
    rr = vkGetQueryPoolResults(
        dev, qp, 0, uint32_t(2 * batch), ts.size() * sizeof(uint64_t),
        ts.data(), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (rr != VK_SUCCESS) {
      std::cerr << "vkGetQueryPoolResults: " << vk_err_str(rr) << "\n";
      break;
    }

    // Sum kernel time across batch
    double sum_ns = 0.0;
    for (int j = 0; j < batch; j++) {
      uint64_t a0 = ts[2 * j];
      uint64_t a1 = ts[2 * j + 1];
      if (a1 > a0) {
        double ticks = double(a1 - a0);
        sum_ns += ticks * timestampPeriodNs;
      }
    }
    kernel_ms.push_back(sum_ns / 1e6);
  }

  // Validation (8x8)
  if (compute_only) {
    VkResult rr = vkResetCommandPool(dev, pool, 0);
    if (rr != VK_SUCCESS) {
      std::cerr << "vkResetCommandPool (barrier): " << vk_err_str(rr) << "\n";
    } else {
      VkCommandBufferBeginInfo bi{};
      bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

      rr = vkBeginCommandBuffer(cmd, &bi);
      if (rr != VK_SUCCESS) {
        std::cerr << "vkBeginCommandBuffer (barrier): " << vk_err_str(rr)
                  << "\n";
      } else {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr,
                             0, nullptr);

        rr = vkEndCommandBuffer(cmd);
        if (rr != VK_SUCCESS) {
          std::cerr << "vkEndCommandBuffer (barrier): " << vk_err_str(rr)
                    << "\n";
        } else {
          VkSubmitInfo si{};
          si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
          si.commandBufferCount = 1;
          si.pCommandBuffers = &cmd;

          VkFence fence = VK_NULL_HANDLE;
          rr = vkCreateFence(dev, &fci, nullptr, &fence);
          if (rr != VK_SUCCESS) {
            std::cerr << "vkCreateFence (barrier): " << vk_err_str(rr) << "\n";
          } else {
            rr = vkQueueSubmit(q, 1, &si, fence);
            if (rr != VK_SUCCESS) {
              std::cerr << "vkQueueSubmit (barrier): " << vk_err_str(rr)
                        << "\n";
            } else {
              rr = vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
              if (rr != VK_SUCCESS) {
                std::cerr << "vkWaitForFences (barrier): " << vk_err_str(rr)
                          << "\n";
              }
            }
            vkDestroyFence(dev, fence, nullptr);
          }
        }
      }
    }
  }
  {
    VkResult rr = vkResetCommandPool(dev, pool, 0);
    if (rr != VK_SUCCESS) {
      std::cerr << "vkResetCommandPool (copyback): " << vk_err_str(rr) << "\n";
    } else {
      VkCommandBufferBeginInfo bi{};
      bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

      rr = vkBeginCommandBuffer(cmd, &bi);
      if (rr != VK_SUCCESS) {
        std::cerr << "vkBeginCommandBuffer (copyback): " << vk_err_str(rr)
                  << "\n";
      } else {
        VkBufferMemoryBarrier b0{};
        b0.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b0.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b0.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b0.buffer = bufC.buf;
        b0.offset = 0;
        b0.size = bytesC;
        b0.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b0.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                             &b0, 0, nullptr);

        VkBufferCopy c{};
        c.srcOffset = 0;
        c.dstOffset = 0;
        c.size = bytesC;
        vkCmdCopyBuffer(cmd, bufC.buf, stageC.buf, 1, &c);

        VkBufferMemoryBarrier b1{};
        b1.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b1.buffer = stageC.buf;
        b1.offset = 0;
        b1.size = bytesC;
        b1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b1.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &b1,
                             0, nullptr);

        rr = vkEndCommandBuffer(cmd);
        if (rr != VK_SUCCESS) {
          std::cerr << "vkEndCommandBuffer (copyback): " << vk_err_str(rr)
                    << "\n";
        } else {
          VkSubmitInfo si{};
          si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
          si.commandBufferCount = 1;
          si.pCommandBuffers = &cmd;

          VkFence fence = VK_NULL_HANDLE;
          rr = vkCreateFence(dev, &fci, nullptr, &fence);
          if (rr != VK_SUCCESS) {
            std::cerr << "vkCreateFence (copyback): " << vk_err_str(rr) << "\n";
          } else {
            rr = vkQueueSubmit(q, 1, &si, fence);
            if (rr != VK_SUCCESS) {
              std::cerr << "vkQueueSubmit (copyback): " << vk_err_str(rr)
                        << "\n";
            } else {
              rr = vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
              if (rr != VK_SUCCESS) {
                std::cerr << "vkWaitForFences (copyback): " << vk_err_str(rr)
                          << "\n";
              }
            }
            vkDestroyFence(dev, fence, nullptr);
          }
        }
      }
    }
  }

  std::vector<float> Cref(M * N, 0.f);
  cpu_ref_8x8(hostA.data(), hostB.data(), Cref.data(), M, N, K, lda, ldb,
              ldc);
  std::vector<float> Cout(M * N, 0.f);
  if (!gcore::rt::vk::read_device_to_host(
          dev, pool, q, bufC, stageC, bytesC,
          [&](const void *ptr, VkDeviceSize sz) {
            std::memcpy(Cout.data(), ptr, static_cast<size_t>(sz));
          },
          &err)) {
    std::cerr << "read_device_to_host failed: " << err << "\n";
    validation_ok = false;
  } else {
    bool ok = true;
    int rm = std::min(M, 8), cn = std::min(N, 8);
    double max_abs = 0.0;
    for (int r0 = 0; r0 < rm; r0++) {
      for (int c0 = 0; c0 < cn; c0++) {
        float got = Cout[r0 * ldc + c0];
        float exp = Cref[r0 * ldc + c0];
        double absd = std::abs(double(got) - double(exp));
        max_abs = std::max(max_abs, absd);
        if (absd > 1e-2)
          ok = false;
      }
    }

    std::cout << "VALIDATION(8x8): " << (ok ? "OK" : "FAILED")
              << "  max_abs_err=" << max_abs << "\n";
    if (!ok)
      validation_ok = false;
  }

  auto stats = [](std::vector<double> v, const char *name) {
    if (v.empty())
      return;
    std::sort(v.begin(), v.end());
    double mean = 0;
    for (auto x : v)
      mean += x;
    mean /= v.size();
    double p50 = v[v.size() / 2];
    double p99 =
        v[(size_t)std::min<size_t>(v.size() - 1, (v.size() * 99) / 100)];
    std::cout << std::fixed << std::setprecision(3);
    std::cout << name << ":\n";
    std::cout << "  samples=" << v.size() << "\n";
    std::cout << "  mean_ms=" << mean << "\n";
    std::cout << "  p50_ms=" << p50 << "\n";
    std::cout << "  p99_ms=" << p99 << "\n";
  };

  if (!kernel_ms.empty()) {
    // Kernel TFLOPs computed from *kernel_ms* (GPU-only) and per-dispatch flops
    // * batch
    double flops_per_dispatch = 2.0 * double(M) * double(N) * double(K);
    // Convert kernel_ms for each iter into TFLOPs for that iter
    std::vector<double> tflops;
    tflops.reserve(kernel_ms.size());
    for (double km : kernel_ms) {
      double secs = (km / 1e3);
      double flops_total = flops_per_dispatch * double(batch);
      tflops.push_back((flops_total / secs) / 1e12);
    }

    std::vector<double> km_sorted = kernel_ms;
    std::sort(km_sorted.begin(), km_sorted.end());
    std::vector<double> tf_sorted = tflops;
    std::sort(tf_sorted.begin(), tf_sorted.end());

    auto mean_of = [](const std::vector<double> &v) {
      double m = 0;
      for (auto x : v)
        m += x;
      return m / std::max<size_t>(1, v.size());
    };

    double km_mean = mean_of(kernel_ms);
    double tf_mean = mean_of(tflops);
    double km_p50 = km_sorted[km_sorted.size() / 2];
    double km_p99 = km_sorted[(size_t)std::min<size_t>(
        km_sorted.size() - 1, (km_sorted.size() * 99) / 100)];
    double tf_p50 = tf_sorted[tf_sorted.size() / 2];
    double tf_p99 = tf_sorted[(size_t)std::min<size_t>(
        tf_sorted.size() - 1, (tf_sorted.size() * 99) / 100)];

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "RESULT vk_gemm_tiled_ts_bench (GPU-only kernel):\n";
    std::cout << "  iters=" << kernel_ms.size() << " batch=" << batch << "\n";
    std::cout << "  kernel_mean_ms=" << km_mean << "  kernel_p50_ms=" << km_p50
              << "  kernel_p99_ms=" << km_p99 << "\n";
    std::cout << "  mean_TFLOPs=" << tf_mean << "  p50_TFLOPs=" << tf_p50
              << "  p99_TFLOPs=" << tf_p99 << "\n";
  } else {
    std::cerr << "No kernel timestamp samples collected.\n";
    timing_ok = false;
  }

  if (!total_ms.empty()) {
    std::cout << "RESULT submit_wait_overhead (host visible):\n";
    stats(total_ms, "submit_wait_ms");
  }
  if (!timing_ok) {
    std::cout << "STATUS=FAILED reason=\"no_timing\"\n";
  } else if (!validation_ok) {
    std::cout << "STATUS=FAILED reason=\"validation_failed\"\n";
  } else {
    std::cout << "STATUS=OK\n";
  }

  cleanup_buffers();

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
