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

static void softmax_ref(const float *x, float *y, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    const float *xr = x + size_t(r) * size_t(cols);
    float *yr = y + size_t(r) * size_t(cols);
    double maxv = double(xr[0]);
    for (int c = 1; c < cols; c++)
      maxv = std::max(maxv, double(xr[c]));
    double sum = 0.0;
    for (int c = 0; c < cols; c++) {
      double e = std::exp(double(xr[c]) - maxv);
      yr[c] = float(e);
      sum += e;
    }
    double inv = 1.0 / sum;
    for (int c = 0; c < cols; c++)
      yr[c] = float(double(yr[c]) * inv);
  }
}

int main(int argc, char **argv) {
  const int rows = argi(argc, argv, "--rows", 256);
  const int cols = argi(argc, argv, "--cols", 1024);
  const int iters = std::max(1, argi(argc, argv, "--iters", 20));

  std::cout << "GRETA CORE Runtime Bench: vk_softmax_bench\n";
  std::cout << "rows=" << rows << " cols=" << cols << " iters=" << iters
            << "\n";

  gcore::rt::vk::Backend b;
  std::string err;
  if (!b.init(&err)) {
    std::cerr << "INIT FAILED: " << err << "\n";
    std::cout << "STATUS=FAILED reason=init_failed\n";
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
    std::cout << "STATUS=SKIPPED reason=gpu_blacklisted\n";
    return 0;
  }

  VkDevice dev = b.device();
  VkPhysicalDevice phys = b.physical_device();
  VkQueue q = b.queue();
  VkCommandPool pool = b.command_pool();

  std::filesystem::path spv_path =
      std::filesystem::current_path() / "build" / "softmax.comp.spv";
  auto spv = read_spv_u32(spv_path);
  if (spv.empty()) {
    std::filesystem::path alt =
        std::filesystem::current_path() / "tools" / "bench" / "runtime" /
        "build" / "softmax.comp.spv";
    spv = read_spv_u32(alt);
    if (!spv.empty())
      spv_path = alt;
  }
  if (spv.empty()) {
    std::cerr << "Failed to read SPIR-V: " << spv_path.string() << "\n";
    std::cout << "STATUS=FAILED reason=shader_read_failed\n";
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
    std::cout << "STATUS=FAILED reason=shader_module_failed\n";
    return 3;
  }

  VkDescriptorSetLayoutBinding binds[2]{};
  for (int i = 0; i < 2; i++) {
    binds[i].binding = uint32_t(i);
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = 2;
  dslci.pBindings = binds;

  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  r = vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);
  if (r != VK_SUCCESS) {
    std::cerr << "vkCreateDescriptorSetLayout failed: " << vk_err_str(r)
              << "\n";
    vkDestroyShaderModule(dev, shader, nullptr);
    return 4;
  }

  struct Push {
    uint32_t rows;
    uint32_t cols;
  };

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = sizeof(Push);

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
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyShaderModule(dev, shader, nullptr);
    return 6;
  }

  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 2;

  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;

  VkDescriptorPool dp = VK_NULL_HANDLE;
  r = vkCreateDescriptorPool(dev, &dpci, nullptr, &dp);
  if (r != VK_SUCCESS) {
    std::cerr << "vkCreateDescriptorPool failed: " << vk_err_str(r) << "\n";
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyShaderModule(dev, shader, nullptr);
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
    vkDestroyDescriptorPool(dev, dp, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyShaderModule(dev, shader, nullptr);
    return 8;
  }

  const VkDeviceSize bytes_x = VkDeviceSize(rows) * VkDeviceSize(cols) * 4;

  gcore::rt::vk::Buffer x_dev{}, y_dev{};
  gcore::rt::vk::Buffer x_stage{}, y_stage{};
  auto cleanup = [&]() {
    gcore::rt::vk::destroy_buffer(dev, &y_stage);
    gcore::rt::vk::destroy_buffer(dev, &x_stage);
    gcore::rt::vk::destroy_buffer(dev, &y_dev);
    gcore::rt::vk::destroy_buffer(dev, &x_dev);
  };

  if (!gcore::rt::vk::create_device_local_buffer(
          phys, dev, bytes_x, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_dev,
          &err) ||
      !gcore::rt::vk::create_device_local_buffer(
          phys, dev, bytes_x, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_dev,
          &err) ||
      !gcore::rt::vk::create_staging_buffer(phys, dev, bytes_x, &x_stage,
                                            &err) ||
      !gcore::rt::vk::create_staging_buffer(phys, dev, bytes_x, &y_stage,
                                            &err)) {
    std::cerr << "buffer alloc failed: " << err << "\n";
    cleanup();
    return 9;
  }

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  auto upload_x = [&](void *ptr, VkDeviceSize sz) {
    auto *dst = reinterpret_cast<float *>(ptr);
    size_t n = size_t(sz / 4);
    for (size_t i = 0; i < n; i++)
      dst[i] = dist(rng);
  };

  if (!gcore::rt::vk::stage_host_to_device(dev, pool, q, x_stage, x_dev, bytes_x,
                                           upload_x, &err)) {
    std::cerr << "upload failed: " << err << "\n";
    cleanup();
    return 10;
  }

  VkDescriptorBufferInfo dbi[2]{};
  dbi[0] = {x_dev.buf, 0, bytes_x};
  dbi[1] = {y_dev.buf, 0, bytes_x};

  VkWriteDescriptorSet wr[2]{};
  for (int i = 0; i < 2; i++) {
    wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr[i].dstSet = ds;
    wr[i].dstBinding = uint32_t(i);
    wr[i].descriptorCount = 1;
    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[i].pBufferInfo = &dbi[i];
  }
  vkUpdateDescriptorSets(dev, 2, wr, 0, nullptr);

  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  r = vkAllocateCommandBuffers(dev, &cbai, &cmd);
  if (r != VK_SUCCESS) {
    std::cerr << "vkAllocateCommandBuffers failed: " << vk_err_str(r) << "\n";
    cleanup();
    return 11;
  }

  VkCommandBufferBeginInfo cbbi{};
  cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  r = vkBeginCommandBuffer(cmd, &cbbi);
  if (r != VK_SUCCESS) {
    std::cerr << "vkBeginCommandBuffer failed: " << vk_err_str(r) << "\n";
    cleanup();
    return 12;
  }

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0,
                          nullptr);

  Push pc{uint32_t(rows), uint32_t(cols)};
  vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &pc);
  vkCmdDispatch(cmd, uint32_t(rows), 1, 1);

  r = vkEndCommandBuffer(cmd);
  if (r != VK_SUCCESS) {
    std::cerr << "vkEndCommandBuffer failed: " << vk_err_str(r) << "\n";
    cleanup();
    return 13;
  }

  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; i++) {
    VkFence fence = VK_NULL_HANDLE;
    r = vkCreateFence(dev, &fci, nullptr, &fence);
    if (r != VK_SUCCESS)
      break;

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    auto t0 = std::chrono::high_resolution_clock::now();
    r = vkQueueSubmit(q, 1, &si, fence);
    if (r != VK_SUCCESS) {
      vkDestroyFence(dev, fence, nullptr);
      break;
    }
    r = vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
    auto t1 = std::chrono::high_resolution_clock::now();
    vkDestroyFence(dev, fence, nullptr);
    if (r != VK_SUCCESS)
      break;

    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    samples.push_back(ms);
  }

  bool timing_ok = !samples.empty();
  if (timing_ok) {
    std::vector<double> tmp = samples;
    std::sort(tmp.begin(), tmp.end());
    double mean = 0.0;
    for (double v : samples)
      mean += v;
    mean /= samples.size();
    double p50 = tmp[tmp.size() / 2];
    size_t p99_idx = (tmp.size() * 99) / 100;
    if (p99_idx >= tmp.size())
      p99_idx = tmp.size() - 1;
    double p99 = tmp[p99_idx];
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "RESULT vk_softmax_bench: mean_ms=" << mean
              << " p50_ms=" << p50 << " p99_ms=" << p99 << "\n";
  } else {
    std::cerr << "No timing samples collected.\n";
  }

  std::vector<float> y(rows * cols);
  if (!gcore::rt::vk::read_device_to_host(
          dev, pool, q, y_dev, y_stage, bytes_x,
          [&](const void *ptr, VkDeviceSize sz) {
            std::memcpy(y.data(), ptr, static_cast<size_t>(sz));
          },
          &err)) {
    std::cerr << "read_device_to_host failed: " << err << "\n";
    timing_ok = false;
  }

  std::vector<float> x_host(rows * cols);
  if (!gcore::rt::vk::read_device_to_host(
          dev, pool, q, x_dev, x_stage, bytes_x,
          [&](const void *ptr, VkDeviceSize sz) {
            std::memcpy(x_host.data(), ptr, static_cast<size_t>(sz));
          },
          &err)) {
    std::cerr << "read_device_to_host(x) failed: " << err << "\n";
    timing_ok = false;
  }

  std::vector<float> ref(rows * cols);
  softmax_ref(x_host.data(), ref.data(), rows, cols);

  double max_abs_sum = 0.0;
  bool valid = true;
  for (int r = 0; r < rows; r++) {
    double sum = 0.0;
    for (int c = 0; c < cols; c++) {
      float v = y[size_t(r) * size_t(cols) + size_t(c)];
      if (v < 0.0f || v > 1.0f)
        valid = false;
      sum += double(v);
    }
    max_abs_sum = std::max(max_abs_sum, std::abs(sum - 1.0));
  }

  double max_abs = 0.0;
  for (size_t i = 0; i < ref.size(); i++)
    max_abs = std::max(max_abs, std::abs(double(ref[i]) - double(y[i])));

  bool validation_ok = valid && max_abs_sum < 1e-4 && max_abs < 5e-3;
  std::cout << "VALIDATION(softmax): " << (validation_ok ? "OK" : "FAILED")
            << " max_abs_sum=" << max_abs_sum << " max_abs_err=" << max_abs
            << "\n";

  cleanup();
  vkDestroyDescriptorPool(dev, dp, nullptr);
  vkDestroyPipeline(dev, pipe, nullptr);
  vkDestroyPipelineLayout(dev, pl, nullptr);
  vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
  vkDestroyShaderModule(dev, shader, nullptr);

  if (!timing_ok) {
    std::cout << "STATUS=FAILED reason=no_timing\n";
    return 1;
  }
  if (!validation_ok) {
    std::cout << "STATUS=FAILED reason=validation_failed\n";
    return 2;
  }
  std::cout << "STATUS=OK\n";
  return 0;
}
