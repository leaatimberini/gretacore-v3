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

static double argd(int argc, char **argv, const char *key, double def) {
  for (int i = 1; i + 1 < argc; i++) {
    if (std::string(argv[i]) == key)
      return std::stod(argv[i + 1]);
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

static void layernorm_ref(const float *x, float *y, const float *gamma,
                          const float *beta, int rows, int cols, double eps) {
  for (int r = 0; r < rows; r++) {
    const float *xr = x + size_t(r) * size_t(cols);
    float *yr = y + size_t(r) * size_t(cols);
    double mean = 0.0;
    double ms = 0.0;
    for (int c = 0; c < cols; c++) {
      double v = double(xr[c]);
      mean += v;
      ms += v * v;
    }
    mean /= double(cols);
    ms /= double(cols);
    double var = ms - mean * mean;
    if (var < 0.0)
      var = 0.0;
    double inv = 1.0 / std::sqrt(var + eps);
    for (int c = 0; c < cols; c++) {
      double v = (double(xr[c]) - mean) * inv;
      yr[c] = float(v * double(gamma[c]) + double(beta[c]));
    }
  }
}

static void rmsnorm_ref(const float *x, float *y, const float *gamma, int rows,
                        int cols, double eps) {
  for (int r = 0; r < rows; r++) {
    const float *xr = x + size_t(r) * size_t(cols);
    float *yr = y + size_t(r) * size_t(cols);
    double ms = 0.0;
    for (int c = 0; c < cols; c++) {
      double v = double(xr[c]);
      ms += v * v;
    }
    ms /= double(cols);
    double inv = 1.0 / std::sqrt(ms + eps);
    for (int c = 0; c < cols; c++)
      yr[c] = float(double(xr[c]) * inv) * gamma[c];
  }
}

int main(int argc, char **argv) {
  const int rows = argi(argc, argv, "--rows", 256);
  const int cols = argi(argc, argv, "--cols", 1024);
  const int iters = std::max(1, argi(argc, argv, "--iters", 20));
  const double eps = argd(argc, argv, "--eps", 1e-5);

  std::cout << "GRETA CORE Runtime Bench: vk_layernorm_rmsnorm_fused_tiled_bench\n";
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
      std::filesystem::current_path() / "build" /
      "layernorm_rmsnorm_fused_tiled.comp.spv";
  auto spv = read_spv_u32(spv_path);
  if (spv.empty()) {
    std::filesystem::path alt =
        std::filesystem::current_path() / "tools" / "bench" / "runtime" /
        "build" / "layernorm_rmsnorm_fused_tiled.comp.spv";
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

  VkDescriptorSetLayoutBinding binds[6]{};
  for (int i = 0; i < 6; i++) {
    binds[i].binding = uint32_t(i);
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = 6;
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
    float eps;
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
  ps.descriptorCount = 6;

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
  const VkDeviceSize bytes_g = VkDeviceSize(cols) * 4;

  gcore::rt::vk::Buffer x_dev{}, gl_dev{}, bl_dev{}, gr_dev{}, yln_dev{},
      yrm_dev{};
  gcore::rt::vk::Buffer x_stage{}, gl_stage{}, bl_stage{}, gr_stage{},
      yln_stage{}, yrm_stage{};
  auto cleanup = [&]() {
    gcore::rt::vk::destroy_buffer(dev, &yrm_stage);
    gcore::rt::vk::destroy_buffer(dev, &yln_stage);
    gcore::rt::vk::destroy_buffer(dev, &gr_stage);
    gcore::rt::vk::destroy_buffer(dev, &bl_stage);
    gcore::rt::vk::destroy_buffer(dev, &gl_stage);
    gcore::rt::vk::destroy_buffer(dev, &x_stage);
    gcore::rt::vk::destroy_buffer(dev, &yrm_dev);
    gcore::rt::vk::destroy_buffer(dev, &yln_dev);
    gcore::rt::vk::destroy_buffer(dev, &gr_dev);
    gcore::rt::vk::destroy_buffer(dev, &bl_dev);
    gcore::rt::vk::destroy_buffer(dev, &gl_dev);
    gcore::rt::vk::destroy_buffer(dev, &x_dev);
  };

  if (!gcore::rt::vk::create_device_local_buffer(
          phys, dev, bytes_x, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_dev,
          &err) ||
      !gcore::rt::vk::create_device_local_buffer(
          phys, dev, bytes_g, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &gl_dev,
          &err) ||
      !gcore::rt::vk::create_device_local_buffer(
          phys, dev, bytes_g, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bl_dev,
          &err) ||
      !gcore::rt::vk::create_device_local_buffer(
          phys, dev, bytes_g, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &gr_dev,
          &err) ||
      !gcore::rt::vk::create_device_local_buffer(
          phys, dev, bytes_x, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &yln_dev,
          &err) ||
      !gcore::rt::vk::create_device_local_buffer(
          phys, dev, bytes_x, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &yrm_dev,
          &err) ||
      !gcore::rt::vk::create_staging_buffer(phys, dev, bytes_x, &x_stage,
                                            &err) ||
      !gcore::rt::vk::create_staging_buffer(phys, dev, bytes_g, &gl_stage,
                                            &err) ||
      !gcore::rt::vk::create_staging_buffer(phys, dev, bytes_g, &bl_stage,
                                            &err) ||
      !gcore::rt::vk::create_staging_buffer(phys, dev, bytes_g, &gr_stage,
                                            &err) ||
      !gcore::rt::vk::create_staging_buffer(phys, dev, bytes_x, &yln_stage,
                                            &err) ||
      !gcore::rt::vk::create_staging_buffer(phys, dev, bytes_x, &yrm_stage,
                                            &err)) {
    std::cerr << "buffer alloc failed: " << err << "\n";
    cleanup();
    return 9;
  }

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  std::vector<float> x_host(rows * cols);
  std::vector<float> gl_host(cols, 1.0f);
  std::vector<float> bl_host(cols, 0.0f);
  std::vector<float> gr_host(cols, 1.0f);

  auto upload_x = [&](void *ptr, VkDeviceSize sz) {
    auto *dst = reinterpret_cast<float *>(ptr);
    size_t n = size_t(sz / 4);
    for (size_t i = 0; i < n; i++) {
      float v = dist(rng);
      dst[i] = v;
      x_host[i] = v;
    }
  };
  auto upload_gl = [&](void *ptr, VkDeviceSize sz) {
    auto *dst = reinterpret_cast<float *>(ptr);
    size_t n = size_t(sz / 4);
    for (size_t i = 0; i < n; i++)
      dst[i] = gl_host[i];
  };
  auto upload_bl = [&](void *ptr, VkDeviceSize sz) {
    auto *dst = reinterpret_cast<float *>(ptr);
    size_t n = size_t(sz / 4);
    for (size_t i = 0; i < n; i++)
      dst[i] = bl_host[i];
  };
  auto upload_gr = [&](void *ptr, VkDeviceSize sz) {
    auto *dst = reinterpret_cast<float *>(ptr);
    size_t n = size_t(sz / 4);
    for (size_t i = 0; i < n; i++)
      dst[i] = gr_host[i];
  };

  if (!gcore::rt::vk::stage_host_to_device(dev, pool, q, x_stage, x_dev, bytes_x,
                                           upload_x, &err) ||
      !gcore::rt::vk::stage_host_to_device(dev, pool, q, gl_stage, gl_dev,
                                           bytes_g, upload_gl, &err) ||
      !gcore::rt::vk::stage_host_to_device(dev, pool, q, bl_stage, bl_dev,
                                           bytes_g, upload_bl, &err) ||
      !gcore::rt::vk::stage_host_to_device(dev, pool, q, gr_stage, gr_dev,
                                           bytes_g, upload_gr, &err)) {
    std::cerr << "upload failed: " << err << "\n";
    cleanup();
    return 10;
  }

  VkDescriptorBufferInfo dbi[6]{};
  dbi[0] = {x_dev.buf, 0, bytes_x};
  dbi[1] = {gl_dev.buf, 0, bytes_g};
  dbi[2] = {bl_dev.buf, 0, bytes_g};
  dbi[3] = {gr_dev.buf, 0, bytes_g};
  dbi[4] = {yln_dev.buf, 0, bytes_x};
  dbi[5] = {yrm_dev.buf, 0, bytes_x};

  VkWriteDescriptorSet wr[6]{};
  for (int i = 0; i < 6; i++) {
    wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr[i].dstSet = ds;
    wr[i].dstBinding = uint32_t(i);
    wr[i].descriptorCount = 1;
    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[i].pBufferInfo = &dbi[i];
  }
  vkUpdateDescriptorSets(dev, 6, wr, 0, nullptr);

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

  Push pc{uint32_t(rows), uint32_t(cols), float(eps)};
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
    std::cout << "RESULT vk_layernorm_rmsnorm_fused_tiled_bench: mean_ms="
              << mean << " p50_ms=" << p50 << " p99_ms=" << p99 << "\n";
  } else {
    std::cerr << "No timing samples collected.\n";
  }

  std::vector<float> yln(rows * cols);
  std::vector<float> yrm(rows * cols);
  if (!gcore::rt::vk::read_device_to_host(
          dev, pool, q, yln_dev, yln_stage, bytes_x,
          [&](const void *ptr, VkDeviceSize sz) {
            std::memcpy(yln.data(), ptr, static_cast<size_t>(sz));
          },
          &err)) {
    std::cerr << "read_device_to_host(y_ln) failed: " << err << "\n";
    timing_ok = false;
  }
  if (!gcore::rt::vk::read_device_to_host(
          dev, pool, q, yrm_dev, yrm_stage, bytes_x,
          [&](const void *ptr, VkDeviceSize sz) {
            std::memcpy(yrm.data(), ptr, static_cast<size_t>(sz));
          },
          &err)) {
    std::cerr << "read_device_to_host(y_rms) failed: " << err << "\n";
    timing_ok = false;
  }

  std::vector<float> ref_ln(rows * cols);
  std::vector<float> ref_rms(rows * cols);
  layernorm_ref(x_host.data(), ref_ln.data(), gl_host.data(), bl_host.data(),
                rows, cols, eps);
  rmsnorm_ref(x_host.data(), ref_rms.data(), gr_host.data(), rows, cols, eps);

  double max_abs_ln = 0.0;
  double max_abs_rms = 0.0;
  for (size_t i = 0; i < ref_ln.size(); i++) {
    max_abs_ln =
        std::max(max_abs_ln, std::abs(double(ref_ln[i]) - double(yln[i])));
    max_abs_rms =
        std::max(max_abs_rms, std::abs(double(ref_rms[i]) - double(yrm[i])));
  }

  bool validation_ok = (max_abs_ln < 5e-3) && (max_abs_rms < 5e-3);
  std::cout << "VALIDATION(layernorm_rmsnorm_fused_tiled): "
            << (validation_ok ? "OK" : "FAILED")
            << " max_abs_err_ln=" << max_abs_ln
            << " max_abs_err_rms=" << max_abs_rms << "\n";

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
