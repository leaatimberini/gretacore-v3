#include "gcore/rt/vk/backend.hpp"
#include "gcore/rt/vk/buffer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring> // <-- FIX: for std::memcpy
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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

int main(int argc, char **argv) {
  const int size_mb = argi(argc, argv, "--size-mb", 64);
  const int iters = argi(argc, argv, "--iters", 50);
  const uint32_t fill_value = 0xA5A5A5A5u;

  std::cout << "GRETA CORE Runtime Bench: vk_fill_bench\n";
  std::cout << "size_mb=" << size_mb << " iters=" << iters << "\n";

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

  // Load SPIR-V generated into build dir:
  // build/fill.comp.spv
  std::filesystem::path spv_path =
      std::filesystem::current_path() / "build" / "fill.comp.spv";
  auto spv = read_spv_u32(spv_path);
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

  VkDescriptorSetLayoutBinding bind0{};
  bind0.binding = 0;
  bind0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bind0.descriptorCount = 1;
  bind0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = 1;
  dslci.pBindings = &bind0;

  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  r = vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);
  if (r != VK_SUCCESS) {
    std::cerr << "vkCreateDescriptorSetLayout failed: " << vk_err_str(r)
              << "\n";
    vkDestroyShaderModule(dev, shader, nullptr);
    std::cout << "STATUS=FAILED reason=\"dsl_failed\"\n";
    return 4;
  }

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = 8;

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
  ps.descriptorCount = 1;

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

  const VkDeviceSize bytes =
      static_cast<VkDeviceSize>(size_mb) * 1024ull * 1024ull;
  const uint32_t n_u32 = static_cast<uint32_t>(bytes / 4ull);

  gcore::rt::vk::Buffer out_dev{};
  gcore::rt::vk::Buffer staging{};
  if (!gcore::rt::vk::create_device_local_buffer(
          phys, dev, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out_dev,
          &err)) {
    std::cerr << "create_device_local_buffer failed: " << err << "\n";
    vkDestroyDescriptorPool(dev, dp, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyShaderModule(dev, shader, nullptr);
    return 9;
  }
  if (!gcore::rt::vk::create_staging_buffer(phys, dev, bytes, &staging, &err)) {
    std::cerr << "create_staging_buffer failed: " << err << "\n";
    gcore::rt::vk::destroy_buffer(dev, &out_dev);
    vkDestroyDescriptorPool(dev, dp, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyShaderModule(dev, shader, nullptr);
    return 9;
  }

  VkDescriptorBufferInfo dbi{};
  dbi.buffer = out_dev.buf;
  dbi.offset = 0;
  dbi.range = bytes;

  VkWriteDescriptorSet w{};
  w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  w.dstSet = ds;
  w.dstBinding = 0;
  w.descriptorCount = 1;
  w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  w.pBufferInfo = &dbi;

  vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);

  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  r = vkAllocateCommandBuffers(dev, &cbai, &cmd);
  if (r != VK_SUCCESS) {
    std::cerr << "vkAllocateCommandBuffers failed: " << vk_err_str(r) << "\n";
    gcore::rt::vk::destroy_buffer(dev, &out_dev);
    vkDestroyDescriptorPool(dev, dp, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyShaderModule(dev, shader, nullptr);
    return 10;
  }

  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

  std::vector<double> secs;
  bool validation_ok = true;
  secs.reserve(static_cast<size_t>(iters));

  const uint32_t wg = 256;
  const uint32_t groups = (n_u32 + wg - 1) / wg;

  for (int it = 0; it < iters; it++) {
    VkResult rr = vkResetCommandPool(dev, pool, 0);
    if (rr != VK_SUCCESS) {
      std::cerr << "vkResetCommandPool failed: " << vk_err_str(rr) << "\n";
      break;
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    rr = vkBeginCommandBuffer(cmd, &bi);
    if (rr != VK_SUCCESS) {
      std::cerr << "vkBeginCommandBuffer failed: " << vk_err_str(rr) << "\n";
      break;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds,
                            0, nullptr);

    struct Push {
      uint32_t value;
      uint32_t n;
    } pc{fill_value, n_u32};
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push),
                       &pc);

    vkCmdDispatch(cmd, groups, 1, 1);

    VkBufferMemoryBarrier b0{};
    b0.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b0.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.buffer = out_dev.buf;
    b0.offset = 0;
    b0.size = bytes;
    b0.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b0.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &b0,
                         0, nullptr);

    VkBufferCopy c{};
    c.srcOffset = 0;
    c.dstOffset = 0;
    c.size = bytes;
    vkCmdCopyBuffer(cmd, out_dev.buf, staging.buf, 1, &c);

    VkBufferMemoryBarrier b1{};
    b1.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.buffer = staging.buf;
    b1.offset = 0;
    b1.size = bytes;
    b1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b1.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &b1, 0,
                         nullptr);

    rr = vkEndCommandBuffer(cmd);
    if (rr != VK_SUCCESS) {
      std::cerr << "vkEndCommandBuffer failed: " << vk_err_str(rr) << "\n";
      break;
    }

    VkFence fence = VK_NULL_HANDLE;
    rr = vkCreateFence(dev, &fci, nullptr, &fence);
    if (rr != VK_SUCCESS) {
      std::cerr << "vkCreateFence failed: " << vk_err_str(rr) << "\n";
      break;
    }

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    auto t0 = std::chrono::steady_clock::now();
    rr = vkQueueSubmit(q, 1, &si, fence);
    if (rr != VK_SUCCESS) {
      std::cerr << "vkQueueSubmit failed: " << vk_err_str(rr) << "\n";
      vkDestroyFence(dev, fence, nullptr);
      break;
    }

    rr = vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
    auto t1 = std::chrono::steady_clock::now();
    vkDestroyFence(dev, fence, nullptr);

    if (rr != VK_SUCCESS) {
      std::cerr << "vkWaitForFences failed: " << vk_err_str(rr) << "\n";
      break;
    }

    std::chrono::duration<double> dt = t1 - t0;
    secs.push_back(dt.count());
  }

  void *p = nullptr;
  if (!gcore::rt::vk::map_buffer(dev, staging, &p, &err)) {
    std::cerr << "map_buffer failed: " << err << "\n";
    validation_ok = false;
  } else {
    uint32_t *u = reinterpret_cast<uint32_t *>(p);
    bool ok = true;
    for (int i = 0; i < 16 && i < (int)n_u32; i++) {
      if (u[i] != fill_value) {
        ok = false;
        break;
      }
    }
    gcore::rt::vk::unmap_buffer(dev, staging);
    std::cout << "VALIDATION: " << (ok ? "OK" : "FAILED") << "\n";
    if (!ok)
      validation_ok = false;
  }

  if (secs.empty()) {
    std::cerr << "No timing samples collected.\n";
    std::cout << "STATUS=FAILED reason=\"no_timing\"\n";
    gcore::rt::vk::destroy_buffer(dev, &staging);
    gcore::rt::vk::destroy_buffer(dev, &out_dev);
    vkDestroyDescriptorPool(dev, dp, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyShaderModule(dev, shader, nullptr);
    return 11;
  }

  std::sort(secs.begin(), secs.end());
  double mean = 0.0;
  for (double s : secs)
    mean += s;
  mean /= secs.size();

  double p50 = secs[secs.size() / 2];
  double p99 = secs[static_cast<size_t>(
      std::min<size_t>(secs.size() - 1, (secs.size() * 99) / 100))];

  auto gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "RESULT vk_fill_bench:\n";
  std::cout << "  samples=" << secs.size() << "\n";
  std::cout << "  mean_ms=" << (mean * 1e3) << "  mean_GiBps=" << (gib / mean)
            << "\n";
  std::cout << "  p50_ms=" << (p50 * 1e3) << "   p50_GiBps=" << (gib / p50)
            << "\n";
  std::cout << "  p99_ms=" << (p99 * 1e3) << "   p99_GiBps=" << (gib / p99)
            << "\n";
  if (!validation_ok) {
    std::cout << "STATUS=FAILED reason=\"validation_failed\"\n";
  } else {
    std::cout << "STATUS=OK\n";
  }

  gcore::rt::vk::destroy_buffer(dev, &staging);
  gcore::rt::vk::destroy_buffer(dev, &out_dev);
  vkDestroyDescriptorPool(dev, dp, nullptr);
  vkDestroyPipeline(dev, pipe, nullptr);
  vkDestroyPipelineLayout(dev, pl, nullptr);
  vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
  vkDestroyShaderModule(dev, shader, nullptr);

  return validation_ok ? 0 : 12;
}
