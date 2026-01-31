#include "gemm_f32_runtime.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace gcore::rt::vk {

static uint32_t ceil_div_u32(uint32_t x, uint32_t d) {
  return (x + d - 1u) / d;
}

const char *gemm_f32_kernel_name(GemmF32KernelId id) {
  switch (id) {
  case GemmF32KernelId::tiled:
    return "tiled";
  }
  return "unknown";
}

const char *gemm_f32_kernel_spv_filename(GemmF32KernelId id) {
  switch (id) {
  case GemmF32KernelId::tiled:
    return "gemm_f32_tiled.comp.spv";
  }
  return "";
}

std::vector<uint32_t>
GemmF32PipelineCache::read_spv_u32(const std::string &filename,
                                  std::string *err) const {
  std::filesystem::path p = std::filesystem::path(shader_dir_) / filename;
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    if (err) {
      std::ostringstream oss;
      oss << "Failed to open SPIR-V: " << p.string()
          << " (set shader_dir o asegurá build/*.spv)";
      *err = oss.str();
    }
    return {};
  }
  std::vector<char> bytes((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
  if (bytes.empty() || (bytes.size() % 4) != 0) {
    if (err) {
      std::ostringstream oss;
      oss << "Invalid SPIR-V (size % 4 != 0): " << p.string();
      *err = oss.str();
    }
    return {};
  }
  std::vector<uint32_t> out(bytes.size() / 4);
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

VkPipeline GemmF32PipelineCache::pipe(GemmF32KernelId kid) const {
  auto it = pipes_.find((uint32_t)kid);
  if (it == pipes_.end())
    return VK_NULL_HANDLE;
  return it->second.pipe;
}

bool GemmF32PipelineCache::ensure_common_layouts(std::string *err) {
  if (dsl_ != VK_NULL_HANDLE && pl_ != VK_NULL_HANDLE)
    return true;

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

  VkResult r = vkCreateDescriptorSetLayout(dev_, &dslci, nullptr, &dsl_);
  if (r != VK_SUCCESS) {
    if (err)
      *err = "vkCreateDescriptorSetLayout failed";
    return false;
  }

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = sizeof(GemmF32PushConstants);

  VkPipelineLayoutCreateInfo plci{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl_;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;

  r = vkCreatePipelineLayout(dev_, &plci, nullptr, &pl_);
  if (r != VK_SUCCESS) {
    if (err)
      *err = "vkCreatePipelineLayout failed";
    return false;
  }

  return true;
}

bool GemmF32PipelineCache::create_entry(VkPhysicalDevice /*phys*/,
                                        GemmF32KernelId kid, Entry &e,
                                        std::string *err) {
  const char *spv_name = gemm_f32_kernel_spv_filename(kid);
  auto spv = read_spv_u32(spv_name, err);
  if (spv.empty())
    return false;

  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spv.size() * sizeof(uint32_t);
  smci.pCode = spv.data();

  VkShaderModule sm = VK_NULL_HANDLE;
  VkResult r = vkCreateShaderModule(dev_, &smci, nullptr, &sm);
  if (r != VK_SUCCESS) {
    if (err)
      *err = "vkCreateShaderModule failed";
    return false;
  }

  VkPipelineShaderStageCreateInfo stage{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = sm;
  stage.pName = "main";

  VkComputePipelineCreateInfo cpci{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage = stage;
  cpci.layout = pl_;

  VkPipeline pipe = VK_NULL_HANDLE;
  r = vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);

  vkDestroyShaderModule(dev_, sm, nullptr);

  if (r != VK_SUCCESS) {
    if (err) {
      std::ostringstream oss;
      oss << "vkCreateComputePipelines failed for kid="
          << gemm_f32_kernel_name(kid);
      *err = oss.str();
    }
    return false;
  }

  e.pipe = pipe;
  e.ready = true;
  return true;
}

bool GemmF32PipelineCache::get_or_create(VkPhysicalDevice phys,
                                         GemmF32KernelId kid,
                                         std::string *err) {
  if (!ensure_common_layouts(err))
    return false;

  auto &e = pipes_[(uint32_t)kid];
  if (e.ready && e.pipe != VK_NULL_HANDLE)
    return true;

  return create_entry(phys, kid, e, err);
}

void GemmF32PipelineCache::destroy() {
  if (dev_ == VK_NULL_HANDLE)
    return;

  for (auto &kv : pipes_) {
    if (kv.second.pipe != VK_NULL_HANDLE) {
      vkDestroyPipeline(dev_, kv.second.pipe, nullptr);
      kv.second.pipe = VK_NULL_HANDLE;
    }
    kv.second.ready = false;
  }
  pipes_.clear();

  if (pl_ != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(dev_, pl_, nullptr);
    pl_ = VK_NULL_HANDLE;
  }
  if (dsl_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(dev_, dsl_, nullptr);
    dsl_ = VK_NULL_HANDLE;
  }
}

bool probe_gemm_f32(VkDevice dev, VkPhysicalDevice phys,
                    GemmF32PipelineCache &cache, GemmF32KernelId kid,
                    std::string *err) {
  if (dev == VK_NULL_HANDLE || phys == VK_NULL_HANDLE)
    return false;
  return cache.get_or_create(phys, kid, err);
}

bool dispatch_gemm_f32(VkDevice dev, VkPhysicalDevice phys, VkCommandBuffer cmd,
                       GemmF32PipelineCache &cache, GemmF32KernelId kid,
                       const GemmF32DispatchParams &p) {
  if (dev == VK_NULL_HANDLE || phys == VK_NULL_HANDLE || cmd == VK_NULL_HANDLE)
    return false;

  if (p.A == VK_NULL_HANDLE || p.B == VK_NULL_HANDLE || p.C == VK_NULL_HANDLE)
    return false;

  std::string err;
  if (!cache.get_or_create(phys, kid, &err)) {
    std::cerr << "dispatch_gemm_f32: pipeline get_or_create failed: " << err
              << "\n";
    return false;
  }

  VkPipeline pipe = cache.pipe(kid);
  if (pipe == VK_NULL_HANDLE)
    return false;

  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 3;

  VkDescriptorPoolCreateInfo dpci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;

  VkDescriptorPool dp = VK_NULL_HANDLE;
  if (vkCreateDescriptorPool(dev, &dpci, nullptr, &dp) != VK_SUCCESS)
    return false;

  VkDescriptorSetLayout dsl = cache.dsl();

  VkDescriptorSetAllocateInfo dsai{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = dp;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;

  VkDescriptorSet ds = VK_NULL_HANDLE;
  if (vkAllocateDescriptorSets(dev, &dsai, &ds) != VK_SUCCESS) {
    vkDestroyDescriptorPool(dev, dp, nullptr);
    return false;
  }

  VkDescriptorBufferInfo dbA{p.A, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo dbB{p.B, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo dbC{p.C, 0, VK_WHOLE_SIZE};

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

  GemmF32RunArgs a = p.args;
  GemmF32PushConstants pc{a.M, a.N, a.K, a.lda, a.ldb, a.ldc};

  VkPipelineLayout pl = cache.pl();

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0,
                          nullptr);
  vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(GemmF32PushConstants), &pc);

  uint32_t gx = ceil_div_u32(a.N, 16u);
  uint32_t gy = ceil_div_u32(a.M, 16u);

  if (gx == 0 || gy == 0) {
    std::cerr << "dispatch_gemm_f32: gx/gy=0 for kid="
              << gemm_f32_kernel_name(kid) << "\n";
    vkDestroyDescriptorPool(dev, dp, nullptr);
    return false;
  }

  vkCmdDispatch(cmd, gx, gy, 1);

  vkDestroyDescriptorPool(dev, dp, nullptr);
  return true;
}

} // namespace gcore::rt::vk
