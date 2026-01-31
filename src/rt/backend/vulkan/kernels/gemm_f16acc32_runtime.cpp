#include "gemm_f16acc32_runtime.hpp"

#include "../autotune/vk_autotune.hpp" // greta::vk_autotune::Cache + probe_device + make_bucket

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace gcore::rt::vk {

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
  std::string t;
  t.reserve(s.size());
  for (char c : s)
    t.push_back((char)std::tolower((unsigned char)c));
  return (t == "1" || t == "true" || t == "yes" || t == "on");
}

static uint32_t ceil_div_u32(uint32_t x, uint32_t d) {
  return (x + d - 1u) / d;
}

std::optional<GemmKernelId> kernel_from_name(const std::string &name) {
  if (name == "tiled_f16acc32")
    return GemmKernelId::tiled_f16acc32;
  if (name == "tiled_vec2")
    return GemmKernelId::tiled_vec2;
  if (name == "tiled_vec2_32x8")
    return GemmKernelId::tiled_vec2_32x8;
  if (name == "tiled_vec2_db")
    return GemmKernelId::tiled_vec2_db;
  if (name == "subgroup")
    return GemmKernelId::subgroup;
  return std::nullopt;
}

const char *kernel_name(GemmKernelId id) {
  switch (id) {
  case GemmKernelId::tiled_f16acc32:
    return "tiled_f16acc32";
  case GemmKernelId::tiled_vec2:
    return "tiled_vec2";
  case GemmKernelId::tiled_vec2_32x8:
    return "tiled_vec2_32x8";
  case GemmKernelId::tiled_vec2_db:
    return "tiled_vec2_db";
  case GemmKernelId::subgroup:
    return "subgroup";
  }
  return "unknown";
}

const char *kernel_spv_filename(GemmKernelId id) {
  switch (id) {
  case GemmKernelId::tiled_f16acc32:
    return "gemm_f16acc32_tiled.comp.spv";
  case GemmKernelId::tiled_vec2:
    return "gemm_f16acc32_tiled_vec2.comp.spv";
  case GemmKernelId::tiled_vec2_32x8:
    return "gemm_f16acc32_tiled_vec2_32x8.comp.spv";
  case GemmKernelId::tiled_vec2_db:
    return "gemm_f16acc32_tiled_vec2_db.comp.spv";
  case GemmKernelId::subgroup:
    return "gemm_f16acc32_subgroup.comp.spv";
  }
  return "";
}

std::vector<uint32_t>
GemmPipelineCache::read_spv_u32(const std::string &filename,
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

VkPipeline GemmPipelineCache::pipe(GemmKernelId kid) const {
  auto it = pipes_.find((uint32_t)kid);
  if (it == pipes_.end())
    return VK_NULL_HANDLE;
  return it->second.pipe;
}

bool GemmPipelineCache::ensure_common_layouts(std::string *err) {
  if (dsl_ != VK_NULL_HANDLE && pl_ != VK_NULL_HANDLE)
    return true;

  // DSL: 3 storage buffers (A,B,C)
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
  pcr.size = sizeof(GemmPushConstants);

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

bool GemmPipelineCache::create_entry(VkPhysicalDevice /*phys*/,
                                     GemmKernelId kid, Entry &e,
                                     std::string *err) {
  // Load SPIR-V
  const char *spv_name = kernel_spv_filename(kid);
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

  // Subgroup: require subgroup size 32 (si y solo si el device lo habilitó)
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT reqSG{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT};
  if (kid == GemmKernelId::subgroup) {
    if (!subgroup_size_control_) {
      vkDestroyShaderModule(dev_, sm, nullptr);
      if (err)
        *err = "Kernel subgroup requiere VK_EXT_subgroup_size_control + "
               "feature subgroupSizeControl (no habilitado en VkDevice)";
      return false;
    }
    reqSG.requiredSubgroupSize = 32;
    stage.flags =
        VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT;
    stage.pNext = &reqSG;
  }

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
      oss << "vkCreateComputePipelines failed for kid=" << kernel_name(kid);
      *err = oss.str();
    }
    return false;
  }

  e.pipe = pipe;
  e.ready = true;
  return true;
}

bool GemmPipelineCache::get_or_create(VkPhysicalDevice phys, GemmKernelId kid,
                                      std::string *err) {
  if (!ensure_common_layouts(err))
    return false;

  auto &e = pipes_[(uint32_t)kid];
  if (e.ready && e.pipe != VK_NULL_HANDLE)
    return true;

  return create_entry(phys, kid, e, err);
}

void GemmPipelineCache::destroy() {
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

std::optional<GemmKernelId>
resolve_gemm_kernel_id_autotune(uint32_t M, uint32_t N, uint32_t K) {
  // 1) FORCE
  {
    std::string forced = getenv_str("GRETA_VK_AUTOTUNE_FORCE");
    if (!forced.empty()) {
      auto kid = kernel_from_name(forced);
      if (kid)
        return kid;
      // si el usuario puso un nombre inválido, seguimos con cache/fallback
    }
  }

  // 2) Cache
  auto di_opt = greta::vk_autotune::probe_device();
  if (di_opt) {
    const std::string device_key = di_opt->key_string();
    const std::string bucket = greta::vk_autotune::make_bucket(M, N, K);

    greta::vk_autotune::Cache cache;
    cache.load();

    auto w = cache.find_winner(device_key, bucket);
    if (w && !w->empty()) {
      auto kid = kernel_from_name(*w);
      if (kid)
        return kid;
    }
  }

  // 3) Fallback estable (tu mejor medido)
  return GemmKernelId::tiled_vec2_32x8;
}

bool probe_gemm_f16acc32(VkDevice dev, VkPhysicalDevice phys,
                         GemmPipelineCache &cache, GemmKernelId kid,
                         std::string *err) {
  if (dev == VK_NULL_HANDLE || phys == VK_NULL_HANDLE)
    return false;
  return cache.get_or_create(phys, kid, err);
}

bool dispatch_gemm_f16acc32(VkDevice dev, VkPhysicalDevice phys,
                            VkCommandBuffer cmd, GemmPipelineCache &cache,
                            GemmKernelId kid, const GemmDispatchParams &p) {
  if (dev == VK_NULL_HANDLE || phys == VK_NULL_HANDLE || cmd == VK_NULL_HANDLE)
    return false;

  if (p.A == VK_NULL_HANDLE || p.B == VK_NULL_HANDLE || p.C == VK_NULL_HANDLE)
    return false;

  // Ensure pipeline exists
  std::string err;
  if (!cache.get_or_create(phys, kid, &err)) {
    std::cerr << "dispatch_gemm_f16acc32: pipeline get_or_create failed: "
              << err << "\n";
    return false;
  }

  VkPipeline pipe = cache.pipe(kid);
  if (pipe == VK_NULL_HANDLE)
    return false;

  // Descriptor set ephemeral (simple): create pool+set per call (safe).
  // Si querés, luego lo optimizamos a un pool persistente por thread.
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

  // Push constants
  GemmRunArgs a = p.args;
  GemmPushConstants pc{a.M, a.N, a.K, a.lda, a.ldb, a.ldc};

  VkPipelineLayout pl = cache.pl();

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0,
                          nullptr);
  vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(GemmPushConstants), &pc);

  // Dispatch mapping (compatible con tu familia tiled actual)
  // gx = ceil(N, 32), gy = ceil(M, 8)
  uint32_t gx = ceil_div_u32(a.N, 32u);
  uint32_t gy = ceil_div_u32(a.M, 8u);

  if (gx == 0 || gy == 0) {
    std::cerr << "dispatch_gemm_f16acc32: gx/gy=0 for kid=" << kernel_name(kid)
              << "\n";
    vkDestroyDescriptorPool(dev, dp, nullptr);
    return false;
  }

  vkCmdDispatch(cmd, gx, gy, 1);

  // Destroy DS pool (safe). Más adelante lo persistimos.
  vkDestroyDescriptorPool(dev, dp, nullptr);
  return true;
}

bool dispatch_gemm_f16acc32_auto(VkDevice dev, VkPhysicalDevice phys,
                                 VkCommandBuffer cmd, GemmPipelineCache &cache,
                                 const GemmDispatchParams &p,
                                 std::string *err) {
  auto kid_opt = resolve_gemm_kernel_id_autotune(p.args.M, p.args.N, p.args.K);
  if (!kid_opt) {
    if (err)
      *err = "resolve_gemm_kernel_id_autotune returned nullopt";
    return false;
  }

  GemmKernelId kid = *kid_opt;
  const bool forced = !getenv_str("GRETA_VK_AUTOTUNE_FORCE").empty();

  if (kid == GemmKernelId::subgroup && !cache.subgroup_size_control_enabled()) {
    if (!forced) {
      kid = GemmKernelId::tiled_vec2_32x8;
    }
  }

  bool ok = dispatch_gemm_f16acc32(dev, phys, cmd, cache, kid, p);
  if (!ok && err) {
    std::ostringstream oss;
    oss << "dispatch_gemm_f16acc32_auto failed for kid=" << kernel_name(kid);
    *err = oss.str();
  }
  return ok;
}

} // namespace gcore::rt::vk
