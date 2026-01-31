#include "gcore/rt/vk/gemm.hpp"
#include "gcore/rt/vk/buffer.hpp"

#include "../kernels/gemm_f16acc32_runtime.hpp"
#include "../kernels/gemm_f32_runtime.hpp"
#include "../autotune/vk_autotune.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
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

static bool env_flag_default(const char *k, bool def) {
  std::string s = getenv_str(k);
  if (s.empty())
    return def;
  std::string t;
  t.reserve(s.size());
  for (char c : s)
    t.push_back((char)std::tolower((unsigned char)c));
  if (t == "1" || t == "true" || t == "yes" || t == "on")
    return true;
  if (t == "0" || t == "false" || t == "no" || t == "off")
    return false;
  return def;
}

static uint32_t env_u32(const char *k, uint32_t def) {
  std::string s = getenv_str(k);
  if (s.empty())
    return def;
  try {
    return static_cast<uint32_t>(std::stoul(s));
  } catch (...) {
    return def;
  }
}

static std::filesystem::path fp16_blacklist_path() {
  auto xdg = std::getenv("XDG_CACHE_HOME");
  std::filesystem::path base;
  if (xdg && *xdg) {
    base = xdg;
  } else {
    auto home = std::getenv("HOME");
    if (home && *home)
      base = std::filesystem::path(home) / ".cache";
    else
      base = std::filesystem::current_path();
  }
  return base / "gretacore" / "vk_fp16_blacklist.txt";
}

static std::optional<std::string> current_device_key() {
  auto di = greta::vk_autotune::probe_device();
  if (!di)
    return std::nullopt;
  return di->key_string();
}

static bool fp16_blacklisted_for_device(const std::string &key) {
  std::ifstream f(fp16_blacklist_path());
  if (!f)
    return false;
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line == key)
      return true;
  }
  return false;
}

static void fp16_blacklist_add(const std::string &key) {
  auto path = fp16_blacklist_path();
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  if (fp16_blacklisted_for_device(key))
    return;
  std::ofstream f(path, std::ios::app);
  if (!f)
    return;
  f << key << "\n";
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

static bool fp16_health_check(Backend *backend, GemmPipelineCache &cache,
                              std::string *err) {
  const uint32_t M = 8, N = 8, K = 8;
  const uint32_t timeout_ms =
      env_u32("GRETA_VK_FP16_HEALTHCHECK_TIMEOUT_MS", 2000);

  const VkDeviceSize sizeA = VkDeviceSize(M * K * sizeof(uint16_t));
  const VkDeviceSize sizeB = VkDeviceSize(K * N * sizeof(uint16_t));
  const VkDeviceSize sizeC = VkDeviceSize(M * N * sizeof(float));

  Buffer bufA, bufB, bufC;
  VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  VkMemoryPropertyFlags props =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  if (!create_buffer(backend->physical_device(), backend->device(), sizeA,
                     usage, props, &bufA, err) ||
      !create_buffer(backend->physical_device(), backend->device(), sizeB,
                     usage, props, &bufB, err) ||
      !create_buffer(backend->physical_device(), backend->device(), sizeC,
                     usage, props, &bufC, err)) {
    return false;
  }

  void *pA = nullptr;
  void *pB = nullptr;
  void *pC = nullptr;
  if (!map_buffer(backend->device(), bufA, &pA, err) ||
      !map_buffer(backend->device(), bufB, &pB, err) ||
      !map_buffer(backend->device(), bufC, &pC, err)) {
    return false;
  }
  std::memset(pA, 0, sizeA);
  std::memset(pB, 0, sizeB);
  std::memset(pC, 0, sizeC);
  unmap_buffer(backend->device(), bufA);
  unmap_buffer(backend->device(), bufB);
  unmap_buffer(backend->device(), bufC);

  VkCommandBufferAllocateInfo cbai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = backend->command_pool();
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(backend->device(), &cbai, &cmd) != VK_SUCCESS) {
    if (err)
      *err = "vkAllocateCommandBuffers failed";
    return false;
  }

  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
    if (err)
      *err = "vkBeginCommandBuffer failed";
    return false;
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
  bms[0].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
  bms[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  bms[1].buffer = bufB.buf;
  bms[1].offset = 0;
  bms[1].size = VK_WHOLE_SIZE;
  bms[1].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
  bms[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  bms[2].buffer = bufC.buf;
  bms[2].offset = 0;
  bms[2].size = VK_WHOLE_SIZE;
  bms[2].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
  bms[2].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 3,
                       bms, 0, nullptr);

  GemmDispatchParams p{};
  p.A = bufA.buf;
  p.B = bufB.buf;
  p.C = bufC.buf;
  p.args.M = M;
  p.args.N = N;
  p.args.K = K;
  p.args.lda = K;
  p.args.ldb = N;
  p.args.ldc = N;

  if (!dispatch_gemm_f16acc32(backend->device(), backend->physical_device(), cmd,
                              cache, GemmKernelId::tiled_f16acc32, p)) {
    if (err)
      *err = "dispatch_gemm_f16acc32 failed";
    return false;
  }

  VkMemoryBarrier mb{};
  mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0,
                       nullptr);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    if (err)
      *err = "vkEndCommandBuffer failed";
    return false;
  }

  bool ok = submit_and_wait(backend->device(), backend->queue(), cmd,
                            timeout_ms, err);

  destroy_buffer(backend->device(), &bufA);
  destroy_buffer(backend->device(), &bufB);
  destroy_buffer(backend->device(), &bufC);
  return ok;
}

static GemmVariant kid_to_variant(GemmKernelId kid) {
  switch (kid) {
  case GemmKernelId::tiled_f16acc32:
    return GemmVariant::TiledF16Acc32;
  case GemmKernelId::tiled_vec2:
    return GemmVariant::TiledVec2;
  case GemmKernelId::tiled_vec2_32x8:
    return GemmVariant::TiledVec2_32x8;
  case GemmKernelId::tiled_vec2_db:
    return GemmVariant::TiledVec2_DB;
  case GemmKernelId::subgroup:
    return GemmVariant::Subgroup32;
  }
  return GemmVariant::TiledVec2_32x8;
}

const char *GemmF16Acc32::variant_name(GemmVariant v) {
  switch (v) {
  case GemmVariant::TiledF16Acc32:
    return "tiled_f16acc32";
  case GemmVariant::TiledVec2:
    return "tiled_vec2";
  case GemmVariant::TiledVec2_32x8:
    return "tiled_vec2_32x8";
  case GemmVariant::TiledVec2_DB:
    return "tiled_vec2_db";
  case GemmVariant::Subgroup32:
    return "subgroup";
  }
  return "unknown";
}

struct GemmF16Acc32::Impl {
  explicit Impl(VkDevice dev) : cache(dev) {}
  GemmPipelineCache cache;
  bool probe_only = false;
};

GemmF16Acc32::~GemmF16Acc32() { shutdown(); }

bool GemmF16Acc32::init(Backend *backend, std::string shader_dir,
                        std::string *err) {
  if (!backend || backend->device() == VK_NULL_HANDLE) {
    if (err)
      *err = "GemmF16Acc32::init: backend inválido o no inicializado";
    return false;
  }
  if (backend->gpu_blacklisted()) {
    if (err)
      *err = "GemmF16Acc32::init: GPU/driver blacklisted (ver GRETA_VK_ALLOW_UNSAFE)";
    return false;
  }
  if (env_true("GRETA_VK_FORCE_FP32") || env_true("GRETA_VK_DISABLE_FP16")) {
    if (err)
      *err = "GemmF16Acc32::init: FP16 forzado OFF (GRETA_VK_FORCE_FP32=1)";
    return false;
  }
  if (!backend->fp16_enabled()) {
    if (err)
      *err = "GemmF16Acc32::init: FP16 no habilitado en VkDevice "
             "(requiere VK_KHR_shader_float16_int8 + VK_KHR_16bit_storage)";
    return false;
  }

  const bool allow_unsafe = env_true("GRETA_VK_FP16_ALLOW_UNSAFE");
  if (!allow_unsafe) {
    auto key = current_device_key();
    if (key && fp16_blacklisted_for_device(*key)) {
      if (err)
        *err = "GemmF16Acc32::init: FP16 deshabilitado por healthcheck previo "
               "(set GRETA_VK_FP16_ALLOW_UNSAFE=1 para forzar)";
      return false;
    }
  }

  // (Re)create impl/cache first (clears previous state)
  shutdown();
  backend_ = backend;

  // Resolve shader dir
  shader_dir_ = std::move(shader_dir);
  if (shader_dir_.empty()) {
    auto envd = getenv_str("GRETA_VK_SHADER_DIR");
    if (!envd.empty())
      shader_dir_ = envd;
    else
      shader_dir_ = (std::filesystem::current_path() / "build").string();
  }

  impl_ = new Impl(backend_->device());
  impl_->cache.set_shader_dir(shader_dir_);

  // Backend decide si habilitó subgroup size control en VkDevice
  impl_->cache.set_subgroup_size_control(
      backend_->subgroup_size_control_enabled());

  impl_->probe_only = env_true("GRETA_VK_PROBE_ONLY");
  if (impl_->probe_only) {
    std::string probe_err;
    auto kid_opt = resolve_gemm_kernel_id_autotune(1, 1, 1);
    GemmKernelId kid = kid_opt.value_or(GemmKernelId::tiled_vec2_32x8);
    if (!probe_gemm_f16acc32(backend_->device(), backend_->physical_device(),
                             impl_->cache, kid, &probe_err)) {
      if (err)
        *err = "GemmF16Acc32::init: probe failed: " + probe_err;
      return false;
    }
  }

  const bool do_healthcheck =
      env_flag_default("GRETA_VK_FP16_HEALTHCHECK", false);
  if (do_healthcheck && !impl_->probe_only && !allow_unsafe) {
    std::string hc_err;
    if (!fp16_health_check(backend_, impl_->cache, &hc_err)) {
      auto key = current_device_key();
      if (key && !env_true("GRETA_VK_FP16_BLACKLIST_NO_WRITE"))
        fp16_blacklist_add(*key);
      if (err)
        *err = "GemmF16Acc32::init: FP16 healthcheck failed: " + hc_err;
      return false;
    }
  }

  last_variant_ = GemmVariant::TiledVec2_32x8;
  last_winner_name_.clear();
  return true;
}

void GemmF16Acc32::shutdown() {
  if (impl_) {
    impl_->cache.destroy();
    delete impl_;
    impl_ = nullptr;
  }
  backend_ = nullptr;
  shader_dir_.clear();
  last_winner_name_.clear();
  last_variant_ = GemmVariant::TiledVec2_32x8;
}

bool GemmF16Acc32::record_dispatch(VkCommandBuffer cmd,
                                   const GemmDispatchDesc &d,
                                   std::string *err) {
  if (!backend_ || backend_->device() == VK_NULL_HANDLE) {
    if (err)
      *err = "GemmF16Acc32::record_dispatch: backend no inicializado";
    return false;
  }
  if (!impl_) {
    if (err)
      *err = "GemmF16Acc32::record_dispatch: impl/cache no inicializado (llamá "
             "init())";
    return false;
  }
  if (impl_->probe_only) {
    if (err)
      *err = "GemmF16Acc32::record_dispatch: probe-only (GRETA_VK_PROBE_ONLY=1)";
    return false;
  }
  if (cmd == VK_NULL_HANDLE) {
    if (err)
      *err = "GemmF16Acc32::record_dispatch: cmd == VK_NULL_HANDLE";
    return false;
  }
  if (d.A == VK_NULL_HANDLE || d.B == VK_NULL_HANDLE || d.C == VK_NULL_HANDLE) {
    if (err)
      *err = "GemmF16Acc32::record_dispatch: buffers A/B/C inválidos";
    return false;
  }
  if (d.M == 0 || d.N == 0 || d.K == 0) {
    if (err)
      *err = "GemmF16Acc32::record_dispatch: dimensiones M/N/K inválidas";
    return false;
  }

  // Build dispatch params for kernels runtime
  GemmDispatchParams p{};
  p.A = d.A;
  p.B = d.B;
  p.C = d.C;

  p.args.M = d.M;
  p.args.N = d.N;
  p.args.K = d.K;
  p.args.lda = d.lda;
  p.args.ldb = d.ldb;
  p.args.ldc = d.ldc;

  bool ok = dispatch_gemm_f16acc32_auto(backend_->device(),
                                       backend_->physical_device(), cmd,
                                       impl_->cache, p, err);
  if (!ok)
    return false;

  // Telemetry: resolve winner name for debugging (best-effort)
  auto kid_opt = resolve_gemm_kernel_id_autotune(d.M, d.N, d.K);
  if (kid_opt) {
    last_winner_name_ = kernel_name(*kid_opt);
    last_variant_ = kid_to_variant(*kid_opt);
  }

  return true;
}

struct GemmF32::Impl {
  explicit Impl(VkDevice dev) : cache(dev) {}
  GemmF32PipelineCache cache;
  bool probe_only = false;
};

GemmF32::~GemmF32() { shutdown(); }

bool GemmF32::init(Backend *backend, std::string shader_dir, std::string *err) {
  if (!backend || backend->device() == VK_NULL_HANDLE) {
    if (err)
      *err = "GemmF32::init: backend inválido o no inicializado";
    return false;
  }
  if (backend->gpu_blacklisted()) {
    if (err)
      *err = "GemmF32::init: GPU/driver blacklisted (ver GRETA_VK_ALLOW_UNSAFE)";
    return false;
  }

  shutdown();
  backend_ = backend;

  shader_dir_ = std::move(shader_dir);
  if (shader_dir_.empty()) {
    auto envd = getenv_str("GRETA_VK_SHADER_DIR");
    if (!envd.empty())
      shader_dir_ = envd;
    else
      shader_dir_ = (std::filesystem::current_path() / "build").string();
  }

  impl_ = new Impl(backend_->device());
  impl_->cache.set_shader_dir(shader_dir_);

  impl_->probe_only = env_true("GRETA_VK_PROBE_ONLY");
  if (impl_->probe_only) {
    std::string probe_err;
    if (!probe_gemm_f32(backend_->device(), backend_->physical_device(),
                        impl_->cache, GemmF32KernelId::tiled, &probe_err)) {
      if (err)
        *err = "GemmF32::init: probe failed: " + probe_err;
      return false;
    }
  }

  return true;
}

void GemmF32::shutdown() {
  if (impl_) {
    impl_->cache.destroy();
    delete impl_;
    impl_ = nullptr;
  }
  backend_ = nullptr;
  shader_dir_.clear();
}

bool GemmF32::record_dispatch(VkCommandBuffer cmd, const GemmDispatchDesc &d,
                              std::string *err) {
  if (!backend_ || backend_->device() == VK_NULL_HANDLE) {
    if (err)
      *err = "GemmF32::record_dispatch: backend no inicializado";
    return false;
  }
  if (!impl_) {
    if (err)
      *err = "GemmF32::record_dispatch: impl/cache no inicializado (llamá init())";
    return false;
  }
  if (impl_->probe_only) {
    if (err)
      *err = "GemmF32::record_dispatch: probe-only (GRETA_VK_PROBE_ONLY=1)";
    return false;
  }
  if (cmd == VK_NULL_HANDLE) {
    if (err)
      *err = "GemmF32::record_dispatch: cmd == VK_NULL_HANDLE";
    return false;
  }
  if (d.A == VK_NULL_HANDLE || d.B == VK_NULL_HANDLE || d.C == VK_NULL_HANDLE) {
    if (err)
      *err = "GemmF32::record_dispatch: buffers A/B/C inválidos";
    return false;
  }
  if (d.M == 0 || d.N == 0 || d.K == 0) {
    if (err)
      *err = "GemmF32::record_dispatch: dimensiones M/N/K inválidas";
    return false;
  }

  GemmF32DispatchParams p{};
  p.A = d.A;
  p.B = d.B;
  p.C = d.C;

  p.args.M = d.M;
  p.args.N = d.N;
  p.args.K = d.K;
  p.args.lda = d.lda;
  p.args.ldb = d.ldb;
  p.args.ldc = d.ldc;

  bool ok = dispatch_gemm_f32(backend_->device(), backend_->physical_device(),
                              cmd, impl_->cache, GemmF32KernelId::tiled, p);
  if (!ok && err)
    *err = "dispatch_gemm_f32 failed";
  return ok;
}

GemmAuto::~GemmAuto() { shutdown(); }

bool GemmAuto::init(Backend *backend, std::string shader_dir,
                    GemmPrecision preferred, std::string *err) {
  backend_ = backend;
  active_precision_ = GemmPrecision::F32;
  fallback_reason_.clear();
  const bool force_fp32 =
      env_true("GRETA_VK_FORCE_FP32") || env_true("GRETA_VK_DISABLE_FP16");

  if (!backend || backend->device() == VK_NULL_HANDLE) {
    if (err)
      *err = "GemmAuto::init: backend inválido o no inicializado";
    return false;
  }

  if (backend->gpu_blacklisted()) {
    if (err)
      *err = "GemmAuto::init: GPU/driver blacklisted (ver GRETA_VK_ALLOW_UNSAFE)";
    return false;
  }

  auto key = current_device_key();
  if (key && fp16_blacklisted_for_device(*key) &&
      !env_true("GRETA_VK_FP16_ALLOW_UNSAFE")) {
    std::ostringstream oss;
    oss << "fp16_blacklisted device_key=" << *key
        << " blacklist_path=" << fp16_blacklist_path().string()
        << " (set GRETA_VK_FP16_ALLOW_UNSAFE=1 to override)";
    fallback_reason_ = oss.str();
  }

  if (!force_fp32 && preferred == GemmPrecision::F16Acc32 &&
      backend->fp16_enabled()) {
    if (f16_.init(backend, shader_dir, err)) {
      active_precision_ = GemmPrecision::F16Acc32;
      fallback_reason_.clear();
      return true;
    }
  }

  if (f32_.init(backend, shader_dir, err)) {
    active_precision_ = GemmPrecision::F32;
    if (fallback_reason_.empty())
      fallback_reason_ = "fallback_to_fp32";
    return true;
  }

  return false;
}

void GemmAuto::shutdown() {
  f16_.shutdown();
  f32_.shutdown();
  backend_ = nullptr;
  active_precision_ = GemmPrecision::F32;
}

bool GemmAuto::record_dispatch(VkCommandBuffer cmd, const GemmDispatchDesc &d,
                               std::string *err) {
  if (active_precision_ == GemmPrecision::F16Acc32)
    return f16_.record_dispatch(cmd, d, err);
  return f32_.record_dispatch(cmd, d, err);
}

} // namespace gcore::rt::vk
