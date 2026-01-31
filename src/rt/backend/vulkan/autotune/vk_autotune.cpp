#include "vk_autotune.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <vulkan/vulkan.h>

namespace greta::vk_autotune {

static std::string trim(std::string s) {
  auto is_ws = [](unsigned char c) { return std::isspace(c) != 0; };
  while (!s.empty() && is_ws((unsigned char)s.front()))
    s.erase(s.begin());
  while (!s.empty() && is_ws((unsigned char)s.back()))
    s.pop_back();
  return s;
}

static bool env_true(const char *k) {
  const char *v = std::getenv(k);
  if (!v || !*v)
    return false;
  std::string s(v);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  return (s == "1" || s == "true" || s == "yes" || s == "on");
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

static bool fp16_blacklisted_for_key(const std::string &key) {
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

bool fp16_blacklisted(const DeviceInfo &di) {
  return fp16_blacklisted_for_key(di.key_string());
}

std::string fp16_blacklist_reason(const DeviceInfo &di) {
  return "fp16_blacklisted device_key=" + di.key_string() +
         " blacklist_path=" + fp16_blacklist_path().string() +
         " (set GRETA_VK_FP16_ALLOW_UNSAFE=1 to override)";
}

std::string fp16_blacklist_path_string() {
  return fp16_blacklist_path().string();
}

void tag_fp16_blacklist_cache(const DeviceInfo &di,
                              const std::string &reason) {
  if (!fp16_blacklisted(di))
    return;
  if (env_flag_true("GRETA_VK_AUTOTUNE_NO_WRITE") ||
      env_true("GRETA_VK_FP16_BLACKLIST_TAG_NO_WRITE"))
    return;
  Cache cache;
  cache.load();
  cache.upsert(di.key_string(), "meta:fp16_blacklist", "1");
  if (!reason.empty())
    cache.upsert(di.key_string(), "meta:fp16_fallback_reason", reason);
  cache.save();
}

std::string DeviceInfo::key_string() const {
  // Keep it stable and explicit. driver_name might be empty; that's ok.
  std::ostringstream oss;
  oss << std::hex;
  oss << "vid=0x" << vendor_id << ";did=0x" << device_id << std::dec;
  oss << ";name=" << device_name;
  oss << ";driver=" << driver_name;
  // subgroup ranges affect kernel viability; include.
  oss << ";sg=(" << reported_subgroup_size << "," << min_subgroup_size << ","
      << max_subgroup_size << ")";
  return oss.str();
}

static void vk_check(VkResult r, const char *msg) {
  if (r != VK_SUCCESS) {
    std::cerr << "Vulkan error: " << msg << " (VkResult=" << r << ")\n";
    std::exit(1);
  }
}

static bool contains_llvmpipe(const char *name) {
  if (!name)
    return false;
  std::string s = name;
  return s.find("llvmpipe") != std::string::npos;
}

static std::string get_driver_name_best_effort(VkPhysicalDevice phy) {
  // Try VK_KHR_driver_properties if available.
  uint32_t extCount = 0;
  if (vkEnumerateDeviceExtensionProperties(phy, nullptr, &extCount, nullptr) !=
      VK_SUCCESS)
    return {};
  std::vector<VkExtensionProperties> exts(extCount);
  if (vkEnumerateDeviceExtensionProperties(phy, nullptr, &extCount,
                                           exts.data()) != VK_SUCCESS)
    return {};

  bool has_driver_props = false;
  for (auto &e : exts) {
    if (std::string(e.extensionName) == "VK_KHR_driver_properties") {
      has_driver_props = true;
      break;
    }
  }
  if (!has_driver_props)
    return {};

  VkPhysicalDeviceDriverPropertiesKHR drv{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 p2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  p2.pNext = &drv;
  vkGetPhysicalDeviceProperties2(phy, &p2);
  return std::string(drv.driverName);
}

static std::vector<VkExtensionProperties>
enumerate_device_extensions(VkPhysicalDevice phy) {
  uint32_t extCount = 0;
  if (vkEnumerateDeviceExtensionProperties(phy, nullptr, &extCount, nullptr) !=
      VK_SUCCESS)
    return {};
  std::vector<VkExtensionProperties> exts(extCount);
  if (extCount) {
    if (vkEnumerateDeviceExtensionProperties(phy, nullptr, &extCount,
                                             exts.data()) != VK_SUCCESS)
      return {};
  }
  return exts;
}

static bool has_extension(const std::vector<VkExtensionProperties> &exts,
                          const char *name) {
  for (auto &e : exts) {
    if (std::string(e.extensionName) == name)
      return true;
  }
  return false;
}

static void probe_subgroup_best_effort(VkPhysicalDevice phy, uint32_t &reported,
                                       uint32_t &minS, uint32_t &maxS) {
  reported = 0;
  minS = 0;
  maxS = 0;

  VkPhysicalDeviceSubgroupProperties sg{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};

  // Check extension presence; if absent, we still get sg.subgroupSize.
  auto exts = enumerate_device_extensions(phy);
  bool has_sg_ctrl = has_extension(exts, "VK_EXT_subgroup_size_control");

  VkPhysicalDeviceProperties2 p2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};

  if (has_sg_ctrl) {
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT sgctrl{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT};
    p2.pNext = &sg;
    sg.pNext = &sgctrl;
    vkGetPhysicalDeviceProperties2(phy, &p2);
    reported = sg.subgroupSize;
    minS = sgctrl.minSubgroupSize;
    maxS = sgctrl.maxSubgroupSize;
  } else {
    p2.pNext = &sg;
    vkGetPhysicalDeviceProperties2(phy, &p2);
    reported = sg.subgroupSize;
  }
}

std::optional<DeviceInfo> probe_device() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "gretacore_autotune_probe";
  app.applicationVersion = 1;
  app.pEngineName = "gretacore";
  app.engineVersion = 1;
  app.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;

  VkInstance inst{};
  VkResult r = vkCreateInstance(&ici, nullptr, &inst);
  if (r != VK_SUCCESS)
    return std::nullopt;

  uint32_t pcount = 0;
  r = vkEnumeratePhysicalDevices(inst, &pcount, nullptr);
  if (r != VK_SUCCESS || pcount == 0) {
    vkDestroyInstance(inst, nullptr);
    return std::nullopt;
  }

  std::vector<VkPhysicalDevice> phys(pcount);
  vk_check(vkEnumeratePhysicalDevices(inst, &pcount, phys.data()),
           "vkEnumeratePhysicalDevices");

  VkPhysicalDevice chosen = phys[0];
  for (auto d : phys) {
    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(d, &p);
    if (!contains_llvmpipe(p.deviceName)) {
      chosen = d;
      break;
    }
  }

  VkPhysicalDeviceProperties p{};
  vkGetPhysicalDeviceProperties(chosen, &p);

  DeviceInfo di;
  di.vendor_id = p.vendorID;
  di.device_id = p.deviceID;
  di.device_name = std::string(p.deviceName);
  di.driver_name = get_driver_name_best_effort(chosen);

  probe_subgroup_best_effort(chosen, di.reported_subgroup_size,
                             di.min_subgroup_size, di.max_subgroup_size);

  auto exts = enumerate_device_extensions(chosen);

  // FP16 capability (best-effort)
  const bool has_f16_int8_ext =
      has_extension(exts, "VK_KHR_shader_float16_int8");
  const bool has_16bit_storage_ext =
      has_extension(exts, "VK_KHR_16bit_storage");

  VkPhysicalDeviceFloat16Int8FeaturesKHR f16i8{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR};
  VkPhysicalDevice16BitStorageFeatures storage16{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  VkPhysicalDeviceFeatures2 feats2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  feats2.pNext = &f16i8;
  f16i8.pNext = &storage16;
  vkGetPhysicalDeviceFeatures2(chosen, &feats2);

  di.fp16_supported = has_f16_int8_ext && has_16bit_storage_ext &&
                      (f16i8.shaderFloat16 == VK_TRUE) &&
                      (storage16.storageBuffer16BitAccess == VK_TRUE);
  di.fp16_reason.clear();
  if (!di.fp16_supported) {
    std::ostringstream oss;
    oss << "fp16 not supported (";
    bool first = true;
    if (!has_f16_int8_ext) {
      oss << "missing VK_KHR_shader_float16_int8";
      first = false;
    }
    if (!has_16bit_storage_ext) {
      if (!first)
        oss << ", ";
      oss << "missing VK_KHR_16bit_storage";
      first = false;
    }
    if (has_f16_int8_ext && !f16i8.shaderFloat16) {
      if (!first)
        oss << ", ";
      oss << "feature shaderFloat16=false";
      first = false;
    }
    if (has_16bit_storage_ext && !storage16.storageBuffer16BitAccess) {
      if (!first)
        oss << ", ";
      oss << "feature storageBuffer16BitAccess=false";
    }
    oss << ")";
    di.fp16_reason = oss.str();
  }

  // Safety/blacklist (match Backend policy)
  di.gpu_blacklisted = false;
  di.blacklist_reason.clear();
  if (!env_true("GRETA_VK_ALLOW_UNSAFE")) {
    const bool is_radv =
        (di.driver_name.find("radv") != std::string::npos);
    const bool is_gfx1103 =
        (di.device_name.find("GFX1103") != std::string::npos);
    if (is_radv && is_gfx1103) {
      di.gpu_blacklisted = true;
      di.blacklist_reason =
          "RADV GFX1103 blacklisted (set GRETA_VK_ALLOW_UNSAFE=1 to override)";
    }
  }

  vkDestroyInstance(inst, nullptr);
  return di;
}

// ------------------------
// Env helpers
// ------------------------
std::optional<std::string> env_get(const char *name) {
  const char *v = std::getenv(name);
  if (!v || !*v)
    return std::nullopt;
  return std::string(v);
}

bool env_flag_true(const char *name) {
  auto v = env_get(name);
  if (!v)
    return false;
  std::string s = *v;
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  return (s == "1" || s == "true" || s == "yes" || s == "on");
}

std::optional<double> env_get_double(const char *name) {
  auto v = env_get(name);
  if (!v)
    return std::nullopt;
  try {
    return std::stod(*v);
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<int> env_get_int(const char *name) {
  auto v = env_get(name);
  if (!v)
    return std::nullopt;
  try {
    return std::stoi(*v);
  } catch (...) {
    return std::nullopt;
  }
}

// ------------------------
// Bucket
// ------------------------
std::string make_bucket(uint32_t M, uint32_t N, uint32_t K) {
  std::ostringstream oss;
  oss << "M" << M << "_N" << N << "_K" << K;
  return oss.str();
}

// ------------------------
// Cache implementation
// ------------------------
static std::string resolve_cache_path() {
  // Prefer XDG_CACHE_HOME, else ~/.cache
  auto xdg = env_get("XDG_CACHE_HOME");
  std::filesystem::path base;
  if (xdg) {
    base = *xdg;
  } else {
    auto home = env_get("HOME");
    if (!home)
      return "vk_autotune_cache.json"; // fallback: CWD
    base = std::filesystem::path(*home) / ".cache";
  }
  std::filesystem::path dir = base / "gretacore";
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  return (dir / "vk_autotune.json").string();
}

std::string Cache::path() const {
  if (path_.empty())
    path_ = resolve_cache_path();
  return path_;
}

void Cache::clear() { entries_.clear(); }

static std::string json_escape(const std::string &s) {
  std::ostringstream o;
  for (char c : s) {
    switch (c) {
    case '\\':
      o << "\\\\";
      break;
    case '"':
      o << "\\\"";
      break;
    case '\n':
      o << "\\n";
      break;
    case '\r':
      o << "\\r";
      break;
    case '\t':
      o << "\\t";
      break;
    default:
      o << c;
      break;
    }
  }
  return o.str();
}

// Extremely small “parser”: reads entries written by save() only.
void Cache::load() {
  std::ifstream f(path());
  if (!f)
    return;

  std::string line;
  std::vector<CacheEntry> tmp;

  while (std::getline(f, line)) {
    line = trim(line);
    if (line.find("\"device_key\"") == std::string::npos)
      continue;

    auto get_field = [&](const std::string &key) -> std::optional<std::string> {
      auto pos = line.find("\"" + key + "\"");
      if (pos == std::string::npos)
        return std::nullopt;
      pos = line.find(":", pos);
      if (pos == std::string::npos)
        return std::nullopt;
      pos = line.find("\"", pos);
      if (pos == std::string::npos)
        return std::nullopt;
      auto end = line.find("\"", pos + 1);
      if (end == std::string::npos)
        return std::nullopt;
      return line.substr(pos + 1, end - (pos + 1));
    };

    auto dk = get_field("device_key");
    auto bk = get_field("bucket");
    auto wn = get_field("winner");
    if (dk && bk && wn) {
      tmp.push_back(CacheEntry{*dk, *bk, *wn});
    }
  }

  entries_ = std::move(tmp);
}

void Cache::save() const {
  std::ofstream f(path(), std::ios::trunc);
  if (!f)
    return;

  f << "{\n";
  f << "  \"version\": 1,\n";
  f << "  \"entries\": [\n";
  for (size_t i = 0; i < entries_.size(); i++) {
    const auto &e = entries_[i];
    f << "    {"
      << "\"device_key\":\"" << json_escape(e.device_key) << "\","
      << "\"bucket\":\"" << json_escape(e.bucket) << "\","
      << "\"winner\":\"" << json_escape(e.winner) << "\""
      << "}";
    if (i + 1 != entries_.size())
      f << ",";
    f << "\n";
  }
  f << "  ]\n";
  f << "}\n";
}

std::optional<std::string> Cache::find_winner(const std::string &device_key,
                                              const std::string &bucket) const {
  for (const auto &e : entries_) {
    if (e.device_key == device_key && e.bucket == bucket)
      return e.winner;
  }
  return std::nullopt;
}

void Cache::upsert(const std::string &device_key, const std::string &bucket,
                   const std::string &winner) {
  for (auto &e : entries_) {
    if (e.device_key == device_key && e.bucket == bucket) {
      e.winner = winner;
      return;
    }
  }
  entries_.push_back(CacheEntry{device_key, bucket, winner});
}

// ------------------------
// Candidate parsing / execution
// ------------------------
std::optional<double> parse_mean_tflops(const std::string &output) {
  // Look for: "mean_TFLOPs=0.393"
  auto pos = output.find("mean_TFLOPs=");
  if (pos == std::string::npos)
    return std::nullopt;
  pos += std::strlen("mean_TFLOPs=");
  // Read until whitespace/newline
  size_t end = pos;
  while (end < output.size() && !std::isspace((unsigned char)output[end]))
    end++;
  std::string num = output.substr(pos, end - pos);
  try {
    return std::stod(num);
  } catch (...) {
    return std::nullopt;
  }
}

static std::string shell_escape(const std::string &s) {
  // Minimal single-quote escaping for sh -lc
  std::string out = "'";
  for (char c : s) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  out += "'";
  return out;
}

CandidateResult run_candidate_command(const std::string &candidate_name,
                                      const std::string &command) {
  CandidateResult res;
  res.name = candidate_name;

  // Use /bin/sh -lc "<command> 2>&1"
  std::string full = "/bin/sh -lc " + shell_escape(command + " 2>&1");

  FILE *pipe = popen(full.c_str(), "r");
  if (!pipe) {
    res.exit_code = 127;
    res.raw_output = "popen() failed";
    return res;
  }

  std::ostringstream oss;
  char buf[4096];
  while (std::fgets(buf, sizeof(buf), pipe)) {
    oss << buf;
  }
  int rc = pclose(pipe);

  // NOTE: pclose returns encoded status in POSIX. For our use, keep it raw;
  // wrappers typically check ==0.
  res.exit_code = rc;
  res.raw_output = oss.str();

  auto t = parse_mean_tflops(res.raw_output);
  res.mean_tflops = t.value_or(0.0);
  return res;
}

std::optional<CandidateResult>
pick_best(const std::vector<CandidateResult> &rs) {
  if (rs.empty())
    return std::nullopt;
  const CandidateResult *best = &rs[0];
  for (size_t i = 1; i < rs.size(); i++) {
    if (rs[i].mean_tflops > best->mean_tflops)
      best = &rs[i];
  }
  return *best;
}

std::optional<CandidateResult>
pick_second_best(const std::vector<CandidateResult> &rs) {
  if (rs.size() < 2)
    return std::nullopt;

  // find best index
  size_t best_i = 0;
  for (size_t i = 1; i < rs.size(); i++) {
    if (rs[i].mean_tflops > rs[best_i].mean_tflops)
      best_i = i;
  }

  // find second best among others
  size_t second_i = (best_i == 0 ? 1 : 0);
  for (size_t i = 0; i < rs.size(); i++) {
    if (i == best_i)
      continue;
    if (rs[i].mean_tflops > rs[second_i].mean_tflops)
      second_i = i;
  }

  return rs[second_i];
}

// ------------------------
// High-level resolve_winner
// ------------------------
static bool candidate_valid_for_device(const Candidate &c,
                                       const DeviceInfo &di) {
  if (!c.requires_subgroup32)
    return true;

  // If we couldn't probe min_subgroup_size, allow and let runtime candidate
  // fail.
  if (di.min_subgroup_size == 0)
    return true;

  // Candidate needs subgroup size 32 to be supported.
  return di.min_subgroup_size <= 32;
}

static bool is_fp16_name(const std::string &name) {
  return name.find("f16") != std::string::npos;
}

static bool candidate_is_fp16(const Candidate &c) {
  return is_fp16_name(c.name) || is_fp16_name(c.exe);
}

static bool name_in_candidates(const std::string &name,
                               const std::vector<Candidate> &cands) {
  for (const auto &c : cands) {
    if (c.name == name)
      return true;
  }
  return false;
}

static std::vector<CandidateResult>
run_all_candidates(const RunArgs &args, const std::string &exe_dir,
                   const std::vector<Candidate> &candidates,
                   const DeviceInfo &di) {

  std::vector<CandidateResult> results;
  results.reserve(candidates.size());

  const bool fp16_blk = fp16_blacklisted(di) && !env_true("GRETA_VK_FP16_ALLOW_UNSAFE");

  for (const auto &c : candidates) {
    if (fp16_blk && candidate_is_fp16(c)) {
      CandidateResult r;
      r.name = c.name;
      r.mean_tflops = 0.0;
      r.exit_code = 0;
      r.raw_output = "SKIPPED (fp16 blacklisted)";
      results.push_back(std::move(r));
      continue;
    }

    if (!candidate_valid_for_device(c, di)) {
      CandidateResult r;
      r.name = c.name;
      r.mean_tflops = 0.0;
      r.exit_code = 0;
      r.raw_output = "SKIPPED (device capability mismatch)";
      results.push_back(std::move(r));
      continue;
    }

    std::string cmd = exe_dir + "/" + c.exe + " --m " + std::to_string(args.M) +
                      " --n " + std::to_string(args.N) + " --k " +
                      std::to_string(args.K) + " --iters " +
                      std::to_string(args.iters) + " --batch " +
                      std::to_string(args.batch);

    auto r = run_candidate_command(c.name, cmd);

    // If command failed, keep TFLOPs=0 but store output for debugging.
    if (r.exit_code != 0 && r.mean_tflops < 0.0)
      r.mean_tflops = 0.0;

    results.push_back(std::move(r));
  }

  return results;
}

ResolveResult resolve_winner(const RunArgs &args, const std::string &exe_dir,
                             const std::vector<Candidate> &candidates) {
  ResolveResult out;

  // 1) Device probe
  auto di_opt = probe_device();
  if (!di_opt) {
    out.winner = ""; // signals failure to caller
    return out;
  }
  const DeviceInfo &di = *di_opt;
  const bool fp16_blk =
      fp16_blacklisted(di) && !env_true("GRETA_VK_FP16_ALLOW_UNSAFE");
  std::string fp16_reason;
  if (fp16_blk) {
    fp16_reason = "fp16_blacklisted device_key=" + di.key_string() +
                  " blacklist_path=" + fp16_blacklist_path().string() +
                  " (set GRETA_VK_FP16_ALLOW_UNSAFE=1 to override)";
  }

  out.device_key = di.key_string();
  out.bucket = make_bucket(args.M, args.N, args.K);

  // 3) Cache load
  Cache cache;
  cache.load();
  out.cache_path = cache.path();

  const bool no_write = env_flag_true("GRETA_VK_AUTOTUNE_NO_WRITE");

  // Tag device in cache if FP16 is blacklisted (auditable).
  if (fp16_blk && !no_write &&
      !env_true("GRETA_VK_FP16_BLACKLIST_TAG_NO_WRITE")) {
    cache.upsert(out.device_key, "meta:fp16_blacklist", "1");
    if (!fp16_reason.empty())
      cache.upsert(out.device_key, "meta:fp16_fallback_reason", fp16_reason);
    cache.save();
  }

  // 2) FORCE winner by ENV (pin)
  //    - semantics: if set, return immediately
  //    - by default we do NOT write cache when forcing (can be changed if
  //    needed)
  auto forced = env_get("GRETA_VK_AUTOTUNE_FORCE");
  if (forced && !forced->empty()) {
    // Optional safety: only accept if it exists in current candidate list.
    if (name_in_candidates(*forced, candidates)) {
      if (fp16_blk && is_fp16_name(*forced)) {
        // Ignore forced FP16 on blacklisted device unless explicitly allowed.
      } else {
        out.winner = *forced;
        out.force_winner = true;

        // If you WANT to persist forced choice, remove this NO_WRITE check.
        if (!no_write) {
          // We only persist if explicitly requested via an extra env.
          // (Keeps FORCE as a temporary override by default.)
          if (env_flag_true("GRETA_VK_AUTOTUNE_PERSIST_FORCE")) {
            cache.upsert(out.device_key, out.bucket, out.winner);
            cache.save();
          }
        }
        return out;
      }
    } else {
      // Unknown forced winner -> ignore and continue to normal autotune.
      // Caller can log if desired.
    }
  }

  if (env_flag_true("GRETA_VK_AUTOTUNE_CLEAR")) {
    cache.clear();
  }

  const bool retune = env_flag_true("GRETA_VK_AUTOTUNE_RETUNE");

  // 4) Cache hit path
  if (!retune) {
    auto w = cache.find_winner(out.device_key, out.bucket);
    if (w) {
      if (!(fp16_blk && is_fp16_name(*w))) {
        out.winner = *w;
        out.used_cache = true;
        return out;
      }
    }
  }

  out.retuned = true;

  // 5) Run candidates
  std::vector<CandidateResult> results =
      run_all_candidates(args, exe_dir, candidates, di);

  // 6) Pick best/second and apply "margin" rerun if needed
  auto best = pick_best(results);
  if (!best) {
    out.winner = "";
    out.results = std::move(results);
    return out;
  }
  auto second = pick_second_best(results);

  out.winner = best->name;
  out.winner_tflops = best->mean_tflops;
  out.second_tflops = second ? second->mean_tflops : 0.0;

  // Optional: enforce minimal viable performance (useful if parsing failed =>
  // 0.0)
  if (auto min_tf = env_get_double("GRETA_VK_AUTOTUNE_MIN_TFLOPS")) {
    if (out.winner_tflops < *min_tf) {
      // Treat as failure: no reliable winner
      out.winner = "";
      out.results = std::move(results);
      return out;
    }
  }

  const double margin =
      env_get_double("GRETA_VK_AUTOTUNE_MARGIN").value_or(1.03);
  const int rerun_iters =
      env_get_int("GRETA_VK_AUTOTUNE_RERUN_ITERS").value_or(60);

  if (second && second->mean_tflops > 0.0 && best->mean_tflops > 0.0) {
    const double ratio = best->mean_tflops / second->mean_tflops;
    if (ratio < margin) {
      // Rerun only the top2 with higher iters to stabilize selection.
      RunArgs args2 = args;
      args2.iters = std::max(args.iters, rerun_iters);

      std::vector<Candidate> top2;
      top2.reserve(2);

      // preserve ordering: best then second
      for (const auto &c : candidates) {
        if (c.name == best->name)
          top2.push_back(c);
      }
      for (const auto &c : candidates) {
        if (c.name == second->name)
          top2.push_back(c);
      }

      auto results2 = run_all_candidates(args2, exe_dir, top2, di);
      auto best2 = pick_best(results2);
      auto second2 = pick_second_best(results2);

      if (best2) {
        out.winner = best2->name;
        out.winner_tflops = best2->mean_tflops;
        out.second_tflops = second2 ? second2->mean_tflops : 0.0;
      }

      // merge outputs for debug visibility (optional)
      // keep original full list in out.results; rerun not included there by
      // default. If you want rerun details, you can append results2 with prefix
      // in name. For now keep it simple.
    }
  }

  // 7) Save cache
  if (!no_write && !out.winner.empty()) {
    cache.upsert(out.device_key, out.bucket, out.winner);
    cache.save();
  }

  out.results = std::move(results);
  return out;
}

} // namespace greta::vk_autotune
