#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace greta::vk_autotune {

// ------------------------
// Device probe (cache key)
// ------------------------
struct DeviceInfo {
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
  std::string device_name;
  std::string driver_name; // best-effort (may be empty)

  // Subgroup size control (best-effort; 0 if unknown)
  uint32_t reported_subgroup_size = 0;
  uint32_t min_subgroup_size = 0;
  uint32_t max_subgroup_size = 0;

  // FP16 capability (best-effort)
  bool fp16_supported = false;
  std::string fp16_reason;

  // Safety/blacklist
  bool gpu_blacklisted = false;
  std::string blacklist_reason;

  std::string key_string() const; // stable cache key
};

// Probe Vulkan for the preferred physical device (skipping llvmpipe if
// possible). Returns nullopt if Vulkan instance/device enumeration fails.
std::optional<DeviceInfo> probe_device();

// Check if this device is blacklisted for FP16 (healthcheck failed before).
bool fp16_blacklisted(const DeviceInfo &di);
// Human-readable reason for FP16 blacklist (includes override hint).
std::string fp16_blacklist_reason(const DeviceInfo &di);
// Path to the FP16 blacklist file.
std::string fp16_blacklist_path_string();
// Ensure FP16 blacklist metadata is written to the autotune cache.
void tag_fp16_blacklist_cache(const DeviceInfo &di,
                              const std::string &reason);

// ------------------------
// Cache
// ------------------------
struct CacheEntry {
  std::string device_key;
  std::string bucket; // e.g. "M1024_N1024_K1024"
  std::string winner; // e.g. "tiled_vec2_32x8"
};

// Simple JSON-ish cache (no external deps).
class Cache {
public:
  void load();       // safe to call multiple times
  void save() const; // writes to disk (best-effort)

  std::optional<std::string> find_winner(const std::string &device_key,
                                         const std::string &bucket) const;

  void upsert(const std::string &device_key, const std::string &bucket,
              const std::string &winner);

  void clear();             // clears in-memory entries
  std::string path() const; // resolved on first call

private:
  mutable std::string path_;
  std::vector<CacheEntry> entries_;
};

// ------------------------
// Candidate execution
// ------------------------
struct CandidateResult {
  std::string name; // stable ID (e.g. "tiled_vec2_db")
  double mean_tflops = 0.0;
  std::string raw_output;
  int exit_code = -1;
};

struct RunArgs {
  uint32_t M = 1024, N = 1024, K = 1024;
  int iters = 30;
  int batch = 100;
};

struct Candidate {
  std::string name; // stable ID stored in cache
  std::string exe;  // executable name (relative to exe_dir)
  // If true, this candidate is only valid if device supports subgroup size 32.
  bool requires_subgroup32 = false;
};

// Execute a command capturing stdout/stderr (merged). Uses /bin/sh -lc.
CandidateResult run_candidate_command(const std::string &candidate_name,
                                      const std::string &command);

// Parse "mean_TFLOPs=" from GRETA bench output. Returns nullopt if not found.
std::optional<double> parse_mean_tflops(const std::string &output);

// Build a shape bucket string used for caching.
std::string make_bucket(uint32_t M, uint32_t N, uint32_t K);

// Env helpers
std::optional<std::string> env_get(const char *name);
bool env_flag_true(const char *name);
std::optional<double> env_get_double(const char *name);
std::optional<int> env_get_int(const char *name);

// Pick best candidate by mean_tflops (ties -> first).
std::optional<CandidateResult>
pick_best(const std::vector<CandidateResult> &rs);

// Pick second best candidate (by mean_tflops). Returns nullopt if <2 entries.
std::optional<CandidateResult>
pick_second_best(const std::vector<CandidateResult> &rs);

// ------------------------
// High-level API (CUDA-like)
// ------------------------
struct ResolveResult {
  std::string device_key;
  std::string bucket;
  std::string winner;
  std::string cache_path;

  bool used_cache = false;
  bool force_winner = false;
  bool retuned = false;

  // Optional: top stats (useful for debugging/telemetry)
  double winner_tflops = 0.0;
  double second_tflops = 0.0;

  // Raw per-candidate results (only filled when tuning happened)
  std::vector<CandidateResult> results;
};

// Main entry point: resolves best candidate for given args + device.
// - exe_dir: directory where candidate executables live (e.g. "./build").
// - candidates: list of candidates to evaluate (names must be stable).
//
// Environment controls:
// - GRETA_VK_AUTOTUNE_FORCE=<name>   : force winner to this name (no tuning)
// - GRETA_VK_AUTOTUNE_RETUNE=1       : ignore cache and tune
// - GRETA_VK_AUTOTUNE_CLEAR=1        : clear in-memory cache after load()
// - GRETA_VK_AUTOTUNE_NO_WRITE=1     : do not save cache file
// - GRETA_VK_AUTOTUNE_MARGIN=1.03    : if top2 within margin => rerun more
// iters
// - GRETA_VK_AUTOTUNE_RERUN_ITERS=60 : iters used for rerun
// - GRETA_VK_AUTOTUNE_MIN_TFLOPS=... : if best < threshold, treat as failure
// (optional)
ResolveResult resolve_winner(const RunArgs &args, const std::string &exe_dir,
                             const std::vector<Candidate> &candidates);

} // namespace greta::vk_autotune
