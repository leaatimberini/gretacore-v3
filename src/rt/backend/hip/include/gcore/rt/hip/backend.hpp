#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

namespace gcore::rt::hip {

struct DeviceInfo {
  int device_id = 0;
  std::string name;
  std::string gcn_arch_name;
  size_t total_global_mem = 0;
  int multi_processor_count = 0;
  int warp_size = 0;
  int major = 0;
  int minor = 0;
};

class Backend {
public:
  Backend() = default;
  ~Backend();

  // Initialize HIP: pick a device and setup context.
  bool init(std::string *err);
  void shutdown();

  int device_id() const { return device_id_; }
  const DeviceInfo &device_info() const { return info_; }

  // Print extended diagnostics (HIP capabilities).
  void print_diagnostics(std::ostream &os) const;

  // Simple synchronization for smoke/latency measurement.
  bool sync(uint64_t *out_wait_ns, std::string *err);

private:
  int device_id_ = -1;
  DeviceInfo info_{};
};

} // namespace gcore::rt::hip
