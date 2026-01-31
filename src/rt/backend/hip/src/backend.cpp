#include "gcore/rt/hip/backend.hpp"

#include <chrono>
#include <hip/hip_runtime.h>
#include <iostream>

namespace gcore::rt::hip {

Backend::~Backend() { shutdown(); }

bool Backend::init(std::string *err) {
  int deviceCount = 0;
  hipError_t res = hipGetDeviceCount(&deviceCount);

  if (res != hipSuccess || deviceCount == 0) {
    if (err)
      *err = "No HIP devices found: " + std::string(hipGetErrorString(res));
    return false;
  }

  // For now, pick device 0.
  // In the future, we might want to pick the best GFX942 or non-VF device.
  device_id_ = 0;
  res = hipSetDevice(device_id_);
  if (res != hipSuccess) {
    if (err)
      *err = "Failed to set HIP device: " + std::string(hipGetErrorString(res));
    return false;
  }

  hipDeviceProp_t props;
  res = hipGetDeviceProperties(&props, device_id_);
  if (res != hipSuccess) {
    if (err)
      *err = "Failed to get HIP device properties: " +
             std::string(hipGetErrorString(res));
    return false;
  }

  info_.device_id = device_id_;
  info_.name = props.name;
  info_.gcn_arch_name = props.gcnArchName;
  info_.total_global_mem = props.totalGlobalMem;
  info_.multi_processor_count = props.multiProcessorCount;
  info_.warp_size = props.warpSize;
  info_.major = props.major;
  info_.minor = props.minor;

  return true;
}

void Backend::shutdown() {
  if (device_id_ >= 0) {
    (void)hipDeviceReset();
    device_id_ = -1;
  }
}

void Backend::print_diagnostics(std::ostream &os) const {
  os << "HIP Device Diagnostics:\n";
  os << "  Name: " << info_.name << "\n";
  os << "  GCN Arch: " << info_.gcn_arch_name << "\n";
  os << "  Total Global Mem: " << (info_.total_global_mem / (1024 * 1024))
     << " MB\n";
  os << "  Multi Processors: " << info_.multi_processor_count << "\n";
  os << "  Warp Size: " << info_.warp_size << "\n";
  os << "  Compute Capability: " << info_.major << "." << info_.minor << "\n";
}

bool Backend::sync(uint64_t *out_wait_ns, std::string *err) {
  auto t0 = std::chrono::steady_clock::now();
  hipError_t res = hipDeviceSynchronize();
  auto t1 = std::chrono::steady_clock::now();

  if (res != hipSuccess) {
    if (err)
      *err =
          "hipDeviceSynchronize failed: " + std::string(hipGetErrorString(res));
    return false;
  }

  if (out_wait_ns) {
    *out_wait_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  }
  return true;
}

} // namespace gcore::rt::hip
