#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifndef GRETA_HAS_HIP
#define GRETA_HAS_HIP 0
#endif

#if GRETA_HAS_HIP
#include <hip/hip_runtime.h>
#endif

static int parse_arg_int(const std::vector<std::string> &args,
                         const std::string &key, int def) {
  for (size_t i = 0; i + 1 < args.size(); i++) {
    if (args[i] == key)
      return std::stoi(args[i + 1]);
  }
  return def;
}

#if GRETA_HAS_HIP
__global__ void noop_kernel() { /* intentionally empty */ }
#endif

int main(int argc, char **argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; i++)
    args.emplace_back(argv[i]);

  const int iters = parse_arg_int(args, "--iters", 200000);

  std::cout << "GRETA CORE Platform Bench: hip_noop_launch\n";
  std::cout << "iters=" << iters << "\n";

#if !GRETA_HAS_HIP
  std::cerr << "HIP not enabled/built. Reconfigure with HIP available.\n";
  return 2;
#else
  int device = 0;
  hipError_t e = hipSetDevice(device);
  if (e != hipSuccess) {
    std::cerr << "hipSetDevice failed: " << hipGetErrorString(e) << "\n";
    return 1;
  }

  hipStream_t stream{};
  e = hipStreamCreate(&stream);
  if (e != hipSuccess) {
    std::cerr << "hipStreamCreate failed: " << hipGetErrorString(e) << "\n";
    return 1;
  }

  // Warmup launches
  for (int i = 0; i < 1000; i++) {
    hipLaunchKernelGGL(noop_kernel, dim3(1), dim3(1), 0, stream);
  }
  hipStreamSynchronize(stream);

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++) {
    hipLaunchKernelGGL(noop_kernel, dim3(1), dim3(1), 0, stream);
  }
  hipStreamSynchronize(stream);
  auto t1 = std::chrono::steady_clock::now();

  std::chrono::duration<double> dt = t1 - t0;
  const double total_us = dt.count() * 1e6;
  const double per_launch_us = total_us / static_cast<double>(iters);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "RESULT hip_noop_launch:\n";
  std::cout << "  total_sec=" << dt.count() << "\n";
  std::cout << "  per_launch_us=" << per_launch_us << "\n";

  hipStreamDestroy(stream);
  return 0;
#endif
}
