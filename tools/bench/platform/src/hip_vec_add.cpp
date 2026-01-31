#include <chrono>
#include <cmath>
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
__global__ void vec_add_kernel(const float *x, const float *y, float *z,
                               int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    z[idx] = x[idx] + y[idx];
}
#endif

int main(int argc, char **argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; i++)
    args.emplace_back(argv[i]);

  const int n = parse_arg_int(args, "--n", 1 << 22);
  const int iters = parse_arg_int(args, "--iters", 200);
  const int warmup = parse_arg_int(args, "--warmup", 10);

  std::cout << "GRETA CORE Platform Bench: hip_vec_add\n";
  std::cout << "n=" << n << " iters=" << iters << " warmup=" << warmup
            << "\n";

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

  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, device) == hipSuccess) {
    std::cout << "device=" << prop.name << "\n";
    std::cout << "gcn_arch=" << prop.gcnArchName << "\n";
  }

  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  std::vector<float> hx(n), hy(n), hz(n);
  for (int i = 0; i < n; i++) {
    hx[i] = float((i % 251) - 125) * 0.01f;
    hy[i] = float((i % 197) - 98) * 0.02f;
  }

  float *dx = nullptr;
  float *dy = nullptr;
  float *dz = nullptr;
  if (hipMalloc(&dx, bytes) != hipSuccess ||
      hipMalloc(&dy, bytes) != hipSuccess ||
      hipMalloc(&dz, bytes) != hipSuccess) {
    std::cerr << "hipMalloc failed\n";
    hipFree(dx);
    hipFree(dy);
    hipFree(dz);
    return 1;
  }

  if (hipMemcpy(dx, hx.data(), bytes, hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(dy, hy.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
    std::cerr << "hipMemcpy H2D failed\n";
    hipFree(dx);
    hipFree(dy);
    hipFree(dz);
    return 1;
  }

  const int block = 256;
  const int grid = (n + block - 1) / block;

  hipStream_t stream{};
  hipStreamCreate(&stream);

  for (int i = 0; i < warmup; i++) {
    hipLaunchKernelGGL(vec_add_kernel, dim3(grid), dim3(block), 0, stream, dx,
                       dy, dz, n);
  }
  hipStreamSynchronize(stream);

  hipEvent_t ev_start{}, ev_stop{};
  hipEventCreate(&ev_start);
  hipEventCreate(&ev_stop);

  hipEventRecord(ev_start, stream);
  for (int i = 0; i < iters; i++) {
    hipLaunchKernelGGL(vec_add_kernel, dim3(grid), dim3(block), 0, stream, dx,
                       dy, dz, n);
  }
  hipEventRecord(ev_stop, stream);
  hipEventSynchronize(ev_stop);

  float kernel_ms = 0.0f;
  hipEventElapsedTime(&kernel_ms, ev_start, ev_stop);

  auto t0 = std::chrono::steady_clock::now();
  if (hipMemcpy(hz.data(), dz, bytes, hipMemcpyDeviceToHost) != hipSuccess) {
    std::cerr << "hipMemcpy D2H failed\n";
  }
  auto t1 = std::chrono::steady_clock::now();

  double max_abs = 0.0;
  for (int i = 0; i < n; i++) {
    double ref = double(hx[i]) + double(hy[i]);
    double err = std::abs(ref - double(hz[i]));
    if (err > max_abs)
      max_abs = err;
  }

  const double bytes_per_iter = double(n) * sizeof(float) * 3.0;
  const double gbps =
      (bytes_per_iter * double(iters)) / (kernel_ms * 1.0e6);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "RESULT hip_vec_add:\n";
  std::cout << "  kernel_ms_total=" << kernel_ms << "\n";
  std::cout << "  kernel_ms_avg=" << (kernel_ms / double(iters)) << "\n";
  std::cout << "  kernel_gbps=" << gbps << "\n";
  std::cout << "  d2h_ms="
            << std::chrono::duration<double, std::milli>(t1 - t0).count()
            << "\n";
  std::cout << "  max_abs_err=" << max_abs << "\n";

  hipEventDestroy(ev_start);
  hipEventDestroy(ev_stop);
  hipStreamDestroy(stream);
  hipFree(dx);
  hipFree(dy);
  hipFree(dz);
  return 0;
#endif
}
