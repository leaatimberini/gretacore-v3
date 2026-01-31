#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef GRETA_HAS_HIP
#define GRETA_HAS_HIP 0
#endif

#if GRETA_HAS_HIP
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#endif

static int parse_arg_int(const std::vector<std::string> &args,
                         const std::string &key, int def) {
  for (size_t i = 0; i + 1 < args.size(); i++) {
    if (args[i] == key)
      return std::stoi(args[i + 1]);
  }
  return def;
}

int main(int argc, char **argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; i++)
    args.emplace_back(argv[i]);

  const int m = parse_arg_int(args, "--m", 1024);
  const int n = parse_arg_int(args, "--n", 1024);
  const int k = parse_arg_int(args, "--k", 1024);
  const int iters = parse_arg_int(args, "--iters", 20);
  const int warmup = parse_arg_int(args, "--warmup", 5);
  const int check = parse_arg_int(args, "--check", 0);
  const int check_samples = parse_arg_int(args, "--check-samples", 8);

  std::cout << "GRETA CORE Platform Bench: hip_gemm\n";
  std::cout << "m=" << m << " n=" << n << " k=" << k << " iters=" << iters
            << " warmup=" << warmup << " check=" << check
            << " check_samples=" << check_samples << "\n";

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

  const size_t bytes_a = static_cast<size_t>(m) * k * sizeof(float);
  const size_t bytes_b = static_cast<size_t>(k) * n * sizeof(float);
  const size_t bytes_c = static_cast<size_t>(m) * n * sizeof(float);

  std::vector<float> ha(static_cast<size_t>(m) * k);
  std::vector<float> hb(static_cast<size_t>(k) * n);
  std::vector<float> hc(static_cast<size_t>(m) * n);

  for (size_t i = 0; i < ha.size(); i++)
    ha[i] = static_cast<float>((i % 251) - 125) * 0.01f;
  for (size_t i = 0; i < hb.size(); i++)
    hb[i] = static_cast<float>((i % 197) - 98) * 0.02f;

  float *da = nullptr;
  float *db = nullptr;
  float *dc = nullptr;
  if (hipMalloc(&da, bytes_a) != hipSuccess ||
      hipMalloc(&db, bytes_b) != hipSuccess ||
      hipMalloc(&dc, bytes_c) != hipSuccess) {
    std::cerr << "hipMalloc failed\n";
    hipFree(da);
    hipFree(db);
    hipFree(dc);
    return 1;
  }

  if (hipMemcpy(da, ha.data(), bytes_a, hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(db, hb.data(), bytes_b, hipMemcpyHostToDevice) != hipSuccess) {
    std::cerr << "hipMemcpy H2D failed\n";
    hipFree(da);
    hipFree(db);
    hipFree(dc);
    return 1;
  }

  hipblasHandle_t handle{};
  if (hipblasCreate(&handle) != HIPBLAS_STATUS_SUCCESS) {
    std::cerr << "hipblasCreate failed\n";
    hipFree(da);
    hipFree(db);
    hipFree(dc);
    return 1;
  }

  hipStream_t stream{};
  hipStreamCreate(&stream);
  hipblasSetStream(handle, stream);
  if (hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST) !=
      HIPBLAS_STATUS_SUCCESS) {
    std::cerr << "hipblasSetPointerMode failed\n";
    hipblasDestroy(handle);
    hipFree(da);
    hipFree(db);
    hipFree(dc);
    return 1;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (int i = 0; i < warmup; i++) {
    hipblasStatus_t st =
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, da, m,
                     db, k, &beta, dc, m);
    if (st != HIPBLAS_STATUS_SUCCESS) {
      std::cerr << "hipblasSgemm warmup failed: " << int(st) << "\n";
      hipblasDestroy(handle);
      hipFree(da);
      hipFree(db);
      hipFree(dc);
      return 1;
    }
  }
  hipStreamSynchronize(stream);

  hipEvent_t ev_start{}, ev_stop{};
  hipEventCreate(&ev_start);
  hipEventCreate(&ev_stop);

  hipEventRecord(ev_start, stream);
  for (int i = 0; i < iters; i++) {
    hipblasStatus_t st =
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, da, m,
                     db, k, &beta, dc, m);
    if (st != HIPBLAS_STATUS_SUCCESS) {
      std::cerr << "hipblasSgemm failed: " << int(st) << "\n";
      hipblasDestroy(handle);
      hipFree(da);
      hipFree(db);
      hipFree(dc);
      return 1;
    }
  }
  hipEventRecord(ev_stop, stream);
  hipEventSynchronize(ev_stop);

  float kernel_ms = 0.0f;
  hipEventElapsedTime(&kernel_ms, ev_start, ev_stop);

  const double ops = 2.0 * double(m) * double(n) * double(k) * double(iters);
  const double tflops = (ops / 1.0e12) / (kernel_ms / 1.0e3);

  double max_abs_err_col = 0.0;
  double max_abs_err_row = 0.0;
  if (check) {
    if (hipMemcpy(hc.data(), dc, bytes_c, hipMemcpyDeviceToHost) !=
        hipSuccess) {
      std::cerr << "hipMemcpy D2H failed\n";
    } else {
      std::mt19937 rng(12345);
      std::uniform_int_distribution<int> dist_m(0, m - 1);
      std::uniform_int_distribution<int> dist_n(0, n - 1);
      const int samples = std::max(1, check_samples);
      for (int s = 0; s < samples; s++) {
        int mi = dist_m(rng);
        int ni = dist_n(rng);
        double acc_col = 0.0;
        double acc_row = 0.0;
        for (int ki = 0; ki < k; ki++) {
          acc_col += double(ha[static_cast<size_t>(ki) * m + mi]) *
                     double(hb[static_cast<size_t>(ni) * k + ki]);
          acc_row += double(ha[static_cast<size_t>(mi) * k + ki]) *
                     double(hb[static_cast<size_t>(ki) * n + ni]);
        }
        double err_col = std::abs(
            acc_col - double(hc[static_cast<size_t>(ni) * m + mi]));
        double err_row = std::abs(
            acc_row - double(hc[static_cast<size_t>(mi) * n + ni]));
        if (err_col > max_abs_err_col)
          max_abs_err_col = err_col;
        if (err_row > max_abs_err_row)
          max_abs_err_row = err_row;
      }
    }
  }

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "RESULT hip_gemm:\n";
  std::cout << "  kernel_ms_total=" << kernel_ms << "\n";
  std::cout << "  kernel_ms_avg=" << (kernel_ms / double(iters)) << "\n";
  std::cout << "  tflops=" << tflops << "\n";
  if (check) {
    std::cout << "  max_abs_err_col=" << max_abs_err_col << "\n";
    std::cout << "  max_abs_err_row=" << max_abs_err_row << "\n";
  }

  hipEventDestroy(ev_start);
  hipEventDestroy(ev_stop);
  hipStreamDestroy(stream);
  hipblasDestroy(handle);
  hipFree(da);
  hipFree(db);
  hipFree(dc);
  return 0;
#endif
}
