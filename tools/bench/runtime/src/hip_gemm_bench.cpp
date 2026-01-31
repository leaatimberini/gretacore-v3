#include "gcore/rt/hip/backend.hpp"
#include "gcore/rt/hip/buffer.hpp"
#include "gcore/rt/hip/kernels/gemm_kernels.hpp"
#include "gcore/rt/hip/stream.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

static void gemm_ref(const float *a, const float *b, float *c, int M, int N,
                     int K, int lda, int ldb, int ldc) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += a[m * lda + k] * b[k * ldb + n];
      }
      c[m * ldc + n] = acc;
    }
  }
}

int main(int argc, char **argv) {
  int M = 1024;
  int N = 1024;
  int K = 1024;
  int iters = 10;
  bool run_mfma = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--M" && i + 1 < argc)
      M = std::stoi(argv[++i]);
    if (arg == "--N" && i + 1 < argc)
      N = std::stoi(argv[++i]);
    if (arg == "--K" && i + 1 < argc)
      K = std::stoi(argv[++i]);
    if (arg == "--iters" && i + 1 < argc)
      iters = std::stoi(argv[++i]);
    if (arg == "--mfma")
      run_mfma = true;
  }

  std::cout << "GRETA CORE Runtime Bench: hip_gemm_bench\n";
  std::cout << "M=" << M << " N=" << N << " K=" << K << " iters=" << iters
            << "\n";
  if (run_mfma)
    std::cout << "Mode: MFMA (Matrix Cores 16x16x4)\n";
  else
    std::cout << "Mode: Tiled (Legacy)\n";

  gcore::rt::hip::Backend backend;
  std::string err;
  if (!backend.init(&err)) {
    std::cerr << "Init failed: " << err << "\n";
    return 1;
  }
  backend.print_diagnostics(std::cout);

  // Strides (Row Major)
  int lda = K;
  int ldb = N;
  int ldc = N;

  size_t bytes_a = static_cast<size_t>(M) * K * sizeof(float);
  size_t bytes_b = static_cast<size_t>(K) * N * sizeof(float);
  size_t bytes_c = static_cast<size_t>(M) * N * sizeof(float);

  gcore::rt::hip::Buffer buf_a, buf_b, buf_c;
  if (!buf_a.allocate(bytes_a, gcore::rt::hip::BufferUsage::DeviceOnly, &err) ||
      !buf_b.allocate(bytes_b, gcore::rt::hip::BufferUsage::DeviceOnly, &err) ||
      !buf_c.allocate(bytes_c, gcore::rt::hip::BufferUsage::DeviceOnly, &err)) {
    std::cerr << "Buffer alloc failed: " << err << "\n";
    return 1;
  }

  gcore::rt::hip::Stream stream;
  if (!stream.init(&err)) {
    std::cerr << "Stream init failed: " << err << "\n";
    return 1;
  }

  // Host data
  std::vector<float> h_a(M * K);
  std::vector<float> h_b(K * N);
  std::vector<float> h_c(M * N);

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (auto &v : h_a)
    v = dist(rng);
  for (auto &v : h_b)
    v = dist(rng);

  if (!buf_a.copy_to_device(h_a.data(), bytes_a, &err) ||
      !buf_b.copy_to_device(h_b.data(), bytes_b, &err)) {
    std::cerr << "H2D failed: " << err << "\n";
    return 1;
  }

  // Warmup
  if (run_mfma) {
    gcore::rt::hip::kernels::launch_gemm_mfma_f32(
        stream.handle(), (const float *)buf_a.data(),
        (const float *)buf_b.data(), (float *)buf_c.data(), M, N, K, lda, ldb,
        ldc);
  } else {
    gcore::rt::hip::kernels::launch_gemm_tiled_f32(
        stream.handle(), (const float *)buf_a.data(),
        (const float *)buf_b.data(), (float *)buf_c.data(), M, N, K, lda, ldb,
        ldc);
  }
  stream.sync();

  // Bench
  std::vector<double> samples;
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (run_mfma) {
      gcore::rt::hip::kernels::launch_gemm_mfma_f32(
          stream.handle(), (const float *)buf_a.data(),
          (const float *)buf_b.data(), (float *)buf_c.data(), M, N, K, lda, ldb,
          ldc);
    } else {
      gcore::rt::hip::kernels::launch_gemm_tiled_f32(
          stream.handle(), (const float *)buf_a.data(),
          (const float *)buf_b.data(), (float *)buf_c.data(), M, N, K, lda, ldb,
          ldc);
    }

    stream.sync();
    auto t1 = std::chrono::high_resolution_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  double sum = 0;
  for (auto s : samples)
    sum += s;
  double avg_ms = sum / samples.size();
  double gflops = (2.0 * M * N * K) /
                  (avg_ms * 1e6); // 1e-3 (ms->s) * 1e9 (FLOPS->GFLOPS) = 1e6

  std::cout << "Mean time: " << avg_ms << " ms\n";
  std::cout << "Perf: " << gflops << " GFLOPS\n";

  // Validate (on small sub-region or full if small)
  if (M <= 1024 && N <= 1024 && K <= 1024) {
    std::cout << "Validating result...\n";
    if (!buf_c.copy_to_host(h_c.data(), bytes_c, &err)) {
      std::cerr << "D2H failed: " << err << "\n";
      return 1;
    }

    std::vector<float> ref_c(M * N);
    gemm_ref(h_a.data(), h_b.data(), ref_c.data(), M, N, K, lda, ldb, ldc);

    double max_diff = 0.0;
    for (size_t i = 0; i < h_c.size(); ++i) {
      double d = std::abs(h_c[i] - ref_c[i]);
      if (d > max_diff)
        max_diff = d;
    }
    std::cout << "Max diff: " << max_diff << "\n";
    if (max_diff > 1e-2) { // Slightly looser tolerance for MFMA which might
                           // have different accumulation order
      std::cout << "STATUS=FAILED\n";
      return 1;
    }
  } else {
    std::cout << "Skipping validation (matrix too large for CPU ref)\n";
  }

  std::cout << "STATUS=OK\n";
  return 0;
}
