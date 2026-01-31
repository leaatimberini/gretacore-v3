#include "gcore/rt/hip/backend.hpp"
#include "gcore/rt/hip/buffer.hpp"
#include "gcore/rt/hip/kernels/basic_kernels.hpp"
#include "gcore/rt/hip/stream.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

static void rmsnorm_ref(const float *x, float *y, const float *gamma, int rows,
                        int cols, double eps) {
  for (int r = 0; r < rows; r++) {
    const float *xr = x + size_t(r) * size_t(cols);
    float *yr = y + size_t(r) * size_t(cols);
    double ms = 0.0;
    for (int c = 0; c < cols; c++) {
      double v = double(xr[c]);
      ms += v * v;
    }
    ms /= double(cols);
    double inv = 1.0 / std::sqrt(ms + eps);
    for (int c = 0; c < cols; c++)
      yr[c] = float(double(xr[c]) * inv) * gamma[c];
  }
}

int main(int argc, char **argv) {
  int rows = 256;
  int cols = 1024;
  int iters = 20;
  float eps = 1e-5f;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--rows" && i + 1 < argc)
      rows = std::stoi(argv[++i]);
    if (arg == "--cols" && i + 1 < argc)
      cols = std::stoi(argv[++i]);
    if (arg == "--iters" && i + 1 < argc)
      iters = std::stoi(argv[++i]);
  }

  std::cout << "GRETA CORE Runtime Bench: hip_rmsnorm_bench\n";
  std::cout << "rows=" << rows << " cols=" << cols << " iters=" << iters
            << "\n";

  gcore::rt::hip::Backend backend;
  std::string err;
  if (!backend.init(&err)) {
    std::cerr << "Init failed: " << err << "\n";
    return 1;
  }
  backend.print_diagnostics(std::cout);

  size_t size_bytes = static_cast<size_t>(rows) * cols * sizeof(float);
  size_t gamma_bytes = static_cast<size_t>(cols) * sizeof(float);

  gcore::rt::hip::Buffer x_buf, y_buf, gamma_buf;
  if (!x_buf.allocate(size_bytes, gcore::rt::hip::BufferUsage::DeviceOnly,
                      &err) ||
      !y_buf.allocate(size_bytes, gcore::rt::hip::BufferUsage::DeviceOnly,
                      &err) ||
      !gamma_buf.allocate(gamma_bytes, gcore::rt::hip::BufferUsage::DeviceOnly,
                          &err)) {
    std::cerr << "Buffer alloc failed: " << err << "\n";
    return 1;
  }

  gcore::rt::hip::Stream stream;
  if (!stream.init(&err)) {
    std::cerr << "Stream init failed: " << err << "\n";
    return 1;
  }

  // Init data
  std::vector<float> x_host(rows * cols);
  std::vector<float> gamma_host(cols);

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &v : x_host)
    v = dist(rng);
  for (auto &v : gamma_host)
    v = 1.0f; // Simplified gamma

  if (!x_buf.copy_to_device(x_host.data(), size_bytes, &err) ||
      !gamma_buf.copy_to_device(gamma_host.data(), gamma_bytes, &err)) {
    std::cerr << "H2D failed: " << err << "\n";
    return 1;
  }

  // Warmup
  gcore::rt::hip::kernels::launch_rmsnorm_naive(
      stream.handle(), (float *)x_buf.data(), (float *)gamma_buf.data(),
      (float *)y_buf.data(), rows, cols, eps);
  stream.sync();

  std::vector<double> samples;
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    gcore::rt::hip::kernels::launch_rmsnorm_naive(
        stream.handle(), (float *)x_buf.data(), (float *)gamma_buf.data(),
        (float *)y_buf.data(), rows, cols, eps);
    stream.sync();
    auto t1 = std::chrono::high_resolution_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  double sum = 0;
  for (auto s : samples)
    sum += s;
  std::cout << "Mean time: " << sum / samples.size() << " ms\n";

  // Validate
  std::vector<float> y_host(rows * cols);
  if (!y_buf.copy_to_host(y_host.data(), size_bytes, &err)) {
    std::cerr << "D2H failed: " << err << "\n";
    return 1;
  }

  std::vector<float> y_ref(rows * cols);
  rmsnorm_ref(x_host.data(), y_ref.data(), gamma_host.data(), rows, cols, eps);

  double max_diff = 0.0;
  for (size_t i = 0; i < y_host.size(); ++i) {
    double d = std::abs(y_host[i] - y_ref[i]);
    if (d > max_diff)
      max_diff = d;
  }

  std::cout << "Max diff: " << max_diff << "\n";
  if (max_diff > 1e-4) {
    std::cout << "STATUS=FAILED\n";
    return 1;
  }

  std::cout << "STATUS=OK\n";
  return 0;
}
