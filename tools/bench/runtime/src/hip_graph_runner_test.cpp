#include "gcore/rt/graph/hip_graph.hpp"
#include "gcore/rt/graph/hip_nodes.hpp"
#include "gcore/rt/hip/backend.hpp"
#include "gcore/rt/hip/buffer.hpp"
#include "gcore/rt/hip/stream.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace gcore::rt::graph;
using namespace gcore::rt::hip;

// Simple CPU Reference GEMM
void gemm_ref(const float *a, const float *b, float *c, int M, int N, int K,
              int lda, int ldb, int ldc) {
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

// Simple CPU Reference RMSNorm
void rmsnorm_ref(const float *x, float *y, const float *gamma, int rows,
                 int cols, float eps) {
  for (int r = 0; r < rows; r++) {
    const float *xr = x + r * cols;
    float *yr = y + r * cols;
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

int main() {
  std::cout << "GRETA CORE: hip_graph_runner_test\n";

  Backend backend;
  std::string err;
  if (!backend.init(&err)) {
    std::cerr << "Backend init failed: " << err << "\n";
    return 1;
  }
  backend.print_diagnostics(std::cout);

  // Dimensions
  uint32_t M = 256;
  uint32_t N = 256;
  uint32_t K = 256;
  float eps = 1e-5f;

  // 1. Allocate Buffers
  Buffer buf_a, buf_b, buf_c, buf_y, buf_gamma;
  size_t bytes_ab = M * K * sizeof(float); // Assuming square for simplicity
  size_t bytes_c = M * N * sizeof(float);
  size_t bytes_gamma = N * sizeof(float);

  if (!buf_a.allocate(bytes_ab, BufferUsage::DeviceOnly, &err) ||
      !buf_b.allocate(bytes_ab, BufferUsage::DeviceOnly, &err) ||
      !buf_c.allocate(bytes_c, BufferUsage::DeviceOnly, &err) ||
      !buf_y.allocate(bytes_c, BufferUsage::DeviceOnly, &err) ||
      !buf_gamma.allocate(bytes_gamma, BufferUsage::DeviceOnly, &err)) {
    std::cerr << "Buffer allocation failed: " << err << "\n";
    return 1;
  }

  // 2. Prepare Data
  std::vector<float> h_a(M * K);
  std::vector<float> h_b(K * N);
  std::vector<float> h_gamma(N, 1.0f);

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (auto &v : h_a)
    v = dist(rng);
  for (auto &v : h_b)
    v = dist(rng);

  if (!buf_a.copy_to_device(h_a.data(), bytes_ab, &err) ||
      !buf_b.copy_to_device(h_b.data(), bytes_ab, &err) ||
      !buf_gamma.copy_to_device(h_gamma.data(), bytes_gamma, &err)) {
    std::cerr << "H2D copy failed: " << err << "\n";
    return 1;
  }

  // 3. Build Graph
  HIPGraph graph;

  // Node 1: Fill C with zeros (just to test FillNode)
  graph.add_node(
      std::make_unique<HIPFillNode>((uint32_t *)buf_c.data(), 0, M * N));

  // Node 2: GEMM (C = A * B)
  graph.add_node(std::make_unique<HIPGemmNode>(
      (const float *)buf_a.data(), (const float *)buf_b.data(),
      (float *)buf_c.data(), M, N, K, K, N, N));

  // Node 3: RMSNorm (Y = RMSNorm(C))
  graph.add_node(std::make_unique<HIPRMSNormNode>(
      (const float *)buf_c.data(), (const float *)buf_gamma.data(),
      (float *)buf_y.data(), M, N, eps));

  // 4. Execute Graph
  Stream stream;
  if (!stream.init(&err)) {
    std::cerr << "Stream init failed: " << err << "\n";
    return 1;
  }

  std::cout << "Executing Graph...\n";
  if (!HIPGraphRunner::execute(stream.handle(), graph, &err)) {
    std::cerr << "Graph execution failed: " << err << "\n";
    return 1;
  }
  stream.sync();

  // 5. Validation
  std::cout << "Validating...\n";
  std::vector<float> h_c_out(M * N);
  std::vector<float> h_y_out(M * N);

  // We check Y (final output)
  if (!buf_y.copy_to_host(h_y_out.data(), bytes_c, &err)) {
    std::cerr << "D2H failed: " << err << "\n";
    return 1;
  }

  // CPU Ref
  std::vector<float> ref_c(M * N);
  std::vector<float> ref_y(M * N);
  gemm_ref(h_a.data(), h_b.data(), ref_c.data(), M, N, K, K, N, N);
  rmsnorm_ref(ref_c.data(), ref_y.data(), h_gamma.data(), M, N, eps);

  double max_diff = 0.0;
  for (size_t i = 0; i < h_y_out.size(); ++i) {
    double d = std::abs(h_y_out[i] - ref_y[i]);
    if (d > max_diff)
      max_diff = d;
  }

  std::cout << "Max diff: " << max_diff << "\n";
  if (max_diff > 1e-3) {
    std::cout << "STATUS=FAILED\n";
    return 1;
  }

  std::cout << "STATUS=OK\n";
  return 0;
}
