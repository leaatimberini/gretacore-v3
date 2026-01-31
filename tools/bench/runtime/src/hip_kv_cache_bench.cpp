#include "gcore/rt/graph/hip_graph.hpp"
#include "gcore/rt/graph/hip_nodes.hpp"
#include "gcore/rt/hip/backend.hpp"
#include "gcore/rt/hip/buffer.hpp"
#include "gcore/rt/hip/stream.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace gcore::rt::graph;
using namespace gcore::rt::hip;

int main() {
  std::cout << "GRETA CORE: hip_kv_cache_bench\n";

  Backend backend;
  std::string err;
  if (!backend.init(&err)) {
    std::cerr << "Backend init failed: " << err << "\n";
    return 1;
  }
  backend.print_diagnostics(std::cout);

  uint32_t num_heads = 32;
  uint32_t head_dim = 128;
  uint32_t max_seq_len = 512;
  uint32_t test_steps = 5;

  size_t cache_bytes = num_heads * max_seq_len * head_dim * sizeof(float);
  size_t new_bytes = num_heads * head_dim * sizeof(float);

  Buffer buf_cache_k, buf_cache_v;
  Buffer buf_new_k, buf_new_v;

  if (!buf_cache_k.allocate(cache_bytes, BufferUsage::DeviceOnly, &err) ||
      !buf_cache_v.allocate(cache_bytes, BufferUsage::DeviceOnly, &err) ||
      !buf_new_k.allocate(new_bytes, BufferUsage::DeviceOnly, &err) ||
      !buf_new_v.allocate(new_bytes, BufferUsage::DeviceOnly, &err)) {
    std::cerr << "Allocation failed: " << err << "\n";
    return 1;
  }

  // Clear caches
  std::vector<float> zero(num_heads * max_seq_len * head_dim, 0.0f);
  buf_cache_k.copy_to_device(zero.data(), cache_bytes, &err);
  buf_cache_v.copy_to_device(zero.data(), cache_bytes, &err);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<std::vector<float>> history_k, history_v;

  Stream stream;
  stream.init(&err);

  for (uint32_t step = 0; step < test_steps; ++step) {
    std::cout << "Step " << step << "...\n";

    std::vector<float> h_new_k(num_heads * head_dim);
    std::vector<float> h_new_v(num_heads * head_dim);
    for (auto &v : h_new_k)
      v = dist(rng);
    for (auto &v : h_new_v)
      v = dist(rng);

    history_k.push_back(h_new_k);
    history_v.push_back(h_new_v);

    buf_new_k.copy_to_device(h_new_k.data(), new_bytes, &err);
    buf_new_v.copy_to_device(h_new_v.data(), new_bytes, &err);

    // We create a one-node graph for each step to simulate dynamic 'pos'
    HIPGraph graph;
    graph.add_node(std::make_unique<HIPKVUpdateNode>(
        (float *)buf_cache_k.data(), (float *)buf_cache_v.data(),
        (const float *)buf_new_k.data(), (const float *)buf_new_v.data(), step,
        max_seq_len, num_heads, head_dim));

    HIPGraphRunner::execute(stream.handle(), graph, &err);
    stream.sync();
  }

  // Final Validation
  std::vector<float> final_cache_k(num_heads * max_seq_len * head_dim);
  std::vector<float> final_cache_v(num_heads * max_seq_len * head_dim);
  buf_cache_k.copy_to_host(final_cache_k.data(), cache_bytes, &err);
  buf_cache_v.copy_to_host(final_cache_v.data(), cache_bytes, &err);

  double max_diff = 0.0;
  for (uint32_t step = 0; step < test_steps; ++step) {
    for (uint32_t h = 0; h < num_heads; ++h) {
      for (uint32_t d = 0; d < head_dim; ++d) {
        uint32_t cache_idx = h * (max_seq_len * head_dim) + step * head_dim + d;
        uint32_t hist_idx = h * head_dim + d;

        double dk =
            std::abs(final_cache_k[cache_idx] - history_k[step][hist_idx]);
        double dv =
            std::abs(final_cache_v[cache_idx] - history_v[step][hist_idx]);
        max_diff = std::max({max_diff, dk, dv});
      }
    }
  }

  std::cout << "KV-Cache Max Diff: " << max_diff << "\n";

  if (max_diff > 1e-6) {
    std::cout << "STATUS=FAILED\n";
    return 1;
  }

  std::cout << "STATUS=OK\n";
  return 0;
}
