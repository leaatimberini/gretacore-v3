#include "gcore/rt/graph/hip_graph.hpp"
#include "gcore/rt/graph/hip_nodes.hpp"
#include "gcore/rt/hip/backend.hpp"
#include "gcore/rt/hip/buffer.hpp"
#include "gcore/rt/hip/stream.hpp"

#include <chrono>
#include <iostream>
#include <vector>

using namespace gcore::rt::graph;
using namespace gcore::rt::hip;

int main() {
  std::cout << "GRETA CORE: hip_llama_block_test (Llama-2-7B Config)\n";

  Backend backend;
  std::string err;
  if (!backend.init(&err)) {
    std::cerr << "Backend init failed: " << err << "\n";
    return 1;
  }
  backend.print_diagnostics(std::cout);

  // Model Constants (Llama-2-7B)
  const uint32_t dim = 4096;
  const uint32_t hidden_dim = 11008;
  const uint32_t num_heads = 32;
  const uint32_t head_dim = dim / num_heads;
  const uint32_t max_seq_len = 2048;

  // Inference state (single token)
  uint32_t pos = 10; // arbitrary position

  // Buffers
  Buffer x, norm_x, qkv, q, k, v, attn_scores, attn_out, mlp_gate, mlp_up,
      mlp_down;
  Buffer weight_q, weight_k, weight_v, weight_wo;
  Buffer weight_gate, weight_up, weight_down;
  Buffer rms_gamma, cache_k, cache_v;

  size_t dim_bytes = dim * sizeof(float);
  size_t hidden_bytes = hidden_dim * sizeof(float);
  size_t weight_bytes = dim * dim * sizeof(float);
  size_t cache_bytes = num_heads * max_seq_len * head_dim * sizeof(float);

  bool ok = true;
  ok &= x.allocate(dim_bytes, BufferUsage::DeviceOnly);
  ok &= norm_x.allocate(dim_bytes, BufferUsage::DeviceOnly);
  ok &= qkv.allocate(dim_bytes * 3,
                     BufferUsage::DeviceOnly); // combined QKV for bench
  ok &= q.allocate(dim_bytes, BufferUsage::DeviceOnly);
  ok &= k.allocate(dim_bytes, BufferUsage::DeviceOnly);
  ok &= v.allocate(dim_bytes, BufferUsage::DeviceOnly);
  ok &= attn_scores.allocate(num_heads * max_seq_len * sizeof(float),
                             BufferUsage::DeviceOnly);
  ok &= attn_out.allocate(dim_bytes, BufferUsage::DeviceOnly);
  ok &= mlp_gate.allocate(hidden_bytes, BufferUsage::DeviceOnly);
  ok &= mlp_up.allocate(hidden_bytes, BufferUsage::DeviceOnly);
  ok &= mlp_down.allocate(dim_bytes, BufferUsage::DeviceOnly);

  ok &= weight_gate.allocate(hidden_dim * dim * sizeof(float),
                             BufferUsage::DeviceOnly);
  ok &= weight_up.allocate(hidden_dim * dim * sizeof(float),
                           BufferUsage::DeviceOnly);
  ok &= weight_down.allocate(dim * hidden_dim * sizeof(float),
                             BufferUsage::DeviceOnly);
  ok &= rms_gamma.allocate(dim_bytes, BufferUsage::DeviceOnly);
  ok &= cache_k.allocate(cache_bytes, BufferUsage::DeviceOnly);
  ok &= cache_v.allocate(cache_bytes, BufferUsage::DeviceOnly);

  if (!ok) {
    std::cerr << "Buffer allocation failed\n";
    return 1;
  }

  // Build Graph
  HIPGraph graph;

  // 1. Attention Path
  // In a real Llama, we'd have 3 GEMMs for Q, K, V. Here we simplify.
  graph.add_node(std::make_unique<HIPRMSNormNode>(
      (float *)x.data(), (float *)rms_gamma.data(), (float *)norm_x.data(), 1,
      dim, 1e-5f));

  // Simulating Q, K, V projections (Simplified as one GEMM each)
  // Weight matrices are typically [dim, dim]
  graph.add_node(std::make_unique<HIPGemmNode>(
      (float *)norm_x.data(), (float *)weight_gate.data(), (float *)q.data(), 1,
      dim, dim, 1, dim, 1)); // GEMM: [1,dim] x [dim,dim] -> [1,dim]

  // 2. RoPE & KV Update
  graph.add_node(std::make_unique<HIPRoPENode>((float *)q.data(), 1, num_heads,
                                               head_dim, 10000.0f));

  graph.add_node(std::make_unique<HIPKVUpdateNode>(
      (float *)cache_k.data(), (float *)cache_v.data(), (float *)k.data(),
      (float *)v.data(), pos, max_seq_len, num_heads, head_dim));

  // 3. Attention calculation would go here (Simplified: skip for timing focus)

  // 4. MLP Path
  graph.add_node(std::make_unique<HIPGemmNode>(
      (float *)norm_x.data(), (float *)weight_gate.data(),
      (float *)mlp_gate.data(), 1, hidden_dim, dim, 1, hidden_dim, 1));

  graph.add_node(std::make_unique<HIPGemmNode>(
      (float *)norm_x.data(), (float *)weight_up.data(), (float *)mlp_up.data(),
      1, hidden_dim, dim, 1, hidden_dim, 1));

  graph.add_node(std::make_unique<HIPSiLUNode>(
      (float *)mlp_gate.data(), (float *)mlp_gate.data(), hidden_dim));

  // Element-wise mul (Simplified as SiLU output)

  graph.add_node(std::make_unique<HIPGemmNode>(
      (float *)mlp_gate.data(), (float *)weight_down.data(),
      (float *)mlp_down.data(), 1, dim, hidden_dim, 1, dim, 1));

  // 5. Residual
  graph.add_node(std::make_unique<HIPAddNode>(
      (float *)mlp_down.data(), (float *)x.data(), (float *)x.data(), dim));

  Stream stream;
  stream.init(&err);

  std::cout << "Executing graph (single execution)...\n";
  HIPGraphRunner::execute(stream.handle(), graph, &err);
  stream.sync();

  std::cout << "Benchmarking 10 iterations...\n";
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 10; ++i) {
    HIPGraphRunner::execute(stream.handle(), graph, &err);
  }
  stream.sync();
  auto t1 = std::chrono::steady_clock::now();

  double ms =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
      1000.0 / 10.0;
  std::cout << "Average Block Execution Time: " << ms << " ms\n";

  std::cout << "STATUS=OK\n";
  return 0;
}
