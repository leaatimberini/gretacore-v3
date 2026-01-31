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

// CPU Reference for Causal Mask
void causal_mask_ref(float *data, int seq_len, float mask_val) {
  for (int r = 0; r < seq_len; r++) {
    for (int c = r + 1; c < seq_len; c++) {
      data[r * seq_len + c] = mask_val;
    }
  }
}

// CPU Reference for RoPE
void rope_ref(float *x, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim,
              float base) {
  for (uint32_t p = 0; p < seq_len; ++p) {
    for (uint32_t h = 0; h < num_heads; ++h) {
      for (uint32_t i = 0; i < head_dim / 2; ++i) {
        float theta =
            (float)p * std::pow(base, -2.0f * (float)i / (float)head_dim);
        float cos_val = std::cos(theta);
        float sin_val = std::sin(theta);

        uint32_t idx0 = p * (num_heads * head_dim) + h * head_dim + 2 * i;
        uint32_t idx1 = idx0 + 1;

        float v0 = x[idx0];
        float v1 = x[idx1];

        x[idx0] = v0 * cos_val - v1 * sin_val;
        x[idx1] = v0 * sin_val + v1 * cos_val;
      }
    }
  }
}

int main() {
  std::cout << "GRETA CORE: hip_attention_bench (RoPE + Mask)\n";

  Backend backend;
  std::string err;
  if (!backend.init(&err)) {
    std::cerr << "Backend init failed: " << err << "\n";
    return 1;
  }
  backend.print_diagnostics(std::cout);

  uint32_t seq_len = 128;
  uint32_t num_heads = 12;
  uint32_t head_dim = 64;
  float mask_val = -1e9f;

  size_t bytes_rope = seq_len * num_heads * head_dim * sizeof(float);
  size_t bytes_mask = seq_len * seq_len * sizeof(float);

  Buffer buf_rope, buf_mask;
  if (!buf_rope.allocate(bytes_rope, BufferUsage::DeviceOnly, &err) ||
      !buf_mask.allocate(bytes_mask, BufferUsage::DeviceOnly, &err)) {
    std::cerr << "Allocation failed: " << err << "\n";
    return 1;
  }

  std::vector<float> h_rope(seq_len * num_heads * head_dim);
  std::vector<float> h_mask(seq_len * seq_len);

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &v : h_rope)
    v = dist(rng);
  for (auto &v : h_mask)
    v = dist(rng);

  std::vector<float> ref_rope = h_rope;
  std::vector<float> ref_mask = h_mask;

  buf_rope.copy_to_device(h_rope.data(), bytes_rope, &err);
  buf_mask.copy_to_device(h_mask.data(), bytes_mask, &err);

  HIPGraph graph;
  graph.add_node(std::make_unique<HIPRoPENode>(
      (float *)buf_rope.data(), seq_len, num_heads, head_dim, 10000.0f));
  graph.add_node(std::make_unique<HIPCausalMaskNode>((float *)buf_mask.data(),
                                                     seq_len, mask_val));

  Stream stream;
  stream.init(&err);

  std::cout << "Executing Attention Graph...\n";
  HIPGraphRunner::execute(stream.handle(), graph, &err);
  stream.sync();

  // Validate Mask
  std::vector<float> m_out(seq_len * seq_len);
  buf_mask.copy_to_host(m_out.data(), bytes_mask, &err);
  causal_mask_ref(ref_mask.data(), seq_len, mask_val);

  double mask_diff = 0.0;
  for (size_t i = 0; i < m_out.size(); ++i) {
    double d = std::abs(m_out[i] - ref_mask[i]);
    if (d > mask_diff)
      mask_diff = d;
  }
  std::cout << "Mask Max Diff: " << mask_diff << "\n";

  // Validate RoPE
  std::vector<float> r_out(h_rope.size());
  buf_rope.copy_to_host(r_out.data(), bytes_rope, &err);
  rope_ref(ref_rope.data(), seq_len, num_heads, head_dim, 10000.0f);

  double rope_diff = 0.0;
  for (size_t i = 0; i < r_out.size(); ++i) {
    double d = std::abs(r_out[i] - ref_rope[i]);
    if (d > rope_diff)
      rope_diff = d;
  }
  std::cout << "RoPE Max Diff: " << rope_diff << "\n";

  if (mask_diff > 1e-5 || rope_diff > 1e-5) {
    std::cout << "STATUS=FAILED\n";
    return 1;
  }

  std::cout << "STATUS=OK\n";
  return 0;
}
