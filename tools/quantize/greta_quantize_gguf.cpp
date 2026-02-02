#include "gcore/inference/weight_loader.hpp"
#include <fstream>
#include <iostream>
#include <vector>

using namespace gcore::inference;

// Helper to check if pointer is null (since GretaQuantInfo just uses void*)
inline bool is_null_quant(const void *ptr) { return ptr == nullptr; }

void write_string(std::ostream &os, const std::string &s) {
  uint32_t len = s.length();
  os.write((char *)&len, 4);
  os.write(s.data(), len);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: greta_quantize_gguf <input.gguf> <output.greta>\n";
    return 1;
  }

  std::string input_path = argv[1];
  std::string output_path = argv[2];

  std::string err;
  auto loader = create_weight_loader(input_path, &err);
  if (!loader) {
    std::cerr << "Failed to open input: " << err << "\n";
    return 1;
  }

  std::ofstream out(output_path, std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "Failed to open output file\n";
    return 1;
  }

  // Header
  out.write("GRETA_W\0", 8);

  // Model Config (Dummy JSON for now, can be improved)
  auto config = loader->get_config();
  std::string config_json = "{}"; // Simplified
  uint32_t json_len = config_json.length();
  out.write((char *)&json_len, 4);
  out.write(config_json.data(), json_len);

  auto tensors = loader->list_tensors();
  uint32_t n_tensors = 0;
  // We only care about weights that we usually quantize
  std::vector<TensorInfo> to_quantize;
  for (const auto &t : tensors) {
    if (t.name.find(".weight") != std::string::npos) {
      to_quantize.push_back(t);
    }
  }
  n_tensors = to_quantize.size();
  out.write((char *)&n_tensors, 4);

  std::cout << "Quantizing " << n_tensors << " tensors...\n";

  // Use HostVisible buffers to avoid needing a GPU for quantization?
  // Actually, WeightLoader currently needs a HIP context and Buffer.
  // I'll assume we HAVE a HIP context as it's required by WeightLoader's deps.
  if (gcore::rt::GretaContext::instance().initialize() !=
      gcore::rt::GretaResult::SUCCESS) {
    std::cerr << "Failed to init HIP context\n";
    return 1;
  }

  for (const auto &t : to_quantize) {
    std::cout << "  - " << t.name << "... " << std::flush;

    gcore::rt::hip::Buffer buffer, scales, head_scales;
    // We use DeviceOnly buffers because load_tensor_int4 expects them.
    // We will copy them back to host to save.
    if (!loader->load_tensor_int4(t.name, buffer, scales, head_scales, &err)) {
      std::cout << "Failed: " << err << "\n";
      continue;
    }

    auto qinfo = buffer.quant_info();

    write_string(out, t.name);
    uint32_t dtype = 4; // INT4
    out.write((char *)&dtype, 4);
    uint32_t rank = t.shape.size();
    out.write((char *)&rank, 4);
    for (auto d : t.shape) {
      uint64_t dd = d;
      out.write((char *)&dd, 8);
    }
    out.write((char *)&qinfo.group_size, 4);
    out.write((char *)&qinfo.num_heads, 4);

    uint64_t w_size = buffer.size();
    uint64_t s_size = scales.size();
    uint64_t h_size =
        is_null_quant(qinfo.head_scales) ? 0 : (qinfo.num_heads * 4);
    // Wait, head_scales might be allocated but null in qinfo?
    // GGUFLoader allocates it only if is_qkv is true.

    std::vector<uint8_t> w_host(w_size);
    std::vector<uint8_t> s_host(s_size);
    std::vector<uint8_t> h_host(h_size);

    buffer.copy_to_host(w_host.data(), w_size, &err);
    scales.copy_to_host(s_host.data(), s_size, &err);
    if (h_size > 0) {
      head_scales.copy_to_host(h_host.data(), h_size, &err);
    }

    out.write((char *)&w_size, 8);
    out.write((char *)&s_size, 8);
    out.write((char *)&h_size, 8);

    out.write((char *)w_host.data(), w_size);
    out.write((char *)s_host.data(), s_size);
    if (h_size > 0) {
      out.write((char *)h_host.data(), h_size);
    }

    std::cout << "Done (" << (w_size + s_size + h_size) / 1024 << " KB)\n";
  }

  std::cout << "\nPre-quantization finished: " << output_path << "\n";
  return 0;
}
