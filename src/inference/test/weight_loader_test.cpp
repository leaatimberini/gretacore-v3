#include "gcore/inference/model_config.hpp"
#include "gcore/inference/weight_loader.hpp"

#include <iostream>

int main(int argc, char *argv[]) {
  std::cout << "GRETA CORE: Weight Loader Test\n";

  // Test ModelConfig
  auto cfg = gcore::inference::ModelConfig::llama2_7b();
  std::cout << "Model Config (Llama-2-7B):\n";
  std::cout << "  dim: " << cfg.dim << "\n";
  std::cout << "  num_heads: " << cfg.num_heads << "\n";
  std::cout << "  num_layers: " << cfg.num_layers << "\n";
  std::cout << "  vocab_size: " << cfg.vocab_size << "\n";
  std::cout << "  hidden_dim: " << cfg.hidden_dim << "\n";
  std::cout << "  param_count: " << cfg.param_count() / 1e9 << "B\n";

  // Test GGUF loader (if path provided)
  if (argc > 1) {
    std::string path = argv[1];
    std::cout << "\nOpening: " << path << "\n";

    std::string err;
    auto loader = gcore::inference::create_weight_loader(path, &err);
    if (!loader) {
      std::cerr << "Error: " << err << "\n";
      return 1;
    }

    auto tensors = loader->list_tensors();
    std::cout << "Found " << tensors.size() << " tensors:\n";
    for (size_t i = 0; i < std::min<size_t>(10, tensors.size()); ++i) {
      std::cout << "  " << tensors[i].name << " [";
      for (size_t j = 0; j < tensors[i].shape.size(); ++j) {
        if (j > 0)
          std::cout << ", ";
        std::cout << tensors[i].shape[j];
      }
      std::cout << "] " << tensors[i].dtype << " ("
                << tensors[i].size_bytes / 1024 << " KB)\n";
    }
    if (tensors.size() > 10) {
      std::cout << "  ... and " << tensors.size() - 10 << " more\n";
    }
  }

  std::cout << "\nSTATUS=OK\n";
  return 0;
}
