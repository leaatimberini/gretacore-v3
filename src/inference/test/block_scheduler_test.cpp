#include "gcore/inference/block_scheduler.hpp"
#include "gcore/inference/model_config.hpp"

#include <iostream>

int main() {
  std::cout << "GRETA CORE: Block Scheduler Test\n";

  auto cfg = gcore::inference::ModelConfig::llama2_7b();
  std::cout << "Config: " << cfg.num_layers << " layers, dim=" << cfg.dim
            << "\n";

  gcore::inference::BlockScheduler scheduler;
  std::string err;

  // Initialize
  if (!scheduler.init(cfg, &err)) {
    std::cerr << "Init failed: " << err << "\n";
    return 1;
  }
  std::cout << "Initialized scheduler for " << scheduler.num_layers()
            << " layers\n";

  // Allocate weights
  std::cout << "Allocating weights...\n";
  if (!scheduler.allocate_weights(&err)) {
    std::cerr << "Weight allocation failed: " << err << "\n";
    return 1;
  }
  std::cout << "Weight buffers allocated\n";

  // Allocate activations for batch=1, seq=128
  std::cout << "Allocating activations (batch=1, seq=128)...\n";
  if (!scheduler.allocate_activations(1, 128, &err)) {
    std::cerr << "Activation allocation failed: " << err << "\n";
    return 1;
  }
  std::cout << "Activation buffers allocated\n";

  // Execute forward pass (skeleton)
  std::cout << "Executing forward pass...\n";
  if (!scheduler.forward(0, 10, &err)) {
    std::cerr << "Forward pass failed: " << err << "\n";
    return 1;
  }
  std::cout << "Forward pass completed (skeleton)\n";

  std::cout << "\nSTATUS=OK\n";
  return 0;
}
