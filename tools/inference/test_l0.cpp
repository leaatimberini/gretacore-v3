#include "gcore/rt/greta_runtime.hpp"
#include <iostream>
#include <vector>

using namespace gcore::rt;

int main() {
  std::cout << "Testing GRETA Runtime L0..." << std::endl;

  GretaContext &ctx = GretaContext::instance();
  if (ctx.initialize() != GretaResult::SUCCESS) {
    std::cerr << "Failed to initialize Greta Context" << std::endl;
    return 1;
  }

  GretaStream *stream = ctx.create_stream();
  if (!stream) {
    std::cerr << "Failed to create Greta Stream" << std::endl;
    return 1;
  }

  const size_t size = 1024 * sizeof(float);
  GretaMemory *mem = ctx.create_memory(size);
  if (!mem) {
    std::cerr << "Failed to allocate Greta Memory" << std::endl;
    delete stream;
    return 1;
  }

  std::cout << "Initialization and Allocation successful." << std::endl;

  stream->synchronize();
  delete mem;
  delete stream;

  return 0;
}
