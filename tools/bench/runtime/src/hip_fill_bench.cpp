#include "gcore/rt/hip/backend.hpp"
#include "gcore/rt/hip/buffer.hpp"
#include "gcore/rt/hip/kernels/basic_kernels.hpp"
#include "gcore/rt/hip/stream.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
  std::cout << "GRETA CORE Runtime Bench: hip_fill_bench\n";

  gcore::rt::hip::Backend backend;
  std::string err;
  if (!backend.init(&err)) {
    std::cerr << "Init failed: " << err << "\n";
    return 1;
  }
  backend.print_diagnostics(std::cout);

  size_t N = 1024 * 1024 * 64; // 64M elements * 4 bytes = 256MB
  uint32_t value = 0xDEADBEEF;

  gcore::rt::hip::Buffer buf;
  if (!buf.allocate(N * sizeof(uint32_t),
                    gcore::rt::hip::BufferUsage::DeviceOnly, &err)) {
    std::cerr << "Buffer allocation failed: " << err << "\n";
    return 1;
  }

  gcore::rt::hip::Stream stream;
  if (!stream.init(&err)) {
    std::cerr << "Stream creation failed: " << err << "\n";
    return 1;
  }

  std::cout << "Running Fill Kernel (N=" << N << ")...\n";
  gcore::rt::hip::kernels::launch_fill(
      stream.handle(), static_cast<uint32_t *>(buf.data()), value, N);

  if (!stream.sync(&err)) {
    std::cerr << "Sync failed: " << err << "\n";
    return 1;
  }

  // Verification
  std::vector<uint32_t> host_data(N);
  if (!buf.copy_to_host(host_data.data(), N * sizeof(uint32_t), &err)) {
    std::cerr << "D2H failed: " << err << "\n";
    return 1;
  }

  size_t errors = 0;
  for (size_t i = 0; i < N; ++i) {
    if (host_data[i] != value) {
      errors++;
      if (errors < 10) {
        std::cerr << "Mismatch at " << i << ": expected " << std::hex << value
                  << " got " << host_data[i] << std::dec << "\n";
      }
    }
  }

  if (errors > 0) {
    std::cout << "STATUS=FAILED errors=" << errors << "\n";
    return 1;
  }

  std::cout << "STATUS=OK\n";
  return 0;
}
