#include "gcore/rt/hip/backend.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

int main() {
  std::cout << "GRETA CORE Runtime Bench: hip_smoke_bench\n";

  gcore::rt::hip::Backend b;
  std::string err;

  auto t0 = std::chrono::steady_clock::now();
  bool ok = b.init(&err);
  auto t1 = std::chrono::steady_clock::now();

  if (!ok) {
    std::cerr << "INIT FAILED: " << err << "\n";
    std::cout << "STATUS=FAILED reason=\"init_failed\"\n";
    return 1;
  }

  auto info = b.device_info();
  std::cout << "Selected device:\n";
  std::cout << "  device_id=" << info.device_id << "\n";
  std::cout << "  name=" << info.name << "\n";
  std::cout << "  gcn_arch=" << info.gcn_arch_name << "\n";

  b.print_diagnostics(std::cout);

  uint64_t sync_ns = 0;
  auto t2 = std::chrono::steady_clock::now();
  ok = b.sync(&sync_ns, &err);
  auto t3 = std::chrono::steady_clock::now();

  if (!ok) {
    std::cerr << "SYNC FAILED: " << err << "\n";
    std::cout << "STATUS=FAILED reason=\"sync_failed\"\n";
    return 2;
  }

  auto init_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

  auto total_sync_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "RESULT hip_smoke_bench:\n";
  std::cout << "  init_ms=" << (init_ns / 1e6) << "\n";
  std::cout << "  device_sync_wait_ms=" << (sync_ns / 1e6) << "\n";
  std::cout << "  device_sync_total_ms=" << (total_sync_ns / 1e6) << "\n";
  std::cout << "STATUS=OK\n";

  return 0;
}
