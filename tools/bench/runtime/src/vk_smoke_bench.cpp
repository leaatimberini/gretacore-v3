#include "gcore/rt/vk/backend.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

int main() {
  std::cout << "GRETA CORE Runtime Bench: vk_smoke_bench\n";

  gcore::rt::vk::Backend b;
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
  std::cout << "  vendor_id=0x" << std::hex << info.vendor_id << std::dec
            << "\n";
  std::cout << "  device_id=0x" << std::hex << info.device_id << std::dec
            << "\n";
  std::cout << "  device_type=" << info.type << "\n";
  std::cout << "  name=" << info.name << "\n";
  std::cout << "  driver_name=" << info.driver_name << "\n";
  b.print_diagnostics(std::cout);

  if (b.gpu_blacklisted()) {
    std::cout << "SKIPPED: GPU blacklisted: " << b.blacklist_reason() << "\n";
    std::cout << "STATUS=SKIPPED reason=\"gpu_blacklisted\"\n";
    return 0;
  }

  uint64_t submit_ns = 0;
  auto t2 = std::chrono::steady_clock::now();
  ok = b.empty_submit(&submit_ns, &err);
  auto t3 = std::chrono::steady_clock::now();

  if (!ok) {
    std::cerr << "EMPTY SUBMIT FAILED: " << err << "\n";
    std::cout << "STATUS=FAILED reason=\"empty_submit_failed\"\n";
    return 2;
  }

  auto init_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

  auto total_submit_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "RESULT vk_smoke_bench:\n";
  std::cout << "  init_ms=" << (init_ns / 1e6) << "\n";
  std::cout << "  empty_submit_wait_ms=" << (submit_ns / 1e6) << "\n";
  std::cout << "  empty_submit_total_ms=" << (total_submit_ns / 1e6) << "\n";
  std::cout << "STATUS=OK\n";

  return 0;
}
