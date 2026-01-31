#include "gcore/rt/telemetry.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static int argi(int argc, char **argv, const char *key, int def) {
  for (int i = 1; i + 1 < argc; i++) {
    if (std::string(argv[i]) == key)
      return std::stoi(argv[i + 1]);
  }
  return def;
}

int main(int argc, char **argv) {
  const int iters = argi(argc, argv, "--iters", 2000000);

  std::cout << "GRETA CORE Runtime Bench: telemetry_bench\n";
  std::cout << "iters=" << iters << "\n";

  gcore::rt::Counter c("timer_ns");

  // Measure overhead of ScopedTimer in tight loop.
  std::vector<double> secs;
  secs.reserve(30);

  for (int round = 0; round < 30; round++) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
      gcore::rt::ScopedTimer t(c);
      // no-op body
    }
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    secs.push_back(dt.count());
  }

  std::sort(secs.begin(), secs.end());
  double mean = 0.0;
  for (double v : secs)
    mean += v;
  mean /= secs.size();

  const double p50 = secs[secs.size() / 2];
  const double p99 = secs[static_cast<size_t>(
      std::min<size_t>(secs.size() - 1, (secs.size() * 99) / 100))];

  auto ns_per = [&](double s) {
    return (s * 1e9) / static_cast<double>(iters);
  };

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "RESULT telemetry_bench:\n";
  std::cout << "  mean_ns_per_scope=" << ns_per(mean) << "\n";
  std::cout << "  p50_ns_per_scope=" << ns_per(p50) << "\n";
  std::cout << "  p99_ns_per_scope=" << ns_per(p99) << "\n";

  std::cout << "COUNTER snapshot:\n";
  std::cout << "  " << c.name() << "=" << c.value() << "\n";

  return 0;
}
