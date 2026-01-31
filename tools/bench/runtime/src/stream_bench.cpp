#include "gcore/rt/stream.hpp"

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
  const int n = argi(argc, argv, "--n", 500000);

  std::cout << "GRETA CORE Runtime Bench: stream_bench\n";
  std::cout << "n=" << n << "\n";

  gcore::rt::Stream s;

  // Benchmark enqueue+flush where each task is no-op
  std::vector<double> secs;
  secs.reserve(30);

  for (int round = 0; round < 30; round++) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) {
      s.enqueue([] {});
    }
    s.flush();
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

  auto per_task_ns = [&](double s) {
    return (s * 1e9) / static_cast<double>(n);
  };

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "RESULT stream_bench:\n";
  std::cout << "  mean_sec=" << mean
            << "  mean_ns_per_task=" << per_task_ns(mean) << "\n";
  std::cout << "  p50_sec=" << p50 << "   p50_ns_per_task=" << per_task_ns(p50)
            << "\n";
  std::cout << "  p99_sec=" << p99 << "   p99_ns_per_task=" << per_task_ns(p99)
            << "\n";

  // Event overhead sanity test
  gcore::rt::Event a, b;
  a.record(s);
  b.record(s);
  b.wait();
  auto dt = a.elapsed_ns(b);

  std::cout << "EVENT sanity:\n";
  std::cout << "  elapsed_ns(a->b)=" << dt << " (should be small, >=0)\n";

  return 0;
}
