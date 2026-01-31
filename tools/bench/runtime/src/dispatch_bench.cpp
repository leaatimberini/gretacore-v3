#include "gcore/rt/dispatch.hpp"

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

  std::cout << "GRETA CORE Runtime Bench: dispatch_bench\n";
  std::cout << "n=" << n << "\n";

  gcore::rt::Stream stream;
  gcore::rt::Dispatcher disp;

  std::vector<double> secs;
  secs.reserve(20);

  for (int round = 0; round < 20; round++) {
    auto t0 = std::chrono::steady_clock::now();

    // Submit N trabajos no-op. No guardamos todos los Events (evita overhead de
    // vector). Esperamos al último Event, que implica que todo lo anterior
    // terminó.
    gcore::rt::Event last;
    for (int i = 0; i < n; i++) {
      last = disp.submit(stream, [] {});
    }
    last.wait();

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

  auto ns_per = [&](double s) { return (s * 1e9) / static_cast<double>(n); };

  auto st = disp.stats();

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "RESULT dispatch_bench:\n";
  std::cout << "  mean_ns_per_submit_and_exec=" << ns_per(mean) << "\n";
  std::cout << "  p50_ns_per_submit_and_exec=" << ns_per(p50) << "\n";
  std::cout << "  p99_ns_per_submit_and_exec=" << ns_per(p99) << "\n";

  std::cout << "DISPATCH stats snapshot:\n";
  std::cout << "  submits=" << st.submits << " completed=" << st.completed
            << " total_work_ns=" << st.total_work_ns << "\n";

  return 0;
}
