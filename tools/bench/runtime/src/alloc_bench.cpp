#include "gcore/rt/allocator.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

static int argi(int argc, char **argv, const char *key, int def) {
  for (int i = 1; i + 1 < argc; i++) {
    if (std::string(argv[i]) == key)
      return std::stoi(argv[i + 1]);
  }
  return def;
}

static size_t argu(int argc, char **argv, const char *key, size_t def) {
  for (int i = 1; i + 1 < argc; i++) {
    if (std::string(argv[i]) == key)
      return static_cast<size_t>(std::stoull(argv[i + 1]));
  }
  return def;
}

int main(int argc, char **argv) {
  const int iters = argi(argc, argv, "--iters", 200000);
  const int ops_per_iter = argi(argc, argv, "--ops", 64);
  const int max_kb = argi(argc, argv, "--max-kb", 256);
  const uint64_t seed =
      static_cast<uint64_t>(argu(argc, argv, "--seed", 12345));

  std::cout << "GRETA CORE Runtime Bench: alloc_bench\n";
  std::cout << "iters=" << iters << " ops=" << ops_per_iter
            << " max_kb=" << max_kb << " seed=" << seed << "\n";

  gcore::rt::HostAllocator alloc;

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> size_dist(1, max_kb * 1024);

  std::vector<void *> ptrs(static_cast<size_t>(ops_per_iter), nullptr);

  // Warmup
  for (int i = 0; i < 10000; i++) {
    void *p = alloc.alloc(static_cast<size_t>(size_dist(rng)), 64);
    alloc.free(p);
  }

  std::vector<double> secs;
  secs.reserve(50);

  for (int round = 0; round < 50; round++) {
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < iters; i++) {
      // allocate batch
      for (int j = 0; j < ops_per_iter; j++) {
        ptrs[static_cast<size_t>(j)] =
            alloc.alloc(static_cast<size_t>(size_dist(rng)), 64);
      }
      // free batch
      for (int j = 0; j < ops_per_iter; j++) {
        alloc.free(ptrs[static_cast<size_t>(j)]);
        ptrs[static_cast<size_t>(j)] = nullptr;
      }
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

  const double total_ops = static_cast<double>(iters) *
                           static_cast<double>(ops_per_iter) *
                           2.0; // alloc+free
  auto ops_per_sec = [&](double s) { return total_ops / s; };

  auto st = alloc.stats();

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "RESULT alloc_bench:\n";
  std::cout << "  mean_sec=" << mean
            << "  mean_ops_per_sec=" << ops_per_sec(mean) << "\n";
  std::cout << "  p50_sec=" << p50 << "   p50_ops_per_sec=" << ops_per_sec(p50)
            << "\n";
  std::cout << "  p99_sec=" << p99 << "   p99_ops_per_sec=" << ops_per_sec(p99)
            << "\n";

  std::cout << "ALLOCATOR stats snapshot:\n";
  std::cout << "  alloc_calls=" << st.alloc_calls
            << " free_calls=" << st.free_calls
            << " reuse_hits=" << st.reuse_hits << " os_allocs=" << st.os_allocs
            << " bytes_in_use=" << st.bytes_in_use
            << " bytes_reserved=" << st.bytes_reserved << "\n";

  return 0;
}
