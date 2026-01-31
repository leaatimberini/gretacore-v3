#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Pointer-chasing latency benchmark.
// Construye un ciclo aleatorio de índices y mide ns por salto.

static size_t parse_arg_size(const std::vector<std::string> &args,
                             const std::string &key, size_t def) {
  for (size_t i = 0; i + 1 < args.size(); i++) {
    if (args[i] == key)
      return static_cast<size_t>(std::stoull(args[i + 1]));
  }
  return def;
}

static int parse_arg_int(const std::vector<std::string> &args,
                         const std::string &key, int def) {
  for (size_t i = 0; i + 1 < args.size(); i++) {
    if (args[i] == key)
      return std::stoi(args[i + 1]);
  }
  return def;
}

int main(int argc, char **argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; i++)
    args.emplace_back(argv[i]);

  const size_t size_mb = parse_arg_size(args, "--size-mb", 256);
  const int iters = parse_arg_int(args, "--iters", 50);
  const uint64_t seed =
      static_cast<uint64_t>(parse_arg_size(args, "--seed", 12345));

  const size_t bytes = size_mb * 1024ull * 1024ull;
  const size_t count = bytes / sizeof(uint32_t);

  std::cout << "GRETA CORE Platform Bench: memlat_cpu\n";
  std::cout << "size_mb=" << size_mb << " iters=" << iters << " seed=" << seed
            << "\n";

  // Alineación 64B
  uint32_t *next = nullptr;
  if (posix_memalign(reinterpret_cast<void **>(&next), 64,
                     count * sizeof(uint32_t)) != 0) {
    std::cerr << "Allocation failed\n";
    return 1;
  }

  // Construye una permutación y crea un ciclo (pointer chasing).
  std::vector<uint32_t> idx(count);
  for (size_t i = 0; i < count; i++)
    idx[i] = static_cast<uint32_t>(i);

  std::mt19937_64 rng(seed);
  std::shuffle(idx.begin(), idx.end(), rng);

  for (size_t i = 0; i + 1 < count; i++)
    next[idx[i]] = idx[i + 1];
  next[idx[count - 1]] = idx[0];

  // Warmup
  volatile uint32_t cur = idx[0];
  for (size_t i = 0; i < 1'000'000; i++)
    cur = next[cur];

  auto measure_once = [&](int steps) -> double {
    volatile uint32_t x = idx[0];
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; i++)
      x = next[x];
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    (void)x;
    return dt.count();
  };

  // Medimos varios rounds y calculamos ns/salto.
  const int steps = 50'000'000; // suficiente para estabilidad
  std::vector<double> ns_per_hop;
  ns_per_hop.reserve(static_cast<size_t>(iters));

  for (int i = 0; i < iters; i++) {
    double sec = measure_once(steps);
    double ns = (sec * 1e9) / static_cast<double>(steps);
    ns_per_hop.push_back(ns);
  }

  std::sort(ns_per_hop.begin(), ns_per_hop.end());
  auto mean = [&] {
    double sum = 0.0;
    for (double v : ns_per_hop)
      sum += v;
    return sum / ns_per_hop.size();
  }();
  double p50 = ns_per_hop[ns_per_hop.size() / 2];
  double p99 = ns_per_hop[static_cast<size_t>(
      std::min<size_t>(ns_per_hop.size() - 1, (ns_per_hop.size() * 99) / 100))];

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "RESULT memlat_cpu:\n";
  std::cout << "  mean_ns_per_hop=" << mean << "\n";
  std::cout << "  p50_ns_per_hop=" << p50 << "\n";
  std::cout << "  p99_ns_per_hop=" << p99 << "\n";

  std::free(next);
  return 0;
}
