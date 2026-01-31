#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

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

// Simple multithreaded memcpy bandwidth test.
// Measures read+write traffic (copy), so effective bytes moved ~ 2 * size per
// iter.
int main(int argc, char **argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; i++)
    args.emplace_back(argv[i]);

  const size_t size_mb = parse_arg_size(args, "--size-mb", 1024);
  const int iters = parse_arg_int(args, "--iters", 6);
  int threads = parse_arg_int(
      args, "--threads", static_cast<int>(std::thread::hardware_concurrency()));
  if (threads <= 0)
    threads = 1;

  const size_t total_bytes = size_mb * 1024ull * 1024ull;
  // Split per thread aligned to 64B.
  const size_t per_thread =
      (total_bytes / static_cast<size_t>(threads)) & ~size_t(63);
  const size_t used_bytes = per_thread * static_cast<size_t>(threads);

  std::cout << "GRETA CORE Platform Bench: membw_cpu\n";
  std::cout << "size_mb=" << size_mb << " iters=" << iters
            << " threads=" << threads
            << " used_bytes=" << used_bytes / (1024.0 * 1024.0) << " MiB\n";

  // Allocate 2 buffers (src/dst) with alignment.
  auto alloc_aligned = [](size_t bytes) -> void * {
    void *p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0)
      return nullptr;
    std::memset(p, 0xA5, bytes);
    return p;
  };

  std::vector<void *> src_ptrs(static_cast<size_t>(threads), nullptr);
  std::vector<void *> dst_ptrs(static_cast<size_t>(threads), nullptr);

  for (int t = 0; t < threads; t++) {
    src_ptrs[static_cast<size_t>(t)] = alloc_aligned(per_thread);
    dst_ptrs[static_cast<size_t>(t)] = alloc_aligned(per_thread);
    if (!src_ptrs[static_cast<size_t>(t)] ||
        !dst_ptrs[static_cast<size_t>(t)]) {
      std::cerr << "Allocation failed\n";
      return 1;
    }
  }

  // Warmup: touch pages to reduce first-touch noise.
  for (int t = 0; t < threads; t++) {
    volatile uint8_t *p =
        reinterpret_cast<volatile uint8_t *>(src_ptrs[static_cast<size_t>(t)]);
    for (size_t i = 0; i < per_thread; i += 4096)
      p[i] ^= 1;
  }

  std::atomic<bool> start{false};
  std::atomic<int> ready{0};

  std::vector<double> iter_seconds(static_cast<size_t>(iters), 0.0);

  for (int iter = 0; iter < iters; iter++) {
    start.store(false, std::memory_order_release);
    ready.store(0, std::memory_order_release);

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));

    for (int t = 0; t < threads; t++) {
      workers.emplace_back([&, t] {
        ready.fetch_add(1, std::memory_order_acq_rel);
        while (!start.load(std::memory_order_acquire)) { /* spin */
        }

        // Copy per-thread chunk.
        std::memcpy(dst_ptrs[static_cast<size_t>(t)],
                    src_ptrs[static_cast<size_t>(t)], per_thread);

        // Prevent dead-code elimination: read one byte.
        volatile uint8_t sink = reinterpret_cast<uint8_t *>(
            dst_ptrs[static_cast<size_t>(t)])[per_thread - 1];
        (void)sink;
      });
    }

    while (ready.load(std::memory_order_acquire) != threads) { /* spin */
    }
    auto t0 = std::chrono::steady_clock::now();
    start.store(true, std::memory_order_release);

    for (auto &th : workers)
      th.join();
    auto t1 = std::chrono::steady_clock::now();

    std::chrono::duration<double> dt = t1 - t0;
    iter_seconds[static_cast<size_t>(iter)] = dt.count();
  }

  // Compute stats.
  std::vector<double> s = iter_seconds;
  std::sort(s.begin(), s.end());
  const double mean = [&] {
    double sum = 0.0;
    for (double v : iter_seconds)
      sum += v;
    return sum / iter_seconds.size();
  }();
  const double p50 = s[s.size() / 2];
  const double p99 = s[static_cast<size_t>(
      std::min<size_t>(s.size() - 1, (s.size() * 99) / 100))];

  // Effective bytes moved for memcpy: read + write.
  const double bytes_moved = 2.0 * static_cast<double>(used_bytes);

  auto bw = [&](double seconds) -> double {
    return (bytes_moved / seconds) / (1024.0 * 1024.0 * 1024.0); // GiB/s
  };

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "RESULT membw_cpu:\n";
  std::cout << "  mean_sec=" << mean << "  mean_GiBps=" << bw(mean) << "\n";
  std::cout << "  p50_sec=" << p50 << "   p50_GiBps=" << bw(p50) << "\n";
  std::cout << "  p99_sec=" << p99 << "   p99_GiBps=" << bw(p99) << "\n";

  for (int t = 0; t < threads; t++) {
    std::free(src_ptrs[static_cast<size_t>(t)]);
    std::free(dst_ptrs[static_cast<size_t>(t)]);
  }

  return 0;
}
