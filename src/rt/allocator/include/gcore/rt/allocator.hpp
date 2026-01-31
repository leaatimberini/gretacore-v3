#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace gcore::rt {

// HostAllocator: caching/pooling allocator for CPU memory.
// - Small allocations go to power-of-two bins with freelists.
// - Large allocations bypass bins.
//
// Thread-safety: coarse-grained mutex (v1). We can evolve to per-bin locks.
class HostAllocator final {
public:
  struct Stats {
    uint64_t alloc_calls = 0;
    uint64_t free_calls = 0;
    uint64_t reuse_hits = 0; // satisfied from freelist
    uint64_t os_allocs = 0;  // new allocations from OS
    uint64_t bytes_in_use = 0;
    uint64_t bytes_reserved = 0; // total allocated from OS
  };

  // bin_min_pow2: smallest bin size = 2^bin_min_pow2 bytes
  // bin_max_pow2: largest bin size = 2^bin_max_pow2 bytes
  // large_threshold_pow2: >= 2^large_threshold_pow2 uses direct allocation
  HostAllocator(int bin_min_pow2 = 6,  // 64 B
                int bin_max_pow2 = 20, // 1 MiB
                int large_threshold_pow2 = 20);

  ~HostAllocator();

  HostAllocator(const HostAllocator &) = delete;
  HostAllocator &operator=(const HostAllocator &) = delete;

  // Allocate `size` bytes with alignment (power of two).
  // Returns nullptr on failure.
  void *alloc(std::size_t size, std::size_t alignment = 64);

  // Free memory previously allocated by this allocator.
  void free(void *p);

  // Returns a snapshot of stats.
  Stats stats() const;

  // Release cached blocks back to OS (best-effort).
  // Useful for tests; not required for normal operation.
  void release();

private:
  struct BlockHeader {
    uint32_t magic;
    uint16_t bin_index; // 0xFFFF for direct allocation
    uint16_t reserved;
    uint64_t requested; // requested payload size
    uint64_t allocated; // payload capacity
    BlockHeader *next;
  };

  static constexpr uint32_t kMagic = 0x47434F52; // 'GCOR'

  int bin_min_pow2_;
  int bin_max_pow2_;
  int large_threshold_pow2_;
  int bin_count_;

  mutable std::mutex mu_;

  std::vector<BlockHeader *> freelists_;
  Stats stats_;

  int size_to_bin(std::size_t size) const;
  std::size_t bin_to_size(int bin) const;

  void *os_alloc(std::size_t bytes, std::size_t alignment);
  void os_free(void *base);

  static std::size_t align_up(std::size_t x, std::size_t a);
};

} // namespace gcore::rt
