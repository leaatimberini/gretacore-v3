#include "gcore/rt/allocator.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <new>

namespace gcore::rt {

static constexpr uint16_t kDirectBin = 0xFFFF;

std::size_t HostAllocator::align_up(std::size_t x, std::size_t a) {
  return (x + (a - 1)) & ~(a - 1);
}

HostAllocator::HostAllocator(int bin_min_pow2, int bin_max_pow2,
                             int large_threshold_pow2)
    : bin_min_pow2_(bin_min_pow2), bin_max_pow2_(bin_max_pow2),
      large_threshold_pow2_(large_threshold_pow2) {
  if (bin_min_pow2_ < 4)
    bin_min_pow2_ = 4;
  if (bin_max_pow2_ < bin_min_pow2_)
    bin_max_pow2_ = bin_min_pow2_;
  if (large_threshold_pow2_ < bin_min_pow2_)
    large_threshold_pow2_ = bin_max_pow2_;

  bin_count_ = (bin_max_pow2_ - bin_min_pow2_) + 1;
  freelists_.assign(static_cast<size_t>(bin_count_), nullptr);
}

HostAllocator::~HostAllocator() { release(); }

int HostAllocator::size_to_bin(std::size_t size) const {
  // include header overhead in bin sizing
  const std::size_t need = size;
  // find smallest pow2 >= need
  std::size_t s = 1ull << static_cast<unsigned>(bin_min_pow2_);
  int bin = 0;
  while (s < need && (bin_min_pow2_ + bin) < bin_max_pow2_) {
    s <<= 1;
    bin++;
  }
  return bin;
}

std::size_t HostAllocator::bin_to_size(int bin) const {
  return 1ull << static_cast<unsigned>(bin_min_pow2_ + bin);
}

void *HostAllocator::os_alloc(std::size_t bytes, std::size_t alignment) {
  // We need space for header + payload with requested alignment.
  // We'll allocate a base pointer aligned at least to alignment.
  void *p = nullptr;
  const std::size_t a = std::max<std::size_t>(alignment, 64);
  if (posix_memalign(&p, a, bytes) != 0)
    return nullptr;
  return p;
}

void HostAllocator::os_free(void *base) { std::free(base); }

void *HostAllocator::alloc(std::size_t size, std::size_t alignment) {
  if (size == 0)
    size = 1;

  const std::size_t a = std::max<std::size_t>(alignment, 64);
  const std::size_t header_bytes = align_up(sizeof(BlockHeader), 64);
  const std::size_t need_payload = align_up(size, a);
  const std::size_t need_total = header_bytes + need_payload;

  std::lock_guard<std::mutex> lk(mu_);
  stats_.alloc_calls++;

  // Large allocations bypass bins
  const std::size_t large_threshold =
      1ull << static_cast<unsigned>(large_threshold_pow2_);
  if (need_payload >= large_threshold) {
    void *base = os_alloc(need_total, a);
    if (!base)
      return nullptr;

    stats_.os_allocs++;
    stats_.bytes_reserved += need_total;
    stats_.bytes_in_use += need_payload;

    auto *h = reinterpret_cast<BlockHeader *>(base);
    std::memset(h, 0, header_bytes);
    h->magic = kMagic;
    h->bin_index = kDirectBin;
    h->requested = size;
    h->allocated = need_payload;
    h->next = nullptr;

    return reinterpret_cast<void *>(reinterpret_cast<std::uint8_t *>(base) +
                                    header_bytes);
  }

  const int bin = size_to_bin(need_payload);
  if (bin < 0 || bin >= bin_count_) {
    // Should not happen with thresholds, but fallback direct
    void *base = os_alloc(need_total, a);
    if (!base)
      return nullptr;

    stats_.os_allocs++;
    stats_.bytes_reserved += need_total;
    stats_.bytes_in_use += need_payload;

    auto *h = reinterpret_cast<BlockHeader *>(base);
    std::memset(h, 0, header_bytes);
    h->magic = kMagic;
    h->bin_index = kDirectBin;
    h->requested = size;
    h->allocated = need_payload;
    h->next = nullptr;

    return reinterpret_cast<void *>(reinterpret_cast<std::uint8_t *>(base) +
                                    header_bytes);
  }

  const std::size_t bin_payload = bin_to_size(bin);
  const std::size_t total = header_bytes + bin_payload;

  // Try freelist
  if (freelists_[static_cast<size_t>(bin)] != nullptr) {
    BlockHeader *h = freelists_[static_cast<size_t>(bin)];
    freelists_[static_cast<size_t>(bin)] = h->next;
    h->next = nullptr;

    stats_.reuse_hits++;
    stats_.bytes_in_use += bin_payload;

    // payload pointer
    return reinterpret_cast<void *>(reinterpret_cast<std::uint8_t *>(h) +
                                    header_bytes);
  }

  // OS allocate new
  void *base = os_alloc(total, a);
  if (!base)
    return nullptr;

  stats_.os_allocs++;
  stats_.bytes_reserved += total;
  stats_.bytes_in_use += bin_payload;

  auto *h = reinterpret_cast<BlockHeader *>(base);
  std::memset(h, 0, header_bytes);
  h->magic = kMagic;
  h->bin_index = static_cast<uint16_t>(bin);
  h->requested = size;
  h->allocated = bin_payload;
  h->next = nullptr;

  return reinterpret_cast<void *>(reinterpret_cast<std::uint8_t *>(base) +
                                  header_bytes);
}

void HostAllocator::free(void *p) {
  if (!p)
    return;

  const std::size_t header_bytes = align_up(sizeof(BlockHeader), 64);
  auto *base = reinterpret_cast<std::uint8_t *>(p) - header_bytes;
  auto *h = reinterpret_cast<BlockHeader *>(base);

  std::lock_guard<std::mutex> lk(mu_);
  stats_.free_calls++;

  if (h->magic != kMagic) {
    // Not a GRETA block: ignore hard fail in v1 (could be UB); we choose
    // safety. In debug builds, this could assert.
    return;
  }

  const uint16_t bin = h->bin_index;
  const std::size_t payload = static_cast<std::size_t>(h->allocated);
  if (stats_.bytes_in_use >= payload)
    stats_.bytes_in_use -= payload;

  if (bin == kDirectBin) {
    // direct free to OS
    const std::size_t total = header_bytes + payload;
    // bytes_reserved tracks OS allocations; reduce on free
    if (stats_.bytes_reserved >= total)
      stats_.bytes_reserved -= total;
    os_free(base);
    return;
  }

  if (bin >= static_cast<uint16_t>(bin_count_)) {
    // corrupt bin index -> safest: free to OS
    const std::size_t total = header_bytes + payload;
    if (stats_.bytes_reserved >= total)
      stats_.bytes_reserved -= total;
    os_free(base);
    return;
  }

  // push to freelist
  h->next = freelists_[static_cast<size_t>(bin)];
  freelists_[static_cast<size_t>(bin)] = h;
}

HostAllocator::Stats HostAllocator::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  return stats_;
}

void HostAllocator::release() {
  const std::size_t header_bytes = align_up(sizeof(BlockHeader), 64);

  std::lock_guard<std::mutex> lk(mu_);
  for (int bin = 0; bin < bin_count_; bin++) {
    BlockHeader *h = freelists_[static_cast<size_t>(bin)];
    freelists_[static_cast<size_t>(bin)] = nullptr;
    while (h) {
      BlockHeader *next = h->next;
      const std::size_t payload = static_cast<std::size_t>(h->allocated);
      const std::size_t total = header_bytes + payload;
      if (stats_.bytes_reserved >= total)
        stats_.bytes_reserved -= total;
      os_free(reinterpret_cast<void *>(h));
      h = next;
    }
  }
}

} // namespace gcore::rt
