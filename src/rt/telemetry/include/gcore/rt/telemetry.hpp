#pragma once

#include <atomic>
#include <cstdint>
#include <string_view>

namespace gcore::rt {

// Monotonic clock helper (ns since unspecified epoch).
uint64_t now_ns();

// A named counter (lock-free increment).
class Counter final {
public:
  explicit Counter(std::string_view name);

  Counter(const Counter &) = delete;
  Counter &operator=(const Counter &) = delete;

  void inc(uint64_t v = 1);
  uint64_t value() const;
  std::string_view name() const;

private:
  std::atomic<uint64_t> v_{0};
  std::string_view name_;
};

// Scoped timer that accumulates elapsed time into a counter (ns).
class ScopedTimer final {
public:
  explicit ScopedTimer(Counter &sink_ns);
  ~ScopedTimer();

  ScopedTimer(const ScopedTimer &) = delete;
  ScopedTimer &operator=(const ScopedTimer &) = delete;

private:
  Counter &sink_;
  uint64_t start_;
};

} // namespace gcore::rt
