#include "gcore/rt/telemetry.hpp"

#include <chrono>

namespace gcore::rt {

uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

Counter::Counter(std::string_view name) : name_(name) {}

void Counter::inc(uint64_t v) { v_.fetch_add(v, std::memory_order_relaxed); }

uint64_t Counter::value() const { return v_.load(std::memory_order_relaxed); }

std::string_view Counter::name() const { return name_; }

ScopedTimer::ScopedTimer(Counter &sink_ns) : sink_(sink_ns), start_(now_ns()) {}

ScopedTimer::~ScopedTimer() {
  uint64_t end = now_ns();
  sink_.inc(end - start_);
}

} // namespace gcore::rt
