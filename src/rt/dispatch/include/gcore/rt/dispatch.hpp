#pragma once

#include <cstdint>
#include <functional>
#include <string_view>

#include "gcore/rt/stream.hpp"
#include "gcore/rt/telemetry.hpp"

namespace gcore::rt {

// Minimal dispatch: submits work onto a Stream and instruments it.
class Dispatcher final {
public:
  struct Stats {
    uint64_t submits = 0;
    uint64_t completed = 0;
    uint64_t total_work_ns = 0; // aggregated measured work time (ns)
  };

  Dispatcher();

  Dispatcher(const Dispatcher &) = delete;
  Dispatcher &operator=(const Dispatcher &) = delete;

  // Submit work onto the stream. Work is executed in-order.
  // - label is used for future tracing/logging.
  // - returns an Event that completes after the work.
  Event submit(Stream &stream, std::function<void()> work,
               std::string_view label = "work");

  Stats stats() const;

private:
  Counter submits_;
  Counter completed_;
  Counter work_ns_;
};

} // namespace gcore::rt
