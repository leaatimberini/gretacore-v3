#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

namespace gcore::rt {

class Stream;

// Event: can be recorded on a Stream, waited on, and used for elapsed time.
class Event final {
public:
  Event();
  ~Event();

  // Ahora Event es copiable/movible (handle)
  Event(const Event &) = default;
  Event &operator=(const Event &) = default;
  Event(Event &&) noexcept = default;
  Event &operator=(Event &&) noexcept = default;

  // Record: completa cuando el stream ejecute el marker.
  void record(Stream &stream);

  // Wait: bloquea hasta que el evento esté completo.
  void wait() const;

  // Tiempo entre dos eventos completos (ns). 0 si alguno no completó.
  uint64_t elapsed_ns(const Event &other) const;

  // Runtime-internal: marca el evento como completo.
  void signal();

private:
  struct State;
  std::shared_ptr<State> st_;
};

class Stream final {
public:
  Stream();
  ~Stream();

  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  // Enqueue a task for execution in-order.
  void enqueue(std::function<void()> fn);

  // Blocks until all queued tasks are finished.
  void flush();

private:
  struct Task {
    std::function<void()> fn;
  };

  std::mutex mu_;
  std::condition_variable cv_;
  std::queue<Task> q_;
  std::thread worker_;
  std::atomic<bool> stop_{false};

  // For flush semantics
  std::atomic<uint64_t> enqueued_{0};
  std::atomic<uint64_t> completed_{0};

  void worker_loop();
};

} // namespace gcore::rt
