#include "gcore/rt/stream.hpp"

namespace gcore::rt {
struct Event::State {
  mutable std::mutex mu;
  mutable std::condition_variable cv;
  bool completed = false;
  std::chrono::steady_clock::time_point tp{};
};

// ---------------- Event ----------------

Event::Event() : st_(std::make_shared<State>()) {}
Event::~Event() = default;

void Event::signal() {
  {
    std::lock_guard<std::mutex> lk(st_->mu);
    st_->completed = true;
    st_->tp = std::chrono::steady_clock::now();
  }
  st_->cv.notify_all();
}

void Event::wait() const {
  std::unique_lock<std::mutex> lk(st_->mu);
  st_->cv.wait(lk, [&] { return st_->completed; });
}

uint64_t Event::elapsed_ns(const Event &other) const {
  // Lock ordenado para evitar deadlock (por dirección del puntero)
  const State *a = st_.get();
  const State *b = other.st_.get();
  if (a == b)
    return 0;

  State *first = const_cast<State *>((a < b) ? a : b);
  State *second = const_cast<State *>((a < b) ? b : a);

  std::scoped_lock lk(first->mu, second->mu);
  if (!st_->completed || !other.st_->completed)
    return 0;

  auto dt = other.st_->tp - st_->tp;
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count());
}

void Event::record(Stream &stream) {
  auto local = *this; // copia del handle
  stream.enqueue([local]() mutable { local.signal(); });
}

// ---------------- Stream ----------------

Stream::Stream() {
  worker_ = std::thread([this] { worker_loop(); });
}

Stream::~Stream() {
  flush();
  stop_.store(true, std::memory_order_release);
  cv_.notify_all();
  if (worker_.joinable())
    worker_.join();
}

void Stream::enqueue(std::function<void()> fn) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    q_.push(Task{std::move(fn)});
    enqueued_.fetch_add(1, std::memory_order_relaxed);
  }
  cv_.notify_one();
}

void Stream::flush() {
  // Wait until completed >= enqueued snapshot
  const uint64_t target = enqueued_.load(std::memory_order_acquire);
  while (completed_.load(std::memory_order_acquire) < target) {
    std::this_thread::yield();
  }
}

void Stream::worker_loop() {
  for (;;) {
    Task task;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] {
        return stop_.load(std::memory_order_acquire) || !q_.empty();
      });
      if (stop_.load(std::memory_order_acquire) && q_.empty())
        return;
      task = std::move(q_.front());
      q_.pop();
    }

    // Execute outside lock
    task.fn();
    completed_.fetch_add(1, std::memory_order_release);
  }
}

} // namespace gcore::rt
