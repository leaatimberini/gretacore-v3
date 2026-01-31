#include "gcore/rt/dispatch.hpp"

#include <atomic>

namespace gcore::rt {

Dispatcher::Dispatcher()
    : submits_("dispatch_submits"), completed_("dispatch_completed"),
      work_ns_("dispatch_work_ns") {}

Event Dispatcher::submit(Stream &stream, std::function<void()> work,
                         std::string_view /*label*/) {
  submits_.inc(1);

  Event done;
  Event done_copy = done; // handle copiable

  stream.enqueue([this, work = std::move(work), done_copy]() mutable {
    {
      ScopedTimer t(work_ns_);
      work();
    }
    completed_.inc(1);
    done_copy.signal();
  });

  return done;
}

Dispatcher::Stats Dispatcher::stats() const {
  Stats s;
  s.submits = submits_.value();
  s.completed = completed_.value();
  s.total_work_ns = work_ns_.value();
  return s;
}

} // namespace gcore::rt
