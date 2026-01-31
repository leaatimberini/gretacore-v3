#pragma once

#include <hip/hip_runtime.h>
#include <string>

namespace gcore::rt::hip {

class Stream {
public:
  Stream();
  ~Stream();

  // Disable copy
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  // Move support
  Stream(Stream &&other) noexcept;
  Stream &operator=(Stream &&other) noexcept;

  // Initialize the stream. Returns false on failure.
  bool init(std::string *err = nullptr);

  // Synchronize the host with this stream.
  bool sync(std::string *err = nullptr);

  // Get the raw HIP stream handle.
  hipStream_t handle() const { return stream_; }

private:
  hipStream_t stream_ = nullptr;
};

} // namespace gcore::rt::hip
