#include "gcore/rt/hip/stream.hpp"

namespace gcore::rt::hip {

Stream::Stream() = default;

Stream::~Stream() {
  if (stream_) {
    (void)hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

Stream::Stream(Stream &&other) noexcept : stream_(other.stream_) {
  other.stream_ = nullptr;
}

Stream &Stream::operator=(Stream &&other) noexcept {
  if (this != &other) {
    if (stream_) {
      (void)hipStreamDestroy(stream_);
    }
    stream_ = other.stream_;
    other.stream_ = nullptr;
  }
  return *this;
}

bool Stream::init(std::string *err) {
  if (stream_)
    return true;

  hipError_t res = hipStreamCreate(&stream_);
  if (res != hipSuccess) {
    if (err)
      *err = "hipStreamCreate failed: " + std::string(hipGetErrorString(res));
    return false;
  }
  return true;
}

bool Stream::sync(std::string *err) {
  if (!stream_)
    return true; // Default stream or uninitialized

  hipError_t res = hipStreamSynchronize(stream_);
  if (res != hipSuccess) {
    if (err)
      *err =
          "hipStreamSynchronize failed: " + std::string(hipGetErrorString(res));
    return false;
  }
  return true;
}

} // namespace gcore::rt::hip
