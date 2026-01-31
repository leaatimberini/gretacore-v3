#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>
#include <string>

namespace gcore::rt::hip {

enum class BufferUsage {
  DeviceOnly,
  HostVisible, // Managed or Pinned
};

class Buffer {
public:
  Buffer() = default;
  ~Buffer();

  bool allocate(size_t size, BufferUsage usage, std::string *err);
  void free();

  void *data() { return ptr_; }
  const void *data() const { return ptr_; }
  size_t size() const { return size_; }

  bool copy_to_device(const void *host_ptr, size_t size, std::string *err);
  bool copy_to_host(void *host_ptr, size_t size, std::string *err);

private:
  void *ptr_ = nullptr;
  size_t size_ = 0;
  BufferUsage usage_ = BufferUsage::DeviceOnly;
};

} // namespace gcore::rt::hip
