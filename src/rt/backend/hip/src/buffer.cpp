#include "gcore/rt/hip/buffer.hpp"

#include <hip/hip_runtime.h>

namespace gcore::rt::hip {

Buffer::~Buffer() { free(); }

bool Buffer::allocate(size_t size, BufferUsage usage, GretaDataType type,
                      std::string *err) {
  free();
  size_ = size;
  usage_ = usage;
  type_ = type;

  hipError_t res;
  if (usage == BufferUsage::HostVisible) {
    res = hipHostMalloc(&ptr_, size);
  } else {
    res = hipMalloc(&ptr_, size);
  }

  if (res != hipSuccess) {
    if (err)
      *err = "HIP allocation failed: " + std::string(hipGetErrorString(res));
    ptr_ = nullptr;
    size_ = 0;
    return false;
  }

  return true;
}

void Buffer::free() {
  if (ptr_) {
    if (usage_ == BufferUsage::HostVisible) {
      (void)hipHostFree(ptr_);
    } else {
      (void)hipFree(ptr_);
    }
    ptr_ = nullptr;
    size_ = 0;
  }
}

bool Buffer::copy_to_device(const void *host_ptr, size_t size,
                            std::string *err) {
  hipError_t res = hipMemcpy(ptr_, host_ptr, size, hipMemcpyHostToDevice);
  if (res != hipSuccess) {
    if (err)
      *err = "hipMemcpy H2D failed: " + std::string(hipGetErrorString(res));
    return false;
  }
  return true;
}

bool Buffer::copy_to_host(void *host_ptr, size_t size, std::string *err) const {
  hipError_t res = hipMemcpy(host_ptr, ptr_, size, hipMemcpyDeviceToHost);
  if (res != hipSuccess) {
    if (err)
      *err = "hipMemcpy D2H failed: " + std::string(hipGetErrorString(res));
    return false;
  }
  return true;
}

bool Buffer::copy_to_host_offset(void *host_ptr, size_t offset, size_t size,
                                 std::string *err) const {
  if (offset + size > size_) {
    if (err)
      *err = "Buffer copy out of bounds: offset=" + std::to_string(offset) +
             ", size=" + std::to_string(size) +
             ", total_size=" + std::to_string(size_);
    return false;
  }
  char *device_ptr = static_cast<char *>(ptr_) + offset;
  hipError_t res = hipMemcpy(host_ptr, device_ptr, size, hipMemcpyDeviceToHost);
  if (res != hipSuccess) {
    if (err)
      *err = "hipMemcpy D2H failed: " + std::string(hipGetErrorString(res));
    return false;
  }
  return true;
}

} // namespace gcore::rt::hip
