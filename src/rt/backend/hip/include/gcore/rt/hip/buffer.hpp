#pragma once
#include "gcore/rt/greta_runtime.hpp"
#include <cstddef>
#include <hip/hip_runtime.h>
#include <string>

namespace gcore::rt::hip {

enum class BufferUsage {
  DeviceOnly,
  HostVisible, // Managed or Pinned
};

class Buffer : public gcore::rt::GretaMemory {
public:
  Buffer() = default;
  ~Buffer();

  bool allocate(size_t size, BufferUsage usage,
                GretaDataType type = GretaDataType::FP32,
                std::string *err = nullptr);
  void free();

  void *data() override { return ptr_; }
  const void *data() const override { return ptr_; }
  size_t size() const override { return size_; }

  GretaDataType data_type() const override { return type_; }
  GretaQuantInfo quant_info() const override { return qinfo_; }

  bool copy_to_device(const void *host_ptr, size_t size, std::string *err);
  bool copy_to_host(void *host_ptr, size_t size, std::string *err) const;
  bool copy_to_host_offset(void *host_ptr, size_t offset, size_t size,
                           std::string *err) const;

  // GretaMemory implementation
  bool copy_from_host(const void *src, size_t size) override {
    return copy_to_device(src, size, nullptr);
  }
  bool copy_to_host(void *dst, size_t size) const override {
    return copy_to_host(dst, size, nullptr);
  }

  void set_quant_info(const GretaQuantInfo &info) { qinfo_ = info; }

private:
  void *ptr_ = nullptr;
  size_t size_ = 0;
  GretaDataType type_ = GretaDataType::FP32;
  GretaQuantInfo qinfo_;
  BufferUsage usage_ = BufferUsage::DeviceOnly;
};

} // namespace gcore::rt::hip
