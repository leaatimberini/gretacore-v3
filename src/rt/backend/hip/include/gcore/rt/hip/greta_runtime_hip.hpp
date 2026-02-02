#pragma once
#include "gcore/rt/greta_runtime.hpp"
#include <hip/hip_runtime.h>
#include <memory>

/**
 * GRETA CORE - HIP Backend Implementation of GPM L0
 */

namespace gcore::rt::hip {

class GretaMemoryHip : public GretaMemory {
public:
  GretaMemoryHip(size_t size, GretaDataType type = GretaDataType::FP32,
                 bool host_visible = false);
  ~GretaMemoryHip() override;

  void *data() override { return ptr_; }
  const void *data() const override { return ptr_; }
  size_t size() const override { return size_; }

  GretaDataType data_type() const override { return type_; }
  GretaQuantInfo quant_info() const override { return qinfo_; }

  bool copy_from_host(const void *src, size_t size) override;
  bool copy_to_host(void *dst, size_t size) const override;

private:
  void *ptr_ = nullptr;
  size_t size_ = 0;
  GretaDataType type_ = GretaDataType::FP32;
  GretaQuantInfo qinfo_;
  bool host_visible_ = false;
};

class GretaEventHip : public GretaEvent {
public:
  GretaEventHip();
  ~GretaEventHip() override;

  void record(GretaStream *stream) override;
  float elapsed_time_since(GretaEvent *start) override;

  hipEvent_t handle() const { return event_; }

private:
  hipEvent_t event_ = nullptr;
};

class GretaStreamHip : public GretaStream {
public:
  GretaStreamHip();
  GretaStreamHip(hipStream_t existing);
  ~GretaStreamHip() override;

  void synchronize() override;
  void wait_event(GretaEvent *event) override;
  void record_event(GretaEvent *event) override;

  hipStream_t handle() const { return stream_; }

private:
  hipStream_t stream_ = nullptr;
  bool own_stream_ = true;
};

class GretaGraphHip : public GretaGraph {
public:
  GretaGraphHip();
  ~GretaGraphHip() override;

  void capture_start(GretaStream *stream) override;
  void capture_end(GretaStream *stream) override;
  GretaResult instantiate() override;
  GretaResult launch(GretaStream *stream) override;

private:
  hipGraph_t graph_ = nullptr;
  hipGraphExec_t instance_ = nullptr;
  bool capturing_ = false;
};

class GretaContextHip : public GretaContext {
public:
  GretaResult initialize() override;
  GretaStream *create_stream() override;
  GretaGraph *create_graph() override;
  GretaEvent *create_event() override;
  GretaMemory *create_memory(size_t size,
                             GretaDataType type = GretaDataType::FP32,
                             bool host_visible = false) override;
};

} // namespace gcore::rt::hip
