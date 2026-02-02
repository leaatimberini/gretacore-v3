#include "gcore/rt/hip/greta_runtime_hip.hpp"
#include <iostream>

namespace gcore::rt::hip {

// --- GretaMemoryHip ---
GretaMemoryHip::GretaMemoryHip(size_t size, GretaDataType type,
                               bool host_visible)
    : size_(size), type_(type), host_visible_(host_visible) {
  hipError_t err;
  if (host_visible) {
    err = hipHostMalloc(&ptr_, size);
  } else {
    err = hipMalloc(&ptr_, size);
  }
  if (err != hipSuccess) {
    ptr_ = nullptr;
    size_ = 0;
  }
}

GretaMemoryHip::~GretaMemoryHip() {
  if (ptr_) {
    if (host_visible_)
      (void)hipHostFree(ptr_);
    else
      (void)hipFree(ptr_);
  }
}

bool GretaMemoryHip::copy_from_host(const void *src, size_t size) {
  if (!ptr_ || size > size_)
    return false;
  return hipMemcpy(ptr_, src, size, hipMemcpyHostToDevice) == hipSuccess;
}

bool GretaMemoryHip::copy_to_host(void *dst, size_t size) const {
  if (!ptr_ || size > size_)
    return false;
  return hipMemcpy(dst, ptr_, size, hipMemcpyDeviceToHost) == hipSuccess;
}

// --- GretaEventHip ---
GretaEventHip::GretaEventHip() { (void)hipEventCreate(&event_); }

GretaEventHip::~GretaEventHip() {
  if (event_)
    (void)hipEventDestroy(event_);
}

void GretaEventHip::record(GretaStream *stream) {
  auto *s = static_cast<GretaStreamHip *>(stream);
  (void)hipEventRecord(event_, s->handle());
}

float GretaEventHip::elapsed_time_since(GretaEvent *start) {
  auto *s = static_cast<GretaEventHip *>(start);
  float ms = 0;
  (void)hipEventElapsedTime(&ms, s->handle(), event_);
  return ms;
}

// --- GretaStreamHip ---
GretaStreamHip::GretaStreamHip() {
  std::cout << "[GRETA_RT] Calling hipStreamCreate..." << std::endl;
  hipError_t err = hipStreamCreate(&stream_);
  if (err != hipSuccess) {
    std::cerr << "[GRETA_RT] hipStreamCreate failed: " << hipGetErrorString(err)
              << std::endl;
  } else {
    std::cout << "[GRETA_RT] hipStreamCreate success" << std::endl;
  }
  own_stream_ = true;
}
GretaStreamHip::GretaStreamHip(hipStream_t existing)
    : stream_(existing), own_stream_(false) {}

GretaStreamHip::~GretaStreamHip() {
  if (stream_ && own_stream_)
    (void)hipStreamDestroy(stream_);
}

void GretaStreamHip::synchronize() {
  if (stream_)
    (void)hipStreamSynchronize(stream_);
}

void GretaStreamHip::wait_event(GretaEvent *event) {
  auto *e = static_cast<GretaEventHip *>(event);
  (void)hipStreamWaitEvent(stream_, e->handle(), 0);
}

void GretaStreamHip::record_event(GretaEvent *event) {
  auto *e = static_cast<GretaEventHip *>(event);
  (void)hipEventRecord(e->handle(), stream_);
}

// --- GretaGraphHip ---
GretaGraphHip::GretaGraphHip() = default;

GretaGraphHip::~GretaGraphHip() {
  if (instance_)
    (void)hipGraphExecDestroy(instance_);
  if (graph_)
    (void)hipGraphDestroy(graph_);
}

void GretaGraphHip::capture_start(GretaStream *stream) {
  if (capturing_)
    return;
  auto *hip_stream = static_cast<GretaStreamHip *>(stream);
  (void)hipStreamBeginCapture(hip_stream->handle(), hipStreamCaptureModeGlobal);
  capturing_ = true;
}

void GretaGraphHip::capture_end(GretaStream *stream) {
  if (!capturing_)
    return;
  auto *hip_stream = static_cast<GretaStreamHip *>(stream);
  (void)hipStreamEndCapture(hip_stream->handle(), &graph_);
  capturing_ = false;
}

GretaResult GretaGraphHip::instantiate() {
  if (!graph_)
    return GretaResult::ERROR_GRAPH_CAPTURE;
  if (instance_)
    (void)hipGraphExecDestroy(instance_);
  hipError_t err = hipGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0);
  return (err == hipSuccess) ? GretaResult::SUCCESS
                             : GretaResult::ERROR_GRAPH_CAPTURE;
}

GretaResult GretaGraphHip::launch(GretaStream *stream) {
  if (!instance_)
    return GretaResult::ERROR_GRAPH_CAPTURE;
  auto *hip_stream = static_cast<GretaStreamHip *>(stream);
  hipError_t err = hipGraphLaunch(instance_, hip_stream->handle());
  return (err == hipSuccess) ? GretaResult::SUCCESS
                             : GretaResult::ERROR_KERNEL_LAUNCH;
}

// --- GretaContextHip ---
GretaResult GretaContextHip::initialize() {
  std::cout << "[GRETA_RT] Initializing HIP Context..." << std::endl;
  int device_count;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "[GRETA_RT] No HIP devices found!" << std::endl;
    return GretaResult::ERROR_INVALID_ARGS;
  }
  std::cout << "[GRETA_RT] Found " << device_count
            << " HIP device(s). Selecting device 0..." << std::endl;
  if (hipSetDevice(0) != hipSuccess) {
    std::cerr << "[GRETA_RT] Failed to select device 0" << std::endl;
    return GretaResult::ERROR_INVALID_ARGS;
  }
  std::cout << "[GRETA_RT] Context initialized successfully" << std::endl;
  return GretaResult::SUCCESS;
}

GretaStream *GretaContextHip::create_stream() { return new GretaStreamHip(); }
GretaGraph *GretaContextHip::create_graph() { return new GretaGraphHip(); }
GretaEvent *GretaContextHip::create_event() { return new GretaEventHip(); }
GretaMemory *GretaContextHip::create_memory(size_t size, GretaDataType type,
                                            bool host_visible) {
  return new GretaMemoryHip(size, type, host_visible);
}

} // namespace gcore::rt::hip

namespace gcore::rt {

// Singleton implementation - defined in the base namespace
GretaContext &GretaContext::instance() {
  static gcore::rt::hip::GretaContextHip ctx;
  return ctx;
}

} // namespace gcore::rt
