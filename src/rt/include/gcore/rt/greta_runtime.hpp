#pragma once
#include <cstddef>
#include <cstdint>

/**
 * GRETA CORE - Runtime Abstraction Layer (L0)
 * Goal: Provide a CUDA-equivalent mental model for AMD hardware.
 */

namespace gcore::rt {

#define GRETA_API_VERSION_MAJOR 0
#define GRETA_API_VERSION_MINOR 1

enum class GretaResult {
  SUCCESS = 0,
  ERROR_OUT_OF_MEM = 1,
  ERROR_GRAPH_CAPTURE = 2,
  ERROR_KERNEL_LAUNCH = 3,
  ERROR_INVALID_ARGS = 4,
  ERROR_NOT_IMPLEMENTED = 5
};

enum class GretaDataType {
  FP32 = 0,
  FP16 = 1,
  BF16 = 2,
  INT8 = 3,
  INT4 = 4,
  Q4_K = 5,
  FP8_E4M3 = 6,
  FP8_E5M2 = 7
};

struct GretaQuantInfo {
  const void *scales = nullptr;
  const void *head_scales = nullptr;
  float zero_point = 0.0f;
  uint32_t group_size = 0; // 0 means per-tensor
  uint32_t num_heads = 0;
};

class GretaStream;

// Event abstraction for profiling and synchronization
class GretaEvent {
public:
  virtual ~GretaEvent() = default;
  virtual void record(GretaStream *stream) = 0;
  virtual float elapsed_time_since(GretaEvent *start) = 0;
};

class GretaMemory {
public:
  virtual ~GretaMemory() = default;
  virtual void *data() = 0;
  virtual const void *data() const = 0;
  virtual size_t size() const = 0; // Total size in bytes

  virtual GretaDataType data_type() const = 0;
  virtual GretaQuantInfo quant_info() const = 0;

  virtual bool copy_from_host(const void *src, size_t size) = 0;
  virtual bool copy_to_host(void *dst, size_t size) const = 0;
};

// Stream abstraction (CUDA Stream equivalent)
class GretaStream {
public:
  virtual ~GretaStream() = default;
  virtual void synchronize() = 0;
  virtual void wait_event(GretaEvent *event) = 0;
  virtual void record_event(GretaEvent *event) = 0;
};

// Graph abstraction (CUDA Graph equivalent)
class GretaGraph {
public:
  virtual ~GretaGraph() = default;
  virtual void capture_start(GretaStream *stream) = 0;
  virtual void capture_end(GretaStream *stream) = 0;
  virtual GretaResult instantiate() = 0;
  virtual GretaResult launch(GretaStream *stream) = 0;
};

// Global Context
class GretaContext {
public:
  static GretaContext &instance();
  virtual GretaResult initialize() = 0;
  virtual GretaStream *create_stream() = 0;
  virtual GretaGraph *create_graph() = 0;
  virtual GretaEvent *create_event() = 0;
  virtual GretaMemory *create_memory(size_t size,
                                     GretaDataType type = GretaDataType::FP32,
                                     bool host_visible = false) = 0;
};

} // namespace gcore::rt
