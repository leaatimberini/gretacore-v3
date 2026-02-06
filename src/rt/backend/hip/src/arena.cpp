#include "gcore/rt/hip/arena.hpp"
#include <iostream>

namespace gcore::rt::hip {

GretaArena::GretaArena(size_t chunk_size) : chunk_size_(chunk_size) {
  if (std::getenv("GRETA_VERBOSE_INFO")) {
    std::cout << "[ARENA] Initialized with chunk size: "
              << chunk_size_ / (1024 * 1024) << " MB" << std::endl;
  }
}

bool GretaArena::allocate(size_t size, void **out_ptr, std::string *err) {
  std::lock_guard<std::mutex> lock(mutex_);

  // GPU Alignment: Round up to 256 bytes for performance/correctness
  const size_t alignment = 256;
  size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;

  // If current chunk cannot fit, allocate a new one
  if (chunks_.empty() || (current_offset_ + aligned_size > chunk_size_)) {
    // Handle case where single allocation > chunk_size
    size_t next_chunk_size = std::max(chunk_size_, aligned_size);

    Buffer new_chunk;
    if (!new_chunk.allocate(next_chunk_size, BufferUsage::DeviceOnly,
                            GretaDataType::FP8_E4M3, err)) {
      return false;
    }

    chunks_.push_back(std::move(new_chunk));
    current_offset_ = 0;

    if (std::getenv("GRETA_VERBOSE_INFO")) {
      std::cout << "[ARENA] Allocated new chunk of "
                << next_chunk_size / (1024 * 1024)
                << " MB. Total chunks: " << chunks_.size() << std::endl;
    }
  }

  // Assign from current chunk
  *out_ptr = static_cast<char *>(chunks_.back().data()) + current_offset_;
  current_offset_ += aligned_size;
  total_allocated_ += aligned_size;

  return true;
}

} // namespace gcore::rt::hip
