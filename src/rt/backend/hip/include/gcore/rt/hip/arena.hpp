#include "gcore/rt/hip/buffer.hpp"
#include <mutex>
#include <string>
#include <vector>

namespace gcore::rt::hip {

class GretaArena {
public:
  GretaArena(size_t chunk_size = 256 * 1024 * 1024); // Default 256MB
  ~GretaArena() = default;

  // Disallow copying
  GretaArena(const GretaArena &) = delete;
  GretaArena &operator=(const GretaArena &) = delete;

  bool allocate(size_t size, void **out_ptr, std::string *err);

  size_t total_allocated() const { return total_allocated_; }
  size_t num_chunks() const { return chunks_.size(); }

private:
  size_t chunk_size_;
  std::vector<Buffer> chunks_;
  size_t current_offset_ = 0;
  size_t total_allocated_ = 0;
  std::mutex mutex_;
};

} // namespace gcore::rt::hip
