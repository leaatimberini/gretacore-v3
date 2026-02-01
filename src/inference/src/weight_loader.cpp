#include "gcore/inference/weight_loader.hpp"

#include <cstring>
#include <fstream>
#include <iostream>

namespace gcore::inference {

// GGUF Magic and Version
static constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"
static constexpr uint32_t GGUF_VERSION = 3;

// GGUF Data Types
enum class GGMLType : uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  I8 = 16,
  I16 = 17,
  I32 = 18,
  COUNT = 19,
};

// Block sizes for quantized formats
static constexpr size_t QK_K = 256; // Super-block size for K-quants
static constexpr size_t QK4_0 = 32; // Block size for Q4_0

// Size of one block for each quantized type
static size_t ggml_block_size(GGMLType type) {
  switch (type) {
  case GGMLType::F32:
    return 1;
  case GGMLType::F16:
    return 1;
  case GGMLType::Q4_0:
    return QK4_0;
  case GGMLType::Q4_1:
    return QK4_0;
  case GGMLType::Q8_0:
    return 32;
  case GGMLType::Q4_K:
    return QK_K;
  case GGMLType::Q5_K:
    return QK_K;
  case GGMLType::Q6_K:
    return QK_K;
  default:
    return 32;
  }
}

static size_t ggml_type_size(GGMLType type) {
  switch (type) {
  case GGMLType::F32:
    return 4;
  case GGMLType::F16:
    return 2;
  case GGMLType::Q4_0:
    return 18; // 2 bytes scale + 16 bytes data for 32 elements
  case GGMLType::Q4_1:
    return 20; // 2 bytes scale + 2 bytes min + 16 bytes data
  case GGMLType::Q8_0:
    return 34; // 2 bytes scale + 32 bytes data
  case GGMLType::Q4_K:
    return 144; // K-quant block: 256 elements
  case GGMLType::Q5_K:
    return 176;
  case GGMLType::Q6_K:
    return 210;
  default:
    return 0;
  }
}

static std::string ggml_type_name(GGMLType type) {
  switch (type) {
  case GGMLType::F32:
    return "F32";
  case GGMLType::F16:
    return "F16";
  case GGMLType::Q4_0:
    return "Q4_0";
  case GGMLType::Q4_1:
    return "Q4_1";
  case GGMLType::Q8_0:
    return "Q8_0";
  case GGMLType::Q4_K:
    return "Q4_K";
  case GGMLType::Q5_K:
    return "Q5_K";
  case GGMLType::Q6_K:
    return "Q6_K";
  default:
    return "UNKNOWN";
  }
}

static uint32_t ggml_type_from_name(const std::string &name) {
  if (name == "F32")
    return static_cast<uint32_t>(GGMLType::F32);
  if (name == "F16")
    return static_cast<uint32_t>(GGMLType::F16);
  if (name == "Q4_0")
    return static_cast<uint32_t>(GGMLType::Q4_0);
  if (name == "Q4_K")
    return static_cast<uint32_t>(GGMLType::Q4_K);
  if (name == "Q5_K")
    return static_cast<uint32_t>(GGMLType::Q5_K);
  if (name == "Q6_K")
    return static_cast<uint32_t>(GGMLType::Q6_K);
  return static_cast<uint32_t>(GGMLType::F32);
}

struct GGUFLoader::Impl {
  std::string path;
  std::ifstream file;
  std::vector<TensorInfo> tensors;
  ModelConfig config;
  bool loaded = false;

  // File mapping
  size_t data_offset = 0;

  bool parse_header(std::string *err) {
    file.seekg(0);

    uint32_t magic;
    file.read(reinterpret_cast<char *>(&magic), 4);
    if (magic != GGUF_MAGIC) {
      *err = "Invalid GGUF magic number";
      return false;
    }

    uint32_t version;
    file.read(reinterpret_cast<char *>(&version), 4);
    if (version < 2 || version > 3) {
      *err = "Unsupported GGUF version: " + std::to_string(version);
      return false;
    }

    uint64_t tensor_count, kv_count;
    file.read(reinterpret_cast<char *>(&tensor_count), 8);
    file.read(reinterpret_cast<char *>(&kv_count), 8);

    // Skip KV pairs for now (TODO: parse for config extraction)
    // For simplicity, we'll use default Llama-2-7B config
    config = ModelConfig::llama2_7b();

    // Skip to tensor info section
    for (uint64_t i = 0; i < kv_count; ++i) {
      skip_kv_pair();
    }

    // Parse tensor info
    tensors.reserve(tensor_count);
    for (uint64_t i = 0; i < tensor_count; ++i) {
      TensorInfo info;
      if (!parse_tensor_info(info, err)) {
        return false;
      }
      tensors.push_back(info);
    }

    // Align to 32 bytes for data section
    size_t pos = file.tellg();
    data_offset = (pos + 31) & ~31ULL;

    // Calculate actual offsets
    size_t current_offset = data_offset;
    for (auto &t : tensors) {
      t.offset = current_offset;
      current_offset += t.size_bytes;
      current_offset = (current_offset + 31) & ~31ULL; // Align
    }

    loaded = true;
    return true;
  }

  void skip_kv_pair() {
    // Read key length + key
    uint64_t key_len;
    file.read(reinterpret_cast<char *>(&key_len), 8);
    file.seekg(key_len, std::ios::cur);

    // Read value type
    uint32_t value_type;
    file.read(reinterpret_cast<char *>(&value_type), 4);

    // Skip value based on type (simplified)
    switch (value_type) {
    case 0: // UINT8
      file.seekg(1, std::ios::cur);
      break;
    case 1: // INT8
      file.seekg(1, std::ios::cur);
      break;
    case 2: // UINT16
      file.seekg(2, std::ios::cur);
      break;
    case 3: // INT16
      file.seekg(2, std::ios::cur);
      break;
    case 4: // UINT32
      file.seekg(4, std::ios::cur);
      break;
    case 5: // INT32
      file.seekg(4, std::ios::cur);
      break;
    case 6: // FLOAT32
      file.seekg(4, std::ios::cur);
      break;
    case 7: // BOOL
      file.seekg(1, std::ios::cur);
      break;
    case 8: { // STRING
      uint64_t len;
      file.read(reinterpret_cast<char *>(&len), 8);
      file.seekg(len, std::ios::cur);
      break;
    }
    case 9: { // ARRAY
      uint32_t arr_type;
      uint64_t arr_len;
      file.read(reinterpret_cast<char *>(&arr_type), 4);
      file.read(reinterpret_cast<char *>(&arr_len), 8);
      // Skip array elements (simplified)
      for (uint64_t i = 0; i < arr_len; ++i) {
        // Recursively skip based on arr_type
        if (arr_type <= 7) {
          file.seekg(8, std::ios::cur); // Max primitive size
        }
      }
      break;
    }
    case 10: // UINT64
    case 11: // INT64
    case 12: // FLOAT64
      file.seekg(8, std::ios::cur);
      break;
    default:
      break;
    }
  }

  bool parse_tensor_info(TensorInfo &info, std::string *err) {
    // Read name
    uint64_t name_len;
    file.read(reinterpret_cast<char *>(&name_len), 8);
    info.name.resize(name_len);
    file.read(info.name.data(), name_len);

    // Read dimensions
    uint32_t n_dims;
    file.read(reinterpret_cast<char *>(&n_dims), 4);
    info.shape.resize(n_dims);
    for (uint32_t i = 0; i < n_dims; ++i) {
      uint64_t dim;
      file.read(reinterpret_cast<char *>(&dim), 8);
      info.shape[i] = dim;
    }

    // Read type
    uint32_t type;
    file.read(reinterpret_cast<char *>(&type), 4);
    info.dtype = ggml_type_name(static_cast<GGMLType>(type));

    // Read offset (relative to data section)
    uint64_t offset;
    file.read(reinterpret_cast<char *>(&offset), 8);

    // Calculate size
    size_t n_elements = 1;
    for (auto d : info.shape) {
      n_elements *= d;
    }
    GGMLType gtype = static_cast<GGMLType>(type);
    size_t type_size = ggml_type_size(gtype);
    size_t block_size = ggml_block_size(gtype);

    if (gtype == GGMLType::F32 || gtype == GGMLType::F16) {
      // Non-quantized: element-wise size
      info.size_bytes = n_elements * type_size;
    } else if (block_size > 0 && type_size > 0) {
      // Quantized: blocks of elements
      size_t n_blocks = (n_elements + block_size - 1) / block_size;
      info.size_bytes = n_blocks * type_size;
    } else {
      // Fallback: approximate
      info.size_bytes = n_elements * 2;
    }

    return true;
  }
};

GGUFLoader::GGUFLoader() : impl_(std::make_unique<Impl>()) {}
GGUFLoader::~GGUFLoader() = default;

bool GGUFLoader::open(const std::string &path, std::string *err) {
  impl_->path = path;
  impl_->file.open(path, std::ios::binary);
  if (!impl_->file.is_open()) {
    *err = "Failed to open file: " + path;
    return false;
  }
  return impl_->parse_header(err);
}

std::vector<TensorInfo> GGUFLoader::list_tensors() const {
  return impl_->tensors;
}

bool GGUFLoader::load_tensor(const std::string &name,
                             gcore::rt::hip::Buffer &buffer, std::string *err) {
  // Find tensor
  const TensorInfo *info = nullptr;
  for (const auto &t : impl_->tensors) {
    if (t.name == name) {
      info = &t;
      break;
    }
  }
  if (!info) {
    *err = "Tensor not found: " + name;
    return false;
  }

  // Allocate host buffer
  std::vector<char> host_data(info->size_bytes);

  // Read from file
  impl_->file.seekg(info->offset);
  impl_->file.read(host_data.data(), info->size_bytes);

  // Copy to GPU
  if (!buffer.allocate(info->size_bytes,
                       gcore::rt::hip::BufferUsage::DeviceOnly, err)) {
    return false;
  }
  if (!buffer.copy_to_device(host_data.data(), info->size_bytes, err)) {
    return false;
  }

  return true;
}

ModelConfig GGUFLoader::get_config() const { return impl_->config; }

// Factory function
std::unique_ptr<WeightLoader> create_weight_loader(const std::string &path,
                                                   std::string *err) {
  if (path.find(".gguf") != std::string::npos) {
    auto loader = std::make_unique<GGUFLoader>();
    if (!loader->open(path, err)) {
      return nullptr;
    }
    return loader;
  }
  // TODO: Add SafeTensorsLoader
  *err = "Unsupported weight format: " + path;
  return nullptr;
}

} // namespace gcore::inference
