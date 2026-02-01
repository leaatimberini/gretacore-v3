#include "gcore/inference/weight_loader.hpp"

#include <cmath>
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

// Q4_K block structure (256 elements = 144 bytes):
// - 2 bytes: d (FP16 super-block scale)
// - 2 bytes: dmin (FP16 super-block min)
// - 12 bytes: scales (6-bit scales packed for 8 sub-blocks)
// - 128 bytes: qs (4-bit quantized values, 256/2 = 128)
struct block_q4_k {
  uint16_t d;         // super-block scale (FP16)
  uint16_t dmin;      // super-block min (FP16)
  uint8_t scales[12]; // scales and mins for 8 sub-blocks
  uint8_t qs[128];    // quantized values (4 bits each)
};

// Convert FP16 to FP32 (simplified)
static float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;

  if (exp == 0) {
    if (mant == 0)
      return sign ? -0.0f : 0.0f;
    // Denormalized
    exp = 1;
    while ((mant & 0x400) == 0) {
      mant <<= 1;
      exp--;
    }
    mant &= ~0x400;
  } else if (exp == 31) {
    return sign ? -INFINITY : INFINITY;
  }

  uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  float result;
  memcpy(&result, &f, sizeof(float));
  return result;
}

// Dequantize Q4_K block to FP32
static void dequantize_q4_k_block(const uint8_t *src, float *dst,
                                  size_t n_elements) {
  const block_q4_k *block = reinterpret_cast<const block_q4_k *>(src);

  float d = fp16_to_fp32(block->d);
  float dmin = fp16_to_fp32(block->dmin);

  uint8_t sc8[8];
  uint8_t m8[8];

  // Extract 6-bit scales and mins according to GGML Q4_K layout
  for (int i = 0; i < 4; ++i) {
    sc8[i] = block->scales[i] & 0x3f;
    sc8[i + 4] = (block->scales[i] >> 6) |
                 ((block->scales[4 + (i / 2)] >> ((i % 2) * 4)) & 0x03) << 2;
    // Wait, the above is still a bit simplified. Let's use exact GGML mapping
  }

  // Refined extraction matching ggml-quants.c:
  sc8[0] = block->scales[0] & 0x3f;
  sc8[1] = block->scales[1] & 0x3f;
  sc8[2] = block->scales[2] & 0x3f;
  sc8[3] = block->scales[3] & 0x3f;
  sc8[4] = (block->scales[0] >> 6) | ((block->scales[4] & 0x0f) << 2);
  sc8[5] = (block->scales[1] >> 6) | ((block->scales[4] >> 4) << 2);
  sc8[6] = (block->scales[2] >> 6) | ((block->scales[5] & 0x0f) << 2);
  sc8[7] = (block->scales[3] >> 6) | ((block->scales[5] >> 4) << 2);

  m8[0] = block->scales[6] & 0x3f;
  m8[1] = block->scales[7] & 0x3f;
  m8[2] = block->scales[8] & 0x3f;
  m8[3] = block->scales[9] & 0x3f;
  m8[4] = (block->scales[6] >> 6) | ((block->scales[10] & 0x0f) << 2);
  m8[5] = (block->scales[7] >> 6) | ((block->scales[10] >> 4) << 2);
  m8[6] = (block->scales[8] >> 6) | ((block->scales[11] & 0x0f) << 2);
  m8[7] = (block->scales[9] >> 6) | ((block->scales[11] >> 4) << 2);

  // Dequantize 8 sub-blocks of 32 elements each
  for (int j = 0; j < 8; ++j) {
    float scale = d * sc8[j];
    float min_val = dmin * m8[j];

    // Each sub-block has 16 bytes of 4-bit values (32 elements)
    for (int l = 0; l < 16; ++l) {
      uint8_t qs = block->qs[j * 16 + l];
      dst[j * 32 + l * 2 + 0] = scale * (qs & 0x0F) - min_val;
      dst[j * 32 + l * 2 + 1] = scale * (qs >> 4) - min_val;
    }
  }
}

// Q6_K block structure (256 elements = 210 bytes)
static void dequantize_q6_k_block(const uint8_t *src, float *dst,
                                  size_t n_elements) {
  // Q6_K layout:
  // - 128 bytes: ql (lower 4 bits)
  // - 64 bytes: qh (upper 2 bits)
  // - 16 bytes: scales (8-bit per 16 elements)
  // - 2 bytes: d (FP16 scale)

  const uint8_t *ql = src;
  const uint8_t *qh = src + 128;
  const int8_t *scales = reinterpret_cast<const int8_t *>(src + 192);
  uint16_t d_raw;
  memcpy(&d_raw, src + 208, 2);
  float d = fp16_to_fp32(d_raw);

  for (int i = 0; i < 256; ++i) {
    int l_idx = i / 2;
    int shift = (i % 2) * 4;
    uint8_t l_val = (ql[l_idx] >> shift) & 0x0F;

    int h_idx = i / 4;
    int h_shift = (i % 4) * 2;
    uint8_t h_val = (qh[h_idx] >> h_shift) & 0x03;

    int q = (l_val | (h_val << 4)) - 32; // 6-bit signed
    int sc_idx = i / 16;
    dst[i] = d * q * scales[sc_idx];
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

static GGMLType ggml_type_from_name(const std::string &name) {
  if (name == "F32")
    return GGMLType::F32;
  if (name == "F16")
    return GGMLType::F16;
  if (name == "Q4_0")
    return GGMLType::Q4_0;
  if (name == "Q4_K")
    return GGMLType::Q4_K;
  if (name == "Q5_K")
    return GGMLType::Q5_K;
  if (name == "Q6_K")
    return GGMLType::Q6_K;
  return GGMLType::F32;
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

    // Sanity check: GGUF usually has < 100 KV pairs
    if (kv_count > 1000) {
      *err = "Metadata KV count too large: " + std::to_string(kv_count);
      return false;
    }

    std::cerr << "[GGUF] Version: " << version << ", Tensors: " << tensor_count
              << ", KV pairs: " << kv_count << "\n";

    // Sanity check
    if (tensor_count > 10000 || kv_count > 10000) {
      *err = "Suspicious tensor/kv counts, file may be corrupted";
      return false;
    }

    // Parse KV pairs
    if (!parse_kv_pairs(kv_count, err)) {
      return false;
    }

    std::cerr << "[GGUF] After KV pairs, position: " << file.tellg() << "\n";

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

  bool parse_kv_pairs(uint64_t count, std::string *err) {
    // Restore defaults first
    config = ModelConfig::llama2_7b();

    for (uint64_t i = 0; i < count; ++i) {
      if (!parse_kv_pair(err))
        return false;
    }
    return true;
  }

  bool parse_kv_pair(std::string *err) {
    uint64_t key_len;
    file.read(reinterpret_cast<char *>(&key_len), 8);
    if (!file || key_len > 1024) {
      *err = "Invalid or too long GGUF key length";
      return false;
    }
    std::string key(key_len, '\0');
    file.read(&key[0], key_len);

    uint32_t value_type;
    file.read(reinterpret_cast<char *>(&value_type), 4);
    if (!file)
      return false;

    if (key == "tokenizer.ggml.tokens" && value_type == 9) {
      uint32_t arr_type;
      uint64_t arr_len;
      file.read(reinterpret_cast<char *>(&arr_type), 4);
      file.read(reinterpret_cast<char *>(&arr_len), 8);
      if (arr_type != 8 || arr_len > 1000000) {
        *err = "Invalid tokenizer tokens array";
        return false;
      }
      config.vocabulary.clear();
      config.vocabulary.reserve(arr_len);
      for (uint64_t i = 0; i < arr_len; ++i) {
        uint64_t slen;
        file.read(reinterpret_cast<char *>(&slen), 8);
        if (slen > 1024) { // Llama-2 tokens are short
          file.seekg(slen, std::ios::cur);
          config.vocabulary.push_back("<too_long>");
          continue;
        }
        std::string s(slen, '\0');
        file.read(&s[0], slen);
        config.vocabulary.push_back(s);
      }
      config.vocab_size = static_cast<uint32_t>(arr_len);
    } else {
      if (!skip_value(value_type, err))
        return false;
    }
    return true;
  }

  bool skip_value(uint32_t value_type, std::string *err) {
    switch (value_type) {
    case 0:
    case 1:
    case 7:
      file.seekg(1, std::ios::cur);
      break;
    case 2:
    case 3:
      file.seekg(2, std::ios::cur);
      break;
    case 4:
    case 5:
    case 6:
      file.seekg(4, std::ios::cur);
      break;
    case 8: { // STRING
      uint64_t len;
      file.read(reinterpret_cast<char *>(&len), 8);
      if (len > 1000000)
        return false;
      file.seekg(len, std::ios::cur);
      break;
    }
    case 9: { // ARRAY
      uint32_t arr_type;
      uint64_t arr_len;
      file.read(reinterpret_cast<char *>(&arr_type), 4);
      file.read(reinterpret_cast<char *>(&arr_len), 8);
      if (arr_len > 1000000)
        return false;
      if (arr_type == 8) { // String array
        for (uint64_t i = 0; i < arr_len; ++i) {
          uint64_t slen;
          file.read(reinterpret_cast<char *>(&slen), 8);
          if (slen > 1000000)
            return false;
          file.seekg(slen, std::ios::cur);
        }
      } else {
        size_t elem_size = 0;
        if (arr_type <= 1 || arr_type == 7)
          elem_size = 1;
        else if (arr_type <= 3)
          elem_size = 2;
        else if (arr_type <= 6)
          elem_size = 4;
        else
          elem_size = 8;
        file.seekg(arr_len * elem_size, std::ios::cur);
      }
      break;
    }
    case 10:
    case 11:
    case 12:
      file.seekg(8, std::ios::cur);
      break;
    default:
      return false;
    }
    return true;
  }

  bool parse_tensor_info(TensorInfo &info, std::string *err) {
    // Read name
    uint64_t name_len;
    file.read(reinterpret_cast<char *>(&name_len), 8);

    // Sanity check: tensor names should be reasonable
    if (name_len > 512) {
      *err = "Tensor name too long: " + std::to_string(name_len);
      return false;
    }

    info.name.resize(name_len);
    file.read(info.name.data(), name_len);

    // Read dimensions
    uint32_t n_dims;
    file.read(reinterpret_cast<char *>(&n_dims), 4);

    // Sanity check: tensors shouldn't have > 8 dimensions
    if (n_dims > 8) {
      *err = "Too many dimensions: " + std::to_string(n_dims);
      return false;
    }

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
  GGMLType gtype = GGMLType::F32;
  for (const auto &t : impl_->tensors) {
    if (t.name == name) {
      info = &t;
      gtype = ggml_type_from_name(t.dtype);
      break;
    }
  }
  if (!info) {
    *err = "Tensor not found: " + name;
    return false;
  }

  // Read raw quantized data from file
  std::vector<uint8_t> raw_data(info->size_bytes);
  impl_->file.seekg(info->offset);
  impl_->file.read(reinterpret_cast<char *>(raw_data.data()), info->size_bytes);

  // Calculate number of elements
  size_t n_elements = 1;
  for (auto d : info->shape)
    n_elements *= d;

  // Dequantize if needed
  std::vector<float> fp32_data;
  const void *upload_data = nullptr;
  size_t upload_size = 0;

  if (gtype == GGMLType::F32) {
    // Already FP32, copy directly
    upload_data = raw_data.data();
    upload_size = info->size_bytes;
  } else if (gtype == GGMLType::F16) {
    // Convert FP16 to FP32
    fp32_data.resize(n_elements);
    const uint16_t *src = reinterpret_cast<const uint16_t *>(raw_data.data());
    for (size_t i = 0; i < n_elements; ++i) {
      fp32_data[i] = fp16_to_fp32(src[i]);
    }
    upload_data = fp32_data.data();
    upload_size = n_elements * sizeof(float);
  } else if (gtype == GGMLType::Q4_K) {
    // Dequantize Q4_K to FP32
    fp32_data.resize(n_elements);
    size_t block_size = ggml_block_size(gtype);
    size_t type_size = ggml_type_size(gtype);
    size_t n_blocks = n_elements / block_size;

    for (size_t b = 0; b < n_blocks; ++b) {
      dequantize_q4_k_block(raw_data.data() + b * type_size,
                            fp32_data.data() + b * block_size, block_size);
    }
    upload_data = fp32_data.data();
    upload_size = n_elements * sizeof(float);
  } else if (gtype == GGMLType::Q6_K) {
    // Dequantize Q6_K to FP32
    fp32_data.resize(n_elements);
    size_t block_size = ggml_block_size(gtype);
    size_t type_size = ggml_type_size(gtype);
    size_t n_blocks = n_elements / block_size;

    for (size_t b = 0; b < n_blocks; ++b) {
      dequantize_q6_k_block(raw_data.data() + b * type_size,
                            fp32_data.data() + b * block_size, block_size);
    }
    upload_data = fp32_data.data();
    upload_size = n_elements * sizeof(float);
  } else {
    *err = "Unsupported quantization type for tensor: " + name + " (" +
           info->dtype + ")";
    return false;
  }

  // Allocate GPU buffer
  if (!buffer.allocate(upload_size, gcore::rt::hip::BufferUsage::DeviceOnly,
                       err)) {
    return false;
  }

  // Copy dequantized data to GPU
  if (!buffer.copy_to_device(upload_data, upload_size, err)) {
    return false;
  }

  return true;
}

// Helper to convert FP32 to FP16
static uint16_t fp32_to_fp16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);

  uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFF;

  if (exp <= 0) {
    return sign; // Zero or denorm
  } else if (exp >= 31) {
    return sign | 0x7C00; // Infinity/NaN
  }

  return sign | (exp << 10) | (mant >> 13);
}

bool GGUFLoader::load_tensor_fp16(const std::string &name,
                                  gcore::rt::hip::Buffer &buffer,
                                  std::string *err) {
  // Find tensor
  const TensorInfo *info = nullptr;
  GGMLType gtype = GGMLType::F32;
  for (const auto &t : impl_->tensors) {
    if (t.name == name) {
      info = &t;
      gtype = ggml_type_from_name(t.dtype);
      break;
    }
  }
  if (!info) {
    *err = "Tensor not found: " + name;
    return false;
  }

  // Read raw data
  std::vector<uint8_t> raw_data(info->size_bytes);
  impl_->file.seekg(info->offset);
  impl_->file.read(reinterpret_cast<char *>(raw_data.data()), info->size_bytes);

  // Calculate elements
  size_t n_elements = 1;
  for (auto d : info->shape)
    n_elements *= d;

  // Dequantize to FP16
  std::vector<uint16_t> fp16_data(n_elements);
  std::vector<float> temp_fp32; // Temp buffer for intermediate dequant

  if (gtype == GGMLType::F32) {
    // Convert FP32 to FP16
    const float *src = reinterpret_cast<const float *>(raw_data.data());
    for (size_t i = 0; i < n_elements; ++i) {
      fp16_data[i] = fp32_to_fp16(src[i]);
    }
  } else if (gtype == GGMLType::F16) {
    // Already FP16, copy directly
    std::memcpy(fp16_data.data(), raw_data.data(), n_elements * 2);
  } else if (gtype == GGMLType::Q4_K) {
    // Dequantize Q4_K to FP32, then convert to FP16
    temp_fp32.resize(n_elements);
    size_t block_size = ggml_block_size(gtype);
    size_t type_size = ggml_type_size(gtype);
    size_t n_blocks = n_elements / block_size;

    for (size_t b = 0; b < n_blocks; ++b) {
      dequantize_q4_k_block(raw_data.data() + b * type_size,
                            temp_fp32.data() + b * block_size, block_size);
    }
    // Convert to FP16
    for (size_t i = 0; i < n_elements; ++i) {
      fp16_data[i] = fp32_to_fp16(temp_fp32[i]);
    }
  } else if (gtype == GGMLType::Q6_K) {
    // Dequantize Q6_K to FP32, then convert to FP16
    temp_fp32.resize(n_elements);
    size_t block_size = ggml_block_size(gtype);
    size_t type_size = ggml_type_size(gtype);
    size_t n_blocks = n_elements / block_size;

    for (size_t b = 0; b < n_blocks; ++b) {
      dequantize_q6_k_block(raw_data.data() + b * type_size,
                            temp_fp32.data() + b * block_size, block_size);
    }
    // Convert to FP16
    for (size_t i = 0; i < n_elements; ++i) {
      fp16_data[i] = fp32_to_fp16(temp_fp32[i]);
    }
  } else {
    *err = "Unsupported type for FP16 load: " + name + " (" + info->dtype + ")";
    return false;
  }

  // Transpose if it's a 2D weight matrix (Linear Layers)
  // GGUF layout is (Inner, Outer) -> data is Outer * Inner
  // Our kernels expect (Outer, Inner) relative to shared memory tiles?
  // Wait, let's be precise:
  // kernels expect b[k * N + n].
  // existing memory is w[n * K + k].
  // So yes, we need to map w[n * K + k] -> b[k * N + n].
  if (info->shape.size() == 2) {
    size_t K = info->shape[0];
    size_t N = info->shape[1];
    std::vector<uint16_t> transposed(n_elements);
    for (size_t n = 0; n < N; ++n) {
      for (size_t k = 0; k < K; ++k) {
        transposed[k * N + n] = fp16_data[n * K + k];
      }
    }
    fp16_data = std::move(transposed);
  }

  // Allocate GPU buffer (FP16 size = n_elements * 2 bytes)
  size_t upload_size = n_elements * sizeof(uint16_t);
  if (!buffer.allocate(upload_size, gcore::rt::hip::BufferUsage::DeviceOnly,
                       err)) {
    return false;
  }

  // Copy to GPU
  if (!buffer.copy_to_device(fp16_data.data(), upload_size, err)) {
    return false;
  }

  return true;
}

ModelConfig GGUFLoader::get_config() const { return impl_->config; }

SafeTensorsLoader::SafeTensorsLoader() {}
SafeTensorsLoader::~SafeTensorsLoader() = default;
bool SafeTensorsLoader::open(const std::string &path, std::string *err) {
  return false;
}
std::vector<TensorInfo> SafeTensorsLoader::list_tensors() const { return {}; }
bool SafeTensorsLoader::load_tensor(const std::string &name,
                                    gcore::rt::hip::Buffer &buffer,
                                    std::string *err) {
  return false;
}
bool SafeTensorsLoader::load_tensor_fp16(const std::string &name,
                                         gcore::rt::hip::Buffer &buffer,
                                         std::string *err) {
  return false;
}
ModelConfig SafeTensorsLoader::get_config() const {
  return ModelConfig::llama2_7b();
}

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
