#include "gcore/inference/weight_loader.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

namespace gcore::inference {

// GGUF Magic and Version
static constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"

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

static constexpr size_t QK_K = 256;
static constexpr size_t QK4_0 = 32;

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
    return 18;
  case GGMLType::Q4_1:
    return 20;
  case GGMLType::Q8_0:
    return 34;
  case GGMLType::Q4_K:
    return 144;
  case GGMLType::Q5_K:
    return 176;
  case GGMLType::Q6_K:
    return 210;
  default:
    return 0;
  }
}

struct block_q4_k {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[128];
};

static float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0) {
    if (mant == 0)
      return sign ? -0.0f : 0.0f;
    exp = 1;
    while ((mant & 0x400) == 0) {
      mant <<= 1;
      exp--;
    }
    mant &= ~0x400;
  } else if (exp == 31)
    return sign ? -INFINITY : INFINITY;
  uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  float result;
  memcpy(&result, &f, 4);
  return result;
}

static uint16_t fp32_to_fp16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFF;
  if (exp <= 0)
    return sign;
  else if (exp >= 31)
    return sign | 0x7C00;
  return sign | (exp << 10) | (mant >> 13);
}

static void dequantize_q4_k_block(const uint8_t *src, float *dst,
                                  size_t n_elements) {
  const block_q4_k *block = reinterpret_cast<const block_q4_k *>(src);
  float d = fp16_to_fp32(block->d);
  float dmin = fp16_to_fp32(block->dmin);
  uint8_t sc8[8], m8[8];
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
  for (int j = 0; j < 8; ++j) {
    float scale = d * sc8[j];
    float min_val = dmin * m8[j];
    for (int l = 0; l < 16; ++l) {
      uint8_t qs = block->qs[j * 16 + l];
      dst[j * 32 + l * 2 + 0] = scale * (qs & 0x0F) - min_val;
      dst[j * 32 + l * 2 + 1] = scale * (qs >> 4) - min_val;
    }
  }
}

static void dequantize_q6_k_block(const uint8_t *src, float *dst,
                                  size_t n_elements) {
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
    int q = (l_val | (h_val << 4)) - 32;
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
  case GGMLType::Q4_K:
    return "Q4_K";
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
  size_t data_offset = 0;

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
    case 8: {
      uint64_t len;
      file.read(reinterpret_cast<char *>(&len), 8);
      if (len > 1000000)
        return false;
      file.seekg(len, std::ios::cur);
      break;
    }
    case 9: {
      uint32_t arr_type;
      uint64_t arr_len;
      file.read(reinterpret_cast<char *>(&arr_type), 4);
      file.read(reinterpret_cast<char *>(&arr_len), 8);
      if (arr_len > 1000000)
        return false;
      if (arr_type == 8) {
        for (uint64_t i = 0; i < arr_len; ++i) {
          uint64_t slen;
          file.read(reinterpret_cast<char *>(&slen), 8);
          if (slen > 1000000)
            return false;
          file.seekg(slen, std::ios::cur);
        }
      } else {
        size_t es = (arr_type <= 1 || arr_type == 7)
                        ? 1
                        : (arr_type <= 3 ? 2 : (arr_type <= 6 ? 4 : 8));
        file.seekg(arr_len * es, std::ios::cur);
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

  bool parse_kv_pair(std::string *err) {
    uint64_t key_len;
    file.read(reinterpret_cast<char *>(&key_len), 8);
    if (!file || key_len > 1024)
      return false;
    std::string key(key_len, '\0');
    file.read(&key[0], key_len);
    uint32_t val_type;
    file.read(reinterpret_cast<char *>(&val_type), 4);
    if (!file)
      return false;
    if (key == "tokenizer.ggml.tokens" && val_type == 9) {
      uint32_t arr_type;
      uint64_t arr_len;
      file.read(reinterpret_cast<char *>(&arr_type), 4);
      file.read(reinterpret_cast<char *>(&arr_len), 8);
      if (arr_type != 8 || arr_len > 1000000)
        return false;
      config.vocabulary.clear();
      config.vocabulary.reserve(arr_len);
      for (uint64_t i = 0; i < arr_len; ++i) {
        uint64_t slen;
        file.read(reinterpret_cast<char *>(&slen), 8);
        if (slen > 1024) {
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
      if (!skip_value(val_type, err))
        return false;
    }
    return true;
  }

  bool parse_tensor_info(TensorInfo &info, std::string *err) {
    uint64_t name_len;
    file.read(reinterpret_cast<char *>(&name_len), 8);
    if (name_len > 512)
      return false;
    info.name.resize(name_len);
    file.read(&info.name[0], name_len);
    uint32_t n_dims;
    file.read(reinterpret_cast<char *>(&n_dims), 4);
    if (n_dims > 8)
      return false;
    info.shape.resize(n_dims);
    size_t n_elements = 1;
    for (uint32_t i = 0; i < n_dims; ++i) {
      uint64_t d;
      file.read(reinterpret_cast<char *>(&d), 8);
      info.shape[i] = d;
      n_elements *= d;
    }
    uint32_t type;
    file.read(reinterpret_cast<char *>(&type), 4);
    info.dtype = ggml_type_name(static_cast<GGMLType>(type));
    uint64_t offset;
    file.read(reinterpret_cast<char *>(&offset), 8);
    GGMLType gtype = static_cast<GGMLType>(type);
    size_t ts = ggml_type_size(gtype), bs = ggml_block_size(gtype);
    if (gtype <= GGMLType::F16)
      info.size_bytes = n_elements * ts;
    else if (bs > 0)
      info.size_bytes = ((n_elements + bs - 1) / bs) * ts;
    else
      info.size_bytes = n_elements * 2;
    return true;
  }

  bool parse_header(std::string *err) {
    file.seekg(0);
    char buf[4];
    file.read(buf, 4);
    if (std::memcmp(buf, "GGUF", 4) != 0) {
      *err = "Not GGUF";
      return false;
    }
    uint32_t version;
    file.read(reinterpret_cast<char *>(&version), 4);
    if (version < 2) {
      *err = "Old GGUF";
      return false;
    }
    uint64_t t_count, kv_count;
    file.read(reinterpret_cast<char *>(&t_count), 8);
    file.read(reinterpret_cast<char *>(&kv_count), 8);
    if (kv_count > 2000)
      return false;
    config = ModelConfig::llama2_7b();
    for (uint64_t i = 0; i < kv_count; ++i)
      if (!parse_kv_pair(err))
        return false;
    tensors.reserve(t_count);
    for (uint64_t i = 0; i < t_count; ++i) {
      TensorInfo info;
      if (!parse_tensor_info(info, err))
        return false;
      tensors.push_back(info);
    }
    size_t pos = file.tellg();
    data_offset = (pos + 31) & ~31ULL;
    size_t cur = data_offset;
    for (auto &t : tensors) {
      t.offset = cur;
      cur += (t.size_bytes + 31) & ~31ULL;
    }
    loaded = true;
    return true;
  }
};

GGUFLoader::GGUFLoader() : impl_(std::make_unique<Impl>()) {}
GGUFLoader::~GGUFLoader() = default;
bool GGUFLoader::open(const std::string &path, std::string *err) {
  impl_->path = path;
  impl_->file.open(path, std::ios::binary);
  if (!impl_->file.is_open())
    return false;
  return impl_->parse_header(err);
}
std::vector<TensorInfo> GGUFLoader::list_tensors() const {
  return impl_->tensors;
}
ModelConfig GGUFLoader::get_config() const { return impl_->config; }

bool GGUFLoader::load_tensor(const std::string &name,
                             gcore::rt::hip::Buffer &buffer, std::string *err) {
  const TensorInfo *it = nullptr;
  for (const auto &t : impl_->tensors)
    if (t.name == name) {
      it = &t;
      break;
    }
  if (!it)
    return false;
  GGMLType gtype = ggml_type_from_name(it->dtype);
  std::vector<uint8_t> raw(it->size_bytes);
  impl_->file.seekg(it->offset);
  impl_->file.read((char *)raw.data(), it->size_bytes);
  size_t n_elem = 1;
  for (auto d : it->shape)
    n_elem *= d;
  std::vector<float> fp32;
  const void *up = nullptr;
  size_t ups = 0;
  if (gtype == GGMLType::F32) {
    up = raw.data();
    ups = it->size_bytes;
  } else if (gtype == GGMLType::F16) {
    fp32.resize(n_elem);
    const uint16_t *s = (uint16_t *)raw.data();
    for (size_t i = 0; i < n_elem; ++i)
      fp32[i] = fp16_to_fp32(s[i]);
    up = fp32.data();
    ups = n_elem * 4;
  } else if (gtype == GGMLType::Q4_K) {
    fp32.resize(n_elem);
    size_t bs = 256, ts = 144, nb = n_elem / bs;
    for (size_t b = 0; b < nb; ++b)
      dequantize_q4_k_block(raw.data() + b * ts, fp32.data() + b * bs, bs);
    up = fp32.data();
    ups = n_elem * 4;
  } else
    return false;
  if (!buffer.allocate(ups, gcore::rt::hip::BufferUsage::DeviceOnly, err))
    return false;
  return buffer.copy_to_device(up, ups, err);
}

bool GGUFLoader::load_tensor_fp16(const std::string &name,
                                  gcore::rt::hip::Buffer &buffer,
                                  std::string *err) {
  const TensorInfo *it = nullptr;
  for (const auto &t : impl_->tensors)
    if (t.name == name) {
      it = &t;
      break;
    }
  if (!it)
    return false;
  GGMLType gtype = ggml_type_from_name(it->dtype);
  std::vector<uint8_t> raw(it->size_bytes);
  impl_->file.seekg(it->offset);
  impl_->file.read((char *)raw.data(), it->size_bytes);
  size_t n_elem = 1;
  for (auto d : it->shape)
    n_elem *= d;
  std::vector<uint16_t> fp16(n_elem);
  if (gtype == GGMLType::F32) {
    const float *s = (float *)raw.data();
    for (size_t i = 0; i < n_elem; ++i)
      fp16[i] = fp32_to_fp16(s[i]);
  } else if (gtype == GGMLType::F16)
    std::memcpy(fp16.data(), raw.data(), n_elem * 2);
  else if (gtype == GGMLType::Q4_K) {
    std::vector<float> tmp(n_elem);
    size_t bs = 256, ts = 144, nb = n_elem / bs;
    for (size_t b = 0; b < nb; ++b)
      dequantize_q4_k_block(raw.data() + b * ts, tmp.data() + b * bs, bs);
    for (size_t i = 0; i < n_elem; ++i)
      fp16[i] = fp32_to_fp16(tmp[i]);
  } else
    return false;
  if (it->shape.size() == 2) {
    size_t K = it->shape[0], N = it->shape[1];
    std::vector<uint16_t> tr(n_elem);
    for (size_t n = 0; n < N; ++n)
      for (size_t k = 0; k < K; ++k)
        tr[k * N + n] = fp16[n * K + k];
    fp16 = std::move(tr);
  }
  size_t ups = n_elem * 2;
  if (!buffer.allocate(ups, gcore::rt::hip::BufferUsage::DeviceOnly, err))
    return false;
  return buffer.copy_to_device(fp16.data(), ups, err);
}

struct SafeTensorsLoader::Impl {};
SafeTensorsLoader::SafeTensorsLoader() : impl_(std::make_unique<Impl>()) {}
SafeTensorsLoader::~SafeTensorsLoader() = default;
bool SafeTensorsLoader::open(const std::string &path, std::string *err) {
  return false;
}
std::vector<TensorInfo> SafeTensorsLoader::list_tensors() const { return {}; }
bool SafeTensorsLoader::load_tensor(const std::string &name,
                                    gcore::rt::hip::Buffer &b, std::string *e) {
  return false;
}
bool SafeTensorsLoader::load_tensor_fp16(const std::string &name,
                                         gcore::rt::hip::Buffer &b,
                                         std::string *e) {
  return false;
}
ModelConfig SafeTensorsLoader::get_config() const {
  return ModelConfig::llama2_7b();
}

std::unique_ptr<WeightLoader> create_weight_loader(const std::string &p,
                                                   std::string *e) {
  if (p.find(".gguf") != std::string::npos) {
    auto l = std::make_unique<GGUFLoader>();
    if (!l->open(p, e))
      return nullptr;
    return l;
  }
  *e = "Unsupported format";
  return nullptr;
}
} // namespace gcore::inference
