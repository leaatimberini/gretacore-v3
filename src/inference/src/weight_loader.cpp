#include "gcore/inference/weight_loader.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <omp.h>

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

    auto read_u32 = [&](uint32_t t, uint32_t &out) -> bool {
      if (t == 4) {
        uint32_t v = 0;
        file.read(reinterpret_cast<char *>(&v), 4);
        out = v;
        return static_cast<bool>(file);
      } else if (t == 5) {
        int32_t v = 0;
        file.read(reinterpret_cast<char *>(&v), 4);
        out = static_cast<uint32_t>(v);
        return static_cast<bool>(file);
      } else if (t == 2) {
        uint16_t v = 0;
        file.read(reinterpret_cast<char *>(&v), 2);
        out = v;
        return static_cast<bool>(file);
      } else if (t == 3) {
        int16_t v = 0;
        file.read(reinterpret_cast<char *>(&v), 2);
        out = static_cast<uint32_t>(v);
        return static_cast<bool>(file);
      } else if (t == 6) {
        float v = 0.0f;
        file.read(reinterpret_cast<char *>(&v), 4);
        out = static_cast<uint32_t>(v);
        return static_cast<bool>(file);
      } else if (t == 7) {
        double v = 0.0;
        file.read(reinterpret_cast<char *>(&v), 8);
        out = static_cast<uint32_t>(v);
        return static_cast<bool>(file);
      }
      return false;
    };

    auto read_f32 = [&](uint32_t t, float &out) -> bool {
      if (t == 6) {
        float v = 0.0f;
        file.read(reinterpret_cast<char *>(&v), 4);
        out = v;
        return static_cast<bool>(file);
      } else if (t == 7) {
        double v = 0.0;
        file.read(reinterpret_cast<char *>(&v), 8);
        out = static_cast<float>(v);
        return static_cast<bool>(file);
      }
      return false;
    };

    if (key == "llama.embedding_length") {
      uint32_t v = 0;
      if (!read_u32(val_type, v))
        return false;
      config.dim = v;
      if (config.num_heads > 0)
        config.head_dim = config.dim / config.num_heads;
      return true;
    }
    if (key == "llama.feed_forward_length") {
      uint32_t v = 0;
      if (!read_u32(val_type, v))
        return false;
      config.hidden_dim = v;
      return true;
    }
    if (key == "llama.block_count") {
      uint32_t v = 0;
      if (!read_u32(val_type, v))
        return false;
      config.num_layers = v;
      return true;
    }
    if (key == "llama.attention.head_count") {
      uint32_t v = 0;
      if (!read_u32(val_type, v))
        return false;
      config.num_heads = v;
      if (config.num_heads_kv == 0)
        config.num_heads_kv = v;
      if (config.num_heads > 0)
        config.head_dim = config.dim / config.num_heads;
      return true;
    }
    if (key == "llama.attention.head_count_kv") {
      uint32_t v = 0;
      if (!read_u32(val_type, v))
        return false;
      config.num_heads_kv = v;
      return true;
    }
    if (key == "llama.context_length") {
      uint32_t v = 0;
      if (!read_u32(val_type, v))
        return false;
      config.max_seq_len = v;
      return true;
    }
    if (key == "llama.rope.freq_base") {
      float v = 0.0f;
      if (!read_f32(val_type, v))
        return false;
      config.rope_base = v;
      return true;
    }
    if (key == "llama.norm_eps") {
      float v = 0.0f;
      if (!read_f32(val_type, v))
        return false;
      config.rms_eps = v;
      return true;
    }

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
    info.offset = offset; // Relative to data section

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
    for (auto &t : tensors) {
      t.offset += data_offset;
    } // Map relative to absolute
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
  } else if (gtype == GGMLType::Q6_K) {
    fp32.resize(n_elem);
    size_t bs = 256, ts = 210, nb = n_elem / bs;
    for (size_t b = 0; b < nb; ++b)
      dequantize_q6_k_block(raw.data() + b * ts, fp32.data() + b * bs, bs);
    up = fp32.data();
    ups = n_elem * 4;
  } else
    return false;
  if (!buffer.allocate(ups, gcore::rt::hip::BufferUsage::DeviceOnly,
                       gcore::rt::GretaDataType::FP32, err))
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
  } else if (gtype == GGMLType::Q6_K) {
    std::vector<float> tmp(n_elem);
    size_t bs = 256, ts = 210, nb = n_elem / bs;
    for (size_t b = 0; b < nb; ++b)
      dequantize_q6_k_block(raw.data() + b * ts, tmp.data() + b * bs, bs);
    for (size_t i = 0; i < n_elem; ++i)
      fp16[i] = fp32_to_fp16(tmp[i]);
  } else
    return false;

  const bool is_kv_weight =
      (name.find("attn_k.weight") != std::string::npos ||
       name.find("attn_v.weight") != std::string::npos);
  if (is_kv_weight && impl_->config.num_heads_kv > 0 &&
      impl_->config.head_dim > 0) {
    const uint32_t kv_dim = impl_->config.num_heads_kv * impl_->config.head_dim;
    const uint32_t model_dim = impl_->config.dim;
    if (it->shape.size() != 2 || kv_dim == 0 || model_dim == 0) {
      if (err) {
        *err = "GQA KV load failed for " + name +
               ": unexpected shape dims or config mismatch. got [" +
               (it->shape.size() > 0 ? std::to_string(it->shape[0]) : "?") +
               "," +
               (it->shape.size() > 1 ? std::to_string(it->shape[1]) : "?") +
               "]";
      }
      return false;
    }

    if (it->shape[0] == model_dim && it->shape[1] == kv_dim) {
      std::vector<uint16_t> transposed((size_t)kv_dim * (size_t)model_dim, 0);
      for (uint32_t r = 0; r < kv_dim; ++r) {
        for (uint32_t c = 0; c < model_dim; ++c) {
          transposed[(size_t)r * model_dim + c] =
              fp16[(size_t)c * kv_dim + r];
        }
      }
      fp16.swap(transposed);
      n_elem = fp16.size();
      std::cout << "[GRETA_LOAD] Transposed " << name
                << " from [D, KV] to [KV, D]" << std::endl;
    } else if (it->shape[0] == kv_dim && it->shape[1] == model_dim) {
      // Already in expected [KV, D] layout.
    } else if (kv_dim == model_dim && it->shape[0] == model_dim &&
               it->shape[1] == model_dim) {
      // Standard MHA layout, no action.
    } else {
      if (err) {
        *err = "GQA KV load failed for " + name +
               ": unsupported shape [" + std::to_string(it->shape[0]) + "," +
               std::to_string(it->shape[1]) + "]";
      }
      return false;
    }
  }

  size_t ups = n_elem * 2;
  if (!buffer.allocate(ups, gcore::rt::hip::BufferUsage::DeviceOnly,
                       gcore::rt::GretaDataType::FP16, err))
    return false;
  return buffer.copy_to_device(fp16.data(), ups, err);
}

bool GGUFLoader::load_tensor_int8(const std::string &name,
                                  gcore::rt::hip::Buffer &buffer,
                                  gcore::rt::hip::Buffer &scales,
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
  std::cout << "[GRETA_LOAD] Loading tensor: " << name
            << " (Type: " << it->dtype << ", Size: " << it->size_bytes
            << " bytes)" << std::endl;
  static bool omp_logged = false;
  if (!omp_logged) {
    std::cout << "[GRETA_LOAD] OpenMP Max Threads: " << omp_get_max_threads()
              << std::endl;
    omp_logged = true;
  }
  std::vector<uint8_t> raw(it->size_bytes);
  impl_->file.seekg(it->offset);
  impl_->file.read((char *)raw.data(), it->size_bytes);

  size_t n_elem = 1;
  for (auto d : it->shape)
    n_elem *= d;

  std::vector<int8_t> weights(n_elem);
  std::vector<float> scale_data;
  uint32_t group_size = 32;

  if (gtype == GGMLType::Q8_0) {
    size_t nb = n_elem / 32;
    scale_data.resize(nb);
    for (size_t b = 0; b < nb; ++b) {
      const uint8_t *src = raw.data() + b * 34;
      uint16_t d_raw;
      std::memcpy(&d_raw, src, 2);
      scale_data[b] = fp16_to_fp32(d_raw);
      std::memcpy(weights.data() + b * 32, src + 2, 32);
    }
  } else {
    // Convert FP32/FP16/Other to INT8 with scales
    std::vector<float> fp32(n_elem);
    const bool is_kv_weight =
        (name.find("attn_k.weight") != std::string::npos ||
         name.find("attn_v.weight") != std::string::npos);
    if (gtype == GGMLType::F32) {
      std::memcpy(fp32.data(), raw.data(), n_elem * 4);
    } else if (gtype == GGMLType::F16) {
      const uint16_t *s = (const uint16_t *)raw.data();
      for (size_t i = 0; i < n_elem; ++i)
        fp32[i] = fp16_to_fp32(s[i]);
    } else if (gtype == GGMLType::Q4_K) {
      size_t bs = 256, ts = 144, nb = n_elem / bs;
      for (size_t b = 0; b < nb; ++b)
        dequantize_q4_k_block(raw.data() + b * ts, fp32.data() + b * bs, bs);
    } else if (gtype == GGMLType::Q6_K) {
      size_t bs = 256, ts = 210, nb = n_elem / bs;
      for (size_t b = 0; b < nb; ++b)
        dequantize_q6_k_block(raw.data() + b * ts, fp32.data() + b * bs, bs);
    } else {
      if (err)
        *err = "Unsupported INT8 conversion for type " + it->dtype;
      return false;
    }

    if (is_kv_weight && impl_->config.num_heads_kv > 0 &&
        impl_->config.head_dim > 0) {
      const uint32_t kv_dim =
          impl_->config.num_heads_kv * impl_->config.head_dim;
      const uint32_t model_dim = impl_->config.dim;
      if (it->shape.size() != 2 || kv_dim == 0 || model_dim == 0) {
        if (err) {
          *err = "GQA KV INT8 load failed for " + name +
                 ": unexpected shape dims or config mismatch. got [" +
                 (it->shape.size() > 0 ? std::to_string(it->shape[0]) : "?") +
                 "," +
                 (it->shape.size() > 1 ? std::to_string(it->shape[1]) : "?") +
                 "]";
        }
        return false;
      }
      if (it->shape[0] == model_dim && it->shape[1] == kv_dim) {
        std::vector<float> transposed((size_t)kv_dim * (size_t)model_dim,
                                      0.0f);
        for (uint32_t r = 0; r < kv_dim; ++r) {
          for (uint32_t c = 0; c < model_dim; ++c) {
            transposed[(size_t)r * model_dim + c] =
                fp32[(size_t)c * kv_dim + r];
          }
        }
        fp32.swap(transposed);
        n_elem = fp32.size();
        std::cout << "[GRETA_LOAD] Transposed " << name
                  << " from [D, KV] to [KV, D]" << std::endl;
      } else if (it->shape[0] == kv_dim && it->shape[1] == model_dim) {
        // Already in expected [KV, D] layout.
      } else if (kv_dim == model_dim && it->shape[0] == model_dim &&
                 it->shape[1] == model_dim) {
        // Standard MHA layout, no action.
      } else {
        if (err) {
          *err = "GQA KV INT8 load failed for " + name +
                 ": unsupported shape [" + std::to_string(it->shape[0]) + "," +
                 std::to_string(it->shape[1]) + "]";
        }
        return false;
      }
    }

    size_t nb = (n_elem + 31) / 32;
    scale_data.resize(nb);
    std::cout << "[GRETA_LOAD] Quantizing " << n_elem << " elements to INT8..."
              << std::endl;
#pragma omp parallel for
    for (size_t b = 0; b < nb; ++b) {
      if (b % 100000 == 0 && b > 0)
        std::cout << "  - Block " << b << "/" << nb << std::endl;
      float max_val = 0.0f;
      for (size_t i = 0; i < 32 && (b * 32 + i) < n_elem; ++i) {
        max_val = std::max(max_val, std::abs(fp32[b * 32 + i]));
      }
      float scale = max_val / 127.0f;
      scale_data[b] = scale;
      float inv_scale = scale > 1e-9f ? 1.0f / scale : 0.0f;
      for (size_t i = 0; i < 32 && (b * 32 + i) < n_elem; ++i) {
        weights[b * 32 + i] = (int8_t)std::round(fp32[b * 32 + i] * inv_scale);
      }
    }
    std::cout << "[GRETA_LOAD] Quantization complete." << std::endl;
  }

  if (!buffer.allocate(n_elem, rt::hip::BufferUsage::DeviceOnly,
                       rt::GretaDataType::INT8, err))
    return false;
  if (!scales.allocate(scale_data.size() * 4, rt::hip::BufferUsage::DeviceOnly,
                       rt::GretaDataType::FP32, err))
    return false;

  if (!buffer.copy_to_device(weights.data(), n_elem, err))
    return false;
  if (!scales.copy_to_device(scale_data.data(), scale_data.size() * 4, err))
    return false;

  gcore::rt::GretaQuantInfo qinfo;
  qinfo.group_size = group_size;
  qinfo.scales = scales.data();
  buffer.set_quant_info(qinfo);

  return true;
}

bool GGUFLoader::load_tensor_int4(const std::string &name,
                                  gcore::rt::hip::Buffer &buffer,
                                  gcore::rt::hip::Buffer &scales,
                                  gcore::rt::hip::Buffer &head_scales,
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
  std::cout << "[GRETA_LOAD] Loading tensor (INT4): " << name
            << " (Type: " << it->dtype << ", Size: " << it->size_bytes
            << " bytes)" << std::endl;

  std::vector<uint8_t> raw(it->size_bytes);
  impl_->file.seekg(it->offset);
  impl_->file.read((char *)raw.data(), it->size_bytes);

  size_t n_elem = 1;
  for (auto d : it->shape)
    n_elem *= d;

  // 1. Dequantize to FP32
  std::vector<float> fp32(n_elem);
  const bool is_kv_weight =
      (name.find("attn_k.weight") != std::string::npos ||
       name.find("attn_v.weight") != std::string::npos);
  if (gtype == GGMLType::F32) {
    std::memcpy(fp32.data(), raw.data(), n_elem * 4);
  } else if (gtype == GGMLType::F16) {
    const uint16_t *s = (const uint16_t *)raw.data();
    for (size_t i = 0; i < n_elem; ++i)
      fp32[i] = fp16_to_fp32(s[i]);
  } else if (gtype == GGMLType::Q4_K) {
    size_t bs = 256, ts = 144, nb = n_elem / bs;
    for (size_t b = 0; b < nb; ++b)
      dequantize_q4_k_block(raw.data() + b * ts, fp32.data() + b * bs, bs);
  } else if (gtype == GGMLType::Q6_K) {
    size_t bs = 256, ts = 210, nb = n_elem / bs;
    for (size_t b = 0; b < nb; ++b)
      dequantize_q6_k_block(raw.data() + b * ts, fp32.data() + b * bs, bs);
  } else {
    if (err)
      *err = "Unsupported INT4 conversion for type " + it->dtype;
    return false;
  }

  if (is_kv_weight && impl_->config.num_heads_kv > 0 &&
      impl_->config.head_dim > 0) {
    const uint32_t kv_dim =
        impl_->config.num_heads_kv * impl_->config.head_dim;
    const uint32_t model_dim = impl_->config.dim;
    if (it->shape.size() != 2 || kv_dim == 0 || model_dim == 0) {
      if (err) {
        *err = "GQA KV INT4 load failed for " + name +
               ": unexpected shape dims or config mismatch. got [" +
               (it->shape.size() > 0 ? std::to_string(it->shape[0]) : "?") +
               "," +
               (it->shape.size() > 1 ? std::to_string(it->shape[1]) : "?") +
               "]";
      }
      return false;
    }
    if (it->shape[0] == model_dim && it->shape[1] == kv_dim) {
      std::vector<float> transposed((size_t)kv_dim * (size_t)model_dim, 0.0f);
      for (uint32_t r = 0; r < kv_dim; ++r) {
        for (uint32_t c = 0; c < model_dim; ++c) {
          transposed[(size_t)r * model_dim + c] =
              fp32[(size_t)c * kv_dim + r];
        }
      }
      fp32.swap(transposed);
      n_elem = fp32.size();
      std::cout << "[GRETA_LOAD] Transposed " << name
                << " from [D, KV] to [KV, D]" << std::endl;
    } else if (it->shape[0] == kv_dim && it->shape[1] == model_dim) {
      // Already in expected [KV, D] layout.
    } else if (kv_dim == model_dim && it->shape[0] == model_dim &&
               it->shape[1] == model_dim) {
      // Standard MHA layout, no action.
    } else {
      if (err) {
        *err = "GQA KV INT4 load failed for " + name +
               ": unsupported shape [" + std::to_string(it->shape[0]) + "," +
               std::to_string(it->shape[1]) + "]";
      }
      return false;
    }
  }

  // 2. Quantize to INT4 and Pack
  size_t n_groups = (n_elem + 31) / 32;
  std::vector<float> scale_data(n_groups);
  std::vector<uint8_t> packed_weights((n_elem + 1) / 2);

  std::cout << "[GRETA_LOAD] Quantizing " << n_elem << " elements to INT4..."
            << std::endl;

#pragma omp parallel for
  for (size_t g = 0; g < n_groups; ++g) {
    if (g % 200000 == 0 && g > 0)
      std::cout << "  - Group " << g << "/" << n_groups << std::endl;

    float max_val = 0.0f;
    for (size_t i = 0; i < 32 && (g * 32 + i) < n_elem; ++i) {
      max_val = std::max(max_val, std::abs(fp32[g * 32 + i]));
    }

    float scale = max_val / 7.0f;
    scale_data[g] = scale;
    float inv_scale = (scale > 1e-9f) ? 1.0f / scale : 0.0f;

    for (size_t i = 0; i < 32 && (g * 32 + i) < n_elem; i += 2) {
      size_t idx0 = g * 32 + i;
      size_t idx1 = idx0 + 1;

      int8_t v0 = (int8_t)std::round(fp32[idx0] * inv_scale);
      v0 = std::max((int8_t)-8, std::min((int8_t)7, v0));

      int8_t v1 = 0;
      if (idx1 < n_elem) {
        v1 = (int8_t)std::round(fp32[idx1] * inv_scale);
        v1 = std::max((int8_t)-8, std::min((int8_t)7, v1));
      }

      packed_weights[idx0 / 2] = (v0 & 0x0F) | ((v1 & 0x0F) << 4);
    }
  }

  // 3. Upload to Device
  if (!buffer.allocate(packed_weights.size(), rt::hip::BufferUsage::DeviceOnly,
                       rt::GretaDataType::INT4, err))
    return false;
  if (!scales.allocate(scale_data.size() * 4, rt::hip::BufferUsage::DeviceOnly,
                       rt::GretaDataType::FP32, err))
    return false;

  if (!buffer.copy_to_device(packed_weights.data(), packed_weights.size(), err))
    return false;
  if (!scales.copy_to_device(scale_data.data(), scale_data.size() * 4, err))
    return false;

  // 4. Per-head Scaling (Phase 5.3)
  uint32_t num_heads = impl_->config.num_heads;
  if (is_kv_weight && impl_->config.num_heads_kv > 0)
    num_heads = impl_->config.num_heads_kv;
  const uint32_t head_dim = impl_->config.head_dim;
  std::vector<float> h_scales(num_heads, 1.0f);
  bool is_qkv = (name.find("attn_q") != std::string::npos ||
                 name.find("attn_k") != std::string::npos ||
                 name.find("attn_v") != std::string::npos);

  if (is_qkv && num_heads > 0 && head_dim > 0) {
    size_t D = impl_->config.dim;
    size_t Dh = head_dim;
#pragma omp parallel for
    for (uint32_t h = 0; h < num_heads; ++h) {
      float h_max = 0.0f;
      for (size_t i = 0; i < Dh * D; ++i) {
        h_max = std::max(h_max, std::abs(fp32[h * Dh * D + i]));
      }
      h_scales[h] = h_max > 1e-9f ? h_max : 1.0f;
    }
    if (!head_scales.allocate(num_heads * 4, rt::hip::BufferUsage::DeviceOnly,
                              rt::GretaDataType::FP32, err))
      return false;
    if (!head_scales.copy_to_device(h_scales.data(), num_heads * 4, err))
      return false;
  }

  gcore::rt::GretaQuantInfo qinfo;
  qinfo.group_size = 32;
  qinfo.scales = scales.data();
  qinfo.head_scales = is_qkv ? head_scales.data() : nullptr;
  qinfo.num_heads = is_qkv ? num_heads : 0;
  buffer.set_quant_info(qinfo);

  return true;
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
bool SafeTensorsLoader::load_tensor_int8(const std::string &name,
                                         gcore::rt::hip::Buffer &buffer,
                                         gcore::rt::hip::Buffer &scales,
                                         std::string *err) {
  return false;
}
bool SafeTensorsLoader::load_tensor_int4(const std::string &name,
                                         gcore::rt::hip::Buffer &buffer,
                                         gcore::rt::hip::Buffer &scales,
                                         gcore::rt::hip::Buffer &head_scales,
                                         std::string *err) {
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
