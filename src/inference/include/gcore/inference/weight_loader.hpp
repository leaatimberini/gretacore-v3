#pragma once

#include "gcore/inference/model_config.hpp"
#include "gcore/rt/hip/buffer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gcore::inference {

/// Tensor metadata from a weight file.
struct TensorInfo {
  std::string name;
  std::vector<size_t> shape;
  size_t offset = 0;
  size_t size_bytes = 0;
  std::string dtype; // "F32", "F16", "BF16", etc.
};

/// Abstract interface for weight loading.
class WeightLoader {
public:
  virtual ~WeightLoader() = default;

  /// Open a weight file and parse metadata.
  virtual bool open(const std::string &path, std::string *err) = 0;

  /// Get list of all tensors in the file.
  virtual std::vector<TensorInfo> list_tensors() const = 0;

  /// Load a tensor into a pre-allocated GPU buffer.
  virtual bool load_tensor(const std::string &name,
                           gcore::rt::hip::Buffer &buffer,
                           std::string *err) = 0;

  /// Load a tensor as FP16.
  virtual bool load_tensor_fp16(const std::string &name,
                                gcore::rt::hip::Buffer &buffer,
                                std::string *err) = 0;

  /// Load a tensor as INT8 with scales.
  virtual bool load_tensor_int8(const std::string &name,
                                gcore::rt::hip::Buffer &buffer,
                                gcore::rt::hip::Buffer &scales,
                                std::string *err) = 0;

  /// Load a tensor as INT4 with scales (packed 2 values per byte).
  virtual bool load_tensor_int4(const std::string &name,
                                gcore::rt::hip::Buffer &buffer,
                                gcore::rt::hip::Buffer &scales,
                                gcore::rt::hip::Buffer &head_scales,
                                std::string *err) = 0;

  /// Get model configuration (if embedded in file).
  virtual ModelConfig get_config() const = 0;
};

/// GGUF format weight loader (llama.cpp compatible).
class GGUFLoader : public WeightLoader {
public:
  GGUFLoader();
  ~GGUFLoader() override;

  bool open(const std::string &path, std::string *err) override;
  std::vector<TensorInfo> list_tensors() const override;
  bool load_tensor(const std::string &name, gcore::rt::hip::Buffer &buffer,
                   std::string *err) override;
  bool load_tensor_fp16(const std::string &name, gcore::rt::hip::Buffer &buffer,
                        std::string *err) override;
  bool load_tensor_int8(const std::string &name, gcore::rt::hip::Buffer &buffer,
                        gcore::rt::hip::Buffer &scales,
                        std::string *err) override;
  bool load_tensor_int4(const std::string &name, gcore::rt::hip::Buffer &buffer,
                        gcore::rt::hip::Buffer &scales,
                        gcore::rt::hip::Buffer &head_scales,
                        std::string *err) override;
  ModelConfig get_config() const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// SafeTensors format weight loader.
class SafeTensorsLoader : public WeightLoader {
public:
  SafeTensorsLoader();
  ~SafeTensorsLoader() override;

  bool open(const std::string &path, std::string *err) override;
  std::vector<TensorInfo> list_tensors() const override;
  bool load_tensor(const std::string &name, gcore::rt::hip::Buffer &buffer,
                   std::string *err) override;
  bool load_tensor_fp16(const std::string &name, gcore::rt::hip::Buffer &buffer,
                        std::string *err) override;
  bool load_tensor_int8(const std::string &name, gcore::rt::hip::Buffer &buffer,
                        gcore::rt::hip::Buffer &scales,
                        std::string *err) override;
  bool load_tensor_int4(const std::string &name, gcore::rt::hip::Buffer &buffer,
                        gcore::rt::hip::Buffer &scales,
                        gcore::rt::hip::Buffer &head_scales,
                        std::string *err) override;
  ModelConfig get_config() const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Factory function to create appropriate loader based on file extension.
std::unique_ptr<WeightLoader> create_weight_loader(const std::string &path,
                                                   std::string *err);

} // namespace gcore::inference
