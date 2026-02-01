#include "gcore/inference/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>

namespace gcore::inference {

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;

bool Tokenizer::load(const std::string &path, std::string *err) {
  // Detect file type and load accordingly
  if (path.find(".json") != std::string::npos) {
    // Load HuggingFace tokenizer.json (simplified)
    std::ifstream file(path);
    if (!file.is_open()) {
      *err = "Failed to open tokenizer file: " + path;
      return false;
    }

    // Simple JSON parsing for vocab (production would use proper JSON)
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    // For now, create a dummy vocabulary
    // Real implementation would parse the JSON
    vocab_.resize(32000);
    for (size_t i = 0; i < vocab_.size(); ++i) {
      vocab_[i] = "<tok_" + std::to_string(i) + ">";
    }

    // Special tokens
    vocab_[0] = "<unk>";
    vocab_[1] = "<s>";
    vocab_[2] = "</s>";

    bos_id_ = 1;
    eos_id_ = 2;
    unk_id_ = 0;
    pad_id_ = 0;

    return true;
  }

  // Fallback: create minimal vocabulary
  vocab_.resize(32000);
  for (size_t i = 0; i < 256; ++i) {
    vocab_[i] = std::string(1, static_cast<char>(i));
  }
  vocab_[0] = "<unk>";
  vocab_[1] = "<s>";
  vocab_[2] = "</s>";

  // ASCII printable characters as tokens (for testing)
  for (int c = 32; c < 127; ++c) {
    vocab_[c] = std::string(1, static_cast<char>(c));
  }

  bos_id_ = 1;
  eos_id_ = 2;
  unk_id_ = 0;
  pad_id_ = 0;

  return true;
}

int32_t Tokenizer::find_token(const std::string &token) const {
  for (size_t i = 0; i < vocab_.size(); ++i) {
    if (vocab_[i] == token) {
      return static_cast<int32_t>(i);
    }
  }
  return unk_id_;
}

std::vector<std::string> Tokenizer::bpe_encode(const std::string &word) const {
  // Simple character-level fallback (no actual BPE merges)
  std::vector<std::string> tokens;
  for (char c : word) {
    tokens.push_back(std::string(1, c));
  }
  return tokens;
}

std::vector<int32_t> Tokenizer::encode(const std::string &text) const {
  std::vector<int32_t> result;

  // Add BOS token
  result.push_back(bos_id_);

  // Simple character-level encoding for now
  for (char c : text) {
    unsigned char uc = static_cast<unsigned char>(c);
    if (uc < vocab_.size()) {
      result.push_back(static_cast<int32_t>(uc));
    } else {
      result.push_back(unk_id_);
    }
  }

  return result;
}

std::string Tokenizer::decode(const std::vector<int32_t> &tokens) const {
  std::string result;
  for (int32_t id : tokens) {
    // Skip special tokens
    if (id == bos_id_ || id == eos_id_ || id == pad_id_) {
      continue;
    }
    result += decode_token(id);
  }
  return result;
}

std::string Tokenizer::decode_token(int32_t token_id) const {
  if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_.size()) {
    return "[ID_" + std::to_string(token_id) + "]";
  }
  std::string t = vocab_[token_id];
  if (t.empty())
    return "[ID_" + std::to_string(token_id) + "]";
  return t;
}

} // namespace gcore::inference
