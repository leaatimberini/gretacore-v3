#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gcore::inference {

/// Simple BPE Tokenizer for Llama-style models.
/// Supports loading vocabulary from tokenizer.model (SentencePiece) or
/// vocab.json.
class Tokenizer {
public:
  Tokenizer();
  ~Tokenizer();

  /// Load vocabulary from a file.
  /// Supports: tokenizer.model (SentencePiece), tokenizer.json (HF), vocab.txt
  bool load(const std::string &path, std::string *err);

  /// Set the vocabulary directly (e.g., from GGUF)
  void set_vocabulary(const std::vector<std::string> &vocab) { vocab_ = vocab; }

  /// Encode text to token IDs.
  std::vector<int32_t> encode(const std::string &text) const;

  /// Decode token IDs to text.
  std::string decode(const std::vector<int32_t> &tokens) const;

  /// Decode a single token ID to text.
  std::string decode_token(int32_t token_id) const;

  /// Get vocabulary size.
  size_t vocab_size() const { return vocab_.size(); }

  /// Special token IDs
  int32_t bos_id() const { return bos_id_; }
  int32_t eos_id() const { return eos_id_; }
  int32_t pad_id() const { return pad_id_; }
  int32_t unk_id() const { return unk_id_; }

private:
  // Vocabulary: token_id -> token_string
  std::vector<std::string> vocab_;

  // Reverse mapping: token_string -> token_id
  // Using simple linear search for now (small vocab)

  // Merge rules for BPE (pair -> merged_token)
  std::vector<std::pair<std::string, std::string>> merges_;

  // Special tokens
  int32_t bos_id_ = 1; // <s>
  int32_t eos_id_ = 2; // </s>
  int32_t pad_id_ = 0; // <pad>
  int32_t unk_id_ = 0; // <unk>

  // Internal helper: find token ID by string
  int32_t find_token(const std::string &token) const;

  // Internal helper: apply BPE merges
  std::vector<std::string> bpe_encode(const std::string &word) const;
};

} // namespace gcore::inference
