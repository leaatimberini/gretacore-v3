#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gcore::inference {

/// Tokenizer wrapper that can use SentencePiece or fallback to ASCII.
class Tokenizer {
public:
  Tokenizer();
  ~Tokenizer();

  /// Load a SentencePiece model (.model or .spm).
  /// If loading fails, falls back to ASCII tokenizer.
  bool load(const std::string &path, std::string *err);

  /// Force ASCII fallback mode (for --demo-tokenizer)
  void use_ascii_fallback();

  /// Set the vocabulary directly (e.g., from GGUF)
  void set_vocabulary(const std::vector<std::string> &vocab);

  /// Encode text to token IDs.
  std::vector<int32_t> encode(const std::string &text) const;

  /// Decode token IDs to text.
  std::string decode(const std::vector<int32_t> &tokens) const;

  /// Decode a single token ID to text.
  std::string decode_token(int32_t token_id) const;

  /// Get vocabulary size.
  size_t vocab_size() const;

  /// Special token IDs
  int32_t bos_id() const;
  int32_t eos_id() const;

  /// Check if using real tokenizer vs fallback
  bool is_using_sentencepiece() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace gcore::inference
