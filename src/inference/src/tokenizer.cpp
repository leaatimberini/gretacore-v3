#include "gcore/inference/tokenizer.hpp"

#ifdef GRETA_USE_SENTENCEPIECE
#include <sentencepiece_processor.h>
#endif

#include <cstdio>
#include <fstream>
#include <iostream>

namespace gcore::inference {

// ============================================================================
// Implementation struct
// ============================================================================

struct Tokenizer::Impl {
  enum class Mode { SENTENCEPIECE, ASCII };
  Mode mode = Mode::ASCII;

#ifdef GRETA_USE_SENTENCEPIECE
  sentencepiece::SentencePieceProcessor processor;
  bool spm_loaded = false;
#endif

  // ASCII fallback vocabulary
  std::vector<std::string> vocab;

  Impl() {
    // Initialize ASCII vocab
    vocab.resize(32000);
    for (size_t i = 0; i < 256; ++i) {
      vocab[i] = std::string(1, static_cast<char>(i));
    }
    vocab[0] = "<unk>";
    vocab[1] = "<s>";
    vocab[2] = "</s>";
  }
};

Tokenizer::Tokenizer() : impl_(std::make_unique<Impl>()) {}
Tokenizer::~Tokenizer() = default;

bool Tokenizer::load(const std::string &path, std::string *err) {
#ifdef GRETA_USE_SENTENCEPIECE
  auto status = impl_->processor.Load(path);
  if (status.ok()) {
    impl_->mode = Impl::Mode::SENTENCEPIECE;
    impl_->spm_loaded = true;
    std::cout << "[TOKENIZER] Loaded SentencePiece model: " << path << "\n";
    return true;
  }
  if (err)
    *err = "SentencePiece load failed: " + status.ToString();
  std::cout << "[TOKENIZER] Warning: " << *err << ". Using ASCII fallback.\n";
#else
  if (err)
    *err = "SentencePiece not compiled in.";
  std::cout << "[TOKENIZER] SentencePiece disabled, using ASCII fallback.\n";
#endif
  impl_->mode = Impl::Mode::ASCII;
  return true; // Fallback is always available
}

void Tokenizer::use_ascii_fallback() {
  impl_->mode = Impl::Mode::ASCII;
#ifdef GRETA_USE_SENTENCEPIECE
  impl_->spm_loaded = false;
#endif
}

void Tokenizer::set_vocabulary(const std::vector<std::string> &vocab) {
  impl_->vocab = vocab;
}

std::vector<int32_t> Tokenizer::encode(const std::string &text) const {
#ifdef GRETA_USE_SENTENCEPIECE
  if (impl_->mode == Impl::Mode::SENTENCEPIECE && impl_->spm_loaded) {
    std::vector<int> ids;
    impl_->processor.Encode(text, &ids);
    return std::vector<int32_t>(ids.begin(), ids.end());
  }
#endif
  // ASCII fallback
  std::vector<int32_t> result;
  result.push_back(1); // BOS
  for (unsigned char c : text) {
    result.push_back(static_cast<int32_t>(c));
  }
  return result;
}

std::string Tokenizer::decode(const std::vector<int32_t> &tokens) const {
#ifdef GRETA_USE_SENTENCEPIECE
  if (impl_->mode == Impl::Mode::SENTENCEPIECE && impl_->spm_loaded) {
    std::vector<int> ids(tokens.begin(), tokens.end());
    std::string text;
    impl_->processor.Decode(ids, &text);
    return text;
  }
#endif
  // ASCII fallback
  std::string result;
  for (int32_t t : tokens) {
    if (t == 1 || t == 2)
      continue; // Skip BOS/EOS
    if (t >= 0 && t < 256) {
      result.push_back(static_cast<char>(t));
    }
  }
  return result;
}

std::string Tokenizer::decode_token(int32_t token_id) const {
#ifdef GRETA_USE_SENTENCEPIECE
  if (impl_->mode == Impl::Mode::SENTENCEPIECE && impl_->spm_loaded) {
    return impl_->processor.IdToPiece(token_id);
  }
#endif
  if (token_id >= 0 && static_cast<size_t>(token_id) < impl_->vocab.size()) {
    return impl_->vocab[token_id];
  }
  return "[ID_" + std::to_string(token_id) + "]";
}

size_t Tokenizer::vocab_size() const {
#ifdef GRETA_USE_SENTENCEPIECE
  if (impl_->mode == Impl::Mode::SENTENCEPIECE && impl_->spm_loaded) {
    return static_cast<size_t>(impl_->processor.GetPieceSize());
  }
#endif
  return impl_->vocab.size();
}

int32_t Tokenizer::bos_id() const {
#ifdef GRETA_USE_SENTENCEPIECE
  if (impl_->mode == Impl::Mode::SENTENCEPIECE && impl_->spm_loaded) {
    return impl_->processor.bos_id();
  }
#endif
  return 1;
}

int32_t Tokenizer::eos_id() const {
#ifdef GRETA_USE_SENTENCEPIECE
  if (impl_->mode == Impl::Mode::SENTENCEPIECE && impl_->spm_loaded) {
    return impl_->processor.eos_id();
  }
#endif
  return 2;
}

bool Tokenizer::is_using_sentencepiece() const {
#ifdef GRETA_USE_SENTENCEPIECE
  return impl_->mode == Impl::Mode::SENTENCEPIECE && impl_->spm_loaded;
#else
  return false;
#endif
}

} // namespace gcore::inference
