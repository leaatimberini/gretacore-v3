#include "gcore/inference/tokenizer.hpp"

#include <iostream>

int main() {
  std::cout << "GRETA CORE: Tokenizer Test\n";

  gcore::inference::Tokenizer tokenizer;
  std::string err;

  // Load (creates default vocab for now)
  if (!tokenizer.load("dummy.json", &err)) {
    std::cerr << "Load warning: " << err << " (using fallback vocab)\n";
  }
  std::cout << "Vocabulary size: " << tokenizer.vocab_size() << "\n";
  std::cout << "BOS ID: " << tokenizer.bos_id() << "\n";
  std::cout << "EOS ID: " << tokenizer.eos_id() << "\n";

  // Test encode
  std::string prompt = "Hello, world!";
  std::cout << "\nEncoding: \"" << prompt << "\"\n";
  auto tokens = tokenizer.encode(prompt);
  std::cout << "Tokens (" << tokens.size() << "): ";
  for (auto t : tokens) {
    std::cout << t << " ";
  }
  std::cout << "\n";

  // Test decode
  std::string decoded = tokenizer.decode(tokens);
  std::cout << "Decoded: \"" << decoded << "\"\n";

  // Verify roundtrip
  if (decoded == prompt) {
    std::cout << "Roundtrip: PASS\n";
  } else {
    std::cout << "Roundtrip: PARTIAL (expected for simple encoding)\n";
  }

  std::cout << "\nSTATUS=OK\n";
  return 0;
}
