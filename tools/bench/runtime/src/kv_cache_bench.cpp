#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static int parse_arg_int(int argc, char **argv, const std::string &key,
                         int def) {
  for (int i = 1; i + 1 < argc; i++) {
    if (argv[i] == key)
      return std::stoi(argv[i + 1]);
  }
  return def;
}

int main(int argc, char **argv) {
  const int layers = parse_arg_int(argc, argv, "--layers", 2);
  const int heads = parse_arg_int(argc, argv, "--heads", 4);
  const int head_dim = parse_arg_int(argc, argv, "--head-dim", 32);
  const int seq_len = parse_arg_int(argc, argv, "--seq-len", 128);
  const int iters = parse_arg_int(argc, argv, "--iters", 10);
  const int tokens_per_iter = parse_arg_int(argc, argv, "--tokens", 8);

  const size_t per_token = static_cast<size_t>(layers) * heads * head_dim;
  const size_t total = per_token * seq_len;
  std::vector<float> k(total, 0.0f);
  std::vector<float> v(total, 0.0f);

  auto t0 = std::chrono::high_resolution_clock::now();
  int last_pos = 0;
  float last_val = 0.0f;
  for (int it = 0; it < iters; it++) {
    for (int t = 0; t < tokens_per_iter; t++) {
      const int pos = (it * tokens_per_iter + t) % seq_len;
      last_pos = pos;
      last_val = static_cast<float>((pos % 97) - 48) * 0.01f;
      for (int l = 0; l < layers; l++) {
        for (int h = 0; h < heads; h++) {
          const size_t base =
              ((static_cast<size_t>(l) * heads + h) * seq_len + pos) *
              head_dim;
          for (int d = 0; d < head_dim; d++) {
            const float val = last_val + static_cast<float>(d) * 1e-4f;
            k[base + d] = val;
            v[base + d] = val * 0.5f;
          }
        }
      }
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double sec =
      std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
          .count();

  bool ok = true;
  const size_t check_base = (static_cast<size_t>(last_pos) * head_dim);
  for (int l = 0; l < layers && ok; l++) {
    for (int h = 0; h < heads && ok; h++) {
      const size_t base =
          ((static_cast<size_t>(l) * heads + h) * seq_len + last_pos) *
          head_dim;
      if (k[base] != last_val)
        ok = false;
      if (v[base] != last_val * 0.5f)
        ok = false;
    }
  }

  const double tokens_total =
      static_cast<double>(iters) * tokens_per_iter * layers * heads;
  const double tokens_per_sec = tokens_total / sec;

  std::cout << "GRETA CORE Runtime Bench: kv_cache_bench\n";
  std::cout << "layers=" << layers << " heads=" << heads
            << " head_dim=" << head_dim << " seq_len=" << seq_len
            << " iters=" << iters << " tokens=" << tokens_per_iter << "\n";
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "RESULT kv_cache_bench:\n";
  std::cout << "  total_sec=" << sec << "\n";
  std::cout << "  tokens_per_sec=" << tokens_per_sec << "\n";
  std::cout << "STATUS=" << (ok ? "OK" : "FAILED") << "\n";
  return ok ? 0 : 1;
}
