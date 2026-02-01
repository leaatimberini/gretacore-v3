#include <bit>
#include <chrono>
#include <cmath>
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

static std::string parse_arg_str(int argc, char **argv, const std::string &key,
                                 const std::string &def) {
  for (int i = 1; i + 1 < argc; i++) {
    if (argv[i] == key)
      return argv[i + 1];
  }
  return def;
}

static float parse_arg_float(int argc, char **argv, const std::string &key,
                             float def) {
  for (int i = 1; i + 1 < argc; i++) {
    if (argv[i] == key)
      return std::stof(argv[i + 1]);
  }
  return def;
}

static uint16_t float_to_half(float f) {
  uint32_t x = std::bit_cast<uint32_t>(f);
  uint32_t sign = (x >> 31) & 0x1;
  int exp = int((x >> 23) & 0xFF) - 127;
  uint32_t mant = x & 0x7FFFFF;
  if (exp > 15) {
    return static_cast<uint16_t>((sign << 15) | 0x7C00);
  }
  if (exp < -14) {
    if (exp < -24)
      return static_cast<uint16_t>(sign << 15);
    mant |= 0x800000;
    int shift = -exp - 14;
    uint32_t m = mant >> (shift + 13);
    return static_cast<uint16_t>((sign << 15) | m);
  }
  uint32_t exp_h = static_cast<uint32_t>(exp + 15);
  uint32_t mant_h = mant >> 13;
  uint32_t round = mant & 0x1FFF;
  if (round > 0x1000 || (round == 0x1000 && (mant_h & 0x1)))
    mant_h++;
  if (mant_h == 0x400) {
    mant_h = 0;
    exp_h++;
    if (exp_h >= 31)
      return static_cast<uint16_t>((sign << 15) | 0x7C00);
  }
  return static_cast<uint16_t>((sign << 15) | (exp_h << 10) | mant_h);
}

static float half_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign << 31;
    } else {
      exp = 127 - 14;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3FF;
      out = (sign << 31) | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = (sign << 31) | 0x7F800000 | (mant << 13);
  } else {
    exp = exp + (127 - 15);
    out = (sign << 31) | (exp << 23) | (mant << 13);
  }
  return std::bit_cast<float>(out);
}

int main(int argc, char **argv) {
  const int m = parse_arg_int(argc, argv, "--m", 128);
  const int n = parse_arg_int(argc, argv, "--n", 128);
  const int k = parse_arg_int(argc, argv, "--k", 128);
  const int iters = parse_arg_int(argc, argv, "--iters", 1);
  const std::string precision =
      parse_arg_str(argc, argv, "--precision", "fp32");
  const float tol_default = (precision == "fp16") ? 1e-2f : 1e-4f;
  const float tol = parse_arg_float(argc, argv, "--tol", tol_default);

  std::vector<float> a_f(static_cast<size_t>(m) * k);
  std::vector<float> b_f(static_cast<size_t>(k) * n);
  std::vector<uint16_t> a_h;
  std::vector<uint16_t> b_h;
  if (precision == "fp16") {
    a_h.resize(a_f.size());
    b_h.resize(b_f.size());
  }

  for (size_t i = 0; i < a_f.size(); i++) {
    const int v = static_cast<int>(i % 251) - 125;
    a_f[i] = static_cast<float>(v) * 0.01f;
    if (!a_h.empty())
      a_h[i] = float_to_half(a_f[i]);
  }
  for (size_t i = 0; i < b_f.size(); i++) {
    const int v = static_cast<int>(i % 197) - 98;
    b_f[i] = static_cast<float>(v) * 0.02f;
    if (!b_h.empty())
      b_h[i] = float_to_half(b_f[i]);
  }

  std::vector<float> c_test(static_cast<size_t>(m) * n, 0.0f);
  std::vector<double> c_ref(static_cast<size_t>(m) * n, 0.0);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int it = 0; it < iters; it++) {
    std::fill(c_test.begin(), c_test.end(), 0.0f);
    std::fill(c_ref.begin(), c_ref.end(), 0.0);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        double acc_ref = 0.0;
        float acc_test = 0.0f;
        for (int kk = 0; kk < k; kk++) {
          float av = a_h.empty()
                         ? a_f[static_cast<size_t>(i) * k + kk]
                         : half_to_float(a_h[static_cast<size_t>(i) * k + kk]);
          float bv = b_h.empty()
                         ? b_f[static_cast<size_t>(kk) * n + j]
                         : half_to_float(b_h[static_cast<size_t>(kk) * n + j]);
          acc_ref += static_cast<double>(av) * static_cast<double>(bv);
          acc_test += av * bv;
        }
        c_ref[static_cast<size_t>(i) * n + j] = acc_ref;
        c_test[static_cast<size_t>(i) * n + j] = acc_test;
      }
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double sec =
      std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
          .count();

  double max_abs_err = 0.0;
  for (size_t i = 0; i < c_test.size(); i++) {
    double err = std::abs(static_cast<double>(c_test[i]) - c_ref[i]);
    if (err > max_abs_err)
      max_abs_err = err;
  }

  std::cout << "GRETA CORE Runtime Bench: gemm_ref_bench\n";
  std::cout << "m=" << m << " n=" << n << " k=" << k << " iters=" << iters
            << " precision=" << precision << " tol=" << tol << "\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "RESULT gemm_ref_bench:\n";
  std::cout << "  total_sec=" << sec << "\n";
  std::cout << "  max_abs_err=" << max_abs_err << "\n";
  const bool ok = (max_abs_err <= tol);
  std::cout << "STATUS=" << (ok ? "OK" : "FAILED") << "\n";
  return ok ? 0 : 1;
}
