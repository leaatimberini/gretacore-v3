#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static int argi(int argc, char **argv, const char *key, int def) {
  for (int i = 1; i + 1 < argc; i++) {
    if (std::string(argv[i]) == key)
      return std::stoi(argv[i + 1]);
  }
  return def;
}

static double argd(int argc, char **argv, const char *key, double def) {
  for (int i = 1; i + 1 < argc; i++) {
    if (std::string(argv[i]) == key)
      return std::stod(argv[i + 1]);
  }
  return def;
}

static std::string args(int argc, char **argv, const char *key,
                        const std::string &def) {
  for (int i = 1; i + 1 < argc; i++) {
    if (std::string(argv[i]) == key)
      return std::string(argv[i + 1]);
  }
  return def;
}

struct Stats {
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double p99_ms = 0.0;
};

static Stats compute_stats(const std::vector<double> &samples) {
  Stats s{};
  if (samples.empty())
    return s;
  double sum = 0.0;
  for (double v : samples)
    sum += v;
  s.mean_ms = sum / samples.size();
  std::vector<double> tmp = samples;
  std::sort(tmp.begin(), tmp.end());
  s.p50_ms = tmp[tmp.size() / 2];
  size_t p99_idx = (tmp.size() * 99) / 100;
  if (p99_idx >= tmp.size())
    p99_idx = tmp.size() - 1;
  s.p99_ms = tmp[p99_idx];
  return s;
}

static void layernorm_ref(const float *x, float *y, const float *gamma,
                          const float *beta, int rows, int cols, double eps) {
  for (int r = 0; r < rows; r++) {
    const float *xr = x + size_t(r) * size_t(cols);
    float *yr = y + size_t(r) * size_t(cols);
    double mean = 0.0;
    for (int c = 0; c < cols; c++)
      mean += double(xr[c]);
    mean /= double(cols);
    double var = 0.0;
    for (int c = 0; c < cols; c++) {
      double d = double(xr[c]) - mean;
      var += d * d;
    }
    var /= double(cols);
    double inv = 1.0 / std::sqrt(var + eps);
    for (int c = 0; c < cols; c++)
      yr[c] = float((double(xr[c]) - mean) * inv) * gamma[c] + beta[c];
  }
}

static void rmsnorm_ref(const float *x, float *y, const float *gamma, int rows,
                        int cols, double eps) {
  for (int r = 0; r < rows; r++) {
    const float *xr = x + size_t(r) * size_t(cols);
    float *yr = y + size_t(r) * size_t(cols);
    double ms = 0.0;
    for (int c = 0; c < cols; c++) {
      double v = double(xr[c]);
      ms += v * v;
    }
    ms /= double(cols);
    double inv = 1.0 / std::sqrt(ms + eps);
    for (int c = 0; c < cols; c++)
      yr[c] = float(double(xr[c]) * inv) * gamma[c];
  }
}

static void softmax_ref(const float *x, float *y, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    const float *xr = x + size_t(r) * size_t(cols);
    float *yr = y + size_t(r) * size_t(cols);
    double maxv = double(xr[0]);
    for (int c = 1; c < cols; c++)
      maxv = std::max(maxv, double(xr[c]));
    double sum = 0.0;
    for (int c = 0; c < cols; c++) {
      double e = std::exp(double(xr[c]) - maxv);
      yr[c] = float(e);
      sum += e;
    }
    double inv = 1.0 / sum;
    for (int c = 0; c < cols; c++)
      yr[c] = float(double(yr[c]) * inv);
  }
}

static bool check_layernorm(const float *y, int rows, int cols,
                            double *max_abs_mean, double *max_abs_var) {
  *max_abs_mean = 0.0;
  *max_abs_var = 0.0;
  for (int r = 0; r < rows; r++) {
    const float *yr = y + size_t(r) * size_t(cols);
    double mean = 0.0;
    for (int c = 0; c < cols; c++)
      mean += double(yr[c]);
    mean /= double(cols);
    double var = 0.0;
    for (int c = 0; c < cols; c++) {
      double d = double(yr[c]) - mean;
      var += d * d;
    }
    var /= double(cols);
    *max_abs_mean = std::max(*max_abs_mean, std::abs(mean));
    *max_abs_var = std::max(*max_abs_var, std::abs(var - 1.0));
  }
  return (*max_abs_mean < 5e-3) && (*max_abs_var < 5e-2);
}

static bool check_rmsnorm(const float *y, int rows, int cols,
                          double *max_abs_rms) {
  *max_abs_rms = 0.0;
  for (int r = 0; r < rows; r++) {
    const float *yr = y + size_t(r) * size_t(cols);
    double ms = 0.0;
    for (int c = 0; c < cols; c++) {
      double v = double(yr[c]);
      ms += v * v;
    }
    ms /= double(cols);
    double rms = std::sqrt(ms);
    *max_abs_rms = std::max(*max_abs_rms, std::abs(rms - 1.0));
  }
  return (*max_abs_rms < 5e-2);
}

static bool check_softmax(const float *y, int rows, int cols,
                          double *max_abs_sum) {
  *max_abs_sum = 0.0;
  for (int r = 0; r < rows; r++) {
    const float *yr = y + size_t(r) * size_t(cols);
    double sum = 0.0;
    for (int c = 0; c < cols; c++) {
      if (yr[c] < 0.0f || yr[c] > 1.0f)
        return false;
      sum += double(yr[c]);
    }
    *max_abs_sum = std::max(*max_abs_sum, std::abs(sum - 1.0));
  }
  return (*max_abs_sum < 1e-4);
}

int main(int argc, char **argv) {
  const int rows = argi(argc, argv, "--rows", 256);
  const int cols = argi(argc, argv, "--cols", 1024);
  const int iters = std::max(1, argi(argc, argv, "--iters", 10));
  const double eps = argd(argc, argv, "--eps", 1e-5);
  const std::string mode = args(argc, argv, "--mode", "all");
  const int seed = argi(argc, argv, "--seed", 12345);

  std::cout << "GRETA CORE Runtime Bench: llm_primitives_bench\n";
  std::cout << "rows=" << rows << " cols=" << cols << " iters=" << iters
            << " mode=" << mode << "\n";

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  std::vector<float> x(size_t(rows) * size_t(cols));
  std::vector<float> y(size_t(rows) * size_t(cols));
  std::vector<float> gamma(cols, 1.0f);
  std::vector<float> beta(cols, 0.0f);

  for (auto &v : x)
    v = dist(rng);

  bool all_ok = true;

  auto run_bench = [&](const std::string &name, auto &&fn, auto &&check_fn) {
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; i++) {
      auto t0 = std::chrono::high_resolution_clock::now();
      fn();
      auto t1 = std::chrono::high_resolution_clock::now();
      double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      samples.push_back(ms);
    }
    Stats st = compute_stats(samples);

    double a = 0.0, b = 0.0;
    bool ok = check_fn(a, b);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "RESULT " << name << ": mean_ms=" << st.mean_ms
              << " p50_ms=" << st.p50_ms << " p99_ms=" << st.p99_ms;
    if (name == "layernorm") {
      std::cout << " max_abs_mean=" << a << " max_abs_var=" << b;
    } else if (name == "rmsnorm") {
      std::cout << " max_abs_rms=" << a;
    } else if (name == "softmax") {
      std::cout << " max_abs_sum=" << a;
    }
    std::cout << "\n";
    if (!ok) {
      std::cout << "VALIDATION(" << name << "): FAILED\n";
      all_ok = false;
    } else {
      std::cout << "VALIDATION(" << name << "): OK\n";
    }
  };

  if (mode == "layernorm" || mode == "all") {
    auto fn = [&]() { layernorm_ref(x.data(), y.data(), gamma.data(),
                                    beta.data(), rows, cols, eps); };
    auto check = [&](double &a, double &b) {
      return check_layernorm(y.data(), rows, cols, &a, &b);
    };
    run_bench("layernorm", fn, check);
  }

  if (mode == "rmsnorm" || mode == "all") {
    auto fn =
        [&]() { rmsnorm_ref(x.data(), y.data(), gamma.data(), rows, cols, eps); };
    auto check = [&](double &a, double &b) {
      (void)b;
      return check_rmsnorm(y.data(), rows, cols, &a);
    };
    run_bench("rmsnorm", fn, check);
  }

  if (mode == "softmax" || mode == "all") {
    auto fn = [&]() { softmax_ref(x.data(), y.data(), rows, cols); };
    auto check = [&](double &a, double &b) {
      (void)b;
      return check_softmax(y.data(), rows, cols, &a);
    };
    run_bench("softmax", fn, check);
  }

  if (all_ok) {
    std::cout << "STATUS=OK\n";
    return 0;
  }
  std::cout << "STATUS=FAILED\n";
  return 1;
}
