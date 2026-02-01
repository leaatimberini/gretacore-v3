#include <assert.h>
#include <gcore/rt/ref/cpu_reference.hpp>
#include <iostream>
#include <vector>

using namespace gcore::rt::ref;

void test_gemm() {
  std::cout << "Testing GEMM..." << std::endl;
  int M = 2, N = 2, K = 2;
  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> C(4, 0.0f);

  CpuReference::gemm(A.data(), B.data(), C.data(), M, N, K);

  // C = [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
  //     [3*5 + 4*7, 3*6 + 4*8]   [43, 50]
  assert(C[0] == 19.0f);
  assert(C[1] == 22.0f);
  assert(C[2] == 43.0f);
  assert(C[3] == 50.0f);
  std::cout << "GEMM OK" << std::endl;
}

void test_rmsnorm() {
  std::cout << "Testing RMSNorm..." << std::endl;
  int rows = 1, cols = 4;
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> weight = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> y(4, 0.0f);

  CpuReference::rmsnorm(x.data(), y.data(), weight.data(), rows, cols, 1e-5f);

  // MS = (1^2 + 2^2 + 3^2 + 4^2) / 4 = (1+4+9+16)/4 = 30/4 = 7.5
  // RMS = sqrt(7.5) approx 2.7386
  float rms = std::sqrt(7.5f);
  for (int i = 0; i < 4; ++i) {
    assert(std::abs(y[i] - (x[i] / rms)) < 1e-5f);
  }
  std::cout << "RMSNorm OK" << std::endl;
}

int main() {
  test_gemm();
  test_rmsnorm();
  std::cout << "All tests passed!" << std::endl;
  return 0;
}
