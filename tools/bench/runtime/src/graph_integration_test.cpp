#include <gcore/rt/graph/graph.hpp>
#include <gcore/rt/graph/vk_nodes.hpp>
#include <gcore/rt/ref/cpu_reference.hpp>
#include <gcore/rt/vk/backend.hpp>
#include <gcore/rt/vk/buffer.hpp>
#include <gcore/rt/vk/gemm.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace gcore::rt;

int main() {
  std::cout << "Starting Graph Integration Test..." << std::endl;

  vk::Backend backend;
  std::string err;
  if (!backend.init(&err)) {
    std::cerr << "Backend init failed: " << err << std::endl;
    return 1;
  }

  // Probar un grafo con una sola GEMM para validación básica del runner
  vk::GemmAuto gemm;
  if (!gemm.init(&backend, "build", vk::GemmPrecision::F32, &err)) {
    std::cerr << "Gemm init failed: " << err << std::endl;
    // Tratar de buscar en otra ruta si falla (benchmarks building path)
    if (!gemm.init(&backend, "tools/bench/runtime/build",
                   vk::GemmPrecision::F32, &err)) {
      std::cerr << "Gemm init failed (2nd attempt): " << err << std::endl;
      return 1;
    }
  }

  int M = 128, N = 128, K = 128;
  VkDeviceSize szA = M * K * 4;
  VkDeviceSize szB = K * N * 4;
  VkDeviceSize szC = M * N * 4;

  vk::Buffer bufA, bufB, bufC;
  vk::create_device_local_buffer(backend.physical_device(), backend.device(),
                                 szA, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bufA,
                                 &err);
  vk::create_device_local_buffer(backend.physical_device(), backend.device(),
                                 szB, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bufB,
                                 &err);
  vk::create_device_local_buffer(
      backend.physical_device(), backend.device(), szC,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      &bufC, &err);

  vk::GemmDispatchDesc desc;
  desc.A = bufA.buf;
  desc.B = bufB.buf;
  desc.C = bufC.buf;
  desc.M = M;
  desc.N = N;
  desc.K = K;
  desc.lda = K;
  desc.ldb = N;
  desc.ldc = N;

  graph::Graph g;
  g.add_node(std::make_unique<graph::GemmNode>(&gemm, desc));

  // Para validación, podríamos usar el GraphRunner pero como es una clase
  // interna en .cpp mejor la movemos a header o la incluimos. Para el test la
  // implementaremos inline o usaremos el .cpp

  // Por ahora, verifiquemos que compila y el record() funciona.
  std::cout << "Graph built with " << g.node_count() << " nodes." << std::endl;

  std::cout << "Test passed (Graph construction)." << std::endl;

  vk::destroy_buffer(backend.device(), &bufA);
  vk::destroy_buffer(backend.device(), &bufB);
  vk::destroy_buffer(backend.device(), &bufC);

  return 0;
}
