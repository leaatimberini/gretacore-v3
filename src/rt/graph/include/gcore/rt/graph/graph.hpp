#pragma once

#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace gcore::rt::graph {

/**
 * @brief Interfaz base para un nodo en el grafo de ejecución.
 */
class GraphNode {
public:
  virtual ~GraphNode() = default;

  /**
   * @brief Registra los comandos del nodo en el commandBuffer.
   * @param cmd CommandBuffer en estado de grabación.
   * @return true si se registró correctamente.
   */
  virtual bool record(VkCommandBuffer cmd, std::string *err) = 0;

  virtual const char *name() const = 0;
};

/**
 * @brief Orquestador de una secuencia de nodos de ejecución.
 */
class Graph {
public:
  Graph() = default;

  void add_node(std::unique_ptr<GraphNode> node) {
    nodes_.push_back(std::move(node));
  }

  /**
   * @brief Registra todos los comandos de los nodos en orden.
   */
  bool record_all(VkCommandBuffer cmd, std::string *err) {
    for (auto &node : nodes_) {
      if (!node->record(cmd, err)) {
        if (err)
          *err = std::string(node->name()) + ": " + *err;
        return false;
      }
    }
    return true;
  }

  size_t node_count() const { return nodes_.size(); }

private:
  std::vector<std::unique_ptr<GraphNode>> nodes_;
};

/**
 * @brief Clase utilitaria para ejecutar un grafo sobre un backend.
 */
class GraphRunner {
public:
  static bool execute(vk::Backend *backend, Graph &graph, std::string *err);
};

} // namespace gcore::rt::graph
