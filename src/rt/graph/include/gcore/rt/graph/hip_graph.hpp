#pragma once

#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace gcore::rt::graph {

/**
 * @brief Base interface for a node in the HIP execution graph.
 */
class HIPGraphNode {
public:
  virtual ~HIPGraphNode() = default;

  /**
   * @brief Enqueues the node's operations into the provided stream.
   * @param stream The HIP stream to record into.
   * @param err Output error string if something fails.
   * @return true if successful.
   */
  virtual bool record(hipStream_t stream, std::string *err) = 0;

  virtual const char *name() const = 0;
};

/**
 * @brief Container for a sequence of execution nodes.
 */
class HIPGraph {
public:
  void add_node(std::unique_ptr<HIPGraphNode> node) {
    nodes_.push_back(std::move(node));
  }

  bool record_all(hipStream_t stream, std::string *err) {
    for (auto &node : nodes_) {
      if (!node->record(stream, err)) {
        if (err)
          *err = std::string(node->name()) + ": " + *err;
        return false;
      }
    }
    return true;
  }

private:
  std::vector<std::unique_ptr<HIPGraphNode>> nodes_;
};

/**
 * @brief Executor for HIP graphs.
 */
class HIPGraphRunner {
public:
  // Simple synchronous execution for now, can be expanded for async
  static bool execute(hipStream_t stream, HIPGraph &graph, std::string *err) {
    return graph.record_all(stream, err);
  }
};

} // namespace gcore::rt::graph
