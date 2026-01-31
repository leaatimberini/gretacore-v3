#pragma once

#include "gcore/rt/graph/graph.hpp"
#include "gcore/rt/vk/gemm.hpp"
#include <string>

namespace gcore::rt::graph {

/**
 * @brief Nodo que ejecuta una operación GEMM usando el backend Vulkan.
 */
class GemmNode : public GraphNode {
public:
  GemmNode(vk::GemmAuto *gemm_op, const vk::GemmDispatchDesc &desc)
      : gemm_op_(gemm_op), desc_(desc) {}

  bool record(VkCommandBuffer cmd, std::string *err) override {
    return gemm_op_->record_dispatch(cmd, desc_, err);
  }

  const char *name() const override { return "GemmNode"; }

private:
  vk::GemmAuto *gemm_op_;
  vk::GemmDispatchDesc desc_;
};

/**
 * @brief Nodo que inserta una barrera de memoria para sincronización.
 */
class SyncNode : public GraphNode {
public:
  SyncNode(VkBuffer buffer, VkDeviceSize size, VkAccessFlags srcAccess,
           VkAccessFlags dstAccess, VkPipelineStageFlags srcStage,
           VkPipelineStageFlags dstStage)
      : buffer_(buffer), size_(size), srcAccess_(srcAccess),
        dstAccess_(dstAccess), srcStage_(srcStage), dstStage_(dstStage) {}

  bool record(VkCommandBuffer cmd, std::string *err) override {
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = srcAccess_;
    barrier.dstAccessMask = dstAccess_;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer_;
    barrier.offset = 0;
    barrier.size = size_;

    vkCmdPipelineBarrier(cmd, srcStage_, dstStage_, 0, 0, nullptr, 1, &barrier,
                         0, nullptr);
    return true;
  }

  const char *name() const override { return "SyncNode"; }

private:
  VkBuffer buffer_;
  VkDeviceSize size_;
  VkAccessFlags srcAccess_;
  VkAccessFlags dstAccess_;
  VkPipelineStageFlags srcStage_;
  VkPipelineStageFlags dstStage_;
};

} // namespace gcore::rt::graph
