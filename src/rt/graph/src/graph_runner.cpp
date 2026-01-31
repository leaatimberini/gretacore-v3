#include "gcore/rt/graph/graph.hpp"
#include "gcore/rt/vk/backend.hpp"
#include <string>

namespace gcore::rt::graph {

bool GraphRunner::execute(vk::Backend *backend, Graph &graph,
                          std::string *err) {
  VkDevice dev = backend->device();
  VkQueue q = backend->queue();
  VkCommandPool pool = backend->command_pool();

  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkResult r = vkAllocateCommandBuffers(dev, &cbai, &cmd);
  if (r != VK_SUCCESS) {
    if (err)
      *err = "vkAllocateCommandBuffers failed";
    return false;
  }

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(cmd, &bi);
  if (!graph.record_all(cmd, err)) {
    vkFreeCommandBuffers(dev, pool, 1, &cmd);
    return false;
  }
  vkEndCommandBuffer(cmd);

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;

  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence;
  vkCreateFence(dev, &fci, nullptr, &fence);

  r = vkQueueSubmit(q, 1, &si, fence);
  if (r != VK_SUCCESS) {
    if (err)
      *err = "vkQueueSubmit failed";
    vkDestroyFence(dev, fence, nullptr);
    vkFreeCommandBuffers(dev, pool, 1, &cmd);
    return false;
  }

  vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
  vkDestroyFence(dev, fence, nullptr);
  vkFreeCommandBuffers(dev, pool, 1, &cmd);

  return true;
}

} // namespace gcore::rt::graph
