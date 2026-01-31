#include "gcore/rt/vk/buffer.hpp"

#include <cstring>
#include <cstdlib>

namespace gcore::rt::vk {

static std::string vk_err_str(VkResult r) {
  return "VkResult=" + std::to_string(static_cast<int>(r));
}

bool find_memory_type(VkPhysicalDevice phys, uint32_t type_bits,
                      VkMemoryPropertyFlags props, uint32_t *out_type_index,
                      std::string *err) {
  VkPhysicalDeviceMemoryProperties mp{};
  vkGetPhysicalDeviceMemoryProperties(phys, &mp);

  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    bool type_ok = (type_bits & (1u << i)) != 0;
    bool props_ok = (mp.memoryTypes[i].propertyFlags & props) == props;
    if (type_ok && props_ok) {
      *out_type_index = i;
      return true;
    }
  }

  if (err) {
    *err = "No suitable memory type found (type_bits=" +
           std::to_string(type_bits) + ").";
  }
  return false;
}

bool create_buffer(VkPhysicalDevice phys, VkDevice dev, VkDeviceSize size,
                   VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_props,
                   Buffer *out, std::string *err) {
  if (!out)
    return false;

  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = size;
  bci.usage = usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult r = vkCreateBuffer(dev, &bci, nullptr, &out->buf);
  if (r != VK_SUCCESS) {
    if (err)
      *err = "vkCreateBuffer failed: " + vk_err_str(r);
    return false;
  }

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(dev, out->buf, &req);

  uint32_t mem_type = 0;
  if (!find_memory_type(phys, req.memoryTypeBits, mem_props, &mem_type, err)) {
    vkDestroyBuffer(dev, out->buf, nullptr);
    out->buf = VK_NULL_HANDLE;
    return false;
  }

  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = mem_type;

  r = vkAllocateMemory(dev, &mai, nullptr, &out->mem);
  if (r != VK_SUCCESS) {
    if (err)
      *err = "vkAllocateMemory failed: " + vk_err_str(r);
    vkDestroyBuffer(dev, out->buf, nullptr);
    out->buf = VK_NULL_HANDLE;
    return false;
  }

  r = vkBindBufferMemory(dev, out->buf, out->mem, 0);
  if (r != VK_SUCCESS) {
    if (err)
      *err = "vkBindBufferMemory failed: " + vk_err_str(r);
    vkFreeMemory(dev, out->mem, nullptr);
    vkDestroyBuffer(dev, out->buf, nullptr);
    out->mem = VK_NULL_HANDLE;
    out->buf = VK_NULL_HANDLE;
    return false;
  }

  out->size = size;
  return true;
}

bool create_device_local_buffer(VkPhysicalDevice phys, VkDevice dev,
                                VkDeviceSize size, VkBufferUsageFlags usage,
                                Buffer *out, std::string *err) {
  VkBufferUsageFlags u = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  auto host_visible = []() {
    const char *v = std::getenv("GCORE_VK_HOST_VISIBLE");
    if (!v)
      return false;
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
           std::strcmp(v, "TRUE") == 0 || std::strcmp(v, "yes") == 0 ||
           std::strcmp(v, "YES") == 0;
  };
  if (host_visible()) {
    VkMemoryPropertyFlags p = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return create_buffer(phys, dev, size, u, p, out, err);
  }
  return create_buffer(phys, dev, size, u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       out, err);
}

bool create_staging_buffer(VkPhysicalDevice phys, VkDevice dev,
                           VkDeviceSize size, Buffer *out,
                           std::string *err) {
  VkBufferUsageFlags u =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkMemoryPropertyFlags p = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  return create_buffer(phys, dev, size, u, p, out, err);
}

void destroy_buffer(VkDevice dev, Buffer *b) {
  if (!b)
    return;
  if (b->buf != VK_NULL_HANDLE)
    vkDestroyBuffer(dev, b->buf, nullptr);
  if (b->mem != VK_NULL_HANDLE)
    vkFreeMemory(dev, b->mem, nullptr);
  b->buf = VK_NULL_HANDLE;
  b->mem = VK_NULL_HANDLE;
  b->size = 0;
}

bool map_buffer(VkDevice dev, const Buffer &b, void **out_ptr,
                std::string *err) {
  if (!out_ptr)
    return false;
  void *p = nullptr;
  VkResult r = vkMapMemory(dev, b.mem, 0, b.size, 0, &p);
  if (r != VK_SUCCESS) {
    if (err)
      *err = "vkMapMemory failed: " + vk_err_str(r);
    return false;
  }
  *out_ptr = p;
  return true;
}

void unmap_buffer(VkDevice dev, const Buffer &b) { vkUnmapMemory(dev, b.mem); }

bool copy_buffer(VkDevice dev, VkCommandPool pool, VkQueue queue,
                 const Buffer &src, const Buffer &dst, VkDeviceSize size,
                 std::string *err) {
  VkCommandBufferAllocateInfo cbai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(dev, &cbai, &cmd) != VK_SUCCESS) {
    if (err)
      *err = "copy_buffer: vkAllocateCommandBuffers failed";
    return false;
  }

  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
    if (err)
      *err = "copy_buffer: vkBeginCommandBuffer failed";
    return false;
  }

  VkBufferCopy region{};
  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = size;
  vkCmdCopyBuffer(cmd, src.buf, dst.buf, 1, &region);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    if (err)
      *err = "copy_buffer: vkEndCommandBuffer failed";
    return false;
  }

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  if (vkCreateFence(dev, &fci, nullptr, &fence) != VK_SUCCESS) {
    if (err)
      *err = "copy_buffer: vkCreateFence failed";
    return false;
  }

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS) {
    vkDestroyFence(dev, fence, nullptr);
    if (err)
      *err = "copy_buffer: vkQueueSubmit failed";
    return false;
  }

  VkResult wr = vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
  vkDestroyFence(dev, fence, nullptr);
  if (wr != VK_SUCCESS) {
    if (err)
      *err = "copy_buffer: vkWaitForFences failed";
    return false;
  }

  return true;
}

bool stage_host_to_device(VkDevice dev, VkCommandPool pool, VkQueue queue,
                          Buffer &staging, const Buffer &device,
                          VkDeviceSize size, HostCallback fill,
                          std::string *err) {
  void *p = nullptr;
  if (map_buffer(dev, device, &p, nullptr)) {
    fill(p, size);
    unmap_buffer(dev, device);
    return true;
  }
  if (!map_buffer(dev, staging, &p, err))
    return false;
  fill(p, size);
  unmap_buffer(dev, staging);
  return copy_buffer(dev, pool, queue, staging, device, size, err);
}

bool read_device_to_host(VkDevice dev, VkCommandPool pool, VkQueue queue,
                         const Buffer &device, Buffer &staging,
                         VkDeviceSize size,
                         const std::function<void(const void *, VkDeviceSize)> &cons,
                         std::string *err) {
  void *p = nullptr;
  if (map_buffer(dev, device, &p, nullptr)) {
    cons(p, size);
    unmap_buffer(dev, device);
    return true;
  }
  if (!copy_buffer(dev, pool, queue, device, staging, size, err))
    return false;
  if (!map_buffer(dev, staging, &p, err))
    return false;
  cons(p, size);
  unmap_buffer(dev, staging);
  return true;
}

} // namespace gcore::rt::vk
