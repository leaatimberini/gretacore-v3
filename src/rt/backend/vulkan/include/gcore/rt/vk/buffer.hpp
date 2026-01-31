#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include <vulkan/vulkan.h>

namespace gcore::rt::vk {

struct Buffer {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
};

bool create_buffer(VkPhysicalDevice phys, VkDevice dev, VkDeviceSize size,
                   VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_props,
                   Buffer *out, std::string *err = nullptr);

// Convenience: device-local buffer with transfer usage bits enabled.
bool create_device_local_buffer(VkPhysicalDevice phys, VkDevice dev,
                                VkDeviceSize size, VkBufferUsageFlags usage,
                                Buffer *out, std::string *err = nullptr);

// Convenience: host-visible staging buffer for transfers.
bool create_staging_buffer(VkPhysicalDevice phys, VkDevice dev,
                           VkDeviceSize size, Buffer *out,
                           std::string *err = nullptr);

void destroy_buffer(VkDevice dev, Buffer *b);

// Map/unmap helpers (only valid if memory is HOST_VISIBLE)
bool map_buffer(VkDevice dev, const Buffer &b, void **out_ptr,
                std::string *err = nullptr);
void unmap_buffer(VkDevice dev, const Buffer &b);

// Blocking copy (uses a one-off command buffer + fence).
bool copy_buffer(VkDevice dev, VkCommandPool pool, VkQueue queue,
                 const Buffer &src, const Buffer &dst, VkDeviceSize size,
                 std::string *err = nullptr);

using HostCallback =
    std::function<void(void *, VkDeviceSize)>; // fill or consume data

bool stage_host_to_device(VkDevice dev, VkCommandPool pool, VkQueue queue,
                          Buffer &staging, const Buffer &device,
                          VkDeviceSize size, HostCallback fill,
                          std::string *err = nullptr);

bool read_device_to_host(VkDevice dev, VkCommandPool pool, VkQueue queue,
                         const Buffer &device, Buffer &staging,
                         VkDeviceSize size,
                         const std::function<void(const void *, VkDeviceSize)> &cons,
                         std::string *err = nullptr);

// Find memory type index given requirements and desired props
bool find_memory_type(VkPhysicalDevice phys, uint32_t type_bits,
                      VkMemoryPropertyFlags props, uint32_t *out_type_index,
                      std::string *err = nullptr);

} // namespace gcore::rt::vk
