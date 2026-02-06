#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

int main() {
  std::cout << "Starting HIP Allocation Probe..." << std::endl;
  int deviceCount = 0;
  hipGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    std::cerr << "No HIP devices found!" << std::endl;
    return 1;
  }
  hipSetDevice(0);

  std::vector<void *> pointers;
  size_t allocSize = 4096; // 4KB as requested
  int count = 0;

  while (true) {
    void *ptr = nullptr;
    hipError_t err = hipMalloc(&ptr, allocSize);
    if (err != hipSuccess) {
      std::cout << "Failed at allocation #" << count + 1 << std::endl;
      std::cout << "Error: " << hipGetErrorString(err) << " (" << (int)err
                << ")" << std::endl;
      break;
    }
    pointers.push_back(ptr);
    count++;
    if (count % 100 == 0) {
      std::cout << "Successfully allocated " << count << " buffers..."
                << std::endl;
    }
  }

  std::cout << "Final count: " << count << " allocations of " << allocSize
            << " bytes each." << std::endl;

  for (void *ptr : pointers) {
    hipFree(ptr);
  }

  return 0;
}
