# GRETA CORE — Backend Selection (GPU Compute Runtime)

Path: docs/en/backend_selection.md  
Version: 1.0  
Date: 2026-01-29  
Author / Owner: Leandro Emanuel Timberini

---

## 1) Purpose

This document records the decision for GRETA CORE’s initial GPU compute backend.  
The goal is to enable a practical, maintainable, high-performance compute runtime for LLM workloads on AMD hardware.

This is a living document: decisions may evolve as drivers, toolchains, or hardware support changes.

---

## 2) Target Hardware & OS (Current Test Platform)

- CPU/APU: AMD Ryzen 5 8600G (APU, RDNA3 iGPU)
- GPU: Integrated RDNA3 (Phoenix class)
- OS: Ubuntu Desktop 22.04 (Kernel: 6.8.0-90-generic)
- Vulkan driver stack: Mesa RADV
- RAM: 16GB DDR5 (currently single DIMM installed in Channel B, therefore likely single-channel)

---

## 3) Observed System Signals

### 3.1 PCI / GPU detection
`lspci` confirms an AMD VGA controller:
- `Advanced Micro Devices, Inc. [AMD/ATI] Device [1002:15bf]`

### 3.2 Kernel driver status (amdgpu)
`dmesg` confirms:
- `amdgpu kernel modesetting enabled`
- VRAM reported: `2048M`
- GTT reported: `6779M`
- KFD node created and device added:
  - `kfd ... added device 1002:15bf`

A display-controller timeout was observed:
- `REG_WAIT timeout ... optc314_disable_crtc`
This is treated as a display-path warning and is not considered a blocker for compute runtime bring-up.

---

## 4) Vulkan Capability (Compute Path)

`vulkaninfo` enumerates two devices:

- GPU0: Integrated GPU on AMD via RADV
  - `deviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`
  - `deviceName = AMD Unknown (RADV GFX1103_R1)`
  - `driverName = radv`

- GPU1: CPU fallback
  - `deviceType = PHYSICAL_DEVICE_TYPE_CPU`
  - `deviceName = llvmpipe (LLVM ...)`
  - `driverName = llvmpipe`

### Loader warning
A Vulkan loader warning is present:
- `Failed to CreateInstance in ICD 1. Skipping ICD.`

Interpretation:
- One installed ICD (e.g., intel/virtio) fails to initialize on this system.
- The Vulkan loader skips it and still successfully loads RADV.
- This is not treated as a blocker. GRETA CORE will explicitly select the AMD GPU and avoid llvmpipe.

To force AMD RADV for benchmarking:
- `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`

---

## 5) ROCm / HIP Status (AMD Compute Stack)

ROCm/HIP is not available on this system at present:
- `hipcc` not installed / not found.
- `rocminfo` is not installed or not reporting.

Given time-to-value, ecosystem variance, and APU support constraints, ROCm/HIP is not selected as the initial backend.
ROCm/HIP remains a required path for LOE-5 framework compatibility (Triton/PyTorch/JAX) once the stack is stable on target systems.

---

## 6) Decision

### Selected Primary Backend (v1)
✅ **Vulkan Compute (Mesa RADV)**

Rationale:
- Already functional on the current AMD APU platform.
- Stable toolchain for cross-vendor portability.
- Explicit device selection is possible (avoid llvmpipe).
- Enables a full “from-scratch” backend that can later interoperate with other stacks.

### Deferred / Optional Backend (future)
⏳ **ROCm / HIP**
- Considered later if it becomes straightforward and stable on the target platforms.

---

## 7) Engineering Implications for GRETA CORE

### 7.1 Runtime architecture alignment
GRETA CORE runtime modules already implemented:
- Host allocator (pooling/caching)
- Stream & Event (control plane primitives)
- Telemetry (low-overhead counters/timers)
- Dispatcher (CPU-only “kernel-like” dispatch)

Vulkan backend will plug into these via:
- Device selection
- Compute queue scheduling
- Synchronization primitives (fence/semaphore)
- Buffer management (device memory allocations)

### 7.2 Next Milestones
1. Vulkan backend bootstrap (instance, device, queue, empty submit smoke test)
2. Device buffer allocation and staging logic
3. First compute kernel:
   - baseline: device memcpy / fill
4. Core compute kernel for LLM primitives:
   - GEMM (matrix multiplication)
   - attention subcomponents (later)

---

## 8) Ownership & Attribution

GRETA CORE is authored and owned by:
**Leandro Emanuel Timberini**

This document is part of the GRETA CORE repository and serves as an official record of engineering decisions.
