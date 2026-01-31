# GRETA CORE – Development Environment

Version: 1.0  
Status: Foundational  
Project Phase: Phase 1 – Runtime Core  
Language: English

---

## Purpose of This Document

This document defines the official development and benchmarking
environment for GRETA CORE.

All benchmarks, performance measurements, and architectural decisions
must be reproducible within this environment.

Any deviation must be explicitly documented.

---

## 1. Operating System

- Distribution: Ubuntu Linux
- Version: 22.04 LTS (Jammy Jellyfish)
- Kernel: Linux 5.15.x (default LTS kernel)

Rationale:
Ubuntu 22.04 LTS provides long-term stability and the best balance
between AMD driver support, ROCm compatibility, and tooling maturity.

---

## 2. Reference Hardware

### CPU / APU
- Vendor: AMD
- Model: Ryzen 5 8600G
- Architecture: Zen 4
- Integrated GPU: RDNA 3

### Memory
- Type: DDR5
- Capacity: 16 GB (current)
- Configuration: Single Channel (temporary)
- Target Configuration: Dual Channel (recommended)

Note:
Single-channel memory significantly limits bandwidth-sensitive
workloads. All benchmark results must record memory configuration.

### Storage
- Type: NVMe SSD
- Minimum Free Space: 50 GB
- Filesystem: ext4

---

## 3. Graphics and Compute Stack

### Kernel Driver
- Driver: amdgpu (in-kernel)
- Verification:
rocminfo
clinfo

If ROCm is not available for the target device, Vulkan Compute is used.

---

## 5. Toolchain

### Compilers
- GCC: >= 11.x
- Clang / LLVM: >= 14.x

### Build Tools
- CMake: >= 3.22
- Ninja (optional but recommended)

### Profiling Tools
- perf
- rocprof (when ROCm is active)
- Vulkan validation layers (when applicable)

---

## 5.1 Packaging & Developer Experience
- Prefer pip/conda installs; Docker is not required for GRETA workflows.
- Same build/run steps should apply on Radeon dev and MI300X cloud.
- GRETA installation bundles torch, triton, and jax.
- Local dev may use a project `.venv` to avoid system Python restrictions.
- Notebook dev without ROCm uses CPU fallback for Triton prototypes.

---

## 6. Power and Thermal Configuration

### CPU Governor
- Mode: performance

Set via:
sudo cpupower frequency-set -g performance


### Thermal Considerations
- Benchmarks must be executed under stable thermal conditions
- Throttling must be detected and documented

Verification:
watch -n1 sensors


---

## 7. Environment Validation Checklist

Before running benchmarks, verify:

- Correct OS and kernel
- amdgpu driver loaded
- Compute backend available (ROCm or Vulkan)
- CPU governor set to performance
- System idle and thermally stable

Any failed check invalidates benchmark results.

---

## 8. Reproducibility Rules

- All environment changes must be logged
- Toolchain versions must be recorded
- Kernel parameters must not be modified silently
- Benchmarks without environment metadata are invalid

---

## Authorship

GRETA CORE is conceived, founded, and led by:

Leandro Emanuel Timberini  
Founder & Lead Systems Architect
