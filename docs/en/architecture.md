# GRETA CORE – Architecture

Version: 1.1  
Status: Active  
Project Phase: Phase 3 – LLM Inference Pipeline (active)  
Language: English

---

## Purpose of This Document

This document defines the technical architecture of GRETA CORE.
It establishes the system boundaries, core components, responsibilities,
and interaction rules between layers.

The goal of this architecture is not flexibility for all use cases,
but maximal control, performance, and long-term maintainability for
LLM workloads on AMD hardware.

---

## Current Status (B3.x)

- The LLM inference pipeline is in a correctness + deep tracing phase.
- LM head routing was isolated (MFMA/VALU) and decode instrumentation added.
- Decode collapse persists; investigation focuses on attention/KV and state.
- B3.20 introduces attention decode verification with reference and KV invariant traces. Results show attn_out diverges from the ref at layer 31 while KV invariants hold; fused+mfma fails at load.
- B3.21 stabilizes `fused+mfma` (Hkv fix + alignment guard rails). MFMA==VALU at decode0, but ref divergence at layer 31 and decode0 collapse persist.
- B3.22 audits high-layer attention precision; divergence vs FP64 ref persists at layer 31 independent of accumulation mode.
- B3.23 isolates softmax in decode0 (layer 31 head 0). QK/softmax match FP64; focus shifts to V accumulation / attn_out path.
- MI300X validation is ongoing with evidence in `docs/AMD/`.

---

## Architectural Overview

GRETA CORE is designed as a **layered but non-leaky system**, where each
layer has a strict and limited responsibility.

The architecture is intentionally minimal and vertical, prioritizing
execution efficiency over abstraction generality.

High-level layers:

1. Platform Layer
2. Runtime Layer
3. Kernel Layer
4. Compiler & Autotuning Layer
5. Integration Layer
6. Tooling Layer

Each layer may depend only on the layer immediately below it.

---

## 1. Platform Layer

### Responsibility
Provide access to AMD hardware through stable, well-defined system
interfaces.

### Components
- Linux kernel (amdgpu driver)
- ROCm/HIP and/or Vulkan Compute (RADV)
- LLVM toolchain (when required)

### Design Rules
- No direct hardware manipulation outside supported drivers
- Platform-specific behavior must be isolated
- Platform layer is replaceable, not embedded

### Explicitly Out of Scope
- Custom kernel drivers
- Firmware modification
- Proprietary driver hacking

---

## 2. Runtime Layer (gcore-rt)

### Responsibility
Control execution, memory, and synchronization with minimal overhead.

### Core Concepts
- Streams
- Events
- Memory pools
- Deterministic scheduling
- Telemetry hooks

### Responsibilities
- Memory allocation and reuse
- Kernel launch orchestration
- Execution ordering
- Timestamping and counters

### Design Rules
- No kernel logic inside runtime
- No framework dependencies
- Runtime must be usable standalone

---

## 3. Kernel Layer (gcore-kernels)

### Responsibility
Implement all performance-critical computation.

### Kernel Categories
- Linear algebra (GEMM)
- Normalization (LayerNorm, RMSNorm)
- Softmax and reductions
- Attention-related primitives
- KV-cache operations

### Design Rules
- Correctness before optimization
- Explicit memory access patterns
- Favor fusion to reduce memory traffic
- Hardware-aware tuning

### Non-Goals
- Full operator coverage
- Training kernels
- Non-LLM primitives

---

## 4. Compiler & Autotuning Layer

### Responsibility
Generate and optimize kernels for specific AMD architectures.

### Components
- Kernel parameter exploration
- Autotuning database
- Optional DSL (MLIR / Triton-like)

### Design Rules
- Empirical performance over static heuristics
- Autotuning must be reproducible
- Generated kernels are first-class artifacts

### Non-Goals
- General-purpose compiler
- Automatic model graph compilation (initially)

---

## 5. Integration Layer (gcore-bridge)

### Responsibility
Expose GRETA CORE functionality to external systems.

### Targets
- Minimal standalone inference runner
- Optional ONNX Runtime execution provider
- Triton frontend bridge (AMD target)
- PyTorch and JAX bridges
- Compatibility layer for cuDNN/TensorRT via AMD equivalents

### Design Rules
- Integration is optional, not required
- No framework logic leaks into runtime or kernels
- Bridges are thin and replaceable
- No NVIDIA binary dependencies; parity via AMD equivalents

---

## 6. Tooling Layer (gcore-tools)

### Responsibility
Provide visibility, diagnostics, and control to developers.

### Tools
- Profiler (timeline, counters)
- Kernel performance reports
- Autotuning visualization
- Regression detection

### Design Rules
- Tooling must not affect runtime performance
- Tooling is strictly optional
- Metrics are canonical

---

## Data Flow Summary

Typical inference execution flow:

1. Model weights loaded
2. Memory pools initialized
3. Streams created
4. Kernels scheduled via runtime
5. Kernels executed on hardware
6. Telemetry collected
7. Results returned to caller

No layer bypasses another layer.

---

## Architectural Invariants

The following invariants must always hold:

- Kernels never depend on frameworks
- Runtime never contains model logic
- Integration never dictates kernel design
- Tooling never alters execution behavior
- Performance regressions are not acceptable without justification

---

## Long-Term Evolution

The architecture is designed to evolve without breaking invariants:

- New kernels extend the kernel layer
- New backends extend the platform layer
- New integrations extend the bridge layer

Core principles remain unchanged.

---

## Authorship

GRETA CORE is conceived, founded, and led by:

Leandro Emanuel Timberini  
Founder & Lead Systems Architect
