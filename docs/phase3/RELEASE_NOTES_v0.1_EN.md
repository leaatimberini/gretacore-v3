# GRETA CORE v0.1 - Phase 3 Release Notes

## Overview
v0.1 marks the first stable release of GRETA CORE for AMD Instinct™ MI300X accelerators. This release focuses on inference correctness, host-overhead elimination via HIP Graphs, and specialized decode-layer performance.

## Key Features

### 1. High-Performance CDNA 3 Port
- Full support for Llama-2-7B forward pass on MI300X.
- Specialized **`lm_head_gemv`** kernel for optimized large-vocabulary decoding.
- **Zero-Copy Argmax**: GPU-side category selection with host-bypass.

### 2. Autonomous Inference (HIP Graphs)
- Native implementation of `hipGraph_t` capture for 32-layer stacks.
- Elimination of host-driver scheduling latency during the autoregressive loop.
- Pointer-based RoPE and Attention kernels for static graph stability.

### 3. Technical Auditability (Tracer)
- Real-time numerical stability monitoring (`GRETA_TRACE_LEVEL`).
- High-resolution block-level profiling (`GRETA_PROFILE_BLOCKS`).

## Performance Targets
- **Average Throughput**: 13+ tokens/s on MI300X (Q4_K).
- **Correctness**: 100% parity with reference GGUF results (0 NaNs).

## Installation
Refer to [PHASE3_REPRO_GUIDE_EN.md](./PHASE3_REPRO_GUIDE_EN.md) for detailed build instructions.
