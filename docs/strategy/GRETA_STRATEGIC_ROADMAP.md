# Strategic Roadmap: The GRETA Platform

**Mission**: Rebuild the "NVIDIA Stack" (cuBLAS, cuDNN, TensorRT) on AMD hardware through native GRETA abstractions.

## Phase 1-4: The Foundation (Completed/Current)
- [x] **MI300X Infrastructure**: Stable HIP/ROCm access and memory management.
- [x] **The Matrix Core (MFMA) Breakthrough**: Custom GEMM kernels outperforming standard libs on CDNA 3.
- [x] **Fusion Archetypes**: Proved 2x speedup in FFN and Attention through memory elimination.
- [/] **L0/L1 Transition**: Formalizing `GretaRuntime` and `GretaCompute` APIs to hide HIP.

## Phase 5: The "H100 Parity" Push
- **Objective**: Match H100/CUDA throughput on Llama-3-8B.
- [ ] **GCC (L1) Hardening**: Complete `GretaCompute` implementation using optimized kernel backends.
- [ ] **Dequantization-in-GEMM**: First-class support for mixed-precision (W4/A16) MFMA ops.
- [ ] **vLLM/TensorRT-LLM Connector**: A prototype shim that allows 3rd party frameworks to call GRETA instead of cuBLAS.

## Phase 6: The GRETA Graph Ecosystem
- **Objective**: Eliminate kernel launch overhead for all generative models.
- [ ] **Auto-Capture**: Intelligent logic that translates a sequence of `GretaCompute` calls into a single `GretaGraph`.
- [ ] **Multi-GPU / Infinity Fabric**: Scale GPM to multi-MI300X nodes using NCCL-equivalent grease.

## Comparison Table: Stack Parity

| NVIDIA/CUDA | AMD/ROCm (Standard) | **GRETA CORE** | Mental Model |
| :--- | :--- | :--- | :--- |
| **cuBLAS** | rocBLAS | **GretaCompute (L1)** | Standard GEMM / GCC |
| **cuDNN** | MIOpen | **GretaCompute (L1)** | Attention / GCC |
| **TensorRT** | AMDMIGraphX | **GretaGraph (L0)** | Static Execution Plan |
| **CUDA Graphs** | HIP Graphs | **GretaGraph (L0)** | Graph Capture |

---

# Roadmap Estratégico: La Plataforma GRETA (ES)

**Misión**: Reconstruir el "Stack de NVIDIA" (cuBLAS, cuDNN, TensorRT) en hardware AMD mediante abstracciones nativas de GRETA.

## Fase 5: El Desafío "HParity"
- **Objetivo**: Igualar el rendimiento de H100/CUDA en Llama-3-8B.
- [ ] **Endurecimiento de GCC (L1)**: Implementación completa de `GretaCompute`.
- [ ] **Dequant-in-GEMM**: Soporte MFMA nativo para precisión mixta (W4/A16).

## Tabla de Comparación: Paridad de Stack

| NVIDIA/CUDA | AMD/ROCm (Estándar) | **GRETA CORE** | Modelo Mental |
| :--- | :--- | :--- | :--- |
| **cuBLAS** | rocBLAS | **GretaCompute (L1)** | GEMM Estándar |
| **cuDNN** | MIOpen | **GretaCompute (L1)** | Atención |
| **TensorRT** | AMDMIGraphX | **GretaGraph (L0)** | Plan de Ejecución |
