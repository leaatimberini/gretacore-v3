# Status Report & Strategic Evaluation - GRETA CORE

**Date:** 2026-01-31
**Author:** Leandro Emanuel Timberini
**Context:** Post-Codex operational continuity

## 1. Diagnosis
The GRETA CORE project has overcome its major structural bottleneck: Vulkan's incompatibility with MI300X hardware. A strategic pivot to the **HIP/ROCm** stack has been executed, achieving native execution on MI300X with scalable performance.

**Status:** The HIP backend (`gcore::rt::hip`) is now the primary engine for Enterprise hardware, while Vulkan remains as an option for consumer and cross-platform hardware.

## 2. Evaluation of Current Stage
We are finalizing **Phase 1 (Base Portability)** and entering **Phase 2 (Applied Optimization)**.
- **Achieved:** Operational HIP backend, validation on MI300X (HBM3 OK), Integrated execution graph, Initial MFMA kernels.
- **Missing:** Asynchronous KV-cache, Attention kernel fusion (FlashAttention style), Full model inference.

## 3. Technical Gaps vs CUDA (Software)
| Area | CUDA / cuDNN | GRETA CORE (Current) | Gap |
| :--- | :--- | :--- | :--- |
| **Dev toolchain** | Extreme Maturity | Incipient | **Critical** |
| **Kernels** | Ultra-tuned libraries (cuBLAS) | Manual "good enough" kernels | High |
| **Dispatch** | Graph API / Stream Capture | Manual QueueSubmit | Medium |
| **Multi-GPU** | Native NVLink / NCCL | Non-existent | **Blocker for large LLMs** |
| **Tooling** | Nsight Systems / Compute | Manual timestamp queries | High |
| **Ecosystem** | Native Pytorch | Experimental bridges | Medium |

## 4. Engineering Actions (Achieved today 2026-01-31)
1.  **MI300X Unlock (SUCCESS):** Pivot from Vulkan to HIP. Confirmed execution on MI300X VF.
2.  **Robust CPU Reference:** Implemented and used to validate all HIP kernels.
3.  **MFMA GEMM (Matrix Cores):** Initial implementation using `v_mfma_f32_16x16x4f32` reaching **~13 TFLOPS**.
4.  **LLM Attention Kernels:** RoPE and Causal Masking implemented and validated in graph.
5.  **KV-Cache & Transformer Block:** KV-Cache management operational and full Llama-2-7B block execution in **2.1 ms**.

## 5. Mid-term Architectural Roadmap
-   **Feb Week 1-2:** Minimum Inference Runner (Transformer Block).
-   **Feb Week 3-4:** Triton-First integration (Functional AMD backend upstream or stable fork).
-   **March:** Intra-node Multi-GPU (Basic tensor sharding).
-   **April:** Persistence and loading of real weights (Llama/Mistral).

## 6. Definition of FIRST RELEASE (v0.1)
**Name:** GRETA CORE v0.1 - "The Red Foundation"
**Target:** System developers and ML hackers.

**Includes:**
-   Dual Backend: **Vulkan** (Consumer) + **HIP** (Enterprise/MI300X).
-   **MFMA** Optimization for CDNA 3.
-   Functional Graph Runner for Transformer blocks.
-   Demos: Validated inference on 8600G and MI300X.

**NOT Included:**
-   Training.
-   NVIDIA support (explicit).
-   High-level Python frontends (beyond demos).

## 7. Strategic Value for AMD
GRETA CORE demonstrates that AMD hardware **does not need CUDA abstraction** to be performant. By using lower-level control:
1.  **Schematic Control:** We exploit AMD memory hierarchy without the "Black Box" of proprietary drivers.
2.  **Universality:** A single binary runs on laptop APUs and MI300X server nodes.
3.  **Independence:** Breaks reliance on monolithic ROCm (giant dockers) for inference.

## 8. Documentation Notes
-   Rigorously keep `docs/es/` and `docs/en/` synchronized.
-   Technical "Truth" resides in the code and whitepapers (`docs/*/strategy/`).
