# Phase 3 Strategy: Technical Closure & Hardware Hardening (MI300X)
**Author:** Leandro Emanuel Timberini

## Technical Status Summary
GRETA CORE has successfully achieved functional stability and baseline performance optimization for Llama-2-7B on AMD MI300X (gfx942).

### Validated KPIs
- **Correctness:** 0 NaNs detected in all layers. Logits are stable within expected FP16/FP32 bounds.
- **Dequantization:** bit-perfect parity for Q4_K and Q6_K compared to reference.
- **Throughput (Decode):** **10.41 tokens/s** (Warp-only GEMV specialization).
- **Throughput (Prefill):** **19.3 tokens/s** (Mixed MFMA/GEMV path).

## Roadmap: Phase 3.1 - 3.3

### Phase 3.1: Hardening & Deep Validation
*Focus: From "it works" to "it is auditable and trustworthy".*

1.  **Parity Validator:** Automatic comparison between GRETA output and CPU Reference/FP32 ground truth.
2.  **Granular Instrumentation:** Implementation of high-resolution HIP timers per logic block:
    -   QKV Projection (Latency/Throughput).
    -   Self-Attention & KV-Update.
    -   FFN (W1, W3, W2 chain).
    -   lm_head (High-vocab latency).
3.  **Numerical Stability Report:** Documentation of FP16 vs FP32 drift and its impact on top-1/top-k token selection.

### Phase 3.2: Performance Scaling
*Focus: Competitive potential and efficiency.*

1.  **lm_head Optimization:** Specialized GEMV for vocabulary (>32k) to saturate memory controllers.
2.  **HIP Graph Integration:** Capture and replay the generation loop to eliminate $300+$ host-side kernel launch latencies.
3.  **KV-Cache Locality:** Alignment and layout optimization for HBM3 burst-mode access.

### Phase 3.3: Technical Release (AMD-facing)
*Focus: Professional delivery and reproducibility.*

1.  **Reproducibility Guide:** One-click script to verify all KPIs on a standard ROCm 7.1 environment.
2.  **Architecture Specifications:** Documentation of the low-level inference engine logic.
3.  **Gap Analysis:** Clear list of current limitations (e.g., P2P across multiple MI300X, FlashAttention tuning).

## Technical Risks & Mitigations

| Risk | Mitigation Strategy |
| :--- | :--- |
| **Launch Overhead:** Host latency dominating decode loop. | Implement HIP Graphs to offload dispatch logic to GPU firmware. |
| **Numeric Drift:** FP16 accumulation causing divergence in long context. | Selectively use FP32 for sensitive reductions (Softmax/Norms). |
| **Memory Bound:** low CU utilization in small batch sizes. | Increase thread-level parallelism (TLP) in GEMV via warp-specialization. |

## Why GRETA CORE? (AMD Value Prop)
Unlike black-box frameworks, GRETA CORE provides:
1.  **Explicit Runtime Control:** No hidden abstraction layers between the logic and ROCm/HIP.
2.  **Native Matrix Core Utilization:** Architecture-specific MFMA usage tuned for gfx942.
3.  **Optimized Mixed-Precision:** Precise balance between Q4_K/Q6_K weights and FP32 activations.
