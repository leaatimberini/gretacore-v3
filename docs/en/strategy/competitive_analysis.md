# Competitive Analysis: LLM Inference Engines

## Current State (2026-01-31)

| Engine | Hardware | Tokens/s (Batch=1) | Notes |
|:---|:---|:---|:---|
| **GRETA CORE** | AMD MI300X | ~95* | Phase 3 initial, custom kernels, demo mode |
| vLLM | NVIDIA H100 | ~180-220 | PagedAttention, optimized kernels |
| TensorRT-LLM | NVIDIA H100 | ~250-300+ | Maximum NVIDIA optimization (FP8/FP16) |
| vLLM | AMD MI300X | ~200-240 | ROCm + standard kernels |

> **Note:** The 95 tok/s value is from demo mode (loop without real computation). Actual throughput with full forward pass will be lower initially.

## Gap Analysis

### Competition Advantages
- **vLLM:** PagedAttention for efficient memory management, years of optimization.
- **TensorRT-LLM:** Access to proprietary NVIDIA libraries (cuBLAS, cuDNN), FP8 quantization.
- **Development time:** Large teams with multiple years of work.

### GRETA CORE Advantages
- **Full stack control:** No opaque dependencies, all code is visible and modifiable.
- **Hardware-specific optimization:** Direct CDNA3 (gfx942) targeting, no generic abstractions.
- **Clean architecture:** Long-term maintainability, no accumulated performance hacks.
- **Complete ownership:** No restrictive licenses or corporate dependencies.

## Optimization Roadmap

### Phase 4A: Memory Optimization (Target: +50% throughput)
| Optimization | Estimated Impact | Priority |
|:---|:---|:---|
| Pure FP16/BF16 (replace FP32) | +2x theoretical FLOPS | High |
| Double buffering in GEMM | +30-50% utilization | High |
| GEMM+RMSNorm+SiLU fusion | -30% memory overhead | Medium |

### Phase 4B: Attention Architecture (Target: +100% throughput)
| Optimization | Estimated Impact | Priority |
|:---|:---|:---|
| FlashAttention v2 (CDNA3) | +2-3x in attention kernel | Critical |
| PagedAttention | Efficient batching, less fragmentation | High |
| Attention tiling for long context | Support 8K+ tokens | Medium |

### Phase 4C: Quantization (Target: +200% throughput)
| Optimization | Estimated Impact | Priority |
|:---|:---|:---|
| INT8 GEMM (Matrix Cores) | +2x throughput | High |
| FP8 (if available on CDNA3) | +2-4x throughput | High |
| GPTQ/AWQ support | Compatibility with quantized models | Medium |

## Long-term Goal
**Target: 200+ tokens/second on MI300X with Llama-2-7B (Batch=1)**

This goal would put us at vLLM level on AMD hardware, validating the viability of an optimized native inference engine.

---
*Strategy document - GRETA CORE Team*
