# Numeric Tolerance Matrix - GRETA CORE

**Date:** 2026-01-31
**Status:** Initial Draft

## Objective
Define acceptable absolute and relative error limits for GRETA CORE operations compared to the CPU Reference (FP64/FP32).

## Tolerances by Data Type

| Operation | Precision | Max Abs Error | Max Rel Error | Note |
|:---|:---|:---|:---|:---|
| **GEMM** | FP32 | 1e-4 | 1e-4 | Accumulation in FP32. |
| **GEMM** | FP16 (Acc32) | 1e-2 | 1e-3 | Expected error due to input truncation. |
| **RMSNorm** | FP32 | 1e-5 | 1e-5 | Sensitive to epsilon. |
| **LayerNorm** | FP32 | 1e-5 | 1e-5 | Includes mean and variance. |
| **Softmax** | FP32 | 1e-6 | 1e-6 | Sum must be exactly 1.0 (approx). |

## Validation Rules
1. **GEMM**: Validated on a sample of 8x8 corners and center for large matrices (>1024), and fully for small matrices.
2. **Norms**: Validated on 100% of elements per row.
3. **Scale**: On MI300X, tolerances may be relaxed by 10% if specific hardware optimizations (like Reduced Precision in registers) are used to sacrifice bits for speed.

## Failure Actions
- If `max_abs_err > tolerance`: The test marks `STATUS=FAILED`.
- If error is consistent (e.g., always 2.0x tolerance): Investigate accumulator overflow.
- If error is random: Investigate race conditions in Shmem/Synchronization.
