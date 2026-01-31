# v0.1 Summary (Draft)

Date: 2026-01-31

## Environment
- Local: Ryzen 5 8600G + AMD Radeon Graphics (RADV Phoenix)
- Remote: Runpod MI300X (RADV GFX940), ROCm 6.1 container
- Remote: AMD Developer Cloud MI300X VF, Ubuntu 24.04.3, kernel 6.8.0-87, ROCm 7.1

## Scope Achieved
- Device-local + staging path active in Vulkan benches and runtime smoke.
- GEMM benchmarks run with compute-only mode; results archived.
- Smoke profiles (ultrasafe/fill/default) pass on Ryzen 5 8600G + RADV Phoenix.
- Tiled RMSNorm/Softmax Vulkan benches added with smoke validation.
- LayerNorm+RMSNorm fused Vulkan bench added with smoke validation.
- LayerNorm+RMSNorm fused tiled Vulkan bench added with smoke validation.
- MI300X Vulkan ICD investigation complete; GPU benches blocked by driver/ICD.

## Key Results
- `vk_gemm_bench` (512^3, compute-only): mean 1.854 ms, 0.145 TFLOPs.
- `vk_gemm_tiled_bench` (512^3, compute-only): mean 1.855 ms, 0.145 TFLOPs.
- `vk_gemm_tiled_ts_bench` (512^3, batch=8, compute-only): kernel mean 9.355 ms, 0.239 TFLOPs.
- `vk_gemm_tiled_ts_bench` (1024^3, batch=20): kernel mean 154.111 ms, 0.279 TFLOPs.
- `vk_gemm_f16acc32_tiled_vec2_ts_bench` (1024^3, batch=20): kernel mean 112.916 ms, 0.380 TFLOPs.
- `vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench` (1024^3, batch=20): kernel mean 109.887 ms, 0.391 TFLOPs.
- `vk_gemm_f16acc32_tiled_vec2_db_ts_bench` (1024^3, batch=20): kernel mean 110.876 ms, 0.387 TFLOPs.
- Smoke: `STATUS=OK`, `max_abs_err=0` on FP32 fallback (FP16 gated by safety).

## Smoke LLM Summary (rows=64, cols=256, iters=5)
| Bench | baseline_mean_ms | tiled_mean_ms | status |
| --- | --- | --- | --- |
| vk_layernorm | 0.084 | 0.049 | OK |
| vk_rmsnorm | 0.087 | 0.044 | OK |
| vk_softmax | 0.155 | 0.048 | OK |

## MI300X HIP Results (ROCm 7.2, gfx942)
| Bench | metric | value | status |
| --- | --- | --- | --- |
| hip_noop_launch (iters=200000) | per_launch_us | 2.3483 | OK |
| hip_vec_add smoke (n=4194304, iters=50) | kernel_gbps | 2689.372 | OK |
| hip_vec_add standard (n=16777216, iters=200) | kernel_gbps | 4747.449 | OK |

## AMD Cloud MI300X VF HIP Smoke (ROCm 7.1, Ubuntu 24.04.3)
| Bench | metric | value | status |
| --- | --- | --- | --- |
| membw_cpu smoke (512 MiB, iters=3) | mean_GiBps | 152.131 | OK |
| memlat_cpu smoke (128 MiB, iters=20) | mean_ns_per_hop | 43.67 | OK |
| hip_noop_launch smoke (iters=100000) | per_launch_us | 1.5508 | OK |
| hip_vec_add smoke (n=4194304, iters=50) | kernel_gbps | 3787.843 | OK |

## Artifacts
- `tools/bench/runtime/results/2026-01-31_vk_gemm_bench_compute_only.txt`
- `tools/bench/runtime/results/2026-01-31_vk_gemm_tiled_bench_compute_only.txt`
- `tools/bench/runtime/results/2026-01-31_vk_gemm_tiled_ts_bench_compute_only.txt`
- `tools/bench/runtime/results/2026-01-31_vk_gemm_tiled_ts_bench_standard.txt`
- `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_ts_bench_standard.txt`
- `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench_standard.txt`
- `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_db_ts_bench_standard.txt`
- `tools/bench/runtime/results/2026-01-31_vk_gemm_tiled_ts_bench_perf.txt`
- `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_ts_bench_perf.txt`
- `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench_perf.txt`
- `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_db_ts_bench_perf.txt`
- `tools/bench/runtime/results/2026-01-31_bench_summary.csv`
- `tools/bench/runtime/results/2026-01-31_llm_primitives_bench_smoke.txt`
- `tools/bench/runtime/results/2026-01-31_vk_layernorm_bench_smoke.txt`
- `tools/bench/runtime/results/2026-01-31_vk_layernorm_tiled_bench_smoke.txt`
- `tools/bench/runtime/results/2026-01-31_vk_rmsnorm_bench_smoke.txt`
- `tools/bench/runtime/results/2026-01-31_vk_softmax_bench_smoke.txt`
- `tools/bench/runtime/results/2026-01-31_vk_rmsnorm_tiled_bench_smoke.txt`
- `tools/bench/runtime/results/2026-01-31_vk_softmax_tiled_bench_smoke.txt`
- `tools/bench/runtime/results/2026-01-31_vk_layernorm_rmsnorm_fused_bench_smoke.txt`
- `tools/bench/runtime/results/2026-01-31_vk_layernorm_rmsnorm_fused_tiled_bench_smoke.txt`
- `tools/bench/runtime/results/2026-01-31_vk_layernorm_bench_standard.txt`
- `tools/bench/runtime/results/2026-01-31_vk_layernorm_tiled_bench_standard.txt`
- `tools/bench/runtime/results/2026-01-31_vk_layernorm_rmsnorm_fused_bench_standard.txt`
- `tools/bench/runtime/results/2026-01-31_vk_layernorm_rmsnorm_fused_tiled_bench_standard.txt`
- `tools/bench/runtime/results/2026-01-31_vk_rmsnorm_bench_standard.txt`
- `tools/bench/runtime/results/2026-01-31_vk_softmax_bench_standard.txt`
- `tools/bench/runtime/results/2026-01-31_vk_rmsnorm_tiled_bench_standard.txt`
- `tools/bench/runtime/results/2026-01-31_vk_softmax_tiled_bench_standard.txt`
- `tools/bench/runtime/results/2026-01-31_vk_layernorm_bench_perf.txt`
- `tools/bench/runtime/results/2026-01-31_vk_layernorm_tiled_bench_perf.txt`
- `tools/bench/runtime/results/2026-01-31_vk_layernorm_rmsnorm_fused_bench_perf.txt`
- `tools/bench/runtime/results/2026-01-31_vk_layernorm_rmsnorm_fused_tiled_bench_perf.txt`
- `tools/bench/runtime/results/2026-01-31_vk_rmsnorm_bench_perf.txt`
- `tools/bench/runtime/results/2026-01-31_vk_softmax_bench_perf.txt`
- `tools/bench/runtime/results/2026-01-31_vk_rmsnorm_tiled_bench_perf.txt`
- `tools/bench/runtime/results/2026-01-31_vk_softmax_tiled_bench_perf.txt`
- `tools/bench/platform/results/2026-01-31_hip_noop_launch.txt` (MI300X Runpod)
- `tools/bench/platform/results/2026-01-31_hip_vec_add_smoke.txt` (MI300X Runpod)
- `tools/bench/platform/results/2026-01-31_hip_vec_add_standard.txt` (MI300X Runpod)
- `tools/bench/platform/results/2026-01-31_membw_cpu_smoke_amdcloud.txt` (AMD Cloud MI300X VF)
- `tools/bench/platform/results/2026-01-31_memlat_cpu_smoke_amdcloud.txt` (AMD Cloud MI300X VF)
- `tools/bench/platform/results/2026-01-31_hip_noop_launch_smoke_amdcloud.txt` (AMD Cloud MI300X VF)
- `tools/bench/platform/results/2026-01-31_hip_vec_add_smoke_amdcloud.txt` (AMD Cloud MI300X VF)

## Standard Bench Summary (1024^3, batch=20)
| Bench | kernel_mean_ms | mean_TFLOPs | status |
| --- | --- | --- | --- |
| vk_gemm_tiled_ts_bench | 154.111 | 0.279 | OK |
| vk_gemm_f16acc32_tiled_vec2_ts_bench | 112.916 | 0.380 | OK |
| vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench | 109.887 | 0.391 | OK |
| vk_gemm_f16acc32_tiled_vec2_db_ts_bench | 110.876 | 0.387 | OK |

## LLM Tiled vs Baseline (rows=256, cols=1024, iters=10)
| Bench | baseline_mean_ms | tiled_mean_ms | status |
| --- | --- | --- | --- |
| vk_layernorm | 0.517 | 0.114 | OK |
| vk_rmsnorm | 0.365 | 0.116 | OK |
| vk_softmax | 1.764 | 0.109 | OK |

## Perf Bench Summary (1024^3, batch=50)
| Bench | kernel_mean_ms | mean_TFLOPs | status |
| --- | --- | --- | --- |
| vk_gemm_tiled_ts_bench | 386.297 | 0.278 | OK |
| vk_gemm_f16acc32_tiled_vec2_ts_bench | 286.908 | 0.374 | OK |
| vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench | 280.098 | 0.383 | OK |
| vk_gemm_f16acc32_tiled_vec2_db_ts_bench | 276.702 | 0.388 | OK |

## LLM Tiled vs Baseline (perf, rows=256, cols=1024, iters=10)
| Bench | baseline_mean_ms | tiled_mean_ms | status |
| --- | --- | --- | --- |
| vk_layernorm | 0.525 | 0.106 | OK |
| vk_rmsnorm | 0.464 | 0.098 | OK |
| vk_softmax | 1.766 | 0.108 | OK |

## Compute-Only Summary (512^3)
| Bench | mean_ms | mean_TFLOPs | status |
| --- | --- | --- | --- |
| vk_gemm_bench | 1.854 | 0.145 | OK |
| vk_gemm_tiled_bench | 1.855 | 0.145 | OK |
| vk_gemm_tiled_ts_bench (batch=8) | 9.355 | 0.239 | OK |

## Notes
- FP16 remains disabled by safety policy on APU; override requires explicit flags.
- Device-local staging is now the default path for compute benchmarks.
- MI300X: RADV vkQueueSubmit returns VkResult=-4 (CS rejected). AMDVLK ICD fails to create instance (ERROR_INCOMPATIBLE_DRIVER). Requires AMDGPU-PRO Vulkan ICD or validated Runpod image.
