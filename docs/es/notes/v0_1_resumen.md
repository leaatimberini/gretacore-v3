# v0.1 Resumen (Borrador)

Fecha: 2026-01-31

## Entorno
- Local: Ryzen 5 8600G + AMD Radeon Graphics (RADV Phoenix)
- Remoto: Runpod MI300X (RADV GFX940), contenedor ROCm 6.1
- Remoto: AMD Developer Cloud MI300X VF, Ubuntu 24.04.3, kernel 6.8.0-87, ROCm 7.2

## Alcance Logrado
- Ruta device-local + staging activa en benches Vulkan y smoke runtime.
- Benchmarks GEMM con modo compute-only ejecutados y archivados.
- Perfiles smoke (ultrasafe/fill/default) pasan en Ryzen 5 8600G + RADV Phoenix.
- Bench Vulkan RMSNorm/Softmax tiled agregados con smoke y validación.
- Bench Vulkan LayerNorm+RMSNorm fused agregado con smoke y validación.
- Bench Vulkan LayerNorm+RMSNorm fused tiled agregado con smoke y validación.
- Investigación de ICD Vulkan en MI300X completada; benches GPU bloqueados por driver/ICD.

## Resultados Clave
- `vk_gemm_bench` (512^3, compute-only): mean 1.854 ms, 0.145 TFLOPs.
- `vk_gemm_tiled_bench` (512^3, compute-only): mean 1.855 ms, 0.145 TFLOPs.
- `vk_gemm_tiled_ts_bench` (512^3, batch=8, compute-only): kernel mean 9.355 ms, 0.239 TFLOPs.
- `vk_gemm_tiled_ts_bench` (1024^3, batch=20): kernel mean 154.111 ms, 0.279 TFLOPs.
- `vk_gemm_f16acc32_tiled_vec2_ts_bench` (1024^3, batch=20): kernel mean 112.916 ms, 0.380 TFLOPs.
- `vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench` (1024^3, batch=20): kernel mean 109.887 ms, 0.391 TFLOPs.
- `vk_gemm_f16acc32_tiled_vec2_db_ts_bench` (1024^3, batch=20): kernel mean 110.876 ms, 0.387 TFLOPs.
- Smoke: `STATUS=OK`, `max_abs_err=0` con fallback FP32 (FP16 gated por seguridad).

## Resumen Smoke LLM (rows=64, cols=256, iters=5)
| Bench | baseline_mean_ms | tiled_mean_ms | status |
| --- | --- | --- | --- |
| vk_layernorm | 0.084 | 0.049 | OK |
| vk_rmsnorm | 0.087 | 0.044 | OK |
| vk_softmax | 0.155 | 0.048 | OK |

## Resultados HIP MI300X (ROCm 7.2, gfx942)
| Bench | métrica | valor | estado |
| --- | --- | --- | --- |
| hip_noop_launch (iters=200000) | per_launch_us | 2.3483 | OK |
| hip_vec_add smoke (n=4194304, iters=50) | kernel_gbps | 2689.372 | OK |
| hip_vec_add estándar (n=16777216, iters=200) | kernel_gbps | 4747.449 | OK |

## AMD Cloud MI300X VF HIP Smoke (ROCm 7.2, Ubuntu 24.04.3)
| Bench | métrica | valor | estado |
| --- | --- | --- | --- |
| membw_cpu smoke (512 MiB, iters=3) | mean_GiBps | 152.131 | OK |
| memlat_cpu smoke (128 MiB, iters=20) | mean_ns_per_hop | 43.67 | OK |
| hip_noop_launch smoke (iters=100000) | per_launch_us | 1.5508 | OK |
| hip_vec_add smoke (n=4194304, iters=50) | kernel_gbps | 3787.843 | OK |
| hip_gemm smoke (m=n=k=512, iters=10) | tflops | 30.173 | OK |

## AMD Cloud MI300X VF Standard/Perf (ROCm 7.2)
| Bench | métrica | valor | estado |
| --- | --- | --- | --- |
| membw_cpu estándar (1024 MiB, iters=6) | mean_GiBps | 152.474 | OK |
| memlat_cpu estándar (256 MiB, iters=50) | mean_ns_per_hop | 62.97 | OK |
| hip_noop_launch estándar (iters=200000) | per_launch_us | 1.5517 | OK |
| hip_vec_add estándar (n=16777216, iters=200) | kernel_gbps | 4398.102 | OK |
| hip_gemm estándar (m=n=k=2048, iters=20) | tflops | 80.369 | OK |
| membw_cpu perf (2048 MiB, iters=8) | mean_GiBps | 177.892 | OK |
| memlat_cpu perf (512 MiB, iters=80) | mean_ns_per_hop | 107.04 | OK |
| hip_noop_launch perf (iters=400000) | per_launch_us | 1.5506 | OK |
| hip_vec_add perf (n=16777216, iters=400) | kernel_gbps | 4370.602 | OK |
| hip_gemm perf (m=n=k=4096, iters=10) | tflops | 111.031 | OK |

## Artefactos
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
- `tools/bench/platform/results/2026-01-31_hip_gemm_smoke_amdcloud.txt` (AMD Cloud MI300X VF)
- `tools/bench/platform/results/2026-01-31_membw_cpu_standard_amdcloud.txt` (AMD Cloud MI300X VF)
- `tools/bench/platform/results/2026-01-31_memlat_cpu_standard_amdcloud.txt` (AMD Cloud MI300X VF)
- `tools/bench/platform/results/2026-01-31_hip_noop_launch_standard_amdcloud.txt` (AMD Cloud MI300X VF)
- `tools/bench/platform/results/2026-01-31_hip_vec_add_standard_amdcloud.txt` (AMD Cloud MI300X VF)
- `tools/bench/platform/results/2026-01-31_hip_gemm_standard_amdcloud.txt` (AMD Cloud MI300X VF)
- `tools/bench/platform/results/2026-01-31_membw_cpu_perf_amdcloud.txt` (AMD Cloud MI300X VF)
- `tools/bench/platform/results/2026-01-31_memlat_cpu_perf_amdcloud.txt` (AMD Cloud MI300X VF)
- `tools/bench/platform/results/2026-01-31_hip_noop_launch_perf_amdcloud.txt` (AMD Cloud MI300X VF)
- `tools/bench/platform/results/2026-01-31_hip_vec_add_perf_amdcloud.txt` (AMD Cloud MI300X VF)
- `tools/bench/platform/results/2026-01-31_hip_gemm_perf_amdcloud.txt` (AMD Cloud MI300X VF)

## Resumen Bench Estándar (1024^3, batch=20)
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

## Resumen Bench Perf (1024^3, batch=50)
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

## Resumen Compute-Only (512^3)
| Bench | mean_ms | mean_TFLOPs | status |
| --- | --- | --- | --- |
| vk_gemm_bench | 1.854 | 0.145 | OK |
| vk_gemm_tiled_bench | 1.855 | 0.145 | OK |
| vk_gemm_tiled_ts_bench (batch=8) | 9.355 | 0.239 | OK |

## Notas
- FP16 permanece deshabilitado por politica de seguridad en APU; requiere flags explicitos.
- Staging device-local es el camino por defecto para benchmarks de compute.
- MI300X: RADV vkQueueSubmit devuelve VkResult=-4 (CS rejected). ICD AMDVLK no crea instancia (ERROR_INCOMPATIBLE_DRIVER). Requiere ICD Vulkan AMDGPU-PRO o imagen Runpod validada.
