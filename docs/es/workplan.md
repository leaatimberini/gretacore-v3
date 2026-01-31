# GRETA CORE – Plan de Trabajo

Version: 1.0
Fecha: 2026-01-31
Idioma: Espanol

---

## Objetivo
Mantener una linea unica de ejecucion con gates medibles y
condiciones de stop claras en hardware local limitado.

---

## Linea de Ejecucion (LOE)

### LOE-1 — Estabilidad del Runtime + Memoria Device-Local
**Meta:** Ejecucion Vulkan estable con memoria device-local y staging seguro.
**Gate:** Todos los perfiles de smoke pasan en 8600G; sin hangs ni timeouts.

**Tareas**
- Implementar allocator de buffers device-local + staging.
- Agregar helpers de copia H2D/D2H y validacion.
- Agregar submit async + sincronizacion con fence/timeline.

**Benchmarks**
- `vk_smoke_bench`
- `vk_fill_bench`
- `vk_gemm_runtime_smoke` (ultrasafe/fill/default)

**Done When**
- 10/10 runs limpios en todos los perfiles smoke.
- Sin cuelgues de GPU; sin errores de validacion.

---

### LOE-2 — Correctitud GEMM + Baselines de Autotune
**Meta:** Paridad de correctitud vs referencia CPU y baselines estables.
**Gate:** Output determinista dentro de tolerancia para FP32 y FP16 (si aplica).

**Tareas**
- Construir GEMM referencia CPU para chequeo de correctitud.
- Agregar reglas de tolerancia por precision.
- Guardar baselines en `tools/bench/runtime/results` con fecha.
- Agregar scripts de presets (local/remoto) + export CSV de resumen.
- Mantener matriz de brechas vs CUDA en `docs/es/strategy/cuda_gap.md`.

**Benchmarks**
- `vk_gemm_bench`
- `vk_gemm_tiled_bench`
- `vk_gemm_auto_ts_bench`

**Done When**
- FP32 correcto en APU.
- FP16 solo si healthcheck limpio y no blacklisted.

---

### LOE-3 — Expansion de Kernels (Primitivas LLM)
**Meta:** Agregar LayerNorm/RMSNorm/Softmax/KV con APIs minimas.
**Gate:** Cada kernel tiene test de correctitud + microbench.

**Tareas**
- [x] Implementar kernels uno por uno con checks de referencia (RMSNorm, Softmax, RoPE).
- [x] Agregar benchmarks minimos y guardar baselines.
- [x] Entregar Pack de Primitivas LLM v1 (LayerNorm, RMSNorm, Softmax).
- [ ] Agregar cache de tuning por shape/dispositivo.

---

### LOE-4 — Runner de Inferencia Minimo
**Meta:** Ejecutar un bloque transformer en GRETA CORE.
**Gate:** Metricas deterministas de tokens/s y latencia.

**Tareas**
- [x] Armar un tiny graph runner para un bloque transformer (`HIPGraphRunner`).
- [x] Agregar ciclo de vida de KV-cache (`HIPKVUpdateNode`).
- [x] Publicar tokens/s + latencia (2.1 ms en MI300X).
- [x] Conectar pipeline RMSNorm + QKV + atención + MLP (Validado en `hip_llama_block_test`).

**Reporte de Cierre Técnico:** [Fase 2 - Cierre Técnico (ES)](docs/es/strategy/phase_2_technical_closure.md) | [Phase 2 - Technical Closure (EN)](docs/en/strategy/phase_2_technical_closure.md)

---

### LOE-5 — Compatibilidad de Frameworks y DX
**Meta:** Mismo código en Radeon dev y MI300X cloud sin cambios.
**Gate:** 1–3 comandos de instalación, sin Docker; path Triton/PyTorch/JAX validado.

**Tareas**
- Implementar puente Triton (target AMD).
- Integrar puentes PyTorch y JAX.
- Mapear paridad cuDNN/TensorRT vía equivalentes AMD.
- Mantener `docs/es/strategy/framework_compat.md`.
- Incluir torch, triton y jax en el instalador de GRETA.
- Definir matriz de versiones + lockfiles (`docs/es/strategy/framework_versions.md`, `tools/compat/lock/`).

---

## Validacion MI300X (Opcional, Pago)
Solo agendar si hay hipotesis con pass/fail medible.

**Hipotesis**
Variantes GEMM FP16 escalan en MI300X sin timeouts.

**Benchmark**
- `vk_gemm_auto_ts_bench --m 4096 --n 4096 --k 4096`
- `vk_gemm_runtime_smoke` (default)

**Pass/Fail**
- Pass: >=10x throughput vs baseline 8600G, cero timeouts.
- Fail: cualquier timeout o <10x throughput.

---

## Ownership
Owner: Leandro Emanuel Timberini

---

## Trabajo Completado (2026-01-31)
- Helper de staging device-local (`stage_host_to_device` / `read_device_to_host`) en `src/rt/backend/vulkan/include/gcore/rt/vk/buffer.hpp` usado por `vk_gemm_bench` y `vk_gemm_runtime_smoke`.
- Benchmarks FP16 vec2 con timestamps ahora usan buffers device-local con staging de upload/readback (`vk_gemm_f16acc32_tiled_vec2_ts_bench`, `vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench`, `vk_gemm_f16acc32_tiled_vec2_db_ts_bench`).
- Runs smoke FP16 vec2 (M=N=K=128, iters=3, batch=5) registrados:
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_ts_bench_smoke.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench_smoke.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_db_ts_bench_smoke.txt`
- Runs estándar FP16 vec2 (M=N=K=1024, iters=10, batch=20) registrados:
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_ts_bench_standard.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench_standard.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_db_ts_bench_standard.txt`
- Scripts de presets + CSV agregado:
  - `tools/bench/runtime/scripts/run_presets_local.sh`
  - `tools/bench/runtime/scripts/run_presets_remote.sh`
  - `tools/bench/runtime/scripts/gen_bench_csv.py`
- Prototipos de compatibilidad frameworks agregados:
  - `tools/compat/triton/vec_add.py`
  - `tools/compat/pytorch/greta_extension_hello.py`
  - `tools/compat/jax/jax_custom_call_hello.py`
- Estado de ejecución local (venv): PyTorch OK, JAX OK, Triton OK (fallback CPU), ROCm requerido para GPU.
- Bench de plataforma HIP agregado:
  - `tools/bench/platform/src/hip_vec_add.cpp`
  - `tools/bench/platform/src/hip_gemm.cpp`
  - Target de build: `hip_vec_add`
  - Target de build: `hip_gemm` (requiere hipBLAS)
- Scripts de presets de plataforma agregados:
  - `tools/bench/platform/scripts/run_presets_local.sh`
  - `tools/bench/platform/scripts/run_presets_remote.sh`
- Bench CPU de primitivas LLM agregado + smoke run:
  - `tools/bench/runtime/build/llm_primitives_bench`
  - `tools/bench/runtime/results/2026-01-31_llm_primitives_bench_smoke.txt`
- Bench Vulkan LayerNorm agregado + smoke run:
  - `tools/bench/runtime/build/vk_layernorm_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_layernorm_bench_smoke.txt`
- Bench Vulkan LayerNorm tiled agregado + smoke run:
  - `tools/bench/runtime/build/vk_layernorm_tiled_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_layernorm_tiled_bench_smoke.txt`
- Bench Vulkan RMSNorm + Softmax agregados + smoke runs:
  - `tools/bench/runtime/build/vk_rmsnorm_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_rmsnorm_bench_smoke.txt`
  - `tools/bench/runtime/build/vk_softmax_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_softmax_bench_smoke.txt`
- Bench Vulkan RMSNorm/Softmax tiled agregados + smoke runs:
  - `tools/bench/runtime/build/vk_rmsnorm_tiled_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_rmsnorm_tiled_bench_smoke.txt`
  - `tools/bench/runtime/build/vk_softmax_tiled_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_softmax_tiled_bench_smoke.txt`
- Bench Vulkan LayerNorm+RMSNorm fused agregado + smoke run:
  - `tools/bench/runtime/build/vk_layernorm_rmsnorm_fused_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_layernorm_rmsnorm_fused_bench_smoke.txt`
- Bench Vulkan LayerNorm+RMSNorm fused tiled agregado + smoke run:
  - `tools/bench/runtime/build/vk_layernorm_rmsnorm_fused_tiled_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_layernorm_rmsnorm_fused_tiled_bench_smoke.txt`
- Runs estándar/perf LLM Vulkan registrados:
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
- Preset verify del runtime registrado (APU):
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_tiled_ts_bench_verify.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_ts_bench_verify.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench_verify.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_db_ts_bench_verify.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_layernorm_bench_verify.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_layernorm_rmsnorm_fused_bench_verify.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_layernorm_rmsnorm_fused_tiled_bench_verify.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_layernorm_tiled_bench_verify.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_rmsnorm_bench_verify.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_rmsnorm_tiled_bench_verify.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_softmax_bench_verify.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_softmax_tiled_bench_verify.txt`
- Benchmarks compute-only registrados:
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_bench_compute_only.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_tiled_bench_compute_only.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_tiled_ts_bench_compute_only.txt`
  Los resultados documentan throughput con `--compute-only=1` en el stack Ryzen 5 8600G / RADV Phoenix.
- Estado Vulkan MI300X (Runpod, contenedor ROCm 6.1):
  - ICD RADV inicializa pero vkQueueSubmit falla con `CS rejected` (VkResult=-4).
  - ICD AMDVLK se instala desde repo amdvlk bionic, pero vkCreateInstance falla (ERROR_INCOMPATIBLE_DRIVER).
  - Acción: requerir ICD Vulkan AMDGPU-PRO o imagen Runpod con Vulkan validado para MI300X.
- Benchmarks HIP MI300X (ROCm 7.2, Runpod) registrados:
  - `tools/bench/platform/results/2026-01-31_hip_noop_launch.txt`
  - `tools/bench/platform/results/2026-01-31_hip_vec_add_smoke.txt`
  - `tools/bench/platform/results/2026-01-31_hip_vec_add_standard.txt`
- AMD Developer Cloud MI300X VF (Ubuntu 24.04.3, ROCm 7.2) smoke registrados:
  - `tools/bench/platform/results/2026-01-31_membw_cpu_smoke_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_memlat_cpu_smoke_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_hip_noop_launch_smoke_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_hip_vec_add_smoke_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_hip_gemm_smoke_amdcloud.txt`
- AMD Developer Cloud MI300X VF standard/perf registrados:
  - `tools/bench/platform/results/2026-01-31_membw_cpu_standard_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_memlat_cpu_standard_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_hip_noop_launch_standard_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_hip_vec_add_standard_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_hip_gemm_standard_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_membw_cpu_perf_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_memlat_cpu_perf_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_hip_noop_launch_perf_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_hip_vec_add_perf_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_hip_gemm_perf_amdcloud.txt`
- Resultados de AMD Cloud refrescados con hip_gemm check habilitado (column-major OK).
- Diagnóstico hip_gemm en AMD Developer Cloud MI300X VF (layout mismatch pendiente):
  - `tools/bench/platform/results/2026-01-31_hip_gemm_check_diag_amdcloud.txt`
- Fix de overflow en init de hip_gemm (underflow de size_t); dump confirma check column-major OK:
  - `tools/bench/platform/results/2026-01-31_hip_gemm_check_dump_amdcloud.txt`
- Check hip_gemm verificado en 512^3 (column-major max_abs_err=0):
  - `tools/bench/platform/results/2026-01-31_hip_gemm_check_fixed_amdcloud.txt`
- Implementación de Backend HIP nativo (`gcore::rt::hip`):
  - Abstracciones de `Backend`, `Buffer`, `Stream` y `GraphRunner` operativas en MI300X.
- Registro de Kernels HIP optimizados:
  - `hip_fill_bench`, `hip_rmsnorm_bench` (Error ~2e-6).
  - `hip_gemm_bench` (Tiled: 12.7 TFLOPS, MFMA: 13.0 TFLOPS).
- Integración de Primitivas LLM en Grafo:
  - Implementación de RoPE (Rotary Embeddings) y Causal Masking.
  - Test de integración `hip_attention_bench` exitoso (STATUS=OK).
- Sincronización de flujo via Git (Push local / Pull remoto) establecida para evitar bloqueos de SSH.
