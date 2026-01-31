# GRETA CORE – Work Plan

Version: 1.0
Date: 2026-01-31
Language: English

---

## Objective
Maintain a single, linear execution line with measurable gates and
clear stop conditions on limited local hardware.

---

## Line of Execution (LOE)

### LOE-1 — Runtime Stability + Device Memory Path
**Goal:** Stable Vulkan execution with device-local memory and safe staging.
**Gate:** All smoke profiles pass on 8600G; no hangs or timeouts.

**Tasks**
- Implement device-local buffer allocator + staging path.
- Add explicit H2D/D2H copy helpers and validation.
- Add async submit + fence/timeline-based synchronization.

**Benchmarks**
- `vk_smoke_bench`
- `vk_fill_bench`
- `vk_gemm_runtime_smoke` (ultrasafe/fill/default)

**Done When**
- 10/10 clean runs across all smoke profiles.
- No GPU hangs; no validation errors.

---

### LOE-2 — GEMM Correctness + Autotune Baselines
**Goal:** Correctness parity vs CPU reference and stable baselines.
**Gate:** Deterministic output within tolerance for FP32 and FP16 (if allowed).

**Tasks**
- Build CPU reference GEMM for correctness check.
- Add tolerance rules per precision.
- Store baseline results in `tools/bench/runtime/results` with date.
- Add preset runner scripts (local/remote) + CSV summary export.
- Maintain CUDA parity gap matrix in `docs/en/strategy/cuda_gap.md`.

**Benchmarks**
- `vk_gemm_bench`
- `vk_gemm_tiled_bench`
- `vk_gemm_auto_ts_bench`

**Done When**
- FP32 correctness OK on APU.
- FP16 only if healthcheck clean and not blacklisted.

---

### LOE-3 — Kernel Expansion (LLM Primitives)
**Goal:** Add LayerNorm/RMSNorm/Softmax/KV primitives with minimal APIs.
**Gate:** Each kernel has a correctness test + microbench.

**Tasks**
- [x] Implement kernels one-by-one with reference checks (RMSNorm, Softmax, RoPE).
- [x] Add minimal benchmarks and store baselines.
- [x] Deliver LLM Primitives Pack v1 (LayerNorm, RMSNorm, Softmax).
- [ ] Add kernel tuning cache for shape/device.

---

### LOE-4 — Minimal Inference Runner
**Goal:** Execute a single transformer block on GRETA CORE.
**Gate:** Deterministic tokens/s and latency metrics.

**Tasks**
- [x] Build a tiny graph runner for transformer block (`HIPGraphRunner`).
- [x] Add KV-cache lifecycle (`HIPKVUpdateNode`).
- [x] Publish tokens/s + latency metrics (2.1 ms on MI300X).
- [x] Wire RMSNorm + QKV + attention + MLP path (Validated in `hip_llama_block_test`).

**Technical Closure Report:** [Phase 2 - Technical Closure (EN)](docs/en/strategy/phase_2_technical_closure.md) | [Fase 2 - Cierre Técnico (ES)](docs/es/strategy/phase_2_technical_closure.md)

---

### LOE-5 — Framework Compatibility & DX
**Goal:** Same code runs on Radeon dev and MI300X cloud without changes.
**Gate:** 1–3 install commands, no Docker required; Triton/PyTorch/JAX path validated.

**Tasks**
- Implement Triton frontend bridge (AMD target).
- Integrate PyTorch and JAX bridges.
- Map cuDNN/TensorRT parity via AMD equivalents.
- Maintain `docs/en/strategy/framework_compat.md`.
- Bundle torch, triton, and jax in the GRETA installer.
- Define version matrix + lock files (`docs/en/strategy/framework_versions.md`, `tools/compat/lock/`).

---

## MI300X Validation (Optional, Paid)
Only schedule if a hypothesis exists with a measurable pass/fail.

**Hypothesis**
FP16 GEMM variants scale on MI300X without timeouts.

**Benchmark**
- `vk_gemm_auto_ts_bench --m 4096 --n 4096 --k 4096`
- `vk_gemm_runtime_smoke` (default)

**Pass/Fail**
- Pass: >=10x throughput vs 8600G baseline, zero timeouts.
- Fail: any timeout or <10x throughput.

---

## Ownership
Owner: Leandro Emanuel Timberini

---

## Completed Work (2026-01-31)
- Device-local + staging helper (`stage_host_to_device` / `read_device_to_host`) now lives in `src/rt/backend/vulkan/include/gcore/rt/vk/buffer.hpp` and is used by `vk_gemm_bench` + `vk_gemm_runtime_smoke`.
- FP16 vec2 timestamped benches now use device-local buffers with staging upload/readback (`vk_gemm_f16acc32_tiled_vec2_ts_bench`, `vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench`, `vk_gemm_f16acc32_tiled_vec2_db_ts_bench`).
- FP16 vec2 smoke runs (M=N=K=128, iters=3, batch=5) recorded:
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_ts_bench_smoke.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench_smoke.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_db_ts_bench_smoke.txt`
- FP16 vec2 standard runs (M=N=K=1024, iters=10, batch=20) recorded:
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_ts_bench_standard.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench_standard.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_f16acc32_tiled_vec2_db_ts_bench_standard.txt`
- Preset runner scripts + CSV summary added:
  - `tools/bench/runtime/scripts/run_presets_local.sh`
  - `tools/bench/runtime/scripts/run_presets_remote.sh`
  - `tools/bench/runtime/scripts/gen_bench_csv.py`
- Framework compatibility prototypes added:
  - `tools/compat/triton/vec_add.py`
  - `tools/compat/pytorch/greta_extension_hello.py`
  - `tools/compat/jax/jax_custom_call_hello.py`
- Prototype run status (local, venv): PyTorch OK, JAX OK, Triton OK (cpu fallback), ROCm build required for GPU.
- HIP platform bench added:
  - `tools/bench/platform/src/hip_vec_add.cpp`
  - `tools/bench/platform/src/hip_gemm.cpp`
  - Build target: `hip_vec_add`
  - Build target: `hip_gemm` (requires hipBLAS)
- Platform preset runner scripts added:
  - `tools/bench/platform/scripts/run_presets_local.sh`
  - `tools/bench/platform/scripts/run_presets_remote.sh`
- LLM primitives CPU bench added + smoke run:
  - `tools/bench/runtime/build/llm_primitives_bench`
  - `tools/bench/runtime/results/2026-01-31_llm_primitives_bench_smoke.txt`
- Vulkan LayerNorm bench added + smoke run:
  - `tools/bench/runtime/build/vk_layernorm_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_layernorm_bench_smoke.txt`
- Vulkan LayerNorm tiled bench added + smoke run:
  - `tools/bench/runtime/build/vk_layernorm_tiled_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_layernorm_tiled_bench_smoke.txt`
- Vulkan RMSNorm + Softmax benches added + smoke runs:
  - `tools/bench/runtime/build/vk_rmsnorm_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_rmsnorm_bench_smoke.txt`
  - `tools/bench/runtime/build/vk_softmax_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_softmax_bench_smoke.txt`
- Vulkan RMSNorm/Softmax tiled benches added + smoke runs:
  - `tools/bench/runtime/build/vk_rmsnorm_tiled_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_rmsnorm_tiled_bench_smoke.txt`
  - `tools/bench/runtime/build/vk_softmax_tiled_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_softmax_tiled_bench_smoke.txt`
- Vulkan LayerNorm+RMSNorm fused bench added + smoke run:
  - `tools/bench/runtime/build/vk_layernorm_rmsnorm_fused_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_layernorm_rmsnorm_fused_bench_smoke.txt`
- Vulkan LayerNorm+RMSNorm fused tiled bench added + smoke run:
  - `tools/bench/runtime/build/vk_layernorm_rmsnorm_fused_tiled_bench`
  - `tools/bench/runtime/results/2026-01-31_vk_layernorm_rmsnorm_fused_tiled_bench_smoke.txt`
- LLM Vulkan standard/perf runs recorded:
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
- Runtime verify preset recorded (APU):
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
- Compute-only benches recorded and archived:
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_bench_compute_only.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_tiled_bench_compute_only.txt`
  - `tools/bench/runtime/results/2026-01-31_vk_gemm_tiled_ts_bench_compute_only.txt`
  These results capture throughput with `--compute-only=1` on the Ryzen 5 8600G/RADV Phoenix stack.
- MI300X Vulkan status (Runpod, ROCm 6.1 container):
  - RADV ICD initializes but vkQueueSubmit fails with `CS rejected` (VkResult=-4).
  - AMDVLK ICD installs from amdvlk bionic repo, but vkCreateInstance fails (ERROR_INCOMPATIBLE_DRIVER).
  - Action: require AMDGPU-PRO Vulkan ICD or Runpod image with validated Vulkan for MI300X.
- MI300X HIP benches (ROCm 7.2, Runpod) recorded:
  - `tools/bench/platform/results/2026-01-31_hip_noop_launch.txt`
  - `tools/bench/platform/results/2026-01-31_hip_vec_add_smoke.txt`
  - `tools/bench/platform/results/2026-01-31_hip_vec_add_standard.txt`
- AMD Developer Cloud MI300X VF (Ubuntu 24.04.3, ROCm 7.2) smoke presets recorded:
  - `tools/bench/platform/results/2026-01-31_membw_cpu_smoke_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_memlat_cpu_smoke_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_hip_noop_launch_smoke_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_hip_vec_add_smoke_amdcloud.txt`
  - `tools/bench/platform/results/2026-01-31_hip_gemm_smoke_amdcloud.txt`
- AMD Developer Cloud MI300X VF standard/perf presets recorded:
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
- AMD Cloud preset results refreshed with hip_gemm checks enabled (column-major OK).
- AMD Developer Cloud MI300X VF hip_gemm check diagnostics recorded (layout mismatch still unresolved):
  - `tools/bench/platform/results/2026-01-31_hip_gemm_check_diag_amdcloud.txt`
- hip_gemm input init overflow fixed (size_t underflow); dump confirms column-major check OK:
  - `tools/bench/platform/results/2026-01-31_hip_gemm_check_dump_amdcloud.txt`
- hip_gemm check verified on 512^3 (column-major max_abs_err=0):
  - `tools/bench/platform/results/2026-01-31_hip_gemm_check_fixed_amdcloud.txt`
- Native HIP Backend implementation (`gcore::rt::hip`):
  - `Backend`, `Buffer`, `Stream`, and `GraphRunner` abstractions operational on MI300X.
- Optimized HIP Kernels registered:
  - `hip_fill_bench`, `hip_rmsnorm_bench` (Error ~2e-6).
  - `hip_gemm_bench` (Tiled: 12.7 TFLOPS, MFMA: 13.0 TFLOPS).
- LLM Primitives Graph Integration:
  - RoPE (Rotary Embeddings) and Causal Masking implemented.
  - `hip_attention_bench` integration test successful (STATUS=OK).
- Git-based synchronization workflow (Local Push / Remote Pull) established to prevent SSH blocking.
