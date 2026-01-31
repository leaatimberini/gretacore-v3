# GRETA CORE: Runtime Validation Checklist (Local Hardware)

## Goal
Provide a deterministic validation flow on low-resource AMD hardware (iGPU/APU),
ensuring correctness and stability before attempting any FP16 or tuning paths.

## Checklist (run in order)
1) **Smoke (ultrasafe, FP32)**
```
GRETA_VK_SMOKE_TIMEOUT_MS=2000 \
GRETA_VK_SMOKE_PROFILE=ultrasafe \
GRETA_VK_SMOKE_TILE=8 \
tools/bench/runtime/build/vk_gemm_runtime_smoke --m 8 --n 8 --k 8
```
Expected: `STATUS=OK`, `active_precision=fp32`.

2) **Smoke (fill)**
```
GRETA_VK_SMOKE_TIMEOUT_MS=2000 \
GRETA_VK_SMOKE_PROFILE=fill \
tools/bench/runtime/build/vk_gemm_runtime_smoke --m 8 --n 8 --k 8
```
Expected: `STATUS=OK`.

3) **Smoke (default, FP32)**
```
GRETA_VK_SMOKE_TIMEOUT_MS=2000 \
tools/bench/runtime/build/vk_gemm_runtime_smoke --m 8 --n 8 --k 8
```
Expected: `STATUS=OK`.

4) **Autotune (FP16 gated)**
```
tools/bench/runtime/build/vk_gemm_auto_ts_bench --m 1024 --n 1024 --k 1024
```
Expected on iGPU/APU with FP16 blacklist: `STATUS=SKIPPED reason="fp16_blacklisted"`.

## Failure Handling
- If any smoke step times out or fails validation, keep FP16 disabled and use
  FP32-only paths.
- Do not force FP16 unless a clear hypothesis is being tested.

## Signals to Record
- `STATUS=...`
- `RUNTIME: active_precision=...`
- `RUNTIME: fallback_reason=...`
- `vk_fp16_blacklist.txt` contents
- `vk_autotune.json` tags (`meta:fp16_blacklist`, `meta:fp16_fallback_reason`)
