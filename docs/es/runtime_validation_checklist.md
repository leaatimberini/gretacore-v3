# GRETA CORE: Checklist de Validacion del Runtime (Hardware Local)

## Objetivo
Definir un flujo de validacion deterministico en hardware AMD limitado
(iGPU/APU), asegurando correccion y estabilidad antes de intentar FP16 o tuning.

## Checklist (en orden)
1) **Smoke (ultrasafe, FP32)**
```
GRETA_VK_SMOKE_TIMEOUT_MS=2000 \
GRETA_VK_SMOKE_PROFILE=ultrasafe \
GRETA_VK_SMOKE_TILE=8 \
tools/bench/runtime/build/vk_gemm_runtime_smoke --m 8 --n 8 --k 8
```
Esperado: `STATUS=OK`, `active_precision=fp32`.

2) **Smoke (fill)**
```
GRETA_VK_SMOKE_TIMEOUT_MS=2000 \
GRETA_VK_SMOKE_PROFILE=fill \
tools/bench/runtime/build/vk_gemm_runtime_smoke --m 8 --n 8 --k 8
```
Esperado: `STATUS=OK`.

3) **Smoke (default, FP32)**
```
GRETA_VK_SMOKE_TIMEOUT_MS=2000 \
tools/bench/runtime/build/vk_gemm_runtime_smoke --m 8 --n 8 --k 8
```
Esperado: `STATUS=OK`.

4) **Autotune (FP16 gateado)**
```
tools/bench/runtime/build/vk_gemm_auto_ts_bench --m 1024 --n 1024 --k 1024
```
Esperado en iGPU/APU con blacklist FP16: `STATUS=SKIPPED reason="fp16_blacklisted"`.

## Manejo de fallas
- Si algun smoke falla (timeout o validacion), mantener FP16 deshabilitado y
  usar solo FP32.
- No forzar FP16 salvo que exista una hipotesis clara a validar.

## Senales a registrar
- `STATUS=...`
- `RUNTIME: active_precision=...`
- `RUNTIME: fallback_reason=...`
- Contenido de `vk_fp16_blacklist.txt`
- Tags en `vk_autotune.json` (`meta:fp16_blacklist`, `meta:fp16_fallback_reason`)
