# B3.17 — Decode LM Head Isolation (MFMA vs VALU, decode-only)

## Resumen Ejecutivo (ES)
Se introdujo un control explícito de ruta del LM head **solo en decode** mediante `GRETA_LMHEAD_FORCE_ROUTE_DECODE=auto|valu|mfma` y se extendieron las trazas JSONL para correlacionar hidden/offsets y la ruta de GEMM en `decode0`. El objetivo es aislar si el colapso proviene del **hidden de decode** o del **LM head en decode**.

## Executive Summary (EN)
We added a **decode-only** LM head route override via `GRETA_LMHEAD_FORCE_ROUTE_DECODE=auto|valu|mfma` and expanded JSONL traces to correlate hidden/offsets and GEMM route at `decode0`. The goal is to isolate whether collapse comes from the **decode hidden** or the **decode LM head**.

## Cambios Implementados (ES)
- Nuevo flag `GRETA_LMHEAD_FORCE_ROUTE_DECODE` (aplica **solo** a decode; prefill queda intacto).
- El LM head ahora registra `lm_head_force_route_decode` en el JSONL de delta.
- Se agregaron campos de trazas en decode: `hidden_ptr`, `hidden_offset_bytes`, `hidden_token_index_used`, `lm_head_*` (M/N/K, lda/ldb/ldc, ptrs).
- El LM head se etiqueta como `lm_head_prefill` o `lm_head_decode` para diferenciar rutas.

## Implemented Changes (EN)
- New flag `GRETA_LMHEAD_FORCE_ROUTE_DECODE` (**decode-only**; prefill unaffected).
- Delta JSONL now records `lm_head_force_route_decode`.
- Added decode trace fields: `hidden_ptr`, `hidden_offset_bytes`, `hidden_token_index_used`, and LM head audit (M/N/K, lda/ldb/ldc, ptrs).
- LM head op is labeled as `lm_head_prefill` or `lm_head_decode` to disambiguate routing.

## Metodología / Methodology
- Flags base:
  - `GRETA_INT4_WEIGHTS=1`
  - `GRETA_MAX_SEQ_LEN=256`
  - `GRETA_TRACE_PREFILL_DECODE_DELTA=1`
  - `GRETA_TRACE_RMS_VERIFY=1`
  - `GRETA_TRACE_LMHEAD_CPU_PROBE=1`
- Decode-only route:
  - `GRETA_LMHEAD_FORCE_ROUTE_DECODE=valu|mfma|auto`
- Output JSONL:
  - `GRETA_TRACE_PREFILL_DECODE_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_17_delta.jsonl`

## Resultados (ES)
Pendiente de ejecución en MI300X (B3.17 runs p4_sys/p5_ba). Se completará con:
- Tabla `prefill_last` vs `decode0` (top1/top2/gap)
- `cpu_probe_agrees_gpu` por fase
- Comparación de rutas (MFMA vs VALU) en decode

## Results (EN)
Pending MI300X execution (B3.17 runs p4_sys/p5_ba). Will be completed with:
- `prefill_last` vs `decode0` table (top1/top2/gap)
- `cpu_probe_agrees_gpu` per phase
- Decode route comparison (MFMA vs VALU)

## Conclusión / Conclusion
Este bloque habilita el aislamiento de la ruta LM head **en decode** sin afectar prefill. La evidencia se completará tras la corrida remota.

---
L.E.T / Leandro Emanuel Timberini
