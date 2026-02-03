# B3.13 — Prefill vs Decode0 Delta (RMSNorm + LM Head Probes)

## Resumen Ejecutivo (ES)
Objetivo: aislar si el colapso en decode (top1=96965) proviene de RMSNorm final o del LM head (dequant/layout/ruta) comparando prefill_last vs decode0 en p4/p5. Se agregaron trazas de delta (hidden/rms/logits + rutas LM head) y un CPU probe mínimo para validar coherencia de logits.

## Executive Summary (EN)
Goal: isolate whether decode collapse (top1=96965) comes from final RMSNorm or LM head (dequant/layout/route) by comparing prefill_last vs decode0 on p4/p5. Added delta traces (hidden/rms/logits + LM head route) and a minimal CPU probe to validate logits coherence.

## Cambios Implementados (ES)
- Trazas `GRETA_TRACE_PREFILL_DECODE_DELTA`, `GRETA_TRACE_RMS_VERIFY`, `GRETA_TRACE_LMHEAD_CPU_PROBE`.
- JSONL `b3_13_delta.jsonl` con líneas `prefill_last` y `decode0`.
- Auditoría de LM head (ruta, quant, layout, tipos, scales hash) y CPU probe opcional.

## Implemented Changes (EN)
- Traces `GRETA_TRACE_PREFILL_DECODE_DELTA`, `GRETA_TRACE_RMS_VERIFY`, `GRETA_TRACE_LMHEAD_CPU_PROBE`.
- JSONL `b3_13_delta.jsonl` with `prefill_last` and `decode0` lines.
- LM head audit (route, quant, layout, dtypes, scales hash) and optional CPU probe.

## Configuración de Run (ES/EN)
- Modelo: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
- Flags:
  - `GRETA_INT4_WEIGHTS=1`
  - `GRETA_MAX_SEQ_LEN=256`
  - `GRETA_TRACE_PREFILL_DECODE_DELTA=1`
  - `GRETA_TRACE_RMS_VERIFY=1`
  - `GRETA_TRACE_LMHEAD_CPU_PROBE=1`
  - `GRETA_TRACE_PREFILL_DECODE_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_13_delta.jsonl`
- Prompts:
  - `p4_sys` (system+user template)
  - `p5_ba` (Buenos Aires)

## Evidencia (ES)
Tabla prefill_last vs decode0 (top1/top2/gap):

| Prompt | prefill_last top1 | decode0 top1 | gap prefill | gap decode0 |
|---|---:|---:|---:|---:|
| p4_sys | TBD | TBD | TBD | TBD |
| p5_ba  | TBD | TBD | TBD | TBD |

Extractos JSONL (máx 2 líneas por prompt):
```
TBD
```

## Evidence (EN)
Prefill_last vs decode0 table (top1/top2/gap):

| Prompt | prefill_last top1 | decode0 top1 | gap prefill | gap decode0 |
|---|---:|---:|---:|---:|
| p4_sys | TBD | TBD | TBD | TBD |
| p5_ba  | TBD | TBD | TBD | TBD |

JSONL excerpts (max 2 lines per prompt):
```
TBD
```

## Interpretación (ES)
Hipótesis ordenadas por evidencia:
1. TBD
2. TBD
3. TBD

## Interpretation (EN)
Evidence-ordered hypotheses:
1. TBD
2. TBD
3. TBD

## Próximo Paso / Next Step (B3.14)
TBD según resultado (RMSNorm vs LM head dequant/layout).

---
L.E.T / Leandro Emanuel Timberini
