# B3.17 — Decode LM Head Isolation (MFMA vs VALU, decode-only)

## Resumen Ejecutivo (ES)
Se introdujo control explícito de ruta del LM head **solo en decode** (`GRETA_LMHEAD_FORCE_ROUTE_DECODE=auto|valu|mfma`) y se extendieron las trazas JSONL para correlacionar hidden/offsets y ruta GEMM en `decode0`. El objetivo fue aislar si el colapso proviene del LM head en decode o del hidden previo.

**Resultado clave:** forzar MFMA/VALU en decode **no cambia** el colapso (`decode0_top1=96965`) y el CPU probe concuerda con GPU en decode. Esto descarta el LM head decode como causa principal y apunta al hidden de decode.

## Executive Summary (EN)
We added a **decode-only** LM head route override (`GRETA_LMHEAD_FORCE_ROUTE_DECODE=auto|valu|mfma`) and expanded JSONL traces to correlate hidden/offsets and GEMM route at `decode0`. The goal was to isolate whether collapse comes from the decode LM head or from the preceding hidden.

**Key result:** forcing MFMA/VALU in decode **does not change** the collapse (`decode0_top1=96965`) and the CPU probe agrees with GPU in decode. This rules out the decode LM head and points to the decode hidden state.

## Metodología / Methodology
- Modelo: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
- Flags base:
  - `GRETA_INT4_WEIGHTS=1`
  - `GRETA_MAX_SEQ_LEN=256`
  - `GRETA_TRACE_PREFILL_DECODE_DELTA=1`
  - `GRETA_TRACE_RMS_VERIFY=1`
  - `GRETA_TRACE_LMHEAD_CPU_PROBE=1`
  - `GRETA_TRACE_HIDDEN_EQUIV=1`
  - `GRETA_TRACE_LAYER_DELTA=1`
- Decode-only:
  - `GRETA_LMHEAD_FORCE_ROUTE_DECODE=mfma|valu`
- JSONL:
  - `/root/gretacore/artifacts/alignment/2026-02-03/b3_17_*_decode_*.jsonl`

## Resultados (ES)
Tabla (prefill_last vs decode0):

| Prompt | Force decode | lm_head_route (decode0) | prefill_last top1 | decode0 top1 | cpu_probe prefill | cpu_probe decode0 |
|---|---|---|---:|---:|---|---|
| p4_sys | mfma | MFMA | 127158 | 96965 | true | true |
| p4_sys | valu | VALU | 127158 | 96965 | true | true |
| p5_ba  | mfma | MFMA | 127158 | 96965 | true | true |
| p5_ba  | valu | VALU | 127158 | 96965 | true | true |

Observaciones:
- El colapso persiste independientemente de la ruta en decode.
- `cpu_probe_agrees_gpu=true` en prefill y decode ⇒ el LM head está consistente con su entrada.
- `hidden_equiv` muestra hashes distintos entre prefill y decode, indicando divergencia previa al LM head.

## Results (EN)
Table (prefill_last vs decode0):

| Prompt | Force decode | lm_head_route (decode0) | prefill_last top1 | decode0 top1 | cpu_probe prefill | cpu_probe decode0 |
|---|---|---|---:|---:|---|---|
| p4_sys | mfma | MFMA | 127158 | 96965 | true | true |
| p4_sys | valu | VALU | 127158 | 96965 | true | true |
| p5_ba  | mfma | MFMA | 127158 | 96965 | true | true |
| p5_ba  | valu | VALU | 127158 | 96965 | true | true |

Notes:
- Collapse persists regardless of the decode route.
- `cpu_probe_agrees_gpu=true` in prefill and decode ⇒ LM head is consistent with its input.
- `hidden_equiv` shows different hashes between prefill and decode, indicating divergence before the LM head.

## Conclusión / Conclusion
El LM head decode **no es la causa primaria**. El foco debe pasar a la transición de hidden entre prefill y decode (B3.18/B3.19).

---
L.E.T / Leandro Emanuel Timberini
