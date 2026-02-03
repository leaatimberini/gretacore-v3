# B3.14 — LM Head Force Route Isolation (MFMA vs VALU)

## Resumen Ejecutivo (ES)
Objetivo: aislar el bug del LM head en prefill forzando la ruta de GEMM (MFMA vs VALU) únicamente para el LM head y comparando prefill_last vs decode0 con CPU probe.

## Executive Summary (EN)
Goal: isolate the LM head bug in prefill by forcing GEMM route (MFMA vs VALU) only for LM head and comparing prefill_last vs decode0 with CPU probe.

## Motivación (ES)
B3.13 mostró discrepancia entre prefill_last y decode0 con CPU probe discordante en prefill y ruta MFMA activa. Se requiere forzar ruta para confirmar que el fallo está en MFMA.

## Motivation (EN)
B3.13 showed prefill_last vs decode0 mismatch with CPU probe disagreement in prefill and MFMA route active. We must force route to confirm MFMA is the culprit.

## Metodología (ES/EN)
- Nuevo switch: `GRETA_LMHEAD_FORCE_ROUTE=mfma|valu|auto` (solo LM head).
- Trazas: `GRETA_TRACE_PREFILL_DECODE_DELTA=1`, `GRETA_TRACE_RMS_VERIFY=1`, `GRETA_TRACE_LMHEAD_CPU_PROBE=1`.
- Salida JSONL: `b3_14_p4_mfma.jsonl`, `b3_14_p5_mfma.jsonl`, `b3_14_p4_valu.jsonl`, `b3_14_p5_valu.jsonl`.
- Modelo: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`.

## Resultados (ES)
Tabla prefill_last vs decode0 (top1/top2/gap):

| Prompt | Route | prefill_last top1 | decode0 top1 | cpu_probe_agrees (prefill/decode) |
|---|---|---:|---:|---|
| p4_sys | mfma | 79 | 96965 | false/true |
| p4_sys | valu | 127158 | 96965 | true/true |
| p5_ba  | mfma | 79 | 96965 | false/true |
| p5_ba  | valu | 127158 | 96965 | true/true |

## Results (EN)
Prefill_last vs decode0 table (top1/top2/gap):

| Prompt | Route | prefill_last top1 | decode0 top1 | cpu_probe_agrees (prefill/decode) |
|---|---|---:|---:|---|
| p4_sys | mfma | 79 | 96965 | false/true |
| p4_sys | valu | 127158 | 96965 | true/true |
| p5_ba  | mfma | 79 | 96965 | false/true |
| p5_ba  | valu | 127158 | 96965 | true/true |

## Conclusión (ES)
MFMA en prefill produce logits inconsistentes (cpu_probe_agrees=false). Forzar VALU corrige coherencia del prefill (cpu_probe_agrees=true).

## Conclusion (EN)
MFMA in prefill yields inconsistent logits (cpu_probe_agrees=false). Forcing VALU restores prefill consistency (cpu_probe_agrees=true).

## Próximo Paso / Next Step (B3.15)
Verificación puntual de layout/pesos del LM head.

---
L.E.T / Leandro Emanuel Timberini
