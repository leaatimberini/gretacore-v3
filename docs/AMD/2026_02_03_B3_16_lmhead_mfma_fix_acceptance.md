# B3.16 — LM Head MFMA Fix (Acceptance)

## Resumen Ejecutivo (ES)
Objetivo: aplicar fix mínimo para evitar la ruta MFMA en LM head y validar que prefill_last y decode0 mantengan coherencia en prompts largos (p4/p5).

## Executive Summary (EN)
Goal: apply a minimal fix to avoid MFMA in LM head and validate that prefill_last and decode0 remain coherent on long prompts (p4/p5).

## Fix (ES)
- LM head en ruta `auto` fuerza VALU (MFMA deshabilitado) para evitar el path defectuoso detectado en B3.14.

## Fix (EN)
- LM head in `auto` forces VALU (MFMA disabled) to avoid the faulty MFMA path identified in B3.14.

## Metodología (ES/EN)
- Modelo: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
- Flags: `GRETA_TRACE_PREFILL_DECODE_DELTA=1`, `GRETA_TRACE_RMS_VERIFY=1`, `GRETA_TRACE_LMHEAD_CPU_PROBE=1`
- Prompts: `p4_sys`, `p5_ba`
- Outputs: `b3_16_p4_auto.jsonl`, `b3_16_p5_auto.jsonl`

## Resultados (ES)
Tabla prefill_last vs decode0 (top1/top2/gap):

| Prompt | prefill_last top1 | decode0 top1 | cpu_probe_agrees (prefill/decode) |
|---|---:|---:|---|
| p4_sys | 127158 | 96965 | true/true |
| p5_ba  | 127158 | 96965 | true/true |

## Results (EN)
Prefill_last vs decode0 table:

| Prompt | prefill_last top1 | decode0 top1 | cpu_probe_agrees (prefill/decode) |
|---|---:|---:|---|
| p4_sys | 127158 | 96965 | true/true |
| p5_ba  | 127158 | 96965 | true/true |

## Conclusión (ES)
Prefill ahora es coherente (cpu_probe_agrees=true) pero decode0 sigue colapsando a 96965; la igualdad prefill_last==decode0 no se cumple.

## Conclusion (EN)
Prefill is now coherent (cpu_probe_agrees=true) but decode0 still collapses to 96965; prefill_last==decode0 is not achieved.

---
L.E.T / Leandro Emanuel Timberini
