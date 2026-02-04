# B3.30 — Layer0 Attention Pipeline Root Cause (Prefill vs Decode0)

## Resumen Ejecutivo (ES)
Objetivo: aislar el primer punto de divergencia en la tubería de atención de la capa 0 entre `prefill_last` y `decode0` usando trazas deterministas (Q/K/V → QK → softmax → P·V → attn_out). Se instrumentó `GRETA_TRACE_ATTN_L0_PIPE` para capturar datos de head0 en layer0 y comparar por fase. Resultado: la primera divergencia aparece en **Q** (vector de consulta) con MAE alto, lo que indica que el problema ocurre **antes** del núcleo de atención (proyección Q/K/V en decode vs prefill), no en softmax ni en P·V. Esto explica por qué `attn_out` diverge desde layer0.

## Executive Summary (EN)
Goal: isolate the first divergence point in the layer0 attention pipeline between `prefill_last` and `decode0` using deterministic traces (Q/K/V → QK → softmax → P·V → attn_out). We instrumented `GRETA_TRACE_ATTN_L0_PIPE` to capture head0 data in layer0 and compare phases. Result: the first divergence appears in **Q** (query vector) with high MAE, indicating the issue happens **before** the attention core (Q/K/V projection in decode vs prefill), not in softmax or P·V. This explains why `attn_out` diverges from layer0 onward.

## Contexto (ES)
- B3.29 mostró el primer mismatch en `attn_out` (layer0) con `x_in` equivalente.
- B3.30 amplía la instrumentación para comparar sub-etapas internas de atención.
- Se usó `GRETA_TRACE_STAGE_DEBUG_INPUT=1` para alinear `decode0` con el último token de prefill (mismo token/pos_id) y aislar diferencias de pipeline, no de input semántico.

## Context (EN)
- B3.29 showed the first mismatch at `attn_out` (layer0) with equivalent `x_in`.
- B3.30 expands instrumentation to compare internal attention sub-stages.
- `GRETA_TRACE_STAGE_DEBUG_INPUT=1` was used to align `decode0` with the last prefill token (same token/pos_id) to isolate pipeline differences, not input semantics.

## Metodología (ES)
Instrumentación activada (layer0, head0, decode0 + prefill_last):
- `GRETA_TRACE_ATTN_L0_PIPE=1`
- `GRETA_TRACE_ATTN_L0_PIPE_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_30_attn_l0_pipe.jsonl`
- `GRETA_TRACE_STAGE_DEBUG_INPUT=1`
- `GRETA_TRACE_PROMPT_ID=p4_sys | p5_ba`

Runner:
- `tools/benchmarks/run_b3_30_mi300x.sh 129.212.184.200 2026-02-03`

Artefactos:
- `artifacts_remote/2026-02-03/b3_30/gretacore_b3_30_artifacts.tgz`
- `artifacts_remote/2026-02-03/b3_30/b3_30_analysis.txt`

## Methodology (EN)
Instrumentation enabled (layer0, head0, decode0 + prefill_last):
- `GRETA_TRACE_ATTN_L0_PIPE=1`
- `GRETA_TRACE_ATTN_L0_PIPE_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_30_attn_l0_pipe.jsonl`
- `GRETA_TRACE_STAGE_DEBUG_INPUT=1`
- `GRETA_TRACE_PROMPT_ID=p4_sys | p5_ba`

Runner:
- `tools/benchmarks/run_b3_30_mi300x.sh 129.212.184.200 2026-02-03`

Artifacts:
- `artifacts_remote/2026-02-03/b3_30/gretacore_b3_30_artifacts.tgz`
- `artifacts_remote/2026-02-03/b3_30/b3_30_analysis.txt`

## Resultados (ES)
Tabla MAE (prefill_last vs decode0) por sub-etapa (layer0/head0):

| Prompt | q MAE | k MAE | v MAE | qk MAE | softmax MAE | pv MAE | attn_out MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| p4_sys | 5.37518e-01 | 1.99312e-01 | 2.68110e-03 | 1.32675e-01 | 6.50352e-04 | 3.46855e-05 | 3.46854e-05 |
| p5_ba  | 8.88882e-01 | 2.55535e-01 | 2.98710e-03 | 1.98819e-01 | 4.05699e-03 | 9.49639e-05 | 9.49638e-05 |

Primera divergencia:
- **Q (query)** es el primer punto de ruptura en ambas prompts.

Observación clave:
- `pos_id` y `seq_len_used` coinciden entre prefill_last y decode0 (debug input), por lo que la diferencia no proviene de índices/posiciones sino del **camino de proyección Q/K/V** (GEMM/MFMA/VALU) entre prefill (S>1) y decode (S=1).

## Results (EN)
MAE table (prefill_last vs decode0) per sub-stage (layer0/head0):

| Prompt | q MAE | k MAE | v MAE | qk MAE | softmax MAE | pv MAE | attn_out MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| p4_sys | 5.37518e-01 | 1.99312e-01 | 2.68110e-03 | 1.32675e-01 | 6.50352e-04 | 3.46855e-05 | 3.46854e-05 |
| p5_ba  | 8.88882e-01 | 2.55535e-01 | 2.98710e-03 | 1.98819e-01 | 4.05699e-03 | 9.49639e-05 | 9.49638e-05 |

First divergence:
- **Q (query)** is the first breaking point for both prompts.

Key observation:
- `pos_id` and `seq_len_used` match between prefill_last and decode0 (debug input), so the difference does not come from indexing/position but from the **Q/K/V projection path** (GEMM/MFMA/VALU) between prefill (S>1) and decode (S=1).

## Conclusión Técnica (ES)
La divergencia se introduce **antes de la atención**: la proyección Q/K/V en decode (S=1) no reproduce la salida de prefill (S>1), incluso con el mismo token/posición. Esto explica el mismatch en `attn_out` y el colapso posterior en logits. La corrección debe enfocarse en el camino de GEMM/GEMV de decode (VALU vs MFMA), o forzar el mismo kernel/ruta para Q/K/V entre prefill y decode.

## Technical Conclusion (EN)
The divergence is introduced **before attention**: the Q/K/V projection in decode (S=1) does not reproduce the prefill (S>1) output even with the same token/position. This explains the mismatch in `attn_out` and the subsequent collapse in logits. The fix must target the decode GEMM/GEMV path (VALU vs MFMA) or force the same kernel/route for Q/K/V across prefill and decode.

## Próximo Paso (B3.31) (ES)
- Auditar el path de GEMM/GEMV de Q/K/V para S=1 (decode), incluyendo ruta MFMA vs VALU.
- Forzar `GRETA_GEMM_FORCE=MFMA` en decode para validar si el mismatch desaparece.
- Si MFMA arregla, corregir el kernel VALU o forzar ruta consistente para decode.

## Next Step (B3.31) (EN)
- Audit the Q/K/V GEMM/GEMV path for S=1 (decode), including MFMA vs VALU route.
- Force `GRETA_GEMM_FORCE=MFMA` in decode to validate if the mismatch disappears.
- If MFMA fixes it, correct the VALU kernel or force a consistent route for decode.

---

**L.E.T / Leandro Emanuel Timberini**
