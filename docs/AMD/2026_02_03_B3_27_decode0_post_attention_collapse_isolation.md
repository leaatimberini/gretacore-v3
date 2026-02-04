# B3.27 — Decode0 Post-Attention Collapse Isolation (StageTrace)

## Resumen Ejecutivo (ES)
B3.26 dejó P*V correcto y QK/softmax alineados, pero el colapso en decode0 persiste. Se implementó un trazado por etapas (StageTrace) para comparar `prefill_last` vs `decode0` en capas [0,1,2,15,31] y puntos críticos (`x_in`, `attn_out`, `x_after_attn`, `mlp_out`, `x_after_mlp`, `final_norm`, `lm_head_in`, `logits`). El resultado mostró que la primera divergencia aparece en `x_in` de la capa 0 (MAE ~1e-2), indicando que la discrepancia se origina antes de la atención/MLP/final RMSNorm y sugiere un problema en la orquestación de decode (token de entrada/pos_id/seq_len) más que en kernels posteriores.

## Executive Summary (EN)
After B3.26 fixed P*V and QK/softmax alignment, decode0 collapse still persisted. We added a StageTrace that compares `prefill_last` vs `decode0` across layers [0,1,2,15,31] and key points (`x_in`, `attn_out`, `x_after_attn`, `mlp_out`, `x_after_mlp`, `final_norm`, `lm_head_in`, `logits`). The first divergence appears at layer-0 `x_in` (MAE ~1e-2), indicating the discrepancy arises before attention/MLP/final RMSNorm. This points to decode orchestration (input token/pos_id/seq_len handling) rather than downstream kernels.

## Contexto (ES)
- B3.23: QK y softmax en decode0 layer 31 coinciden con referencia FP64.
- B3.24–B3.26: P*V y layout/stride de V corregidos; pv_mae ~1e-7 en p4/p5.
- Aun así, `prefill_last_top1 != decode0_top1` (colapso a 96965).

## Context (EN)
- B3.23: decode0 layer-31 QK/softmax match FP64 reference.
- B3.24–B3.26: P*V and V layout/stride fixed; pv_mae ~1e-7 on p4/p5.
- Yet `prefill_last_top1 != decode0_top1` (collapse to 96965).

## Metodología (ES/EN)
- Flags:
  - `GRETA_TRACE_STAGE=1`
  - `GRETA_TRACE_STAGE_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_27_stage.jsonl`
  - `GRETA_TRACE_STAGE_LAYERS="0,1,2,15,31"`
  - `GRETA_TRACE_STAGE_POINTS="attn_out,x_after_attn,mlp_out,x_after_mlp,final_norm,lm_head_in,logits,x_in"`
  - `GRETA_TRACE_STAGE_PHASES="prefill_last,decode0"`
- Prompts:
  - `p4_sys` (system + user template)
  - `p5_ba` (Buenos Aires)
- Binario: `tools/inference/build/greta_infer` (MI300X)

## Resultados (ES)
Primer punto de divergencia (prefill_last vs decode0):

| Prompt | First mismatch | Layer | Point | Métrica |
|---|---|---:|---|---|
| p4_sys | Sí | 0 | x_in | sample_mae 0.0114 |
| p5_ba  | Sí | 0 | x_in | sample_mae 0.00996 |

Observación clave: `x_in` en decode0 usa buffer single-token (`token_index=0`) con `pos_id=205`, mientras `prefill_last` usa `token_index=204` en secuencia larga (`seq_len=205`). La divergencia aparece antes de atención/MLP.

## Results (EN)
First divergence point (prefill_last vs decode0):

| Prompt | First mismatch | Layer | Point | Metric |
|---|---|---:|---|---|
| p4_sys | Yes | 0 | x_in | sample_mae 0.0114 |
| p5_ba  | Yes | 0 | x_in | sample_mae 0.00996 |

Key observation: decode0 `x_in` uses single-token buffer (`token_index=0`) with `pos_id=205`, while `prefill_last` uses `token_index=204` in a long sequence (`seq_len=205`). Divergence appears before attention/MLP.

## Evidencia (ES/EN)
- JSONL: `artifacts_remote/2026-02-03/b3_27/root/gretacore/artifacts/alignment/2026-02-03/b3_27_stage.jsonl`
- Logs: `artifacts_remote/2026-02-03/b3_27/root/gretacore/artifacts/alignment/2026-02-03/b3_27_p4.log` / `b3_27_p5.log`
- Análisis: `artifacts_remote/2026-02-03/b3_27/b3_27_analysis.txt`

## Conclusión (ES)
La divergencia prefill_last → decode0 aparece en `x_in` (layer 0). Esto apunta a la selección de token/posición en decode (orquestación) como punto de quiebre temprano, no a kernels de atención/MLP/RMSNorm. No se aplica fix en este bloque; se recomienda un B3.28 para validar la coherencia del token de entrada (ID real usado, embedding esperado) y la transición pos_id/seq_len.

## Conclusion (EN)
The prefill_last → decode0 divergence appears at `x_in` (layer 0). This indicates the decode input token/position selection as the earliest breakpoint, not attention/MLP/RMSNorm kernels. No fix is applied in this block; B3.28 should validate decode input token ID, embedding correctness, and pos_id/seq_len transition.

---

L.E.T / Leandro Emanuel Timberini
