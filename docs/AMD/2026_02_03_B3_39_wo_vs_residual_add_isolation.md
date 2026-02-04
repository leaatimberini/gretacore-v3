# B3.39 — WO vs Residual Add Isolation (Decode0 Collapse)

## Resumen Ejecutivo (ES)
B3.39 aisla el primer mismatch **después** de Q/K/V: ocurre en **WO (attn_proj)**, no en la suma residual. En prompts de contexto/largos, `attn_out` sigue coincidiendo (MAE ~1e‑10), pero `wo_out` ya difiere (MAE ~7.4e‑04 a 1.35e‑03), y `x_after_attn` hereda esa diferencia. Por lo tanto, la causa inmediata está en la proyección WO (GEMM/packing/layout) en decode0.

## Executive Summary (EN)
B3.39 isolates the first mismatch **after** Q/K/V: it occurs at **WO (attn_proj)**, not at the residual add. For context/long prompts, `attn_out` still matches (MAE ~1e‑10), but `wo_out` already diverges (MAE ~7.4e‑04 to 1.35e‑03), and `x_after_attn` inherits that delta. The immediate root cause is the WO projection (GEMM/packing/layout) in decode0.

---

## Contexto / Context
- B3.36: Q/K/V alineados prefill vs decode.
- B3.38: mismatch aparece en `x_after_attn` (post‑QKV).
- B3.39: separar **WO** vs **residual add**.

## Metodología / Method
- Prompts: `p0_short` (control), `p4_sys`, `p6_long`
- Modo: E1 (alignment activo)
- Trazas: `GRETA_TRACE_STAGE` con puntos:
  `x_in, attn_out, wo_out, x_after_attn, ffn_norm, mlp_out, x_out, final_norm, lm_head_in, logits`
- Output:
  `b3_39_wo_resid_<prompt>_E1.jsonl`

## Resultados (ES)
**Tabla resumida (de `b3_39_analysis.txt`):**

| Prompt | Exp | attn_out_mae | wo_out_mae | x_in_mae | x_after_attn_mae | first_mismatch_stage | prefill_last_top1 | decode0_top1 | collapse_96965 |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| p0_short | E1 | ~2.13e‑11 | ~1.32e‑10 | 0.0 | ~9.80e‑11 | NONE | 96965 | 96965 | true |
| p4_sys | E1 | ~1.68e‑10 | 7.40e‑04 | 0.0 | 7.40e‑04 | wo_out | 93590 | 96965 | true |
| p6_long | E1 | ~7.00e‑10 | 1.35e‑03 | 0.0 | 1.35e‑03 | wo_out | 52196 | 96965 | true |

**Interpretación:**
- `attn_out` coincide (MAE ~1e‑10).
- `wo_out` es el primer punto de divergencia → el problema está en **WO GEMM** o en su configuración para decode0.
- `x_in` coincide; la suma residual no introduce error adicional.

## Results (EN)
**Summary table (from `b3_39_analysis.txt`):**

| Prompt | Exp | attn_out_mae | wo_out_mae | x_in_mae | x_after_attn_mae | first_mismatch_stage | prefill_last_top1 | decode0_top1 | collapse_96965 |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| p0_short | E1 | ~2.13e‑11 | ~1.32e‑10 | 0.0 | ~9.80e‑11 | NONE | 96965 | 96965 | true |
| p4_sys | E1 | ~1.68e‑10 | 7.40e‑04 | 0.0 | 7.40e‑04 | wo_out | 93590 | 96965 | true |
| p6_long | E1 | ~7.00e‑10 | 1.35e‑03 | 0.0 | 1.35e‑03 | wo_out | 52196 | 96965 | true |

**Interpretation:**
- `attn_out` matches (MAE ~1e‑10).
- `wo_out` is the first divergence → issue is in **WO GEMM** or its decode0 configuration.
- `x_in` matches; residual add is not the source.

## Evidencia / Evidence
- Analyzer: `artifacts_remote/2026-02-03/b3_39/b3_39_analysis.txt`
- JSONL: `artifacts_remote/2026-02-03/b3_39/root/gretacore/artifacts/alignment/2026-02-03/`
- Logs: `artifacts_remote/2026-02-03/b3_39/root/gretacore/artifacts/alignment/2026-02-03/b3_39_E1_*.log`
- Commit de ejecución: `5379fe2`

## Conclusión / Conclusion
**La divergencia inicia en `wo_out`.** La causa inmediata es la proyección WO en decode0 (layout/packing/stride o ruta GEMM), no la suma residual.

## Próximo paso / Next step
- **B3.40:** aislar y corregir layout/packing/stride de WO en decode0 (comparar GEMM S>1 vs S=1, y forzar ruta homogénea).

---

L.E.T / Leandro Emanuel Timberini
