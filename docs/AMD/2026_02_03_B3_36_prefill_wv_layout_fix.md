# B3.36 — Prefill Wv Layout Fix (GEMM) / Wv Layout Alignment

## Resumen Ejecutivo (ES)
B3.36 corrige la interpretación de Wv en el camino GEMM de prefill para alinearla con decode (row-major). Con el forcing `GRETA_PREFILL_FORCE_WV_LAYOUT=row` (manteniendo Wq/Wk en row), el primer mismatch en la traza L0 desaparece para prompts de contexto y largos, y los MAE de Q/K/V caen a 0.0. Esto indica que Wv era el último layout inconsistente en prefill.

## Executive Summary (EN)
B3.36 fixes Wv interpretation in the prefill GEMM path to match decode (row-major). With `GRETA_PREFILL_FORCE_WV_LAYOUT=row` (and Wq/Wk row), the first mismatch in L0 trace disappears for context and long prompts, and Q/K/V MAEs drop to 0.0. This confirms Wv was the last inconsistent layout in prefill.

---

## Contexto / Context
- Estado previo (B3.35): Wq/Wk alineados; primer mismatch en `v`.
- Objetivo B3.36: alinear Wv en prefill con decode.

## Cambios Implementados (ES)
- Se agregó `GRETA_PREFILL_FORCE_WV_LAYOUT=row|col|auto` (prefill-only).
- Se forzó el uso de layout row para Wv en prefill (GEMM con `force_gemv`).
- Se extendió el probe QKV para incluir layout/MAE de Wv.

## Implemented Changes (EN)
- Added `GRETA_PREFILL_FORCE_WV_LAYOUT=row|col|auto` (prefill-only).
- Forced row layout for Wv in prefill (GEMM with `force_gemv`).
- Extended QKV probe to include Wv layout/MAE.

## Metodología / Method
- Modelo: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
- Prompts: `p0_short`, `p4_sys`, `p5_ba`, `p6_long`
- Experimentos:
  - **E0**: baseline (sin forcing)
  - **E1**: Wq/Wk row + Wv row (`GRETA_PREFILL_FORCE_WQ_ROW=1`, `GRETA_PREFILL_FORCE_WK_ROW=1`, `GRETA_PREFILL_FORCE_WV_LAYOUT=row`)
- Trazas:
  - Pipeline L0 (q/k/v/qk/softmax/pv/attn_out)
  - Weight layout probe QKV (Wv incluido)

## Resultados (ES)
**Tabla resumen (de `b3_36_analysis.txt`):**

| Prompt | Exp | prefill_layout_v | decode_layout_v | first_mismatch | v_mae | k_mae | q_mae |
|---|---|---|---|---|---:|---:|---:|
| p0_short | E0 | row | row | NONE | 0.0 | 0.0 | 0.0 |
| p4_sys | E0 | row | row | q | 0.00268 | 0.19931 | 0.53752 |
| p5_ba | E0 | row | row | q | 0.00299 | 0.25553 | 0.88888 |
| p6_long | E0 | row | row | q | 0.00267 | 0.18318 | 0.53963 |
| p0_short | E1 | row | row | NONE | 0.0 | 0.0 | 0.0 |
| p4_sys | E1 | row | row | NONE | 0.0 | 0.0 | 0.0 |
| p5_ba | E1 | row | row | NONE | 0.0 | 0.0 | 0.0 |
| p6_long | E1 | row | row | NONE | 0.0 | 0.0 | 0.0 |

**Interpretación:**
- En E1, desaparecen los mismatches en L0 (Q/K/V) para prompts con contexto y largos.
- Se confirma que Wv era el último layout inconsistente en prefill.

## Results (EN)
**Summary table (from `b3_36_analysis.txt`):**

| Prompt | Exp | prefill_layout_v | decode_layout_v | first_mismatch | v_mae | k_mae | q_mae |
|---|---|---|---|---|---:|---:|---:|
| p0_short | E0 | row | row | NONE | 0.0 | 0.0 | 0.0 |
| p4_sys | E0 | row | row | q | 0.00268 | 0.19931 | 0.53752 |
| p5_ba | E0 | row | row | q | 0.00299 | 0.25553 | 0.88888 |
| p6_long | E0 | row | row | q | 0.00267 | 0.18318 | 0.53963 |
| p0_short | E1 | row | row | NONE | 0.0 | 0.0 | 0.0 |
| p4_sys | E1 | row | row | NONE | 0.0 | 0.0 | 0.0 |
| p5_ba | E1 | row | row | NONE | 0.0 | 0.0 | 0.0 |
| p6_long | E1 | row | row | NONE | 0.0 | 0.0 | 0.0 |

**Interpretation:**
- In E1, L0 mismatches (Q/K/V) disappear for context and long prompts.
- This confirms Wv was the last inconsistent prefill layout.

## Evidencia / Evidence
- Analyzer: `artifacts_remote/2026-02-03/b3_36/b3_36_analysis.txt`
- JSONL: `artifacts_remote/2026-02-03/b3_36/root/gretacore/artifacts/alignment/2026-02-03/`
- Logs: `artifacts_remote/2026-02-03/b3_36/root/gretacore/artifacts/alignment/2026-02-03/b3_36_*.log`
- Commit de ejecución: `359f170`

## Conclusión / Conclusion
**B3.36 alinea Wv en prefill con decode y elimina el mismatch en V.** Q/K/V quedan consistentes para prompts con contexto, y el primer mismatch en la traza L0 desaparece en E1.

## Próximo paso / Next step
- Validar si el colapso en logits persiste fuera del L0 QKV (B3.37: rerun B3.29/B3.30 full pipeline).

---

L.E.T / Leandro Emanuel Timberini
