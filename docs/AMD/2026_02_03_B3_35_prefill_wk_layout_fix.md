# B3.35 — Prefill Wk Layout Fix (GEMM) / Wk Layout Alignment

## Resumen Ejecutivo (ES)
En B3.34 se corrigió Wq para prefill y el primer mismatch se movió a `k`. En B3.35 se unificó la interpretación de Wk en el camino GEMM de prefill para igualar la semántica de decode (row-major). El resultado es que `k_mae` cae a 0.0 y el primer mismatch avanza a `v`, confirmando que la discrepancia en K era el bloqueador inmediato. Queda pendiente corregir Wv (B3.36).

## Executive Summary (EN)
B3.34 fixed Wq for prefill and shifted the first mismatch to `k`. In B3.35 we unified the Wk interpretation in the prefill GEMM path to match decode (row-major). As a result, `k_mae` drops to 0.0 and the first mismatch moves to `v`, confirming K layout was the immediate blocker. Next step is Wv (B3.36).

---

## Contexto / Context
- Proyecto: GRETA CORE (MI300X, CDNA3)
- Problema: colapso determinista en decode0
- Estado previo: Wq corregido; primer mismatch en `k`
- Objetivo B3.35: alinear Wk en prefill con decode (row-major)

## Cambios Implementados (ES)
- Se forzó la interpretación de Wk en prefill a row-major (misma semántica que decode).
- Se extendió la traza del pipeline L0 para registrar layout/MAE de K.
- Se creó runner/analyzer B3.35 para comparar E0 (baseline) vs E1 (fix Wk row).

## Implemented Changes (EN)
- Forced Wk interpretation in prefill to row-major (same semantics as decode).
- Extended L0 pipeline trace to record K layout/MAE.
- Added B3.35 runner/analyzer to compare E0 (baseline) vs E1 (Wk row fix).

## Metodología / Method
- Modelo: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
- Prompts: `p0_short`, `p4_sys`, `p5_ba`, `p6_long`
- Experimentos:
  - **E0**: baseline (sin forcing)
  - **E1**: prefill Wq+Wk row (`GRETA_PREFILL_FORCE_WQ_ROW=1`, `GRETA_PREFILL_FORCE_WK_ROW=1`)
- Trazas:
  - Pipeline layer0 (q/k/v/qk/softmax/pv/attn_out)
  - Weight layout probe para Wk

## Resultados (ES)
**Tabla resumen (extraída de `b3_35_analysis.txt`):**

| Prompt | Exp | prefill_layout_k | decode_layout_k | first_mismatch | k_mae | q_mae |
|---|---|---|---|---|---:|---:|
| p0_short | E0 | row | row | NONE | 0.0 | 0.0 |
| p4_sys | E0 | row | row | q | 0.1993 | 0.5375 |
| p5_ba | E0 | row | row | q | 0.2555 | 0.8889 |
| p6_long | E0 | row | row | q | 0.1832 | 0.5396 |
| p0_short | E1 | row | row | NONE | 0.0 | 0.0 |
| p4_sys | E1 | row | row | v | 0.0 | 0.0 |
| p5_ba | E1 | row | row | v | 0.0 | 0.0 |
| p6_long | E1 | row | row | v | 0.0 | 0.0 |

**Interpretación:**
- En E1, `k_mae = 0.0` para prompts de contexto, y el primer mismatch se mueve a `v`.
- Q y K quedan alineados entre prefill y decode.

## Results (EN)
**Summary table (from `b3_35_analysis.txt`):**

| Prompt | Exp | prefill_layout_k | decode_layout_k | first_mismatch | k_mae | q_mae |
|---|---|---|---|---|---:|---:|
| p0_short | E0 | row | row | NONE | 0.0 | 0.0 |
| p4_sys | E0 | row | row | q | 0.1993 | 0.5375 |
| p5_ba | E0 | row | row | q | 0.2555 | 0.8889 |
| p6_long | E0 | row | row | q | 0.1832 | 0.5396 |
| p0_short | E1 | row | row | NONE | 0.0 | 0.0 |
| p4_sys | E1 | row | row | v | 0.0 | 0.0 |
| p5_ba | E1 | row | row | v | 0.0 | 0.0 |
| p6_long | E1 | row | row | v | 0.0 | 0.0 |

**Interpretation:**
- In E1, `k_mae = 0.0` for context prompts and the first mismatch moves to `v`.
- Q and K are aligned between prefill and decode.

## Evidencia / Evidence
- Analyzer: `artifacts_remote/2026-02-03/b3_35/b3_35_analysis.txt`
- JSONL: `artifacts_remote/2026-02-03/b3_35/root/gretacore/artifacts/alignment/2026-02-03/` (E0/E1 runs)
- Logs: `artifacts_remote/2026-02-03/b3_35/root/gretacore/artifacts/alignment/2026-02-03/b3_35_*.log`
- Commit de ejecución: `6a36c6d`

## Conclusión / Conclusion
**B3.35 confirma que Wk en prefill estaba interpretado con un layout incorrecto.** La corrección unifica K entre prefill y decode y desplaza el primer mismatch a `v`. El siguiente paso es B3.36: alinear Wv en prefill para cerrar la cadena Q/K/V.

## Próximo paso / Next step
- **B3.36:** Fix Wv layout/packing en prefill GEMM y revalidación de pipeline layer0.

---

L.E.T / Leandro Emanuel Timberini
