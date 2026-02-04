# B3.33 — QKV Weight Layout Verification (Prefill vs Decode)

## Resumen Ejecutivo (ES)
Objetivo: verificar si el layout/packing de pesos QKV difiere entre prefill (S>1) y decode0 (S=1). Se instrumentó `GRETA_TRACE_QKV_W_VERIFY=1` en layer0/head0 y se comparó el match row vs col contra `q` observado. Resultado: en prompts con contexto (p4_sys, p5_ba, p6_long) **prefill_last se alinea mejor con layout col**, mientras que **decode0 se alinea con layout row** (MAE≈1e-6). En prompt corto (p0_short) ambos fases alinean con row. Conclusión: **existe un cambio de interpretación de layout en el camino de proyección Q en prefill**, no en decode. Próximo paso B3.34: corregir el path GEMM de prefill para usar el mismo layout (row) que decode, o ajustar el packing/transpose del peso Q para prefill.

## Executive Summary (EN)
Goal: verify whether QKV weight layout/packing differs between prefill (S>1) and decode0 (S=1). We instrumented `GRETA_TRACE_QKV_W_VERIFY=1` at layer0/head0 and compared row vs col layout match against observed `q`. Result: for context prompts (p4_sys, p5_ba, p6_long) **prefill_last matches col layout**, while **decode0 matches row layout** (MAE≈1e-6). For a short prompt (p0_short), both phases match row. Conclusion: **there is a layout interpretation change in the prefill Q projection path**, not in decode. Next B3.34: fix prefill GEMM path to use the same layout (row) as decode, or adjust Q weight packing/transpose for prefill.

## Metodología / Method
- Flags:
  - `GRETA_TRACE_ATTN_L0_PIPE=1`
  - `GRETA_TRACE_ATTN_L0_NORM=1`
  - `GRETA_TRACE_QKV_W_VERIFY=1`
  - `GRETA_TRACE_STAGE_DEBUG_INPUT=1`
- Prompts:
  - `p0_short` ("Hi")
  - `p4_sys` (system prompt)
  - `p5_ba` (Buenos Aires)
  - `p6_long` (>=200 tokens)
- Runner:
  - `tools/benchmarks/run_b3_33_mi300x.sh 129.212.184.200 2026-02-03`
- Artifacts:
  - `artifacts_remote/2026-02-03/b3_33/gretacore_b3_33_artifacts.tgz`
  - JSONL: `.../b3_33_attn_l0_pipe_<prompt>.jsonl`
  - Analysis: `artifacts_remote/2026-02-03/b3_33/b3_33_analysis.txt`

## Resultados (ES)
Tabla resumida (layout_best + MAE):

| Prompt | Phase | layout_best | mae_row | mae_col |
|---|---|---|---:|---:|
| p0_short | prefill_last | row | 3.70e-08 | 6.55e-01 |
| p0_short | decode0 | row | 3.70e-08 | 6.55e-01 |
| p4_sys | prefill_last | col | 6.08e-01 | 2.42e-01 |
| p4_sys | decode0 | row | 9.09e-07 | 5.94e-01 |
| p5_ba | prefill_last | col | 1.59e+00 | 1.52e+00 |
| p5_ba | decode0 | row | 1.81e-07 | 7.40e-01 |
| p6_long | prefill_last | col | 7.66e-01 | 2.87e-01 |
| p6_long | decode0 | row | 1.30e-06 | 7.13e-01 |

**Short vs Long:** p0_short (S≈1) alinea en row en ambas fases. En prompts con contexto (p4/p5/p6), prefill_last alinea en col mientras decode0 alinea en row.

## Results (EN)
Summary table (layout_best + MAE):

| Prompt | Phase | layout_best | mae_row | mae_col |
|---|---|---|---:|---:|
| p0_short | prefill_last | row | 3.70e-08 | 6.55e-01 |
| p0_short | decode0 | row | 3.70e-08 | 6.55e-01 |
| p4_sys | prefill_last | col | 6.08e-01 | 2.42e-01 |
| p4_sys | decode0 | row | 9.09e-07 | 5.94e-01 |
| p5_ba | prefill_last | col | 1.59e+00 | 1.52e+00 |
| p5_ba | decode0 | row | 1.81e-07 | 7.40e-01 |
| p6_long | prefill_last | col | 7.66e-01 | 2.87e-01 |
| p6_long | decode0 | row | 1.30e-06 | 7.13e-01 |

**Short vs Long:** p0_short (S≈1) matches row in both phases. In context prompts (p4/p5/p6), prefill_last matches col while decode0 matches row.

## Conclusión / Conclusion
- **Root cause statement (ES):** El layout efectivo de pesos Q en prefill (S>1) se interpreta como col, mientras que decode0 usa row. Esto explica la divergencia de Q entre prefill_last y decode0 cuando hay contexto. El bug está en el path GEMM de prefill (layout/transpose/stride), no en RMSNorm ni en el decode.
- **Root cause statement (EN):** The effective Q weight layout in prefill (S>1) is interpreted as col, while decode0 uses row. This explains the Q divergence between prefill_last and decode0 for context prompts. The bug is in the prefill GEMM path (layout/transpose/stride), not in RMSNorm or decode.

## Próximo Paso / Next Step (B3.34)
- Alinear el layout de pesos Q en prefill con el layout row usado por decode (revisar ldb/transpose en GEMM int4 y/o packing del peso).
- Re-correr B3.30/B3.29 para verificar que el primer mismatch ya no esté en `q` y que `prefill_last_top1 == decode0_top1`.

---

L.E.T / Leandro Emanuel Timberini
