# B3.34 — Prefill Wq Layout Fix (GEMV-forced) and Acceptance

## Resumen Ejecutivo (ES)
Objetivo: alinear la interpretación del layout de Wq en prefill (S>1) con decode0 (S=1). En B3.33 se observó que prefill_last alineaba con layout **col**, mientras decode0 con **row**. En B3.34 se aplicó un fix mínimo: forzar el path de Q en prefill a ejecutar GEMV (row-major) vía `GRETA_PREFILL_FORCE_WQ_ROW=1` / `GRETA_PREFILL_QKV_LAYOUT=row`. Resultado: en prompts con contexto (p4_sys, p5_ba, p6_long) **prefill_last y decode0 alinean ambos con row**, y la divergencia ya no ocurre en `q` (q_mae=0.0). El primer mismatch se desplaza a `k`. Esto confirma que el problema era el path GEMM de prefill para Wq.

## Executive Summary (EN)
Goal: align Wq layout interpretation in prefill (S>1) with decode0 (S=1). B3.33 showed prefill_last matching **col** while decode0 matched **row**. In B3.34 we applied a minimal fix: force prefill Q to run GEMV (row-major) via `GRETA_PREFILL_FORCE_WQ_ROW=1` / `GRETA_PREFILL_QKV_LAYOUT=row`. Result: for context prompts (p4_sys, p5_ba, p6_long) **both prefill_last and decode0 match row**, and divergence is no longer at `q` (q_mae=0.0). The first mismatch moves to `k`. This confirms the prefill GEMM path for Wq was the root cause.

## Cambio Implementado (ES)
- Kernel INT4 GEMM permite forzar GEMV aun con M>32 cuando `GRETA_PREFILL_FORCE_WQ_ROW=1`.
- En `GretaCompute::gemm`, si `op_label=attn_q_prefill` y `GRETA_PREFILL_FORCE_WQ_ROW=1` o `GRETA_PREFILL_QKV_LAYOUT=row`, se fuerza GEMV.

## Implemented Change (EN)
- INT4 GEMM kernel supports a forced GEMV path even when M>32 under `GRETA_PREFILL_FORCE_WQ_ROW=1`.
- In `GretaCompute::gemm`, if `op_label=attn_q_prefill` and `GRETA_PREFILL_FORCE_WQ_ROW=1` or `GRETA_PREFILL_QKV_LAYOUT=row`, GEMV is forced.

## Metodología / Method
- Runner: `tools/benchmarks/run_b3_34_mi300x.sh 129.212.184.200 2026-02-03`
- Flags:
  - `GRETA_TRACE_ATTN_L0_PIPE=1`
  - `GRETA_TRACE_ATTN_L0_NORM=1`
  - `GRETA_TRACE_QKV_W_VERIFY=1`
  - `GRETA_TRACE_STAGE_DEBUG_INPUT=1`
  - E1 only: `GRETA_PREFILL_FORCE_WQ_ROW=1` (and `GRETA_PREFILL_QKV_LAYOUT=row`)
- Prompts:
  - p0_short, p4_sys, p5_ba, p6_long
- Artifacts:
  - `artifacts_remote/2026-02-03/b3_34/gretacore_b3_34_artifacts.tgz`
  - Analysis: `artifacts_remote/2026-02-03/b3_34/b3_34_analysis.txt`

## Resultados (ES)
Tabla resumen (extracto):

| Prompt | Exp | prefill_layout | decode_layout | first_mismatch | q_mae |
|---|---|---|---|---|---:|
| p4_sys | E0 | col | row | q | 0.5375 |
| p4_sys | E1 | row | row | k | 0.0 |
| p5_ba | E0 | col | row | q | 0.8889 |
| p5_ba | E1 | row | row | k | 0.0 |
| p6_long | E0 | col | row | q | 0.5396 |
| p6_long | E1 | row | row | k | 0.0 |

**Resultado clave:** prefill Wq ahora coincide con decode0; el primer mismatch se mueve a `k`.

## Results (EN)
Summary table (excerpt):

| Prompt | Exp | prefill_layout | decode_layout | first_mismatch | q_mae |
|---|---|---|---|---|---:|
| p4_sys | E0 | col | row | q | 0.5375 |
| p4_sys | E1 | row | row | k | 0.0 |
| p5_ba | E0 | col | row | q | 0.8889 |
| p5_ba | E1 | row | row | k | 0.0 |
| p6_long | E0 | col | row | q | 0.5396 |
| p6_long | E1 | row | row | k | 0.0 |

**Key outcome:** prefill Wq now matches decode0; first mismatch moves to `k`.

## Conclusión / Conclusion
- **ES:** El bug de layout en prefill Q quedó corregido con GEMV forzado. El siguiente foco es **Wk en prefill**, que ahora es el primer punto de divergencia.
- **EN:** The prefill Q layout bug is fixed with forced GEMV. The next focus is **Wk in prefill**, now the first divergence point.

## Próximo Paso (B3.35)
- Extender el mismo fix a Wk/Wv en prefill o corregir el layout/stride en GEMM para K.
- Re-ejecutar B3.30 para confirmar que el primer mismatch ya no ocurre en Q ni K.

---

L.E.T / Leandro Emanuel Timberini
