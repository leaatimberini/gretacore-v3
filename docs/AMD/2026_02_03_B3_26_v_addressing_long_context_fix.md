# B3.26 — V Addressing (Long Context) Fix + Validation

## Resumen Ejecutivo (ES)
Se instrumentó `GRETA_TRACE_V_ADDR` para validar direccionamiento de V/K en decode0 (layer 31, head 0) y se ajustó el cálculo de referencia de P·V a secuencia completa (scope=full). El trazado confirma `kv_pos` correcto (match con el token actual) y layout row-major. El aumento de `pv_mae` en p4 era un artefacto de ventana parcial; al usar referencia full-seq, p4 queda en rango ≤1e-7. **El colapso decode0→96965 persiste** (prefill_last_top1=127158, decode0_top1=96965), por lo que el problema principal no es V addressing sino dinámica de decode posterior.

## Executive Summary (EN)
We added `GRETA_TRACE_V_ADDR` to validate V/K addressing in decode0 (layer 31, head 0) and corrected the P·V reference to full-sequence scope (scope=full). The trace confirms correct `kv_pos` and row-major layout. The elevated p4 `pv_mae` was a windowing artifact; with full-seq reference, p4 is within ≤1e-7. **Decode collapse to 96965 persists** (prefill_last_top1=127158, decode0_top1=96965), so the root cause is not V addressing but decode dynamics downstream.

## Metodología / Method
- Commit: `987613f3aa469c72f5190144b20c9c99b1884981`
- Modelo: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
- Flags:
  - `GRETA_INT4_WEIGHTS=1`
  - `GRETA_MAX_SEQ_LEN=256`
  - `GRETA_TRACE_ATTN_VACC=1`
  - `GRETA_TRACE_ATTN_LAYER=31`
  - `GRETA_TRACE_ATTN_HEAD=0`
  - `GRETA_TRACE_ATTN_KEYS_WINDOW=64`
  - `GRETA_TRACE_ATTN_DIMS_SAMPLE=16`
  - `GRETA_TRACE_ATTN_OUT=.../b3_26_attn_vacc.jsonl`
  - `GRETA_TRACE_V_ADDR=1`
  - `GRETA_TRACE_V_ADDR_OUT=.../b3_26_v_addr.jsonl`
  - `GRETA_TRACE_PREFILL_DECODE_DELTA=1`
- Prompts:
  - `p4_sys`: system+user template
  - `p5_ba`: “Write one short sentence about Buenos Aires.”

## Resultados (ES)
### Diagnóstico V-addr (decode0, layer31, head0)
- `mae_v_pos = 0` (match exacto con cache en `pos_id`)
- `mae_v_prev/next` altos (≈0.078–0.083)
- `mae_v_col` alto (≈0.098–0.106)
**Conclusión:** `kv_pos` correcto y layout row-major confirmado.

### Comparativa (B3.24/B3.25 vs B3.26)
| Prompt | v_layout_best | pv_scope | pv_mae (antes) | pv_mae (B3.26) | attn_out_mae (B3.26) | prefill_last_top1 | decode0_top1 |
|---|---|---|---:|---:|---:|---:|---:|
| p4_sys | row | full | 2.45e-03 (window) | 1.19e-07 | 1.19e-07 | 127158 | 96965 |
| p5_ba  | row | full | 6.19e-08 | 6.19e-08 | 6.19e-08 | 127158 | 96965 |

### Evidencia (paths)
- `artifacts_remote/2026-02-03/b3_26/root/gretacore/artifacts/alignment/2026-02-03/b3_26_attn_vacc.jsonl`
- `artifacts_remote/2026-02-03/b3_26/root/gretacore/artifacts/alignment/2026-02-03/b3_26_v_addr.jsonl`
- `artifacts_remote/2026-02-03/b3_26/b3_26_analysis.txt`

## Results (EN)
### V-addr diagnosis (decode0, layer31, head0)
- `mae_v_pos = 0` (exact match at `pos_id`)
- `mae_v_prev/next` high (≈0.078–0.083)
- `mae_v_col` high (≈0.098–0.106)
**Conclusion:** `kv_pos` is correct and row-major layout confirmed.

### Comparison (B3.24/B3.25 vs B3.26)
| Prompt | v_layout_best | pv_scope | pv_mae (before) | pv_mae (B3.26) | attn_out_mae (B3.26) | prefill_last_top1 | decode0_top1 |
|---|---|---|---:|---:|---:|---:|---:|
| p4_sys | row | full | 2.45e-03 (window) | 1.19e-07 | 1.19e-07 | 127158 | 96965 |
| p5_ba  | row | full | 6.19e-08 | 6.19e-08 | 6.19e-08 | 127158 | 96965 |

### Evidence (paths)
- `artifacts_remote/2026-02-03/b3_26/root/gretacore/artifacts/alignment/2026-02-03/b3_26_attn_vacc.jsonl`
- `artifacts_remote/2026-02-03/b3_26/root/gretacore/artifacts/alignment/2026-02-03/b3_26_v_addr.jsonl`
- `artifacts_remote/2026-02-03/b3_26/b3_26_analysis.txt`

## Conclusión / Conclusion
- **V addressing correcto** (row-major, `kv_pos` OK).
- El incremento de `pv_mae` en p4 era un artefacto de ventana parcial; la referencia full-seq elimina el sesgo.
- **El colapso decode0→96965 persiste**; el root-cause está fuera de V addressing (siguiente foco: dinámica de decode / post-attention).

## Próximo paso / Next Step (B3.27)
Instrumentar y aislar dinámica de decode posterior a attention (estado residual / MLP / lm head en decode) con métricas prefill_last vs decode0, manteniendo trazas full-seq cuando corresponda.

---

L.E.T / Leandro Emanuel Timberini
