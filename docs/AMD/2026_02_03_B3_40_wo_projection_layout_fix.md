# B3.40 — WO Projection Layout Fix (Prefill vs Decode)

## Resumen Ejecutivo (ES)
B3.39 aisló el primer mismatch en `wo_out` (salida de WO), con `attn_out` y `x_in` iguales. En B3.40 añadimos un verificador de layout para WO y un control de layout (`GRETA_WO_LAYOUT_FORCE`) y confirmamos que el layout correcto es **row**. Al alinear la interpretación de WO en prefill con decode, `wo_out_mae` cae a ~1e-10 y el mismatch en `x_after_attn` desaparece en E1 (aligned). El colapso de decode0 a 96965 persiste, por lo que el siguiente paso debe aislar la etapa posterior a WO (B3.41).

## Executive Summary (EN)
B3.39 isolated the first mismatch at `wo_out` (WO projection output), while `attn_out` and `x_in` matched. In B3.40 we added a WO layout verifier and a layout override (`GRETA_WO_LAYOUT_FORCE`) and confirmed the correct layout is **row**. Aligning prefill WO semantics with decode drops `wo_out_mae` to ~1e-10 and eliminates the `x_after_attn` mismatch in E1 (aligned). The decode0 collapse to 96965 persists, so the next step must isolate the stage after WO (B3.41).

## Cambios Implementados (ES)
- Verificador WO (`GRETA_TRACE_WO_W_VERIFY=1`) con comparación row vs col y métricas MAE.
- Control de layout WO (`GRETA_WO_LAYOUT_FORCE=row|col|auto`).
- Prefill WO respeta `GRETA_PREFILL_QKV_LAYOUT=row` para alinear semántica con decode.

## Implemented Changes (EN)
- WO verifier (`GRETA_TRACE_WO_W_VERIFY=1`) with row vs col MAE comparison.
- WO layout control (`GRETA_WO_LAYOUT_FORCE=row|col|auto`).
- Prefill WO honors `GRETA_PREFILL_QKV_LAYOUT=row` to align with decode semantics.

## Metodología / Method
- Runner: `tools/benchmarks/run_b3_40_mi300x.sh`
- Prompts: `p0_short`, `p4_sys`, `p6_long`
- Modo: E1 (aligned Q/K/V) + matriz WO:
  - E1W0 = auto
  - E1W1 = row
  - E1W2 = col
- JSONL: `/root/gretacore/artifacts/alignment/2026-02-03/b3_40_wo_resid_<prompt>_<exp>.jsonl`
- Análisis: `tools/benchmarks/analyze_b3_40_wo_fix.py`

## Resultados (ES)
Tabla resumida (p4_sys y p6_long):

| Prompt | Exp | wo_out_mae | x_after_attn_mae | wo_layout_best | wo_layout_used | prefill_top1 | decode0_top1 |
|---|---|---:|---:|---|---|---:|---:|
| p4_sys | E1W0 | 2.31e-10 | 1.85e-10 | row|row | auto | 93590 | 96965 |
| p4_sys | E1W1 | 2.31e-10 | 1.85e-10 | row|row | row  | 93590 | 96965 |
| p4_sys | E1W2 | 7.40e-04 | 7.40e-04 | col|row | col  | 93590 | 96965 |
| p6_long| E1W0 | 3.07e-10 | 2.27e-10 | row|row | auto | 99668 | 96965 |
| p6_long| E1W1 | 3.07e-10 | 2.27e-10 | row|row | row  | 99668 | 96965 |
| p6_long| E1W2 | 1.35e-03 | 1.35e-03 | col|row | col  | 52196 | 96965 |

**Conclusión:** WO correcto = **row**. Forzar col reintroduce el mismatch. Con WO en row, el mismatch en `x_after_attn` desaparece.

## Results (EN)
Summary table (p4_sys and p6_long):

| Prompt | Exp | wo_out_mae | x_after_attn_mae | wo_layout_best | wo_layout_used | prefill_top1 | decode0_top1 |
|---|---|---:|---:|---|---|---:|---:|
| p4_sys | E1W0 | 2.31e-10 | 1.85e-10 | row|row | auto | 93590 | 96965 |
| p4_sys | E1W1 | 2.31e-10 | 1.85e-10 | row|row | row  | 93590 | 96965 |
| p4_sys | E1W2 | 7.40e-04 | 7.40e-04 | col|row | col  | 93590 | 96965 |
| p6_long| E1W0 | 3.07e-10 | 2.27e-10 | row|row | auto | 99668 | 96965 |
| p6_long| E1W1 | 3.07e-10 | 2.27e-10 | row|row | row  | 99668 | 96965 |
| p6_long| E1W2 | 1.35e-03 | 1.35e-03 | col|row | col  | 52196 | 96965 |

**Conclusion:** WO correct layout is **row**. Forcing col reintroduces the mismatch. With WO in row, `x_after_attn` mismatch disappears.

## Próximo Paso (B3.41)
Aislar el siguiente punto de divergencia post-WO (probablemente en FFN o en logits). Mantener E1 aligned y repetir stage trace para ubicar el primer mismatch.

## Next Step (B3.41)
Isolate the next divergence point after WO (likely FFN or logits). Keep E1 aligned and rerun stage trace to locate the first mismatch.

---

L.E.T / Leandro Emanuel Timberini
