# B3.25 — V Layout Fix Acceptance (decode0)

## Resumen Ejecutivo (ES)
Objetivo: corregir el layout/stride de V en decode para que **v_layout_best=row** y reducir el error en P*V/attn_out, con impacto directo en la coherencia prefill_last→decode0. Se corrigió el uso de KV cache por layer en decode (se pasa el slice correcto al kernel), y se re-ejecutó B3.25 con trazas `GRETA_TRACE_ATTN_VACC`.

Resultado: **v_layout_best pasó a row** en p4_sys y p5_ba. `pv_mae` se redujo drásticamente en p5_ba (~6.19e-08), pero en p4_sys sigue en ~2.45e-03. La coherencia **prefill_last_top1 == decode0_top1 aún no se restablece** (prefill_last=127158, decode0=96965). Esto indica que el layout de V por layer estaba mal, pero aún queda un problema de dinámica/estado en decode0 (posiblemente sensibilidad de prompt corto o atención residual en capas altas).

## Executive Summary (EN)
Goal: fix V layout/stride in decode so **v_layout_best=row** and reduce P*V/attn_out error, restoring prefill_last→decode0 coherence. We corrected layer-sliced KV usage in decode and re-ran B3.25 with `GRETA_TRACE_ATTN_VACC`.

Result: **v_layout_best is now row** for p4_sys and p5_ba. `pv_mae` dropped sharply in p5_ba (~6.19e-08), but remains ~2.45e-03 in p4_sys. **prefill_last_top1 == decode0_top1 is still not restored** (prefill_last=127158, decode0=96965). This indicates the per-layer V layout issue is fixed, but decode0 still collapses for short prompts; further investigation is required.

## Metodología / Methodology
Commit: `616330d8db0cf8a4427ee82e9b7598aa194ec166`

Flags:
- `GRETA_TRACE_ATTN_VACC=1`
- `GRETA_TRACE_ATTN_LAYER=31`
- `GRETA_TRACE_ATTN_HEAD=0`
- `GRETA_TRACE_ATTN_KEYS_WINDOW=64`
- `GRETA_TRACE_ATTN_DIMS_SAMPLE=16`
- `GRETA_TRACE_ATTN_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_25_attn_vacc.jsonl`
- `GRETA_TRACE_PREFILL_DECODE_DELTA=1`
- `GRETA_TRACE_PREFILL_DECODE_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_25_p{4,5}_prefill_decode.jsonl`

Prompts:
- `p4_sys`
- `p5_ba`

Artefactos (local):
- `artifacts_remote/2026-02-03/b3_25/root/gretacore/artifacts/alignment/2026-02-03/b3_25_attn_vacc.jsonl`
- `artifacts_remote/2026-02-03/b3_25/root/gretacore/artifacts/alignment/2026-02-03/b3_25_p4_prefill_decode.jsonl`
- `artifacts_remote/2026-02-03/b3_25/root/gretacore/artifacts/alignment/2026-02-03/b3_25_p5_prefill_decode.jsonl`
- `artifacts_remote/2026-02-03/b3_25/b3_25_analysis.txt`

## Resultados / Results

### B3.24 (baseline) vs B3.25
| Prompt | v_layout_best (B3.24) | v_layout_best (B3.25) | pv_mae (B3.24) | pv_mae (B3.25) | attn_out_mae (B3.25) | prefill_last_top1 | decode0_top1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| p4_sys | col | row | 0.0375317 | 0.00245193 | 0.00245193 | 127158 | 96965 |
| p5_ba  | col | row | 0.0480429 | 6.18786e-08 | 6.18786e-08 | 127158 | 96965 |

Interpretación: el layout de V por layer quedó corregido (row). Sin embargo, **decode0 aún colapsa**; el error residual en p4_sys sugiere sensibilidad a prompt corto o un problema adicional en capas altas (posiblemente combinación de heads o path de atención posterior).

## Conclusión / Conclusion
El fix corrige el layout/stride de V en decode (v_layout_best=row) y reduce fuertemente el error P*V en prompts largos, pero **no elimina el colapso decode0**. Se requiere un siguiente paso para aislar por qué p4_sys aún muestra error (posible estado/normalización en decode0 o sensibilidad al prompt corto).

## Próximo Paso / Next Step (B3.26)
Auditar el path de decode0 en prompts cortos (p4_sys) con foco en:
1) normalización residual y combine de heads post‑attn,
2) sensibilidad del kernel a window_len/seq_len pequeño,
3) verificación por head adicional (head 1/2) para descartar anomalía específica de head 0.

---
L.E.T / Leandro Emanuel Timberini
