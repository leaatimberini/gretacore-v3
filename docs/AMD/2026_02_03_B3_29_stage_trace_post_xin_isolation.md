# B3.29 — StageTrace Post‑x_in Isolation (prefill_last vs decode0)

## Resumen Ejecutivo (ES)
B3.28 logró equivalencia exacta de `x_in` entre `prefill_last` y `decode0` (MAE=0), pero el colapso en logits persistió. En B3.29 se activó StageTrace completo (puntos post‑x_in) y se identificó el **primer punto de divergencia**: `attn_out` en la **layer 0** (MAE ~3.36e‑05 para p4_sys y ~9.76e‑05 para p5_ba). Esto confirma que la divergencia nace en la **atención decode** inmediatamente después de `x_in`, no en MLP/final RMS/LM head.

## Executive Summary (EN)
B3.28 achieved exact `x_in` equivalence between `prefill_last` and `decode0` (MAE=0), but logits collapse persisted. In B3.29 we enabled full StageTrace (post‑x_in points) and identified the **first divergence** at `attn_out` in **layer 0** (MAE ~3.36e‑05 for p4_sys and ~9.76e‑05 for p5_ba). This confirms the divergence originates in **decode attention** immediately after `x_in`, not in MLP/final RMS/LM head.

## Metodología / Method
Flags:
- `GRETA_TRACE_STAGE=1`
- `GRETA_TRACE_STAGE_DEBUG_INPUT=1`
- `GRETA_TRACE_STAGE_LAYERS="0,1,2,15,31"`
- `GRETA_TRACE_STAGE_POINTS="x_in,attn_out,x_after_attn,mlp_out,x_out,final_rms,lm_head_in,logits"`
- `GRETA_TRACE_STAGE_PHASES="prefill_last,decode0"`

Prompts:
- `p4_sys` (system+user template)
- `p5_ba` (Buenos Aires)

Artefactos:
- JSONL: `artifacts_remote/2026-02-03/b3_29/root/gretacore/artifacts/alignment/2026-02-03/b3_29_stage.jsonl`
- Logs: `artifacts_remote/2026-02-03/b3_29/root/gretacore/artifacts/alignment/2026-02-03/b3_29_p4.log`, `b3_29_p5.log`
- Analysis: `artifacts_remote/2026-02-03/b3_29/b3_29_analysis.txt`

## Resultados (ES)
Primer punto de divergencia:

| Prompt | Layer | Point | MAE |
|---|---:|---|---:|
| p4_sys | 0 | attn_out | 3.36048e‑05 |
| p5_ba  | 0 | attn_out | 9.75960e‑05 |

Resumen: `x_in` coincide (MAE=0), pero `attn_out` ya diverge en layer 0. Esto explica el colapso posterior en logits.

## Results (EN)
First divergence point:

| Prompt | Layer | Point | MAE |
|---|---:|---|---:|
| p4_sys | 0 | attn_out | 3.36048e‑05 |
| p5_ba  | 0 | attn_out | 9.75960e‑05 |

Summary: `x_in` matches (MAE=0), but `attn_out` already diverges at layer 0. This explains the later logits collapse.

## Conclusión (ES)
La discrepancia aparece **inmediatamente después de x_in** en `attn_out` de layer 0. El foco debe pasar a la **ruta de atención decode** (kernel/path, uso de KV/cache, o diferencia prefill vs decode en atención). Se propone B3.30 para aislar Q/K/V y la ruta de atención en layer 0 con referencia controlada.

## Conclusion (EN)
The discrepancy appears **immediately after x_in** at layer‑0 `attn_out`. Focus must move to the **decode attention path** (kernel/path, KV/cache usage, or prefill vs decode attention differences). B3.30 should isolate Q/K/V and attention routing at layer 0 with a controlled reference.

---

L.E.T / Leandro Emanuel Timberini
