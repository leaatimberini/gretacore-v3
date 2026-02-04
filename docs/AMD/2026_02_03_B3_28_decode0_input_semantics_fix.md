# B3.28 — Decode0 Input Semantics Fix (x_in Equivalence)

## Resumen Ejecutivo (ES)
B3.27 mostró que la primera divergencia prefill_last↔decode0 aparece en `x_in` (layer 0). En B3.28 se agregó un trazo de semántica de entrada y un override controlado para que decode0 use el token del último prefill (y el mismo `seq_start`), logrando **x_in idéntico** entre prefill_last y decode0 (MAE=0.0). Aun así, `prefill_last_top1 != decode0_top1` persiste (colapso a 96965), por lo que la divergencia ocurre **después** de `x_in` y requiere trazas de etapas posteriores (B3.29).

## Executive Summary (EN)
B3.27 showed the first prefill_last↔decode0 divergence at `x_in` (layer 0). In B3.28 we added input-semantics tracing and a controlled override so decode0 uses the last prefill token (and matching `seq_start`), yielding **identical x_in** (MAE=0.0). However, `prefill_last_top1 != decode0_top1` persists (collapse to 96965), so the divergence occurs **after** `x_in` and requires downstream stage tracing (B3.29).

## Cambio Implementado (ES)
- `GRETA_TRACE_STAGE_DEBUG_INPUT=1` agrega campos de semántica de entrada en JSONL:
  `x_in_src_kind`, `x_in_token_index_used`, `x_in_offset_bytes`, `x_in_ptr`, `x_in_alloc_bytes`, `prompt_tokens`, `kv_pos`, `decode_step`.
- Override controlado en decode0 (solo cuando `GRETA_TRACE_STAGE_DEBUG_INPUT=1`):
  usa el token del último prefill y `seq_start = prompt_len-1` para alinear `x_in`.

## Implemented Change (EN)
- `GRETA_TRACE_STAGE_DEBUG_INPUT=1` adds input-semantics fields in JSONL:
  `x_in_src_kind`, `x_in_token_index_used`, `x_in_offset_bytes`, `x_in_ptr`, `x_in_alloc_bytes`, `prompt_tokens`, `kv_pos`, `decode_step`.
- Controlled decode0 override (only with `GRETA_TRACE_STAGE_DEBUG_INPUT=1`):
  uses the last prefill token and `seq_start = prompt_len-1` to align `x_in`.

## Metodología / Method
Flags:
- `GRETA_TRACE_STAGE=1`
- `GRETA_TRACE_STAGE_DEBUG_INPUT=1`
- `GRETA_TRACE_STAGE_LAYERS="0"`
- `GRETA_TRACE_STAGE_POINTS="x_in,logits"`
- `GRETA_TRACE_STAGE_PHASES="prefill_last,decode0"`

Prompts:
- `p4_sys` (system + user template)
- `p5_ba` (Buenos Aires)

## Resultados (ES)
**x_in equivalencia (layer 0):**

| Prompt | x_in MAE (prefill_last vs decode0) |
|---|---:|
| p4_sys | 0.0 |
| p5_ba  | 0.0 |

**Logits (top1) aún divergentes:**

| Prompt | prefill_last top1 | decode0 top1 | gap prefill | gap decode0 |
|---|---:|---:|---:|---:|
| p4_sys | 127158 | 96965 | 0.95866 | 1.42329 |
| p5_ba  | 127158 | 96965 | 0.95619 | 1.43258 |

Conclusión: la equivalencia de `x_in` está lograda, pero el colapso persiste; la divergencia ocurre **después de `x_in`**.

## Results (EN)
**x_in equivalence (layer 0):**

| Prompt | x_in MAE (prefill_last vs decode0) |
|---|---:|
| p4_sys | 0.0 |
| p5_ba  | 0.0 |

**Logits top1 still diverge:**

| Prompt | prefill_last top1 | decode0 top1 | gap prefill | gap decode0 |
|---|---:|---:|---:|---:|
| p4_sys | 127158 | 96965 | 0.95866 | 1.42329 |
| p5_ba  | 127158 | 96965 | 0.95619 | 1.43258 |

Conclusion: `x_in` equivalence is achieved, but collapse persists; divergence occurs **after `x_in`**.

## Evidencia / Evidence
- JSONL: `artifacts_remote/2026-02-03/b3_28/root/gretacore/artifacts/alignment/2026-02-03/b3_28_stage.jsonl`
- Logs: `artifacts_remote/2026-02-03/b3_28/root/gretacore/artifacts/alignment/2026-02-03/b3_28_p4.log` / `b3_28_p5.log`
- Análisis: `artifacts_remote/2026-02-03/b3_28/b3_28_analysis.txt`

## Próximo paso / Next step (B3.29)
Rehabilitar StageTrace en puntos posteriores (`attn_out`, `x_after_attn`, `mlp_out`, `x_after_mlp`, `final_norm`, `lm_head_in`) **bajo el override de input** para ubicar la primera etapa donde diverge decode0.

---

L.E.T / Leandro Emanuel Timberini
