# B3.22 — Attention Precision Audit (Layer 31)

## Resumen Ejecutivo (ES)
Objetivo: aislar si el colapso `decode0→96965` proviene de precisión/acumulación en atención de capas altas (layer 31). Se agregó referencia CPU de alta precisión (FP64) para decode0 y un toggle de acumulación `GRETA_ATTN_ACCUM=fp16|fp32`. Resultado: la divergencia vs referencia **persiste** (MAE≈0.042–0.044 en layer 31) y el colapso `prefill_last_top1 != decode0_top1` no se corrige. La sensibilidad a acumulación FP16 es baja frente al error observado, por lo que la causa probable es **softmax/escala/orden numérico** o un comportamiento no capturado por la referencia actual.

## Executive Summary (EN)
Goal: isolate whether `decode0→96965` collapse comes from attention precision/accumulation in high layers (layer 31). Added a high‑precision CPU reference (FP64) for decode0 and an accumulation toggle `GRETA_ATTN_ACCUM=fp16|fp32`. Result: divergence vs reference **persists** (MAE≈0.042–0.044 at layer 31) and `prefill_last_top1 != decode0_top1` remains. FP16 accumulation sensitivity is small relative to the observed error, suggesting **softmax/scale/order effects** or a mismatch not captured by the current reference path.

## Por qué MI300X expone este problema (ES)
En MI300X, el rango dinámico de Q/K/V en capas altas tiende a amplificar errores numéricos en atención (softmax/escala/acumulación). Esto hace visibles divergencias que en entornos CPU/GPU menores pueden pasar desapercibidas. La instrumentación determinista confirma que el problema no es KV ni LM head sino precisión/orden de cómputo en decode.

## Why MI300X exposes this issue (EN)
On MI300X, the dynamic range of Q/K/V in high layers amplifies numeric errors in attention (softmax/scale/accumulation), making divergences visible that might be hidden on smaller CPU/GPU setups. Deterministic traces confirm the issue is not KV or LM head but decode attention precision/order of computation.

## Metodología / Methodology
Commit: `cccf53cac3b32675e42ff34ec32bde7c1bc7d0c4`

Flags principales:
- `GRETA_TRACE_ATTN_REF=1`
- `GRETA_TRACE_ATTN_REF_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_22_attn_precision.jsonl`
- `GRETA_TRACE_ATTN_LAYERS="31"`
- `GRETA_TRACE_ATTN_POINTS="q,k,v,attn_out"`
- `GRETA_ATTN_ACCUM=fp32|fp16`
- `GRETA_TRACE_PREFILL_DECODE=1`

Prompts:
- `p4_sys` (system+user template)
- `p5_ba` (Buenos Aires)

Artefactos (local):
- `artifacts_remote/2026-02-03/b3_22/root/gretacore/artifacts/alignment/2026-02-03/b3_22_attn_precision.jsonl`
- `artifacts_remote/2026-02-03/b3_22/root/gretacore/artifacts/alignment/2026-02-03/b3_22_p4_prefill_decode.jsonl`
- `artifacts_remote/2026-02-03/b3_22/root/gretacore/artifacts/alignment/2026-02-03/b3_22_p5_prefill_decode.jsonl`

## Resultados (ES)
### Prefill vs Decode (top1)
| Prompt | prefill_last_top1 | decode0_top1 | gap prefill | gap decode |
| --- | ---: | ---: | ---: | ---: |
| p4_sys | 127158 | 96965 | 0.95866 | 1.34019 |
| p5_ba  | 127158 | 96965 | 0.95619 | 1.34019 |

### Precisión vs referencia (layer 31)
**FP32 acumulación (GPU vs ref FP64):**
- p4_sys: `attn_ref_mae=0.0419899`, `attn_ref_max_diff=0.286283`
- p5_ba: `attn_ref_mae=0.0438694`, `attn_ref_max_diff=0.28688`

**FP16 acumulación (ref FP16 vs ref FP64):**
- p4_sys: `attn_accum_mae=2.45971e-04`, `attn_accum_max_diff=2.45022e-03`
- p5_ba: `attn_accum_mae=2.94730e-04`, `attn_accum_max_diff=3.67962e-03`

Interpretación: la divergencia GPU→ref es mucho mayor que el error inducido por FP16 en la referencia, por lo que la causa principal no es la acumulación FP16.

## Results (EN)
### Prefill vs Decode (top1)
| Prompt | prefill_last_top1 | decode0_top1 | gap prefill | gap decode |
| --- | ---: | ---: | ---: | ---: |
| p4_sys | 127158 | 96965 | 0.95866 | 1.34019 |
| p5_ba  | 127158 | 96965 | 0.95619 | 1.34019 |

### Precision vs reference (layer 31)
**FP32 accumulation (GPU vs FP64 ref):**
- p4_sys: `attn_ref_mae=0.0419899`, `attn_ref_max_diff=0.286283`
- p5_ba: `attn_ref_mae=0.0438694`, `attn_ref_max_diff=0.28688`

**FP16 accumulation (FP16 ref vs FP64 ref):**
- p4_sys: `attn_accum_mae=2.45971e-04`, `attn_accum_max_diff=2.45022e-03`
- p5_ba: `attn_accum_mae=2.94730e-04`, `attn_accum_max_diff=3.67962e-03`

Interpretation: GPU→ref divergence is much larger than the FP16 accumulation error in the reference, indicating accumulation precision is not the primary cause.

## Conclusión / Conclusion
La evidencia sugiere que el colapso decode **no** se explica por acumulación FP16/FP32, y la divergencia en layer 31 persiste incluso con referencia FP64. El próximo paso (B3.23) debe aislar si el problema proviene de **softmax/escala/orden de cómputo** o de un desajuste en la ruta de referencia vs kernel.

## Why this matters (ES)
Esto refuerza que GRETA puede medir con precisión las diferencias de atención en MI300X y separar errores de acumulación de problemas más profundos en softmax/orden numérico, acelerando el diagnóstico hacia una corrección estable de decode.

## Why this matters (EN)
This reinforces GRETA’s ability to precisely measure attention discrepancies on MI300X and separate accumulation errors from deeper softmax/order issues, accelerating diagnosis toward a stable decode fix.

---
L.E.T / Leandro Emanuel Timberini
