# B3.23 — Softmax Isolation (decode0, layer 31)

## Resumen Ejecutivo (ES)
Objetivo: aislar si la divergencia en atención decode (layer 31) aparece **antes** del softmax (QK) o **dentro** del softmax/escala. Se implementó un trace específico de decode0 para layer 31, head 0, con ventana de keys y comparación GPU vs referencia CPU FP64. Resultado: **QK y softmax coinciden** (MAE ~1e-6 y ~1e-8 respectivamente). La divergencia observada en B3.22 no se origina en QK ni en softmax para la ventana trazada; el foco pasa al **acumulado con V / salida de atención**.

## Executive Summary (EN)
Goal: isolate whether decode attention divergence (layer 31) originates **before** softmax (QK) or **inside** softmax/scale. Implemented a decode0 trace for layer 31, head 0, with key window and GPU vs CPU FP64 comparison. Result: **QK and softmax match** (MAE ~1e-6 and ~1e-8 respectively). The B3.22 divergence is not caused by QK or softmax (for the traced window); focus shifts to **V accumulation / attention output path**.

## Contexto (ES)
B3.22 mostró divergencia vs referencia FP64 en `attn_out` de layer 31 (MAE ~0.042–0.044), independiente de FP16/FP32 accumulation. B3.23 valida si el origen está en QK/softmax.

## Context (EN)
B3.22 showed FP64 reference divergence in layer‑31 `attn_out` (MAE ~0.042–0.044), independent of FP16/FP32 accumulation. B3.23 validates whether the origin lies in QK/softmax.

## Metodología / Methodology
Commit: `123ad06e1357c81faa3854d6e6129be935e3c3d5`

Flags:
- `GRETA_TRACE_ATTN_SOFTMAX=1`
- `GRETA_TRACE_ATTN_LAYER=31`
- `GRETA_TRACE_ATTN_HEAD=0`
- `GRETA_TRACE_ATTN_KEYS_WINDOW=64`
- `GRETA_TRACE_ATTN_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_23_attn_softmax.jsonl`
- `GRETA_TRACE_PROMPT_ID=p4_sys|p5_ba`

Prompts:
- `p4_sys`
- `p5_ba`

Artefactos (local):
- `artifacts_remote/2026-02-03/b3_23/root/gretacore/artifacts/alignment/2026-02-03/b3_23_attn_softmax.jsonl`
- `artifacts_remote/2026-02-03/b3_23/b3_23_analysis.txt`

## Resultados (ES)
### QK y Softmax (GPU vs CPU FP64)
| Prompt | qk_mae | qk_max_diff | softmax_mae | softmax_max_diff | Veredicto |
| --- | ---: | ---: | ---: | ---: | --- |
| p4_sys | 1.45017e-06 | 5.39595e-06 | 3.02422e-08 | 1.02311e-06 | mixed/low |
| p5_ba  | 1.73674e-06 | 4.48005e-06 | 3.33810e-08 | 6.85972e-07 | mixed/low |

Interpretación: QK y softmax están alineados entre GPU y referencia FP64 en la ventana trazada. No hay señal de bug en QK ni en softmax.

## Results (EN)
### QK and Softmax (GPU vs CPU FP64)
| Prompt | qk_mae | qk_max_diff | softmax_mae | softmax_max_diff | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| p4_sys | 1.45017e-06 | 5.39595e-06 | 3.02422e-08 | 1.02311e-06 | mixed/low |
| p5_ba  | 1.73674e-06 | 4.48005e-06 | 3.33810e-08 | 6.85972e-07 | mixed/low |

Interpretation: QK and softmax are aligned between GPU and FP64 reference for the traced window. No evidence of a QK or softmax bug.

## Conclusión / Conclusion
El origen del colapso decode **no** está en QK ni en softmax. El siguiente paso (B3.24) debe aislar el **acumulado con V / attn_out** (layout, stride, orden numérico o path fused), porque allí es donde aún se observa la divergencia.

## Próximo Paso / Next Step (B3.24)
Implementar trazas específicas del acumulado con V (pre/post scale) y comparar `attn_out` GPU vs referencia FP64 para la misma ventana y head. Verificar si la discrepancia aparece en la suma ponderada de V o en el path fused.

---
L.E.T / Leandro Emanuel Timberini
