# B3.10 Attractor Validation (Top1 Multi-Step + Prefill↔Decode)

Fecha: 2026-02-03
Bloque: B3.10

## Resumen ejecutivo (ES)
Se ejecutó un set de prompts controlados con trazas de landscape y prefill/decode para verificar si el attractor (top1=96965) persiste en steps 0..15 y si hay coherencia entre `prefill_last_top1` y `decode0_top1`. Con embedding corregido (`row_major_match`), los prompts cortos siguen fijando top1=96965 en steps 0..15. En prompts largos (sys / Buenos Aires) el step 0 cambia (top1=79), pero `decode0_top1` vuelve a 96965 en el step 1. Esto indica **bug residual en dinámica de decode** (no un attractor “normal” de embedding).

## Executive summary (EN)
We ran a controlled prompt matrix with landscape and prefill/decode traces to determine whether the attractor (top1=96965) persists across steps 0..15 and whether `prefill_last_top1` matches `decode0_top1`. With embedding fixed (`row_major_match`), short prompts still lock top1=96965 across steps 0..15. For longer prompts (system / Buenos Aires), step 0 changes (top1=79), but `decode0_top1` returns to 96965 at step 1. This indicates a **residual decode dynamics bug**, not a normal embedding attractor.

## Configuración de run (MI300X)
```
export GRETA_INT4_WEIGHTS=1
export GRETA_MAX_SEQ_LEN=256
export GRETA_TRACE_EMBED_VERIFY=1
export GRETA_EMBED_LAYOUT=row
export GRETA_TRACE_LOGITS=1
export GRETA_TRACE_LANDSCAPE=1
export GRETA_TRACE_PREFILL_DECODE=1
```
Prompts: p0_hi, p1_hi_space, p2_hi_nl, p3_q, p4_sys, p5_ba.

## Evidencia del probe
```
[GRETA_TRACE_EMBED_VERIFY] token=105 seq_idx=2 mae_row=0 mae_col=0.0129823 max_row=0 max_col=0.063498 layout_used=row layout_probe_best=row_major_match
```

## Tabla A — Top1 por step (0..3) + uniq_top1 (0..15)
| Prompt | uniq_top1_0_15 | top1_step0 | top1_step1 | top1_step2 | top1_step3 |
|---|---|---|---|---|---|
| p0_hi | 1 | 96965 | 96965 | 96965 | 96965 |
| p1_hi_space | 1 | 96965 | 96965 | 96965 | 96965 |
| p2_hi_nl | 1 | 96965 | 96965 | 96965 | 96965 |
| p3_q | 1 | 96965 | 96965 | 96965 | 96965 |
| p4_sys | 2 | 79 | 96965 | 96965 | 96965 |
| p5_ba | 2 | 79 | 96965 | 96965 | 96965 |

## Tabla B — Prefill vs Decode (top1)
| Prompt | prefill_last_step | prefill_last_top1 | decode0_step | decode0_top1 |
|---|---|---|---|---|
| p0_hi | 0 | 96965 | 1 | 96965 |
| p1_hi_space | 0 | 96965 | 1 | 96965 |
| p2_hi_nl | 0 | 96965 | 1 | 96965 |
| p3_q | 0 | 96965 | 1 | 96965 |
| p4_sys | 0 | 79 | 1 | 96965 |
| p5_ba | 0 | 79 | 1 | 96965 |

## Conclusión
- **Decisión:** BUG residual (NO OK).
- Justificación: `prefill_last_top1` cambia para prompts largos, pero `decode0_top1` cae de inmediato a 96965 y se mantiene en steps siguientes. Esto es consistente con un problema en la dinámica de decode o en el head final (RMSNorm / LM head).

## Próximo paso (B3.11)
Investigar RMSNorm final y LM head (layout / escala / dequant) y repetir una corrida con trazas multi-step para verificar que `decode0_top1` siga el prefill en prompts largos.

L.E.T / Leandro Emanuel Timberini
