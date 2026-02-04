# B3.21 — Attention Decode MFMA Fix (fused path) + Shadow Validation

## Resumen Ejecutivo (ES)
Objetivo: estabilizar `fused+mfma` en decode y verificar si el colapso `decode0→96965` proviene del kernel de atención. Se corrigió el uso de `Hkv` en el kernel fused decode (GQA) y se agregaron guard rails de alignment. Se implementó un shadow compare MFMA vs VALU en decode0. Resultado: `fused+mfma` deja de fallar (sin `hipMemcpy H2D invalid argument`) y MFMA==VALU (MAE=0) en todas las capas trazadas, pero `attn_out` sigue divergiendo del ref en layer 31 y el colapso prefill→decode persiste. El foco pasa a **precisión/acumulación o referencia** en atención decode en capas altas.

## Executive Summary (EN)
Goal: stabilize `fused+mfma` in decode and verify whether the `decode0→96965` collapse comes from the attention kernel. Fixed `Hkv` usage in fused decode (GQA) and added alignment guard rails. Implemented MFMA vs VALU shadow compare at decode0. Result: `fused+mfma` no longer fails (no `hipMemcpy H2D invalid argument`) and MFMA==VALU (MAE=0) for all traced layers, but `attn_out` still diverges from the reference at layer 31 and the prefill→decode collapse persists. Focus shifts to **precision/accumulation or reference path** in high-layer decode attention.

## Estado Inicial (ES)
B3.20 mostró divergencia `attn_out` vs ref en layer 31 y `fused+mfma` fallaba en load. `prefill_last_top1=127158`, `decode0_top1=96965` persistía en p4/p5.

## Initial State (EN)
B3.20 showed `attn_out` vs ref divergence at layer 31 and `fused+mfma` failed at load. `prefill_last_top1=127158`, `decode0_top1=96965` persisted in p4/p5.

## Cambios Implementados (ES)
- Fix GQA en fused decode: usar `Hkv` (no `Hq`) al lanzar `launch_fused_rope_kv_update_decode`.
- Guard rails de alignment (Q/K/V/O/KV) con logging bajo `GRETA_TRACE_ATTN_ALIGN=1` o `GRETA_TRACE_ATTN_DECODE_VERIFY=1`.
- Shadow compare en decode0: MFMA vs VALU con `GRETA_ATTN_DECODE_MFMA_SHADOW=1` y JSONL dedicado.

## Implemented Changes (EN)
- GQA fix in fused decode: use `Hkv` (not `Hq`) when launching `launch_fused_rope_kv_update_decode`.
- Alignment guard rails (Q/K/V/O/KV) with logging under `GRETA_TRACE_ATTN_ALIGN=1` or `GRETA_TRACE_ATTN_DECODE_VERIFY=1`.
- Decode0 shadow compare: MFMA vs VALU via `GRETA_ATTN_DECODE_MFMA_SHADOW=1` and dedicated JSONL.

## Metodología / Methodology
Commit (run): `13d491f7e7916c1ef5595dbfd73b94841dfa63e2`

Flags principales:
- `GRETA_FORCE_ATTN_DECODE_KERNEL=fused`
- `GRETA_FORCE_ATTN_DECODE_MATMUL=mfma`
- `GRETA_ATTN_DECODE_MFMA_SHADOW=1`
- `GRETA_ATTN_DECODE_MFMA_SHADOW_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_21_attn_mfma_vs_valu.jsonl`
- `GRETA_TRACE_ATTN_DECODE_VERIFY=1`
- `GRETA_TRACE_ATTN_DECODE_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_21_attn_verify.jsonl`
- `GRETA_ATTN_DECODE_REF=1`
- `GRETA_TRACE_KV_INVARIANTS=1`

Prompts:
- `p4_sys` (system+user template)
- `p5_ba` (Buenos Aires)

Artefactos (local):
- `artifacts_remote/2026-02-03/b3_21/root/gretacore/artifacts/alignment/2026-02-03/b3_21_attn_mfma_vs_valu.jsonl`
- `artifacts_remote/2026-02-03/b3_21/root/gretacore/artifacts/alignment/2026-02-03/b3_21_attn_verify.jsonl`
- `artifacts_remote/2026-02-03/b3_21/root/gretacore/artifacts/alignment/2026-02-03/b3_21_p4_prefill_decode.jsonl`
- `artifacts_remote/2026-02-03/b3_21/root/gretacore/artifacts/alignment/2026-02-03/b3_21_p5_prefill_decode.jsonl`

## Resultados (ES)
### B3.21A — MFMA vs VALU (shadow, MAE)

**p4_sys (seq_len=206)**
| Layer | attn_out_mae | attn_out_max_diff | mfma_hash == valu_hash |
| --- | ---: | ---: | --- |
| 0 | 0 | 0 | true |
| 1 | 0 | 0 | true |
| 2 | 0 | 0 | true |
| 31 | 0 | 0 | true |

**p5_ba (seq_len=46)**
| Layer | attn_out_mae | attn_out_max_diff | mfma_hash == valu_hash |
| --- | ---: | ---: | --- |
| 0 | 0 | 0 | true |
| 1 | 0 | 0 | true |
| 2 | 0 | 0 | true |
| 31 | 0 | 0 | true |

Conclusión: MFMA y VALU son equivalentes en decode0 bajo fused path después del fix.

### B3.21B — attn_out vs ref (MAE)

**p4_sys (seq_len=206)**
| Layer | attn_out_mae | kv_invariant_ok |
| --- | ---: | --- |
| 0 | 6.00923e-11 | true |
| 1 | 6.57923e-04 | true |
| 2 | 9.48245e-04 | true |
| 31 | 4.19899e-02 | true |

**p5_ba (seq_len=46)**
| Layer | attn_out_mae | kv_invariant_ok |
| --- | ---: | --- |
| 0 | 5.34566e-11 | true |
| 1 | 8.78943e-04 | true |
| 2 | 1.18028e-03 | true |
| 31 | 4.38694e-02 | true |

Interpretación: divergencia significativa persiste en layer 31, aun con MFMA==VALU y KV invariants OK.

### B3.21C — Prefill vs Decode (readout)

| Prompt | prefill_last_top1 | decode0_top1 | gap prefill | gap decode |
| --- | ---: | ---: | ---: | ---: |
| p4_sys | 127158 | 96965 | 0.95866 | 1.34019 |
| p5_ba  | 127158 | 96965 | 0.95619 | 1.34019 |

El colapso `decode0→96965` persiste.

## Results (EN)
### B3.21A — MFMA vs VALU (shadow, MAE)

**p4_sys (seq_len=206)**
| Layer | attn_out_mae | attn_out_max_diff | mfma_hash == valu_hash |
| --- | ---: | ---: | --- |
| 0 | 0 | 0 | true |
| 1 | 0 | 0 | true |
| 2 | 0 | 0 | true |
| 31 | 0 | 0 | true |

**p5_ba (seq_len=46)**
| Layer | attn_out_mae | attn_out_max_diff | mfma_hash == valu_hash |
| --- | ---: | ---: | --- |
| 0 | 0 | 0 | true |
| 1 | 0 | 0 | true |
| 2 | 0 | 0 | true |
| 31 | 0 | 0 | true |

Conclusion: MFMA and VALU are equivalent at decode0 under the fused path after the fix.

### B3.21B — attn_out vs ref (MAE)

**p4_sys (seq_len=206)**
| Layer | attn_out_mae | kv_invariant_ok |
| --- | ---: | --- |
| 0 | 6.00923e-11 | true |
| 1 | 6.57923e-04 | true |
| 2 | 9.48245e-04 | true |
| 31 | 4.19899e-02 | true |

**p5_ba (seq_len=46)**
| Layer | attn_out_mae | kv_invariant_ok |
| --- | ---: | --- |
| 0 | 5.34566e-11 | true |
| 1 | 8.78943e-04 | true |
| 2 | 1.18028e-03 | true |
| 31 | 4.38694e-02 | true |

Interpretation: significant divergence persists at layer 31 even with MFMA==VALU and KV invariants OK.

### B3.21C — Prefill vs Decode (readout)

| Prompt | prefill_last_top1 | decode0_top1 | gap prefill | gap decode |
| --- | ---: | ---: | ---: | ---: |
| p4_sys | 127158 | 96965 | 0.95866 | 1.34019 |
| p5_ba  | 127158 | 96965 | 0.95619 | 1.34019 |

The `decode0→96965` collapse persists.

## Impacto en MI300X (ES)
- `fused+mfma` ahora ejecuta sin `hipMemcpy H2D invalid argument`.
- Shadow compare MFMA/VALU confirma equivalencia en decode0.
- La divergencia vs ref en layer 31 persiste, por lo que el foco se mantiene en atención decode de capas altas.

## Impact on MI300X (EN)
- `fused+mfma` now runs without `hipMemcpy H2D invalid argument`.
- MFMA/VALU shadow compare confirms equivalence at decode0.
- Ref divergence at layer 31 persists, so focus remains on high-layer decode attention.

## Conclusión / Conclusion
`fused+mfma` quedó estable y MFMA==VALU bajo shadow compare. Sin embargo, `attn_out` sigue divergiendo del ref en layer 31 y el colapso prefill→decode continúa. El candidato raíz pasa a **precisión/acumulación o ref path** en atención decode de capas altas (no KV, no LM head, no MFMA vs VALU).

## Próximo Paso / Next Step (B3.22)
Refinar el path de referencia (high-precision / no-fused) y aislar si la divergencia en layer 31 es numérica o de layout/stride; validar si la corrección debe aplicarse en la acumulación/softmax o en la referencia.

## Why this matters (ES)
Esta etapa demuestra que GRETA puede estabilizar kernels críticos (fused+MFMA) y aislar la divergencia a una zona precisa (layer 31) con trazabilidad determinista en MI300X, reduciendo el ciclo de diagnóstico y habilitando adopción industrial.

## Why this matters (EN)
This step shows GRETA can stabilize critical kernels (fused+MFMA) and pinpoint divergence to a specific region (layer 31) with deterministic traces on MI300X, reducing diagnosis cycle time and enabling industrial adoption.

---
L.E.T / Leandro Emanuel Timberini
