# B3.20 — Attention Decode / KV / Estado (Isolation)

## Resumen Ejecutivo (ES)
Objetivo: aislar si el colapso `decode0→96965` proviene de inconsistencias en Attention Decode (Q/K/V, KV-cache, offsets o kernel path). Se agregaron trazas de verificación con referencia CPU, invariantes de KV y una matriz de rutas forzadas. Resultado: las invariantes de KV se mantienen, pero el `attn_out` diverge del ref en capas altas (layer 31) y el path `fused+mfma` falla en load. El candidato raíz pasa a **kernel path/precision de atención en decode**, no a KV offset.

## Executive Summary (EN)
Goal: isolate whether `decode0→96965` collapse comes from Attention Decode inconsistencies (Q/K/V, KV-cache, offsets, or kernel path). Added CPU reference verification, KV invariants, and a forced route matrix. Result: KV invariants hold, but `attn_out` diverges from the ref at high layers (layer 31) and the `fused+mfma` path fails at load. Root-cause candidate shifts to **decode attention kernel path/precision**, not KV offsets.

## Estado Inicial (ES)
B3.19 falló: `prefill_last_top1=127158` y `decode0_top1=96965` en p4/p5. CPU probe concuerda con GPU en decode → LM head no es causa primaria.

## Initial State (EN)
B3.19 failed: `prefill_last_top1=127158` and `decode0_top1=96965` on p4/p5. CPU probe agrees with GPU in decode → LM head is not the primary cause.

## Metodología / Methodology
Commit: `1448f28e195f1dcfe27da2ab1c7e653dcd115de9`

Flags principales:
- `GRETA_TRACE_ATTN_DECODE_VERIFY=1`
- `GRETA_TRACE_KV_INVARIANTS=1`
- `GRETA_TRACE_ATTN_LAYERS="0,1,2,31"`
- `GRETA_TRACE_ATTN_POINTS="q,k,v,attn_out,x_out"`
- `GRETA_ATTN_DECODE_REF=1`

Matriz (p4_sys, p5_ba, `max-tokens=2`):
- `GRETA_FORCE_ATTN_DECODE_KERNEL=auto|manual|fused`
- `GRETA_FORCE_ATTN_DECODE_MATMUL=auto|valu|mfma`

Outputs (local):
- `artifacts_remote/2026-02-03/b3_20/artifacts/alignment/2026-02-03/b3_20_p4_run_attn_verify.jsonl`
- `artifacts_remote/2026-02-03/b3_20/artifacts/alignment/2026-02-03/b3_20_p5_run_attn_verify.jsonl`
- `artifacts_remote/2026-02-03/b3_20/artifacts/alignment/2026-02-03/b3_20_matrix_summary.txt`

## Resultados (ES)
### B3.20A — `attn_out` vs `attn_out_ref` (MAE por layer)

**p4_sys (auto/auto, manual path)**
| Layer | attn_out_mae |
| --- | ---: |
| 0 | 6.00923e-11 |
| 1 | 6.57923e-04 |
| 2 | 9.48245e-04 |
| 31 | 4.19899e-02 |

**p5_ba (auto/auto, manual path)**
| Layer | attn_out_mae |
| --- | ---: |
| 0 | 5.34566e-11 |
| 1 | 8.78943e-04 |
| 2 | 1.18028e-03 |
| 31 | 4.38694e-02 |

Interpretación: divergencia significativa en layer 31 (MAE ≈ 4e-2) aun con KV invariants OK.

### B3.20B — KV invariants

| Prompt | Layer | kv_invariant_ok |
| --- | ---: | --- |
| p4_sys | 0,1,2,31 | true |
| p5_ba  | 0,1,2,31 | true |

No se observaron violaciones de offsets/posiciones en KV.

### B3.20C — Matriz kernel/path vs `decode0_top1`

| Config | p4_sys decode0_top1 | p5_ba decode0_top1 | Estado |
| --- | ---: | ---: | --- |
| auto / auto | 127158 | 127158 | OK |
| manual / valu | 127158 | 127158 | OK |
| manual / mfma | 127158 | 127158 | OK |
| fused / valu | 127158 | 127158 | OK |
| fused / mfma | ERROR | ERROR | `hipMemcpy H2D failed: invalid argument` |

Notas: `fused+mfma` falla durante load, sin generar JSONL de atención.

## Results (EN)
### B3.20A — `attn_out` vs `attn_out_ref` (MAE per layer)

**p4_sys (auto/auto, manual path)**
| Layer | attn_out_mae |
| --- | ---: |
| 0 | 6.00923e-11 |
| 1 | 6.57923e-04 |
| 2 | 9.48245e-04 |
| 31 | 4.19899e-02 |

**p5_ba (auto/auto, manual path)**
| Layer | attn_out_mae |
| --- | ---: |
| 0 | 5.34566e-11 |
| 1 | 8.78943e-04 |
| 2 | 1.18028e-03 |
| 31 | 4.38694e-02 |

Interpretation: significant divergence at layer 31 (MAE ≈ 4e-2) despite KV invariants OK.

### B3.20B — KV invariants

| Prompt | Layer | kv_invariant_ok |
| --- | ---: | --- |
| p4_sys | 0,1,2,31 | true |
| p5_ba  | 0,1,2,31 | true |

No KV offset/position violations observed.

### B3.20C — Kernel/path matrix vs `decode0_top1`

| Config | p4_sys decode0_top1 | p5_ba decode0_top1 | Status |
| --- | ---: | ---: | --- |
| auto / auto | 127158 | 127158 | OK |
| manual / valu | 127158 | 127158 | OK |
| manual / mfma | 127158 | 127158 | OK |
| fused / valu | 127158 | 127158 | OK |
| fused / mfma | ERROR | ERROR | `hipMemcpy H2D failed: invalid argument` |

## Conclusión / Conclusion
La evidencia apunta a **divergencia en el kernel de atención decode** (especialmente en capas altas), no a offsets de KV. `fused+mfma` no es estable (falla en load), lo que limita su uso en decode. Siguiente paso (B3.21): auditar kernel path de atención decode y su precisión/acumulación en layers altos, e investigar por qué `fused+mfma` falla con este GGUF/config.

## Por qué importa (ES)
Esta trazabilidad demuestra que GRETA puede aislar bugs de decode a nivel kernel/KV de forma determinista en VFs MI300X, acelerando adopción industrial y justificando financiamiento/credits de AMD.

## Why this matters (EN)
This demonstrates GRETA’s ability to isolate decode bugs at the kernel/KV level with deterministic traces on MI300X VFs, accelerating industrial adoption and justifying AMD credits/funding.

---
L.E.T / Leandro Emanuel Timberini
