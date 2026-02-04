# B3.31 — Decode Q/K/V Projection Route Isolation

## Resumen Ejecutivo (ES)
Objetivo: aislar si el mismatch en Q/K/V de decode proviene de la **ruta de proyección** (MFMA vs VALU) o del uso de GEMM vs GEMV. Se implementaron flags para forzar ruta y GEMM en decode y se ejecutó una matriz E0–E4. Resultado: **la MAE de Q no cambia** entre rutas (MFMA/VALU) ni con GEMM forzado. Conclusión: el problema **no es la ruta de kernel** sino la **entrada a la proyección** (RMSNorm / norm_out) o un problema de pesos/layout en la proyección que afecta tanto MFMA como VALU.

## Executive Summary (EN)
Goal: isolate whether decode Q/K/V mismatch is caused by **projection route** (MFMA vs VALU) or GEMM vs GEMV. We added flags to force route and GEMM in decode and ran an E0–E4 matrix. Result: **Q MAE does not change** across routes (MFMA/VALU) or when forcing GEMM. Conclusion: the issue is **not the kernel route** but the **projection input** (RMSNorm / norm_out) or a weights/layout issue affecting both MFMA and VALU.

## Metodología (ES)
Flags añadidos:
- `GRETA_QKV_FORCE_ROUTE=mfma|valu|auto`
- `GRETA_QKV_FORCE_GEMM=1`

Runner:
- `tools/benchmarks/run_b3_31_mi300x.sh 129.212.184.200 2026-02-03`

Trazas:
- `GRETA_TRACE_ATTN_L0_PIPE=1`
- `GRETA_TRACE_STAGE_DEBUG_INPUT=1` (alinear decode0 con prefill_last)

Artefactos locales:
- `artifacts_remote/2026-02-03/b3_31/gretacore_b3_31_artifacts.tgz`
- `artifacts_remote/2026-02-03/b3_31/b3_31_analysis.txt`

## Methodology (EN)
Added flags:
- `GRETA_QKV_FORCE_ROUTE=mfma|valu|auto`
- `GRETA_QKV_FORCE_GEMM=1`

Runner:
- `tools/benchmarks/run_b3_31_mi300x.sh 129.212.184.200 2026-02-03`

Traces:
- `GRETA_TRACE_ATTN_L0_PIPE=1`
- `GRETA_TRACE_STAGE_DEBUG_INPUT=1` (align decode0 with prefill_last)

Local artifacts:
- `artifacts_remote/2026-02-03/b3_31/gretacore_b3_31_artifacts.tgz`
- `artifacts_remote/2026-02-03/b3_31/b3_31_analysis.txt`

## Resultados (ES)
Matriz de rutas (MAE Q/K/V prefill_last vs decode0, layer0/head0):

| Exp | Prompt | Q MAE | K MAE | V MAE | Q route | K route | V route | force_route | force_gemm |
|---|---|---:|---:|---:|---|---|---|---|---|
| E0 | p4_sys | 5.37518e-01 | 1.99312e-01 | 2.68110e-03 | VALU | VALU | VALU | auto | false |
| E0 | p5_ba  | 8.88882e-01 | 2.55535e-01 | 2.98710e-03 | VALU | VALU | VALU | auto | false |
| E1 | p4_sys | 5.37518e-01 | 1.99312e-01 | 2.68110e-03 | VALU | VALU | VALU | valu | false |
| E1 | p5_ba  | 8.88882e-01 | 2.55535e-01 | 2.98710e-03 | VALU | VALU | VALU | valu | false |
| E2 | p4_sys | 5.37518e-01 | 1.99312e-01 | 2.68110e-03 | MFMA | MFMA | MFMA | mfma | false |
| E2 | p5_ba  | 8.88882e-01 | 2.55535e-01 | 2.98710e-03 | MFMA | MFMA | MFMA | mfma | false |
| E3 | p4_sys | 5.37518e-01 | 1.99312e-01 | 2.68110e-03 | MFMA | MFMA | MFMA | mfma | true |
| E3 | p5_ba  | 8.88882e-01 | 2.55535e-01 | 2.98710e-03 | MFMA | MFMA | MFMA | mfma | true |
| E4 | p4_sys | 5.37518e-01 | 1.99312e-01 | 2.68110e-03 | VALU | VALU | VALU | valu | true |
| E4 | p5_ba  | 8.88882e-01 | 2.55535e-01 | 2.98710e-03 | VALU | VALU | VALU | valu | true |

Conclusión: la MAE de Q no cambia entre rutas; forzar GEMM tampoco reduce el error.

## Results (EN)
Route matrix (Q/K/V MAE prefill_last vs decode0, layer0/head0):

| Exp | Prompt | Q MAE | K MAE | V MAE | Q route | K route | V route | force_route | force_gemm |
|---|---|---:|---:|---:|---|---|---|---|---|
| E0 | p4_sys | 5.37518e-01 | 1.99312e-01 | 2.68110e-03 | VALU | VALU | VALU | auto | false |
| E0 | p5_ba  | 8.88882e-01 | 2.55535e-01 | 2.98710e-03 | VALU | VALU | VALU | auto | false |
| E1 | p4_sys | 5.37518e-01 | 1.99312e-01 | 2.68110e-03 | VALU | VALU | VALU | valu | false |
| E1 | p5_ba  | 8.88882e-01 | 2.55535e-01 | 2.98710e-03 | VALU | VALU | VALU | valu | false |
| E2 | p4_sys | 5.37518e-01 | 1.99312e-01 | 2.68110e-03 | MFMA | MFMA | MFMA | mfma | false |
| E2 | p5_ba  | 8.88882e-01 | 2.55535e-01 | 2.98710e-03 | MFMA | MFMA | MFMA | mfma | false |
| E3 | p4_sys | 5.37518e-01 | 1.99312e-01 | 2.68110e-03 | MFMA | MFMA | MFMA | mfma | true |
| E3 | p5_ba  | 8.88882e-01 | 2.55535e-01 | 2.98710e-03 | MFMA | MFMA | MFMA | mfma | true |
| E4 | p4_sys | 5.37518e-01 | 1.99312e-01 | 2.68110e-03 | VALU | VALU | VALU | valu | true |
| E4 | p5_ba  | 8.88882e-01 | 2.55535e-01 | 2.98710e-03 | VALU | VALU | VALU | valu | true |

Conclusion: Q MAE is unchanged across routes; forcing GEMM does not reduce error.

## Conclusión Técnica (ES)
El mismatch de Q/K/V en decode **no depende** de MFMA vs VALU ni de GEMM vs GEMV. El siguiente foco debe ser el **input a la proyección**: salida de RMSNorm (norm_out) o un problema de layout/packing de pesos que impacta a ambas rutas. Se requiere instrumentar y comparar `norm_out` prefill_last vs decode0, y/o verificar el layout efectivo de los pesos Q/K/V en decode.

## Technical Conclusion (EN)
Decode Q/K/V mismatch is **not caused** by MFMA vs VALU or GEMM vs GEMV. The next focus must be the **projection input**: RMSNorm output (norm_out) or a weights/layout issue affecting both routes. We need to instrument and compare `norm_out` prefill_last vs decode0, and/or verify effective Q/K/V weight layout in decode.

## Próximo Paso (B3.32) (ES)
- Instrumentar `norm_out` prefill_last vs decode0 (MAE y hashes).
- Verificar el layout efectivo de pesos Q/K/V para decode (probe de 1–2 filas como en EMBED_VERIFY).
- Si `norm_out` ya diverge, auditar RMSNorm decode (kernel y parámetros).

## Next Step (B3.32) (EN)
- Instrument `norm_out` prefill_last vs decode0 (MAE and hashes).
- Verify effective Q/K/V weight layout for decode (row/col probe like EMBED_VERIFY).
- If `norm_out` already diverges, audit decode RMSNorm kernel/params.

---

**L.E.T / Leandro Emanuel Timberini**
