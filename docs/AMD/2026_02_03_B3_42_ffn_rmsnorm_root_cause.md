# B3.42 — RMSNorm pre-MLP Root Cause Isolation

## Resumen Ejecutivo (ES)
Objetivo: explicar por qué `ffn_norm` diverge entre `prefill_last` y `decode0` (B3.41). Se instrumentó el RMSNorm pre-MLP con trazas de entrada/salida/pesos, offsets y estadísticas. Resultado: **la entrada al RMSNorm no coincide** entre fases (MAE ~1e-2), mientras los pesos y epsilon coinciden. Esto indica un problema de **selección/offset del input** (bucket A), no de pesos ni de la matemática del RMSNorm.

## Executive Summary (EN)
Goal: explain why `ffn_norm` diverges between `prefill_last` and `decode0` (B3.41). We instrumented pre-MLP RMSNorm with input/output/weight traces, offsets, and stats. Result: **RMSNorm input differs** between phases (MAE ~1e-2), while weights and epsilon match. This indicates an **input selection/offset issue** (bucket A), not weight layout nor RMSNorm math.

---

## Metodología (ES)
- Flags:
  - `GRETA_TRACE_RMSNORM=1`
  - `GRETA_TRACE_RMSNORM_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_42_rmsnorm_<prompt>_E1.jsonl`
  - `GRETA_TRACE_RMSNORM_LAYERS="0"`
  - `GRETA_TRACE_RMSNORM_SAMPLE=1024`
- Prompts: `p0_short`, `p4_sys`, `p6_long`.
- Modo: E1 (prefill alignment flags activos).
- Se compararon input/output/weight en RMSNorm (slice 1024 floats), y offsets/ptrs.

## Method (EN)
- Flags:
  - `GRETA_TRACE_RMSNORM=1`
  - `GRETA_TRACE_RMSNORM_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_42_rmsnorm_<prompt>_E1.jsonl`
  - `GRETA_TRACE_RMSNORM_LAYERS="0"`
  - `GRETA_TRACE_RMSNORM_SAMPLE=1024`
- Prompts: `p0_short`, `p4_sys`, `p6_long`.
- Mode: E1 (prefill alignment flags enabled).
- Compared RMSNorm input/output/weight slices and ptr/offset metadata.

---

## Resultados (ES)
Fuente: `artifacts_remote/2026-02-03/b3_42/b3_42_analysis.txt`

| Prompt | input_mae | weight_mae | output_mae | eps_match | sumsq_delta | inv_rms_delta | input_ptr_match | input_offset_match | weight_hash_match | Bucket |
|---|---:|---:|---:|---|---:|---:|---|---|---|---|
| p0_short | 1.241e-02 | 0.0 | 1.533e-01 | true | 1.83e-05 | 5.83e-06 | false | false | true | A_input_selection |
| p4_sys | 1.522e-02 | 0.0 | 1.675e-01 | true | 1.02e-04 | 2.31e+01 | false | false | true | A_input_selection |
| p6_long | 1.351e-02 | 0.0 | 1.539e-01 | true | 1.28e-04 | 2.74e+01 | false | false | true | A_input_selection |

**Interpretación:**
- `weight_hash_match=true` y `eps_match=true`: pesos/epsilon no son el problema.
- `input_mae` > 1e-2 y offsets/ptrs distintos: la entrada al RMSNorm no coincide entre fases.
- Conclusión: **Bucket A** (selección/offset del input pre-MLP) es la causa raíz inmediata.

## Results (EN)
Source: `artifacts_remote/2026-02-03/b3_42/b3_42_analysis.txt`

| Prompt | input_mae | weight_mae | output_mae | eps_match | sumsq_delta | inv_rms_delta | input_ptr_match | input_offset_match | weight_hash_match | Bucket |
|---|---:|---:|---:|---|---:|---:|---|---|---|---|
| p0_short | 1.241e-02 | 0.0 | 1.533e-01 | true | 1.83e-05 | 5.83e-06 | false | false | true | A_input_selection |
| p4_sys | 1.522e-02 | 0.0 | 1.675e-01 | true | 1.02e-04 | 2.31e+01 | false | false | true | A_input_selection |
| p6_long | 1.351e-02 | 0.0 | 1.539e-01 | true | 1.28e-04 | 2.74e+01 | false | false | true | A_input_selection |

**Interpretation:**
- `weight_hash_match=true` and `eps_match=true`: weights/epsilon are not the issue.
- `input_mae` > 1e-2 and differing offsets/ptrs show RMSNorm input differs between phases.
- Conclusion: **Bucket A** (input selection/offset) is the immediate root cause.

---

## Comandos (ES/EN)
```bash
./tools/benchmarks/run_b3_42_mi300x.sh 129.212.184.200 2026-02-03
python3 tools/benchmarks/analyze_b3_42_rmsnorm.py \
  --dir artifacts_remote/2026-02-03/b3_42/root/gretacore/artifacts/alignment/2026-02-03 \
  --out artifacts_remote/2026-02-03/b3_42/b3_42_analysis.txt
```

---

## Conclusión (ES)
El primer desvío post‑WO proviene de la **entrada al RMSNorm pre‑MLP**: la selección/offset del input en `decode0` no coincide con `prefill_last`. La corrección debe enfocarse en **alinear el buffer/offset** usado como input al RMSNorm en decode.

## Conclusion (EN)
The first post‑WO divergence comes from **the RMSNorm pre‑MLP input**: decode0 input selection/offset does not match prefill_last. The fix must align the **input buffer/offset** used by RMSNorm in decode.

---

## Próximo paso (B3.43)
Alinear la selección del input para RMSNorm en decode0:
- verificar si `activations_.x` usa offset correcto para `seq_start` (token lógico),
- garantizar que el RMSNorm opere sobre el mismo token lógico que prefill_last.

---

L.E.T / Leandro Emanuel Timberini
