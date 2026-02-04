# B3.41 — Post-WO Collapse Isolation (prefill_last vs decode0)

## Resumen Ejecutivo (ES)
Objetivo: aislar el primer punto de divergencia **después de WO** (ya corregido) comparando `prefill_last` vs `decode0` en modo alineado (E1). Se instrumentó un trazado post-WO con hashes/estadísticas y muestras controladas. El resultado muestra que el primer mismatch aparece en **ffn_norm** (RMSNorm previo al MLP). Esto desplaza el foco a la ruta de RMSNorm/FFN en decode0 (pesos/epsilon/precisión o buffer selection), no a atención ni WO.

## Executive Summary (EN)
Goal: isolate the first divergence **after WO** (already fixed) by comparing `prefill_last` vs `decode0` in aligned mode (E1). We added a post-WO trace with hashes/stats and sampled slices. The earliest mismatch appears at **ffn_norm** (RMSNorm before MLP). This shifts root-cause investigation to RMSNorm/FFN path in decode0 (weights/epsilon/precision or buffer selection), not attention nor WO.

---

## Contexto / Context
- B3.40 fixed WO projection (row layout). `wo_out` and `x_after_attn` now match (MAE ~1e-10).
- Decode collapse persists (decode0 → 96965) for context prompts.
- B3.41 isolates the first mismatch **after** x_after_attn.

---

## Metodología (ES)
- Modo: E1 (prefill alignment flags activos).
- Prompts: `p0_short`, `p4_sys`, `p6_long`.
- Trazas: `GRETA_TRACE_POST_WO=1` con salida JSONL por prompt.
- Se compararon muestras (slice fijo) por etapa post-WO:
  `ffn_norm`, `mlp_out`, `x_after_mlp`, `x_out`, `final_rms`, `lm_head_in`.

## Method (EN)
- Mode: E1 (prefill alignment flags enabled).
- Prompts: `p0_short`, `p4_sys`, `p6_long`.
- Traces: `GRETA_TRACE_POST_WO=1` with per-prompt JSONL output.
- Stages compared via fixed slices:
  `ffn_norm`, `mlp_out`, `x_after_mlp`, `x_out`, `final_rms`, `lm_head_in`.

---

## Resultados / Results (B3.41 Analyzer)
Source: `artifacts_remote/2026-02-03/b3_41/b3_41_analysis.txt`

| prompt | exp | first_mismatch_stage | ffn_norm_mae | mlp_out_mae | x_after_mlp_mae | x_out_mae | prefill_last_top1 | decode0_top1 | collapse_96965 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| p0_short | E1 | ffn_norm | 1.533e-01 | 3.590e-01 | 3.593e-01 | 3.593e-01 | 96965 | 96965 | true |
| p4_sys | E1 | ffn_norm | 1.675e-01 | 3.159e-01 | 3.173e-01 | 3.173e-01 | 93590 | 96965 | true |
| p6_long | E1 | ffn_norm | 1.539e-01 | 6.088e-01 | 6.086e-01 | 6.086e-01 | 99668 | 96965 | true |

**Interpretación / Interpretation:**
- `attn_out`, `wo_out`, `x_after_attn` no divergen.
- El primer mismatch aparece en **ffn_norm** (RMSNorm previo a MLP), y se propaga a `mlp_out` y `x_out`.
- Implica que el colapso está post-WO y pre-MLP, probablemente en RMSNorm/FFN path (pesos/epsilon/precisión/selección de buffer).

---

## Comandos (ES/EN)
```bash
# Remote run (scripted)
./tools/benchmarks/run_b3_41_mi300x.sh 129.212.184.200 2026-02-03

# Local analysis
python3 tools/benchmarks/analyze_b3_41_post_wo.py \
  --dir artifacts_remote/2026-02-03/b3_41/root/gretacore/artifacts/alignment/2026-02-03 \
  --out artifacts_remote/2026-02-03/b3_41/b3_41_analysis.txt
```

---

## Conclusión (ES)
El primer punto de divergencia **después de WO** se encuentra en **ffn_norm**. La corrección debe enfocarse en la ruta de RMSNorm/FFN en decode0 (pesos/epsilon/precisión o selección de buffer). No es un problema de atención ni de WO.

## Conclusion (EN)
The first divergence **after WO** occurs at **ffn_norm**. The fix must target RMSNorm/FFN path in decode0 (weights/epsilon/precision or buffer selection). This is not an attention nor WO issue.

---

## Próximo paso (B3.42)
Auditar y aislar la ruta de **RMSNorm pre-MLP** entre prefill_last y decode0:
- verificar pesos/epsilon,
- ruta de precisión (FP16/FP32),
- buffers/offsets usados en decode.

---

L.E.T / Leandro Emanuel Timberini
