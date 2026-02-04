# B3.38 — Post‑QKV Root‑Cause Isolation (Decode0 Collapse)

## Resumen Ejecutivo (ES)
B3.38 compara prefill_last vs decode0 **después** de Q/K/V, usando trazas post‑QKV y la tubería L0. Resultado: el **primer mismatch ocurre en `x_after_attn` (residual add)** para prompts de contexto y largos, mientras `attn_out` sigue coincidiendo (MAE ~1e‑10). Esto indica que el problema no está en Q/K/V ni en atención, sino en el **output projection (WO) o la suma residual** en decode0.

## Executive Summary (EN)
B3.38 compares prefill_last vs decode0 **after** Q/K/V using post‑QKV traces. Result: the **first mismatch occurs at `x_after_attn` (residual add)** for context and long prompts, while `attn_out` still matches (MAE ~1e‑10). This indicates the issue is not in Q/K/V or attention, but in the **output projection (WO) or the residual add** in decode0.

---

## Contexto / Context
- B3.36: Q/K/V alignment fixed in prefill (E1)
- B3.37: collapse persists despite Q/K/V alignment
- Objetivo B3.38: ubicar el primer mismatch **después** de Q/K/V

## Metodología / Method
- Prompts: `p0_short`, `p4_sys`, `p6_long`
- Modo: E1 (prefill alignment activo)
- Trazas:
  - `attn_l0_pipe` (q/k/v/qk/softmax/pv/attn_out)
  - `stage_trace` (x_after_attn, ffn_norm, mlp_out, x_out, final_norm, lm_head_in)
- Output JSONL único por prompt:
  - `b3_38_postqkv_<prompt>_E1.jsonl`

## Resultados (ES)
**Tabla resumida (de `b3_38_analysis.txt`):**

| Prompt | Exp | first_mismatch_stage | attn_out_mae | resid_mae | ffn_norm_mae | mlp_out_mae | x_out_mae | prefill_last_top1 | decode0_top1 | collapse_96965 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| p0_short | E1 | NONE | ~2.1e‑11 | ~9.8e‑11 | ~2.5e‑09 | ~3.9e‑08 | ~3.0e‑08 | 96965 | 96965 | true |
| p4_sys | E1 | x_after_attn | ~1.68e‑10 | 7.40e‑04 | 9.71e‑03 | 2.42e‑01 | 2.42e‑01 | 93590 | 96965 | true |
| p6_long | E1 | x_after_attn | ~7.00e‑10 | 1.35e‑03 | 1.79e‑02 | 2.73e‑01 | 2.73e‑01 | 52196 | 96965 | true |

**Interpretación:**
- `attn_out` coincide (MAE ~1e‑10) → atención correcta.
- Mismatch aparece en `x_after_attn` → problema en **WO GEMM** o **residual add** para decode0.

## Results (EN)
**Summary table (from `b3_38_analysis.txt`):**

| Prompt | Exp | first_mismatch_stage | attn_out_mae | resid_mae | ffn_norm_mae | mlp_out_mae | x_out_mae | prefill_last_top1 | decode0_top1 | collapse_96965 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| p0_short | E1 | NONE | ~2.1e‑11 | ~9.8e‑11 | ~2.5e‑09 | ~3.9e‑08 | ~3.0e‑08 | 96965 | 96965 | true |
| p4_sys | E1 | x_after_attn | ~1.68e‑10 | 7.40e‑04 | 9.71e‑03 | 2.42e‑01 | 2.42e‑01 | 93590 | 96965 | true |
| p6_long | E1 | x_after_attn | ~7.00e‑10 | 1.35e‑03 | 1.79e‑02 | 2.73e‑01 | 2.73e‑01 | 52196 | 96965 | true |

**Interpretation:**
- `attn_out` matches (MAE ~1e‑10) → attention core is correct.
- Mismatch starts at `x_after_attn` → issue in **WO GEMM** or **residual add** for decode0.

## Evidencia / Evidence
- Analyzer: `artifacts_remote/2026-02-03/b3_38/b3_38_analysis.txt`
- JSONL: `artifacts_remote/2026-02-03/b3_38/root/gretacore/artifacts/alignment/2026-02-03/`
- Logs: `artifacts_remote/2026-02-03/b3_38/root/gretacore/artifacts/alignment/2026-02-03/b3_38_E1_*.log`
- Commit de ejecución: `74f7388`

## Conclusión / Conclusion
**El primer mismatch post‑QKV aparece en `x_after_attn`.** Esto apunta a la proyección de salida de atención (WO) o a la suma residual en decode0. Q/K/V y attn_out están alineados; el fallo está inmediatamente después de WO o en la operación de suma residual.

## Próximo paso / Next step
- **B3.39:** aislar WO GEMM vs residual add en decode0 (trazas específicas de `mlp_out` pre‑residual y `x` residual input para confirmar cuál diverge primero).

---

L.E.T / Leandro Emanuel Timberini
