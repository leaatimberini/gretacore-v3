# B3.37 — Full Pipeline Acceptance (Prefill vs Decode)

## Resumen Ejecutivo (ES)
B3.37 valida el síntoma end-to-end tras alinear Q/K/V en prefill (B3.36 E1). Resultado: **la colapsación determinista en decode0 persiste**. En prompts de contexto y largo, `prefill_last_top1 != decode0_top1`, y decode0 cae inmediatamente a `96965` con `uniq_top1_0_15 = 1`. La alineación Q/K/V no es suficiente para resolver el colapso; el problema está aguas abajo (p.ej. atención/MLP/residual/LM head u otra transición post-QKV).

## Executive Summary (EN)
B3.37 validates the end-to-end symptom after aligning Q/K/V in prefill (B3.36 E1). Result: **deterministic decode0 collapse persists**. For context/long prompts, `prefill_last_top1 != decode0_top1`, and decode0 immediately falls to `96965` with `uniq_top1_0_15 = 1`. Q/K/V alignment is not sufficient; the failure is downstream (attention/MLP/residual/LM head or other post-QKV transition).

---

## Contexto / Context
- B3.36: Q/K/V equivalencia prefill_last vs decode0 lograda en layer0.
- Objetivo B3.37: confirmar si el síntoma (colapso) se corrige en el pipeline completo.

## Metodología / Method
- Modelo: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
- Prompts: `p0_short`, `p4_sys`, `p5_ba`, `p6_long`
- Experimentos:
  - **E0**: baseline (sin forcing)
  - **E1**: forcing de alineación Q/K/V en prefill (`GRETA_PREFILL_FORCE_WQ_ROW=1`, `GRETA_PREFILL_FORCE_WK_ROW=1`, `GRETA_PREFILL_FORCE_WV_LAYOUT=row`)
- Flags:
  - `GRETA_TRACE_PREFILL_DECODE=1`
  - `--greedy --debug-decode 16 --max-tokens 16`
- Output: JSONL `b3_37_*_prefill_decode_*.jsonl` + logs

## Resultados (ES)
**Tabla resumida (de `b3_37_analysis.txt`):**

| Prompt | Exp | prefill_last_top1 | decode0_top1 | match | uniq_top1_0_15 | top1_s0..s3 | gap_prefill | gap_decode | collapse_96965 |
|---|---|---:|---:|---|---:|---|---:|---:|---|
| p0_short | E0 | 96965 | 96965 | true | 1 | 96965,96965,96965,96965 | 1.4504 | 1.44461 | true |
| p4_sys | E0 | 93590 | 96965 | false | 1 | 96965,96965,96965,96965 | 2.57582 | 1.43443 | true |
| p5_ba | E0 | 198 | 96965 | false | 1 | 96965,96965,96965,96965 | 0.210462 | 1.43889 | true |
| p6_long | E0 | 52196 | 96965 | false | 1 | 96965,96965,96965,96965 | 0.0299759 | 1.44244 | true |
| p0_short | E1 | 96965 | 96965 | true | 1 | 96965,96965,96965,96965 | 1.4504 | 1.44461 | true |
| p4_sys | E1 | 93590 | 96965 | false | 1 | 96965,96965,96965,96965 | 2.57947 | 1.43282 | true |
| p5_ba | E1 | 198 | 96965 | false | 1 | 96965,96965,96965,96965 | 0.211458 | 1.43919 | true |
| p6_long | E1 | 52196 | 96965 | false | 1 | 96965,96965,96965,96965 | 0.0298271 | 1.44187 | true |

**Interpretación:**
- E1 no cambia el colapso: decode0 sigue fijado en 96965 para prompts con contexto.
- El problema **no está en Q/K/V prefill vs decode**, y debe localizarse más adelante en el pipeline.

## Results (EN)
**Summary table (from `b3_37_analysis.txt`):**

| Prompt | Exp | prefill_last_top1 | decode0_top1 | match | uniq_top1_0_15 | top1_s0..s3 | gap_prefill | gap_decode | collapse_96965 |
|---|---|---:|---:|---|---:|---|---:|---:|---|
| p0_short | E0 | 96965 | 96965 | true | 1 | 96965,96965,96965,96965 | 1.4504 | 1.44461 | true |
| p4_sys | E0 | 93590 | 96965 | false | 1 | 96965,96965,96965,96965 | 2.57582 | 1.43443 | true |
| p5_ba | E0 | 198 | 96965 | false | 1 | 96965,96965,96965,96965 | 0.210462 | 1.43889 | true |
| p6_long | E0 | 52196 | 96965 | false | 1 | 96965,96965,96965,96965 | 0.0299759 | 1.44244 | true |
| p0_short | E1 | 96965 | 96965 | true | 1 | 96965,96965,96965,96965 | 1.4504 | 1.44461 | true |
| p4_sys | E1 | 93590 | 96965 | false | 1 | 96965,96965,96965,96965 | 2.57947 | 1.43282 | true |
| p5_ba | E1 | 198 | 96965 | false | 1 | 96965,96965,96965,96965 | 0.211458 | 1.43919 | true |
| p6_long | E1 | 52196 | 96965 | false | 1 | 96965,96965,96965,96965 | 0.0298271 | 1.44187 | true |

**Interpretation:**
- E1 does not change the collapse: decode0 still locks to 96965 for context prompts.
- The issue is **not** in prefill vs decode Q/K/V; it is downstream in the pipeline.

## Evidencia / Evidence
- Analyzer: `artifacts_remote/2026-02-03/b3_37/b3_37_analysis.txt`
- JSONL: `artifacts_remote/2026-02-03/b3_37/root/gretacore/artifacts/alignment/2026-02-03/`
- Logs: `artifacts_remote/2026-02-03/b3_37/root/gretacore/artifacts/alignment/2026-02-03/b3_37_*.log`
- Commit de ejecución: `e5f6cd3`

## Conclusión / Conclusion
**B3.37 FAIL (síntoma persiste).** La alineación Q/K/V en prefill no elimina la colapsación; el siguiente aislamiento debe enfocarse en etapas posteriores (attention output, MLP, residuals, final norm, LM head o selección de logits post-QKV).

## Próximo paso / Next step
- **B3.38:** aislamiento post-QKV (attention/MLP/residual/final norm/LM head) con trazas comparativas prefill_last vs decode0.

---

L.E.T / Leandro Emanuel Timberini
