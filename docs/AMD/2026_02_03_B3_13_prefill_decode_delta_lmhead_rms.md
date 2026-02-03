# B3.13 — Prefill vs Decode0 Delta (RMSNorm + LM Head Probes)

## Resumen Ejecutivo (ES)
Objetivo: aislar si el colapso en decode (top1=96965) proviene de RMSNorm final o del LM head (dequant/layout/ruta) comparando prefill_last vs decode0 en p4/p5. Se agregaron trazas de delta (hidden/rms/logits + rutas LM head) y un CPU probe mínimo para validar coherencia de logits.

## Executive Summary (EN)
Goal: isolate whether decode collapse (top1=96965) comes from final RMSNorm or LM head (dequant/layout/route) by comparing prefill_last vs decode0 on p4/p5. Added delta traces (hidden/rms/logits + LM head route) and a minimal CPU probe to validate logits coherence.

## Cambios Implementados (ES)
- Trazas `GRETA_TRACE_PREFILL_DECODE_DELTA`, `GRETA_TRACE_RMS_VERIFY`, `GRETA_TRACE_LMHEAD_CPU_PROBE`.
- JSONL `b3_13_delta.jsonl` con líneas `prefill_last` y `decode0`.
- Auditoría de LM head (ruta, quant, layout, tipos, scales hash) y CPU probe opcional.

## Implemented Changes (EN)
- Traces `GRETA_TRACE_PREFILL_DECODE_DELTA`, `GRETA_TRACE_RMS_VERIFY`, `GRETA_TRACE_LMHEAD_CPU_PROBE`.
- JSONL `b3_13_delta.jsonl` with `prefill_last` and `decode0` lines.
- LM head audit (route, quant, layout, dtypes, scales hash) and optional CPU probe.

## Configuración de Run (ES/EN)
- Modelo: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
- Flags:
  - `GRETA_INT4_WEIGHTS=1`
  - `GRETA_MAX_SEQ_LEN=256`
  - `GRETA_TRACE_PREFILL_DECODE_DELTA=1`
  - `GRETA_TRACE_RMS_VERIFY=1`
  - `GRETA_TRACE_LMHEAD_CPU_PROBE=1`
  - `GRETA_TRACE_PREFILL_DECODE_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_13_delta.jsonl`
- Prompts:
  - `p4_sys` (system+user template)
  - `p5_ba` (Buenos Aires)

## Evidencia (ES)
Tabla prefill_last vs decode0 (top1/top2/gap):

| Prompt | prefill_last top1 | decode0 top1 | gap prefill | gap decode0 |
|---|---:|---:|---:|---:|
| p4_sys | 79 | 96965 | 0.9936 | 1.4103 |
| p5_ba  | 79 | 96965 | 0.9929 | 1.4166 |

Extractos JSONL (máx 2 líneas por prompt):
```
{"phase":"prefill_last","step":0,"tokens_total":205,"seq_len":205,"pos_id":204,"hidden_hash":14305319198029099000,"hidden_min":-984.791,"hidden_max":3861.65,"hidden_mean":2063.56,"rms_hash":2109987265846918650,"rms_min":-1.10202,"rms_max":4.32661,"rms_mean":2.20968,"rms_sumsq":26284.2,"rms_eps":1e-05,"rms_weight_dtype":"unknown","rms_input_dtype":"FP32","logits_hash":10258821295820573148,"top1_id":79,"top1_logit":12.0042,"top2_id":18,"top2_logit":11.0106,"gap":0.993584,"lm_head_route":"MFMA","lm_head_quant_mode":"FP16","lm_head_layout_used":"N x K (row_major)","lm_head_dtype_a":0,"lm_head_dtype_b":1,"lm_head_accum_dtype":0,"lm_head_perhead_enabled":true,"lm_head_scales_ptr":0,"lm_head_scales_hash":0,"lm_head_head_scales_ptr":0,"lm_head_head_scales_hash":0,"cpu_probe_top1_id":127158,"cpu_probe_agrees_gpu":false,"cpu_probe_prefill_top1":127158,"cpu_probe_decode0_top1":96965}
{"phase":"decode0","step":1,"tokens_total":206,"seq_len":1,"pos_id":205,"hidden_hash":12423339041135058718,"hidden_min":-14515.1,"hidden_max":5624.58,"hidden_mean":3839.34,"rms_hash":6402361988135106224,"rms_min":-7.59063,"rms_max":3.52933,"rms_mean":2.4467,"rms_sumsq":26432.2,"rms_eps":1e-05,"rms_weight_dtype":"unknown","rms_input_dtype":"FP32","logits_hash":11245132921339454899,"top1_id":96965,"top1_logit":13.3366,"top2_id":198,"top2_logit":11.9263,"gap":1.41033,"lm_head_route":"VALU","lm_head_quant_mode":"FP16","lm_head_layout_used":"N x K (row_major)","lm_head_dtype_a":0,"lm_head_dtype_b":1,"lm_head_accum_dtype":0,"lm_head_perhead_enabled":true,"lm_head_scales_ptr":0,"lm_head_scales_hash":0,"lm_head_head_scales_ptr":0,"lm_head_head_scales_hash":0,"cpu_probe_top1_id":96965,"cpu_probe_agrees_gpu":true,"cpu_probe_prefill_top1":127158,"cpu_probe_decode0_top1":96965}
{"phase":"prefill_last","step":0,"tokens_total":45,"seq_len":45,"pos_id":44,"hidden_hash":16785816201846341193,"hidden_min":-983.323,"hidden_max":3858.82,"hidden_mean":2060.49,"rms_hash":12618115530950980002,"rms_min":-1.10202,"rms_max":4.32718,"rms_mean":2.20967,"rms_sumsq":26284.1,"rms_eps":1e-05,"rms_weight_dtype":"unknown","rms_input_dtype":"FP32","logits_hash":9755417487457510791,"top1_id":79,"top1_logit":11.9995,"top2_id":18,"top2_logit":11.0066,"gap":0.992851,"lm_head_route":"MFMA","lm_head_quant_mode":"FP16","lm_head_layout_used":"N x K (row_major)","lm_head_dtype_a":0,"lm_head_dtype_b":1,"lm_head_accum_dtype":0,"lm_head_perhead_enabled":true,"lm_head_scales_ptr":0,"lm_head_scales_hash":0,"lm_head_head_scales_ptr":0,"lm_head_head_scales_hash":0,"cpu_probe_top1_id":127158,"cpu_probe_agrees_gpu":false,"cpu_probe_prefill_top1":127158,"cpu_probe_decode0_top1":96965}
{"phase":"decode0","step":1,"tokens_total":46,"seq_len":1,"pos_id":45,"hidden_hash":139466943382835816,"hidden_min":-14585.5,"hidden_max":5634.72,"hidden_mean":3849.96,"rms_hash":6346392189903066248,"rms_min":-7.60584,"rms_max":3.52812,"rms_mean":2.44654,"rms_sumsq":26432.2,"rms_eps":1e-05,"rms_weight_dtype":"unknown","rms_input_dtype":"FP32","logits_hash":3369302181601321463,"top1_id":96965,"top1_logit":13.3381,"top2_id":198,"top2_logit":11.9216,"gap":1.41656,"lm_head_route":"VALU","lm_head_quant_mode":"FP16","lm_head_layout_used":"N x K (row_major)","lm_head_dtype_a":0,"lm_head_dtype_b":1,"lm_head_accum_dtype":0,"lm_head_perhead_enabled":true,"lm_head_scales_ptr":0,"lm_head_scales_hash":0,"lm_head_head_scales_ptr":0,"lm_head_head_scales_hash":0,"cpu_probe_top1_id":96965,"cpu_probe_agrees_gpu":true,"cpu_probe_prefill_top1":127158,"cpu_probe_decode0_top1":96965}
```

## Evidence (EN)
Prefill_last vs decode0 table (top1/top2/gap):

| Prompt | prefill_last top1 | decode0 top1 | gap prefill | gap decode0 |
|---|---:|---:|---:|---:|
| p4_sys | 79 | 96965 | 0.9936 | 1.4103 |
| p5_ba  | 79 | 96965 | 0.9929 | 1.4166 |

JSONL excerpts (max 2 lines per prompt):
```
{"phase":"prefill_last","step":0,"tokens_total":205,"seq_len":205,"pos_id":204,"hidden_hash":14305319198029099000,"hidden_min":-984.791,"hidden_max":3861.65,"hidden_mean":2063.56,"rms_hash":2109987265846918650,"rms_min":-1.10202,"rms_max":4.32661,"rms_mean":2.20968,"rms_sumsq":26284.2,"rms_eps":1e-05,"rms_weight_dtype":"unknown","rms_input_dtype":"FP32","logits_hash":10258821295820573148,"top1_id":79,"top1_logit":12.0042,"top2_id":18,"top2_logit":11.0106,"gap":0.993584,"lm_head_route":"MFMA","lm_head_quant_mode":"FP16","lm_head_layout_used":"N x K (row_major)","lm_head_dtype_a":0,"lm_head_dtype_b":1,"lm_head_accum_dtype":0,"lm_head_perhead_enabled":true,"lm_head_scales_ptr":0,"lm_head_scales_hash":0,"lm_head_head_scales_ptr":0,"lm_head_head_scales_hash":0,"cpu_probe_top1_id":127158,"cpu_probe_agrees_gpu":false,"cpu_probe_prefill_top1":127158,"cpu_probe_decode0_top1":96965}
{"phase":"decode0","step":1,"tokens_total":206,"seq_len":1,"pos_id":205,"hidden_hash":12423339041135058718,"hidden_min":-14515.1,"hidden_max":5624.58,"hidden_mean":3839.34,"rms_hash":6402361988135106224,"rms_min":-7.59063,"rms_max":3.52933,"rms_mean":2.4467,"rms_sumsq":26432.2,"rms_eps":1e-05,"rms_weight_dtype":"unknown","rms_input_dtype":"FP32","logits_hash":11245132921339454899,"top1_id":96965,"top1_logit":13.3366,"top2_id":198,"top2_logit":11.9263,"gap":1.41033,"lm_head_route":"VALU","lm_head_quant_mode":"FP16","lm_head_layout_used":"N x K (row_major)","lm_head_dtype_a":0,"lm_head_dtype_b":1,"lm_head_accum_dtype":0,"lm_head_perhead_enabled":true,"lm_head_scales_ptr":0,"lm_head_scales_hash":0,"lm_head_head_scales_ptr":0,"lm_head_head_scales_hash":0,"cpu_probe_top1_id":96965,"cpu_probe_agrees_gpu":true,"cpu_probe_prefill_top1":127158,"cpu_probe_decode0_top1":96965}
{"phase":"prefill_last","step":0,"tokens_total":45,"seq_len":45,"pos_id":44,"hidden_hash":16785816201846341193,"hidden_min":-983.323,"hidden_max":3858.82,"hidden_mean":2060.49,"rms_hash":12618115530950980002,"rms_min":-1.10202,"rms_max":4.32718,"rms_mean":2.20967,"rms_sumsq":26284.1,"rms_eps":1e-05,"rms_weight_dtype":"unknown","rms_input_dtype":"FP32","logits_hash":9755417487457510791,"top1_id":79,"top1_logit":11.9995,"top2_id":18,"top2_logit":11.0066,"gap":0.992851,"lm_head_route":"MFMA","lm_head_quant_mode":"FP16","lm_head_layout_used":"N x K (row_major)","lm_head_dtype_a":0,"lm_head_dtype_b":1,"lm_head_accum_dtype":0,"lm_head_perhead_enabled":true,"lm_head_scales_ptr":0,"lm_head_scales_hash":0,"lm_head_head_scales_ptr":0,"lm_head_head_scales_hash":0,"cpu_probe_top1_id":127158,"cpu_probe_agrees_gpu":false,"cpu_probe_prefill_top1":127158,"cpu_probe_decode0_top1":96965}
{"phase":"decode0","step":1,"tokens_total":46,"seq_len":1,"pos_id":45,"hidden_hash":139466943382835816,"hidden_min":-14585.5,"hidden_max":5634.72,"hidden_mean":3849.96,"rms_hash":6346392189903066248,"rms_min":-7.60584,"rms_max":3.52812,"rms_mean":2.44654,"rms_sumsq":26432.2,"rms_eps":1e-05,"rms_weight_dtype":"unknown","rms_input_dtype":"FP32","logits_hash":3369302181601321463,"top1_id":96965,"top1_logit":13.3381,"top2_id":198,"top2_logit":11.9216,"gap":1.41656,"lm_head_route":"VALU","lm_head_quant_mode":"FP16","lm_head_layout_used":"N x K (row_major)","lm_head_dtype_a":0,"lm_head_dtype_b":1,"lm_head_accum_dtype":0,"lm_head_perhead_enabled":true,"lm_head_scales_ptr":0,"lm_head_scales_hash":0,"lm_head_head_scales_ptr":0,"lm_head_head_scales_hash":0,"cpu_probe_top1_id":96965,"cpu_probe_agrees_gpu":true,"cpu_probe_prefill_top1":127158,"cpu_probe_decode0_top1":96965}
```

## Interpretación (ES)
Hipótesis ordenadas por evidencia:
1. La ruta de LM head cambia MFMA (prefill) → VALU (decode0) y el CPU probe no coincide en prefill pero sí en decode0, lo que sugiere inconsistencia en el path MFMA del LM head.
2. RMSNorm no aparece como causa primaria porque el CPU probe en decode0 valida los logits GPU con el mismo hidden/RMS; el problema parece específico de la ruta de cómputo del LM head en prefill.
3. Si persiste: forzar MFMA vs VALU y auditar scales/per-head en MFMA.

## Interpretation (EN)
Evidence-ordered hypotheses:
1. LM head route flips MFMA (prefill) → VALU (decode0), and the CPU probe disagrees on prefill but agrees on decode0, pointing to an MFMA LM head path inconsistency.
2. RMSNorm is unlikely the primary cause because decode0 CPU probe validates GPU logits for the same hidden/RMS; the issue looks specific to the LM head compute path in prefill.
3. If it persists: force MFMA vs VALU and audit scales/per-head in MFMA.

## Próximo Paso / Next Step (B3.14)
B3.14: forzar LM head a VALU/MFMA por separado y comparar logits prefill vs decode para aislar el bug en la ruta MFMA (scales/layout).

---
L.E.T / Leandro Emanuel Timberini
