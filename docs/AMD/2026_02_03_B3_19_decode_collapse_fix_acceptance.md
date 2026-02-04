# B3.19 — Decode Collapse Fix (Attention Decode seq_len)

## Resumen Ejecutivo (ES)
Se aplicó un fix mínimo en el **decode attention**: el kernel ahora usa `seq_len = pos + 1` (no `pos`) para incluir el token actual en la atención causal. El objetivo es evitar el colapso inmediato en `decode0` (top1=96965) y alinear `prefill_last_top1` con `decode0_top1`.

## Executive Summary (EN)
We applied a minimal fix in **decode attention**: the kernel now uses `seq_len = pos + 1` (not `pos`) to include the current token in causal attention. The goal is to prevent immediate collapse at `decode0` (top1=96965) and align `prefill_last_top1` with `decode0_top1`.

## Cambio Implementado (ES)
- `flash_attention_decode_kernel_p`: `seq_len = *pos_ptr + 1`.

## Implemented Change (EN)
- `flash_attention_decode_kernel_p`: `seq_len = *pos_ptr + 1`.

## Metodología / Methodology
- Modelo: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
- Flags base:
  - `GRETA_INT4_WEIGHTS=1`
  - `GRETA_MAX_SEQ_LEN=256`
  - `GRETA_TRACE_PREFILL_DECODE_DELTA=1`
  - `GRETA_TRACE_RMS_VERIFY=1`
  - `GRETA_TRACE_LMHEAD_CPU_PROBE=1`
  - `GRETA_TRACE_HIDDEN_EQUIV=1`
  - `GRETA_TRACE_LAYER_DELTA=1`
- Prompts:
  - `p4_sys`
  - `p5_ba`

## Resultados (ES)
Pendiente de ejecución en MI300X (acceptance p4/p5). Se completará con:
- Tabla `prefill_last_top1` vs `decode0_top1`
- `readout_mismatch` y `cpu_probe_agrees_gpu`
- Evidencia JSONL/logs

## Results (EN)
Pending MI300X execution (acceptance p4/p5). Will be completed with:
- `prefill_last_top1` vs `decode0_top1` table
- `readout_mismatch` and `cpu_probe_agrees_gpu`
- JSONL/log evidence

---
L.E.T / Leandro Emanuel Timberini
