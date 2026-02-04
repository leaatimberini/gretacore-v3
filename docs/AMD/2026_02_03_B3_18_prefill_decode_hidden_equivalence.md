# B3.18 — Prefill vs Decode Hidden Equivalence + Layer Delta

## Resumen Ejecutivo (ES)
Se agregaron trazas para verificar si `decode0` parte del mismo estado lógico que `prefill_last`. Incluye un registro explícito de **equivalencia de hidden** y un delta por capa (0 y última) en decode, con hashes y estadísticas mínimas.

## Executive Summary (EN)
We added traces to verify whether `decode0` starts from the same logical state as `prefill_last`. This includes an explicit **hidden equivalence** record and a per-layer delta (layer 0 and last) during decode with hashes and basic stats.

## Cambios Implementados (ES)
- `GRETA_TRACE_HIDDEN_EQUIV=1`: emite una línea JSONL comparando `prefill_last` vs `decode0` (hash + stats).
- `GRETA_TRACE_LAYER_DELTA=1`: registra hashes/estadísticas de `attn_out`, `mlp_out`, `x_out` en layers 0 y last (decode, seq_len=1).
- Output por defecto en el mismo JSONL de prefill/decode (`GRETA_TRACE_PREFILL_DECODE_OUT`).

## Implemented Changes (EN)
- `GRETA_TRACE_HIDDEN_EQUIV=1`: emits a JSONL line comparing `prefill_last` vs `decode0` (hash + stats).
- `GRETA_TRACE_LAYER_DELTA=1`: records `attn_out`, `mlp_out`, `x_out` hashes/stats for layers 0 and last (decode, seq_len=1).
- Output defaults to the same prefill/decode JSONL (`GRETA_TRACE_PREFILL_DECODE_OUT`).

## Metodología / Methodology
- Flags base:
  - `GRETA_TRACE_PREFILL_DECODE_DELTA=1`
  - `GRETA_TRACE_HIDDEN_EQUIV=1`
  - `GRETA_TRACE_LAYER_DELTA=1`
- Output:
  - `GRETA_TRACE_PREFILL_DECODE_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_18_delta.jsonl`

## Resultados (ES)
Pendiente de ejecución en MI300X. Se completará con:
- Línea `hidden_equiv` (prefill vs decode)
- Deltas por layer 0/last (attn_out/mlp_out/x_out)
- Interpretación de divergencia (si existe)

## Results (EN)
Pending MI300X execution. Will be completed with:
- `hidden_equiv` line (prefill vs decode)
- Layer 0/last deltas (attn_out/mlp_out/x_out)
- Divergence interpretation (if any)

## Conclusión / Conclusion
Este bloque permite verificar si el estado de decode proviene del prefill correcto o si hay divergencia temprana por capa.

---
L.E.T / Leandro Emanuel Timberini
