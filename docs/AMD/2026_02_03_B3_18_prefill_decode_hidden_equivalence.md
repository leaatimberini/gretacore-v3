# B3.18 — Prefill vs Decode Hidden Equivalence + Layer Delta

## Resumen Ejecutivo (ES)
Se agregaron trazas para verificar si `decode0` parte del mismo estado lógico que `prefill_last` y un delta por capa (0 y última) durante decode. Los resultados muestran **hashes distintos** entre prefill y decode y deltas por capa presentes en 0 y 31.

## Executive Summary (EN)
We added traces to verify whether `decode0` starts from the same logical state as `prefill_last` and a per-layer delta (layer 0 and last) during decode. Results show **different hashes** between prefill and decode, with layer deltas captured at layers 0 and 31.

## Metodología / Methodology
- Flags:
  - `GRETA_TRACE_PREFILL_DECODE_DELTA=1`
  - `GRETA_TRACE_HIDDEN_EQUIV=1`
  - `GRETA_TRACE_LAYER_DELTA=1`
- Output:
  - `/root/gretacore/artifacts/alignment/2026-02-03/b3_17_*_decode_*.jsonl`

## Resultados (ES)
Hidden equivalence (prefill_hash → decode_hash):

| Prompt | prefill_hash | decode_hash |
|---|---:|---:|
| p4_sys | 14305319198029099000 | 16786868500833975907 |
| p5_ba  | 16785816201846341193 | 1662617505111799027 |

Layer delta (decode):
- Se registraron `layer_delta` para **layer 0** y **layer 31** en ambas corridas.

## Results (EN)
Hidden equivalence (prefill_hash → decode_hash):

| Prompt | prefill_hash | decode_hash |
|---|---:|---:|
| p4_sys | 14305319198029099000 | 16786868500833975907 |
| p5_ba  | 16785816201846341193 | 1662617505111799027 |

Layer delta (decode):
- `layer_delta` events were captured for **layer 0** and **layer 31** in both runs.

## Conclusión / Conclusion
`decode0` no reutiliza el mismo hidden que `prefill_last`; hay divergencia temprana en el pipeline de decode. Esto apoya un fix en la ruta de decode antes del LM head (B3.19).

---
L.E.T / Leandro Emanuel Timberini
