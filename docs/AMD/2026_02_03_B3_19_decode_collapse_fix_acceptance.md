# B3.19 — Decode Collapse Fix (Attention Decode seq_len)

## Resumen Ejecutivo (ES)
Se aplicó un fix mínimo en el **decode attention**: el kernel ahora usa `seq_len = pos + 1` (no `pos`) para incluir el token actual en la atención causal. La aceptación se ejecutó en p4_sys y p5_ba.

**Resultado:** el colapso **persiste** (`decode0_top1=96965`). El CPU probe concuerda con GPU en decode, por lo que el LM head no es la causa principal. El problema sigue en la ruta de decode antes del LM head (atención/KV/estado).

## Executive Summary (EN)
We applied a minimal fix in **decode attention**: the kernel now uses `seq_len = pos + 1` (not `pos`) to include the current token in causal attention. Acceptance ran on p4_sys and p5_ba.

**Result:** collapse **persists** (`decode0_top1=96965`). CPU probe agrees with GPU in decode, so the LM head is not the primary cause. The issue remains in the decode path before the LM head (attention/KV/state).

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
Tabla (prefill_last vs decode0):

| Prompt | prefill_last top1 | decode0 top1 | cpu_probe prefill | cpu_probe decode0 |
|---|---:|---:|---|---|
| p4_sys | 127158 | 96965 | true | true |
| p5_ba  | 127158 | 96965 | true | true |

Hidden equivalence (prefill_hash → decode_hash):

| Prompt | prefill_hash | decode_hash |
|---|---:|---:|
| p4_sys | 14305319198029099000 | 4459200984117844040 |
| p5_ba  | 16785816201846341193 | 17322585814358657131 |

## Results (EN)
Table (prefill_last vs decode0):

| Prompt | prefill_last top1 | decode0 top1 | cpu_probe prefill | cpu_probe decode0 |
|---|---:|---:|---|---|
| p4_sys | 127158 | 96965 | true | true |
| p5_ba  | 127158 | 96965 | true | true |

Hidden equivalence (prefill_hash → decode_hash):

| Prompt | prefill_hash | decode_hash |
|---|---:|---:|
| p4_sys | 14305319198029099000 | 4459200984117844040 |
| p5_ba  | 16785816201846341193 | 17322585814358657131 |

## Conclusión / Conclusion
**FIX FAIL.** El ajuste de `seq_len = pos + 1` no elimina el colapso. La evidencia sugiere un problema en el pipeline de decode antes del LM head, probablemente en atención decode o KV cache/estado.

**Next Step (B3.20):** auditar `attention_decode` y `kv_cache` con comparación directa prefill vs decode (mismatches de Q/K/V y escala), o forzar una ruta de atención alternativa validada.

---
L.E.T / Leandro Emanuel Timberini
