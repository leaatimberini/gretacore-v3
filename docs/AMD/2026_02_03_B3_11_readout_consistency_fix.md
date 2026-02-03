# B3.11 — Readout Consistency (Prefill→Decode) Fix + Trace

**Fecha:** 2026-02-03
**Objetivo:** alinear el readout (prefill→decode) y aislar offsets/stride en D2H para logits/hidden, con trazas JSONL detalladas y corrección mínima del offset de logits en decode.

---

## Resumen Ejecutivo (ES)
En B3.10 se observó que `prefill_last_top1` difiere de `decode0_top1` (ej. 79→96965) con prompts largos. Se sospecha un bug residual en el readout de decode (slice/offset) o en el buffer de logits. En B3.11 se agregaron trazas de consistencia y se corrigió la ubicación del output de LM Head en decode para que el logits buffer respete el índice lógico del token (`seq_start`), eliminando offsets ambiguos.

## Executive Summary (EN)
B3.10 showed `prefill_last_top1 != decode0_top1` (e.g., 79→96965) for long prompts. A residual decode readout issue (slice/offset) or logits buffer placement was suspected. B3.11 adds readout-consistency traces and fixes LM Head logits placement in decode so the logits buffer honors the logical token index (`seq_start`), removing ambiguous offsets.

---

## Problema Observado (ES)
- Prefill último token: top1 estable (ej. 79) en prompts largos.
- Decode paso 0: top1 colapsa a 96965 y queda fijo.
- Logs previos mostraron offsets D2H diferentes entre prefill y decode.

## Observed Issue (EN)
- Prefill last token: stable top1 (e.g., 79) on long prompts.
- Decode step 0: top1 collapses to 96965 and stays fixed.
- Prior logs showed D2H offsets differing between prefill and decode.

---

## Hipótesis (ES)
- El slice/offset de logits en decode no respeta el índice lógico del token.
- El readout de hidden/logits no está alineado con `tokens_total-1`.

## Hypothesis (EN)
- Decode logits slice/offset does not respect the logical token index.
- Readout of hidden/logits is misaligned vs `tokens_total-1`.

---

## Cambios Implementados (ES)
1. **Trace JSONL de consistencia** bajo `GRETA_TRACE_READOUT=1` con:
   - `tokens_total`, `seq_len`, `pos_id`.
   - `hidden_src_ptr`, `hidden_alloc_bytes`, `hidden_stride_bytes`, `hidden_offset_bytes`.
   - `hidden_token_index_used`, `expected_last_index`, `readout_mismatch`.
   - `rms_in_ptr`, `rms_out_ptr`, `lm_in_ptr`, `rms_hash`.
   - `logits_ptr`, `logits_offset_bytes`, `logits_hash`, `top1/top2/gap`.
2. **Fix LM Head (decode):** la salida de logits ahora se escribe en el offset lógico `seq_start * vocab`, para que el buffer sea consistente con el índice de token.
3. **Sample greedy GPU:** usa `logits_offset_bytes` para leer el slot correcto.

## Implemented Changes (EN)
1. **Readout consistency JSONL trace** under `GRETA_TRACE_READOUT=1` with:
   - `tokens_total`, `seq_len`, `pos_id`.
   - `hidden_src_ptr`, `hidden_alloc_bytes`, `hidden_stride_bytes`, `hidden_offset_bytes`.
   - `hidden_token_index_used`, `expected_last_index`, `readout_mismatch`.
   - `rms_in_ptr`, `rms_out_ptr`, `lm_in_ptr`, `rms_hash`.
   - `logits_ptr`, `logits_offset_bytes`, `logits_hash`, `top1/top2/gap`.
2. **LM Head fix (decode):** logits output now writes at logical offset `seq_start * vocab` so the buffer matches token index.
3. **Greedy GPU sample:** uses `logits_offset_bytes` to read the correct slot.

---

## Reproducción (ES)
```bash
# Build local y luego run en MI300X
export GRETA_INT4_WEIGHTS=1
export GRETA_MAX_SEQ_LEN=256
export GRETA_TRACE_EMBED_VERIFY=1
export GRETA_EMBED_LAYOUT=row
export GRETA_TRACE_READOUT=1
export GRETA_TRACE_READOUT_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_11_readout_consistency.jsonl

BIN=/root/gretacore/tools/inference/build/greta_infer
MODEL=/root/gretacore/models/llama3_8b_q4/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf

$BIN --model "$MODEL" --prompt "<PROMPT>" --max-tokens 16 --greedy --debug-decode 4
```

## Reproduction (EN)
```bash
# Build locally and run on MI300X
export GRETA_INT4_WEIGHTS=1
export GRETA_MAX_SEQ_LEN=256
export GRETA_TRACE_EMBED_VERIFY=1
export GRETA_EMBED_LAYOUT=row
export GRETA_TRACE_READOUT=1
export GRETA_TRACE_READOUT_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_11_readout_consistency.jsonl

BIN=/root/gretacore/tools/inference/build/greta_infer
MODEL=/root/gretacore/models/llama3_8b_q4/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf

$BIN --model "$MODEL" --prompt "<PROMPT>" --max-tokens 16 --greedy --debug-decode 4
```

---

## Evidencia (MI300X)

Tabla resumen:
| Prompt | prefill_last_top1 | decode0_top1 | readout_mismatch |
| --- | --- | --- | --- |
| p4_sys | 79 | 96965 | true |
| p5_ba | 79 | 96965 | true |

Extracto JSONL (`b3_11_readout_consistency.jsonl`):
```json
{"phase":"prefill_last","step":0,"tokens_total":205,"seq_len":205,"pos_id":204,"token_index":204,"expected_last_index":204,"hidden_token_index_used":204,"readout_mismatch":false,"hidden_src_ptr":126375807680512,"hidden_alloc_bytes":4194304,"hidden_stride_bytes":16384,"hidden_offset_bytes":3342336,"hidden_hash":14305319198029099000,"hidden_min":-984.791,"hidden_max":3861.65,"hidden_mean":2063.56,"rms_in_ptr":126375807680512,"rms_out_ptr":126375801389056,"lm_in_ptr":126375801389056,"rms_offset_bytes":3342336,"rms_hash":2109987265846918650,"rms_min":-1.10202,"rms_max":4.32661,"rms_mean":2.20968,"logits_offset_bytes":104656896,"logits_ptr":126356088160256,"logits_hash":10258821295820573148,"logits_min":-13.492,"logits_max":12.0042,"logits_mean":-0.582097,"top1_id":79,"top1_logit":12.0042,"top2_id":18,"top2_logit":11.0106,"gap":0.993584,"vocab":128256}
{"phase":"decode","step":1,"tokens_total":206,"seq_len":1,"pos_id":205,"token_index":205,"expected_last_index":205,"hidden_token_index_used":0,"readout_mismatch":true,"hidden_src_ptr":126375807680512,"hidden_alloc_bytes":4194304,"hidden_stride_bytes":16384,"hidden_offset_bytes":0,"hidden_hash":12423339041135058718,"hidden_min":-14515.1,"hidden_max":5624.58,"hidden_mean":3839.34,"rms_in_ptr":126375807680512,"rms_out_ptr":126375801389056,"lm_in_ptr":126375801389056,"rms_offset_bytes":0,"rms_hash":6402361988135106224,"rms_min":-7.59063,"rms_max":3.52933,"rms_mean":2.4467,"logits_offset_bytes":105169920,"logits_ptr":126356088160256,"logits_hash":11245132921339454899,"logits_min":-13.0573,"logits_max":13.3366,"logits_mean":-0.459103,"top1_id":96965,"top1_logit":13.3366,"top2_id":198,"top2_logit":11.9263,"gap":1.41033,"vocab":128256}
{"phase":"prefill_last","step":0,"tokens_total":45,"seq_len":45,"pos_id":44,"token_index":44,"expected_last_index":44,"hidden_token_index_used":44,"readout_mismatch":false,"hidden_src_ptr":123265295056896,"hidden_alloc_bytes":4194304,"hidden_stride_bytes":16384,"hidden_offset_bytes":720896,"hidden_hash":16785816201846341193,"hidden_min":-983.323,"hidden_max":3858.82,"hidden_mean":2060.49,"rms_in_ptr":123265295056896,"rms_out_ptr":123265288765440,"lm_in_ptr":123265288765440,"rms_offset_bytes":720896,"rms_hash":12618115530950980002,"rms_min":-1.10202,"rms_max":4.32718,"rms_mean":2.20967,"logits_offset_bytes":22573056,"logits_ptr":123265026621440,"logits_hash":9755417487457510791,"logits_min":-13.4885,"logits_max":11.9995,"logits_mean":-0.582035,"top1_id":79,"top1_logit":11.9995,"top2_id":18,"top2_logit":11.0066,"gap":0.992851,"vocab":128256}
```

Extracto logs (`Top tokens`):
```
b3_11_p4_sys.log
  Top tokens: 79(12.0042) 18(11.0106) 95(10.7514) 64938(10.7406) 59036(9.7833)
  Top tokens: 96965(13.3366) 198(11.9263) 99668(11.4669) 52263(11.185) 17309(10.4398)

b3_11_p5_ba.log
  Top tokens: 79(11.9995) 18(11.0066) 95(10.7475) 64938(10.7399) 59036(9.78043)
  Top tokens: 96965(13.3381) 198(11.9216) 99668(11.465) 52263(11.1899) 17309(10.4357)
```

## Evidence (MI300X)

Summary table:
| Prompt | prefill_last_top1 | decode0_top1 | readout_mismatch |
| --- | --- | --- | --- |
| p4_sys | 79 | 96965 | true |
| p5_ba | 79 | 96965 | true |

JSONL snippet (`b3_11_readout_consistency.jsonl`):
```json
{"phase":"prefill_last","step":0,"tokens_total":205,"seq_len":205,"pos_id":204,"token_index":204,"expected_last_index":204,"hidden_token_index_used":204,"readout_mismatch":false,"hidden_src_ptr":126375807680512,"hidden_alloc_bytes":4194304,"hidden_stride_bytes":16384,"hidden_offset_bytes":3342336,"hidden_hash":14305319198029099000,"hidden_min":-984.791,"hidden_max":3861.65,"hidden_mean":2063.56,"rms_in_ptr":126375807680512,"rms_out_ptr":126375801389056,"lm_in_ptr":126375801389056,"rms_offset_bytes":3342336,"rms_hash":2109987265846918650,"rms_min":-1.10202,"rms_max":4.32661,"rms_mean":2.20968,"logits_offset_bytes":104656896,"logits_ptr":126356088160256,"logits_hash":10258821295820573148,"logits_min":-13.492,"logits_max":12.0042,"logits_mean":-0.582097,"top1_id":79,"top1_logit":12.0042,"top2_id":18,"top2_logit":11.0106,"gap":0.993584,"vocab":128256}
{"phase":"decode","step":1,"tokens_total":206,"seq_len":1,"pos_id":205,"token_index":205,"expected_last_index":205,"hidden_token_index_used":0,"readout_mismatch":true,"hidden_src_ptr":126375807680512,"hidden_alloc_bytes":4194304,"hidden_stride_bytes":16384,"hidden_offset_bytes":0,"hidden_hash":12423339041135058718,"hidden_min":-14515.1,"hidden_max":5624.58,"hidden_mean":3839.34,"rms_in_ptr":126375807680512,"rms_out_ptr":126375801389056,"lm_in_ptr":126375801389056,"rms_offset_bytes":0,"rms_hash":6402361988135106224,"rms_min":-7.59063,"rms_max":3.52933,"rms_mean":2.4467,"logits_offset_bytes":105169920,"logits_ptr":126356088160256,"logits_hash":11245132921339454899,"logits_min":-13.0573,"logits_max":13.3366,"logits_mean":-0.459103,"top1_id":96965,"top1_logit":13.3366,"top2_id":198,"top2_logit":11.9263,"gap":1.41033,"vocab":128256}
{"phase":"prefill_last","step":0,"tokens_total":45,"seq_len":45,"pos_id":44,"token_index":44,"expected_last_index":44,"hidden_token_index_used":44,"readout_mismatch":false,"hidden_src_ptr":123265295056896,"hidden_alloc_bytes":4194304,"hidden_stride_bytes":16384,"hidden_offset_bytes":720896,"hidden_hash":16785816201846341193,"hidden_min":-983.323,"hidden_max":3858.82,"hidden_mean":2060.49,"rms_in_ptr":123265295056896,"rms_out_ptr":123265288765440,"lm_in_ptr":123265288765440,"rms_offset_bytes":720896,"rms_hash":12618115530950980002,"rms_min":-1.10202,"rms_max":4.32718,"rms_mean":2.20967,"logits_offset_bytes":22573056,"logits_ptr":123265026621440,"logits_hash":9755417487457510791,"logits_min":-13.4885,"logits_max":11.9995,"logits_mean":-0.582035,"top1_id":79,"top1_logit":11.9995,"top2_id":18,"top2_logit":11.0066,"gap":0.992851,"vocab":128256}
```

Log excerpts (`Top tokens`):
```
b3_11_p4_sys.log
  Top tokens: 79(12.0042) 18(11.0106) 95(10.7514) 64938(10.7406) 59036(9.7833)
  Top tokens: 96965(13.3366) 198(11.9263) 99668(11.4669) 52263(11.185) 17309(10.4398)

b3_11_p5_ba.log
  Top tokens: 79(11.9995) 18(11.0066) 95(10.7475) 64938(10.7399) 59036(9.78043)
  Top tokens: 96965(13.3381) 198(11.9216) 99668(11.465) 52263(11.1899) 17309(10.4357)
```

---

## Resultado / Next
- **NO OK**: `prefill_last_top1 != decode0_top1` en p4/p5 y `readout_mismatch=true` en decode.
- Próximo paso sugerido: auditar RMSNorm final + LM head dequant/scales y validar si el hidden usado en decode debe reflejar el índice lógico del token (o si el colapso proviene de normalización/escala).

## Result / Next
- **NO OK**: `prefill_last_top1 != decode0_top1` on p4/p5 and `readout_mismatch=true` in decode.
- Suggested next step: audit final RMSNorm + LM head dequant/scales and confirm whether decode hidden should reflect the logical token index (or if collapse is driven by normalization/scale).

---

**Firma:**
L.E.T / Leandro Emanuel Timberini
