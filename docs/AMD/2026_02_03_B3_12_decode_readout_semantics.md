# B3.12 — Decode Readout Semantics (Logical vs Physical Index)

**Fecha:** 2026-02-03

## Resumen Ejecutivo (ES)
En decode, el buffer de hidden es **single-token** `[1, D]` y el índice físico válido es siempre `0`. El readout anterior comparaba contra el índice lógico `tokens_total-1`, provocando `readout_mismatch` y colapso de `decode0_top1`. En B3.12 se separa explícitamente índice lógico vs físico, se introduce `readout_buffer_kind`, y se fuerza el readout a respetar la semántica correcta de buffer single-token.

## Executive Summary (EN)
In decode, the hidden buffer is **single-token** `[1, D]` and the only valid physical index is `0`. The prior readout compared against the logical index `tokens_total-1`, producing `readout_mismatch` and a collapse of `decode0_top1`. B3.12 explicitly separates logical vs physical indices, introduces `readout_buffer_kind`, and enforces correct single-token readout semantics.

---

## Qué estaba roto (ES)
- En decode: buffer single-token, pero se comparaba `used_index` contra el índice lógico.
- Esto marcaba `readout_mismatch` y contribuía al colapso inmediato de `decode0_top1`.

## What was broken (EN)
- In decode: single-token buffer, but `used_index` was compared against the logical index.
- This flagged `readout_mismatch` and contributed to immediate `decode0_top1` collapse.

---

## Fix aplicado (ES)
- `readout_buffer_kind = "single_token"` en decode.
- `used_index = 0` (índice físico) y `logical_last_index = tokens_total-1`.
- `readout_mismatch` **no** compara contra `used_index` cuando el buffer es single-token.
- `hidden_source_tag = decode_hidden_single_token` para identificar el origen correcto del hidden.

## Fix applied (EN)
- `readout_buffer_kind = "single_token"` in decode.
- `used_index = 0` (physical index) and `logical_last_index = tokens_total-1`.
- `readout_mismatch` **does not** compare against `used_index` when the buffer is single-token.
- `hidden_source_tag = decode_hidden_single_token` to tag the correct hidden source.

---

## Evidencia (MI300X)

Tabla p4/p5:
| Prompt | prefill_last_top1 | decode0_top1 |
| --- | --- | --- |
| p4_sys | 79 | 96965 |
| p5_ba | 79 | 96965 |

Conteo `readout_mismatch=true`: **0**

Extracto JSONL (`b3_12_readout.jsonl`):
```json
{"phase":"prefill_last","readout_buffer_kind":"seq","hidden_source_tag":"prefill_hidden_seq","step":0,"tokens_total":205,"seq_len":205,"pos_id":204,"token_index":204,"used_index":204,"logical_last_index":204,"expected_last_index":204,"hidden_token_index_used":204,"readout_mismatch":false,"hidden_src_ptr":125303806492672,"hidden_alloc_bytes":4194304,"hidden_stride_bytes":16384,"hidden_offset_bytes":3342336,"hidden_hash":14305319198029099000,"hidden_min":-984.791,"hidden_max":3861.65,"hidden_mean":2063.56,"rms_in_ptr":125303806492672,"rms_out_ptr":125303800201216,"lm_in_ptr":125303800201216,"rms_offset_bytes":3342336,"rms_hash":2109987265846918650,"rms_min":-1.10202,"rms_max":4.32661,"rms_mean":2.20968,"logits_offset_bytes":104656896,"logits_ptr":125284086972416,"logits_hash":10258821295820573148,"logits_min":-13.492,"logits_max":12.0042,"logits_mean":-0.582097,"top1_id":79,"top1_logit":12.0042,"top2_id":18,"top2_logit":11.0106,"gap":0.993584,"vocab":128256}
{"phase":"decode","readout_buffer_kind":"single_token","hidden_source_tag":"decode_hidden_single_token","step":1,"tokens_total":206,"seq_len":1,"pos_id":205,"token_index":205,"used_index":0,"logical_last_index":205,"expected_last_index":205,"hidden_token_index_used":0,"readout_mismatch":false,"hidden_src_ptr":125303806492672,"hidden_alloc_bytes":4194304,"hidden_stride_bytes":16384,"hidden_offset_bytes":0,"hidden_hash":12423339041135058718,"hidden_min":-14515.1,"hidden_max":5624.58,"hidden_mean":3839.34,"rms_in_ptr":125303806492672,"rms_out_ptr":125303800201216,"lm_in_ptr":125303800201216,"rms_offset_bytes":0,"rms_hash":6402361988135106224,"rms_min":-7.59063,"rms_max":3.52933,"rms_mean":2.4467,"logits_offset_bytes":105169920,"logits_ptr":125284086972416,"logits_hash":11245132921339454899,"logits_min":-13.0573,"logits_max":13.3366,"logits_mean":-0.459103,"top1_id":96965,"top1_logit":13.3366,"top2_id":198,"top2_logit":11.9263,"gap":1.41033,"vocab":128256}
{"phase":"prefill_last","readout_buffer_kind":"seq","hidden_source_tag":"prefill_hidden_seq","step":0,"tokens_total":45,"seq_len":45,"pos_id":44,"token_index":44,"used_index":44,"logical_last_index":44,"expected_last_index":44,"hidden_token_index_used":44,"readout_mismatch":false,"hidden_src_ptr":130983489699840,"hidden_alloc_bytes":4194304,"hidden_stride_bytes":16384,"hidden_offset_bytes":720896,"hidden_hash":16785816201846341193,"hidden_min":-983.323,"hidden_max":3858.82,"hidden_mean":2060.49,"rms_in_ptr":130983489699840,"rms_out_ptr":130983483408384,"lm_in_ptr":130983483408384,"rms_offset_bytes":720896,"rms_hash":12618115530950980002,"rms_min":-1.10202,"rms_max":4.32718,"rms_mean":2.20967,"logits_offset_bytes":22573056,"logits_ptr":130983225458688,"logits_hash":9755417487457510791,"logits_min":-13.4885,"logits_max":11.9995,"logits_mean":-0.582035,"top1_id":79,"top1_logit":11.9995,"top2_id":18,"top2_logit":11.0066,"gap":0.992851,"vocab":128256}
```

## Evidence (MI300X)

Table p4/p5:
| Prompt | prefill_last_top1 | decode0_top1 |
| --- | --- | --- |
| p4_sys | 79 | 96965 |
| p5_ba | 79 | 96965 |

`readout_mismatch=true` count: **0**

JSONL snippet (`b3_12_readout.jsonl`):
```json
{"phase":"prefill_last","readout_buffer_kind":"seq","hidden_source_tag":"prefill_hidden_seq","step":0,"tokens_total":205,"seq_len":205,"pos_id":204,"token_index":204,"used_index":204,"logical_last_index":204,"expected_last_index":204,"hidden_token_index_used":204,"readout_mismatch":false,"hidden_src_ptr":125303806492672,"hidden_alloc_bytes":4194304,"hidden_stride_bytes":16384,"hidden_offset_bytes":3342336,"hidden_hash":14305319198029099000,"hidden_min":-984.791,"hidden_max":3861.65,"hidden_mean":2063.56,"rms_in_ptr":125303806492672,"rms_out_ptr":125303800201216,"lm_in_ptr":125303800201216,"rms_offset_bytes":3342336,"rms_hash":2109987265846918650,"rms_min":-1.10202,"rms_max":4.32661,"rms_mean":2.20968,"logits_offset_bytes":104656896,"logits_ptr":125284086972416,"logits_hash":10258821295820573148,"logits_min":-13.492,"logits_max":12.0042,"logits_mean":-0.582097,"top1_id":79,"top1_logit":12.0042,"top2_id":18,"top2_logit":11.0106,"gap":0.993584,"vocab":128256}
{"phase":"decode","readout_buffer_kind":"single_token","hidden_source_tag":"decode_hidden_single_token","step":1,"tokens_total":206,"seq_len":1,"pos_id":205,"token_index":205,"used_index":0,"logical_last_index":205,"expected_last_index":205,"hidden_token_index_used":0,"readout_mismatch":false,"hidden_src_ptr":125303806492672,"hidden_alloc_bytes":4194304,"hidden_stride_bytes":16384,"hidden_offset_bytes":0,"hidden_hash":12423339041135058718,"hidden_min":-14515.1,"hidden_max":5624.58,"hidden_mean":3839.34,"rms_in_ptr":125303806492672,"rms_out_ptr":125303800201216,"lm_in_ptr":125303800201216,"rms_offset_bytes":0,"rms_hash":6402361988135106224,"rms_min":-7.59063,"rms_max":3.52933,"rms_mean":2.4467,"logits_offset_bytes":105169920,"logits_ptr":125284086972416,"logits_hash":11245132921339454899,"logits_min":-13.0573,"logits_max":13.3366,"logits_mean":-0.459103,"top1_id":96965,"top1_logit":13.3366,"top2_id":198,"top2_logit":11.9263,"gap":1.41033,"vocab":128256}
{"phase":"prefill_last","readout_buffer_kind":"seq","hidden_source_tag":"prefill_hidden_seq","step":0,"tokens_total":45,"seq_len":45,"pos_id":44,"token_index":44,"used_index":44,"logical_last_index":44,"expected_last_index":44,"hidden_token_index_used":44,"readout_mismatch":false,"hidden_src_ptr":130983489699840,"hidden_alloc_bytes":4194304,"hidden_stride_bytes":16384,"hidden_offset_bytes":720896,"hidden_hash":16785816201846341193,"hidden_min":-983.323,"hidden_max":3858.82,"hidden_mean":2060.49,"rms_in_ptr":130983489699840,"rms_out_ptr":130983483408384,"lm_in_ptr":130983483408384,"rms_offset_bytes":720896,"rms_hash":12618115530950980002,"rms_min":-1.10202,"rms_max":4.32718,"rms_mean":2.20967,"logits_offset_bytes":22573056,"logits_ptr":130983225458688,"logits_hash":9755417487457510791,"logits_min":-13.4885,"logits_max":11.9995,"logits_mean":-0.582035,"top1_id":79,"top1_logit":11.9995,"top2_id":18,"top2_logit":11.0066,"gap":0.992851,"vocab":128256}
```

---

## Resultado / Next
- **NO OK**: `prefill_last_top1 != decode0_top1` en p4/p5, aunque `readout_mismatch=false`.
- Próximo paso: B3.13 (auditar RMSNorm final + LM head dequant/layout y escalas).

---

**Firma:**
L.E.T / Leandro Emanuel Timberini
