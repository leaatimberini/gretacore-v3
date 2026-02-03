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

## Evidencia (pendiente de run MI300X)
- Tabla `prefill_last_top1` vs `decode0_top1` (p4_sys / p5_ba).
- Extracto JSONL `b3_11_readout_consistency.jsonl` (2–3 líneas).
- Extracto logs `b3_11_p4_sys.log`, `b3_11_p5_ba.log`.

## Evidence (pending MI300X run)
- Table `prefill_last_top1` vs `decode0_top1` (p4_sys / p5_ba).
- JSONL snippet `b3_11_readout_consistency.jsonl` (2–3 lines).
- Log extracts `b3_11_p4_sys.log`, `b3_11_p5_ba.log`.

---

## Resultado / Next (TBD)
- **OK** si `prefill_last_top1 == decode0_top1` en p4/p5 y `readout_mismatch=false`.
- **NO OK** si decode0 vuelve a 96965 con prefill_last diferente → auditar RMSNorm / LM head dequant / scales.

---

**Firma:**
L.E.T / Leandro Emanuel Timberini
