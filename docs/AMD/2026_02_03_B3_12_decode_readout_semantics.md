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

## Evidencia (pendiente de run MI300X)
- Tabla p4/p5: `prefill_last_top1` vs `decode0_top1`.
- Conteo de `readout_mismatch=true`.
- Extracto JSONL con `readout_buffer_kind`, `used_index`, `logical_last_index`.

## Evidence (pending MI300X run)
- Table p4/p5: `prefill_last_top1` vs `decode0_top1`.
- Count of `readout_mismatch=true`.
- JSONL snippet with `readout_buffer_kind`, `used_index`, `logical_last_index`.

---

## Resultado / Next (TBD)
- **OK** si `prefill_last_top1 == decode0_top1` y `readout_mismatch=false`.
- **NO OK** → pasar a B3.13 (RMSNorm final + LM head dequant/layout).

---

**Firma:**
L.E.T / Leandro Emanuel Timberini
