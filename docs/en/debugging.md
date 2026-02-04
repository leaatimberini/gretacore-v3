# GRETA CORE – Debugging (LLM Pipeline)

Version: 1.0  
Status: Active  
Project Phase: Phase 3 – LLM Inference Pipeline (B3.x)  
Language: English

---

## Goal
Provide a reproducible debugging flow to isolate errors in the LLM inference pipeline, focusing on prefill↔decode coherence and validation of compute routes (LM head, attention, KV cache).

## Principles
- Local-first: all changes land in the local repo before MI300X validation.
- Minimal traces gated by flags.
- Evidence always exported as JSONL and documented in `docs/AMD/`.

## Key Flags (B3.x)
- `GRETA_TRACE_PREFILL_DECODE_DELTA=1`  
  Emits JSONL with `prefill_last` and `decode0` (hash/stats/top1/top2/gap + LM head audit).
- `GRETA_TRACE_LMHEAD_CPU_PROBE=1`  
  Runs a minimal CPU matvec to validate GPU↔CPU coherence.
- `GRETA_TRACE_HIDDEN_EQUIV=1`  
  Records hidden equivalence prefill↔decode.
- `GRETA_TRACE_LAYER_DELTA=1`  
  Hash/stats of `attn_out`, `mlp_out`, `x_out` for layer 0 and last (decode).
- `GRETA_LMHEAD_FORCE_ROUTE=valu|mfma`  
  Forces LM head route (prefill+decode) to isolate MFMA/VALU.
- `GRETA_LMHEAD_FORCE_ROUTE_DECODE=valu|mfma`  
  Forces LM head route **decode-only**.

## Recommended Debug Flow
1. **Baseline**: run p4_sys and p5_ba with delta traces.
2. **Isolate LM head**: force MFMA/VALU and compare `cpu_probe_agrees_gpu`.
3. **Hidden equivalence**: verify divergence between `prefill_last` and `decode0`.
4. **Layer delta**: inspect early divergence (layer 0 / last).
5. **Attention/KV**: if LM head is consistent, audit decode attention and cache.

## Evidence & Reporting
- JSONL and logs must be packed and downloaded locally.
- Each block produces an AMD report under `docs/AMD/` with ES/EN, tables, and excerpts.

---
L.E.T / Leandro Emanuel Timberini
