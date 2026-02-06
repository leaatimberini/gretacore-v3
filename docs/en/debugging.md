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
- `GRETA_TRACE_ATTN_DECODE_VERIFY=1`  
  Verifies decode attention with a reference (hash/MAE).
- `GRETA_ATTN_DECODE_REF=1`  
  Enables recompute reference on decode0.
- `GRETA_TRACE_ATTN_SOFTMAX=1`  
  Captures decode0 softmax isolation (QK/softmax window vs FP64).
- `GRETA_TRACE_ATTN_LAYER=31`  
  Selects a single layer for softmax isolation (default last).
- `GRETA_TRACE_ATTN_HEAD=0`  
  Selects a single head for softmax isolation.
- `GRETA_TRACE_ATTN_KEYS_WINDOW=64`  
  Window size on each side of the current position for softmax isolation.
- `GRETA_TRACE_ATTN_OUT=/root/gretacore/artifacts/alignment/.../b3_23_attn_softmax.jsonl`  
  JSONL output path for softmax isolation traces.
- `GRETA_TRACE_ATTN_L0_PIPE=1`  
  Traces the layer0 attention pipeline (Q/K/V/QK/softmax/P·V/attn_out) for `prefill_last` vs `decode0`.
- `GRETA_TRACE_ATTN_L0_PIPE_OUT=/root/gretacore/artifacts/alignment/.../b3_30_attn_l0_pipe.jsonl`  
  JSONL output path for layer0 attention pipeline trace.
- `GRETA_TRACE_ATTN_L0_NORM=1`  
  Adds `attn_norm_in` and `attn_norm_out` (RMSNorm input/output) to the layer0 trace.
- `GRETA_TRACE_QKV_W_VERIFY=1`  
  Verifies QKV weight layout/packing for layer0 (row vs col) during `prefill_last` vs `decode0`.
- `GRETA_TRACE_WO_W_VERIFY=1`  
- `GRETA_TRACE_POST_WO=1`  
- `GRETA_TRACE_POST_WO_OUT=/root/gretacore/artifacts/alignment/.../b3_41_post_wo.jsonl`  
- `GRETA_TRACE_POST_WO_LAYERS="0"`  
- `GRETA_TRACE_POST_WO_SAMPLE=1024`  
- `GRETA_TRACE_POST_WO_PHASES="prefill_last,decode0"`  
- `GRETA_TRACE_RMSNORM=1`  
- `GRETA_TRACE_RMSNORM_OUT=/root/gretacore/artifacts/alignment/.../b3_42_rmsnorm.jsonl`  
- `GRETA_TRACE_RMSNORM_LAYERS="0"`  
- `GRETA_TRACE_RMSNORM_SAMPLE=1024`  
- `GRETA_TRACE_RMSNORM_PHASES="prefill_last,decode0"`  
  Verifies WO weight layout/packing for layer0 (row vs col) during `prefill_last` vs `decode0`.
- `GRETA_PREFILL_FORCE_WQ_ROW=1`  
  Forces prefill Q projection to use the row-major interpretation (B3.34).
- `GRETA_PREFILL_FORCE_WK_ROW=1`  
  Forces prefill K projection to use the row-major interpretation (B3.35).
- `GRETA_PREFILL_FORCE_WV_LAYOUT=row|col|auto`  
  Forces prefill V projection to use row/col layout (B3.36; recommend `row`).
- `GRETA_PREFILL_QKV_LAYOUT=row|col|auto`  
  Explicit layout selector for prefill QKV projection (row recommended for B3.34/B3.35).
- `GRETA_WO_LAYOUT_FORCE=row|col|auto`  
  Forces WO layout interpretation (B3.40; recommend `row`).
- `GRETA_QKV_FORCE_ROUTE=mfma|valu|auto`  
  Forces Q/K/V projection route in **decode** (S=1).
- `GRETA_QKV_FORCE_GEMM=1`  
  Forces GEMM for decode (disables fused QKV GEMV path).
- `GRETA_TRACE_PROMPT_ID=p4_sys`  
  Optional prompt label for trace attribution.
- `GRETA_TRACE_ATTN_LAYERS="0,1,2,31"`  
  Selects layers for decode attention traces.
- `GRETA_TRACE_ATTN_POINTS="q,k,v,attn_out,x_out"`  
  Selects tensors for decode attention traces.
- `GRETA_TRACE_STAGE_POINTS="x_in,attn_out,wo_out,x_after_attn,ffn_norm,mlp_out,x_out,final_norm,lm_head_in,logits"`  
  Example stage trace points (includes `wo_out` for attention output projection).
- `GRETA_TRACE_KV_INVARIANTS=1`  
  Checks KV cache offsets/positions invariants.
- `GRETA_FORCE_ATTN_DECODE_KERNEL=auto|manual|fused`  
  Forces decode attention kernel path.
- `GRETA_FORCE_ATTN_DECODE_MATMUL=auto|valu|mfma`  
  Forces GEMM route in decode (Q/K/V/O) to isolate kernels.
- `GRETA_LMHEAD_FORCE_ROUTE=valu|mfma`  
  Forces LM head route (prefill+decode) to isolate MFMA/VALU.
- `GRETA_LMHEAD_FORCE_ROUTE_DECODE=valu|mfma`  
  Forces LM head route **decode-only**.
- `GRETA_TRACE_STAGE=1`  
  Emits per-stage JSONL for `prefill_last` vs `decode0`.
- `GRETA_TRACE_STAGE_OUT=/root/gretacore/artifacts/alignment/.../b3_27_stage.jsonl`  
  JSONL output path for stage trace.
- `GRETA_TRACE_STAGE_LAYERS="0,1,2,15,31"`  
  Select layers for stage trace.
- `GRETA_TRACE_STAGE_POINTS="x_in,attn_out,x_after_attn,mlp_out,x_out,x_after_mlp,final_rms,lm_head_in,logits"`  
  Select tensors for stage trace.
- `GRETA_TRACE_STAGE_PHASES="prefill_last,decode0"`  
  Select phases for stage trace.
- `GRETA_TRACE_STAGE_DEBUG_INPUT=1`  
  Adds input semantics fields (`x_in_src_kind`, `x_in_token_index_used`, `x_in_offset_bytes`, `x_in_ptr`, `x_in_alloc_bytes`, `prompt_tokens`, `kv_pos`, `decode_step`).

**B3.23 note:** QK and softmax match FP64 in decode0 (layer 31 head 0, windowed). Divergence is more likely in V accumulation / `attn_out` path.
**B3.27 note:** First divergence appears at layer-0 `x_in`, indicating decode input semantics mismatch (before attention/MLP).
**B3.59 note:** `x_in` divergence resolved. No zeroing found. Perfect hash consistency confirmed with standardized metadata (`token_id`, `route`). Divergence is likely further downstream.

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
