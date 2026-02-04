# GRETA CORE

GRETA CORE is a long-term engineering project focused on building a
high-performance, minimal, CUDA-like compute stack for AMD hardware,
designed specifically for Large Language Models (LLMs).

The project exists to break the current CUDA lock-in by addressing the
problem at its root: software.

---

## Motivation

The modern AI ecosystem is dominated by a single compute platform.
This dominance has created artificial barriers to entry, inflated
hardware costs, and limited innovation.

GRETA CORE approaches this problem from a software-first perspective,
aiming to unlock the full potential of AMD hardware through a focused,
performance-driven compute stack.

---

## Philosophy

- Software over hardware
- Full stack control
- Minimalism over bloat
- Performance over abstraction
- Long-term engineering discipline

---

## What GRETA CORE Is

- A custom compute runtime for AMD hardware
- A kernel-first LLM execution stack
- A CUDA-like developer experience without replicating CUDA
- A long-term research and engineering initiative
- An install that bundles torch, triton, and jax (no extra installs required)

---

## What GRETA CORE Is Not

- Not a CUDA fork
- Not a thin wrapper around existing frameworks
- Not a general-purpose GPU compute platform
- Not a short-term optimization project

---

## Project Status

**Phase 3 – LLM Inference Pipeline (B3.x active)**

- B3.14–B3.16: LM head route isolation; MFMA disabled by default for LM head; VALU coherent in prefill.
- B3.17–B3.18: Decode LM head isolation, hidden equivalence, and per-layer delta traces for decode.
- B3.19: Decode attention `seq_len = pos + 1` fix attempted; decode0 collapse persists.
- B3.20: Attention decode isolation (attn verify/ref, KV invariants, forced kernel/matmul matrix). KV invariants OK; attn_out diverges from ref in layer 31; fused+mfma fails at load.
- B3.21: Fused+MFMA decode stabilized (Hkv fix + alignment guards). MFMA==VALU under shadow compare, but attn_out vs ref still diverges at layer 31 and decode0 collapse persists.
- B3.22: High-layer attention precision audit; divergence vs FP64 ref persists at layer 31 independent of FP16/FP32 accumulation.
- B3.23: Softmax isolation (decode0, layer 31 head 0) shows QK and softmax match FP64 (MAE ~1e-6 / ~1e-8). Focus shifts to V accumulation / attn_out path.
- B3.24–B3.26: V accumulation/layout fixed; PV matches ref (pv_mae ~1e-7), but decode0 collapse persists.
- B3.27: StageTrace shows first mismatch at layer-0 `x_in` between `prefill_last` and `decode0`.
- B3.28: Input semantics tracing to align decode0 `x_in` with `prefill_last` for isolation (debug-only override).
- MI300X validation ongoing; AMD reports under `docs/AMD/`.

**Phase 1 – Runtime Core (completed)**
**Phase 2 – Kernel Dominance (completed)**

## v0.1 Public Release (Definition of Done)

- Runtime core stable on APU (no hangs across smoke profiles).
- Vulkan backend supports device-local buffers + staging.
- GEMM correctness validated vs CPU reference (FP32 required, FP16 optional).
- Baseline benchmarks recorded with environment metadata.

### Baseline Snapshot (2026-01-29, Ryzen 5 8600G APU)

| Benchmark | Metric | Value |
| --- | --- | --- |
| vk_smoke_bench | empty_submit_wait_ms | 1.845 |
| dispatch_bench | mean_ns_per_submit_and_exec | 312.787 |
| stream_bench | mean_ns_per_task | 60.594 |
| telemetry_bench | mean_ns_per_scope | 39.693 |
| alloc_bench | mean_ops_per_sec | 98,623,523.700 |

---

## Documentation

- Whitepaper: `docs/en/whitepaper.md`
- Roadmap: `docs/en/roadmap.md`
- Work plan: `docs/en/workplan.md`
- Debugging guide: `docs/en/debugging.md`
- FP16 healthcheck: `docs/en/runtime_fp16_healthcheck.md`
- Safety profiles: `docs/en/runtime_safety_profiles.md`
- Validation checklist: `docs/en/runtime_validation_checklist.md`
- Compatibility matrix: `docs/en/compatibility.md`
- Framework compatibility plan: `docs/en/strategy/framework_compat.md`
- Framework version matrix: `docs/en/strategy/framework_versions.md`
- Framework prototypes: `tools/compat/README.md`

## Reproduce B3.23 (MI300X)

Local-first workflow (required):
1) Edit locally → `git commit` → `git push`
2) On MI300X: `git pull --rebase` → build → run
3) Tar artifacts and `scp` back to local

```bash
export GRETA_TRACE_ATTN_SOFTMAX=1
export GRETA_TRACE_ATTN_LAYER=31
export GRETA_TRACE_ATTN_HEAD=0
export GRETA_TRACE_ATTN_KEYS_WINDOW=64
export GRETA_TRACE_ATTN_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_23_attn_softmax.jsonl
# Optional: label runs for analysis
export GRETA_TRACE_PROMPT_ID=p4_sys

# Decode attention context (optional, if also tracing decode verification)
export GRETA_TRACE_ATTN_DECODE_VERIFY=1
export GRETA_TRACE_KV_INVARIANTS=1
export GRETA_TRACE_ATTN_LAYERS="0,1,2,31"
export GRETA_TRACE_ATTN_POINTS="q,k,v,attn_out,x_out"
export GRETA_ATTN_DECODE_REF=1
```

## Debug Tracing (StageTrace)

```bash
export GRETA_TRACE_STAGE=1
export GRETA_TRACE_STAGE_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_28_stage.jsonl
export GRETA_TRACE_STAGE_LAYERS="0,1,2,15,31"
export GRETA_TRACE_STAGE_POINTS="x_in,attn_out,x_after_attn,mlp_out,x_out,x_after_mlp,final_rms,lm_head_in,logits"
export GRETA_TRACE_STAGE_PHASES="prefill_last,decode0"
export GRETA_TRACE_STAGE_DEBUG_INPUT=1
export GRETA_TRACE_PROMPT_ID=p4_sys
```

Note: `GRETA_TRACE_STAGE_DEBUG_INPUT=1` adds input semantics fields and enables a decode0 input override used for isolation runs (B3.28).

## Debug Tracing (Layer0 Attention Pipeline)

```bash
export GRETA_TRACE_ATTN_L0_PIPE=1
export GRETA_TRACE_ATTN_L0_PIPE_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_30_attn_l0_pipe.jsonl
# Include attention norm input/output (B3.32)
export GRETA_TRACE_ATTN_L0_NORM=1
# Verify QKV weight layout/packing for layer0 (B3.33)
export GRETA_TRACE_QKV_W_VERIFY=1
# Verify WO weight layout/packing for layer0 (B3.40)
export GRETA_TRACE_WO_W_VERIFY=1
# Force prefill Q projection to use row layout (B3.34)
export GRETA_PREFILL_FORCE_WQ_ROW=1
# Force prefill K projection to use row layout (B3.35)
export GRETA_PREFILL_FORCE_WK_ROW=1
# Force prefill V projection to use row layout (B3.36)
export GRETA_PREFILL_FORCE_WV_LAYOUT=row
# Alternative: explicit layout selector (row|col|auto)
export GRETA_PREFILL_QKV_LAYOUT=row
# Force WO layout interpretation (row|col|auto) for diagnostics (B3.40)
export GRETA_WO_LAYOUT_FORCE=row
# Recommended when comparing prefill_last vs decode0 for the same token/pos
export GRETA_TRACE_STAGE_DEBUG_INPUT=1
```

## Decode QKV Route Forcing (B3.31)

```bash
# Force projection route for Q/K/V in decode (S=1)
export GRETA_QKV_FORCE_ROUTE=mfma   # or valu, auto
# Force GEMM path for decode (disable fused gemv)
export GRETA_QKV_FORCE_GEMM=1
```

---

## Author

GRETA CORE was conceived, founded, and is led by:

**Leandro Emanuel Timberini**  
Founder & Lead Systems Architect

---

## License

License to be defined.
