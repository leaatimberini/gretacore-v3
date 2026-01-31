# CUDA Parity Gap Matrix (Draft)

Date: 2026-01-31

## Objective
Define the minimum software capabilities GRETA must deliver so developers choose it over CUDA/ROCm for AI workloads.

## Why CUDA Wins (Reality Check)
- **Language ubiquity**: most AI code assumes CUDA.
- **Library maturity**: cuDNN/TensorRT deliver large speedups with no extra work.
- **Scale software**: NCCL + NVLink are battle-tested at multi-GPU scale.
- **Developer experience**: fast onboarding, rich docs, community fixes.

## GRETA Strategic Response (Pillars)
1) **No-translation kernels** (Triton-class experience)
   - Goal: write once, run on AMD without HIP hand-translation.
   - Gate: a set of reference kernels (GEMM, LayerNorm, Softmax, RMSNorm) running from a single frontend.

2) **Zero-effort libraries**
   - Goal: common workloads run fast without manual tuning.
   - Gate: measurable speedups vs baseline kernels using pre-tuned paths.
   - Note: parity via AMD equivalents (MIOpen/rocBLAS/MIGraphX), not NVIDIA binaries.

3) **Unified developer path (UAI-like)**
   - Goal: same workflow on Radeon dev boxes and Instinct data center GPUs.
   - Gate: identical build + runtime steps across device tiers.

4) **Scaling-ready runtime**
   - Goal: multi-GPU communication hooks ready for NCCL/RCCL-style integration.
   - Gate: API surface defined even if backend is staged.

## Immediate Deliverables
- Publish this gap matrix (EN/ES) and keep it updated per milestone.
- Define 3 GRETA “differentiator” tasks for LOE-3/LOE-4.
- Add pass/fail criteria for each pillar.
- Publish framework compatibility plan in `docs/en/strategy/framework_compat.md`.

## Differentiator Tasks (LOE-3 / LOE-4)
1) **LLM Primitives Pack v1**
   - Scope: LayerNorm, RMSNorm, Softmax (stable) with CPU reference checks.
   - Pass: correctness within tolerance; microbench results archived.

2) **Single Transformer Block Runner**
   - Scope: minimal runner with RMSNorm + QKV + attention + MLP (FP16/FP32).
   - Pass: deterministic outputs across 5 runs; metrics (ms, tokens/s) recorded.

3) **Kernel Tuning Cache**
   - Scope: cache best kernel config per shape/device.
   - Pass: cache hit-rate ≥80% on repeated runs; speedup vs baseline logged.

## Pass/Fail Criteria (By Pillar)
1) **No-translation kernels**
   - Pass: all reference kernels run from a single frontend on AMD without HIP edits.
   - Fail: any kernel requires manual HIP translation.

2) **Zero-effort libraries**
   - Pass: ≥1.5x speedup vs baseline on 3 workloads.
   - Fail: no measurable speedup across workloads.

3) **Unified developer path**
   - Pass: identical build+run steps on Radeon + Instinct.
   - Fail: device-specific steps required.

4) **Scaling-ready runtime**
   - Pass: multi-GPU API surface defined and smoke-tested (single-node).
   - Fail: no API or untestable integration points.
