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
- Next: B3.20 attention/KV decode audit to eliminate decode collapse.
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

---

## Author

GRETA CORE was conceived, founded, and is led by:

**Leandro Emanuel Timberini**  
Founder & Lead Systems Architect

---

## License

License to be defined.
