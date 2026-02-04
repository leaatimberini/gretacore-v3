# GRETA CORE – Roadmap

Version: 1.0  
Status: Foundational  
Project Phase: Phase 3 – LLM Inference Pipeline (active)  
Language: English

---

## Purpose of This Roadmap

This roadmap defines the long-term execution plan for GRETA CORE.
It establishes clear phases, objectives, deliverables, and success
criteria in order to maintain technical focus and architectural
coherence over a multi-year timeline.

GRETA CORE is not optimized for speed of delivery.
It is optimized for correctness, performance, and sustainability.

---

## Current Status (2026-02-04)

- Phase 1 (Runtime Core): **completed**.
- Phase 2 (Kernel Dominance): **completed** (13 TFLOPS GEMM, 2.1ms Llama Block).
- Phase 3 (LLM Inference Pipeline): **active** (B3.x).
- B3.14–B3.16: LM head isolation; MFMA disabled for LM head; prefill coherent with VALU.
- B3.17–B3.18: decode traces (LM head force route, hidden equivalence, layer delta).
- B3.19: `seq_len = pos + 1` fix in attention decode did not remove collapse.
- B3.20: attention decode isolation (attn verify/ref, KV invariants, route matrix). KV invariants OK; attn_out diverges from ref at layer 31; `fused+mfma` fails at load.
- B3.21: `fused+mfma` stabilized (Hkv fix + alignment guard rails). MFMA==VALU at decode0, but ref divergence at layer 31 and decode0 collapse persist.
- B3.22: high-layer attention precision audit; divergence vs FP64 ref persists at layer 31 independent of accumulation mode.
- B3.23: softmax isolation at decode0 (layer 31 head 0). QK and softmax match FP64; focus shifts to V accumulation / attn_out path.
- B3.24–B3.26: V layout/addressing in decode fixed; P·V consistent; decode0 collapse persists.
- B3.27–B3.29: post-x_in stage trace; first mismatch localized at `attn_out` (layer 0).
- B3.30–B3.32: layer0 isolation; `attn_norm_out` ok; first mismatch at Q; not MFMA/VALU route-dependent.
- B3.33: Wq layout verification: prefill_last=col, decode0=row for context prompts.
- B3.34: fix prefill Wq layout (GEMM) to align with decode (row).
- B3.35: fix prefill Wk layout (GEMM) to align with decode (row).
- B3.36: fix prefill Wv layout (GEMM) to align with decode (row).
- MI300X validation ongoing; evidence under `docs/AMD/`.

## Phase 0 – Foundations

### Objective
Establish the intellectual, architectural, and operational foundations
of the project before writing production code.

### Scope
- Documentation
- Architecture definition
- Benchmark design
- Environment preparation

### Deliverables
- Whitepaper (EN / ES)
- README (EN / ES)
- Roadmap (EN / ES)
- Architectural principles
- Benchmark specification
- Repository structure

### Success Criteria
- Clear project vision and scope
- Reproducible benchmark definitions
- No production code written prematurely

### Explicitly Out of Scope
- Kernel optimization
- Runtime implementation
- Framework integration

---

## Phase 1 – Runtime Core

### Objective
Build a minimal, deterministic compute runtime capable of managing
memory, execution streams, and kernel launches with low overhead.

### Scope
- Custom runtime (gcore-rt)
- Memory pooling and reuse
- Stream and event abstraction
- Telemetry and timing

### Deliverables
- Memory allocator (pool-based)
- Stream/event API
- Kernel launch wrapper
- Runtime microbenchmarks

### Success Criteria
- Stable runtime under stress tests
- Deterministic execution behavior
- Measurable overhead reduction compared to generic runtimes

### Explicitly Out of Scope
- Advanced kernel optimizations
- LLM pipelines
- Compiler automation

---

## Phase 2 – Kernel Dominance (LLM Primitives)

### Objective
Achieve high-performance implementations of the core kernels required
for LLM inference.

### Scope
- Kernel-first development
- Focus on memory traffic reduction
- Hardware-aware tuning

### Deliverables
- GEMM (FP16 / BF16)
- LayerNorm / RMSNorm
- Softmax
- KV-cache operations
- Initial kernel fusion

### Success Criteria
- Correctness validated against reference implementations
- Competitive performance on target AMD hardware
- Stable performance across long inference runs

### Explicitly Out of Scope
- Full model execution
- Training or backpropagation
- Developer tooling polish

---

## Phase 3 – LLM Inference Pipeline

### Objective
Enable end-to-end LLM inference using GRETA CORE components.

### Scope
- Minimal inference runtime
- Operator scheduling
- Memory lifecycle management

### Deliverables
- Transformer block execution
- Quantized and/or FP16 inference path
- End-to-end inference benchmarks (tokens/s, latency)

### Success Criteria
- Successful execution of at least one target LLM
- Stable throughput and latency metrics
- No reliance on external CUDA-based stacks

### Explicitly Out of Scope
- Distributed inference
- Multi-GPU execution
- Training workloads

---

## Phase 4 – Developer Experience & Tooling

### Objective
Improve usability without sacrificing performance or control.

### Scope
- Profiling
- Debugging
- Autotuning workflows

### Deliverables
- Integrated profiler
- Kernel performance visualization
- Autotuning database
- Documentation for developers

### Success Criteria
- Clear visibility into performance bottlenecks
- Reproducible tuning results
- Reduced iteration time for kernel development

---

## Phase 5 – Ecosystem Expansion

### Objective
Expand GRETA CORE beyond its initial targets while preserving
architectural integrity.

### Scope
- Additional operators
- Framework bridges
- Broader hardware support

### Deliverables
- Framework compatibility layer (Triton / PyTorch / JAX)
- Parity mapping for cuDNN/TensorRT via AMD equivalents
- Extended kernel library
- Deployment-oriented tooling

### Success Criteria
- Maintainable growth without architectural degradation
- Measurable adoption by external users
- Continued performance competitiveness
- Same code and install steps across Radeon dev and MI300X cloud targets

---

## Governance Principles

- Architectural decisions remain centralized
- Performance metrics override abstraction preferences
- Minimalism is enforced at every layer
- Every dependency must justify its existence

---

## Authorship

GRETA CORE is conceived, founded, and led by:

Leandro Emanuel Timberini  
Founder & Lead Systems Architect
