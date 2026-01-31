GRETA CORE
A Software-First Compute Stack for AMD Hardware and Large Language Models
1. Abstract

The rapid expansion of Large Language Models (LLMs) has exposed a critical structural weakness in the modern AI ecosystem: a near-total dependency on NVIDIA’s CUDA platform. This dependency has resulted in a de facto monopoly over AI compute, driving hardware costs to unsustainable levels and artificially limiting innovation.

GRETA CORE is a long-term engineering initiative aimed at breaking this dependency by building a minimal, high-performance, CUDA-like compute stack for AMD hardware, designed specifically for LLM workloads.

Rather than competing at the hardware level, GRETA CORE focuses on software dominance: full control of the runtime, kernel libraries, memory management, and execution model. The project is built from first principles, prioritizing performance, transparency, and long-term sustainability over short-term compatibility.

2. Problem Statement
2.1 The CUDA Lock-In

CUDA is not merely a programming API; it is an ecosystem that tightly couples hardware, software, tooling, and developer workflows. Over time, this coupling has created a self-reinforcing loop:

Frameworks optimize for CUDA first.

Tooling assumes CUDA semantics.

Developers default to NVIDIA hardware.

Alternative platforms are treated as secondary or experimental.

As a result, the AI ecosystem is no longer hardware-agnostic. It is CUDA-centric.

2.2 Rising Hardware Costs

The dominance of CUDA has led to an artificial scarcity of “usable” AI hardware. GPUs that are technically capable of LLM workloads are excluded by software limitations rather than physical constraints.

This has caused:

Exponential increases in GPU prices.

Reduced accessibility for independent developers and researchers.

Centralization of AI capabilities in large organizations.

The problem is not hardware performance, but software availability and optimization.

2.3 AMD Hardware: Capable but Underserved

AMD produces competitive CPUs, GPUs, and APUs with modern memory hierarchies and compute capabilities. However, AMD hardware is systematically underutilized for LLM workloads due to:

Fragmented software stacks.

Incomplete or overly generic abstractions.

Performance-critical kernels lagging behind CUDA equivalents.

Tooling that prioritizes breadth over depth.

The absence of a focused, LLM-first compute stack has left AMD hardware operating far below its potential.

3. Philosophy

GRETA CORE is guided by a set of non-negotiable principles.

3.1 Software Over Hardware

Hardware limitations are finite. Software limitations are not.

GRETA CORE assumes that software is the primary bottleneck in democratizing AI compute. By mastering the software stack, existing hardware can be pushed significantly further.

3.2 Full Stack Control

Performance cannot be achieved through wrappers and abstractions alone.

GRETA CORE seeks full control over:

Runtime execution

Memory allocation and reuse

Kernel scheduling

Data movement

Autotuning and fusion

External components are used only when they provide measurable value and can be modified or replaced if necessary.

3.3 Minimalism and Performance

Every layer of abstraction introduces overhead.

GRETA CORE rejects feature bloat, unused generality, and unnecessary dependencies. The stack is intentionally narrow, optimized for a specific class of workloads: LLM inference and related compute patterns.

If a component does not improve performance, stability, or developer control, it does not belong in the system.

4. What is GRETA CORE

GRETA CORE is:

A custom compute runtime designed for AMD hardware.

A kernel-first LLM execution stack.

A CUDA-like developer experience, without replicating CUDA itself.

A long-term research and engineering effort, not a short-term product.

A platform designed to evolve alongside LLM architectures.

5. What GRETA CORE Is Not

GRETA CORE is not:

A fork of CUDA.

A thin wrapper around existing frameworks.

A general-purpose GPU compute platform.

A short-term optimization project.

A competitor focused on marketing parity rather than technical substance.

Compatibility is a goal, but performance and control take precedence.

6. Technical Vision
6.1 Runtime

The GRETA CORE runtime is responsible for:

Explicit stream and event management.

Deterministic kernel scheduling.

High-performance memory pooling and reuse.

Low-overhead kernel launch mechanisms.

Integrated telemetry and profiling hooks.

The runtime is designed to minimize interaction with the operating system during steady-state execution.

6.2 Kernel Libraries

The core of GRETA CORE lies in its kernel implementations.

Initial focus areas include:

GEMM (FP16, BF16, and quantized variants).

LayerNorm and RMSNorm.

Softmax and attention-related primitives.

KV-cache management operations.

Fused kernels to minimize memory traffic.

Kernel correctness is mandatory. Kernel performance is paramount.

6.3 Compiler and Autotuning

Rather than relying solely on static kernel implementations, GRETA CORE will incorporate:

Kernel parameter exploration.

Hardware-aware autotuning.

Controlled kernel fusion.

Cost models driven by empirical data.

This enables adaptation across different AMD architectures without sacrificing performance.

6.4 Framework Integration

GRETA CORE does not aim to replace existing ML frameworks. Instead, it integrates selectively through:

Custom execution providers.

Minimal runtime bridges.

Direct invocation paths for performance-critical workloads.

The goal is to enable LLM execution without forcing developers to abandon familiar tooling.

7. Long-Term Roadmap

GRETA CORE is a multi-year initiative.

High-level phases include:

Foundational runtime and benchmarking.

Kernel performance parity for core LLM operations.

End-to-end LLM inference pipelines.

Developer tooling and profiling.

Ecosystem expansion and broader model support.

Each phase is evaluated by measurable performance and stability criteria.

8. Impact on the AI Ecosystem

By lowering the software barrier to effective AI compute on AMD hardware, GRETA CORE aims to:

Increase competition in the AI hardware market.

Reduce costs for developers and organizations.

Decentralize access to LLM capabilities.

Encourage innovation beyond a single-vendor ecosystem.

This impact is achieved through engineering, not policy.

9. Conclusion

The current dominance of CUDA is not inevitable. It is the result of sustained software investment, not insurmountable technical superiority.

GRETA CORE exists to prove that software control, minimalism, and long-term engineering discipline can unlock the full potential of alternative hardware platforms.

This project is not easy. It is not fast. And it is not guaranteed.

But it is necessary.



Version: 1.0
Status: Foundational Draft
Project Phase: Phase 0 – Foundations

## Authorship

GRETA CORE is an independent engineering project conceived, founded,
and led by:

Leandro Emanuel Timberini  
Founder & Lead Systems Architect

All architectural decisions, long-term vision, and foundational
principles originate from this authorship.
