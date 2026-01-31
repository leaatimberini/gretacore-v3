# GRETA CORE – Benchmarks

Version: 1.0  
Status: Foundational  
Project Phase: Phase 1 – Runtime Core  
Language: English

---

## Purpose of This Document

This document defines the official benchmarking methodology for
GRETA CORE.

Benchmarks are the authoritative mechanism for:
- validating correctness
- measuring performance
- detecting regressions
- guiding architectural decisions

No optimization, refactor, or feature is considered valid without
benchmark evidence.

---

## Benchmarking Principles

All benchmarks in GRETA CORE must satisfy the following conditions:

1. Reproducible
2. Deterministic
3. Minimal
4. Isolated
5. Comparable over time

Benchmarks that fail any of these criteria are invalid.

---

## Benchmark Categories

GRETA CORE benchmarks are divided into four categories:

1. Platform Benchmarks
2. Runtime Benchmarks
3. Kernel Benchmarks
4. End-to-End Benchmarks

Each category serves a distinct purpose and must not be conflated.

---

## 1. Platform Benchmarks

### Objective
Measure raw hardware and system limits independently of GRETA CORE.

### Metrics
- Memory bandwidth (GB/s)
- Memory latency
- Kernel launch latency
- CPU–GPU synchronization cost

### Examples
- Sequential memory read/write
- Random memory access
- Empty kernel launch

### Purpose
Establish upper bounds and identify non-software bottlenecks.

---

## 2. Runtime Benchmarks

### Objective
Measure overhead introduced by the GRETA CORE runtime.

### Metrics
- Allocation latency
- Memory reuse efficiency
- Stream scheduling overhead
- Event timing accuracy

### Examples
- Allocate/free cycles
- Stream synchronization
- No-op kernel dispatch through runtime

### Purpose
Ensure runtime overhead remains minimal and predictable.

---

## 3. Kernel Benchmarks

### Objective
Measure performance and correctness of individual kernels.

### Metrics
- Throughput (GFLOPs / tokens/s where applicable)
- Latency (mean, p50, p99)
- Memory traffic (bytes moved)
- Numerical correctness

### Kernel Classes
- GEMM
- Norms (LayerNorm, RMSNorm)
- Softmax
- Reductions
- KV-cache ops

### Rules
- One kernel per benchmark
- Fixed input sizes
- Verified output against reference

---

## 4. End-to-End Benchmarks

### Objective
Measure real-world performance of composed execution paths.

### Metrics
- Tokens per second
- End-to-end latency
- Memory footprint
- Stability over time

### Examples
- Transformer block execution
- Mini LLM inference loop
- KV-cache growth test

### Purpose
Validate that improvements translate into real gains.

---

## Benchmark Environment Requirements

All benchmarks must record:

- Hardware configuration
- Driver and runtime versions
- Compiler versions
- Power and clock settings (when applicable)

Benchmarks without environment metadata are invalid.

---

## Benchmark Execution Rules

- Benchmarks must run on a quiescent system
- Thermal throttling must be avoided or documented
- Multiple runs required for variance analysis
- Results must be stored in versioned form

---

## Regression Policy

Any change that causes:
- ≥5% performance regression
- increased variance
- increased memory usage

must be flagged and reviewed.

Regressions are treated as bugs.

---

## Output Format

Benchmark results must include:

- Raw measurements
- Aggregated statistics
- Environment metadata
- Timestamp

Human-readable and machine-readable formats are required.

---

## Authorship

GRETA CORE is conceived, founded, and led by:

Leandro Emanuel Timberini  
Founder & Lead Systems Architect
