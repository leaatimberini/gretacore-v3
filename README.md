# GRETA CORE

**Status**: Phase 3 up to B3.60

GRETA CORE is a long-term engineering project focused on building a
high-performance, minimal, CUDA-like compute stack for AMD hardware,
designed specifically for Large Language Models (LLMs).

The project exists to break the current CUDA lock-in by addressing the
problem at its root: software.

---

## Phase 3 Progress (B3.xx Audit Series)

| Milestone | Status | Description |
|-----------|--------|-------------|
| B3.52 | ✅ PASS | KV cache addressing fix |
| B3.55-B3.58 | ✅ PASS | Root cause isolation (RoPE/Q-proj/RMSNorm) |
| B3.59 | ✅ PASS | Embedding + StageDebugInput audit (no zeroing) |
| B3.60 | ✅ PASS | Attention Block bisect (Layer0 pipeline verified) |

**Documentation**:
- [Progress Index](docs/PROGRESS.md)
- [AMD Reports Index](docs/AMD/INDEX.md)

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

## Documentation

- [Progress Tracking](docs/PROGRESS.md)
- [AMD Audit Reports](docs/AMD/INDEX.md)
- [Benchmarking Tools](tools/benchmarks/)

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/leaatimberini/gretacore.git
cd gretacore

# Check documentation
cat docs/PROGRESS.md
cat docs/AMD/INDEX.md
```

---

## Project Structure

```
├── src/           # Core runtime implementation
├── tools/         # Benchmarking and diagnostic tools
├── docs/          # Documentation and AMD reports
├── models/        # Model definitions
├── build/         # Build artifacts (not versioned)
├── README.md      # This file
└── .gitignore    # Git ignore rules
```

---

## Contributing

This is a long-term engineering initiative. All contributions must align
with the project's philosophy of minimalism, performance, and full
stack control.

Contributions should focus exclusively on:
- Source code
- Technical documentation
- Reproducible benchmarks
- Verifiable audits

Changes unrelated to the inference engine or its documentation will not be accepted.
