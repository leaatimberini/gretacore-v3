# GRETA CORE - Phase 3 Reproducibility Guide

This guide describes how to reproduce the Phase 3 performance and correctness results on AMD Instinct™ accelerators.

## Hardware Requirements
- **Accelerator**: AMD Instinct™ MI300X (HBM3)
- **Host RAM**: 128GB+ (minimum for large model loading)
- **Disk**: 50GB+ for model checkpoints and build artifacts

## Software Prerequisites
- **OS**: Ubuntu 22.04 LTS or 24.04 LTS (Noble)
- **ROCm™ Stack**: Version 6.2 or 7.1
- **Compilers**: `hipcc` (Clang-based)
- **Build System**: CMake 3.22+ & Make

## Build Instructions

### 1. Build the Engine
```bash
cd gretacore
mkdir -p tools/inference/build
cd tools/inference/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### 2. Prepare the Model
Ensure you have a Llama-2-7B model in GGUF format (Q4_K_M recommended).
Default path for benchmarks: `/root/models/llama-2-7b-chat.Q4_K_M.gguf`.

## Execution and Validation

### Standard Inference
```bash
./greta_infer --model [path_to_model] --prompt "Hi" --max-tokens 32 --greedy
```

### Performance Benchmarking (with HIP Graphs)
```bash
GRETA_HIP_GRAPH=1 GRETA_VERBOSE_INFO=1 ./greta_infer --model [path_to_model] --prompt "Hi" --max-tokens 32 --greedy
```

### Numerical Audit (Tracer)
```bash
GRETA_TRACE_LEVEL=1 GRETA_PROFILE_BLOCKS=1 ./greta_infer --model [path_to_model] --prompt "Hi" --max-tokens 5 --greedy
```

## Automated Benchmarking
Use the provided script to execute the full Phase 3 matrix:
```bash
cd tools/phase3
./run_bench.sh [path_to_model]
```
Results will be exported to `bench_results.csv`.

## Troubleshooting
- **NaNs in output**: Check if `GRETA_TRACE_LEVEL=1` is active to identify the first failing block.
- **HIP Graph Failures**: Ensure you are using a compatible ROCm version and that `S=1` (decode phase) for graph capture.
