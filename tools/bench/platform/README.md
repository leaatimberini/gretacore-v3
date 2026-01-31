# GRETA CORE — Platform Benchmarks

Path: tools/bench/platform/README.md  
Version: 1.0  
Language: EN/ES (single file; bilingual sections)

## Purpose
These benchmarks measure platform/hardware limits independently of GRETA CORE runtime:
- CPU memory bandwidth (DDR5 throughput)
- GPU kernel launch overhead (HIP no-op) when available
- GPU vector add bandwidth (HIP) when available
- GPU GEMM throughput (HIP+hipBLAS) when available

## Build (Ubuntu 22.04) [EN]
From repo root:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build linux-tools-common linux-tools-generic
cmake -S tools/bench/platform -B tools/bench/platform/build -G Ninja
cmake --build tools/bench/platform/build
```

### HIP (optional) [EN]
```bash
export PATH=/opt/rocm/bin:$PATH
export HIP_PATH=/opt/rocm
export CMAKE_PREFIX_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
export GRETA_HIP_ARCH=gfx942
cmake -S tools/bench/platform -B tools/bench/platform/build -G Ninja -DGRETA_ENABLE_HIP=ON -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
cmake --build tools/bench/platform/build
```

## Run presets [EN]
```bash
tools/bench/platform/scripts/run_presets_local.sh smoke
tools/bench/platform/scripts/run_presets_local.sh standard
tools/bench/platform/scripts/run_presets_local.sh perf
tools/bench/platform/scripts/run_presets_local.sh verify
```

Enable hip_gemm checks in presets:
```bash
export GRETA_HIP_GEMM_CHECK=1
export GRETA_HIP_GEMM_CHECK_SAMPLES=8
tools/bench/platform/scripts/run_presets_local.sh smoke
```

Remote:
```bash
tools/bench/platform/scripts/run_presets_remote.sh user@host /workspace/gretacore smoke
```

Generate CSV summary:
```bash
tools/bench/platform/scripts/gen_bench_csv.py tools/bench/platform/results
```

### Standalone (HIP) [EN]
```bash
tools/bench/platform/build/hip_gemm --m 2048 --n 2048 --k 2048 --iters 20 --warmup 5 --check 1 --check-samples 8
```

## Construcción (Ubuntu 22.04) [ES]
Desde el root del repo:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build linux-tools-common linux-tools-generic
cmake -S tools/bench/platform -B tools/bench/platform/build -G Ninja
cmake --build tools/bench/platform/build
```

### HIP (opcional) [ES]
```bash
export PATH=/opt/rocm/bin:$PATH
export HIP_PATH=/opt/rocm
export CMAKE_PREFIX_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
export GRETA_HIP_ARCH=gfx942
cmake -S tools/bench/platform -B tools/bench/platform/build -G Ninja -DGRETA_ENABLE_HIP=ON -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
cmake --build tools/bench/platform/build
```

## Presets de ejecución [ES]
```bash
tools/bench/platform/scripts/run_presets_local.sh smoke
tools/bench/platform/scripts/run_presets_local.sh standard
tools/bench/platform/scripts/run_presets_local.sh perf
tools/bench/platform/scripts/run_presets_local.sh verify
```

Habilitar checks de hip_gemm en presets:
```bash
export GRETA_HIP_GEMM_CHECK=1
export GRETA_HIP_GEMM_CHECK_SAMPLES=8
tools/bench/platform/scripts/run_presets_local.sh smoke
```

Remoto:
```bash
tools/bench/platform/scripts/run_presets_remote.sh user@host /workspace/gretacore smoke
```

Generar resumen CSV:
```bash
tools/bench/platform/scripts/gen_bench_csv.py tools/bench/platform/results
```

### Standalone (HIP) [ES]
```bash
tools/bench/platform/build/hip_gemm --m 2048 --n 2048 --k 2048 --iters 20 --warmup 5 --check 1 --check-samples 8
```
