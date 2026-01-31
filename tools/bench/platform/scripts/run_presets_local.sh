#!/usr/bin/env bash
set -euo pipefail

preset="${1:-standard}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/build"
results="$root/results"

if [[ ! -d "$build" ]]; then
  echo "build directory not found: $build" >&2
  exit 2
fi
mkdir -p "$results"

date_str="${DATE_OVERRIDE:-$(date +%F)}"

run() {
  local bin="$1"; shift
  "${build}/${bin}" "$@" | tee "${results}/${date_str}_${bin}_${preset}.txt"
}

case "$preset" in
  smoke)
    MEMBW_SIZE_MB=512; MEMBW_ITERS=3
    MEMLAT_SIZE_MB=128; MEMLAT_ITERS=20
    HIP_NOOP_ITERS=100000
    HIP_VEC_N=$((1 << 22)); HIP_VEC_ITERS=50
    HIP_GEMM_M=512; HIP_GEMM_N=512; HIP_GEMM_K=512; HIP_GEMM_ITERS=10; HIP_GEMM_WARMUP=5
    ;;
  verify)
    MEMBW_SIZE_MB=256; MEMBW_ITERS=3
    MEMLAT_SIZE_MB=64; MEMLAT_ITERS=10
    HIP_NOOP_ITERS=50000
    HIP_VEC_N=$((1 << 20)); HIP_VEC_ITERS=20
    HIP_GEMM_M=512; HIP_GEMM_N=512; HIP_GEMM_K=512; HIP_GEMM_ITERS=5; HIP_GEMM_WARMUP=2
    HIP_GEMM_CHECK=1
    HIP_GEMM_CHECK_SAMPLES=8
    ;;
  standard)
    MEMBW_SIZE_MB=1024; MEMBW_ITERS=6
    MEMLAT_SIZE_MB=256; MEMLAT_ITERS=50
    HIP_NOOP_ITERS=200000
    HIP_VEC_N=$((1 << 24)); HIP_VEC_ITERS=200
    HIP_GEMM_M=2048; HIP_GEMM_N=2048; HIP_GEMM_K=2048; HIP_GEMM_ITERS=20; HIP_GEMM_WARMUP=5
    ;;
  perf)
    MEMBW_SIZE_MB=2048; MEMBW_ITERS=8
    MEMLAT_SIZE_MB=512; MEMLAT_ITERS=80
    HIP_NOOP_ITERS=400000
    HIP_VEC_N=$((1 << 24)); HIP_VEC_ITERS=400
    HIP_GEMM_M=4096; HIP_GEMM_N=4096; HIP_GEMM_K=4096; HIP_GEMM_ITERS=10; HIP_GEMM_WARMUP=5
    ;;
  *)
    echo "unknown preset: $preset" >&2
    exit 1
    ;;
 esac

run membw_cpu --size-mb "$MEMBW_SIZE_MB" --iters "$MEMBW_ITERS"
run memlat_cpu --size-mb "$MEMLAT_SIZE_MB" --iters "$MEMLAT_ITERS"

if [[ -x "${build}/hip_noop_launch" ]]; then
  run hip_noop_launch --iters "$HIP_NOOP_ITERS"
else
  echo "hip_noop_launch not found; skipping HIP benches" >&2
fi

if [[ -x "${build}/hip_vec_add" ]]; then
  run hip_vec_add --n "$HIP_VEC_N" --iters "$HIP_VEC_ITERS"
else
  echo "hip_vec_add not found; skipping HIP benches" >&2
fi

if [[ -x "${build}/hip_gemm" ]]; then
  HIP_GEMM_CHECK="${HIP_GEMM_CHECK:-${GRETA_HIP_GEMM_CHECK:-0}}"
  HIP_GEMM_CHECK_SAMPLES="${HIP_GEMM_CHECK_SAMPLES:-${GRETA_HIP_GEMM_CHECK_SAMPLES:-8}}"
  run hip_gemm --m "$HIP_GEMM_M" --n "$HIP_GEMM_N" --k "$HIP_GEMM_K" \
    --iters "$HIP_GEMM_ITERS" --warmup "$HIP_GEMM_WARMUP" \
    --check "$HIP_GEMM_CHECK" --check-samples "$HIP_GEMM_CHECK_SAMPLES"
else
  echo "hip_gemm not found; skipping HIP GEMM bench" >&2
fi
