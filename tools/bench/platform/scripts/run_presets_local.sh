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
    ;;
  standard)
    MEMBW_SIZE_MB=1024; MEMBW_ITERS=6
    MEMLAT_SIZE_MB=256; MEMLAT_ITERS=50
    HIP_NOOP_ITERS=200000
    HIP_VEC_N=$((1 << 24)); HIP_VEC_ITERS=200
    ;;
  perf)
    MEMBW_SIZE_MB=2048; MEMBW_ITERS=8
    MEMLAT_SIZE_MB=512; MEMLAT_ITERS=80
    HIP_NOOP_ITERS=400000
    HIP_VEC_N=$((1 << 24)); HIP_VEC_ITERS=400
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
