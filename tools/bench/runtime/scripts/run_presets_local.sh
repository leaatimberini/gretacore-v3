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
    M=128; N=128; K=128; iters=3; batch=5
    LLM_ROWS=64; LLM_COLS=256; LLM_ITERS=5
    ;;
  verify)
    M=64; N=64; K=64; iters=2; batch=2
    LLM_ROWS=32; LLM_COLS=128; LLM_ITERS=3
    ;;
  standard)
    M=1024; N=1024; K=1024; iters=10; batch=20
    LLM_ROWS=256; LLM_COLS=1024; LLM_ITERS=10
    ;;
  perf)
    M=1024; N=1024; K=1024; iters=10; batch=50
    LLM_ROWS=256; LLM_COLS=1024; LLM_ITERS=10
    ;;
  *)
    echo "unknown preset: $preset" >&2
    exit 1
    ;;
 esac

run vk_gemm_tiled_ts_bench --m "$M" --n "$N" --k "$K" --iters "$iters" --batch "$batch"
run vk_gemm_f16acc32_tiled_vec2_ts_bench --m "$M" --n "$N" --k "$K" --iters "$iters" --batch "$batch"
run vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench --m "$M" --n "$N" --k "$K" --iters "$iters" --batch "$batch"
run vk_gemm_f16acc32_tiled_vec2_db_ts_bench --m "$M" --n "$N" --k "$K" --iters "$iters" --batch "$batch"
run vk_layernorm_bench --rows "$LLM_ROWS" --cols "$LLM_COLS" --iters "$LLM_ITERS"
run vk_layernorm_rmsnorm_fused_bench --rows "$LLM_ROWS" --cols "$LLM_COLS" --iters "$LLM_ITERS"
run vk_layernorm_rmsnorm_fused_tiled_bench --rows "$LLM_ROWS" --cols "$LLM_COLS" --iters "$LLM_ITERS"
run vk_layernorm_tiled_bench --rows "$LLM_ROWS" --cols "$LLM_COLS" --iters "$LLM_ITERS"
run vk_rmsnorm_bench --rows "$LLM_ROWS" --cols "$LLM_COLS" --iters "$LLM_ITERS"
run vk_rmsnorm_tiled_bench --rows "$LLM_ROWS" --cols "$LLM_COLS" --iters "$LLM_ITERS"
run vk_softmax_bench --rows "$LLM_ROWS" --cols "$LLM_COLS" --iters "$LLM_ITERS"
run vk_softmax_tiled_bench --rows "$LLM_ROWS" --cols "$LLM_COLS" --iters "$LLM_ITERS"

python3 "$root/scripts/gen_bench_csv.py" "$results"
