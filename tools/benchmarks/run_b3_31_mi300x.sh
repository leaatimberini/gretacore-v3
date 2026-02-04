#!/usr/bin/env bash
set -euo pipefail

IP="${1:-129.212.184.200}"
DATE="${2:-2026-02-03}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR_LOCAL="$ROOT_DIR/artifacts_remote/${DATE}/b3_31"

cd "$ROOT_DIR"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: working tree has modified tracked files. Commit first." >&2
  exit 1
fi

# Local build (best-effort)
cmake -S tools/inference -B tools/inference/build || echo "WARN: local cmake configure failed"
cmake --build tools/inference/build -j"$(nproc)" || echo "WARN: local build failed"

# Push latest HEAD before remote run
git push origin HEAD
git ls-remote origin HEAD

ssh root@"$IP" <<'EOF_REMOTE'
set -euo pipefail
cd /root/gretacore

git fetch origin
git checkout main
git pull --rebase
git rev-parse HEAD

cmake -S /root/gretacore/tools/inference -B /root/gretacore/tools/inference/build
cd /root/gretacore/tools/inference/build
make -B -j$(nproc)

OUTDIR=/root/gretacore/artifacts/alignment/2026-02-03
mkdir -p "$OUTDIR"

export GRETA_INT4_WEIGHTS=1
export GRETA_MAX_SEQ_LEN=256
export GRETA_TRACE_ATTN_L0_PIPE=1
export GRETA_TRACE_STAGE_DEBUG_INPUT=1

MODEL=/root/gretacore/models/llama3_8b_q4/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
BIN=/root/gretacore/tools/inference/build/greta_infer

run_one () {
  local exp="$1"
  local prompt_id="$2"
  local prompt_file="$3"
  local force_route="$4"
  local force_gemm="$5"

  export GRETA_TRACE_PROMPT_ID="$prompt_id"
  export GRETA_TRACE_ATTN_L0_PIPE_OUT="$OUTDIR/b3_31_${exp}_attn_l0_pipe_${prompt_id}.jsonl"

  if [[ -n "$force_route" ]]; then
    export GRETA_QKV_FORCE_ROUTE="$force_route"
  else
    unset GRETA_QKV_FORCE_ROUTE
  fi
  if [[ "$force_gemm" == "1" ]]; then
    export GRETA_QKV_FORCE_GEMM=1
  else
    unset GRETA_QKV_FORCE_GEMM
  fi

  rm -f "$OUTDIR/b3_31_${exp}_attn_l0_pipe_${prompt_id}.jsonl"

  $BIN --model "$MODEL" --prompt-file "$prompt_file" --max-tokens 2 --greedy \
    2>&1 | tee "$OUTDIR/b3_31_${exp}_${prompt_id}.log"
}

# Experiment matrix
# E0: baseline auto
run_one E0 p4_sys /root/gretacore/tools/benchmarks/prompts/p4_sys.txt "" 0
run_one E0 p5_ba  /root/gretacore/tools/benchmarks/prompts/p5_ba.txt  "" 0

# E1: force VALU
run_one E1 p4_sys /root/gretacore/tools/benchmarks/prompts/p4_sys.txt "valu" 0
run_one E1 p5_ba  /root/gretacore/tools/benchmarks/prompts/p5_ba.txt  "valu" 0

# E2: force MFMA
run_one E2 p4_sys /root/gretacore/tools/benchmarks/prompts/p4_sys.txt "mfma" 0
run_one E2 p5_ba  /root/gretacore/tools/benchmarks/prompts/p5_ba.txt  "mfma" 0

# E3: force GEMM + MFMA
run_one E3 p4_sys /root/gretacore/tools/benchmarks/prompts/p4_sys.txt "mfma" 1
run_one E3 p5_ba  /root/gretacore/tools/benchmarks/prompts/p5_ba.txt  "mfma" 1

# E4: force GEMM + VALU
run_one E4 p4_sys /root/gretacore/tools/benchmarks/prompts/p4_sys.txt "valu" 1
run_one E4 p5_ba  /root/gretacore/tools/benchmarks/prompts/p5_ba.txt  "valu" 1

unset GRETA_TRACE_PROMPT_ID
unset GRETA_QKV_FORCE_ROUTE
unset GRETA_QKV_FORCE_GEMM

mkdir -p /root

tar -czf /root/gretacore_b3_31_artifacts.tgz \
  $OUTDIR/b3_31_*.log \
  $OUTDIR/b3_31_*_attn_l0_pipe_*.jsonl

ls -lh /root/gretacore_b3_31_artifacts.tgz
EOF_REMOTE

mkdir -p "$OUTDIR_LOCAL"
scp root@"$IP":/root/gretacore_b3_31_artifacts.tgz "$OUTDIR_LOCAL/"

tar -xzf "$OUTDIR_LOCAL/gretacore_b3_31_artifacts.tgz" -C "$OUTDIR_LOCAL/"

python3 tools/benchmarks/analyze_b3_31_qkv_route_matrix.py \
  --dir "$OUTDIR_LOCAL/root/gretacore/artifacts/alignment/$DATE" \
  --out "$OUTDIR_LOCAL/b3_31_analysis.txt"
