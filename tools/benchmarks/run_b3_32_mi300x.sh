#!/usr/bin/env bash
set -euo pipefail

IP="${1:-129.212.184.200}"
DATE="${2:-2026-02-03}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR_LOCAL="$ROOT_DIR/artifacts_remote/${DATE}/b3_32"

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
export GRETA_MAX_SEQ_LEN=1024
export GRETA_TRACE_ATTN_L0_PIPE=1
export GRETA_TRACE_ATTN_L0_NORM=1
export GRETA_TRACE_STAGE_DEBUG_INPUT=1

MODEL=/root/gretacore/models/llama3_8b_q4/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
BIN=/root/gretacore/tools/inference/build/greta_infer

run_one () {
  local prompt_id="$1"
  local prompt_file="$2"
  export GRETA_TRACE_PROMPT_ID="$prompt_id"
  export GRETA_TRACE_ATTN_L0_PIPE_OUT="$OUTDIR/b3_32_attn_l0_pipe_${prompt_id}.jsonl"
  rm -f "$OUTDIR/b3_32_attn_l0_pipe_${prompt_id}.jsonl"
  $BIN --model "$MODEL" --prompt-file "$prompt_file" --max-tokens 2 --greedy \
    2>&1 | tee "$OUTDIR/b3_32_${prompt_id}.log"
}

run_one p0_short /root/gretacore/tools/benchmarks/prompts/p0_short_hi.txt
run_one p4_sys   /root/gretacore/tools/benchmarks/prompts/p4_sys.txt
run_one p5_ba    /root/gretacore/tools/benchmarks/prompts/p5_ba.txt
run_one p6_long  /root/gretacore/tools/benchmarks/prompts/p6_long.txt

unset GRETA_TRACE_PROMPT_ID

mkdir -p /root

tar -czf /root/gretacore_b3_32_artifacts.tgz \
  $OUTDIR/b3_32_*.log \
  $OUTDIR/b3_32_attn_l0_pipe_*.jsonl

ls -lh /root/gretacore_b3_32_artifacts.tgz
EOF_REMOTE

mkdir -p "$OUTDIR_LOCAL"
scp root@"$IP":/root/gretacore_b3_32_artifacts.tgz "$OUTDIR_LOCAL/"

tar -xzf "$OUTDIR_LOCAL/gretacore_b3_32_artifacts.tgz" -C "$OUTDIR_LOCAL/"

python3 tools/benchmarks/analyze_b3_32_normout_vs_q.py \
  --dir "$OUTDIR_LOCAL/root/gretacore/artifacts/alignment/$DATE" \
  --out "$OUTDIR_LOCAL/b3_32_analysis.txt"
