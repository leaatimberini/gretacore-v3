#!/usr/bin/env bash
set -euo pipefail

IP="${1:-129.212.184.200}"
DATE="${2:-2026-02-03}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR_LOCAL="$ROOT_DIR/artifacts_remote/${DATE}/b3_28"

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
export GRETA_TRACE_STAGE=1
export GRETA_TRACE_STAGE_DEBUG_INPUT=1
export GRETA_TRACE_STAGE_OUT=$OUTDIR/b3_28_stage.jsonl
export GRETA_TRACE_STAGE_LAYERS="0"
export GRETA_TRACE_STAGE_POINTS="x_in,logits"
export GRETA_TRACE_STAGE_PHASES="prefill_last,decode0"

MODEL=/root/gretacore/models/llama3_8b_q4/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
BIN=/root/gretacore/tools/inference/build/greta_infer

P4="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.\n<|eot_id|><|start_header_id|>user<|end_header_id|>\nHi\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
P5="Write one short sentence about Buenos Aires."

rm -f "$OUTDIR/b3_28_stage.jsonl"

export GRETA_TRACE_PROMPT_ID=p4_sys
$BIN --model "$MODEL" --prompt "$P4" --max-tokens 2 --greedy --debug-decode 1 \
  2>&1 | tee "$OUTDIR/b3_28_p4.log"

export GRETA_TRACE_PROMPT_ID=p5_ba
$BIN --model "$MODEL" --prompt "$P5" --max-tokens 2 --greedy --debug-decode 1 \
  2>&1 | tee "$OUTDIR/b3_28_p5.log"

unset GRETA_TRACE_PROMPT_ID

mkdir -p /root

tar -czf /root/gretacore_b3_28_artifacts.tgz \
  $OUTDIR/b3_28_stage.jsonl \
  $OUTDIR/b3_28_p4.log \
  $OUTDIR/b3_28_p5.log

ls -lh /root/gretacore_b3_28_artifacts.tgz
EOF_REMOTE

mkdir -p "$OUTDIR_LOCAL"
scp root@"$IP":/root/gretacore_b3_28_artifacts.tgz "$OUTDIR_LOCAL/"

tar -xzf "$OUTDIR_LOCAL/gretacore_b3_28_artifacts.tgz" -C "$OUTDIR_LOCAL/"

python3 tools/benchmarks/analyze_b3_27_stage.py \
  --jsonl "$OUTDIR_LOCAL/root/gretacore/artifacts/alignment/$DATE/b3_28_stage.jsonl" \
  --out "$OUTDIR_LOCAL/b3_28_analysis.txt"
