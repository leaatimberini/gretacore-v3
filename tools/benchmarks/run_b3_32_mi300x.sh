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

echo "=== ROCM-SMI PRE ==="
command -v rocm-smi >/dev/null && rocm-smi --showmemuse --showuse --showpids || true
echo "=== KILL CANDIDATES ==="
docker stop rocm-gpt-oss open-webui >/dev/null 2>&1 || true
pkill -f "vllm" || true
pkill -f "open-webui" || true
pkill -f "python.*vllm" || true
pkill -f "python.*gpt-oss" || true
pkill -f "python" || true
pkill -f "greta_infer" || true
pkill -f "greta_server" || true
sleep 2
echo "=== ROCM-SMI POST ==="
command -v rocm-smi >/dev/null && rocm-smi --showmemuse --showuse --showpids || true

cmake -S /root/gretacore/tools/inference -B /root/gretacore/tools/inference/build
cd /root/gretacore/tools/inference/build
make -B -j$(nproc)

OUTDIR=/root/gretacore/artifacts/alignment/2026-02-03
mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/b3_32_* || true

export GRETA_INT4_WEIGHTS=1
export GRETA_TRACE_ATTN_L0_PIPE=1
export GRETA_TRACE_ATTN_L0_NORM=1
export GRETA_TRACE_STAGE_DEBUG_INPUT=1

MODEL=/root/gretacore/models/llama3_8b_q4/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
BIN=/root/gretacore/tools/inference/build/greta_infer

run_one () {
  local prompt_id="$1"
  local prompt_file="$2"
  local max_seq="$3"
  local rc=0
  export GRETA_MAX_SEQ_LEN="$max_seq"
  export GRETA_TRACE_PROMPT_ID="$prompt_id"
  export GRETA_TRACE_ATTN_L0_PIPE_OUT="$OUTDIR/b3_32_attn_l0_pipe_${prompt_id}.jsonl"
  rm -f "$OUTDIR/b3_32_attn_l0_pipe_${prompt_id}.jsonl"
  set +e
  $BIN --model "$MODEL" --prompt-file "$prompt_file" --max-tokens 2 --greedy \
    2>&1 | tee "$OUTDIR/b3_32_${prompt_id}.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "RUN_FAILED ${prompt_id} rc=${rc}" >&2
    return $rc
  fi
}

fail=0
run_one p0_short /root/gretacore/tools/benchmarks/prompts/p0_short_hi.txt 256 || fail=1
run_one p4_sys   /root/gretacore/tools/benchmarks/prompts/p4_sys.txt 256 || fail=1
run_one p5_ba    /root/gretacore/tools/benchmarks/prompts/p5_ba.txt 256 || fail=1
run_one p6_long  /root/gretacore/tools/benchmarks/prompts/p6_long.txt 768 || fail=1

unset GRETA_TRACE_PROMPT_ID

mkdir -p /root

tar -czf /root/gretacore_b3_32_artifacts.tgz \
  $OUTDIR/b3_32_*.log \
  $OUTDIR/b3_32_attn_l0_pipe_*.jsonl \
  2>/dev/null || true

ls -lh /root/gretacore_b3_32_artifacts.tgz

exit $fail
EOF_REMOTE

mkdir -p "$OUTDIR_LOCAL"
scp root@"$IP":/root/gretacore_b3_32_artifacts.tgz "$OUTDIR_LOCAL/"

tar -xzf "$OUTDIR_LOCAL/gretacore_b3_32_artifacts.tgz" -C "$OUTDIR_LOCAL/"

python3 tools/benchmarks/analyze_b3_32_normout_vs_q.py \
  --dir "$OUTDIR_LOCAL/root/gretacore/artifacts/alignment/$DATE" \
  --out "$OUTDIR_LOCAL/b3_32_analysis.txt"
