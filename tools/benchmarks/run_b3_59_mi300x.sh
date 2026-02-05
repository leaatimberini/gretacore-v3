#!/bin/bash
# run_b3_59_mi300x.sh
# Embedding + StageDebugInput Zeroing Audit

set -e

# Config
OUT_DIR="artifacts/b3_59_embedding_audit_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

PROMPTS=(
    "tools/benchmarks/prompts/p0_short.txt"
    "tools/benchmarks/prompts/p6_len_16.txt"
    "tools/benchmarks/prompts/p6_len_32.txt"
)

export GRETA_TRACE_STAGE=1
export GRETA_TRACE_STAGE_LAYERS="0" # Layer 0 is enough for embed_out and initial x_in
export GRETA_TRACE_STAGE_POINTS="embed_out,x_in"
export GRETA_TRACE_STAGE_PHASES="prefill_last,decode0"
export GRETA_TRACE_STAGE_DEBUG_INPUT=1

for prompt in "${PROMPTS[@]}"; do
    prompt_name=$(basename "$prompt" .txt)
    echo "--- Running $prompt_name ---"
    
    export GRETA_TRACE_STAGE_OUT="$OUT_DIR/${prompt_name}_trace.jsonl"
    export GRETA_TRACE_PROMPT_ID="$prompt_name"
    
    ./build/inference/greta_core_test --prompt_file "$prompt" --max_tokens 5
done

echo "--- Generating report ---"
python3 tools/benchmarks/analyze_b3_59_embedding_debug_input.py \
    --input_dir "$OUT_DIR" \
    --output "$OUT_DIR/report_b3_59.md"

echo "Artifacts saved to $OUT_DIR"
tar -czf "${OUT_DIR}.tgz" "$OUT_DIR"
echo "Done: ${OUT_DIR}.tgz"
