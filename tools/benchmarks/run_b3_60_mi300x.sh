#!/bin/bash
# run_b3_60_mi300x.sh
# Attention Block Input/Output Bisect Audit (B3.60)

set -e

# Canonical Config
DATE=$(date +%Y-%m-%d)
B3_ID="b3_60"
OUT_DIR="artifacts_remote/$DATE/$B3_ID"
mkdir -p "$OUT_DIR/run" "$OUT_DIR/traces"

PROMPTS=(
    "tools/benchmarks/prompts/p0_short.txt"
    "tools/benchmarks/prompts/p6_len_16.txt"
    "tools/benchmarks/prompts/p6_len_32.txt"
)

# Build Target
BINARY="./tools/inference/build/greta_infer"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary $BINARY not found. Build it in tools/inference/build first."
    exit 1
fi

# B3.60 Unified Flags
export GRETA_B3_60=1
export GRETA_TRACE_STAGE=1
export GRETA_TRACE_STAGE_LAYERS="0"
export GRETA_TRACE_STAGE_DEBUG_INPUT=1

# Attention Block Trace Points
export GRETA_TRACE_B3_60_ATTN_IN=1
export GRETA_TRACE_B3_60_QKV=1
export GRETA_TRACE_B3_60_KV_STATE=1
export GRETA_TRACE_B3_60_ATTN_OUT=1
export GRETA_TRACE_B3_60_RESIDUAL=1

# Trace Output Directory
export GRETA_TRACE_DIR="$OUT_DIR/traces"

for prompt in "${PROMPTS[@]}"; do
    prompt_name=$(basename "$prompt" .txt)
    echo "--- Running $prompt_name ---"
    
    export GRETA_TRACE_PROMPT_ID="$prompt_name"
    export GRETA_TRACE_STAGE_OUT="$OUT_DIR/traces/${prompt_name}_trace.jsonl"
    
    $BINARY --model models/greta-v1.gguf --prompt-file "$prompt" --max-tokens 5 --greedy > "$OUT_DIR/run/${prompt_name}.log" 2>&1
done

echo "--- Generating report ---"
python3 tools/benchmarks/analyze_b3_60_attention_block.py \
    --input_dir "$OUT_DIR/traces" \
    --output "$OUT_DIR/b3_60_analysis.txt"

echo "--- Packaging artifacts ---"
tar -czf "$OUT_DIR/gretacore_${B3_ID}_artifacts.tgz" -C "$OUT_DIR" run traces b3_60_analysis.txt
echo "Done: $OUT_DIR/gretacore_${B3_ID}_artifacts.tgz"
