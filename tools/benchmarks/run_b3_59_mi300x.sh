#!/bin/bash
# run_b3_59_mi300x.sh
# Embedding + StageDebugInput Zeroing Audit (MI300X Execution)

set -e

# Canonical Config
DATE=$(date +%Y-%m-%d)
B3_ID="b3_59"
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

export GRETA_VERBOSE_INFO=1
export GRETA_TRACE_STAGE=1
export GRETA_TRACE_STAGE_LAYERS="0" 
export GRETA_TRACE_STAGE_POINTS="embd_w_hash,embed_out,x_in"
export GRETA_TRACE_STAGE_PHASES="prefill_last,decode0"
export GRETA_TRACE_STAGE_DEBUG_INPUT=1

for prompt in "${PROMPTS[@]}"; do
    prompt_name=$(basename "$prompt" .txt)
    echo "--- Running $prompt_name ---"
    
    export GRETA_TRACE_STAGE_OUT="$OUT_DIR/traces/${prompt_name}_trace.jsonl"
    export GRETA_TRACE_PROMPT_ID="$prompt_name"
    
    $BINARY --model models/greta-v1.gguf --prompt-file "$prompt" --max-tokens 5 --greedy > "$OUT_DIR/run/${prompt_name}.log" 2>&1
done

echo "--- Generating report ---"
python3 tools/benchmarks/analyze_b3_59_embedding_debug_input.py \
    --input_dir "$OUT_DIR/traces" \
    --output "$OUT_DIR/b3_59_analysis.txt"

echo "--- Packaging artifacts ---"
tar -czf "$OUT_DIR/gretacore_${B3_ID}_artifacts.tgz" -C "$OUT_DIR" run traces b3_59_analysis.txt
echo "Done: $OUT_DIR/gretacore_${B3_ID}_artifacts.tgz"
