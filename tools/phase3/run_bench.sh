#!/bin/bash
# GRETA CORE - Phase 3.3 Benchmark Suite
# Usage: ./run_bench.sh [model_path]

MODEL_PATH=${1:-"/root/models/llama-2-7b-chat.Q4_K_M.gguf"}
OUTPUT_CSV="bench_results.csv"

echo "timestamp,model,prompt,max_tokens,greedy,hip_graph,spec_lm_head,ttft_ms,tokens_per_sec,total_ms" > $OUTPUT_CSV

PROMPTS=("Hi" "What is the capital of France?")
MAX_TOKENS=(32)

# Matrix: Graph (0, 1) x SpecLM (0, 1)
for graph in 0 1; do
    for speclm in 0 1; do
        for prompt in "${PROMPTS[@]}"; do
            for tokens in "${MAX_TOKENS[@]}"; do
                echo "Running: Graph=$graph, SpecLM=$speclm, Tokens=$tokens, Prompt='$prompt'"
                
                # Run and capture output
                # We use GRETA_VERBOSE_INFO=1 to confirm system state
                OUT=$(GRETA_HIP_GRAPH=$graph GRETA_USE_SPECIALIZED_LM_HEAD=$speclm GRETA_VERBOSE_INFO=1 ./greta_infer --model "$MODEL_PATH" --prompt "$prompt" --max-tokens "$tokens" --greedy 2>&1)
                
                # Extract metrics
                TTFT=$(echo "$OUT" | grep "Time to first token:" | awk '{print $5}')
                TPS=$(echo "$OUT" | grep "Tokens/second:" | awk '{print $2}')
                TOTAL=$(echo "$OUT" | grep "Total time:" | awk '{print $3}')
                TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
                
                echo "$TIMESTAMP,$MODEL_PATH,\"$prompt\",$tokens,1,$graph,$speclm,$TTFT,$TPS,$TOTAL" >> $OUTPUT_CSV
            done
        done
    done
done

echo "Benchmark complete. Results saved to $OUTPUT_CSV"
