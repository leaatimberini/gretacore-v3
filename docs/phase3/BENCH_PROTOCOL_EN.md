# GRETA CORE - Phase 3 Benchmark Protocol

This protocol defines the methodology for measuring performance and correctness on AMD Instinct™ hardware.

## 1. Metrics Definition
- **TTFT (Time To First Token)**: Initial latency from prompt submission to the first generated token (ms). Measures prefill efficiency.
- **T/s (Tokens per Second)**: Average generation throughput. Measures decode efficiency.
- **ms/layer**: Execution time per transformer layer (ms).
- **Correctness**: Absolute parity check against a reference output (no NaNs, stable logits).

## 2. Experimental Matrix
Every benchmark run must iterate through the following configuration space:

| Variable | Values | Description |
| :--- | :--- | :--- |
| **HIP Graphs** | `ON` (1), `OFF` (0) | Evaluates host-side scheduling overhead reduction. |
| **Specialized LM_HEAD**| `ON` (1), `OFF` (0) | Compares GEMV-based output vs. standard GEMM. |
| **Prompts** | Short, Medium | "Hi", "What is the capital of France?" |

## 3. Environment Control
- **GPU Frequency**: Fixed to maximum (e.g., via `rocm-smi`).
- **Cooling**: Ensure no thermal throttling.
- **Variables**:
    - `GRETA_HIP_GRAPH`
    - `GRETA_USE_SPECIALIZED_LM_HEAD`
    - `GRETA_VERBOSE_INFO=1`

## 4. Data Collection
Results are gathered using `tools/phase3/run_bench.sh` and saved in `bench_results.csv` following the schema in `docs/phase3/bench_schema.csv`.

### CSV Columns:
`timestamp, model, prompt, max_tokens, greedy, hip_graph, spec_lm_head, ttft_ms, tokens_per_sec, total_ms`
