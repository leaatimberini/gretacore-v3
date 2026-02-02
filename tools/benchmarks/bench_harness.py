import subprocess
import time
import sys
import re
import os

def run_bench(model_path, prompt, max_tokens=256):
    env = os.environ.copy()
    env["GRETA_INT4_WEIGHTS"] = "1"
    env["GRETA_PROFILE_BLOCKS"] = "0" # Disable for clean throughput
    
    cmd = [
        "./greta_infer",
        "--model", model_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--greedy"
    ]
    
    print(f"Running benchmark: {' '.join(cmd)}")
    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=".")
    stdout, stderr = process.communicate()
    end_time = time.time()
    
    if process.returncode != 0:
        print(f"Error: {stderr}")
        return None
        
    # Extract stats
    stats = {}
    pattern_ttft = r"Time to first token: ([\d.]+) ms"
    pattern_tokens = r"Generated tokens: (\d+)"
    pattern_total_time = r"Total time: ([\d.]+) ms"
    
    m_ttft = re.search(pattern_ttft, stdout)
    m_tokens = re.search(pattern_tokens, stdout)
    m_total = re.search(pattern_total_time, stdout)
    
    if m_ttft: stats["ttft"] = float(m_ttft.group(1))
    if m_tokens: stats["tokens"] = int(m_tokens.group(1))
    if m_total: stats["total_time_ms"] = float(m_total.group(1))
    
    # Calculate steady-state (assuming warmup of 32 tokens)
    # TotalTime = TTFT + (T - 1) * AvgDecodeTime
    # We want to be more precise if possible, but greta_infer doesn't yet log per-token time.
    # For now, we report the average over all but the first token.
    if "total_time_ms" in stats and "tokens" in stats and stats["tokens"] > 1:
        decode_time = stats["total_time_ms"] - stats["ttft"]
        stats["decode_tok_s"] = (stats["tokens"] - 1) / (decode_time / 1000.0)
    
    return stats

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 bench_harness.py <model_path> <prompt>")
        sys.exit(1)
        
    model = sys.argv[1]
    prompt = sys.argv[2]
    
    results = run_bench(model, prompt)
    if results:
        print("\n" + "="*40)
        print("       GRETA CORE BENCHMARK RESULTS")
        print("="*40)
        print(f"TTFT:            {results.get('ttft'):.2f} ms")
        print(f"Tokens Gen:       {results.get('tokens')}")
        print(f"Decode (Steady):  {results.get('decode_tok_s'):.2f} tok/s")
        print(f"Total Time:      {results.get('total_time_ms'):.2f} ms")
        print("="*40)
