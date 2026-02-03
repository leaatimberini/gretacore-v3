import subprocess
import time
import sys
import os
import re
import json
import statistics

def run_bench(model_path, prompt, max_tokens=128, perhead=True):
    env = os.environ.copy()
    env["GRETA_PERHEAD_QKV"] = "1" if perhead else "0"
    
    cmd = [
        "tools/inference/build/greta_infer",
        "--model", model_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--greedy"
    ]
    
    print(f"Benchmarking {'PERHEAD' if perhead else 'BASELINE'}...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error: {stderr}")
        return None
        
    ttft = 0
    total_ms = 0
    tokens = 0
    
    for line in stdout.split('\n'):
        if "Time to first token:" in line:
            m = re.search(r"([\d.]+)", line)
            if m: ttft = float(m.group(1))
        elif "Total time:" in line:
            m = re.search(r"([\d.]+)", line)
            if m: total_ms = float(m.group(1))
        elif "Generated tokens:" in line:
            m = re.search(r"(\d+)", line)
            if m: tokens = int(m.group(1))
            
    decode_ms = total_ms - ttft
    tps = (tokens - 1) / (decode_ms / 1000.0) if decode_ms > 0 and tokens > 1 else 0
    return tps

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 overhead_bench.py <model_path>")
        sys.exit(1)
        
    model = sys.argv[1]
    prompt = "Describe the importance of low-latency in LLM inference."
    n_runs = 3
    steps = 128
    
    baseline_tps = []
    print("Collecting Baseline (PH=0)...")
    for _ in range(n_runs):
        tps = run_bench(model, prompt, max_tokens=steps, perhead=False)
        if tps: baseline_tps.append(tps)
        
    perhead_tps = []
    print("Collecting Per-Head (PH=1)...")
    for _ in range(n_runs):
        tps = run_bench(model, prompt, max_tokens=steps, perhead=True)
        if tps: perhead_tps.append(tps)
        
    if not baseline_tps or not perhead_tps:
        print("Failed to collect benchmark data.")
        return
        
    avg_baseline = statistics.mean(baseline_tps)
    avg_perhead = statistics.mean(perhead_tps)
    overhead = (avg_baseline - avg_perhead) / avg_baseline * 100
    
    print("\nBenchmark Results:")
    print(f"  Baseline TPS: {avg_baseline:.2f}")
    print(f"  Per-Head TPS: {avg_perhead:.2f}")
    print(f"  Overhead:     {overhead:.2f}%")
    
    if overhead <= 2.0:
        print("SUCCESS: Overhead is within the 2% limit.")
    else:
        print("WARNING: Overhead exceeds 2%. Optimization required.")

if __name__ == "__main__":
    main()
