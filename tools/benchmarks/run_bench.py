#!/usr/bin/env python3
import json
import subprocess
import os
import sys

# ES — Runner de Benchmarks GRETA CORE
# EN — GRETA CORE Benchmark Runner

def run_command(cmd, env=None):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=env)
    stdout, stderr = process.communicate()
    return stdout.decode(), stderr.decode(), process.returncode

def parse_stats(output):
    stats = {}
    for line in output.split('\n'):
        if "Prompt tokens:" in line:
            stats['prompt_tokens'] = int(line.split(':')[-1].strip())
        if "Generated tokens:" in line:
            stats['generated_tokens'] = int(line.split(':')[-1].strip())
        if "Tokens/second:" in line:
            stats['tps'] = float(line.split(':')[-1].strip())
        if "Time to first token:" in line:
            stats['ttft'] = float(line.split(':')[-1].strip().split()[0])
    return stats

def main():
    # Load reference H100 data
    with open('reference_h100_cuda.json', 'r') as f:
        reference = json.load(f)

    # Simplified scenario execution (since we don't assume PyYAML)
    scenarios = [
        {"name": "Llama-3-8B Prefill Short", "bs": 1, "tokens": 128},
        {"name": "Llama-3-8B Decode Batch 1", "bs": 1, "tokens": 32},
    ]

    print(f"\n{'='*80}")
    print(f"{'GRETA CORE BENCHMARK REPORT':^80}")
    print(f"{'Target: MI300X vs Reference H100':^80}")
    print(f"{'='*80}\n")

    for s in scenarios:
        print(f"[*] Running: {s['name']} (Batch Size: {s['bs']})")
        
        env = os.environ.copy()
        env["GRETA_PROFILE_BLOCKS"] = "1"
        
        # Note: In a real run we would use actual weights, here we might run in demo mode if weights are missing
        cmd = f"../inference/build/greta_infer --batch-size {s['bs']} --max-tokens {s['tokens']}"
        
        stdout, stderr, code = run_command(cmd, env=env)
        
        if code != 0:
            print(f" [!] Error running benchmark: {stderr}")
            continue

        stats = parse_stats(stdout)
        
        # Find competitive reference
        ref_val = "N/A"
        if "Decode" in s['name'] and s['bs'] == 1:
            ref_val = reference['benchmarks']['decode'][0]['tokens_per_second']
            perf_ratio = (stats['tps'] / ref_val) * 100
            print(f"  - GRETA MI300X: {stats['tps']:.2f} tok/s")
            print(f"  - CUDA H100 Ref: {ref_val:.2f} tok/s")
            print(f"  - Performance: {perf_ratio:.1f}% of H100")
        
        print(f"{'-'*80}")

    print("\n[✔] Benchmark complete. Audit logs captured in GRETA_L1_AUDIT.\n")

if __name__ == "__main__":
    main()
