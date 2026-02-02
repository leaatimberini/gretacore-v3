import subprocess
import time
import sys
import os
import re
import json

def run_bench(model_path, prompt, max_tokens=256, use_demo=False):
    env = os.environ.copy()
    env["GRETA_INT4_WEIGHTS"] = "1"
    env["GRETA_PROFILE_BLOCKS"] = "0"
    
    cmd = [
        "./greta_infer",
        "--model", model_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--greedy"
    ]
    if use_demo:
        cmd.append("--demo-tokenizer")
    
    print(f"Running benchmark: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=".")
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error: {stderr}")
        return None
        
    # Extraer métricas del stdout
    ttft = 0
    tokens = 0
    tps_e2e = 0
    total_ms = 0
    
    for line in stdout.split('\n'):
        if "Time to first token:" in line:
            m = re.search(r"([\d.]+)", line)
            if m: ttft = float(m.group(1))
        elif "Generated tokens:" in line:
            m = re.search(r"(\d+)", line)
            if m: tokens = int(m.group(1))
        elif "Tokens/second:" in line:
            m = re.search(r"([\d.]+)", line)
            if m: tps_e2e = float(m.group(1))
        elif "Total time:" in line:
            m = re.search(r"([\d.]+)", line)
            if m: total_ms = float(m.group(1))

    # Calcular steady-state decode real (excluyendo TTFT del total)
    # TotalTime = TTFT + DecodeTime
    decode_ms = total_ms - ttft
    # El primer token ya está en TTFT, así que quedan (tokens - 1) tokens en decode_ms
    decode_tok_s = (tokens - 1) / (decode_ms / 1000.0) if decode_ms > 0 and tokens > 1 else 0
    
    return {
        "ttft": ttft,
        "tokens": tokens,
        "decode_tok_s_steady": decode_tok_s,
        "tps_e2e": tps_e2e,
        "total_time_ms": total_ms,
        "tokenizer": "ASCII" if use_demo else "SentencePiece"
    }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 bench_harness.py <model_path> <prompt> [--json]")
        sys.exit(1)
        
    model = sys.argv[1]
    prompt = sys.argv[2]
    as_json = "--json" in sys.argv
    
    # Baseline Compute (Demo Tokenizer)
    res_compute = run_bench(model, prompt, use_demo=True)
    
    # E2E (SentencePiece)
    res_e2e = run_bench(model, prompt, use_demo=False)
    
    if as_json:
        print(json.dumps({
            "compute_only": res_compute,
            "e2e": res_e2e,
            "metadata": {
                "gpu": "MI300X",
                "rocm": "7.x",
                "timestamp": time.time()
            }
        }, indent=2))
    else:
        print("\n" + "="*40)
        print("       GRETA CORE BENCHMARK RESULTS")
        print("="*40)
        if res_compute:
            print(f"TTFT (Compute):    {res_compute['ttft']:.2f} ms")
            print(f"Decode (Steady):   {res_compute['decode_tok_s_steady']:.2f} tok/s")
        if res_e2e:
            print(f"TTFT (E2E):        {res_e2e['ttft']:.2f} ms")
            print(f"Tokens Gen:        {res_e2e['tokens']}")
            print(f"Tokenizer:         {res_e2e['tokenizer']}")
        print("="*40)
