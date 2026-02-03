import subprocess
import json
import os
import sys
import argparse
import statistics

def run_infer(model, prompt, max_tokens, perhead=True):
    env = os.environ.copy()
    env["GRETA_PERHEAD_QKV"] = "1" if perhead else "0"
    
    cmd = [
        "tools/inference/build/greta_infer",
        "--model", model,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--alignment",
        "--greedy" # Use greedy for deterministic comparison
    ]
    
    print(f"Running {'PERHEAD' if perhead else 'BASELINE'}...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    
    steps = []
    for line in process.stdout:
        if line.startswith("[ALIGNMENT_STEP]"):
            try:
                data = json.loads(line.replace("[ALIGNMENT_STEP]", "").strip())
                steps.append(data)
            except json.JSONDecodeError:
                pass
    
    process.wait()
    if process.returncode != 0:
        print(f"Error running greta_infer: {process.stderr.read()}")
        return None
        
    return steps

def compare(baseline, perhead):
    if not baseline or not perhead:
        return None
    
    n = min(len(baseline), len(perhead))
    matches = 0
    stats_diff = {
        "min": [],
        "max": [],
        "avg": []
    }
    
    for i in range(n):
        b = baseline[i]
        p = perhead[i]
        
        if b["token_id"] == p["token_id"]:
            matches += 1
            
        stats_diff["min"].append(abs(b["stats"]["min"] - p["stats"]["min"]))
        stats_diff["max"].append(abs(b["stats"]["max"] - p["stats"]["max"]))
        stats_diff["avg"].append(abs(b["stats"]["avg"] - p["stats"]["avg"]))
        
    return {
        "match_rate": matches / n if n > 0 else 0,
        "steps_compared": n,
        "avg_min_diff": statistics.mean(stats_diff["min"]) if stats_diff["min"] else 0,
        "avg_max_diff": statistics.mean(stats_diff["max"]) if stats_diff["max"] else 0,
        "avg_avg_diff": statistics.mean(stats_diff["avg"]) if stats_diff["avg"] else 0,
        "perhead_nans": sum(s["stats"]["nan"] for s in perhead),
        "perhead_infs": sum(s["stats"]["inf"] for s in perhead)
    }

def main():
    parser = argparse.ArgumentParser(description="GRETA Alignment Runner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Describe the importance of low-latency in LLM inference.")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--output", default="alignment_results.json")
    args = parser.parse_args()
    
    baseline_steps = run_infer(args.model, args.prompt, args.steps, perhead=False)
    perhead_steps = run_infer(args.model, args.prompt, args.steps, perhead=True)
    
    results = compare(baseline_steps, perhead_steps)
    
    if results:
        report = {
            "model": args.model,
            "prompt": args.prompt,
            "steps": args.steps,
            "results": results
        }
        
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
            
        print("\nAlignment Report:")
        print(f"  Top-1 Match Rate: {results['match_rate']:.2%}")
        print(f"  Avg Logit Avg Diff: {results['avg_avg_diff']:.6f}")
        print(f"  NaNs detected: {results['perhead_nans']}")
        print(f"  Infs detected: {results['perhead_infs']}")
        print(f"\nFull results saved to {args.output}")
    else:
        print("Failed to collect alignment data.")

if __name__ == "__main__":
    main()
