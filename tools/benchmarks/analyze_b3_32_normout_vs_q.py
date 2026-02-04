#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

STAGES = ["attn_norm_in", "attn_norm_out", "q", "k", "v"]


def load_rows(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def mae(a: List[float], b: List[float]):
    if not a or not b:
        return None
    n = min(len(a), len(b))
    if n == 0:
        return None
    return sum(abs(a[i] - b[i]) for i in range(n)) / n


def analyze_file(path: Path):
    rows = load_rows(path)
    pre = None
    dec = None
    for r in rows:
        if r.get("event") != "attn_l0_pipe":
            continue
        if r.get("phase") == "prefill_last":
            pre = r
        elif r.get("phase") == "decode0":
            dec = r
    return pre, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    base = Path(args.dir)
    files = sorted(base.glob("b3_32_attn_l0_pipe_*.jsonl"))
    expected = {"p0_short", "p4_sys", "p5_ba", "p6_long"}
    found = set()
    for f in files:
        name = f.name
        if "b3_32_attn_l0_pipe_" in name:
            prompt = name.split("b3_32_attn_l0_pipe_")[1].split(".jsonl")[0]
            found.add(prompt)
    missing = sorted(expected - found)
    if not files:
        print(f"No JSONL files found under {base}")
        if missing:
            print(f"MISSING_PROMPTS: {', '.join(missing)}")
        return

    lines = []
    lines.append(f"B3.32 norm_out vs Q analysis: {base}")
    lines.append("prompt\tnorm_out_mae\tq_mae\tfirst_mismatch_stage")

    for path in files:
        pre, dec = analyze_file(path)
        if not pre or not dec:
            continue
        prompt = pre.get("prompt_id", "unknown")
        norm_out_mae = mae(pre.get("attn_norm_out", []), dec.get("attn_norm_out", []))
        q_mae = mae(pre.get("q", []), dec.get("q", []))

        first = "NONE"
        for stage in STAGES:
            m = mae(pre.get(stage, []), dec.get(stage, []))
            if m is not None and m > 1e-7:
                first = stage
                break

        lines.append(
            f"{prompt}\t{(norm_out_mae if norm_out_mae is not None else 'NA')}\t{(q_mae if q_mae is not None else 'NA')}\t{first}"
        )

    if missing:
        lines.append(f"MISSING_PROMPTS\t{', '.join(missing)}")

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
