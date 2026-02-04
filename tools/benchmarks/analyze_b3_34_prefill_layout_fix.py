#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

STAGES = ["attn_norm_out", "q", "k", "v", "qk", "softmax", "pv", "attn_out"]


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


def mae(a: List[float], b: List[float]) -> Optional[float]:
    if not a or not b:
        return None
    n = min(len(a), len(b))
    if n == 0:
        return None
    return sum(abs(a[i] - b[i]) for i in range(n)) / n


def analyze_file(path: Path):
    pre = None
    dec = None
    for r in load_rows(path):
        if r.get("event") != "attn_l0_pipe":
            continue
        if r.get("phase") == "prefill_last":
            pre = r
        elif r.get("phase") == "decode0":
            dec = r
    return pre, dec


def first_mismatch(pre: Dict, dec: Dict) -> Tuple[str, Optional[float]]:
    for stage in STAGES:
        m = mae(pre.get(stage, []), dec.get(stage, []))
        if m is not None and m > 1e-7:
            return stage, m
    return "NONE", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    base = Path(args.dir)
    files = sorted(base.glob("b3_34_E*_attn_l0_pipe_*.jsonl"))
    lines = []
    lines.append(f"B3.34 prefill layout fix analysis: {base}")
    lines.append(
        "prompt\texp\tprefill_layout\tdecode_layout\tfirst_mismatch\tq_mae\tprefill_top1\tdecode0_top1"
    )

    for path in files:
        name = path.name
        try:
            exp = name.split("_attn_l0_pipe_")[0].split("b3_34_")[1]
            prompt = name.split("_attn_l0_pipe_")[1].split(".jsonl")[0]
        except Exception:
            continue
        pre, dec = analyze_file(path)
        if not pre or not dec:
            continue

        pre_layout = pre.get("q_weight_layout_best", "unknown")
        dec_layout = dec.get("q_weight_layout_best", "unknown")
        q_mae = mae(pre.get("q", []), dec.get("q", []))
        first, _ = first_mismatch(pre, dec)
        lines.append(
            f"{prompt}\t{exp}\t{pre_layout}\t{dec_layout}\t{first}\t{q_mae}\tNA\tNA"
        )

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
