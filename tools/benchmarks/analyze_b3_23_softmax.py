#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean

def load_rows(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize(rows):
    grouped = {}
    for i, r in enumerate(rows):
        key = r.get("prompt_id") or f"line_{i}"
        grouped.setdefault(key, []).append(r)

    summary = []
    for key, items in grouped.items():
        qk_mae = mean([it.get("qk_mae", 0.0) for it in items])
        qk_max = max([it.get("qk_max_diff", 0.0) for it in items] + [0.0])
        sm_mae = mean([it.get("softmax_mae", 0.0) for it in items])
        sm_max = max([it.get("softmax_max_diff", 0.0) for it in items] + [0.0])
        summary.append((key, qk_mae, qk_max, sm_mae, sm_max))
    return summary


def verdict(qk_mae, sm_mae):
    if qk_mae < 1e-5 and sm_mae > 1e-3:
        return "softmax/scaling"
    if qk_mae > 1e-3:
        return "pre-softmax (Q/K)"
    return "mixed/low"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    path = Path(args.jsonl)
    rows = load_rows(path)
    summary = summarize(rows)

    lines = []
    lines.append(f"Input: {path}")
    lines.append("prompt_id\tqk_mae\tqk_max_diff\tsoftmax_mae\tsoftmax_max_diff\tverdict")
    for key, qk_mae, qk_max, sm_mae, sm_max in summary:
        lines.append(
            f"{key}\t{qk_mae:.6g}\t{qk_max:.6g}\t{sm_mae:.6g}\t{sm_max:.6g}\t{verdict(qk_mae, sm_mae)}"
        )

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
