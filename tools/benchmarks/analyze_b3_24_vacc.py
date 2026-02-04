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
        v_layout = items[-1].get("v_layout_probe", {}).get("v_layout_best", "")
        pv_mae = mean([it.get("pv_mae", 0.0) for it in items])
        pv_max = max([it.get("pv_max_diff", 0.0) for it in items] + [0.0])
        attn_mae = mean([it.get("attn_out_mae", 0.0) for it in items])
        attn_max = max([it.get("attn_out_max_diff", 0.0) for it in items] + [0.0])
        summary.append((key, v_layout, pv_mae, pv_max, attn_mae, attn_max))
    return summary


def verdict(v_layout, pv_mae, attn_mae):
    if v_layout == "col":
        return "V layout mismatch"
    if pv_mae > 1e-3:
        return "P*V accumulation"
    if attn_mae > 1e-3:
        return "writeback/head-combine"
    return "low/mixed"


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
    lines.append(
        "prompt_id\tv_layout_best\tpv_mae\tpv_max_diff\tattn_out_mae\tattn_out_max_diff\tverdict"
    )
    for key, v_layout, pv_mae, pv_max, attn_mae, attn_max in summary:
        lines.append(
            f"{key}\t{v_layout}\t{pv_mae:.6g}\t{pv_max:.6g}\t{attn_mae:.6g}\t{attn_max:.6g}\t{verdict(v_layout, pv_mae, attn_mae)}"
        )

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
