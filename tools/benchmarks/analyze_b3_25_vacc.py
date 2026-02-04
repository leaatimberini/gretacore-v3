#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean


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


def summarize_vacc(rows):
    grouped = {}
    for i, r in enumerate(rows):
        key = r.get("prompt_id") or f"line_{i}"
        grouped.setdefault(key, []).append(r)

    summary = {}
    for key, items in grouped.items():
        v_layout = items[-1].get("v_layout_probe", {}).get("v_layout_best", "")
        pv_mae = mean([it.get("pv_mae", 0.0) for it in items])
        pv_max = max([it.get("pv_max_diff", 0.0) for it in items] + [0.0])
        attn_mae = mean([it.get("attn_out_mae", 0.0) for it in items])
        attn_max = max([it.get("attn_out_max_diff", 0.0) for it in items] + [0.0])
        summary[key] = {
            "v_layout_best": v_layout,
            "pv_mae": pv_mae,
            "pv_max_diff": pv_max,
            "attn_out_mae": attn_mae,
            "attn_out_max_diff": attn_max,
        }
    return summary


def extract_top1(path: Path):
    rows = load_rows(path)
    prefill = None
    decode = None
    for r in rows:
        phase = r.get("phase")
        if phase == "prefill_last":
            prefill = r.get("top1_id")
        elif phase == "decode0":
            decode = r.get("top1_id")
    return prefill, decode


def verdict(v_layout, pv_mae, attn_mae):
    if v_layout and v_layout != "row":
        return "V layout mismatch"
    if pv_mae > 1e-3:
        return "P*V accumulation"
    if attn_mae > 1e-3:
        return "writeback/head-combine"
    return "low/mixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vacc-jsonl", required=True)
    ap.add_argument("--p4-jsonl", required=True)
    ap.add_argument("--p5-jsonl", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    vacc_rows = load_rows(Path(args.vacc_jsonl))
    vacc_summary = summarize_vacc(vacc_rows)

    p4_prefill, p4_decode = extract_top1(Path(args.p4_jsonl))
    p5_prefill, p5_decode = extract_top1(Path(args.p5_jsonl))

    lines = []
    lines.append(f"VACC JSONL: {args.vacc_jsonl}")
    lines.append(
        "prompt_id\tv_layout_best\tpv_mae\tpv_max_diff\tattn_out_mae\tattn_out_max_diff\tprefill_last_top1\tdecode0_top1\tverdict"
    )

    for key, stats in vacc_summary.items():
        prefill = p4_prefill if key == "p4_sys" else p5_prefill if key == "p5_ba" else None
        decode = p4_decode if key == "p4_sys" else p5_decode if key == "p5_ba" else None
        lines.append(
            f"{key}\t{stats['v_layout_best']}\t{stats['pv_mae']:.6g}\t{stats['pv_max_diff']:.6g}\t{stats['attn_out_mae']:.6g}\t{stats['attn_out_max_diff']:.6g}\t{prefill}\t{decode}\t{verdict(stats['v_layout_best'], stats['pv_mae'], stats['attn_out_mae'])}"
        )

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
