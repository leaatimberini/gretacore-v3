#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


def parse_name(path: Path) -> Tuple[str, str]:
    name = path.name.replace("b3_40_wo_resid_", "").replace(".jsonl", "")
    parts = name.split("_")
    if len(parts) < 2:
        return name, "E1W0"
    exp = parts[-1]
    prompt = "_".join(parts[:-1])
    return prompt, exp


def analyze_file(path: Path):
    pre: Dict[str, List[float]] = {}
    dec: Dict[str, List[float]] = {}
    logits_prefill = None
    logits_decode = None
    wo_pre = None
    wo_dec = None
    for r in load_rows(path):
        if r.get("event") == "stage_trace":
            phase = r.get("phase")
            point = r.get("point")
            if phase not in ("prefill_last", "decode0"):
                continue
            if point:
                if phase == "prefill_last":
                    pre[point] = r.get("sample", [])
                else:
                    dec[point] = r.get("sample", [])
        elif r.get("event") == "stage_logits":
            if r.get("phase") == "prefill_last":
                logits_prefill = r
            elif r.get("phase") == "decode0":
                logits_decode = r
        elif r.get("event") == "wo_verify":
            if r.get("phase") == "prefill_last":
                wo_pre = r
            elif r.get("phase") == "decode0":
                wo_dec = r
    return pre, dec, logits_prefill, logits_decode, wo_pre, wo_dec


def format_pair(a: Optional[float], b: Optional[float]) -> str:
    if a is None and b is None:
        return "n/a"
    if b is None:
        return f"{a}"
    if a is None:
        return f"n/a/{b}"
    return f"{a}/{b}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    base = Path(args.dir)
    files = sorted(base.glob("b3_40_wo_resid_*_E1W*.jsonl"))

    lines = []
    lines.append(f"B3.40 WO fix analysis: {base}")
    lines.append(
        "prompt\texp\two_out_mae\tx_after_attn_mae\two_layout_best\two_layout_used\two_mae_row\two_mae_col\tprefill_last_top1\tdecode0_top1\tmatch"
    )

    for path in files:
        prompt, exp = parse_name(path)
        pre, dec, lp, ld, wo_pre, wo_dec = analyze_file(path)
        if not pre or not dec:
            continue

        wo_out_mae = mae(pre.get("wo_out", []), dec.get("wo_out", []))
        x_after_mae = mae(pre.get("x_after_attn", []), dec.get("x_after_attn", []))

        pre_top1 = lp.get("top1_id", -1) if lp else -1
        dec_top1 = ld.get("top1_id", -1) if ld else -1
        match = pre_top1 == dec_top1 if pre_top1 >= 0 and dec_top1 >= 0 else False

        layout_best = None
        layout_used = None
        mae_row = None
        mae_col = None
        if wo_pre or wo_dec:
            pre_best = wo_pre.get("wo_layout_best") if wo_pre else "n/a"
            dec_best = wo_dec.get("wo_layout_best") if wo_dec else "n/a"
            layout_best = f"{pre_best}|{dec_best}"
            layout_used = wo_pre.get("wo_layout_used") if wo_pre else (wo_dec.get("wo_layout_used") if wo_dec else "auto")
            mae_row = format_pair(
                wo_pre.get("wo_mae_row") if wo_pre else None,
                wo_dec.get("wo_mae_row") if wo_dec else None,
            )
            mae_col = format_pair(
                wo_pre.get("wo_mae_col") if wo_pre else None,
                wo_dec.get("wo_mae_col") if wo_dec else None,
            )

        lines.append(
            f"{prompt}\t{exp}\t{wo_out_mae}\t{x_after_mae}\t{layout_best}\t{layout_used}\t{mae_row}\t{mae_col}\t{pre_top1}\t{dec_top1}\t{str(match).lower()}"
        )

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
