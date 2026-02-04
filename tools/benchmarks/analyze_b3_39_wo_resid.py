#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


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
    pre: Dict[str, List[float]] = {}
    dec: Dict[str, List[float]] = {}
    logits_prefill = None
    logits_decode = None
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
    return pre, dec, logits_prefill, logits_decode


def first_mismatch(wo_mae: Optional[float], x_after_mae: Optional[float]) -> str:
    if wo_mae is not None and wo_mae > 1e-6:
        return "wo_out"
    if x_after_mae is not None and x_after_mae > 1e-6:
        return "x_after_attn"
    return "NONE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    base = Path(args.dir)
    files = sorted(base.glob("b3_39_wo_resid_*_E1.jsonl"))

    lines = []
    lines.append(f"B3.39 WO vs residual analysis: {base}")
    lines.append(
        "prompt\texp\tattn_out_mae\two_out_mae\tx_in_mae\tx_after_attn_mae\tfirst_mismatch_stage\tprefill_last_top1\tdecode0_top1\tcollapse_96965"
    )

    for path in files:
        name = path.name
        try:
            core = name.replace("b3_39_wo_resid_", "").replace(".jsonl", "")
            prompt = core.replace("_E1", "")
            exp = "E1"
        except Exception:
            continue

        pre, dec, lp, ld = analyze_file(path)
        if not pre or not dec:
            continue

        attn_out_mae = mae(pre.get("attn_out", []), dec.get("attn_out", []))
        wo_out_mae = mae(pre.get("wo_out", []), dec.get("wo_out", []))
        x_in_mae = mae(pre.get("x_in", []), dec.get("x_in", []))
        x_after_mae = mae(pre.get("x_after_attn", []), dec.get("x_after_attn", []))

        first = first_mismatch(wo_out_mae, x_after_mae)

        pre_top1 = lp.get("top1_id", -1) if lp else -1
        dec_top1 = ld.get("top1_id", -1) if ld else -1
        collapse = False
        if dec_top1 == 96965:
            collapse = True

        lines.append(
            f"{prompt}\t{exp}\t{attn_out_mae}\t{wo_out_mae}\t{x_in_mae}\t{x_after_mae}\t{first}\t{pre_top1}\t{dec_top1}\t{str(collapse).lower()}"
        )

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
