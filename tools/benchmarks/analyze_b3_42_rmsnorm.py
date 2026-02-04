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
    name = path.name.replace("b3_42_rmsnorm_", "").replace(".jsonl", "")
    parts = name.split("_")
    if len(parts) < 2:
        return name, "E1"
    exp = parts[-1]
    prompt = "_".join(parts[:-1])
    return prompt, exp


def bucketize(input_mae: Optional[float], weight_mae: Optional[float], output_mae: Optional[float],
              input_ptr_match: bool, input_offset_match: bool, eps_match: bool,
              sumsq_delta: Optional[float]) -> str:
    # A: input buffer/offset mismatch
    if (not input_ptr_match) or (not input_offset_match) or (input_mae is not None and input_mae > 1e-6):
        return "A_input_selection"
    # C: weight mismatch
    if weight_mae is not None and weight_mae > 1e-6:
        return "C_weight_layout"
    # B: eps/stat mismatch
    if not eps_match or (sumsq_delta is not None and sumsq_delta > 1e-6):
        return "B_norm_stats"
    # D/E: math/precision path
    if output_mae is not None and output_mae > 1e-6:
        return "D_kernel_math"
    return "NONE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    base = Path(args.dir)
    files = sorted(base.glob("b3_42_rmsnorm_*_E1.jsonl"))

    lines = []
    lines.append(f"B3.42 RMSNorm analysis: {base}")
    lines.append(
        "prompt\texp\tinput_mae\tweight_mae\toutput_mae\teps_match\tsumsq_delta\tinv_rms_delta\tinput_ptr_match\tinput_offset_match\tweight_hash_match\troot_cause_bucket"
    )

    for path in files:
        prompt, exp = parse_name(path)
        pre = None
        dec = None
        for r in load_rows(path):
            if r.get("event") != "rmsnorm_trace":
                continue
            if r.get("phase") == "prefill_last":
                pre = r
            elif r.get("phase") == "decode0":
                dec = r
        if not pre or not dec:
            continue

        input_mae = mae(pre.get("input_sample", []), dec.get("input_sample", []))
        weight_mae = mae(pre.get("weight_sample", []), dec.get("weight_sample", []))
        output_mae = mae(pre.get("output_sample", []), dec.get("output_sample", []))

        eps_match = abs(pre.get("eps", 0.0) - dec.get("eps", 0.0)) < 1e-12
        sumsq_delta = None
        if pre.get("sumsq") is not None and dec.get("sumsq") is not None:
            sumsq_delta = abs(pre.get("sumsq") - dec.get("sumsq"))
        inv_rms_delta = None
        if pre.get("inv_rms") is not None and dec.get("inv_rms") is not None:
            inv_rms_delta = abs(pre.get("inv_rms") - dec.get("inv_rms"))

        input_ptr_match = pre.get("input_ptr") == dec.get("input_ptr")
        input_offset_match = pre.get("input_offset_bytes") == dec.get("input_offset_bytes")
        weight_hash_match = pre.get("weight_hash") == dec.get("weight_hash")

        bucket = bucketize(input_mae, weight_mae, output_mae,
                           input_ptr_match, input_offset_match,
                           eps_match, sumsq_delta)

        lines.append(
            f"{prompt}\t{exp}\t{input_mae}\t{weight_mae}\t{output_mae}\t"
            f"{str(eps_match).lower()}\t{sumsq_delta}\t{inv_rms_delta}\t"
            f"{str(input_ptr_match).lower()}\t{str(input_offset_match).lower()}\t"
            f"{str(weight_hash_match).lower()}\t{bucket}"
        )

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
