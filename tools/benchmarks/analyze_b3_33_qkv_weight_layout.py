#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


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


def analyze_file(path: Path) -> Dict[str, Dict[str, float]]:
    rows = load_rows(path)
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        if r.get("event") != "attn_l0_pipe":
            continue
        phase = r.get("phase")
        if phase not in ("prefill_last", "decode0"):
            continue
        out[phase] = {
            "layout_best": r.get("q_weight_layout_best", "unknown"),
            "mae_row": r.get("q_weight_mae_row", None),
            "mae_col": r.get("q_weight_mae_col", None),
        }
    return out


def mismatch_label(prefill: Optional[Dict], decode: Optional[Dict]) -> str:
    if not prefill or not decode:
        return "missing_phase"
    layout_pre = prefill.get("layout_best", "unknown")
    layout_dec = decode.get("layout_best", "unknown")
    if layout_pre != layout_dec:
        return "layout_diff"
    if layout_dec == "col":
        return "layout_col"
    try:
        mae_dec = float(decode.get("mae_row"))
    except (TypeError, ValueError):
        mae_dec = None
    if mae_dec is not None and mae_dec > 1e-3:
        return "decode_weight_mismatch"
    return "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    base = Path(args.dir)
    files = sorted(base.glob("b3_33_attn_l0_pipe_*.jsonl"))
    expected = {"p0_short", "p4_sys", "p5_ba", "p6_long"}
    found = set()
    for f in files:
        name = f.name
        if "b3_33_attn_l0_pipe_" in name:
            prompt = name.split("b3_33_attn_l0_pipe_")[1].split(".jsonl")[0]
            found.add(prompt)
    missing = sorted(expected - found)

    lines = []
    lines.append(f"B3.33 QKV weight layout verify: {base}")
    lines.append("prompt\tphase\tlayout_best\tmae_row\tmae_col\tfirst_mismatch")

    for path in files:
        prompt = path.name.split("b3_33_attn_l0_pipe_")[1].split(".jsonl")[0]
        phases = analyze_file(path)
        pre = phases.get("prefill_last")
        dec = phases.get("decode0")
        mismatch = mismatch_label(pre, dec)

        for phase in ("prefill_last", "decode0"):
            row = phases.get(phase)
            if not row:
                continue
            lines.append(
                f"{prompt}\t{phase}\t{row.get('layout_best')}\t{row.get('mae_row')}\t{row.get('mae_col')}\t{mismatch}"
            )

    if missing:
        lines.append(f"MISSING_PROMPTS\t{', '.join(missing)}")

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
