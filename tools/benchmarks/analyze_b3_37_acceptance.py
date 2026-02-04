#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_rows(path: Path) -> List[Dict]:
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


def analyze_file(path: Path):
    prefill = None
    decode_steps: Dict[int, Dict] = {}
    for r in load_rows(path):
        phase = r.get("phase")
        step = int(r.get("step", 0))
        if phase == "prefill_last":
            prefill = r
        elif phase == "decode":
            decode_steps[step] = r
    return prefill, decode_steps


def top1_ids(dec: Dict[int, Dict], max_step: int) -> List[int]:
    ids = []
    for s in range(1, max_step + 1):
        r = dec.get(s)
        if r is None:
            continue
        ids.append(int(r.get("top1_id", -1)))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    base = Path(args.dir)
    files = sorted(base.glob("b3_37_E*_prefill_decode_*.jsonl"))

    lines = []
    lines.append(f"B3.37 acceptance analysis: {base}")
    lines.append(
        "prompt\texp\tprefill_last_top1\tdecode0_top1\tmatch\tuniq_top1_0_15\ttop1_s0\ttop1_s1\ttop1_s2\ttop1_s3\tgap_prefill\tgap_decode\tcollapse_to_96965"
    )

    for path in files:
        name = path.name
        try:
            exp = name.split("_prefill_decode_")[0].split("b3_37_")[1]
            prompt = name.split("_prefill_decode_")[1].split(".jsonl")[0]
        except Exception:
            continue

        prefill, dec = analyze_file(path)
        if not prefill or not dec:
            continue

        pre_top1 = int(prefill.get("top1_id", -1))
        pre_gap = float(prefill.get("gap", 0.0))

        dec0 = dec.get(1)
        dec0_top1 = int(dec0.get("top1_id", -1)) if dec0 else -1
        dec0_gap = float(dec0.get("gap", 0.0)) if dec0 else 0.0

        ids = top1_ids(dec, 16)
        uniq = len(set(ids)) if ids else 0

        top1_s0 = dec.get(1, {}).get("top1_id", -1)
        top1_s1 = dec.get(2, {}).get("top1_id", -1)
        top1_s2 = dec.get(3, {}).get("top1_id", -1)
        top1_s3 = dec.get(4, {}).get("top1_id", -1)

        collapse = False
        if dec0_top1 == 96965:
            collapse = True
        if uniq == 1 and ids and ids[0] == 96965:
            collapse = True

        match = pre_top1 == dec0_top1

        lines.append(
            f"{prompt}\t{exp}\t{pre_top1}\t{dec0_top1}\t{str(match).lower()}\t{uniq}\t{top1_s0}\t{top1_s1}\t{top1_s2}\t{top1_s3}\t{pre_gap}\t{dec0_gap}\t{str(collapse).lower()}"
        )

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
