#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

STAGES = ["q", "k", "v"]


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


def mae_max(a: List[float], b: List[float]):
    if not a or not b:
        return None, None
    n = min(len(a), len(b))
    if n == 0:
        return None, None
    total = 0.0
    maxd = 0.0
    for i in range(n):
        d = abs(a[i] - b[i])
        total += d
        if d > maxd:
            maxd = d
    return total / n, maxd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    base = Path(args.dir)
    files = sorted(base.glob("b3_31_E*_attn_l0_pipe_*.jsonl"))
    if not files:
        print(f"No JSONL files found under {base}")
        return

    lines = []
    lines.append(f"B3.31 QKV route matrix: {base}")
    lines.append("exp\tprompt\tq_mae\tk_mae\tv_mae\tq_route\tk_route\tv_route\tqkv_force_route\tqkv_force_gemm")

    best = None

    for path in files:
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
        if not pre or not dec:
            continue

        exp = "unknown"
        prompt = pre.get("prompt_id", "unknown")
        name = path.name
        if "_E" in name:
            exp = name.split("_attn_l0_pipe")[0].split("b3_31_")[-1]

        q_mae, _ = mae_max(pre.get("q", []), dec.get("q", []))
        k_mae, _ = mae_max(pre.get("k", []), dec.get("k", []))
        v_mae, _ = mae_max(pre.get("v", []), dec.get("v", []))

        q_route = dec.get("q_route_used", "")
        k_route = dec.get("k_route_used", "")
        v_route = dec.get("v_route_used", "")
        qkv_force_route = dec.get("qkv_force_route", "")
        qkv_force_gemm = dec.get("qkv_force_gemm", "")

        if q_mae is not None:
            lines.append(
                f"{exp}\t{prompt}\t{q_mae:.6g}\t{k_mae:.6g}\t{v_mae:.6g}\t{q_route}\t{k_route}\t{v_route}\t{qkv_force_route}\t{qkv_force_gemm}"
            )
            if best is None or q_mae < best[0]:
                best = (q_mae, exp, prompt, q_route)

    if best:
        lines.append(
            f"BEST_Q_MAE\t{best[0]:.6g}\t{best[1]}\t{best[2]}\t{best[3]}"
        )

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
