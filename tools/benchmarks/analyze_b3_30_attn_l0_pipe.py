#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

STAGES = ["q", "k", "v", "qk", "softmax", "pv", "attn_out"]


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


def format_meta(r):
    keys = [
        "seq_len_used",
        "pos_id",
        "kv_pos",
        "token_index",
        "k_layout_used",
        "v_layout_used",
        "kv_layer_stride_bytes",
        "kv_head_stride_bytes",
        "kv_pos_stride_bytes",
    ]
    return {k: r.get(k) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    rows = load_rows(Path(args.jsonl))
    data: Dict[Tuple[str, str], dict] = {}

    for r in rows:
        if r.get("event") != "attn_l0_pipe":
            continue
        prompt = r.get("prompt_id", "unknown")
        phase = r.get("phase", "")
        key = (prompt, phase)
        data[key] = r

    prompts = sorted({k[0] for k in data.keys()})
    lines = []
    lines.append(f"ATTN_L0_PIPE JSONL: {args.jsonl}")

    for prompt in prompts:
        pre = data.get((prompt, "prefill_last"))
        dec = data.get((prompt, "decode0"))
        if not pre or not dec:
            lines.append(f"{prompt}\tmissing_phase\tprefill={pre is not None}\tdecode0={dec is not None}")
            continue

        lines.append(f"\nPROMPT {prompt}")
        lines.append("stage\tmae\tmax_diff")
        first_mismatch = None
        for stage in STAGES:
            mae, maxd = mae_max(pre.get(stage, []), dec.get(stage, []))
            if mae is None:
                lines.append(f"{stage}\tNA\tNA")
                continue
            lines.append(f"{stage}\t{mae:.6g}\t{maxd:.6g}")
            if first_mismatch is None and mae > 1e-7:
                first_mismatch = (stage, mae, maxd)

        if first_mismatch:
            stage, mae, maxd = first_mismatch
            lines.append(
                f"FIRST_MISMATCH\tstage={stage}\tmae={mae:.6g}\tmax_diff={maxd:.6g}"
            )
        else:
            lines.append("FIRST_MISMATCH\tNONE")

        meta_pre = format_meta(pre)
        meta_dec = format_meta(dec)
        lines.append(f"meta_prefill\t{meta_pre}")
        lines.append(f"meta_decode0\t{meta_dec}")

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
