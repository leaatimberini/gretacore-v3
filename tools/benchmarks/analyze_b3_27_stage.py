#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

POINT_ORDER = [
    "x_in",
    "attn_out",
    "x_after_attn",
    "mlp_out",
    "x_after_mlp",
    "final_norm",
    "lm_head_in",
    "logits",
]

LAYER_ORDER = [0, 1, 2, 15, 31, 32, -1]


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


def sample_mae(a, b):
    if not a or not b:
        return None
    n = min(len(a), len(b))
    if n == 0:
        return None
    return sum(abs(a[i] - b[i]) for i in range(n)) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    rows = load_rows(Path(args.jsonl))
    data: Dict[Tuple[str, str, int, str], dict] = {}

    for r in rows:
        prompt = r.get("prompt_id", "")
        phase = r.get("phase", "")
        point = r.get("point", "")
        layer = r.get("layer", -999)
        if prompt == "":
            prompt = "unknown"
        key = (prompt, phase, int(layer), point)
        data[key] = r

    prompts = sorted({k[0] for k in data.keys()})

    lines = []
    lines.append(f"STAGE JSONL: {args.jsonl}")

    for prompt in prompts:
        first_mismatch = None
        details = None
        for layer in LAYER_ORDER:
            for point in POINT_ORDER:
                key_prefill = (prompt, "prefill_last", layer, point)
                key_decode = (prompt, "decode0", layer, point)
                if key_prefill not in data or key_decode not in data:
                    continue
                pre = data[key_prefill]
                dec = data[key_decode]
                mismatch = False
                metric = ""
                if point == "logits":
                    pre_top1 = pre.get("top1_id")
                    dec_top1 = dec.get("top1_id")
                    pre_gap = pre.get("gap")
                    dec_gap = dec.get("gap")
                    if pre_top1 != dec_top1:
                        mismatch = True
                        metric = f"top1 {pre_top1}->{dec_top1}"
                    elif pre_gap is not None and dec_gap is not None and abs(pre_gap - dec_gap) > 1e-3:
                        mismatch = True
                        metric = f"gap {pre_gap:.4g}->{dec_gap:.4g}"
                else:
                    pre_sample = pre.get("sample") or []
                    dec_sample = dec.get("sample") or []
                    mae = sample_mae(pre_sample, dec_sample)
                    if mae is not None and mae > 1e-6:
                        mismatch = True
                        metric = f"sample_mae {mae:.6g}"
                    else:
                        if pre.get("hash") != dec.get("hash"):
                            mismatch = True
                            metric = "hash_mismatch"
                if mismatch:
                    first_mismatch = (layer, point)
                    details = metric
                    break
            if first_mismatch:
                break

        if first_mismatch:
            layer, point = first_mismatch
            lines.append(
                f"{prompt}\tfirst_mismatch\tlayer={layer}\tpoint={point}\t{details}"
            )
        else:
            lines.append(f"{prompt}\tfirst_mismatch\tNONE")

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
