#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

POINT_ORDER = [
    "x_in",
    "attn_out",
    "x_after_attn",
    "mlp_out",
    "x_out",
    "x_after_mlp",
    "final_rms",
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


def sample_mae(a: List[float], b: List[float]):
    if not a or not b:
        return None
    n = min(len(a), len(b))
    if n == 0:
        return None
    return sum(abs(a[i] - b[i]) for i in range(n)) / n


def normalize_point(point: str) -> str:
    if point == "final_norm":
        return "final_rms"
    return point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="Path to stage trace JSONL")
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    rows = load_rows(Path(args.jsonl))
    data: Dict[Tuple[str, str, int, str], dict] = {}

    for r in rows:
        prompt = r.get("prompt_id", "unknown")
        phase = r.get("phase", "")
        point = normalize_point(r.get("point", ""))
        layer = int(r.get("layer", -999))
        key = (prompt, phase, layer, point)
        data[key] = r

    prompts = sorted({k[0] for k in data.keys()})

    lines = []
    lines.append(f"STAGE JSONL: {args.jsonl}")

    for prompt in prompts:
        lines.append("")
        lines.append(f"PROMPT={prompt}")
        lines.append("layer\tpoint\tmae\thash_match\tpre_top1\tdec_top1\tgap_pre\tgap_dec")
        first_mismatch = None
        first_detail = None

        for layer in LAYER_ORDER:
            for point in POINT_ORDER:
                key_prefill = (prompt, "prefill_last", layer, point)
                key_decode = (prompt, "decode0", layer, point)
                if key_prefill not in data or key_decode not in data:
                    continue
                pre = data[key_prefill]
                dec = data[key_decode]
                pre_hash = pre.get("hash")
                dec_hash = dec.get("hash")
                hash_match = (pre_hash == dec_hash)
                mae = None
                pre_top1 = pre.get("top1_id")
                dec_top1 = dec.get("top1_id")
                pre_gap = pre.get("gap")
                dec_gap = dec.get("gap")

                if point == "logits":
                    mae = None
                else:
                    pre_sample = pre.get("sample") or []
                    dec_sample = dec.get("sample") or []
                    mae = sample_mae(pre_sample, dec_sample)

                lines.append(
                    f"{layer}\t{point}\t{'' if mae is None else f'{mae:.6g}'}\t{hash_match}\t"
                    f"{'' if pre_top1 is None else pre_top1}\t{'' if dec_top1 is None else dec_top1}\t"
                    f"{'' if pre_gap is None else f'{pre_gap:.6g}'}\t{'' if dec_gap is None else f'{dec_gap:.6g}'}"
                )

                mismatch = False
                if point == "logits":
                    if pre_top1 is not None and dec_top1 is not None and pre_top1 != dec_top1:
                        mismatch = True
                else:
                    if mae is not None and mae > 0:
                        mismatch = True
                    elif pre_hash is not None and dec_hash is not None and not hash_match:
                        mismatch = True

                if mismatch and first_mismatch is None:
                    first_mismatch = (layer, point)
                    if point == "logits":
                        first_detail = f"top1 {pre_top1}->{dec_top1}"
                    else:
                        first_detail = f"mae {mae:.6g}" if mae is not None else "hash_mismatch"

        if first_mismatch:
            layer, point = first_mismatch
            lines.append(
                f"FIRST_MISMATCH\tlayer={layer}\tpoint={point}\t{first_detail}"
            )
        else:
            lines.append("FIRST_MISMATCH\tNONE")

    output = "\n".join(lines) + "\n"
    print(output)
    if args.out:
        Path(args.out).write_text(output)


if __name__ == "__main__":
    main()
