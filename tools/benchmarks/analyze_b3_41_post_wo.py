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


STAGES = [
    "x_in",
    "attn_out",
    "wo_out",
    "x_after_attn",
    "ffn_norm",
    "mlp_out",
    "x_after_mlp",
    "x_out",
    "final_rms",
    "lm_head_in",
]

POST_WO_STAGES = [
    "ffn_norm",
    "mlp_out",
    "x_after_mlp",
    "x_out",
    "final_rms",
    "lm_head_in",
]


def parse_name(path: Path) -> Tuple[str, str]:
    name = path.name.replace("b3_41_post_wo_", "").replace(".jsonl", "")
    parts = name.split("_")
    if len(parts) < 2:
        return name, "E1"
    exp = parts[-1]
    prompt = "_".join(parts[:-1])
    return prompt, exp


def analyze_file(path: Path):
    pre: Dict[str, List[float]] = {}
    dec: Dict[str, List[float]] = {}
    logits_pre = None
    logits_dec = None
    for r in load_rows(path):
        if r.get("event") == "post_wo_trace":
            phase = r.get("phase")
            point = r.get("point")
            if phase not in ("prefill_last", "decode0"):
                continue
            if point:
                if phase == "prefill_last":
                    pre[point] = r.get("sample", [])
                else:
                    dec[point] = r.get("sample", [])
        elif r.get("event") == "post_wo_logits":
            if r.get("phase") == "prefill_last":
                logits_pre = r
            elif r.get("phase") == "decode0":
                logits_dec = r
    return pre, dec, logits_pre, logits_dec


def first_mismatch(stage_mae: Dict[str, Optional[float]], threshold: float) -> str:
    for stage in POST_WO_STAGES:
        val = stage_mae.get(stage)
        if val is not None and val > threshold:
            return stage
    return "NONE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    base = Path(args.dir)
    files = sorted(base.glob("b3_41_post_wo_*_E1.jsonl"))

    lines = []
    lines.append(f"B3.41 post-WO analysis: {base}")
    lines.append(
        "prompt\texp\tfirst_mismatch_stage\tffn_norm_mae\tmlp_out_mae\tx_after_mlp_mae\tx_out_mae\tfinal_rms_mae\tlm_head_in_mae\tlogits_mean_delta\tprefill_last_top1\tdecode0_top1\tcollapse_96965"
    )

    for path in files:
        prompt, exp = parse_name(path)
        pre, dec, lp, ld = analyze_file(path)
        if not pre or not dec:
            continue

        stage_mae: Dict[str, Optional[float]] = {}
        for stage in STAGES:
            stage_mae[stage] = mae(pre.get(stage, []), dec.get(stage, []))

        first = first_mismatch(stage_mae, 1e-6)

        pre_top1 = lp.get("top1_id", -1) if lp else -1
        dec_top1 = ld.get("top1_id", -1) if ld else -1
        collapse = (dec_top1 == 96965)

        logits_mean_delta = None
        if lp and ld:
            logits_mean_delta = abs(lp.get("logits_mean", 0.0) - ld.get("logits_mean", 0.0))

        lines.append(
            f"{prompt}\t{exp}\t{first}\t{stage_mae.get('ffn_norm')}\t{stage_mae.get('mlp_out')}\t{stage_mae.get('x_after_mlp')}\t{stage_mae.get('x_out')}\t{stage_mae.get('final_rms')}\t{stage_mae.get('lm_head_in')}\t{logits_mean_delta}\t{pre_top1}\t{dec_top1}\t{str(collapse).lower()}"
        )

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
