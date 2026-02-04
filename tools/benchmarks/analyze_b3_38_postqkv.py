#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ATTN_STAGES = ["q", "k", "v", "qk", "softmax", "pv", "attn_out"]
POST_STAGES = [
    "x_after_attn",
    "ffn_norm",
    "mlp_out",
    "x_after_mlp",
    "x_out",
    "final_norm",
    "lm_head_in",
]
STAGE_ORDER = ATTN_STAGES + POST_STAGES


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


def parse_prompt(path: Path) -> Tuple[str, str]:
    name = path.name
    # b3_38_postqkv_<prompt>_E1.jsonl
    core = name.replace("b3_38_postqkv_", "").replace(".jsonl", "")
    if core.endswith("_E1"):
        return core[:-3], "E1"
    return core, "E1"


def analyze_file(path: Path):
    pre: Dict[str, Dict] = {}
    dec: Dict[str, Dict] = {}
    logits_prefill = None
    logits_decode = None

    for r in load_rows(path):
        event = r.get("event")
        phase = r.get("phase")
        if phase not in ("prefill_last", "decode0"):
            continue

        if event == "attn_l0_pipe":
            if phase == "prefill_last":
                pre.update({k: r.get(k, []) for k in ATTN_STAGES})
            else:
                dec.update({k: r.get(k, []) for k in ATTN_STAGES})
        elif event == "stage_trace":
            point = r.get("point")
            if point in POST_STAGES or point == "attn_out":
                if phase == "prefill_last":
                    pre[point] = r.get("sample", [])
                else:
                    dec[point] = r.get("sample", [])
        elif event == "stage_logits":
            if phase == "prefill_last":
                logits_prefill = r
            elif phase == "decode0":
                logits_decode = r

    return pre, dec, logits_prefill, logits_decode


def first_mismatch(stage_mae: Dict[str, Optional[float]], threshold: float) -> str:
    for stage in STAGE_ORDER:
        m = stage_mae.get(stage)
        if m is not None and m > threshold:
            return stage
    return "NONE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=False)
    ap.add_argument("--threshold", type=float, default=1e-6)
    args = ap.parse_args()

    base = Path(args.dir)
    files = sorted(base.glob("b3_38_postqkv_*_E1.jsonl"))

    lines = []
    lines.append(f"B3.38 post-QKV analysis: {base}")
    lines.append(
        "prompt\texp\tfirst_mismatch_stage\tq_mae\tk_mae\tv_mae\tqk_mae\tsoftmax_mae\tpv_mae\tattn_out_mae\tresid_mae\tffn_norm_mae\tmlp_out_mae\tx_out_mae\tprefill_last_top1\tdecode0_top1\tcollapse_96965"
    )

    for path in files:
        prompt, exp = parse_prompt(path)
        pre, dec, lp, ld = analyze_file(path)
        if not pre or not dec:
            continue

        stage_mae: Dict[str, Optional[float]] = {}
        for stage in STAGE_ORDER:
            stage_mae[stage] = mae(pre.get(stage, []), dec.get(stage, []))

        first = first_mismatch(stage_mae, args.threshold)

        pre_top1 = lp.get("top1_id", -1) if lp else -1
        dec_top1 = ld.get("top1_id", -1) if ld else -1
        collapse = False
        if dec_top1 == 96965:
            collapse = True

        lines.append(
            f"{prompt}\t{exp}\t{first}\t"
            f"{stage_mae.get('q')}\t{stage_mae.get('k')}\t{stage_mae.get('v')}\t"
            f"{stage_mae.get('qk')}\t{stage_mae.get('softmax')}\t{stage_mae.get('pv')}\t"
            f"{stage_mae.get('attn_out')}\t{stage_mae.get('x_after_attn')}\t{stage_mae.get('ffn_norm')}\t"
            f"{stage_mae.get('mlp_out')}\t{stage_mae.get('x_out')}\t{pre_top1}\t{dec_top1}\t{str(collapse).lower()}"
        )

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
