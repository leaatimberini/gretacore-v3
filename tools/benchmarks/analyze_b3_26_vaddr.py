#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean


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


def summarize_vacc(rows):
    grouped = {}
    for i, r in enumerate(rows):
        key = r.get("prompt_id") or f"line_{i}"
        grouped.setdefault(key, []).append(r)

    summary = {}
    for key, items in grouped.items():
        v_layout = items[-1].get("v_layout_probe", {}).get("v_layout_best", "")
        pv_mae = mean([it.get("pv_mae", 0.0) for it in items])
        pv_max = max([it.get("pv_max_diff", 0.0) for it in items] + [0.0])
        attn_mae = mean([it.get("attn_out_mae", 0.0) for it in items])
        attn_max = max([it.get("attn_out_max_diff", 0.0) for it in items] + [0.0])
        pv_scope = items[-1].get("pv_scope", "")
        summary[key] = {
            "v_layout_best": v_layout,
            "pv_mae": pv_mae,
            "pv_max_diff": pv_max,
            "attn_out_mae": attn_mae,
            "attn_out_max_diff": attn_max,
            "pv_scope": pv_scope,
        }
    return summary


def summarize_vaddr(rows):
    grouped = {}
    for i, r in enumerate(rows):
        key = r.get("prompt_id") or f"line_{i}"
        grouped.setdefault(key, []).append(r)

    summary = {}
    for key, items in grouped.items():
        r = items[-1]
        mae_v_pos = r.get("mae_v_pos", 0.0)
        mae_v_prev = r.get("mae_v_prev", 0.0)
        mae_v_next = r.get("mae_v_next", 0.0)
        mae_v_col = r.get("mae_v_col", 0.0)
        candidates = {
            "pos": mae_v_pos,
            "prev": mae_v_prev,
            "next": mae_v_next,
            "col": mae_v_col,
        }
        best_key = min(candidates, key=candidates.get)
        summary[key] = {
            "mae_v_pos": mae_v_pos,
            "mae_v_prev": mae_v_prev,
            "mae_v_next": mae_v_next,
            "mae_v_col": mae_v_col,
            "best": best_key,
        }
    return summary


def extract_top1(path: Path):
    rows = load_rows(path)
    prefill = None
    decode = None
    for r in rows:
        phase = r.get("phase")
        if phase == "prefill_last":
            prefill = r.get("top1_id")
        elif phase == "decode0":
            decode = r.get("top1_id")
    return prefill, decode


def verdict(v_layout, pv_mae, vaddr_best):
    if vaddr_best in ("prev", "next"):
        return f"kv_pos_shift({vaddr_best})"
    if vaddr_best == "col" or (v_layout and v_layout != "row"):
        return "V layout mismatch"
    if pv_mae > 1e-3:
        return "P*V accumulation"
    return "low/mixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vacc-jsonl", required=True)
    ap.add_argument("--vaddr-jsonl", required=True)
    ap.add_argument("--p4-jsonl", required=True)
    ap.add_argument("--p5-jsonl", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    vacc_rows = load_rows(Path(args.vacc_jsonl))
    vacc_summary = summarize_vacc(vacc_rows)

    vaddr_rows = load_rows(Path(args.vaddr_jsonl))
    vaddr_summary = summarize_vaddr(vaddr_rows)

    p4_prefill, p4_decode = extract_top1(Path(args.p4_jsonl))
    p5_prefill, p5_decode = extract_top1(Path(args.p5_jsonl))

    lines = []
    lines.append(f"VACC JSONL: {args.vacc_jsonl}")
    lines.append(f"VADDR JSONL: {args.vaddr_jsonl}")
    lines.append(
        "prompt_id\tv_layout_best\tpv_scope\tpv_mae\tpv_max_diff\tattn_out_mae\tattn_out_max_diff\tmae_v_pos\tmae_v_prev\tmae_v_next\tmae_v_col\tvaddr_best\tprefill_last_top1\tdecode0_top1\tverdict"
    )

    keys = set(vacc_summary.keys()) | set(vaddr_summary.keys())
    for key in sorted(keys):
        vacc = vacc_summary.get(key, {})
        vaddr = vaddr_summary.get(key, {})
        prefill = p4_prefill if key == "p4_sys" else p5_prefill if key == "p5_ba" else None
        decode = p4_decode if key == "p4_sys" else p5_decode if key == "p5_ba" else None
        lines.append(
            f"{key}\t"
            f"{vacc.get('v_layout_best','')}\t"
            f"{vacc.get('pv_scope','')}\t"
            f"{vacc.get('pv_mae',0.0):.6g}\t"
            f"{vacc.get('pv_max_diff',0.0):.6g}\t"
            f"{vacc.get('attn_out_mae',0.0):.6g}\t"
            f"{vacc.get('attn_out_max_diff',0.0):.6g}\t"
            f"{vaddr.get('mae_v_pos',0.0):.6g}\t"
            f"{vaddr.get('mae_v_prev',0.0):.6g}\t"
            f"{vaddr.get('mae_v_next',0.0):.6g}\t"
            f"{vaddr.get('mae_v_col',0.0):.6g}\t"
            f"{vaddr.get('best','')}\t"
            f"{prefill}\t{decode}\t"
            f"{verdict(vacc.get('v_layout_best',''), vacc.get('pv_mae',0.0), vaddr.get('best',''))}"
        )

    output = "\n".join(lines)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n")


if __name__ == "__main__":
    main()
