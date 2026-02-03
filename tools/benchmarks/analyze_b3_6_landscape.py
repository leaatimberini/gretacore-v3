#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def extract_top(entry: Dict[str, Any]) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[float]]:
    top1 = entry.get("top1")
    top2 = entry.get("top2")
    top1_id = None
    top1_logit = None
    top2_id = None
    top2_logit = None
    if isinstance(top1, dict):
        top1_id = top1.get("id")
        top1_logit = top1.get("logit")
    if isinstance(top2, dict):
        top2_id = top2.get("id")
        top2_logit = top2.get("logit")
    if top1_id is None:
        top1_id = entry.get("top1_id")
    if top1_logit is None:
        top1_logit = entry.get("top1_logit")
    if top2_id is None:
        top2_id = entry.get("top2_id")
    if top2_logit is None:
        top2_logit = entry.get("top2_logit")
    return top1_id, top1_logit, top2_id, top2_logit


def entropy_from_topk(topk: List[Dict[str, Any]]) -> Optional[float]:
    logits: List[float] = []
    for item in topk:
        logit = item.get("logit")
        if logit is None:
            continue
        logits.append(float(logit))
    if not logits:
        return None
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    denom = sum(exps)
    if denom == 0:
        return None
    probs = [e / denom for e in exps]
    ent = 0.0
    for p in probs:
        if p > 0:
            ent -= p * math.log(p)
    return ent


def mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def analyze(readout: List[Dict[str, Any]], prefill: List[Dict[str, Any]], landscape: List[Dict[str, Any]]):
    prefill_steps = {e.get("step") for e in prefill if e.get("phase") == "prefill"}
    decode_steps = {e.get("step") for e in prefill if e.get("phase") == "decode"}
    prefill_steps = {s for s in prefill_steps if isinstance(s, int)}
    decode_steps = {s for s in decode_steps if isinstance(s, int)}

    def phase_for_step(step: Optional[int]) -> str:
        if step in prefill_steps:
            return "prefill"
        if step in decode_steps:
            return "decode"
        return "unknown"

    rows = []
    gap_values = []
    entropy_values = []
    top1_prefill = []
    top1_decode = []

    for entry in landscape:
        step = entry.get("step")
        phase = phase_for_step(step)
        top1_id, top1_logit, top2_id, top2_logit = extract_top(entry)
        gap = entry.get("gap")
        if gap is None and top1_logit is not None and top2_logit is not None:
            gap = float(top1_logit) - float(top2_logit)
        ent = entry.get("entropy_topk")
        if ent is None:
            top5 = entry.get("top5")
            if isinstance(top5, list):
                ent = entropy_from_topk(top5)
        if isinstance(gap, (int, float)):
            if phase == "decode" or phase == "unknown":
                gap_values.append(float(gap))
        if isinstance(ent, (int, float)):
            if phase == "decode" or phase == "unknown":
                entropy_values.append(float(ent))

        if phase == "prefill" and top1_id is not None:
            top1_prefill.append(top1_id)
        if (phase == "decode" or phase == "unknown") and top1_id is not None:
            top1_decode.append(top1_id)

        rows.append(
            {
                "step": step,
                "phase": phase,
                "top1_id": top1_id,
                "top1_logit": top1_logit,
                "top2_id": top2_id,
                "top2_logit": top2_logit,
                "gap": gap,
                "entropy_topk": ent,
            }
        )

    uniq_top1_prefill = len(set(top1_prefill)) if top1_prefill else 0
    uniq_top1_decode = len(set(top1_decode)) if top1_decode else 0

    gap_mean, gap_std = mean_std(gap_values)
    ent_mean, ent_std = mean_std(entropy_values)

    prefill_last_top1 = None
    decode0_top1 = None
    if prefill_steps and rows:
        prefill_rows = [r for r in rows if r["phase"] == "prefill"]
        prefill_rows = [r for r in prefill_rows if isinstance(r.get("step"), int)]
        prefill_rows.sort(key=lambda r: r["step"])
        if prefill_rows:
            prefill_last_top1 = prefill_rows[-1]["top1_id"]
    decode_rows = [r for r in rows if r["phase"] == "decode"]
    decode_rows = [r for r in decode_rows if isinstance(r.get("step"), int)]
    decode_rows.sort(key=lambda r: r["step"])
    if decode_rows:
        decode0_top1 = decode_rows[0]["top1_id"]
    elif rows:
        rows_sorted = [r for r in rows if isinstance(r.get("step"), int)]
        rows_sorted.sort(key=lambda r: r["step"])
        if rows_sorted:
            decode0_top1 = rows_sorted[0]["top1_id"]

    readout_steps = [e.get("step") for e in readout if isinstance(e.get("step"), int)]
    landscape_steps = [e.get("step") for e in landscape if isinstance(e.get("step"), int)]
    readout_step_set = set(readout_steps)
    landscape_step_set = set(landscape_steps)

    readout_inconsistent = False
    if not readout_steps or not landscape_steps:
        readout_inconsistent = True
    else:
        if not readout_step_set.issubset(landscape_step_set):
            readout_inconsistent = True

    hidden_ptrs = [e.get("hidden_ptr") for e in readout if e.get("hidden_ptr") is not None]
    logits_ptrs = [e.get("logits_ptr") for e in readout if e.get("logits_ptr") is not None]
    hidden_hashes = [e.get("hidden_hash") for e in readout if e.get("hidden_hash") is not None]
    logits_hashes = [e.get("logits_hash") for e in readout if e.get("logits_hash") is not None]
    token_indices = [e.get("token_index") for e in readout if isinstance(e.get("token_index"), int)]

    readout_stats = {
        "hidden_ptr_unique": len(set(hidden_ptrs)) if hidden_ptrs else 0,
        "logits_ptr_unique": len(set(logits_ptrs)) if logits_ptrs else 0,
        "hidden_hash_unique": len(set(hidden_hashes)) if hidden_hashes else 0,
        "logits_hash_unique": len(set(logits_hashes)) if logits_hashes else 0,
        "token_index_monotonic": token_indices == sorted(token_indices)
        if token_indices
        else False,
    }

    conclusion = "B"
    if readout_inconsistent:
        conclusion = "D"
    else:
        if uniq_top1_decode == 1:
            if uniq_top1_prefill > 1:
                conclusion = "C"
            else:
                if gap_mean is not None and ent_mean is not None:
                    if gap_mean >= 2.0 and ent_mean <= 1.5:
                        conclusion = "A"
                    else:
                        conclusion = "B"
                else:
                    conclusion = "B"
        else:
            if uniq_top1_prefill > 1 and uniq_top1_decode < uniq_top1_prefill:
                conclusion = "C"
            else:
                conclusion = "B"

    return {
        "rows": rows,
        "uniq_top1_prefill": uniq_top1_prefill,
        "uniq_top1_decode": uniq_top1_decode,
        "gap_mean": gap_mean,
        "gap_std": gap_std,
        "entropy_mean": ent_mean,
        "entropy_std": ent_std,
        "prefill_last_top1": prefill_last_top1,
        "decode0_top1": decode0_top1,
        "readout_inconsistent": readout_inconsistent,
        "readout_stats": readout_stats,
        "conclusion": conclusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readout", required=True)
    parser.add_argument("--prefill", required=True)
    parser.add_argument("--landscape", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    readout = load_jsonl(args.readout)
    prefill = load_jsonl(args.prefill)
    landscape = load_jsonl(args.landscape)

    result = analyze(readout, prefill, landscape)

    tables_path = os.path.join(args.out, "tables.csv")
    with open(tables_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "phase",
                "top1_id",
                "top1_logit",
                "top2_id",
                "top2_logit",
                "gap",
                "entropy_topk",
            ],
        )
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow(row)

    summary_path = os.path.join(args.out, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# B3.7 Analysis Summary\n\n")
        f.write(f"uniq_top1_prefill: {result['uniq_top1_prefill']}\n")
        f.write(f"uniq_top1_decode: {result['uniq_top1_decode']}\n")
        f.write(f"gap_mean: {result['gap_mean']}\n")
        f.write(f"gap_std: {result['gap_std']}\n")
        f.write(f"entropy_mean: {result['entropy_mean']}\n")
        f.write(f"entropy_std: {result['entropy_std']}\n")
        f.write(f"prefill_last_top1: {result['prefill_last_top1']}\n")
        f.write(f"decode0_top1: {result['decode0_top1']}\n")
        f.write(f"readout_inconsistent: {result['readout_inconsistent']}\n")
        f.write(f"readout_stats: {result['readout_stats']}\n")
        f.write(f"conclusion: {result['conclusion']}\n")

    plots_path = os.path.join(args.out, "plots_disabled.txt")
    with open(plots_path, "w", encoding="utf-8") as f:
        f.write("Plots are disabled for this run. Use tables.csv for inspection.\n")

    print(f"uniq_top1_prefill={result['uniq_top1_prefill']}")
    print(f"uniq_top1_decode={result['uniq_top1_decode']}")
    print(f"gap_mean={result['gap_mean']}")
    print(f"gap_std={result['gap_std']}")
    print(f"prefill_last_top1={result['prefill_last_top1']}")
    print(f"decode0_top1={result['decode0_top1']}")
    print(f"conclusion={result['conclusion']}")


if __name__ == "__main__":
    main()
