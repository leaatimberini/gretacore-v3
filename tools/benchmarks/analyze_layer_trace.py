#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path
from collections import defaultdict


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_layer_trace.py <jsonl>")
        return 1

    inp = Path(sys.argv[1])
    data = defaultdict(lambda: {"hashes": [], "min": [], "max": [], "mean": [], "nan": 0, "inf": 0})
    steps_seen = set()

    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if is_number(rec.get("step")):
                steps_seen.add(int(rec["step"]))

            if rec.get("type") == "step_header":
                continue

            layer = rec.get("layer")
            tag = rec.get("tag")
            if layer is None or tag is None:
                continue
            key = (int(layer), str(tag))
            d = data[key]
            if is_number(rec.get("hash")):
                d["hashes"].append(int(rec["hash"]))
            if is_number(rec.get("min")):
                d["min"].append(float(rec["min"]))
            if is_number(rec.get("max")):
                d["max"].append(float(rec["max"]))
            if is_number(rec.get("mean")):
                d["mean"].append(float(rec["mean"]))
            if is_number(rec.get("nan")):
                d["nan"] += int(rec["nan"])
            if is_number(rec.get("inf")):
                d["inf"] += int(rec["inf"])

    max_step = max(steps_seen) if steps_seen else -1
    print("steps_seen: {}".format(max_step + 1 if max_step >= 0 else 0))
    print("layer\ttag\tuniq_hash\tconst_from_step1\tmin(min)\tmax(max)\tmean(mean)\tnan\tinf")

    for (layer, tag) in sorted(data.keys()):
        d = data[(layer, tag)]
        hashes = d["hashes"]
        uniq_hash = len(set(hashes)) if hashes else 0
        const_from_step1 = False
        if len(hashes) > 1:
            const_from_step1 = len(set(hashes[1:])) == 1
        min_min = min(d["min"]) if d["min"] else math.nan
        max_max = max(d["max"]) if d["max"] else math.nan
        mean_mean = sum(d["mean"]) / len(d["mean"]) if d["mean"] else math.nan
        print("{}\t{}\t{}\t{}\t{:.6g}\t{:.6g}\t{:.6g}\t{}\t{}".format(
            layer,
            tag,
            uniq_hash,
            const_from_step1,
            min_min,
            max_max,
            mean_mean,
            d["nan"],
            d["inf"],
        ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
