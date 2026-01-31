#!/usr/bin/env python3
import csv
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "tools/bench/runtime/results")
files = sorted(root.glob("*.txt"))
rows = []

for path in files:
    text = path.read_text(errors="ignore")
    bench = None
    m = n = k = iters = batch = compute_only = None
    km = kp50 = kp99 = tf = tf50 = tf99 = None
    status = None

    for line in text.splitlines():
        if line.startswith("GRETA CORE Runtime Bench:"):
            bench = line.split(":", 1)[1].strip()
        if line.startswith("M="):
            for key in ["M", "N", "K", "iters", "batch", "compute_only"]:
                m2 = re.search(rf"\b{key}=(\d+)", line)
                if m2:
                    val = int(m2.group(1))
                    if key == "M":
                        m = val
                    elif key == "N":
                        n = val
                    elif key == "K":
                        k = val
                    elif key == "iters":
                        iters = val
                    elif key == "batch":
                        batch = val
                    elif key == "compute_only":
                        compute_only = val
        if "kernel_mean_ms=" in line:
            m1 = re.search(r"kernel_mean_ms=([0-9.]+)", line)
            m2 = re.search(r"kernel_p50_ms=([0-9.]+)", line)
            m3 = re.search(r"kernel_p99_ms=([0-9.]+)", line)
            if m1:
                km = float(m1.group(1))
            if m2:
                kp50 = float(m2.group(1))
            if m3:
                kp99 = float(m3.group(1))
        if "mean_TFLOPs=" in line:
            m1 = re.search(r"mean_TFLOPs=([0-9.]+)", line)
            m2 = re.search(r"p50_TFLOPs=([0-9.]+)", line)
            m3 = re.search(r"p99_TFLOPs=([0-9.]+)", line)
            if m1:
                tf = float(m1.group(1))
            if m2:
                tf50 = float(m2.group(1))
            if m3:
                tf99 = float(m3.group(1))
        if line.startswith("STATUS="):
            status = line.split("=", 1)[1].strip()

    rows.append(
        {
            "file": path.name,
            "bench": bench,
            "M": m,
            "N": n,
            "K": k,
            "iters": iters,
            "batch": batch,
            "compute_only": compute_only,
            "kernel_mean_ms": km,
            "kernel_p50_ms": kp50,
            "kernel_p99_ms": kp99,
            "mean_TFLOPs": tf,
            "p50_TFLOPs": tf50,
            "p99_TFLOPs": tf99,
            "status": status,
        }
    )

out = root / "2026-01-31_bench_summary.csv"
if rows:
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(out)
else:
    print("no results found", file=sys.stderr)
    sys.exit(1)
