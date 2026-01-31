#!/usr/bin/env python3
import csv
import datetime as dt
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "tools/bench/platform/results")
files = sorted(root.glob("*.txt"))
rows = []

num_kv = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=([-+0-9.]+(?:e[-+0-9]+)?)")

def parse_name(name: str):
    date = None
    env = None
    preset = None
    rest = name
    if len(name) > 11 and name[10] == "_":
        date = name[:10]
        rest = name[11:]
    parts = rest.split("_")
    if parts and parts[-1] == "amdcloud":
        env = "amdcloud"
        parts = parts[:-1]
    if parts and parts[-1] in ("smoke", "standard", "perf", "compute_only", "compute"):
        preset = parts[-1]
        parts = parts[:-1]
    bench = "_".join(parts) if parts else None
    return date, bench, preset, env

for path in files:
    text = path.read_text(errors="ignore")
    row = {
        "file": path.name,
        "bench": None,
        "date": None,
        "preset": None,
        "env": None,
    }
    date, bench, preset, env = parse_name(path.stem)
    if date:
        row["date"] = date
    if bench:
        row["bench"] = bench
    if preset:
        row["preset"] = preset
    if env:
        row["env"] = env

    for line in text.splitlines():
        if line.startswith("GRETA CORE Platform Bench:"):
            row["bench"] = line.split(":", 1)[1].strip()
        for key, val in num_kv.findall(line):
            try:
                row[key] = float(val) if "." in val or "e" in val or "E" in val else int(val)
            except ValueError:
                row[key] = val

    rows.append(row)

if not rows:
    print("no results found", file=sys.stderr)
    sys.exit(1)

keys = []
for r in rows:
    for k in r.keys():
        if k not in keys:
            keys.append(k)

out_date = dt.date.today().isoformat()
out = root / f"{out_date}_platform_summary.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(out)
