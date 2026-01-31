#!/usr/bin/env bash
set -euo pipefail

remote="${1:?user@host required}"
remote_dir="${2:-/tmp/greta}"
preset="${3:-standard}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -az --delete "$root/" "$remote:$remote_dir/"
ssh "$remote" "cd '$remote_dir' && tools/bench/runtime/scripts/run_presets_local.sh '$preset'"
rsync -az "$remote:$remote_dir/tools/bench/runtime/results/" "$root/results/"

python3 "$root/scripts/gen_bench_csv.py" "$root/results"
