#!/usr/bin/env bash
set -euo pipefail

remote="${1:-}"
remote_dir="${2:-}"
preset="${3:-standard}"

if [[ -z "$remote" || -z "$remote_dir" ]]; then
  echo "usage: $0 user@host /path/to/gretacore [preset]" >&2
  exit 1
fi

ssh "$remote" "cd '$remote_dir' && tools/bench/platform/scripts/run_presets_local.sh '$preset'"
