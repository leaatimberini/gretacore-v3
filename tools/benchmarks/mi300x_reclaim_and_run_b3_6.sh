#!/usr/bin/env bash
set -euo pipefail
IP="${1:-129.212.184.200}"

# 1) Snapshot rápido del estado antes de tocar nada
ssh root@"$IP" 'set -e;
echo "=== HOST ==="; hostname; uptime;
echo "=== DATE ==="; date -u;
echo "=== ROCM-SMI MEM ===";
command -v rocm-smi >/dev/null && rocm-smi --showmemuse --showuse --showpids || echo "rocm-smi not found";
echo "=== TOP GPU PROCS ===";
(ps -eo pid,ppid,user,etimes,cmd --sort=-etimes | head -n 40) || true;
'

# 2) Kill selectivo de procesos típicos que comen VRAM (solo si existen)
#   - vLLM/open-webui, servidores python, restos de greta_infer
ssh root@"$IP" 'set -e;
echo "=== KILL CANDIDATES ===";
kill_match() {
  local pattern="$1"
  local self="$$"
  local pids
  pids=$(pgrep -f "$pattern" || true)
  for pid in $pids; do
    if [ "$pid" != "$self" ]; then
      kill -9 "$pid" || true
    fi
  done
}
kill_match "vllm"
kill_match "open-webui"
kill_match "python.*vllm"
kill_match "greta_infer"
kill_match "greta_server"
sleep 2
echo "=== AFTER KILL ROCM-SMI ===";
command -v rocm-smi >/dev/null && rocm-smi --showmemuse --showuse --showpids || true;
'

# 3) Verificar si VRAM liberó lo suficiente (si sigue saturada, lo vamos a registrar igual)
ssh root@"$IP" 'set -e;
echo "=== CHECK FREE MEM ===";
command -v rocm-smi >/dev/null && rocm-smi --showmemuse --showuse --showpids || true;
'

# 4) Re-pull limpio del repo (remoto efímero; garantizar estado idéntico a GitHub)
ssh root@"$IP" 'set -e;
cd /root/gretacore
git fetch origin
git reset --hard origin/main
git clean -fd
git rev-parse HEAD
'

# 5) Build
ssh root@"$IP" 'set -e;
cd /root/gretacore/tools/inference/build
make -B -j$(nproc)
'

# 6) Run B3.6 (INT4 + seq_len recortado)
ssh root@"$IP" 'set -e;
OUTDIR=/root/gretacore/artifacts/alignment/2026-02-03
mkdir -p "$OUTDIR"
export GRETA_INT4_WEIGHTS=1
export GRETA_MAX_SEQ_LEN=256
export GRETA_TRACE_READOUT=1
export GRETA_TRACE_READOUT_OUT=$OUTDIR/b3_6_readout.jsonl
export GRETA_TRACE_PREFILL_DECODE=1
export GRETA_TRACE_PREFILL_DECODE_OUT=$OUTDIR/b3_6_prefill_decode.jsonl
export GRETA_TRACE_LANDSCAPE=1
export GRETA_TRACE_LANDSCAPE_OUT=$OUTDIR/b3_6_landscape.jsonl
MODEL=/root/gretacore/models/llama3_8b_q4/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
cd /root/gretacore/tools/inference/build
( ./greta_infer --model "$MODEL" --prompt "Hi" --max-tokens 16 --greedy --debug-decode 16 ) \
  2>&1 | tee $OUTDIR/b3_6_run.log
ls -lh $OUTDIR/b3_6_* || true
'

# 7) Empaquetar artefactos pase lo que pase (para bajar al local)
ssh root@"$IP" 'set -e;
cd /root/gretacore
tar -czf /root/gretacore_b3_6_artifacts.tgz \
  artifacts/alignment/2026-02-03/b3_6_run.log \
  artifacts/alignment/2026-02-03/b3_6_readout.jsonl \
  artifacts/alignment/2026-02-03/b3_6_prefill_decode.jsonl \
  artifacts/alignment/2026-02-03/b3_6_landscape.jsonl \
  docs/AMD/2026_02_03_B3_6_decode_readout_landscape_rerun.md \
  2>/dev/null || true
ls -lh /root/gretacore_b3_6_artifacts.tgz || true
'
