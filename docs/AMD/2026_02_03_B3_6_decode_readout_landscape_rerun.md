# B3.6 Rerun: Decode Readout + Landscape (D2H Illegal Access Fix)

Fecha: 2026-02-03
Bloque: B3.6 (continuidad B3.7 si aplica)

## Resumen ejecutivo (ES)
Se reforzó el camino D2H de tracing para aislar y prevenir el crash `hipMemcpy D2H failed: illegal memory access` durante B3.6. Se agregaron guard rails de punteros/bounds, validación de forma (config) y atribución de kernels (hipGetLastError + hipDeviceSynchronize) bajo los flags de tracing, con logging detallado para identificar offsets/ptrs erróneos. El objetivo inmediato es re-ejecutar B3.6 en MI300X y generar los JSONL de readout/prefill-decode/landscape sin crash.

## Executive summary (EN)
The D2H tracing path was hardened to isolate and prevent the `hipMemcpy D2H failed: illegal memory access` crash in B3.6. We added pointer/bounds guard rails, shape validation (model config), and kernel attribution (hipGetLastError + hipDeviceSynchronize) under tracing flags, with detailed logging to pinpoint wrong offsets/ptrs. Immediate goal: rerun B3.6 on MI300X and produce the readout/prefill-decode/landscape JSONL without crashing.

## Cambios realizados (código)
- Guard rails en D2H:
  - `ptr != nullptr` y `offset_bytes + size_bytes <= alloc_bytes` (con manejo seguro de overflow).
  - Check de `host_ptr` en copias D2H.
- Logging D2H detallado (solo con `GRETA_TRACE_*`):
  - `tensor_name`, `step`, `layer`, `src_ptr`, `alloc_bytes`, `offset_bytes`, `size_bytes`.
- Validación de forma (solo con `GRETA_TRACE_*`):
  - `vocab_size`, `dim`, `num_layers`, `num_heads`, `num_heads_kv`, `head_dim`.
  - Consistencia: `dim % num_heads == 0`, `head_dim == dim / num_heads`, `num_heads_kv <= num_heads`, `num_heads % num_heads_kv == 0`.
- Atribución de kernels (solo con `GRETA_TRACE_*`):
  - `hipGetLastError()` + `hipDeviceSynchronize()` tras kernels críticos/GretaCompute.
  - Se instrumentaron `Fused RoPE+KV Update` y `Flash Attention Prefill`.
- Build unblock:
  - Se removió `src/rt/backend/hip/src/arena.cpp` de `tools/inference/CMakeLists.txt` porque el archivo no existe en este repo y no hay referencias activas al mismo.
  - Se eliminó `tools/inference/CMakeCache.txt` (archivo de build trackeado) para permitir reconfiguración out-of-source en distintos entornos.
- GQA KV workaround:
  - Expansión de `attn_k.weight`/`attn_v.weight` cuando `num_heads_kv < num_heads` para evitar accesos ilegales (replica heads KV por grupo).

## Reproducción local (CPU-only / validación de offsets)
Comandos previstos:
```bash
cd /media/leandro/D08A27808A2762683/gretacore/gretacore_local_clean
mkdir -p tools/inference/build
cd tools/inference/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
Nota: en este entorno local, luego de remover `tools/inference/CMakeCache.txt` y ajustar CMakeLists, `cmake` y `make` completaron correctamente. Se observaron warnings de `hipStreamSynchronize` ignorando `[[nodiscard]]` en `layer_trace.cpp` (sin impacto funcional para B3.6).

## Ejecución MI300X (B3.6)
```bash
OUTDIR=/root/gretacore/artifacts/alignment/2026-02-03
export GRETA_TRACE_READOUT=1
export GRETA_TRACE_READOUT_OUT=$OUTDIR/b3_6_readout.jsonl
export GRETA_TRACE_PREFILL_DECODE=1
export GRETA_TRACE_PREFILL_DECODE_OUT=$OUTDIR/b3_6_prefill_decode.jsonl
export GRETA_TRACE_LANDSCAPE=1
export GRETA_TRACE_LANDSCAPE_OUT=$OUTDIR/b3_6_landscape.jsonl
MODEL=/root/gretacore/models/llama3_8b_q4/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
cd /root/gretacore/tools/inference/build
./greta_infer --model "$MODEL" --prompt "Hi" --max-tokens 16 --greedy --debug-decode 16 \
  2>&1 | tee $OUTDIR/b3_6_run.log
```

## JSONL esperados (campos)
### GRETA_TRACE_READOUT / GRETA_TRACE_PREFILL_DECODE
- `phase` (prefill/decode)
- `step`
- `token_index`
- `hidden_offset`
- `hidden_ptr`
- `hidden_hash`
- `hidden_min`, `hidden_max`, `hidden_mean`
- `logits_offset`
- `logits_ptr`
- `logits_hash`
- `logits_min`, `logits_max`, `logits_mean`
- `vocab`

### GRETA_TRACE_LANDSCAPE
- `step`
- `top1` {`id`, `logit`}
- `top2` {`id`, `logit`}
- `gap`
- `entropy_topk`
- `top5` [{`id`, `logit`}...]

## Logs adicionales esperados
Cuando `GRETA_TRACE_*` esté activo:
- `[GRETA_TRACE_D2H] tensor=... step=... layer=... src_ptr=... alloc_bytes=... offset_bytes=... size_bytes=...`
- `[GRETA_TRACE_SHAPE] ...` si hay inconsistencia de configuración.
Durante carga de pesos (si aplica GQA):
- `[GRETA_LOAD] Expanded blk.X.attn_[k|v].weight KV heads A -> B (repeat per group)`

## Criterio de éxito B3.6
- No hay crash en `hipMemcpy D2H`.
- Se generan `b3_6_readout.jsonl`, `b3_6_prefill_decode.jsonl`, `b3_6_landscape.jsonl` y `b3_6_run.log`.
- Readout/landscape muestran variación coherente por step.

L.E.T / Leandro Emanuel Timberini
