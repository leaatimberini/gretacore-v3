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
- GQA KV soporte:
  - Ajuste de pipeline a `num_heads_kv` (buffers/KV-cache/kernels) y transposición automática de `attn_k.weight`/`attn_v.weight` si vienen en layout `[D, KV]` (se normaliza a `[KV, D]`).
  - Decode usa kernel no-fused con RoPE aplicado en scheduler; prefill y decode mapean `kv_head = head / group`.

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
export GRETA_INT4_WEIGHTS=1   # usar int4 si VRAM disponible es baja
export GRETA_MAX_SEQ_LEN=256  # reduce KV-cache/activations si VRAM está saturada
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
- `[GRETA_LOAD] Transposed blk.X.attn_[k|v].weight from [D, KV] to [KV, D]`

## Criterio de éxito B3.6
- No hay crash en `hipMemcpy D2H`.
- Se generan `b3_6_readout.jsonl`, `b3_6_prefill_decode.jsonl`, `b3_6_landscape.jsonl` y `b3_6_run.log`.
- Readout/landscape muestran variación coherente por step.

## Estado VRAM MI300X (observado)
- `rocm-smi` reportó VRAM total ~205.8GB, usada ~196.8GB (quedan ~9GB). Esto puede forzar el uso de `GRETA_INT4_WEIGHTS=1` para evitar OOM durante carga de pesos.

## Incidente: saturación VRAM MI300X (evidencia + mitigación)
### Evidencia (rocm-smi --showmemuse --showpids)
Antes del reclaim (2026-02-03):
```
GPU[0] : GPU Memory Allocated (VRAM%): 37
KFD process information:
PID    PROCESS NAME     GPU(s)  VRAM USED
248944 python           1       77420392448
248342 vllm             0       0
248879 VLLM::EngineCor  0       0
```
Después de reclaim:
```
GPU[0] : GPU Memory Allocated (VRAM%): 0
KFD process information:
No KFD PIDs currently running
```

### Impacto (ES)
- La corrida full de B3.6 en MI300X sigue fallando por OOM en el `Embedding Lookup` aun con VRAM liberada y `GRETA_MAX_SEQ_LEN=256`.
- No se generaron JSONL (`b3_6_readout.jsonl`, `b3_6_prefill_decode.jsonl`, `b3_6_landscape.jsonl`).

### Impact (EN)
- The full B3.6 run on MI300X still fails with OOM at `Embedding Lookup` even after reclaim and `GRETA_MAX_SEQ_LEN=256`.
- JSONL artifacts were not produced (`b3_6_readout.jsonl`, `b3_6_prefill_decode.jsonl`, `b3_6_landscape.jsonl`).

### Mitigación / Mitigation
- Reclaim VRAM (kill vLLM / python VRAM holders) before running.
- Use `GRETA_INT4_WEIGHTS=1` and `GRETA_MAX_SEQ_LEN=256` to minimize footprint.
- Intento de Plan Mini con modelo pequeño no fue posible: `tinyllama.gguf` no existe; `dummy_v1.gguf` no es GGUF; `greta-v1.gguf` es equivalente a 8B y reproduce el OOM en `Embedding Lookup`.

### Próximo paso / Next step
- Ejecutar B3.6 en una instancia MI300X con VRAM libre y sin servicios vLLM/Open-WebUI, o provisionar un modelo GGUF realmente pequeño para validar el pipeline de trazas.

## Resultados (MI300X)
- Run con `GRETA_INT4_WEIGHTS=1` y `GRETA_MAX_SEQ_LEN=256` (Llama3 8B Q4):
  - Carga de pesos completó, pero falló en `Embedding Lookup launch failed: out of memory`.
  - No se generaron JSONL (`b3_6_readout.jsonl`, `b3_6_prefill_decode.jsonl`, `b3_6_landscape.jsonl`).
  - `b3_6_run.log` contiene el detalle del error.
- Plan Mini:
  - `tinyllama.gguf` no existe en `/root/gretacore/models`.
  - `dummy_v1.gguf` no es GGUF (falla con `Failed to open model: Not GGUF`).
  - `greta-v1.gguf` reproduce el mismo OOM en `Embedding Lookup`.
- Próximo paso requerido: instancia MI300X con VRAM libre estable o modelo GGUF pequeño disponible.

## Reintento post-reboot (2026-02-03 17:21 UTC)
- VRAM inicial en 0% según `rocm-smi` tras reboot y reclaim.
- Run con `GRETA_INT4_WEIGHTS=1` y `GRETA_MAX_SEQ_LEN=256` (Llama3 8B Q4) completó generación.
- JSONL generados:
  - `b3_6_readout.jsonl` (5.6K)
  - `b3_6_prefill_decode.jsonl` (5.6K)
  - `b3_6_landscape.jsonl` (4.3K)
- Log:
  - `b3_6_run.log` (68K)

L.E.T / Leandro Emanuel Timberini
