# B3.5 Layer Trace Root Cause — GRETA CORE (MI300X)

## ES — Resumen Ejecutivo
La repetición greedy (top1 fijo=96965 “ISTA”) sigue ocurriendo. En esta iteración se restauró y corrigió la instrumentación de LayerTrace y se añadió parsing de metadatos GGUF para obtener configuración real del modelo (dim/hidden/heads/kv_heads/context/rope/eps). El run B3.5 aún falla con **hipMemcpy D2H illegal memory access** durante generación, por lo que el JSONL de trazas no se regeneró. Este fallo bloquea la verificación Q/K/V con el modelo Llama‑3 8B y requiere corrección adicional en el pipeline de decode/hidden/logits.

## ES — Qué se hizo
1) **Recuperación del LayerTrace** (per‑layer, JSONL) y analizador.
2) **Soporte FP16 en trazas** (k/v/attn_out/mlp_*), evitando lecturas inválidas por tamaño.
3) **Parsing GGUF** de parámetros reales del modelo:
   - `llama.embedding_length` → `dim`
   - `llama.feed_forward_length` → `hidden_dim`
   - `llama.block_count` → `num_layers`
   - `llama.attention.head_count` → `num_heads`
   - `llama.attention.head_count_kv` → `num_heads_kv`
   - `llama.context_length`, `llama.rope.freq_base`, `llama.norm_eps`
4) **Inicialización del scheduler** con config GGUF real antes de alocar buffers.
5) **Tokenizer** usando vocab GGUF cuando está disponible.

## ES — Resultado
- Configuración GGUF se detecta correctamente (ej. hidden=14336, vocab=128256).
- La ejecución falla en generación con:
  - `hipMemcpy D2H failed: an illegal memory access was encountered`
- El archivo de trazas **no fue regenerado** en esta corrida (timestamp no cambió), por lo que **Q/K/V aún no se verifican** en JSONL.

## ES — Evidencia (paths)
- Run log (error): `/root/gretacore/artifacts/alignment/2026-02-03/b3_5_trace_run.log`
- Último JSONL previo (sin Q/K/V): `/root/gretacore/artifacts/alignment/2026-02-03/b3_5_layer_trace.jsonl`
- Verificación GitHub HEAD: `/root/gretacore/artifacts/alignment/2026-02-03/b3_5_github_head_final.txt`

## ES — Comando reproducible
```bash
export MODEL=/root/gretacore/models/llama3_8b_q4/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
export OUTDIR=/root/gretacore/artifacts/alignment/2026-02-03
mkdir -p "$OUTDIR"

export GRETA_TRACE_LAYER=1
export GRETA_TRACE_LAYER_OUT=$OUTDIR/b3_5_layer_trace.jsonl
export GRETA_TRACE_LAYER_LAYERS="0,1,2,15,31"
export GRETA_TRACE_LAYER_POINTS="x,norm_out,q,k,v,attn_out,ffn_norm,mlp_out,x_out"

/root/gretacore/tools/inference/build/greta_infer \
  --model "$MODEL" --prompt "Hi" --max-tokens 16 --greedy --debug-decode 16 \
  2>&1 | tee $OUTDIR/b3_5_trace_run.log
```

## ES — Hipótesis actual
El error D2H indica acceso inválido en GPU, probablemente en la ruta de decode/logits o en el manejo de buffers con configuración GQA (kv_heads). La instrumentación de capa está lista pero no alcanza a completar el run.

## ES — Próximos pasos (B3.6)
1) Aislar el kernel responsable del illegal access (activar chequeos HIP por kernel).
2) Validar tamaños/strides del logits buffer y del hidden final usado por lm_head.
3) Re‑ejecutar B3.5 con trazas Q/K/V completas y generar JSONL nuevo.

---

## EN — Executive Summary
Greedy repetition (top1 fixed=96965 “ISTA”) persists. In this iteration, LayerTrace instrumentation was restored and fixed, and GGUF metadata parsing was added to obtain the real model config (dim/hidden/heads/kv_heads/context/rope/eps). The B3.5 run still fails with **hipMemcpy D2H illegal memory access** during generation, so the trace JSONL was not regenerated. This blocks Q/K/V verification for Llama‑3 8B and requires further decode/hidden/logits fixes.

## EN — What was done
1) **LayerTrace restored** (per‑layer, JSONL) and analyzer added.
2) **FP16 trace support** (k/v/attn_out/mlp_*), avoiding invalid reads by size.
3) **GGUF parsing** for real model parameters:
   - `llama.embedding_length` → `dim`
   - `llama.feed_forward_length` → `hidden_dim`
   - `llama.block_count` → `num_layers`
   - `llama.attention.head_count` → `num_heads`
   - `llama.attention.head_count_kv` → `num_heads_kv`
   - `llama.context_length`, `llama.rope.freq_base`, `llama.norm_eps`
4) **Scheduler init** uses GGUF config before buffer allocation.
5) **Tokenizer** uses GGUF vocab when available.

## EN — Result
- GGUF config is detected correctly (e.g., hidden=14336, vocab=128256).
- Generation fails with:
  - `hipMemcpy D2H failed: an illegal memory access was encountered`
- Trace file **was not regenerated** in this run (timestamp unchanged), so **Q/K/V are still missing** from JSONL.

## EN — Evidence (paths)
- Run log (error): `/root/gretacore/artifacts/alignment/2026-02-03/b3_5_trace_run.log`
- Last JSONL (no Q/K/V): `/root/gretacore/artifacts/alignment/2026-02-03/b3_5_layer_trace.jsonl`
- GitHub HEAD verification: `/root/gretacore/artifacts/alignment/2026-02-03/b3_5_github_head_final.txt`

## EN — Repro command
```bash
export MODEL=/root/gretacore/models/llama3_8b_q4/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
export OUTDIR=/root/gretacore/artifacts/alignment/2026-02-03
mkdir -p "$OUTDIR"

export GRETA_TRACE_LAYER=1
export GRETA_TRACE_LAYER_OUT=$OUTDIR/b3_5_layer_trace.jsonl
export GRETA_TRACE_LAYER_LAYERS="0,1,2,15,31"
export GRETA_TRACE_LAYER_POINTS="x,norm_out,q,k,v,attn_out,ffn_norm,mlp_out,x_out"

/root/gretacore/tools/inference/build/greta_infer \
  --model "$MODEL" --prompt "Hi" --max-tokens 16 --greedy --debug-decode 16 \
  2>&1 | tee $OUTDIR/b3_5_trace_run.log
```

## EN — Current hypothesis
The D2H failure indicates an invalid GPU access, likely in decode/logits or buffer management under GQA (kv_heads). LayerTrace is ready but the run does not complete yet.

## EN — Next steps (B3.6)
1) Isolate the kernel causing illegal access (HIP error checks per kernel).
2) Validate logits buffer sizes/strides and the final hidden used by lm_head.
3) Re‑run B3.5 to regenerate JSONL with full Q/K/V traces.

---

L.E.T / Leandro Emanuel Timberini
