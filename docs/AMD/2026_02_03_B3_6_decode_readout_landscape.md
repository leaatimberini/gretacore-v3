# B3.6 — Decode Readout & Logits Landscape (Readout/Prefill–Decode)

## ES — Resumen ejecutivo
Se implementó instrumentación de **readout exacto del último token** y **landscape de logits** (top1/top2/top5, gap, entropía topK) para cerrar la degeneración greedy. La ejecución en MI300X **falló antes de generar tokens** por error de `hipMemcpy D2H` (illegal memory access), por lo que **no se produjeron JSONL** de readout/landscape. El fallo es reproducible y queda documentado con logs y commit.

## ES — Objetivo
Verificar:
1) El hidden_state exacto usado por lm_head (offset/puntero/hash).
2) Equivalencia prefill vs decode.
3) Landscape de logits por paso.

## ES — Metodología reproducible
Comandos (MI300X):
```
cd /root/gretacore/tools/inference/build
make -B -j$(nproc)

OUTDIR=/root/gretacore/artifacts/alignment/2026-02-03
mkdir -p "$OUTDIR"
export GRETA_TRACE_READOUT=1
export GRETA_TRACE_READOUT_OUT=$OUTDIR/b3_6_readout.jsonl
export GRETA_TRACE_PREFILL_DECODE=1
export GRETA_TRACE_PREFILL_DECODE_OUT=$OUTDIR/b3_6_prefill_decode.jsonl
export GRETA_TRACE_LANDSCAPE=1
export GRETA_TRACE_LANDSCAPE_OUT=$OUTDIR/b3_6_landscape.jsonl
MODEL=/root/gretacore/models/llama3_8b_q4/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
/root/gretacore/tools/inference/build/greta_infer --model "$MODEL" --prompt "Hi" --max-tokens 16 --greedy --debug-decode 16 \
  2>&1 | tee $OUTDIR/b3_6_run.log
```

## ES — Resultados
- Resultado: **ERROR** `hipMemcpy D2H failed: an illegal memory access was encountered`
- Tokens generados: 0
- No se generaron JSONL de readout/landscape por abort temprano (solo log del run).

## ES — Evidencia
- Log: `/root/gretacore/artifacts/alignment/2026-02-03/b3_6_run.log`
- Commit: `6602886af2c2caef1b31ff57a5b0cf04dae6e558`

## ES — Impacto para AMD vs H100/CUDA
La instrumentación B3.6 agrega trazabilidad detallada del readout (offset/puntero/hash) y del landscape de logits, elevando la capacidad de diagnóstico por encima del enfoque “caja negra”. La falla D2H indica un problema de robustez del pipeline de copia/offset que debe resolverse para sostener comparabilidad con H100+CUDA y mantener calidad estable en decode.

## ES — Próximos pasos
1) Aislar el `hipMemcpy D2H` (validar tamaños/offsets de logits/hidden para prefill y decode, y chequear prompt_tokens.size() > 0).
2) Agregar chequeos de límites previos a `copy_to_host_offset` y confirmar `logits_` allocation.
3) Re-ejecutar B3.6 con los JSONL habilitados.

---

## EN — Executive summary
We implemented **exact last-token readout** and **logits landscape tracing** (top1/top2/top5, gap, topK entropy) to close the greedy degeneration. The MI300X run **failed before token generation** with `hipMemcpy D2H` illegal memory access, therefore **no readout/landscape JSONL** artifacts were produced. The failure is reproducible and documented with logs and commit.

## EN — Objective
Verify:
1) The exact hidden_state used by lm_head (offset/pointer/hash).
2) Prefill vs decode equivalence.
3) Per-step logits landscape.

## EN — Reproducible methodology
Commands (MI300X):
```
cd /root/gretacore/tools/inference/build
make -B -j$(nproc)

OUTDIR=/root/gretacore/artifacts/alignment/2026-02-03
mkdir -p "$OUTDIR"
export GRETA_TRACE_READOUT=1
export GRETA_TRACE_READOUT_OUT=$OUTDIR/b3_6_readout.jsonl
export GRETA_TRACE_PREFILL_DECODE=1
export GRETA_TRACE_PREFILL_DECODE_OUT=$OUTDIR/b3_6_prefill_decode.jsonl
export GRETA_TRACE_LANDSCAPE=1
export GRETA_TRACE_LANDSCAPE_OUT=$OUTDIR/b3_6_landscape.jsonl
MODEL=/root/gretacore/models/llama3_8b_q4/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
/root/gretacore/tools/inference/build/greta_infer --model "$MODEL" --prompt "Hi" --max-tokens 16 --greedy --debug-decode 16 \
  2>&1 | tee $OUTDIR/b3_6_run.log
```

## EN — Results
- Result: **ERROR** `hipMemcpy D2H failed: an illegal memory access was encountered`
- Generated tokens: 0
- No readout/landscape JSONL artifacts due to early abort (run log only).

## EN — Evidence
- Log: `/root/gretacore/artifacts/alignment/2026-02-03/b3_6_run.log`
- Commit: `6602886af2c2caef1b31ff57a5b0cf04dae6e558`

## EN — AMD impact vs H100/CUDA
B3.6 instrumentation adds fine-grained readout and logits landscape tracing, improving auditability beyond black-box inference. The D2H failure points to a pipeline robustness issue (copy/offset) that must be resolved to sustain H100+CUDA-grade stability and decode quality.

## EN — Next steps
1) Isolate the `hipMemcpy D2H` failure (validate logits/hidden sizes and offsets for prefill/decode, ensure prompt_tokens.size() > 0).
2) Add bounds checks before `copy_to_host_offset` and verify `logits_` allocation.
3) Re-run B3.6 with JSONL artifacts enabled.

L.E.T / Leandro Emanuel Timberini
