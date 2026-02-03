# B3.8 Embedding Layout Verification (GRETA_TRACE_EMBED_VERIFY)

Fecha: 2026-02-03
Bloque: B3.8

## Resumen ejecutivo (ES)
Se ejecutó un run corto en MI300X con `GRETA_TRACE_EMBED_VERIFY=1` para confirmar si el attractor (`top1=96965`) se debe a layout incorrecto del embedding. El probe reportó **match con layout col_major**, lo que indica que el embedding lookup está leyendo en formato `d * vocab + token`. Resultado: **Outcome A (mismatch del layout esperado row_major)**, por lo que el próximo fix (B3.9) debe corregir el layout del embedding weight o la indexación en el kernel.

## Executive summary (EN)
We ran a short MI300X job with `GRETA_TRACE_EMBED_VERIFY=1` to determine whether the attractor (`top1=96965`) is caused by an embedding layout issue. The probe reported **col_major match**, meaning the embedding lookup reads as `d * vocab + token`. Result: **Outcome A (row_major mismatch)**, so the next fix (B3.9) should correct the embedding weight layout or the indexing in the embedding kernel.

## Config exacta del run
```
export GRETA_INT4_WEIGHTS=1
export GRETA_MAX_SEQ_LEN=256
export GRETA_TRACE_EMBED_VERIFY=1
export GRETA_TRACE_DECODE_STATE=1
export GRETA_TRACE_LOGITS=1

MODEL=/root/gretacore/models/llama3_8b_q4/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf

/root/gretacore/tools/inference/build/greta_infer \
  --model "$MODEL" \
  --prompt "Hi" \
  --max-tokens 16 \
  --greedy \
  --debug-decode 16
```
Nota: usar `tools/inference/build/greta_infer`. El wrapper `tools/inference/greta_infer` no era ejecutable (Permission denied) en MI300X.

## Evidencia (extracto log, 1 línea)
```
[GRETA_TRACE_EMBED_VERIFY] token=105 seq_idx=2 mae_row=0.0129823 mae_col=0 max_row=0.063498 max_col=0 layout=col_major_match
```

## Resultado
- **Outcome A**: Mismatch confirmado entre layout esperado (row_major) y observado (col_major_match).
- Implica que el kernel de embedding actual está leyendo en orden columna o que el weight layout está transpuesto respecto al modelo.

## Próximo paso recomendado (B3.9)
Corregir el layout del embedding weight o la indexación del kernel para usar row_major (token * dim + d), y revalidar el attractor.

L.E.T / Leandro Emanuel Timberini
