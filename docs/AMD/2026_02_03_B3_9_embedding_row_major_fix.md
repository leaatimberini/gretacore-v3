# B3.9 Embedding Row-Major Fix (Runtime Toggle)

Fecha: 2026-02-03
Bloque: B3.9

## Resumen ejecutivo (ES)
B3.8 confirmó `col_major_match` en el probe de embedding, lo que sugiere lectura transpuesta vs el layout esperado `row_major`. Se implementó un toggle de runtime `GRETA_EMBED_LAYOUT` con default `row` y se extendió el probe para reportar `layout_used` y `layout_probe_best`. El objetivo es validar si el attractor (`top1=96965`) desaparece al forzar row-major.

## Executive summary (EN)
B3.8 confirmed `col_major_match` in the embedding probe, suggesting a transposed read vs the expected `row_major` layout. We added a runtime toggle `GRETA_EMBED_LAYOUT` with default `row` and extended the probe to report `layout_used` and `layout_probe_best`. The goal is to validate whether the attractor (`top1=96965`) disappears when forcing row-major.

## Cambio implementado
- Toggle de layout: `GRETA_EMBED_LAYOUT=row|col` (default `row`).
- Embedding lookup usa row-major por defecto: `token * dim + d`.
- Probe extendido: `layout_used`, `layout_probe_best`, `mae_row`, `mae_col`.

## Comandos de reproducción (MI300X)
```
export GRETA_INT4_WEIGHTS=1
export GRETA_MAX_SEQ_LEN=256
export GRETA_TRACE_EMBED_VERIFY=1
export GRETA_TRACE_DECODE_STATE=1
export GRETA_TRACE_LOGITS=1
export GRETA_EMBED_LAYOUT=row

MODEL=/root/gretacore/models/llama3_8b_q4/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
BIN=/root/gretacore/tools/inference/build/greta_infer

$BIN --model "$MODEL" --prompt "Hi" --max-tokens 16 --greedy --debug-decode 16
$BIN --model "$MODEL" --prompt "Write one short sentence about Buenos Aires." --max-tokens 32 --greedy --debug-decode 16
```

## Criterio de éxito
- `layout_used=row` y `layout_probe_best=row_major_match` (mae_row≈0).
- `top1` deja de ser constante 96965 al menos en steps 0..3 para algún prompt simple.

## Resultados (pendiente)
Se completará tras el run MI300X B3.9.

L.E.T / Leandro Emanuel Timberini
