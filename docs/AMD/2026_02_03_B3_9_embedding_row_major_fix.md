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
Run MI300X completado con `GRETA_EMBED_LAYOUT=row`.

Extracto (probe):
```
[GRETA_TRACE_EMBED_VERIFY] token=105 seq_idx=2 mae_row=0 mae_col=0.0129823 max_row=0 max_col=0.063498 layout_used=row layout_probe_best=row_major_match
```

Top tokens (Prompt 1: "Hi"):
```
Top tokens: 96965(13.3429) 198(11.8958) 99668(11.4639) 52263(11.2216) 86537(10.4427)
```

Top tokens (Prompt 2: "Write one short sentence about Buenos Aires."):
```
Top tokens: 79(11.9995) 18(11.0066) 95(10.7475) 64938(10.7399) 59036(9.78043)
```

Conclusión: **PARTIAL**. El probe confirma `row_major_match`, pero el prompt corto aún muestra top1=96965 y no hay evidencia multi-step de variación. Se requiere un run con trazas multi-step para confirmar la desaparición total del attractor.

Next B3.10 (propuesto): ejecutar un run con trazas multi-step de top1 (o landscape JSONL) en prompts variados para confirmar que top1 cambia en steps 0..3; si persiste, enfocar RMSNorm/LM head.

L.E.T / Leandro Emanuel Timberini
