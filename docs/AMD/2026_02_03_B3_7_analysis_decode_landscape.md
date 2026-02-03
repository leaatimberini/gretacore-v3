# B3.7 Analysis: Decode Landscape Degeneration (B3.6 JSONL)

Fecha: 2026-02-03
Bloque: B3.7 (análisis + fix mínimo)

## Resumen ejecutivo (ES)
Se analizó B3.6 usando `b3_6_readout.jsonl`, `b3_6_prefill_decode.jsonl` y `b3_6_landscape.jsonl`. El `top1` permanece constante (id=96965) desde prefill hasta decode, con `gap` medio 1.3075 y entropía top-k ~2.692. Los punteros de logits/hidden son estables y los hashes varían por step, por lo que no se observa inconsistencia de trazas. Clasificación: **Caso B (decode repetitivo inducido)**. Fix mínimo aplicado: **probe de verificación de embedding** (`GRETA_TRACE_EMBED_VERIFY=1`) para validar layout/stride del embedding vs salida real.

## Executive summary (EN)
We analyzed B3.6 using `b3_6_readout.jsonl`, `b3_6_prefill_decode.jsonl`, and `b3_6_landscape.jsonl`. `top1` stays constant (id=96965) from prefill through decode, with mean `gap` 1.3075 and top-k entropy ~2.692. Logits/hidden pointers are stable and hashes vary per step, so traces are consistent. Classification: **Case B (repetitive decode)**. Minimal fix applied: **embedding verification probe** (`GRETA_TRACE_EMBED_VERIFY=1`) to validate embedding layout/stride vs actual output.

## Inputs (local)
- TGZ: `artifacts_remote/2026-02-03/gretacore_b3_6_artifacts.tgz`
- Extracted:
- `artifacts_remote/2026-02-03/extracted/artifacts/alignment/2026-02-03/b3_6_readout.jsonl`
- `artifacts_remote/2026-02-03/extracted/artifacts/alignment/2026-02-03/b3_6_prefill_decode.jsonl`
- `artifacts_remote/2026-02-03/extracted/artifacts/alignment/2026-02-03/b3_6_landscape.jsonl`
- Log: `artifacts_remote/2026-02-03/extracted/artifacts/alignment/2026-02-03/b3_6_run.log`

## Resultado cuantitativo (prefill vs decode)
| Métrica | Prefill | Decode |
|---|---|---|
| uniq_top1 | 1 | 1 |
| top1 (ej.) | 96965 | 96965 |
| gap mean | 1.3075 | 1.3075 |
| gap std | 0.0825 | 0.0825 |
| entropy_topk mean | 2.6925 | 2.6925 |

## Evidencia (extractos JSONL)
`b3_6_landscape.jsonl`:
```
{"step":0,"top1":{"id":96965,"logit":13.3423},"top2":{"id":198,"logit":11.8925},"gap":1.44973,"entropy_topk":2.68817}
{"step":1,"top1":{"id":96965,"logit":13.3429},"top2":{"id":198,"logit":11.9056},"gap":1.43734,"entropy_topk":2.68491}
```
`b3_6_prefill_decode.jsonl`:
```
{"phase":"prefill","step":0,"token_index":2,"logits_min":-13.0639,"logits_max":13.3423,"logits_mean":-0.457307}
{"phase":"decode","step":1,"token_index":3,"logits_min":-13.0584,"logits_max":13.3429,"logits_mean":-0.457419}
```

## Diagnóstico (Caso B)
- `top1` constante desde el primer step, pero `gap` moderado y entropía top-k no colapsada.
- `readout` y `prefill_decode` son consistentes en steps/ptrs/hashes.
- Esto sugiere landscape degenerado o embedding/normalization desalineado, no un bug de trazas.

## Fix mínimo aplicado
Se agregó **probe de verificación de embedding** bajo `GRETA_TRACE_EMBED_VERIFY=1`:
- Compara el embedding calculado (salida real) contra dos layouts posibles del tensor `token_embd`:
- `row_major` (offset = token * dim)
- `col_major` (offset = d * vocab + token)
- Reporta `mae_row`, `mae_col` y layout más cercano.

## Next run plan (MI300X)
```bash
OUTDIR=/root/gretacore/artifacts/alignment/2026-02-03
export GRETA_INT4_WEIGHTS=1
export GRETA_MAX_SEQ_LEN=256
export GRETA_TRACE_EMBED_VERIFY=1
./greta_infer --model /root/gretacore/models/llama3_8b_q4/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
  --prompt "Hi" --max-tokens 16 --greedy --debug-decode 16 \
  2>&1 | tee $OUTDIR/b3_7_run.log
```

L.E.T / Leandro Emanuel Timberini
