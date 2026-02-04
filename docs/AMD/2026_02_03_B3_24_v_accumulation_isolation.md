# B3.24 — V Accumulation Isolation (decode0, layer 31)

## Resumen Ejecutivo (ES)
Objetivo: aislar si la divergencia en decode0 (layer 31) proviene del **layout de V**, del **acumulado P*V**, o del **writeback/combina de heads**. Se instrumentó un trace específico (decode0, layer 31, head 0) con ventana de keys y muestreo de dims, comparando GPU vs referencia CPU FP64. Resultado: **v_layout_best = col** con MAE ~0.038–0.048 y `pv_mae` igual a `attn_out_mae`. Conclusión: el problema apunta a **interpretación/layout de V en el path decode** (row/col o stride), no al softmax ni a la acumulación numérica.

## Executive Summary (EN)
Goal: isolate whether decode0 divergence (layer 31) originates in **V layout**, **P*V accumulation**, or **writeback/head combine**. We added a decode0 trace (layer 31, head 0) with key window + dim sampling, comparing GPU vs CPU FP64. Result: **v_layout_best = col** with MAE ~0.038–0.048 and `pv_mae` equals `attn_out_mae`. Conclusion: the issue points to **V layout/interpretation in the decode path** (row/col or stride), not softmax nor accumulation precision.

## Metodología / Methodology
Commit: `21b87f602d88c139521f9ae81082b6b81cd36c80`

Flags:
- `GRETA_TRACE_ATTN_VACC=1`
- `GRETA_TRACE_ATTN_LAYER=31`
- `GRETA_TRACE_ATTN_HEAD=0`
- `GRETA_TRACE_ATTN_KEYS_WINDOW=64`
- `GRETA_TRACE_ATTN_DIMS_SAMPLE=16`
- `GRETA_TRACE_ATTN_OUT=/root/gretacore/artifacts/alignment/2026-02-03/b3_24_attn_vacc.jsonl`
- `GRETA_TRACE_PROMPT_ID=p4_sys|p5_ba`

Prompts:
- `p4_sys`
- `p5_ba`

Artefactos (local):
- `artifacts_remote/2026-02-03/b3_24/root/gretacore/artifacts/alignment/2026-02-03/b3_24_attn_vacc.jsonl`
- `artifacts_remote/2026-02-03/b3_24/b3_24_analysis.txt`

## Resultados / Results
Tabla (layout + error):

| Prompt | v_layout_best | pv_mae | pv_max_diff | attn_out_mae | attn_out_max_diff |
| --- | --- | ---: | ---: | ---: | ---: |
| p4_sys | col | 0.0375317 | 0.0769076 | 0.0375317 | 0.0769076 |
| p5_ba  | col | 0.0480429 | 0.0870157 | 0.0480429 | 0.0870157 |

Interpretación: el best‑fit consistente es **col_major**, lo que indica que el path de decode estaría leyendo V con un layout/stride incorrecto (o asumiendo transposición opuesta). El error aparece ya en P*V y se arrastra a `attn_out`.

## Conclusión / Conclusion
El colapso en decode0 no proviene del softmax ni de precisión de acumulación, sino de **layout/stride de V en decode**. Se requiere corregir la interpretación de V (row/col, stride, o transposición) en la ruta de atención decode, y revalidar contra el ref FP64.

## Próximo Paso / Next Step (B3.25)
Aplicar el fix mínimo en el path decode para V (layout/stride o transposición), repetir p4/p5 con `GRETA_TRACE_ATTN_VACC` y confirmar `v_layout_best=row` + `pv_mae` ~0 y `attn_out_mae` ~0.

---
L.E.T / Leandro Emanuel Timberini
