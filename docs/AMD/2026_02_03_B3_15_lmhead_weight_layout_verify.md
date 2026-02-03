# B3.15 — LM Head Weight Layout Verification

## Resumen Ejecutivo (ES)
Objetivo: verificar layout/stride del LM head weight (W_vocab) comparando predicciones CPU (row vs col) con logits GPU para tokens fijos. Esto valida si el layout de pesos es coherente con la ruta de cómputo.

## Executive Summary (EN)
Goal: verify LM head weight layout/stride (W_vocab) by comparing CPU predictions (row vs col) with GPU logits for fixed tokens. This validates whether weight layout matches the compute path.

## Metodología (ES/EN)
- Flag: `GRETA_TRACE_LMHEAD_W_VERIFY=1`
- Output: `b3_15_lmhead_w_verify.jsonl`
- Tokens verificados: `79`, `96965`, `12345`
- Ventana: W[token_id,0:16] y W[0:16,token_id]
- Métricas: `abs_err_row`, `abs_err_col`, `best_layout`, hashes de ventana

## Resultados (ES)
Resultados del JSONL: best_layout=row_major_match para 79/96965/12345, pero row_window/col_window=0 y row_logit/col_logit=0. Esto indica lectura de pesos nula o ruta de verificación no compatible con el formato real (posible INT4/packing).

## Results (EN)
JSONL shows best_layout=row_major_match for 79/96965/12345, but row/col windows are all zeros and row/col logits are 0. This suggests null weight reads or a verify path incompatible with the real weight format (likely INT4/packing).

## Conclusión (ES)
La verificación directa de W en FP16 no es concluyente; se requiere verificación compatible con INT4/packing o desactivar MFMA mientras se corrige.

## Conclusion (EN)
Direct FP16 W verification is inconclusive; need INT4/packing-aware verification or disable MFMA while fixing.

## Próximo Paso / Next Step (B3.16)
Aplicar fix mínimo en LM head MFMA según evidencia.

---
L.E.T / Leandro Emanuel Timberini
