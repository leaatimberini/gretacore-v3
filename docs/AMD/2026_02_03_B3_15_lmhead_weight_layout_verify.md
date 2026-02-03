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
TBD (llenar con `best_layout` y errores por token).

## Results (EN)
TBD (fill with `best_layout` and per-token errors).

## Conclusión (ES)
TBD según layout dominante (row vs col).

## Conclusion (EN)
TBD based on dominant layout (row vs col).

## Próximo Paso / Next Step (B3.16)
Aplicar fix mínimo en LM head MFMA según evidencia.

---
L.E.T / Leandro Emanuel Timberini
