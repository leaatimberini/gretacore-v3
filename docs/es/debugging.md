# GRETA CORE – Debugging (LLM Pipeline)

Versión: 1.0  
Estado: Activo  
Fase del Proyecto: Fase 3 – Pipeline de Inferencia LLM (B3.x)  
Idioma: Español

---

## Objetivo
Establecer un flujo de depuración reproducible para aislar errores en el pipeline de inferencia LLM, con foco en coherencia prefill↔decode y validación de rutas de cómputo (LM head, atención, KV cache).

## Principios
- Local-first: todo cambio se aplica en el repo local antes de validar en MI300X.
- Trazas mínimas y controladas por flags.
- Evidencia siempre exportada en JSONL y documentada en `docs/AMD/`.

## Flags Principales (B3.x)
- `GRETA_TRACE_PREFILL_DECODE_DELTA=1`  
  Emite JSONL con `prefill_last` y `decode0` (hash/stats/top1/top2/gap + auditoría LM head).
- `GRETA_TRACE_LMHEAD_CPU_PROBE=1`  
  Ejecuta un matvec CPU mínimo para validar coherencia GPU↔CPU.
- `GRETA_TRACE_HIDDEN_EQUIV=1`  
  Registra equivalencia de hidden prefill↔decode.
- `GRETA_TRACE_LAYER_DELTA=1`  
  Hash/stats de `attn_out`, `mlp_out`, `x_out` en layers 0 y last (decode).
- `GRETA_TRACE_ATTN_DECODE_VERIFY=1`  
  Verifica atención decode con referencia (hash/MAE).
- `GRETA_ATTN_DECODE_REF=1`  
  Habilita recompute de referencia en decode0.
- `GRETA_TRACE_ATTN_SOFTMAX=1`  
  Captura aislamiento de softmax en decode0 (QK/softmax ventana vs FP64).
- `GRETA_TRACE_ATTN_LAYER=31`  
  Selecciona una capa para aislamiento de softmax (default last).
- `GRETA_TRACE_ATTN_HEAD=0`  
  Selecciona un head para aislamiento de softmax.
- `GRETA_TRACE_ATTN_KEYS_WINDOW=64`  
  Tamaño de ventana alrededor de la posición actual.
- `GRETA_TRACE_ATTN_OUT=/root/gretacore/artifacts/alignment/.../b3_23_attn_softmax.jsonl`  
  Path de salida JSONL para aislamiento de softmax.
- `GRETA_TRACE_PROMPT_ID=p4_sys`  
  Etiqueta opcional del prompt para trazas.
- `GRETA_TRACE_ATTN_LAYERS="0,1,2,31"`  
  Selecciona layers para trazas de atención decode.
- `GRETA_TRACE_ATTN_POINTS="q,k,v,attn_out,x_out"`  
  Selecciona tensores para trazas de atención decode.
- `GRETA_TRACE_KV_INVARIANTS=1`  
  Chequeos de offsets/posiciones en KV cache.
- `GRETA_FORCE_ATTN_DECODE_KERNEL=auto|manual|fused`  
  Fuerza ruta de kernel de atención decode.
- `GRETA_FORCE_ATTN_DECODE_MATMUL=auto|valu|mfma`  
  Fuerza ruta de GEMM en decode (Q/K/V/O) para aislar kernels.
- `GRETA_LMHEAD_FORCE_ROUTE=valu|mfma`  
  Fuerza ruta LM head (prefill+decode) para aislar MFMA/VALU.
- `GRETA_LMHEAD_FORCE_ROUTE_DECODE=valu|mfma`  
  Fuerza ruta LM head **solo en decode**.

**Nota B3.23:** QK y softmax coinciden con FP64 en decode0 (layer 31 head 0, ventana). La divergencia es más probable en el acumulado de V / `attn_out`.

## Flujo Recomendada de Diagnóstico
1. **Baseline**: ejecutar p4_sys y p5_ba con trazas delta.
2. **Aislar LM head**: forzar rutas MFMA/VALU y comparar `cpu_probe_agrees_gpu`.
3. **Equivalencia de hidden**: verificar divergencias entre `prefill_last` y `decode0`.
4. **Layer delta**: revisar divergencias tempranas (layer 0 / last).
5. **Atención/KV**: si LM head es consistente, auditar atención decode y cache.

## Evidencia y Reporte
- JSONL y logs se empaquetan y descargan al local.
- Cada bloque genera un informe AMD en `docs/AMD/` con ES/EN, tablas y extractos.

---
