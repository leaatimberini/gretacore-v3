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
- `GRETA_TRACE_ATTN_L0_PIPE=1`  
  Traza la tubería de atención en layer0 (Q/K/V/QK/softmax/P·V/attn_out) para `prefill_last` vs `decode0`.
- `GRETA_TRACE_ATTN_L0_PIPE_OUT=/root/gretacore/artifacts/alignment/.../b3_30_attn_l0_pipe.jsonl`  
  Path de salida JSONL para la traza de layer0.
- `GRETA_TRACE_ATTN_L0_NORM=1`  
  Incluye `attn_norm_in` y `attn_norm_out` (RMSNorm entrada/salida) en la traza de layer0.
- `GRETA_TRACE_QKV_W_VERIFY=1`  
  Verifica el layout/packing de pesos QKV en layer0 (row vs col) entre `prefill_last` y `decode0`.
- `GRETA_TRACE_WO_W_VERIFY=1`  
- `GRETA_TRACE_POST_WO=1`  
- `GRETA_TRACE_POST_WO_OUT=/root/gretacore/artifacts/alignment/.../b3_41_post_wo.jsonl`  
- `GRETA_TRACE_POST_WO_LAYERS="0"`  
- `GRETA_TRACE_POST_WO_SAMPLE=1024`  
- `GRETA_TRACE_POST_WO_PHASES="prefill_last,decode0"`  
- `GRETA_TRACE_RMSNORM=1`  
- `GRETA_TRACE_RMSNORM_OUT=/root/gretacore/artifacts/alignment/.../b3_42_rmsnorm.jsonl`  
- `GRETA_TRACE_RMSNORM_LAYERS="0"`  
- `GRETA_TRACE_RMSNORM_SAMPLE=1024`  
- `GRETA_TRACE_RMSNORM_PHASES="prefill_last,decode0"`  
  Verifica el layout/packing de pesos WO en layer0 (row vs col) entre `prefill_last` y `decode0`.
- `GRETA_PREFILL_FORCE_WQ_ROW=1`  
  Fuerza la proyección Q en prefill a usar layout row (B3.34).
- `GRETA_PREFILL_FORCE_WK_ROW=1`  
  Fuerza la proyección K en prefill a usar layout row (B3.35).
- `GRETA_PREFILL_FORCE_WV_LAYOUT=row|col|auto`  
  Fuerza la proyección V en prefill a usar layout row/col (B3.36; recomendado `row`).
- `GRETA_PREFILL_QKV_LAYOUT=row|col|auto`  
  Selector explícito de layout para la proyección QKV en prefill (row recomendado para B3.34/B3.35).
- `GRETA_WO_LAYOUT_FORCE=row|col|auto`  
  Fuerza la interpretación de layout para WO (B3.40; recomendado `row`).
- `GRETA_QKV_FORCE_ROUTE=mfma|valu|auto`  
  Fuerza la ruta de proyección Q/K/V en **decode** (S=1).
- `GRETA_QKV_FORCE_GEMM=1`  
  Fuerza el uso de GEMM en decode (deshabilita el GEMV fused de QKV).
- `GRETA_TRACE_PROMPT_ID=p4_sys`  
  Etiqueta opcional del prompt para trazas.
- `GRETA_TRACE_ATTN_LAYERS="0,1,2,31"`  
  Selecciona layers para trazas de atención decode.
- `GRETA_TRACE_ATTN_POINTS="q,k,v,attn_out,x_out"`  
  Selecciona tensores para trazas de atención decode.
- `GRETA_TRACE_STAGE_POINTS="x_in,attn_out,wo_out,x_after_attn,ffn_norm,mlp_out,x_out,final_norm,lm_head_in,logits"`  
  Ejemplo de puntos de traza de stage (incluye `wo_out` para la proyección de salida de atención).
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
- `GRETA_TRACE_STAGE=1`  
  Emite JSONL por etapas para `prefill_last` vs `decode0`.
- `GRETA_TRACE_STAGE_OUT=/root/gretacore/artifacts/alignment/.../b3_27_stage.jsonl`  
  Path de salida JSONL para StageTrace.
- `GRETA_TRACE_STAGE_LAYERS="0,1,2,15,31"`  
  Selecciona capas para StageTrace.
- `GRETA_TRACE_STAGE_POINTS="x_in,attn_out,x_after_attn,mlp_out,x_out,x_after_mlp,final_rms,lm_head_in,logits"`  
  Selecciona tensores para StageTrace.
- `GRETA_TRACE_STAGE_PHASES="prefill_last,decode0"`  
  Selecciona fases para StageTrace.
- `GRETA_TRACE_STAGE_DEBUG_INPUT=1`  
  Agrega campos de semántica de entrada (`x_in_src_kind`, `x_in_token_index_used`, `x_in_offset_bytes`, `x_in_ptr`, `x_in_alloc_bytes`, `prompt_tokens`, `kv_pos`, `decode_step`).

**Nota B3.23:** QK y softmax coinciden con FP64 en decode0 (layer 31 head 0, ventana). La divergencia es más probable en el acumulado de V / `attn_out`.
**Nota B3.27:** La primera divergencia aparece en `x_in` de layer 0, indicando mismatch en semántica de entrada de decode (antes de attention/MLP).
**Nota B3.59:** Divergencia en `x_in` resuelta. No se encontró zeroing. Consistencia de hash perfecta confirmada con metadata estandarizada (`token_id`, `route`). La divergencia es probable aguas abajo.

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
