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
- `GRETA_LMHEAD_FORCE_ROUTE=valu|mfma`  
  Fuerza ruta LM head (prefill+decode) para aislar MFMA/VALU.
- `GRETA_LMHEAD_FORCE_ROUTE_DECODE=valu|mfma`  
  Fuerza ruta LM head **solo en decode**.

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
L.E.T / Leandro Emanuel Timberini
