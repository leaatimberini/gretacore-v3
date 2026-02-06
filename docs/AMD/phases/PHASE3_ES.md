# GRETA CORE Fase 3: Optimización AMD MI300X

**Fecha de Inicio:** 2026-01-XX  
**Hardware:** AMD MI300X  
**Estado:** EN PROGRESO  
**Rama Actual:** `b3_59_embedding_debug_input_audit`

---

## Resumen Ejecutivo

La Fase 3 se enfoca en optimizar GRETA CORE para hardware AMD MI300X, abordar problemas de precisión y validar el pipeline de inferencia completo.

---

## Objetivos

1. **Validación de Precisión**: Asegurar precisión float16 en todas las operaciones
2. **Integración de Pipeline Completo**: Validar inferencia end-to-end
3. **Optimización de Backend AMD**: Aprovechar características específicas de MI300X
4. **Reproducibilidad**: Documentar todos los experimentos y resultados

---

## Hitos Completados

| Hito | ID B3 | Estado | Fecha |
|------|-------|--------|-------|
| Layer Trace Root Cause | B3.5 | ✅ HECHO | 2026-02-03 |
| Decode Readout Analysis | B3.6 | ✅ HECHO | 2026-02-03 |
| Embedding Layout Fix | B3.8, B3.9 | ✅ HECHO | 2026-02-03 |
| LMHead Isolation | B3.13-B3.17 | ✅ HECHO | 2026-02-03 |
| Attention Pipeline | B3.20-B3.30 | ✅ HECHO | 2026-02-03 |
| QKV/Projection Fixes | B3.31-B3.36 | ✅ HECHO | 2026-02-03 |
| Full Pipeline Acceptance | B3.37 | ✅ HECHO | 2026-02-03 |
| FFN RMSNorm Root Cause | B3.42 | ✅ HECHO | 2026-02-03 |
| Embedding Audit | B3.59 | ✅ HECHO | 2026-02-05 |

---

## Enfoque Actual (B3.59)

**Tarea:** Auditoría Embedding/DebugInput  
**Estado:** ✅ COMPLETADO  
**Resultado:** No se encontró zeroing - coincidencia perfecta de hash entre prefill/decode

### Detalles Técnicos
- **Flags Usados:** `GRETA_TRACE_STAGE=1`, `GRETA_TRACE_STAGE_DEBUG_INPUT=1`
- **Problema:** Ambigüedad en el reporte de `embedding_out/x_in`
- **Solución:** Metadatos estandarizados de `StageTrace` (`token_id`, `route`)

---

## Componentes de Arquitectura Abordados

### 1. Capa de Embedding ✅
- Verificación de layout (B3.8)
- Fix row major (B3.9)
- Auditoría de debug input (B3.59)

### 2. Mecanismo de Attention ✅
- Aislamiento de proyección QKV (B3.31)
- Aislamiento de attention decode (B3.20)
- Aceptación de fix MFMA (B3.21)

### 3. LMHead ✅
- Aislamiento de route (B3.14)
- Verificación de weight layout (B3.15)
- Fix MFMA (B3.16)

### 4. Proyecciones de Salida ✅
- Layout de proyección WO (B3.40)
- Aislamiento de post-WO collapse (B3.41)

### 5. FFN/Normalización ✅
- FFN RMSNorm root cause (B3.42)
- V addressing long context (B3.26)

---

## Próximos Pasos

1. **B3.60**: TBD - Próxima tarea de optimización
2. **Benchmarking de Rendimiento**: Validación de rendimiento de pipeline completo
3. **Documentación**: Completar reporte técnico de Fase 3

---

## Artefactos Clave

| Artefacto | Ubicación | Reportes Asociados |
|-----------|----------|-------------------|
| Análisis B3.59 | `artifacts_remote/2026-02-05/b3_59/` | B3.59 |
| Análisis B3.42 | `artifacts_remote/2026-02-04/b3_58/` | B3.58 |
| Pipeline B3.37 | `artifacts_remote/2026-02-03/` | B3.37 |

---

## Documentación Relacionada

| Documento | Descripción |
|-----------|-------------|
| [Índice de Reportes AMD](./INDEX_ES.md) | Todos los reportes AMD de Fase 3 |
| [PROGRESS_ES.md](../PROGRESS_ES.md) | Seguimiento completo de progreso |
| [REPRODUCIBILITY_ES.md](../REPRODUCIBILITY_ES.md) | Cómo reproducir resultados |

---

## Especificaciones de Hardware

| Componente | Especificación |
|------------|---------------|
| GPU | AMD MI300X |
| Precisión | float16 |
| Backend | HIP |

---

*Mantenido por: Leandro Emanuel Timberini*  
*Última Actualización: 2026-02-06*
