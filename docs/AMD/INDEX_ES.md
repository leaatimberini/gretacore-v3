# Índice de Reportes AMD de GRETA CORE

**Repositorio:** gretacore  
**Rama:** `b3_59_embedding_debug_input_audit`  
**Última Actualización:** 2026-02-06  
**Total de Reportes:** 40

---

## Navegación Rápida

| Categoría | Reportes | Ubicación |
|-----------|----------|-----------|
| Todos los Reportes | 40 | [reports/](./reports/) |
| Phase 3 | Ver abajo | [phases/PHASE3_ES.md](./phases/PHASE3_ES.md) |
| Progreso | Ver abajo | [docs/PROGRESS_ES.md](../PROGRESS_ES.md) |

---

## Reportes por Fecha

### 2026-02-05 (Más Reciente)

| ID | Reporte | Estado | Artefactos |
|----|---------|--------|------------|
| B3.59 | Auditoría Embedding/DebugInput | ✅ PASÓ | [`2026-02-05/b3_59/`](../artifacts_remote/2026-02-05/b3_59/) |

### 2026-02-03 (Análisis Histórico)

| ID | Reporte | Estado | Área de Enfoque |
|----|---------|--------|-----------------|
| B3.5 | Layer Trace Root Cause | ✅ PASÓ | Embedding/Layer Trace |
| B3.6 | Decode Readout Landscape | ✅ PASÓ | Decode Readout Analysis |
| B3.6_rerun | Decode Readout Landscape (Rerun) | ✅ PASÓ | Decode Readout Verification |
| B3.7 | Analysis Decode Landscape | ✅ PASÓ | Decode Pipeline Analysis |
| B3.8 | Embedding Layout Verification | ✅ PASÓ | Embedding Layout |
| B3.9 | Embedding Row Major Fix | ✅ PASÓ | Embedding Data Format |
| B3.10 | Attractor Validation | ✅ PASÓ | Attractor Behavior |
| B3.11 | Readout Consistency Fix | ✅ PASÓ | Readout Consistency |
| B3.12 | Decode Readout Semantics | ✅ PASÓ | Decode Readout Semantics |
| B3.13 | Prefill/Decode Delta LMHead RMS | ✅ PASÓ | LMHead RMS Analysis |
| B3.14 | LMHead Force Route Isolation | ✅ PASÓ | LMHead Route Isolation |
| B3.15 | LMHead Weight Layout Verify | ✅ PASÓ | LMHead Weight Layout |
| B3.16 | LMHead MFMA Fix Acceptance | ✅ PASÓ | LMHead MFMA Fix |
| B3.17 | Decode LMHead Isolation | ✅ PASÓ | Decode LMHead |
| B3.18 | Prefill/Decode Hidden Equivalence | ✅ PASÓ | Hidden State Equivalence |
| B3.19 | Decode Collapse Fix Acceptance | ✅ PASÓ | Decode Collapse Fix |
| B3.20 | Attention Decode Isolation | ✅ PASÓ | Attention Decode Isolation |
| B3.21 | Attention Decode MFMA Fix | ✅ PASÓ | Attention MFMA Fix |
| B3.22 | Attention Precision Root Cause | ✅ PASÓ | Attention Precision |
| B3.23 | Softmax Isolation Decode0 | ✅ PASÓ | Softmax Decode0 |
| B3.24 | V Accumulation Isolation | ✅ PASÓ | V Accumulation |
| B3.25 | V Layout Fix Acceptance | ✅ PASÓ | V Layout Fix |
| B3.26 | V Addressing Long Context Fix | ✅ PASÓ | V Addressing Long Context |
| B3.27 | Decode0 Post-Attention Collapse | ✅ PASÓ | Decode0 Post-Attention |
| B3.28 | Decode0 Input Semantics Fix | ✅ PASÓ | Decode0 Input Semantics |
| B3.29 | Stage Trace Post-XIN Isolation | ✅ PASÓ | Stage Trace Post-XIN |
| B3.30 | Layer0 Attention Pipeline Root Cause | ✅ PASÓ | Layer0 Attention Pipeline |
| B3.31 | Decode QKV Projection Route | ✅ PASÓ | QKV Projection Route |
| B3.32 | Normout QKV Weight Probe | ✅ PASÓ | Normout QKV Weight |
| B3.33 | QKV Weight Layout Verification | ✅ PASÓ | QKV Weight Layout |
| B3.34 | Prefill WQ Layout Fix | ✅ PASÓ | Prefill WQ Layout |
| B3.35 | Prefill WK Layout Fix | ✅ PASÓ | Prefill WK Layout |
| B3.36 | Prefill WV Layout Fix | ✅ PASÓ | Prefill WV Layout |
| B3.37 | Full Pipeline Acceptance | ✅ PASÓ | Full Pipeline |
| B3.38 | Post-QKV Root Cause Isolation | ✅ PASÓ | Post-QKV Root Cause |
| B3.39 | WO vs Residual Add Isolation | ✅ PASÓ | WO vs Residual Add |
| B3.40 | WO Projection Layout Fix | ✅ PASÓ | WO Projection Layout |
| B3.41 | Post-WO Collapse Isolation | ✅ PASÓ | Post-WO Collapse |
| B3.42 | FFN RMSNorm Root Cause | ✅ PASÓ | FFN RMSNorm |

---

## Reportes por Categoría

### Documentos Centrales
| Documento | Versión EN | Descripción |
|-----------|------------|-------------|
| [phases/PHASE3_ES.md](./phases/PHASE3_ES.md) | [phases/PHASE3.md](./phases/PHASE3.md) | Resumen ejecutivo e hitos |
| [phases/AMD_TECH_REPORT_ES.md](./phases/AMD_TECH_REPORT_ES.md) | [phases/AMD_TECH_REPORT_EN.md](./phases/AMD_TECH_REPORT_EN.md) | Reporte técnico completo |
| [phases/BENCH_PROTOCOL_ES.md](./phases/BENCH_PROTOCOL_ES.md) | [phases/BENCH_PROTOCOL_EN.md](./phases/BENCH_PROTOCOL_EN.md) | Protocolo de benchmarking |

### Guías de Reproducción
| Documento | Versión EN | Descripción |
|-----------|------------|-------------|
| [phases/PHASE3_REPRO_GUIDE_ES.md](./phases/PHASE3_REPRO_GUIDE_ES.md) | [phases/PHASE3_REPRO_GUIDE_EN.md](./phases/PHASE3_REPRO_GUIDE_EN.md) | Reproducción paso a paso |
| [phases/REPRO_CHECKLIST_ES.md](./phases/REPRO_CHECKLIST_ES.md) | [phases/REPRO_CHECKLIST_EN.md](./phases/REPRO_CHECKLIST_EN.md) | Lista de verificación |

### Notas de Lanzamiento
| Documento | Versión EN | Descripción |
|-----------|------------|-------------|
| [phases/RELEASE_NOTES_v0.1_ES.md](./phases/RELEASE_NOTES_v0.1_ES.md) | [phases/RELEASE_NOTES_v0.1_EN.md](./phases/RELEASE_NOTES_v0.1_EN.md) | Notas de release v0.1 |

### Esquemas
| Documento | Descripción |
|-----------|-------------|
| [phases/bench_schema.csv](./phases/bench_schema.csv) | Definiciones de esquema de benchmark |

---

### Embedding
- B3.5, B3.8, B3.9, B3.59

### Attention
- B3.20, B3.21, B3.22, B3.23, B3.24, B3.25, B3.26, B3.30

### LMHead
- B3.13, B3.14, B3.15, B3.16, B3.17, B3.19

### QKV/Projection
- B3.31, B3.32, B3.33, B3.34, B3.35, B3.36, B3.38, B3.39, B3.40, B3.41

### FFN/Normalization
- B3.42

### Readout/Decode
- B3.6, B3.6_rerun, B3.7, B3.11, B3.12, B3.18, B3.27, B3.28, B3.29

### Full Pipeline
- B3.10, B3.37

---

## Referencia de Artefactos

| Ubicación de Artefacto | Reportes Asociados | Estado |
|------------------------|-------------------|--------|
| `artifacts_remote/2026-02-03/` | B3.5 - B3.42 | ✅ Disponible |
| `artifacts_remote/2026-02-04/` | B3.53 - B3.59 | ✅ Disponible |
| `artifacts_remote/2026-02-05/` | B3.59 | ✅ Disponible |

---

## Documentación Relacionada

| Documento | Descripción |
|-----------|-------------|
| [PROGRESS_ES.md](../PROGRESS_ES.md) | Índice completo de progreso con detalles técnicos |
| [ROADMAP_ES.md](../ROADMAP_ES.md) | Hoja de ruta del proyecto |
| [DEBUGGING_ES.md](../DEBUGGING_ES.md) | Guía de depuración |
| [REPRODUCIBILITY_ES.md](../REPRODUCIBILITY_ES.md) | Cómo reproducir resultados AMD |

---

*Mantenido por: Leandro Emanuel Timberini*  
*Hardware: AMD MI300X*
