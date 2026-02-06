# GRETA CORE - Reporte Técnico de la Fase 3 (AMD MI300X)

## Resumen Ejecutivo
GRETA CORE es un motor de inferencia de LLM de alto rendimiento diseñado desde cero para arquitecturas AMD. La Fase 3 marca el éxito de la portabilidad del pipeline completo de Llama-2-7B al AMD Instinct™ MI300X, utilizando kernels especializados y aceleración mediante HIP Graphs para minimizar el overhead del host.

## Hitos Técnicos

### 1. Estabilidad Numérica (Fase 3.1)
- **Garantía Zero-NaN**: Validación automatizada mediante tracer en las 32 capas del transformer.
- **Auditabilidad**: Estadísticas por bloque (Mín/Máx/Media) para todos los tensores internos.

### 2. Kernels Especializados (Fase 3.2)
- **lm_head optimizado para Decode**: Sustitución de GEMM estándar por un kernel GEMV wave-parallel para vocabularios de 32k+, logrando una mejor utilización de HBM3 en inferencia de bajo batch.
- **Argmax en GPU**: Eliminación de la sincronización CPU-GPU para el decode greedy.

### 3. Aceleración mediante HIP Graphs
- **Overhead de Programación Zero**: El loop central de decode se captura en un `hipGraph_t`, reduciendo el T3D (Time to Dispatch) a casi cero.
- **Dependencias Estáticas**: Todos los kernels de atención (RoPE, FlashAttention) refactorizados para compatibilidad con grafos estáticos.

## Análisis de Rendimiento
**Hardware**: AMD Instinct™ MI300X (Single GPU)
**Modelo**: Llama-2-7B (Q4_K_M)

| Métrica | Valor Medido |
| :--- | :--- |
| **Rendimiento (Decode)** | 13.01 tokens/s |
| **Latencia por capa** | ~2.1 ms |
| **TTFT (3 tokens)** | 221 ms |

**Bottlenecks Actuales**:
La ejecución está limitada actualmente por instrucciones en RMSNorm no fusionado y patrones de memoria subóptimos en la actualización de la KV Cache.

## Hoja de Ruta: Fase 4 (Fusión Completa de Kernels)
Para alcanzar los picos teóricos del MI300X (~5.3 TB/s HBM3), la Fase 4 implementará:
- **Bloques fusionados de atención-norma**.
- **Tiling en memoria compartida** para dequantización + GEMM.
- **Pre-fetching asíncrono de KV-Cache**.

## Conclusión
GRETA CORE demuestra que las arquitecturas especializadas y agnósticas al host (HIP Graphs) proporcionan la base para una operación de LLM hiper-eficiente en CDNA 3. El motor está listo para el escalado de rendimiento a nivel de producción.
