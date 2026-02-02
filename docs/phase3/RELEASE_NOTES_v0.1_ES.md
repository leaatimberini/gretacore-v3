# GRETA CORE v0.1 - Notas de Lanzamiento de la Fase 3

## Resumen
La versión 0.1 marca el primer lanzamiento estable de GRETA CORE para aceleradores AMD Instinct™ MI300X. Este lanzamiento se centra en la corrección de la inferencia, la eliminación del overhead del host mediante HIP Graphs y el rendimiento especializado de las capas de decode.

## Características Clave

### 1. Port de CDNA 3 de Alto Rendimiento
- Soporte completo para el forward pass de Llama-2-7B en MI300X.
- Kernel especializado **`lm_head_gemv`** para un decode optimizado de vocabularios grandes.
- **Argmax Zero-Copy**: Selección de categoría en el lado de la GPU con bypass del host.

### 2. Inferencia Autónoma (HIP Graphs)
- Implementación nativa de captura `hipGraph_t` para stacks de 32 capas.
- Eliminación de la latencia de programación del driver host durante el loop autoregresivo.
- Kernels de RoPE y Atención basados en punteros para la estabilidad de grafos estáticos.

### 3. Auditabilidad Técnica (Tracer)
- Monitoreo de estabilidad numérica en tiempo real (`GRETA_TRACE_LEVEL`).
- Perfilado de alta resolución a nivel de bloque (`GRETA_PROFILE_BLOCKS`).

## Objetivos de Rendimiento
- **Rendimiento Promedio**: 13+ tokens/s en MI300X (Q4_K).
- **Corrección**: 100% de paridad con los resultados de referencia de GGUF (0 NaNs).

## Instalación
Consulte la [PHASE3_REPRO_GUIDE_ES.md](./PHASE3_REPRO_GUIDE_ES.md) para obtener instrucciones detalladas de compilación.
