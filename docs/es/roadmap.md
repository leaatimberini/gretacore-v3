# GRETA CORE – Hoja de Ruta

Versión: 1.0  
Estado: Fundacional  
Fase del Proyecto: Fase 3 – Pipeline de Inferencia LLM (activo)  
Idioma: Español

---

## Propósito de esta Hoja de Ruta

Esta hoja de ruta define el plan de ejecución a largo plazo de GRETA
CORE. Establece fases claras, objetivos, entregables y criterios de
éxito con el fin de mantener foco técnico y coherencia arquitectónica
a lo largo de un proyecto de varios años.

GRETA CORE no está optimizado para velocidad de entrega.
Está optimizado para corrección, rendimiento y sostenibilidad.

---

## Estado Actual (2026-02-04)

- Fase 1 (Runtime Core): **completada**.
- Fase 2 (Dominio de Kernels): **completada** (13 TFLOPS GEMM, 2.1ms Llama Block).
- Fase 3 (Pipeline de Inferencia LLM): **activa** (B3.x).
- B3.14–B3.16: aislación de LM head; MFMA deshabilitado en LM head; prefill coherente con VALU.
- B3.17–B3.18: trazas de decode (LM head force route, hidden equivalence, layer delta).
- B3.19: fix de `seq_len = pos + 1` en attention decode no resolvió el colapso.
- B3.20: aislamiento de attention decode (attn verify/ref, invariantes KV, matriz de rutas). KV invariants OK; divergencia attn_out vs ref en layer 31; `fused+mfma` falla en load.
- B3.21: `fused+mfma` estabilizado (fix de Hkv + guard rails de alignment). MFMA==VALU en decode0, pero persiste la divergencia vs ref en layer 31 y el colapso decode0.
- B3.22: auditoría de precisión en capas altas; divergencia vs referencia FP64 persiste en layer 31 independiente del modo de acumulación.
- B3.23: aislamiento de softmax en decode0 (layer 31 head 0). QK y softmax coinciden con FP64; el foco pasa a acumulado de V / attn_out.
- B3.24–B3.26: V layout/addressing en decode corregido; P·V consistente; persiste colapso en decode0.
- B3.27–B3.29: stage trace post-x_in; primer mismatch localizado en `attn_out` (layer 0).
- B3.30–B3.32: aislamiento en layer0; `attn_norm_out` ok; primer mismatch en Q; ruta no depende de MFMA/VALU.
- B3.33: verificación layout Wq: prefill_last=col, decode0=row en prompts con contexto.
- B3.34: corregir layout de Wq en prefill (GEMM) para alinear con decode (row).
- B3.35: corregir layout de Wk en prefill (GEMM) para alinear con decode (row).
- B3.36: corregir layout de Wv en prefill (GEMM) para alinear con decode (row).
- Validación MI300X en curso; evidencia en `docs/AMD/`.

## Fase 0 – Fundaciones

### Objetivo
Establecer las bases intelectuales, arquitectónicas y operativas del
proyecto antes de escribir código de producción.

### Alcance
- Documentación
- Definición de arquitectura
- Diseño de benchmarks
- Preparación del entorno

### Entregables
- Whitepaper (EN / ES)
- README (EN / ES)
- Roadmap (EN / ES)
- Principios arquitectónicos
- Especificación de benchmarks
- Estructura del repositorio

### Criterios de Éxito
- Visión y alcance claramente definidos
- Benchmarks reproducibles
- Ausencia de código prematuro

### Fuera de Alcance
- Optimización de kernels
- Implementación de runtime
- Integración con frameworks

---

## Fase 1 – Núcleo del Runtime

### Objetivo
Construir un runtime de cómputo mínimo y determinista capaz de
gestionar memoria, streams de ejecución y lanzamientos de kernels con
bajo overhead.

### Alcance
- Runtime personalizado (gcore-rt)
- Pooling y reutilización de memoria
- Abstracción de streams y eventos
- Telemetría y medición de tiempo

### Entregables
- Allocator de memoria (por pools)
- API de streams/eventos
- Wrapper de lanzamiento de kernels
- Microbenchmarks del runtime

### Criterios de Éxito
- Runtime estable bajo stress
- Comportamiento determinista
- Overhead medible inferior a runtimes genéricos

### Fuera de Alcance
- Optimización avanzada de kernels
- Pipelines de LLM
- Automatización de compilador

---

## Fase 2 – Dominio de Kernels (Primitivas LLM)

### Objetivo
Lograr implementaciones de alto rendimiento de los kernels esenciales
para inferencia de LLMs.

### Alcance
- Desarrollo kernel-first
- Reducción del tráfico de memoria
- Tuning consciente del hardware

### Entregables
- GEMM (FP16 / BF16)
- LayerNorm / RMSNorm
- Softmax
- Operaciones de KV-cache
- Fusión inicial de kernels

### Criterios de Éxito
- Corrección validada contra referencias
- Rendimiento competitivo en hardware AMD objetivo
- Estabilidad en ejecuciones prolongadas

### Fuera de Alcance
- Ejecución completa de modelos
- Entrenamiento o backpropagation
- Herramientas avanzadas para desarrolladores

---

## Fase 3 – Pipeline de Inferencia LLM

### Objetivo
Permitir inferencia LLM end-to-end utilizando componentes de GRETA
CORE.

### Alcance
- Runtime de inferencia mínimo
- Planificación de operadores
- Gestión del ciclo de vida de memoria

### Entregables
- Ejecución de bloques transformer
- Ruta de inferencia FP16 y/o cuantizada
- Benchmarks end-to-end (tokens/s, latencia)

### Criterios de Éxito
- Ejecución exitosa de al menos un LLM objetivo
- Métricas estables de throughput y latencia
- Independencia de stacks CUDA externos

### Fuera de Alcance
- Inferencia distribuida
- Ejecución multi-GPU
- Workloads de entrenamiento

---

## Fase 4 – Experiencia de Desarrollo y Tooling

### Objetivo
Mejorar la usabilidad sin sacrificar rendimiento ni control.

### Alcance
- Profiling
- Debugging
- Flujos de autotuning

### Entregables
- Profiler integrado
- Visualización de rendimiento de kernels
- Base de datos de autotuning
- Documentación para desarrolladores

### Criterios de Éxito
- Visibilidad clara de cuellos de botella
- Resultados de tuning reproducibles
- Reducción del tiempo de iteración

---

## Fase 5 – Expansión del Ecosistema

### Objetivo
Expandir GRETA CORE más allá de sus objetivos iniciales preservando la
integridad arquitectónica.

### Alcance
- Operadores adicionales
- Bridges con frameworks
- Soporte de hardware ampliado

### Entregables
- Capa de compatibilidad de frameworks (Triton / PyTorch / JAX)
- Paridad cuDNN/TensorRT vía equivalentes AMD
- Biblioteca de kernels extendida
- Herramientas orientadas a despliegue

### Criterios de Éxito
- Crecimiento mantenible sin degradación arquitectónica
- Adopción medible por usuarios externos
- Competitividad sostenida de rendimiento
- Mismo código y pasos de instalación entre Radeon dev y MI300X cloud

---

## Principios de Gobernanza

- Las decisiones arquitectónicas permanecen centralizadas
- Las métricas de rendimiento prevalecen sobre preferencias de abstracción
- El minimalismo se aplica en todas las capas
- Cada dependencia debe justificar su existencia

---

## Autoría

GRETA CORE fue concebido, fundado y es liderado por:

Leandro Emanuel Timberini  
Fundador y Arquitecto Principal de Sistemas
