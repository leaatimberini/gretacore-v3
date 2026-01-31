# GRETA CORE – Benchmarks

Versión: 1.0  
Estado: Fundacional  
Fase del Proyecto: Fase 1 – Núcleo del Runtime  
Idioma: Español

---

## Propósito de este Documento

Este documento define la metodología oficial de benchmarking de
GRETA CORE.

Los benchmarks son el mecanismo autoritativo para:
- validar corrección
- medir rendimiento
- detectar regresiones
- guiar decisiones arquitectónicas

Ninguna optimización, refactor o funcionalidad se considera válida
sin evidencia de benchmark.

---

## Principios de Benchmarking

Todos los benchmarks de GRETA CORE deben cumplir:

1. Reproducibilidad
2. Determinismo
3. Minimalismo
4. Aislamiento
5. Comparabilidad en el tiempo

Los benchmarks que no cumplan estos criterios son inválidos.

---

## Categorías de Benchmarks

Los benchmarks de GRETA CORE se dividen en cuatro categorías:

1. Benchmarks de Plataforma
2. Benchmarks de Runtime
3. Benchmarks de Kernels
4. Benchmarks End-to-End

Cada categoría cumple un propósito específico y no debe mezclarse.

---

## 1. Benchmarks de Plataforma

### Objetivo
Medir los límites crudos del hardware y del sistema, independientemente
de GRETA CORE.

### Métricas
- Ancho de banda de memoria (GB/s)
- Latencia de memoria
- Latencia de lanzamiento de kernels
- Costo de sincronización CPU–GPU

### Ejemplos
- Lectura/escritura secuencial de memoria
- Accesos aleatorios
- Lanzamiento de kernel vacío

### Propósito
Establecer cotas superiores e identificar cuellos no atribuibles al
software.

---

## 2. Benchmarks de Runtime

### Objetivo
Medir el overhead introducido por el runtime de GRETA CORE.

### Métricas
- Latencia de asignación
- Eficiencia de reutilización de memoria
- Overhead de planificación de streams
- Precisión de eventos y temporización

### Ejemplos
- Ciclos allocate/free
- Sincronización de streams
- Dispatch de kernel vacío vía runtime

### Propósito
Garantizar que el overhead del runtime sea mínimo y predecible.

---

## 3. Benchmarks de Kernels

### Objetivo
Medir rendimiento y corrección de kernels individuales.

### Métricas
- Throughput (GFLOPs / tokens/s cuando aplique)
- Latencia (media, p50, p99)
- Tráfico de memoria
- Corrección numérica

### Clases de Kernels
- GEMM
- Normalización (LayerNorm, RMSNorm)
- Softmax
- Reducciones
- Operaciones de KV-cache

### Reglas
- Un kernel por benchmark
- Tamaños de entrada fijos
- Salida verificada contra referencia

---

## 4. Benchmarks End-to-End

### Objetivo
Medir rendimiento real de rutas de ejecución compuestas.

### Métricas
- Tokens por segundo
- Latencia end-to-end
- Uso de memoria
- Estabilidad en el tiempo

### Ejemplos
- Ejecución de bloque transformer
- Loop de inferencia LLM mínimo
- Prueba de crecimiento de KV-cache

### Propósito
Validar que las mejoras se traduzcan en ganancias reales.

---

## Requisitos del Entorno de Benchmark

Todo benchmark debe registrar:

- Configuración de hardware
- Versiones de drivers y runtime
- Versiones de compilador
- Configuración de energía y clocks (si aplica)

Benchmarks sin metadatos de entorno son inválidos.

---

## Reglas de Ejecución

- El sistema debe estar en estado estable
- Evitar throttling térmico o documentarlo
- Múltiples corridas para analizar variancia
- Resultados versionados

---

## Política de Regresiones

Cualquier cambio que cause:
- ≥5% de regresión de rendimiento
- incremento de variancia
- mayor uso de memoria

debe ser marcado y revisado.

Las regresiones se tratan como bugs.

---

## Formato de Salida

Los resultados deben incluir:

- Mediciones crudas
- Estadísticas agregadas
- Metadatos del entorno
- Timestamp

Se requieren formatos legibles por humanos y máquinas.

---

## Autoría

GRETA CORE fue concebido, fundado y es liderado por:

Leandro Emanuel Timberini  
Fundador y Arquitecto Principal de Sistemas
