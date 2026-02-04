# GRETA CORE – Arquitectura

Versión: 1.1  
Estado: Activo  
Fase del Proyecto: Fase 3 – Pipeline de Inferencia LLM (activo)  
Idioma: Español

---

## Propósito de este Documento

Este documento define la arquitectura técnica de GRETA CORE.
Establece los límites del sistema, los componentes centrales,
sus responsabilidades y las reglas de interacción entre capas.

El objetivo de esta arquitectura no es la flexibilidad para todos
los casos de uso, sino el máximo control, rendimiento y
mantenibilidad a largo plazo para cargas de trabajo de LLMs en
hardware AMD.

---

## Estado Actual (B3.x)

- El pipeline de inferencia LLM está en fase de corrección y trazas profundas.
- Se aisló el LM head (rutas MFMA/VALU) y se agregó instrumentación de decode.
- El colapso en decode persiste; la investigación se centra en atención/KV y estado.
- B3.20 introduce verificación de attention decode con referencia y trazas de invariantes KV. Resultado: `attn_out` diverge del ref en layer 31 mientras KV invariants se mantiene; `fused+mfma` falla en load.
- B3.21 estabiliza `fused+mfma` (fix de Hkv + guard rails de alignment). MFMA==VALU en decode0, pero persiste divergencia vs ref en layer 31 y el colapso decode0.
- B3.22 audita precisión en capas altas; divergencia vs referencia FP64 persiste en layer 31 independiente del modo de acumulación.
- B3.23 aísla softmax en decode0 (layer 31 head 0). QK/softmax coinciden con FP64; el foco pasa a acumulado de V / attn_out.
- Validación MI300X en curso con evidencia en `docs/AMD/`.

---

## Visión General de la Arquitectura

GRETA CORE está diseñado como un sistema **estratificado pero no
permeable**, donde cada capa tiene una responsabilidad estricta
y limitada.

La arquitectura es deliberadamente mínima y vertical, priorizando
eficiencia de ejecución por sobre generalidad de abstracción.

Capas de alto nivel:

1. Capa de Plataforma
2. Capa de Runtime
3. Capa de Kernels
4. Capa de Compilador y Autotuning
5. Capa de Integración
6. Capa de Tooling

Cada capa solo puede depender de la capa inmediatamente inferior.

---

## 1. Capa de Plataforma

### Responsabilidad
Proveer acceso al hardware AMD mediante interfaces del sistema
estables y bien definidas.

### Componentes
- Kernel Linux (driver amdgpu)
- ROCm/HIP y/o Vulkan Compute (RADV)
- Toolchain LLVM (cuando sea necesario)

### Reglas de Diseño
- No manipulación directa de hardware fuera de drivers soportados
- Comportamientos específicos deben aislarse
- La capa de plataforma es reemplazable, no embebida

### Fuera de Alcance
- Drivers personalizados
- Modificación de firmware
- Hackeo de drivers propietarios

---

## 2. Capa de Runtime (gcore-rt)

### Responsabilidad
Controlar ejecución, memoria y sincronización con mínimo overhead.

### Conceptos Centrales
- Streams
- Eventos
- Pools de memoria
- Planificación determinista
- Hooks de telemetría

### Responsabilidades
- Asignación y reutilización de memoria
- Orquestación de lanzamientos de kernels
- Orden de ejecución
- Medición de tiempo y contadores

### Reglas de Diseño
- El runtime no contiene lógica de kernels
- Sin dependencias de frameworks
- El runtime debe poder usarse de forma autónoma

---

## 3. Capa de Kernels (gcore-kernels)

### Responsabilidad
Implementar todo el cómputo crítico de rendimiento.

### Categorías de Kernels
- Álgebra lineal (GEMM)
- Normalización (LayerNorm, RMSNorm)
- Softmax y reducciones
- Primitivas de atención
- Operaciones de KV-cache

### Reglas de Diseño
- Corrección antes que optimización
- Patrones explícitos de acceso a memoria
- Priorizar fusión para reducir tráfico
- Tuning consciente del hardware

### No Objetivos
- Cobertura completa de operadores
- Kernels de entrenamiento
- Primitivas no relacionadas con LLMs

---

## 4. Capa de Compilador y Autotuning

### Responsabilidad
Generar y optimizar kernels para arquitecturas AMD específicas.

### Componentes
- Exploración de parámetros
- Base de datos de autotuning
- DSL opcional (tipo MLIR / Triton)

### Reglas de Diseño
- Rendimiento empírico sobre heurísticas estáticas
- Autotuning reproducible
- Kernels generados como artefactos de primera clase

### No Objetivos
- Compilador de propósito general
- Compilación automática de grafos de modelos (inicialmente)

---

## 5. Capa de Integración (gcore-bridge)

### Responsabilidad
Exponer funcionalidades de GRETA CORE a sistemas externos.

### Objetivos
- Runner de inferencia autónomo mínimo
- Execution Provider opcional para ONNX Runtime
- Puente Triton (target AMD)
- Puentes PyTorch y JAX
- Capa de compatibilidad cuDNN/TensorRT vía equivalentes AMD

### Reglas de Diseño
- Integración opcional, nunca obligatoria
- Ninguna lógica de framework se filtra al runtime o kernels
- Los bridges deben ser delgados y reemplazables
- Sin dependencias de binarios NVIDIA; paridad vía equivalentes AMD

---

## 6. Capa de Tooling (gcore-tools)

### Responsabilidad
Brindar visibilidad, diagnóstico y control a los desarrolladores.

### Herramientas
- Profiler (líneas de tiempo, contadores)
- Reportes de rendimiento de kernels
- Visualización de autotuning
- Detección de regresiones

### Reglas de Diseño
- El tooling no debe afectar el rendimiento del runtime
- Uso opcional
- Las métricas son la fuente de verdad

---

## Flujo de Datos

Flujo típico de inferencia:

1. Carga de pesos del modelo
2. Inicialización de pools de memoria
3. Creación de streams
4. Planificación de kernels vía runtime
5. Ejecución en hardware
6. Recolección de telemetría
7. Retorno de resultados

Ninguna capa salta a otra capa.

---

## Invariantes Arquitectónicos

Los siguientes invariantes deben cumplirse siempre:

- Los kernels no dependen de frameworks
- El runtime no contiene lógica de modelo
- La integración no dicta diseño de kernels
- El tooling no altera la ejecución
- Las regresiones de rendimiento no se aceptan sin justificación

---

## Evolución a Largo Plazo

La arquitectura está diseñada para evolucionar sin romper invariantes:

- Nuevos kernels extienden la capa de kernels
- Nuevos backends extienden la capa de plataforma
- Nuevas integraciones extienden la capa de bridges

Los principios centrales permanecen inmutables.

---

## Autoría

GRETA CORE fue concebido, fundado y es liderado por:

Leandro Emanuel Timberini  
Fundador y Arquitecto Principal de Sistemas
