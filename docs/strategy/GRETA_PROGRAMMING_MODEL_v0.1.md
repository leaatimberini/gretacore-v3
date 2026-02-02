# GRETA Programming Model v0.1

**Strategic Objective**: To provide a CUDA-like development experience and runtime performance on AMD (CDNA 3) hardware, enabling seamless migration for AI workloads.

## 1. Core Philosophy
The GRETA Programming Model (GPM) acts as a high-level contract between the AI model definition and the hardware-specific execution. 

- **Abstraction over Specialization**: The developer defines *Intents* (e.g., "Fused Attention Matrix-Vector product"), and the GRETA Runtime selects the optimal execution path (MFMA mixed-precision, VALU, or Fused-VALU).
- **CUDA Syntax/Mental Model**: Memory management, streams (Graphs), and synchronization follow the CUDA mental model.
- **MI300X as a First-Class Citizen**: Leverage the MI300X unified memory architecture and HBM3 bandwidth without exposing non-standard primitives.

## 2. API Surface Levels

### L0: GRETA Runtime (GRT)
The system-level abstraction of HIP/Graphs.
- `GretaGraph`: Equivalent to `cudaGraph`. Captures multiple kernel launches into a single executable unit.
- `GretaStream`: Equivalent to `cudaStream`.
- `GretaMemory`: Resource-tracked allocations with automatic alignment.

### L1: GRETA Compute Core (GCC)
The replacement for cuBLAS/cuDNN.
- `GretaGEMM`: Standard matrix multiplication, auto-selecting MFMA layouts.
- `GretaAttention`: Optimized flash-attention kernels (Decode/Prefill).
- `GretaNorm`: Fused normalization kernels (RMSNorm, LayerNorm).

### L2: GRETA Fused Layer (GFL)
The "Platform Secret Sauce."
- Logic that combines GEMM epilogues (Act, Bias, Scaling) into single-pass kernels to bypass memory bandwidth bottlenecks.

## 3. Performance & Stability Baseline
- **Execution Target**: MI300X (gfx942).
- **Performance Baseline**: NVIDIA H100 (Hoppper) + CUDA 12.x.
- **Reproducibility**: Binaries and kernels must produce bit-perfect results across runs. Floating point non-determinism in atomic reductions must be minimized.

---

# Modelo de Programación GRETA v0.1 (ES)

**Objetivo Estratégico**: Proporcionar una experiencia de desarrollo similar a CUDA y un rendimiento de runtime óptimo en hardware AMD (CDNA 3), permitiendo la migración fluida de cargas de trabajo de IA.

## 1. Filosofía Central
El Modelo de Programación GRETA (GPM) actúa como un contrato de alto nivel entre la definición del modelo de IA y la ejecución específica en hardware.

- **Abstracción sobre Especialización**: El desarrollador define *Intenciones* (ej: "Producto Matriz-Vector de Atención Fusionada"), y el Runtime de GRETA selecciona la ruta de ejecución óptima (MFMA precisión mixta, VALU, o VALU Fusionado).
- **Sintaxis y Modelo Mental de CUDA**: La gestión de memoria, streams (Graphs) y sincronización siguen el modelo mental de CUDA.
- **MI300X como Ciudadano de Primera Clase**: Aprovecha la arquitectura de memoria unificada de MI300X y el ancho de banda HBM3 sin exponer primitivas no estándares.

## 2. Superficie de la API

### L0: Runtime de GRETA (GRT)
- `GretaGraph`: Equivalente a `cudaGraph`.
- `GretaStream`: Equivalente a `cudaStream`.
- `GretaMemory`: Asignaciones con seguimiento de recursos y alineación automática.

### L1: Núcleo de Cómputo GRETA (GCC)
Reemplazo para cuBLAS/cuDNN.
- `GretaGEMM`: Multiplicación de matrices estándar con selección automática de layouts MFMA.
- `GretaAttention`: Kernels de atención optimizados (Decode/Prefill).

### L2: Capa Fusionada GRETA (GFL)
Kernels que combinan epílogos de GEMM (Activación, Bias, Escala) para evitar cuellos de botella en el ancho de banda de memoria.
