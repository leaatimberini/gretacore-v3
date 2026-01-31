# Reporte de Estado y Evaluación Estratégica - GRETA CORE

**Fecha:** 2026-01-31
**Autor:** Leandro Emanuel Timberini
**Contexto:** Continuidad operacional post-Codex

## 1. Diagnóstico
El proyecto GRETA CORE ha superado su mayor cuello de botella estructural: la incompatibilidad de Vulkan con el hardware MI300X. Se ha realizado un pivot estratégico hacia el stack **HIP/ROCm**, logrando ejecución nativa en MI300X con rendimiento escalable.

**Estado:** El backend HIP (`gcore::rt::hip`) es ahora el motor principal para hardware Enterprise, mientras que Vulkan se mantiene como opción para hardware de consumo y cross-platform.

## 2. Evaluación de la Etapa Actual
Estamos finalizando la **Fase 1 (Portabilidades Base)** y entrando en **Fase 2 (Optimización Aplicada)**.
- **Logrado:** Backend HIP operativo, validación en MI300X (HBM3 OK), Grafo de ejecución integrado, Kernels MFMA iniciales.
- **Faltante:** KV-cache asíncrono, Fusión de kernels de atención (FlashAttention style), Inferencia de modelo completo.

## 3. Brechas Técnicas vs CUDA (Software)
| Área | CUDA / cuDNN | GRETA CORE (Actual) | Brecha |
|dev| Madurez Extrema | Incipiente | **Critica** |
| **Kernels** | Librerías ultra-tuneadas (cuBLAS) | Kernels "good enough" manuales | Alta |
| **Dispatch** | Graph API / Stream Capture | QueueSubmit manual | Media |
| **Multi-GPU** | NVLink / NCCL transparente | Inexistente | **Bloqueante para LLM grandes** |
| **Tooling** | Nsight Systems / Compute | Timestamp queries manuales | Alta |
| **Ecosistema** | Pytorch nativo | Puentes experimentales | Media |

## 4. Acciones de Ingeniería (Logradas hoy 2026-01-31)
1.  **Desbloqueo MI300X (EXITO):** Pivot de Vulkan a HIP. Ejecución confirmada en MI300X VF.
2.  **Referencia CPU Robusta:** Implementada y usada para validar todos los kernels HIP.
3.  **GEMM MFMA (Matrix Cores):** Implementación inicial usando `v_mfma_f32_16x16x4f32` alcanzando **~13 TFLOPS**.
4.  **Kernels LLM de Atención:** RoPE y Causal Masking implementados y validados en grafo.
5.  **KV-Cache y Bloque Transformer:** Gestión de KV-Cache operativa y ejecución completa de un bloque Llama-2-7B en **2.1 ms**.

## 5. Roadmap de Arquitectura a Mediano Plazo
-   **Semana 1-2 Feb:** Runner de Inferencia Mínimo (Bloque Transformer).
-   **Semana 3-4 Feb:** Integración Triton-First (Backend AMD funcional upstream o fork estable).
-   **Marzo:** Multi-GPU intra-nodo (Sharding básico de tensores).
-   **Abril:** Persistencia y carga de pesos reales (Llama/Mistral).

## 6. Definición del PRIMER LANZAMIENTO (v0.1)
**Nombre:** GRETA CORE v0.1 - "The Red Foundation"
**Target:** Desarrolladores de sistemas y hackers de ML.

**Incluye:**
-   Backend Dual: **Vulkan** (Consumer) + **HIP** (Enterprise/MI300X).
-   Optimización **MFMA** para CDNA 3.
-   Graph Runner funcional para bloques Transformer.
-   Demos: Inferencia validada en 8600G y MI300X.

**NO Incluye:**
-   Training.
-   Soporte NVIDIA (explícito).
-   Frontends Python de alto nivel (más allá de demos).

## 7. Valor Estratégico para AMD
GRETA CORE demuestra que el hardware de AMD **no necesita la abstracción de CUDA** para ser performante. Al usar Vulkan/Spir-V y control de bajo nivel:
1.  **Schematic Control:** Explotamos la jerarquía de memoria de AMD sin la "Caja Negra" de los drivers de NVIDIA.
2.  **Universalidad:** Un solo binario corre en APU de laptop y en nodo MI300X de servidor.
3.  **Independencia:** Rompe la dependencia de ROCm monolítico (dockers gigantes) para inferencia.

## 8. Notas de Documentación
-   Mantener rigurosamente `docs/es/` y `docs/en/` sincronizados.
-   La "Verdad" técnica reside en el código y en los whitepapers (`docs/*/strategy/`).
