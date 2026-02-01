# Análisis Competitivo: Engines de Inferencia LLM

## Estado Actual (2026-01-31)

| Engine | Hardware | Tokens/s (Batch=1) | Notas |
|:---|:---|:---|:---|
| **GRETA CORE** | AMD MI300X | ~95* | Fase 3 inicial, kernels custom, modo demo |
| vLLM | NVIDIA H100 | ~180-220 | PagedAttention, kernels optimizados |
| TensorRT-LLM | NVIDIA H100 | ~250-300+ | Máxima optimización NVIDIA (FP8/FP16) |
| vLLM | AMD MI300X | ~200-240 | ROCm + kernels estándar |

> **Nota:** El valor de 95 tok/s es del modo demo (loop sin cómputo real). El throughput real con forward pass completo será menor inicialmente.

## Análisis de la Brecha

### Ventajas de la Competencia
- **vLLM:** PagedAttention para gestión eficiente de memoria, años de optimización.
- **TensorRT-LLM:** Acceso a librerías propietarias NVIDIA (cuBLAS, cuDNN), quantización FP8.
- **Tiempo de desarrollo:** Equipos grandes con múltiples años de trabajo.

### Ventajas de GRETA CORE
- **Control total del stack:** Sin dependencias opacas, todo el código es visible y modificable.
- **Optimización para hardware específico:** Directamente CDNA3 (gfx942), no abstracciones genéricas.
- **Arquitectura limpia:** Mantenibilidad a largo plazo, no hacks de rendimiento acumulados.
- **Ownership completo:** Sin licencias restrictivas ni dependencias corporativas.

## Roadmap de Optimización

### Fase 4A: Optimización de Memoria (Target: +50% throughput)
| Optimización | Impacto Estimado | Prioridad |
|:---|:---|:---|
| FP16/BF16 puro (reemplazar FP32) | +2x FLOPS teóricos | Alta |
| Double buffering en GEMM | +30-50% utilización | Alta |
| Fusión GEMM+RMSNorm+SiLU | -30% overhead de memoria | Media |

### Fase 4B: Arquitectura de Atención (Target: +100% throughput)
| Optimización | Impacto Estimado | Prioridad |
|:---|:---|:---|
| FlashAttention v2 (CDNA3) | +2-3x en kernel de atención | Crítica |
| PagedAttention | Batching eficiente, menos fragmentación | Alta |
| Attention tiling para contexto largo | Soporte 8K+ tokens | Media |

### Fase 4C: Quantización (Target: +200% throughput)
| Optimización | Impacto Estimado | Prioridad |
|:---|:---|:---|
| INT8 GEMM (Matrix Cores) | +2x throughput | Alta |
| FP8 (si disponible en CDNA3) | +2-4x throughput | Alta |
| GPTQ/AWQ support | Compatibilidad con modelos cuantizados | Media |

## Objetivo a Largo Plazo
**Meta: 200+ tokens/segundo en MI300X con Llama-2-7B (Batch=1)**

Este objetivo nos pondría al nivel de vLLM en hardware AMD, validando la viabilidad de un engine de inferencia nativo optimizado.

---
*Documento de estrategia - GRETA CORE Team*
