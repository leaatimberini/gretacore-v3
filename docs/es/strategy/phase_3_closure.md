# Plan de Cierre Técnico: Fase 3 (MI300X)
**Autor:** Leandro Emanuel Timberini

## Resumen del Estado Técnico
GRETA CORE ha alcanzado estabilidad funcional y optimización de base para Llama-2-7B en AMD MI300X (gfx942).

### KPIs Validados
- **Correctitud:** 0 NaNs detectados. Logits estables.
- **Dequantización:** Paridad exacta para Q4_K y Q6_K.
- **Rendimiento (Decode):** **10.41 tokens/s** (Especialización Warp-only GEMV).
- **Rendimiento (Prefill):** **19.3 tokens/s** (Camino mixto MFMA/GEMV).

## Hoja de Ruta: Fase 3.1 - 3.3

### Fase 3.1: Hardening y Validación Profunda
*Objetivo: Pasar de "funciona" a "es auditable y confiable".*

1.  **Validador de Paridad:** Comparación automática entre GRETA y la referencia CPU/FP32.
2.  **Instrumentación Granular:** Timers de alta resolución (HIP) por bloque lógico:
    -   Proyección QKV (Latencia/Throughput).
    -   Self-Attention y KV-Update.
    -   FFN (cadena W1, W3, W2).
    -   lm_head (latencia con vocabulario grande).
3.  **Reporte de Estabilidad Numérica:** Documentación del drift FP16 vs FP32 y su impacto en la selección de tokens (top-1/top-k).

### Fase 3.2: Optimización de Segundo Orden
*Objetivo: Potencial competitivo y eficiencia real.*

1.  **Optimización lm_head:** GEMV especializado para vocabularios de más de 32k elementos.
2.  **Integración HIP Graphs:** Captura y repetición del loop de generación para eliminar más de 300 latencias de lanzamiento de kernels desde el host.
3.  **Localidad de KV-Cache:** Alineación y optimización de layout para acceso en modo ráfaga (burst) de HBM3.

### Fase 3.3: Release Técnico (AMD-facing)
*Objetivo: Entrega profesional y reproducibilidad.*

1.  **Guía de Reproducción:** Script para verificar todos los KPIs en un entorno estándar ROCm 7.1.
2.  **Especificaciones de Arquitectura:** Documentación de la lógica de bajo nivel del motor de inferencia.
3.  **Análisis de Gaps:** Lista clara de limitaciones actuales (ej. P2P entre múltiples GPUs, ajuste fino de FlashAttention).

## Riesgos Técnicos y Mitigación

| Riesgo | Estrategia de Mitigación |
| :--- | :--- |
| **Overhead de Lanzamiento:** Latencia del host dominando el ciclo. | Implementar HIP Graphs para delegar la lógica al firmware de la GPU. |
| **Drift Numérico:** Acumulación en FP16 causando divergencia. | Uso selectivo de FP32 en reducciones críticas (Softmax/Norms). |
| **Límite de Memoria:** Baja utilización de CUs en batch size pequeño. | Aumentar el paralelismo (TLP) en GEMV mediante wave-specialization. |

## ¿Por qué GRETA CORE? (Valor para AMD)
A diferencia de los frameworks de "caja negra", GRETA CORE ofrece:
1.  **Control Explícito del Runtime:** Sin capas de abstracción ocultas entre la lógica y ROCm/HIP.
2.  **Uso Nativo de Matrix Cores:** Uso de MFMA optimizado específicamente para gfx942 (CDNA 3).
3.  **Precisión Mixta Optimizada:** Equilibrio exacto entre pesos Q4_K/Q6_K y activaciones FP32.
