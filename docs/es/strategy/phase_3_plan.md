# Fase 3: Pipeline de Inferencia LLM

## Objetivo
Ejecutar inferencia end-to-end de un modelo LLM (Llama-2-7B) utilizando el backend HIP nativo sobre MI300X, con tokenización y generación autoregresiva de texto.

## Entregables Clave
1. **Scheduler de Bloques:** Ejecutar N capas transformer de forma secuencial.
2. **Cargador de Pesos:** Leer tensores desde formato SafeTensors/GGUF.
3. **Tokenizer:** Integrar SentencePiece o BPE para entrada/salida de texto.
4. **Generación Autoregresiva:** Bucle de predicción token-a-token con sampling.
5. **Benchmarks E2E:** Métricas de tokens/s y latencia.

## Dependencias de Fase 2
Esta fase depende directamente de los componentes validados en la Fase 2:
- **GEMM MFMA:** 13 TFLOPS validado.
- **RMSNorm, Softmax:** Error < 1e-6.
- **RoPE, Causal Mask:** Validados en grafo.
- **KV-Cache:** Actualizaciones O(1) con fidelidad bit-perfect.
- **Graph Runner:** Ejecución determinista de cadenas de kernels.

## Arquitectura Propuesta

```
┌─────────────────────────────────────────────────────────────────┐
│                     greta_infer (CLI)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐      │
│  │ Tokenizer   │  │ Generator    │  │ BlockScheduler     │      │
│  │  (BPE/SP)   │→ │  (Autoregr.) │→ │  (N layers graph)  │      │
│  └─────────────┘  └──────────────┘  └────────────────────┘      │
│                                               │                 │
│  ┌─────────────────────────────────┐          │                 │
│  │ WeightLoader                    │←─────────┘                 │
│  │  (SafeTensors / GGUF)           │                            │
│  └─────────────────────────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               HIP Backend (gcore::rt::hip)                      │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────┐     │
│  │ GEMM   │ │RMSNorm │ │ RoPE   │ │ SiLU   │ │ KV-Cache   │     │
│  │ MFMA   │ │        │ │        │ │        │ │            │     │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    AMD Instinct MI300X
```

## Estrategia de Implementación
1. **Semana 1:** Weight Loader + Model Config + estructuras base.
2. **Semana 2:** Block Scheduler + pool de buffers + ejecución multi-capa.
3. **Semana 3:** Tokenizer + Generator + bucle autoregresivo.
4. **Semana 4:** CLI + E2E test + optimización inicial.

## Criterios de Éxito
- Ejecutar `greta_infer` con pesos reales de Llama-2-7B.
- Generar texto coherente (>50 tokens).
- Medir tokens/s >= 10 tok/s (baseline conservador).
- Sin crashes ni memory leaks.

## Riesgos y Mitigaciones
| Riesgo | Mitigación |
| :--- | :--- |
| Formatos de peso complejos | Empezar con GGUF (más simple) |
| Memory pressure en MI300X | Pre-allocar todos los buffers al inicio |
| Bajo throughput inicial | Optimizar después de validar correctitud |

## Propietario
**Fase 3 Owner:** Leandro Emanuel Timberini
