# GRETA CORE - Protocolo de Benchmark de la Fase 3

Este protocolo define la metodología para medir el rendimiento y la corrección en el hardware AMD Instinct™.

## 1. Definición de Métricas
- **TTFT (Time To First Token)**: Latencia inicial desde el envío del prompt hasta el primer token generado (ms). Mide la eficiencia del prefill.
- **T/s (Tokens por Segundo)**: Rendimiento promedio de generación. Mide la eficiencia del decode.
- **ms/capa**: Tiempo de ejecución por capa del transformer (ms).
- **Corrección**: Verificación de paridad absoluta contra una salida de referencia (sin NaNs, logits estables).

## 2. Matriz Experimental
Cada ejecución de benchmark debe iterar a través del siguiente espacio de configuración:

| Variable | Valores | Descripción |
| :--- | :--- | :--- |
| **HIP Graphs** | `ON` (1), `OFF` (0) | Evalúa la reducción del overhead de programación del host. |
| **LM_HEAD Especializado**| `ON` (1), `OFF` (0) | Compara la salida basada en GEMV vs. GEMM estándar. |
| **Prompts** | Corto, Medio | "Hola", "¿Cuál es la capital de Francia?" |

## 3. Control del Entorno
- **Frecuencia de la GPU**: Fijada al máximo (ej. mediante `rocm-smi`).
- **Enfriamiento**: Asegurar que no haya thermal throttling.
- **Variables**:
    - `GRETA_HIP_GRAPH`
    - `GRETA_USE_SPECIALIZED_LM_HEAD`
    - `GRETA_VERBOSE_INFO=1`

## 4. Recolección de Datos
Los resultados se recopilan utilizando `tools/phase3/run_bench.sh` y se guardan en `bench_results.csv` siguiendo el esquema definido en `docs/phase3/bench_schema.csv`.

### Columnas del CSV:
`timestamp, model, prompt, max_tokens, greedy, hip_graph, spec_lm_head, ttft_ms, tokens_per_sec, total_ms`
