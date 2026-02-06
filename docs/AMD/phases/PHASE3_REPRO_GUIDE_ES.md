# GRETA CORE - Guía de Reproducción de la Fase 3

Esta guía describe cómo reproducir los resultados de rendimiento y corrección de la Fase 3 en aceleradores AMD Instinct™.

## Requisitos de Hardware
- **Acelerador**: AMD Instinct™ MI300X (HBM3)
- **RAM del Host**: 128GB+ (mínimo para la carga de modelos grandes)
- **Disco**: 50GB+ para checkpoints del modelo y artefactos de compilación

## Requisitos de Software
- **S.O.**: Ubuntu 22.04 LTS o 24.04 LTS (Noble)
- **Stack ROCm™**: Versión 6.2 o 7.1
- **Compiladores**: `hipcc` (basado en Clang)
- **Sistema de Compilación**: CMake 3.22+ y Make

## Instrucciones de Compilación

### 1. Compilar el Motor
```bash
cd gretacore
mkdir -p tools/inference/build
cd tools/inference/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### 2. Preparar el Modelo
Asegúrese de tener un modelo Llama-2-7B en formato GGUF (se recomienda Q4_K_M).
Ruta predeterminada para benchmarks: `/root/models/llama-2-7b-chat.Q4_K_M.gguf`.

## Ejecución y Validación

### Inferencia Estándar
```bash
./greta_infer --model [ruta_al_modelo] --prompt "Hola" --max-tokens 32 --greedy
```

### Benchmark de Rendimiento (con HIP Graphs)
```bash
GRETA_HIP_GRAPH=1 GRETA_VERBOSE_INFO=1 ./greta_infer --model [ruta_al_modelo] --prompt "Hola" --max-tokens 32 --greedy
```

### Auditoría Numérica (Tracer)
```bash
GRETA_TRACE_LEVEL=1 GRETA_PROFILE_BLOCKS=1 ./greta_infer --model [ruta_al_modelo] --prompt "Hola" --max-tokens 5 --greedy
```

## Benchmarking Automatizado
Utilice el script proporcionado para ejecutar la matriz completa de la Fase 3:
```bash
cd tools/phase3
./run_bench.sh [ruta_al_modelo]
```
Los resultados se exportarán a `bench_results.csv`.

## Resolución de Problemas
- **NaNs en la salida**: Verifique si `GRETA_TRACE_LEVEL=1` está activo para identificar el primer bloque con fallos.
- **Fallos en HIP Graph**: Asegúrese de estar utilizando una versión compatible de ROCm y que `S=1` (fase de decode) para la captura del grafo.
