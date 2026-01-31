# Matriz de Tolerancia Numérica - GRETA CORE

**Fecha:** 2026-01-31
**Estado:** Borrador Inicial

## Objetivo
Definir los límites aceptables de error absoluto y relativo para las operaciones de GRETA CORE comparadas con la Referencia CPU (FP64/FP32).

## Tolerancias por Tipo de Dato

| Operación | Precisión | Max Abs Error | Max Rel Error | Nota |
|:---|:---|:---|:---|:---|
| **GEMM** | FP32 | 1e-4 | 1e-4 | Acumulación en FP32. |
| **GEMM** | FP16 (Acc32) | 1e-2 | 1e-3 | Error esperado por truncamiento de inputs. |
| **RMSNorm** | FP32 | 1e-5 | 1e-5 | Sensible a epsilon. |
| **LayerNorm** | FP32 | 1e-5 | 1e-5 | Incluye media y varianza. |
| **Softmax** | FP32 | 1e-6 | 1e-6 | La suma debe ser exactamente 1.0 (aprox). |

## Reglas de Validación
1. **GEMM**: Se valida sobre una muestra de 8x8 esquinas y centro para matrices grandes (>1024), y completa para matrices pequeñas.
2. **Normas**: Se valida sobre el 100% de los elementos por fila.
3. **Escala**: En MI300X, las tolerancias pueden relajarse un 10% si se usa optimización de hardware específica (como Reduced Precision en registros) que sacrifique bits por velocidad.

## Acciones en Falla
- Si `max_abs_err > tolerancia`: El test marca `STATUS=FAILED`.
- Si el error es consistente (ej: siempre 2.0x la tolerancia): Investigar overflow en acumuladores.
- Si el error es aleatorio: Investigar condiciones de carrera en Shmem/Sincronización.
