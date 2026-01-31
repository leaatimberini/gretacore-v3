# Matriz de Brechas vs CUDA (Borrador)

Fecha: 2026-01-31

## Objetivo
Definir las capacidades mínimas de software que GRETA debe entregar para que los desarrolladores la elijan por encima de CUDA/ROCm en cargas de IA.

## Por qué CUDA gana (realidad)
- **Ubicuidad del lenguaje**: la mayoría del código IA asume CUDA.
- **Madurez de librerías**: cuDNN/TensorRT entregan grandes mejoras sin esfuerzo extra.
- **Escala**: NCCL + NVLink están probados en miles de GPUs.
- **Experiencia dev**: onboarding rápido, docs y comunidad.

## Respuesta estratégica de GRETA (pilares)
1) **Kernels sin traducción** (experiencia tipo Triton)
   - Meta: escribir una vez, correr en AMD sin HIP manual.
   - Gate: kernels referencia (GEMM, LayerNorm, Softmax, RMSNorm) desde un único frontend.

2) **Librerías de “cero esfuerzo”**
   - Meta: workloads comunes rápidos sin tuning manual.
   - Gate: speedups medibles vs kernels baseline con rutas pre‑tuneadas.
   - Nota: paridad vía equivalentes AMD (MIOpen/rocBLAS/MIGraphX), no binarios NVIDIA.

3) **Ruta dev unificada (estilo UAI)**
   - Meta: mismo flujo en Radeon y en Instinct.
   - Gate: mismos pasos de build + runtime en todos los tiers.

4) **Runtime listo para escala**
   - Meta: superficie de API preparada para NCCL/RCCL.
   - Gate: API definida aunque el backend sea gradual.

## Entregables inmediatos
- Publicar esta matriz (EN/ES) y mantenerla actualizada por hito.
- Definir 3 tareas diferenciadoras de GRETA para LOE‑3/LOE‑4.
- Agregar criterios de pase/fallo por pilar.
- Publicar plan de compatibilidad en `docs/es/strategy/framework_compat.md`.

## Tareas Diferenciadoras (LOE‑3 / LOE‑4)
1) **Pack de Primitivas LLM v1**
   - Alcance: LayerNorm, RMSNorm, Softmax (estable) con referencia CPU.
   - Pase: correctitud dentro de tolerancia; microbench archivado.

2) **Runner de Bloque Transformer**
   - Alcance: runner mínimo con RMSNorm + QKV + atención + MLP (FP16/FP32).
   - Pase: salidas deterministas en 5 corridas; métricas (ms, tokens/s) registradas.

3) **Cache de Tuning de Kernels**
   - Alcance: cache de mejor configuración por shape/dispositivo.
   - Pase: hit-rate ≥80% en corridas repetidas; speedup vs baseline registrado.

## Criterios de Pase/Fallo (Por pilar)
1) **Kernels sin traducción**
   - Pase: todos los kernels referencia desde un único frontend en AMD sin HIP manual.
   - Fallo: cualquier kernel exige traducción HIP manual.

2) **Librerías “cero esfuerzo”**
   - Pase: ≥1.5x speedup vs baseline en 3 workloads.
   - Fallo: sin speedup medible.

3) **Ruta dev unificada**
   - Pase: mismos pasos de build+run en Radeon + Instinct.
   - Fallo: pasos específicos por dispositivo.

4) **Runtime listo para escala**
   - Pase: API multi‑GPU definida y smoke‑test (single‑node).
   - Fallo: sin API o sin puntos de integración testeables.
