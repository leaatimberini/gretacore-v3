# Plan de Compatibilidad de Frameworks (Borrador)

Fecha: 2026-01-31

## Objetivo
Permitir que el mismo código corra en Radeon (dev) y MI300X (cloud) sin cambios y sin flujos obligatorios de Docker.

## Alcance
- **Triton** (OpenAI): frontend de kernels
- **PyTorch**: integración de training/inference
- **JAX**: investigación + stack de compilación
- **Librerías NVIDIA‑only** (cuDNN / TensorRT): paridad vía equivalentes AMD

## No‑Objetivos
- Distribuir binarios NVIDIA
- Depender de CUDA para el runtime GRETA

## Estrategia de Compatibilidad
1) **Ruta Triton‑first**
   - Frontend Triton soportado para targets AMD.
   - Kernels GRETA expuestos como ops compatibles cuando aplique.

2) **Puente PyTorch**
   - Wheels ROCm como base.
   - Ops GRETA + dispatch para kernels críticos.

3) **Puente JAX (PJRT/XLA)**
   - Backend AMD para JAX.
   - Custom calls GRETA para kernels calientes.

4) **Paridad cuDNN/TensorRT**
   - Mapear a equivalentes AMD (MIOpen, rocBLAS, MIGraphX) o kernels GRETA.
   - Fallbacks documentados con expectativas de performance claras.

## Packaging & DX (Sin Docker Obligatorio)
- Instalación de un comando (pip/conda) para dev y cloud.
- Mismos pasos de build/run en Radeon y MI300X.
- Resultados capturados con scripts estándar + CSV.
- El instalador de GRETA incluye torch, triton y jax (sin instalaciones extra).

## Gates de Pase/Fallo
- **Mismo código** en Radeon y MI300X sin ediciones.
- **Instalación** ≤ 3 comandos; no requiere Docker.
- **Paridad frameworks**: kernels clave corren vía Triton/PyTorch/JAX.
- **Resultados reproducibles**: benchmarks archivados con metadata.
- **Lock de versiones**: versiones pinneadas en `docs/es/strategy/framework_versions.md`.

## Artefactos Prototipo
- `tools/compat/triton/vec_add.py`
- `tools/compat/pytorch/greta_extension_hello.py`
- `tools/compat/jax/jax_custom_call_hello.py`

## Criterios de Estado del Prototipo
- Triton: `STATUS=OK` en backend AMD con ROCm.
- PyTorch: `STATUS=OK` en custom op (baseline CPU); path GPU en LOE‑5.
- JAX: `STATUS=OK` en backend; custom call queda TODO.
- Fallback CPU de Triton es aceptable en notebooks sin ROCm.
