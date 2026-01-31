# GRETA CORE

GRETA CORE es un proyecto de ingeniería a largo plazo enfocado en la
construcción de un stack de cómputo mínimo, de alto rendimiento y de
estilo CUDA para hardware AMD, diseñado específicamente para Modelos
de Lenguaje de Gran Escala (LLMs).

El proyecto existe para romper el lock-in actual de CUDA atacando el
problema en su raíz: el software.

---

## Motivación

El ecosistema moderno de inteligencia artificial está dominado por una
única plataforma de cómputo. Esta dominancia ha creado barreras de
entrada artificiales, incrementado el costo del hardware y limitado
la innovación.

GRETA CORE aborda este problema desde una perspectiva software-first,
buscando liberar todo el potencial del hardware AMD mediante un stack
de cómputo enfocado y orientado al rendimiento.

---

## Filosofía

- Software por sobre hardware
- Control total del stack
- Minimalismo sobre bloat
- Rendimiento por sobre abstracción
- Disciplina de ingeniería a largo plazo

---

## Qué es GRETA CORE

- Un runtime de cómputo personalizado para hardware AMD
- Un stack de ejecución LLM kernel-first
- Una experiencia de desarrollo tipo CUDA sin replicar CUDA
- Una iniciativa de investigación e ingeniería a largo plazo
- Una instalación que incluye torch, triton y jax (sin instalaciones extra)

---

## Qué NO es GRETA CORE

- No es un fork de CUDA
- No es un wrapper delgado sobre frameworks existentes
- No es una plataforma de cómputo GPU de propósito general
- No es un proyecto de optimización de corto plazo

---

## Estado del Proyecto

**Fase 1 – Runtime Core (activo)**
**Fase 2 – Dominio de Kernels (prototipos tempranos)**

- Backend Vulkan implementado y usado por benches.
- Kernels GEMM FP16 entrada / FP32 acumulacion (FP16 gated).
- Autotuning con cache persistente activo.
- Herramientas de smoke/bench para gate de seguridad.

## v0.1 Release Publico (Definition of Done)

- Runtime core estable en APU (sin hangs en perfiles smoke).
- Backend Vulkan con buffers device-local + staging.
- Correctitud GEMM validada vs referencia CPU (FP32 requerido, FP16 opcional).
- Benchmarks baseline registrados con metadata de entorno.

### Baseline Snapshot (2026-01-29, Ryzen 5 8600G APU)

| Benchmark | Metrica | Valor |
| --- | --- | --- |
| vk_smoke_bench | empty_submit_wait_ms | 1.845 |
| dispatch_bench | mean_ns_per_submit_and_exec | 312.787 |
| stream_bench | mean_ns_per_task | 60.594 |
| telemetry_bench | mean_ns_per_scope | 39.693 |
| alloc_bench | mean_ops_per_sec | 98,623,523.700 |

---

## Documentación

- Whitepaper: `docs/es/whitepaper.md`
- Roadmap: `docs/es/roadmap.md`
- Plan de trabajo: `docs/es/workplan.md`
- Healthcheck FP16: `docs/es/runtime_fp16_healthcheck.md`
- Perfiles de seguridad: `docs/es/runtime_safety_profiles.md`
- Checklist de validación: `docs/es/runtime_validation_checklist.md`
- Matriz de compatibilidad: `docs/es/compatibility.md`
- Plan de compatibilidad de frameworks: `docs/es/strategy/framework_compat.md`
- Matriz de versiones de frameworks: `docs/es/strategy/framework_versions.md`
- Prototipos de frameworks: `tools/compat/README.md`

---

## Autor

GRETA CORE fue concebido, fundado y es liderado por:

**Leandro Emanuel Timberini**  
Fundador y Arquitecto Principal de Sistemas

---

## Licencia

Licencia a definir.
