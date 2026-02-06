# GRETA CORE

**Estado**: Fase 3 hasta B3.60

GRETA CORE es un proyecto de ingeniería a largo plazo enfocado en la
construcción de un stack de cómputo mínimo, de alto rendimiento y de
estilo CUDA para hardware AMD, diseñado específicamente para Modelos
de Lenguaje de Gran Escala (LLMs).

El proyecto existe para romper el lock-in actual de CUDA atacando el
problema en su raíz: el software.

---

## Progreso Fase 3 (Serie de Auditorías B3.xx)

| Hito | Estado | Descripción |
|------|--------|-------------|
| B3.52 | ✅ PASS | Fix de direccionamiento KV cache |
| B3.55-B3.58 | ✅ PASS | Aislamiento de causa raíz (RoPE/Q-proj/RMSNorm) |
| B3.59 | ✅ PASS | Auditoría Embedding + StageDebugInput (sin zeroing) |
| B3.60 | ✅ PASS | Bisect Attention Block (pipeline Layer0 verificado) |

**Documentación**:
- [Índice de Progreso](docs/PROGRESS.md)
- [Índice de Reportes AMD](docs/AMD/INDEX.md)

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
- No es un proyecto de optimización a corto plazo

---

## Documentación

- [Seguimiento de Progreso](docs/PROGRESS.md)
- [Reportes de Auditoría AMD](docs/AMD/INDEX.md)
- [Herramientas de Benchmarking](tools/benchmarks/)

---

## Inicio Rápido

```bash
# Clonar el repositorio
git clone https://github.com/leaatimberini/gretacore.git
cd gretacore

# Ver documentación
cat docs/PROGRESS.md
cat docs/AMD/INDEX.md
```

---

## Estructura del Proyecto

```
├── src/           # Implementación del runtime core
├── tools/         # Herramientas de benchmarking y diagnóstico
├── docs/          # Documentación y reportes AMD
├── models/        # Definiciones de modelos
├── build/         # Artefactos de build (no versionado)
├── README.md      # Este archivo (versión EN)
└── .gitignore    # Reglas de git ignore
```

---

## Contribuyendo

Esta es una iniciativa de ingeniería a largo plazo. Todas las contribuciones
deben alinearse con la filosofía del proyecto de minimalismo, rendimiento
y control total del stack.

Las contribuciones deben centrarse exclusivamente en:
- Código fuente
- Documentación técnica
- Benchmarks reproducibles
- Auditorías verificables

No se aceptan cambios que no estén directamente relacionados con el motor
de inferencia o su documentación.
