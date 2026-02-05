# GRETA Core AI Agent Workspace Rules

## 1. Local-First Discipline
- Realizar todos los cambios en local primero.
- Sincronizar (push) antes de ejecutar en remoto (MI300X).

## 2. Artifact Canonical Structure
- Los artefactos **SIEMPRE** se guardan en: `artifacts_remote/YYYY-MM-DD/b3_XX/`
- Estructura interna:
  - `gretacore_b3_XX_artifacts.tgz`: Bundle completo de la ejecución.
  - `b3_XX_analysis.txt`: Resumen técnico del análisis.
  - `run/`: Logs de ejecución de consola.
  - `traces/`: Archivos `.jsonl` extraídos.

## 3. Mandatory Documentation
- Cada bloque `B3.xx` **DEBE** actualizar:
  - `docs/PROGRESS.md`: Índice de progreso.
  - `docs/CHANGELOG.md`: Diario de trabajo.
  - `docs/es/debugging.md` / `docs/en/debugging.md`: Guías de debugging actualizadas.
- Reportes AMD: `docs/AMD/YYYY_MM_DD_B3_XX_<titulo>.md`

## 4. Build Environment
- No usar `cmake` en el root del repo.
- Usar `tools/inference/build` con `make` o `cmake` relativo a esa carpeta.

## 5. Branching
- Trabajar siempre en ramas descriptivas (ej: `b3_59_embedding_debug_input_audit`).
