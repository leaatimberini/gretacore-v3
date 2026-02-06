# Plan de Migración del Repositorio GRETA CORE

**Fecha:** 2026-02-06  
**Estado:** PLANIFICADO  
**Objetivo:** Repositorio `gretacore` limpio (nuevo, sin historial)

---

## Objetivo

Migrar la documentación y artefactos de GRETA CORE a un nuevo repositorio limpio (`gretacore`) que:

1. Contenga solo contenido público-seguro
2. Excluya todas las referencias a agentes/IA/skills/herramientas
3. Mantenga documentación bilingüe completa (ES/EN)
4. Preserve todos los reportes técnicos AMD y artefactos

---

## Lista de Verificación Pre-Migración

### 1. Contenido a EXCLUIR (Prohibido)

| Ruta | Razón |
|------|-------|
| `.agents/` | Herramientas de agentes IA |
| `.kilocode/` | Herramientas de agentes IA |
| `AGENTS.md` | Contiene referencias a agents/skills |
| `docs/WORKSPACE_RULES.md` | Referencias a herramientas internas |
| `tools/diagnostics/` | Herramientas de desarrollo |
| `artifacts/` | Outputs de debug |
| `.venv/` | Entorno virtual Python |

### 2. Contenido a INCLUIR (Público)

| Ruta | Descripción |
|------|-------------|
| `src/` | Implementación core |
| `tools/` | Herramientas públicas (excl. diagnostics) |
| `docs/` | Toda la documentación (bilingüe) |
| `models/` | Placeholders de modelos |
| `README.md` | README del proyecto |
| `README_ES.md` | README en español |
| `implementation_plan.md` | Hoja de ruta de implementación |

### 3. Contenido a ARCHIVAR (Mantener para referencia, no migrar)

| Ruta | Acción |
|------|--------|
| `artifacts_remote/_rescued_from_*/` | Mantener en repositorio archive separado |

---

## Pasos de Migración

### Paso 1: Crear Nuevo Repositorio

```bash
# Crear nuevo repositorio en GitHub: gretacore
# Clonar localmente
mkdir gretacore
cd gretacore
git init
git remote add origin git@github.com:leaatimberini/gretacore.git
```

### Paso 2: Copiar Contenido Público

```bash
# Desde el repo canónico
cd /media/leandro/D08A27808A2762683/gretacore/gretacore

# Copiar código fuente
cp -r src/ /tmp/gretacore_migration/

# Copiar herramientas (excluyendo diagnostics)
cp -r tools/ /tmp/gretacore_migration/
rm -rf tools/diagnostics/

# Copiar documentación (selectiva)
cp -r docs/ /tmp/gretacore_migration/
rm -f docs/WORKSPACE_RULES.md

# Copiar archivos root
cp README.md README_ES.md implementation_plan.md .gitignore \
   LICENSE.md CHANGELOG.md /tmp/gretacore_migration/
```

### Paso 3: Verificar Contenido

```bash
cd /tmp/gretacore_migration
# Verificar que no hay contenido prohibido
grep -r "agents" . --include="*.md" || echo "No se encontraron referencias a agents"
grep -r "kilocode" . --include="*.md" || echo "No se encontraron referencias a kilocode"
```

### Paso 4: Commit y Push

```bash
cd /tmp/gretacore_migration
git add .
git commit -m "chore: migración inicial desde gretacore (repo limpio)

- Excluido: agents, skills, referencias a herramientas
- Incluido: src, tools (excl. diagnostics), docs
- Documentación bilingüe mantenida"
git push origin main
```

---

## Mapeo de Documentación

| Ruta Original | Nueva Ruta |
|---------------|-----------|
| `docs/AMD/` | `docs/AMD/` (completo) |
| `docs/AMD/INDEX.md` | `docs/AMD/INDEX.md` |
| `docs/AMD/INDEX_ES.md` | `docs/AMD/INDEX_ES.md` |
| `docs/AMD/phases/` | `docs/AMD/phases/` |
| `docs/en/` | `docs/en/` |
| `docs/es/` | `docs/es/` |
| `docs/PROGRESS.md` | `docs/PROGRESS.md` |

---

## Validación Post-Migración

1. **Verificación GitHub**
   - Verificar tamaño del repositorio
   - Confirmar historial de commits
   - Verificar reglas de protección de ramas

2. **Verificación de Contenido**
   ```bash
   # Ejecutar en el nuevo repo
   find . -name "*.md" -exec grep -l "agents\|kilocode\|skills" {} \;
   # Debe retornar vacío
   ```

3. **Verificación Documentación**
   - Verificar que existen archivos de índice bilingües
   - Confirmar accesibilidad de reportes AMD

---

## Plan de Rollback

Si se descubren problemas después de la migración:

```bash
# Restaurar desde backup
cd /tmp/gretacore_backup
git push --force origin main
```

---

## Repositorios de Archivo (Futuro)

Después de una migración exitosa:

1. **gretacore-private**: Para herramientas internas, configs de agentes
2. **gretacore-artifacts**: Para artefactos rescatados (archivos grandes)

---

## Cronograma

| Fase | Fecha | Estado |
|------|------|--------|
| Creación del plan | 2026-02-06 | ✅ HECHO |
| Selección de contenido | 2026-02-06 | ✅ HECHO |
| Ejecución de migración | TBD | PENDIENTE |
| Validación | TBD | PENDIENTE |

---

*Mantenido por: Leandro Emanuel Timberini*
