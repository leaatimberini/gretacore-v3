# AGENTS.md

## Reglas de Trabajo para Agentes de IA

Este documento establece las reglas y mejores prácticas para todos los agentes de IA que trabajan en el proyecto GRETA CORE.

### ÚNICO REPO DE TRABAJO

**Repo canonical:** `/media/leandro/D08A27808A2762683/gretacore/gretacore/`

**ABSOLUTAMENTE PROHIBIDO:**
- ❌ Crear clones/checkouts en carpetas hermanas del repo canonical
- ❌ Crear subdirectorios como `gretacore_b3_XX`, `gretacore_local_clean`, `temp-checkout`, etc.
- ❌ Trabajar fuera del directorio `/media/leandro/D08A27808A2762683/gretacore/gretacore/`

### CÓMO TRABAJAR CORRECTAMENTE

#### Opción 1: Ramas Git
```bash
cd /media/leandro/D08A27808A2762683/gretacore/gretacore
git switch -c nueva-rama-trabajo
# Trabajar...
git add .
git commit -m "descripción del trabajo"
```

#### Opción 2: Stash (para cambios temporales)
```bash
git stash push -m "trabajo temporal"
# ... hacer otras tareas ...
git stash pop
```

#### Opción 3: Reset controlado (con backup)
```bash
cd /media/leandro/D08A27808A2762683/gretacore/gretacore
# Hacer backup primero:
cp -r .git/refs/heads/* /tmp/backup_heads/
# Luego hacer reset si es necesario
```

### RECUPERACIÓN DE EMERGENCIAS

Si por error se creó trabajo en una carpeta hermana:

1. **Identificar cambios:**
   ```bash
   cd /media/leandro/D08A27808A2762683/gretacore/<carpeta-extra>
   git diff > /tmp/rescate.patch
   git diff --stat
   ```

2. **Rescatar al repo canonical:**
   ```bash
   cd /media/leandro/D08A27808A2762683/gretacore/gretacore
   git apply /tmp/rescate.patch
   ```

3. **Verificar:**
   ```bash
   git status --porcelain
   ```

### SKILLS COMPARTIDOS

Los skills de agentes están centralizados en:
- Source of Truth: `~/.gemini/antigravity/skills/`
- Accesibles via symlinks en:
  - `.agents/skills/`
  - `.kilocode/skills/`

**NO modificar skills directamente en el repo.** Los cambios deben hacerse en `~/.gemini/antigravity/skills/` y se reflejarán automáticamente via symlinks.

### ARCHIVOS EXCLUIDOS LOCALMENTE

Los siguientes patrones están excluidos en `.git/info/exclude`:
- `.agents/` - symlinks a skills
- `.kilocode/` - symlinks a skills
- `.venv/` - entorno virtual Python
- `artifacts/` - outputs de debug
- `tools/diagnostics/` - herramientas de diagnóstico

### NOTAS DE SEGURIDAD

- ❌ NO incluir secretos, API keys, o tokens en commits
- ❌ NO hacer push a remotos sin aprobación
- Usar `.git/info/exclude` para archivos sensibles locales (NO modificar `.gitignore`)

---

**Última actualización:** 2026-02-06
