# Reglas del Workspace - GRETA CORE

## ⚠️ REGLAS CRÍTICAS

### 1. Abrir IDE SOLO en el Repo Canonical

```bash
# ✓ CORRECTO - Abrir VSCode aquí:
cd /media/leandro/D08A27808A2762683/gretacore/gretacore

# ✗ INCORRECTO - NO abrir el directorio padre:
cd /media/leandro/D08A27808A2762683/gretacore
```

### 2. Nunca Abrir el Directorio Padre

- El directorio `/media/leandro/D08A27808A2762683/gretacore/` NO es un workspace
- Solo `/media/leandro/D08A27808A2762683/gretacore/gretacore/` es el workspace válido
- Abrir el padre puede causar que agentes creen siblings

### 3. Antes de Sesiones de Agentes: Ejecutar Guard

```bash
cd /media/leandro/D08A27808A2762683/gretacore/gretacore
./tools/guard_no_sibling_checkouts.sh
```

**Salida esperada:** `=== OK ===` (exit code 0)

---

## 🚨 Si Aparece un Sibling (Carpeta Hermana)

### Síntomas
- El guard script devuelve `=== FAIL ===`
- Hay carpetas como `temp-checkout`, `gretacore_local_clean`, `gretacore_b3_XX`

### Procedimiento de Recuperación

1. **Rescatar cambios del sibling:**
   ```bash
   cd /media/leandro/D08A27808A2762683/gretacore/<sibling>
   git diff > /tmp/rescate.patch
   ```

2. **Aplicar en el repo canonical:**
   ```bash
   cd /media/leandro/D08A27808A2762683/gretacore/gretacore
   git apply /tmp/rescate.patch
   ```

3. **Verificar:**
   ```bash
   git status --porcelain
   ```

4. **Eliminar el sibling:**
   ```bash
   rm -rf /media/leandro/D08A27808A2762683/gretacore/<sibling>
   ```

5. **Confirmar que el workspace está limpio:**
   ```bash
   ./tools/guard_no_sibling_checkouts.sh
   ```

---

## 📁 Estructura del Workspace

```
/media/leandro/D08A27808A2762683/gretacore/          ← NO ABRIR (solo contenedor)
└── gretacore/                         ← ✓ ABIERTURA VÁLIDA
    ├── .git/
    ├── .agents/skills/                ← Symlinks a ~/.gemini/antigravity/skills/
    ├── .kilocode/skills/              ← Symlinks a ~/.gemini/antigravity/skills/
    ├── tools/
    │   └── guard_no_sibling_checkouts.sh  ← VERIFICAR ANTES DE CADA SESIÓN
    ├── docs/
    │   └── WORKSPACE_RULES.md         ← ESTE ARCHIVO
    ├── AGENTS.md                      ← REGLAS COMPLETAS
    └── ...
```

---

## 🛠️ Cómo Trabajar Correctamente

### Para Nuevo Trabajo
```bash
cd /media/leandro/D08A27808A2762683/gretacore/gretacore
git switch -c nueva-rama-trabajo
# ... hacer cambios ...
git add .
git commit -m "descripción"
```

### Para Cambios Temporales
```bash
git stash push -m "trabajo temporal"
# ... hacer otras tareas ...
git stash pop
```

### Para Limpiar Estado Local
```bash
# Respaldo primero:
cp -r .git/refs/heads/* /tmp/backup_heads/
# Luego reset:
git reset --hard HEAD
```

---

## 📚 Documentación Relacionada

- [`AGENTS.md`](../AGENTS.md) - Reglas completas para agentes de IA
- [`tools/guard_no_sibling_checkouts.sh`](../tools/guard_no_sibling_checkouts.sh) - Script de verificación

---

**Última actualización:** 2026-02-06
