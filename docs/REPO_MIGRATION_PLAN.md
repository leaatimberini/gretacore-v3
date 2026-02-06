# GRETA CORE Repository Migration Plan

**Date:** 2026-02-06  
**Status:** PLANNED  
**Target:** Clean `gretacore` repository (new, no history)

---

## Objective

Migrate GRETA CORE documentation and artifacts to a new clean repository (`gretacore`) that:

1. Contains only public-safe content
2. Excludes all agent/AI/skills/tooling references
3. Maintains full bilingual documentation (ES/EN)
4. Preserves all AMD technical reports and artifacts

---

## Pre-Migration Checklist

### 1. Content to EXCLUDE (Prohibited)

| Path | Reason |
|------|--------|
| `.agents/` | AI agent tooling |
| `.kilocode/` | AI agent tooling |
| `AGENTS.md` | Contains agent/skills references |
| `docs/WORKSPACE_RULES.md` | Internal tooling references |
| `tools/diagnostics/` | Development tooling |
| `artifacts/` | Debug outputs |
| `.venv/` | Python virtual environment |

### 2. Content to INCLUDE (Public)

| Path | Description |
|------|-------------|
| `src/` | Core implementation |
| `tools/` | Public tools (excl. diagnostics) |
| `docs/` | All documentation (bilingual) |
| `models/` | Model placeholders |
| `README.md` | Project README |
| `README_ES.md` | Spanish README |
| `implementation_plan.md` | Implementation roadmap |

### 3. Content to ARCHIVE (Keep for reference, not migrate)

| Path | Action |
|------|--------|
| `artifacts_remote/_rescued_from_*/` | Keep in separate archive repo |

---

## Migration Steps

### Step 1: Create New Repository

```bash
# Create new repository on GitHub: gretacore
# Clone locally
mkdir gretacore
cd gretacore
git init
git remote add origin git@github.com:leaatimberini/gretacore.git
```

### Step 2: Copy Public Content

```bash
# From canonical repo
cd /media/leandro/D08A27808A2762683/gretacore/gretacore

# Copy source
cp -r src/ /tmp/gretacore_migration/

# Copy tools (excluding diagnostics)
cp -r tools/ /tmp/gretacore_migration/
rm -rf tools/diagnostics/

# Copy docs (selective)
cp -r docs/ /tmp/gretacore_migration/
rm -f docs/WORKSPACE_RULES.md

# Copy root files
cp README.md README_ES.md implementation_plan.md .gitignore \
   LICENSE.md CHANGELOG.md /tmp/gretacore_migration/
```

### Step 3: Verify Content

```bash
cd /tmp/gretacore_migration
# Verify no prohibited content
grep -r "agents" . --include="*.md" || echo "No agent references found"
grep -r "kilocode" . --include="*.md" || echo "No kilocode references found"
```

### Step 4: Commit and Push

```bash
cd /tmp/gretacore_migration
git add .
git commit -m "chore: initial migration from gretacore (clean repo)

- Excluded: agents, skills, tooling references
- Included: src, tools (excl. diagnostics), docs
- Bilingual documentation maintained"
git push origin main
```

---

## Documentation Mapping

| Original Path | New Path |
|---------------|----------|
| `docs/AMD/` | `docs/AMD/` (full) |
| `docs/AMD/INDEX.md` | `docs/AMD/INDEX.md` |
| `docs/AMD/INDEX_ES.md` | `docs/AMD/INDEX_ES.md` |
| `docs/AMD/phases/` | `docs/AMD/phases/` |
| `docs/en/` | `docs/en/` |
| `docs/es/` | `docs/es/` |
| `docs/PROGRESS.md` | `docs/PROGRESS.md` |

---

## Post-Migration Validation

1. **GitHub Verification**
   - Check repository size
   - Verify commit history
   - Confirm branch protection rules

2. **Content Verification**
   ```bash
   # Run on new repo
   find . -name "*.md" -exec grep -l "agents\|kilocode\|skills" {} \;
   # Should return empty
   ```

3. **Documentation Check**
   - Verify bilingual index files exist
   - Confirm AMD reports are accessible

---

## Rollback Plan

If issues are discovered after migration:

```bash
# Restore from backup
cd /tmp/gretacore_backup
git push --force origin main
```

---

## Archive Repositories (Future)

After successful migration:

1. **gretacore-private**: For internal tooling, agent configs
2. **gretacore-artifacts**: For rescued artifacts (large files)

---

## Timeline

| Phase | Date | Status |
|-------|------|--------|
| Plan creation | 2026-02-06 | ✅ DONE |
| Content selection | 2026-02-06 | ✅ DONE |
| Migration execution | TBD | PENDING |
| Validation | TBD | PENDING |

---

*Maintained by: Leandro Emanuel Timberini*
