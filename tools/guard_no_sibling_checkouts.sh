#!/bin/bash
# =============================================================================
# GUARD: No Sibling Checkouts
# =============================================================================
# Este script verifica que NO existan carpetas sisters (repos git hermanos)
# que puedan causar divergencia de trabajo.
#
# Uso: ./tools/guard_no_sibling_checkouts.sh
# Salida: "OK" si no hay siblings, "FAIL" + lista de siblings si los hay
# Código de salida: 0 si OK, 1 si FAIL
# =============================================================================

set -e

CANONICAL_REPO="/media/leandro/D08A27808A2762683/gretacore/gretacore"
PARENT_DIR="/media/leandro/D08A27808A2762683/gretacore"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "=== GUARD: No Sibling Checkouts ==="
echo ""

# 1. Verificar que el repo canonical existe
if [ ! -d "$CANONICAL_REPO/.git" ]; then
    echo -e "${RED}ERROR: No se encontró .git en el repo canonical: $CANONICAL_REPO${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Repo canonical existe: $CANONICAL_REPO${NC}"

# 2. Listar todos los directorios en el parent
echo ""
echo "Directorios en $PARENT_DIR:"
ls -la "$PARENT_DIR" | grep "^d" | awk '{print $NF}' | grep -v "^\.$" | grep -v "^\.\.$"

# 3. Buscar siblings prohibidos (carpetas .git en el parent que NO sean el canonical)
echo ""
echo "Buscando siblings prohibidos..."

SIBLING_FOUND=0
SIBLING_LIST=""

for dir in "$PARENT_DIR"/*/; do
    # Skip si no es un directorio
    [ -d "$dir" ] || continue
    
    # Extraer nombre de la carpeta
    dirname=$(basename "$dir")
    
    # Skip el repo canonical
    if [ "$dirname" = "gretacore" ]; then
        continue
    fi
    
    # Skip archivos especiales no-problemáticos
    case "$dirname" in
        *.code-workspace)
            continue
            ;;
    esac
    
    # Verificar si tiene .git (es un repo)
    if [ -d "$dir/.git" ]; then
        SIBLING_FOUND=1
        SIBLING_LIST="$SIBLING_LIST\n  - $dir (tiene .git)"
        echo -e "${RED}✗ SIBLING DETECTADO: $dir${NC}"
    fi
done

# 4. Resultado
echo ""
if [ $SIBLING_FOUND -eq 1 ]; then
    echo -e "${RED}=== FAIL ===${NC}"
    echo "Se detectaron carpetas sisters con .git:"
    echo -e "$SIBLING_LIST"
    echo ""
    echo "ACCIÓN REQUERIDA:"
    echo "  1. Rescatar cambios: git diff > /tmp/rescate.patch"
    echo "  2. Aplicar en canonical: cd $CANONICAL_REPO && git apply /tmp/rescate.patch"
    echo "  3. Eliminar sibling: rm -rf $PARENT_DIR/<sibling>"
    exit 1
else
    echo -e "${GREEN}=== OK ===${NC}"
    echo "No se detectaron carpetas sisters."
    echo "Workspace seguro: solo existe $CANONICAL_REPO"
    exit 0
fi
