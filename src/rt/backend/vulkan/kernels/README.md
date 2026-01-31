# GRETA Vulkan Runtime Kernels

## Objetivo
Mover la lógica de creación de pipelines + dispatch desde `tools/bench/runtime/src/*`
al runtime (`src/rt/backend/vulkan`), manteniendo:
- Autotune winner por device + bucket (cache JSON)
- Fallback si no hay soporte (ej: subgroup32)

## Requisitos
Los `.spv` deben existir en:
- `$GRETA_VK_SPV_DIR` (si está seteado), o
- `<cwd>/build/*.spv` (compatibilidad con los benches)

## Pendiente (obligatorio antes de usar en producción)
Copiar exactamente desde los benches:
- VkDescriptorSetLayoutBinding (bindings y tipos)
- VkPushConstantRange (si aplica)
- Cálculo de dispatch gx/gy por kernel (tile sizes)
