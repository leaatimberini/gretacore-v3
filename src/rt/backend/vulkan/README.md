# GRETA CORE — Vulkan Backend (v1)

Path: src/rt/backend/vulkan/README.md  
Version: 1.0  
Language: EN/ES (bilingual)

## EN
This module provides the minimal Vulkan backend bootstrap:
- Instance creation
- Physical device selection (prefer AMD RADV GPU, avoid llvmpipe)
- Logical device + compute queue
- Command pool/buffer and empty submit (baseline)

Future:
- buffer allocator (device)
- pipeline cache
- SPIR-V compute pipeline (first kernels)

## ES
Este módulo provee el arranque mínimo del backend Vulkan:
- Creación de instancia
- Selección de GPU física (preferir AMD RADV, evitar llvmpipe)
- Dispositivo lógico + cola de compute
- Command pool/buffer y submit vacío (baseline)

Futuro:
- allocator de buffers (device)
- cache de pipelines
- pipeline compute SPIR-V (primeros kernels)
