# GRETA CORE — Selección de Backend (Runtime de Cómputo GPU)

Path: docs/es/backend_selection.md  
Version: 1.0  
Fecha: 2026-01-29  
Autor / Propietario: Leandro Emanuel Timberini

---

## 1) Propósito

Este documento deja asentada la decisión del backend inicial de cómputo GPU para GRETA CORE.  
El objetivo es habilitar un runtime práctico, mantenible y de alto rendimiento para cargas LLM sobre hardware AMD.

Es un documento vivo: las decisiones pueden evolucionar si cambian drivers, toolchains o soporte de hardware.

---

## 2) Hardware y OS objetivo (plataforma actual de pruebas)

- CPU/APU: AMD Ryzen 5 8600G (APU, iGPU RDNA3)
- GPU: Integrada RDNA3 (clase Phoenix)
- OS: Ubuntu Desktop 22.04 (Kernel: 6.8.0-90-generic)
- Stack Vulkan: Mesa RADV
- RAM: 16GB DDR5 (actualmente 1 DIMM instalado en Channel B, por lo tanto probablemente single-channel)

---

## 3) Señales observadas del sistema

### 3.1 Detección PCI / GPU
`lspci` confirma un controlador VGA AMD:
- `Advanced Micro Devices, Inc. [AMD/ATI] Device [1002:15bf]`

### 3.2 Estado del driver de kernel (amdgpu)
`dmesg` confirma:
- `amdgpu kernel modesetting enabled`
- VRAM reportada: `2048M`
- GTT reportada: `6779M`
- Se crea nodo KFD y se agrega el dispositivo:
  - `kfd ... added device 1002:15bf`

Se observó un warning del display-controller:
- `REG_WAIT timeout ... optc314_disable_crtc`
Se trata como advertencia del camino de display y no se considera bloqueante para el bring-up del runtime de cómputo.

---

## 4) Capacidad Vulkan (camino compute)

`vulkaninfo` enumera dos dispositivos:

- GPU0: iGPU AMD con RADV
  - `deviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`
  - `deviceName = AMD Unknown (RADV GFX1103_R1)`
  - `driverName = radv`

- GPU1: fallback por CPU
  - `deviceType = PHYSICAL_DEVICE_TYPE_CPU`
  - `deviceName = llvmpipe (LLVM ...)`
  - `driverName = llvmpipe`

### Warning del loader
Existe un warning del loader Vulkan:
- `Failed to CreateInstance in ICD 1. Skipping ICD.`

Interpretación:
- Un ICD instalado (por ejemplo intel/virtio) falla al inicializar en este sistema.
- El loader lo saltea y carga RADV correctamente.
- No es bloqueante. GRETA CORE seleccionará explícitamente la GPU AMD y evitará llvmpipe.

Para forzar AMD RADV en benchmarks:
- `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`

---

## 5) Estado ROCm / HIP (stack de cómputo AMD)

ROCm/HIP no está disponible actualmente en este sistema:
- `hipcc` no está instalado / no se encuentra.
- `rocminfo` no está instalado o no reporta.

Por time-to-value, variabilidad del ecosistema y restricciones frecuentes en APUs, ROCm/HIP no se elige como backend inicial.
ROCm/HIP sigue siendo requerido para la compatibilidad de frameworks (LOE‑5: Triton/PyTorch/JAX) cuando el stack sea estable en los sistemas objetivo.

---

## 6) Decisión

### Backend primario seleccionado (v1)
✅ **Vulkan Compute (Mesa RADV)**

Justificación:
- Ya está funcionando sobre la APU AMD actual.
- Toolchain estable y portable (no ata a un vendor).
- Permite selección explícita del dispositivo (evitar llvmpipe).
- Habilita construir un backend “desde cero” y luego interoperar con otros stacks.

### Backend diferido / opcional (futuro)
⏳ **ROCm / HIP**
- Se reconsidera más adelante si resulta simple y estable en las plataformas objetivo.

---

## 7) Implicancias de ingeniería para GRETA CORE

### 7.1 Alineación con la arquitectura del runtime
Módulos ya implementados en el runtime GRETA CORE:
- Host allocator (pooling/caching)
- Stream & Event (primitivas de plano de control)
- Telemetry (contadores/timers de bajo overhead)
- Dispatcher (dispatch CPU-only estilo “kernel”)

El backend Vulkan se integrará vía:
- selección de dispositivo
- scheduling en cola compute
- sincronización (fences/semaphores)
- manejo de buffers y memoria (allocations device)

### 7.2 Próximos hitos
1. Bootstrap backend Vulkan (instance, device, queue, empty submit smoke test)
2. Allocator de buffers device + staging
3. Primer kernel compute real:
   - baseline: memcpy/fill en device
4. Kernel central para primitivas LLM:
   - GEMM (multiplicación de matrices)
   - componentes de attention (después)

---

## 8) Autoría y propiedad

GRETA CORE pertenece y es desarrollado por:
**Leandro Emanuel Timberini**

Este documento forma parte del repositorio GRETA CORE y funciona como registro oficial de decisiones de ingeniería.
