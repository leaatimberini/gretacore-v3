# GRETA CORE – Entorno de Desarrollo

Versión: 1.0  
Estado: Fundacional  
Fase del Proyecto: Fase 1 – Núcleo del Runtime  
Idioma: Español

---

## Propósito de este Documento

Este documento define el entorno oficial de desarrollo y benchmarking
de GRETA CORE.

Todos los benchmarks, mediciones de rendimiento y decisiones
arquitectónicas deben ser reproducibles dentro de este entorno.

Cualquier desviación debe documentarse explícitamente.

---

## 1. Sistema Operativo

- Distribución: Ubuntu Linux
- Versión: 22.04 LTS (Jammy Jellyfish)
- Kernel: Linux 5.15.x (kernel LTS por defecto)

Justificación:
Ubuntu 22.04 LTS ofrece estabilidad a largo plazo y el mejor equilibrio
entre soporte de drivers AMD, compatibilidad ROCm y madurez de tooling.

---

## 2. Hardware de Referencia

### CPU / APU
- Fabricante: AMD
- Modelo: Ryzen 5 8600G
- Arquitectura: Zen 4
- GPU Integrada: RDNA 3

### Memoria
- Tipo: DDR5
- Capacidad: 16 GB (actual)
- Configuración: Single Channel (temporal)
- Configuración Objetivo: Dual Channel (recomendado)

Nota:
La memoria single-channel limita severamente workloads sensibles al
ancho de banda. Todos los benchmarks deben registrar la configuración
de memoria.

### Almacenamiento
- Tipo: SSD NVMe
- Espacio libre mínimo: 50 GB
- Sistema de archivos: ext4

---

## 3. Stack Gráfico y de Cómputo

### Driver de Kernel
- Driver: amdgpu (incluido en el kernel)
- Verificación:
lspci | grep VGA
dmesg | grep amdgpu

### Backends de Cómputo
Primario:
- ROCm / HIP (preferido cuando esté disponible)

Secundario:
- Vulkan Compute (RADV) como alternativa o backend comparativo

El backend utilizado debe registrarse en todos los benchmarks.

---

## 4. Configuración ROCm (cuando aplique)

- Versión ROCm: a fijar por release
- Runtime HIP: habilitado
- Visibilidad del dispositivo verificada explícitamente

Verificación:
rocminfo
clinfo

Si ROCm no está disponible para el dispositivo objetivo, se utiliza
Vulkan Compute.

---

## 5. Toolchain

### Compiladores
- GCC: >= 11.x
- Clang / LLVM: >= 14.x

### Herramientas de Build
- CMake: >= 3.22
- Ninja (opcional, recomendado)

### Herramientas de Profiling
- perf
- rocprof (cuando ROCm esté activo)
- Capas de validación Vulkan (cuando aplique)

---

## 5.1 Packaging y Experiencia Dev
- Preferir installs pip/conda; Docker no es requerido para GRETA.
- Mismos pasos de build/run en Radeon dev y MI300X cloud.
- La instalación de GRETA incluye torch, triton y jax.
- En desarrollo local se usa `.venv` para evitar restricciones del Python del sistema.
- En notebooks sin ROCm, Triton usa fallback CPU para prototipos.

---

## 6. Configuración de Energía y Térmica

### Gobernador de CPU
- Modo: performance

Configurar con:
sudo cpupower frequency-set -g performance

### Consideraciones Térmicas
- Los benchmarks deben ejecutarse en condiciones térmicas estables
- El throttling debe detectarse y documentarse

Verificación:
watch -n1 sensors

---

## 7. Checklist de Validación del Entorno

Antes de ejecutar benchmarks, verificar:

- SO y kernel correctos
- Driver amdgpu cargado
- Backend de cómputo disponible (ROCm o Vulkan)
- Gobernador de CPU en performance
- Sistema en estado estable y sin carga

Cualquier fallo invalida los resultados.

---

## 8. Reglas de Reproducibilidad

- Todo cambio de entorno debe registrarse
- Las versiones del toolchain deben documentarse
- No se permiten cambios silenciosos de parámetros de kernel
- Benchmarks sin metadatos de entorno son inválidos

---

## Autoría

GRETA CORE fue concebido, fundado y es liderado por:

Leandro Emanuel Timberini  
Fundador y Arquitecto Principal de Sistemas
