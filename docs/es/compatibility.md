# GRETA CORE: Matriz de Compatibilidad AMD (Seguridad del Runtime)

Esta matriz define defaults seguros por tier de hardware. El objetivo es
prevenir cuelgues de GPU manteniendo un camino claro para habilitar FP16.

## Leyenda
- **FP16 default**: si FP16 se habilita sin override explicito.
- **Smoke recomendado**: perfil + tile size para gate de estabilidad.
- **Override**: flags para forzar FP16 o rutas inseguras.

## Tiers

### Tier D — Dev Solo CPU (sin ROCm)
- **FP16 default**: OFF
- **Smoke recomendado**: prototipos CPU-only
- **Notas**: Triton corre en fallback CPU; path GPU requiere ROCm.

### Tier A — iGPU / APU (RDNA3 iGPU, ej. Phoenix)
- **FP16 default**: OFF (healthcheck + blacklist activos)
- **Smoke recomendado**: `ultrasafe`, `SMOKE_TILE=8`
- **Notas**: usar fallback FP32. Solo habilitar FP16 con hipotesis clara.
- **Override**: `GRETA_VK_FP16_ALLOW_UNSAFE=1`

### Tier B — dGPU media (RDNA2/RDNA3 consumer)
- **FP16 default**: OFF salvo healthcheck limpio
- **Smoke recomendado**: `ultrasafe` → `default`
- **Notas**: habilitar FP16 solo tras multiples ejecuciones limpias.
- **Override**: `GRETA_VK_FP16_ALLOW_UNSAFE=1`

### Tier C — dGPU datacenter (CDNA / Instinct)
- **FP16 default**: ON si healthcheck pasa
- **Smoke recomendado**: `default`
- **Notas**: respetar blacklist; recolectar telemetria.
- **Override**: `GRETA_VK_FP16_ALLOW_UNSAFE=1` (solo tests controlados)

## Gate Smoke Requerido (todos los tiers)
1) `ultrasafe` (FP32)
2) `fill` (FP32)
3) `default` (FP32)
4) FP16 opcional si healthcheck limpio

## Defaults de entorno
- `GRETA_VK_SMOKE_PROFILE=ultrasafe`
- `GRETA_VK_SMOKE_TILE=8`
- `GRETA_VK_SMOKE_TIMEOUT_MS=2000`
