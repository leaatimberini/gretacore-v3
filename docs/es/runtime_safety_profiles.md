# GRETA CORE: Perfiles de Seguridad del Runtime

## Proposito
Los perfiles de seguridad permiten que GRETA CORE se adapte a todo el hardware
AMD, controlando que features se habilitan y cuan agresivo es el runtime al
lanzar trabajo GPU. Esto es critico para estabilidad en iGPU/APU y mantiene una
ruta hacia alto rendimiento en dGPU estables.

## Perfiles (runtime smoke)
Estos perfiles se usan en `vk_gemm_runtime_smoke` y funcionan como gate minimo
y deterministico antes de habilitar rutas mas agresivas.

### Perfil: `ultrasafe`
- FP16 deshabilitado.
- Workgroups pequenos y dispatch conservador.
- Dims padded a boundary del tile.
- Barreras extra para reducir riesgos.
- Recomendado para: drivers nuevos, iGPU/APU, hardware desconocido.

### Perfil: `fill`
- FP16 deshabilitado.
- Ejecuta un shader simple de fill como sanity check de GPU.
- Recomendado para: confirmar submit + escritura/visibilidad de memoria.

### Perfil: `default`
- FP16 deshabilitado por defecto salvo override.
- Dispatch estandar con overhead minimo.
- Recomendado para: desarrollo normal cuando `ultrasafe` pasa.

## Variables de entorno (smoke)
- `GRETA_VK_SMOKE_PROFILE=ultrasafe|fill|default`
- `GRETA_VK_SMOKE_ALLOW_FP16=1` (inseguro en hardware inestable)
- `GRETA_VK_SMOKE_TIMEOUT_MS=2000`
- `GRETA_VK_SMOKE_TILE=8|16|32`

## Flujo recomendado
1) `ultrasafe` pasa
2) `fill` pasa
3) `default` pasa en FP32
4) FP16 opcional (solo si healthcheck limpio y sin blacklist)

## Regla de decision
Si algun perfil falla (timeout o error de validacion), el device se considera
FP16 inseguro y el runtime queda en FP32 hasta override explicito.
