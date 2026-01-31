# GRETA CORE: Healthcheck FP16 y Blacklist

## Resumen
GRETA CORE ejecuta un healthcheck defensivo de FP16 para evitar cuelgues de GPU
en drivers o rutas de hardware inestables. Si el dispatch FP16 se considera
inseguro, el dispositivo queda en blacklist y el runtime cae a FP32.

Esto permite:
- Ejecucion estable en hardware iGPU/APU.
- Fallback seguro sin desactivar la GPU.
- Decisiones auditables por device (cache + archivo de blacklist).

## Archivos
- Blacklist FP16 (por device key):
  - `~/.cache/gretacore/vk_fp16_blacklist.txt`
- Cache de autotune (con tags):
  - `~/.cache/gretacore/vk_autotune.json`
  - Tag: `bucket=meta:fp16_blacklist`, `winner=1`
  - Tag: `bucket=meta:fp16_fallback_reason`, `winner=<razon>`

## Variables de entorno
### Runtime
- `GRETA_VK_FP16_HEALTHCHECK=1`
  Ejecuta el healthcheck de FP16 al init (default: OFF).
- `GRETA_VK_FP16_HEALTHCHECK_TIMEOUT_MS=2000`
  Timeout del healthcheck.
- `GRETA_VK_FP16_ALLOW_UNSAFE=1`
  Fuerza FP16 incluso si esta blacklisted (peligroso).
- `GRETA_VK_FP16_BLACKLIST_NO_WRITE=1`
  No escribir `vk_fp16_blacklist.txt`.

### Autotune
- `GRETA_VK_FP16_ALLOW_UNSAFE=1`
  Permite candidatos FP16 aunque esten blacklisted.
- `GRETA_VK_FP16_BLACKLIST_TAG_NO_WRITE=1`
  No escribir el tag `meta:fp16_blacklist` en cache.

## Comportamiento esperado
1) Si el device esta blacklisted, FP16 se omite.
2) Autotune no ejecuta candidatos FP16.
3) Runtime cae a FP32 y continua.
4) Si autotune se saltea por blacklist FP16, el cache igual queda taggeado con
   `meta:fp16_blacklist` y `meta:fp16_fallback_reason`.

## Telemetria en runtime
Cuando hay fallback, se imprime:
```
RUNTIME: active_precision=fp32
RUNTIME: fallback_reason=fp16_blacklisted device_key=... blacklist_path=...
```

## Por que importa
GRETA CORE debe adaptarse al hardware disponible. Este mecanismo garantiza
estabilidad sin perder aceleracion GPU, incluso en iGPU donde FP16 puede ser
inestable.
