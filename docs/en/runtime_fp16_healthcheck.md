# GRETA CORE: FP16 Healthcheck & Blacklist

## Overview
GRETA CORE performs a defensive FP16 healthcheck to avoid GPU hangs on unstable
drivers or hardware paths. If FP16 dispatch is detected as unsafe, the device is
marked as FP16-blacklisted and the runtime falls back to FP32.

This enables:
- Stable execution on iGPU/APU-class hardware.
- Safe fallback without disabling the GPU entirely.
- Auditable device-level decisions (cache + blacklist file).

## Files
- FP16 blacklist file (per device key):
  - `~/.cache/gretacore/vk_fp16_blacklist.txt`
- Autotune cache (with tags):
  - `~/.cache/gretacore/vk_autotune.json`
  - Tag: `bucket=meta:fp16_blacklist`, `winner=1`
  - Tag: `bucket=meta:fp16_fallback_reason`, `winner=<reason>`

## Environment Variables
### Runtime
- `GRETA_VK_FP16_HEALTHCHECK=1`
  Runs FP16 healthcheck on init (default: OFF).
- `GRETA_VK_FP16_HEALTHCHECK_TIMEOUT_MS=2000`
  Timeout for healthcheck submit.
- `GRETA_VK_FP16_ALLOW_UNSAFE=1`
  Force FP16 even if blacklisted (dangerous).
- `GRETA_VK_FP16_BLACKLIST_NO_WRITE=1`
  Do not write `vk_fp16_blacklist.txt`.

### Autotune
- `GRETA_VK_FP16_ALLOW_UNSAFE=1`
  Allows FP16 candidates even if blacklisted.
- `GRETA_VK_FP16_BLACKLIST_TAG_NO_WRITE=1`
  Do not write the cache tag `meta:fp16_blacklist`.

## Expected Behavior
1) If the device is blacklisted, FP16 is skipped.
2) Autotune candidates using FP16 are skipped automatically.
3) Runtime falls back to FP32 and continues execution.
4) If autotune is skipped due to FP16 blacklist, the cache is still tagged with
   `meta:fp16_blacklist` and `meta:fp16_fallback_reason`.

## Runtime Telemetry
When the runtime falls back, it prints:
```
RUNTIME: active_precision=fp32
RUNTIME: fallback_reason=fp16_blacklisted device_key=... blacklist_path=...
```

## Why This Matters
GRETA CORE must adapt to the hardware available. This mechanism ensures
stability while retaining GPU acceleration, even on iGPUs where FP16 paths can
be unstable.
