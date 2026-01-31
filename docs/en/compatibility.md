# GRETA CORE: AMD Compatibility Matrix (Runtime Safety)

This matrix defines safe defaults by hardware tier. It is designed to prevent
GPU hangs while keeping a clear path toward FP16 enablement.

## Legend
- **FP16 default**: whether FP16 is enabled without explicit override.
- **Recommended smoke**: profile + tile size for stability gate.
- **Override**: env flags required to force FP16 or unsafe paths.

## Tiers

### Tier D — CPU-Only Dev (No ROCm)
- **FP16 default**: OFF
- **Recommended smoke**: CPU-only prototypes
- **Notes**: Triton runs in CPU fallback mode; GPU path requires ROCm.

### Tier A — iGPU / APU (RDNA3 iGPU, e.g. Phoenix)
- **FP16 default**: OFF (healthcheck + blacklist enforced)
- **Recommended smoke**: `ultrasafe`, `SMOKE_TILE=8`
- **Notes**: use FP32 fallback. Only enable FP16 with a strict hypothesis.
- **Override**: `GRETA_VK_FP16_ALLOW_UNSAFE=1`

### Tier B — Midrange dGPU (RDNA2/RDNA3 consumer)
- **FP16 default**: OFF unless healthcheck passes cleanly
- **Recommended smoke**: `ultrasafe` → `default`
- **Notes**: enable FP16 only after repeated clean runs.
- **Override**: `GRETA_VK_FP16_ALLOW_UNSAFE=1`

### Tier C — Datacenter dGPU (CDNA / Instinct)
- **FP16 default**: ON after healthcheck passes
- **Recommended smoke**: `default`
- **Notes**: still honor blacklist; collect telemetry for regressions.
- **Override**: `GRETA_VK_FP16_ALLOW_UNSAFE=1` (only for controlled tests)

## Required Smoke Gate (all tiers)
1) `ultrasafe` (FP32)
2) `fill` (FP32)
3) `default` (FP32)
4) Optional FP16 if healthcheck is clean

## Environment Defaults
- `GRETA_VK_SMOKE_PROFILE=ultrasafe`
- `GRETA_VK_SMOKE_TILE=8`
- `GRETA_VK_SMOKE_TIMEOUT_MS=2000`
