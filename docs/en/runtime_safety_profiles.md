# GRETA CORE: Runtime Safety Profiles

## Purpose
Safety profiles allow GRETA CORE to adapt to a wide range of AMD hardware by
controlling which features are enabled and how aggressively the runtime executes
GPU work. This is critical for iGPU/APU stability while preserving a path to
higher performance on stable dGPU environments.

## Profiles (runtime smoke)
These profiles are used by `vk_gemm_runtime_smoke` and are intended as a
minimal, deterministic gate before enabling more aggressive paths.

### Profile: `ultrasafe`
- FP16 disabled.
- Small workgroups and conservative dispatch.
- Padded dims to tile boundary.
- Extra barriers to reduce hazard risk.
- Recommended for: new drivers, iGPU/APU, unknown devices.

### Profile: `fill`
- FP16 disabled.
- Runs a simple compute fill shader as a GPU sanity check.
- Recommended for: confirming queue submission + memory write/visibility.

### Profile: `default`
- FP16 disabled by default unless explicitly allowed.
- Standard dispatch path with minimal overhead.
- Recommended for: typical development once `ultrasafe` passes.

## Environment Variables (smoke)
- `GRETA_VK_SMOKE_PROFILE=ultrasafe|fill|default`
- `GRETA_VK_SMOKE_ALLOW_FP16=1` (unsafe on unstable devices)
- `GRETA_VK_SMOKE_TIMEOUT_MS=2000`
- `GRETA_VK_SMOKE_TILE=8|16|32`

## Recommended Flow
1) `ultrasafe` passes
2) `fill` passes
3) `default` passes in FP32
4) Optional FP16 (only if healthcheck is clean and no blacklist)

## Decision Rule
If any profile fails (timeout or validation error), the device is treated as
FP16-unsafe and runtime remains in FP32 until explicit override.
