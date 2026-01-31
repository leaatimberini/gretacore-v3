# Framework Version Matrix (Draft)

Date: 2026-01-31

## Objective
Lock framework versions shipped with GRETA so install is deterministic and identical across Radeon dev and MI300X cloud.

## Targets
- **Radeon Dev** (APU/dGPU)
- **MI300X Cloud** (Instinct)

## Version Matrix (Candidate)
| Component | Radeon Dev | MI300X Cloud | Notes |
| --- | --- | --- | --- |
| ROCm | 7.1.1 (candidate) | 7.1.1 (candidate) | ROCm 7.1.1 supports PyTorch 2.9. citeturn1search4 |
| torch | 2.9 (ROCm build) | 2.9 (ROCm build) | ROCm 7.1.1 enables support for PyTorch 2.9. citeturn1search4 |
| triton | Bundled w/ ROCm PyTorch wheels | Bundled w/ ROCm PyTorch wheels | AMD docs install Triton via ROCm PyTorch wheels. citeturn1search2 |
| jax | TBD (blocked) | TBD (blocked) | ROCm JAX install path is inconsistent; no official rocm extra. citeturn0search3 |

## Lock Files
- `tools/compat/lock/greta_requirements.txt`
- `tools/compat/lock/greta_environment.yml`

## Pass Criteria
- Same versions across targets (unless vendor requires split builds).
- All compatibility prototypes report `STATUS=OK`.
- JAX ROCm install path resolved and pinned.
