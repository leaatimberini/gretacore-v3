# Framework Compatibility Plan (Draft)

Date: 2026-01-31

## Objective
Enable the same code to run on Radeon dev machines and MI300X in the cloud without changes and without Docker-only workflows.

## Scope
- **Triton** (OpenAI): kernel authoring frontend
- **PyTorch**: training/inference integration
- **JAX**: research + compiler stack integration
- **NVIDIA-only libraries** (cuDNN / TensorRT): parity via AMD equivalents

## Non-Goals
- Shipping NVIDIA binaries
- Depending on CUDA for GRETA runtime

## Compatibility Strategy
1) **Triton-first kernel path**
   - Triton frontend supported for AMD backend targets.
   - GRETA kernels exposed as Triton-compatible ops where needed.

2) **PyTorch bridge**
   - ROCm wheels as baseline.
   - GRETA custom ops + dispatch for critical kernels.

3) **JAX bridge (PJRT/XLA)**
   - Use AMD backend for JAX.
   - GRETA custom calls for hot kernels.

4) **cuDNN/TensorRT parity mapping**
   - Map to AMD equivalents (MIOpen, rocBLAS, MIGraphX) or GRETA kernels.
   - Provide documented fallbacks with clear performance expectations.

## Packaging & DX (No Docker Requirement)
- One-command install (pip/conda) for dev and cloud.
- Same build/run steps on Radeon and MI300X.
- Results captured via standardized benchmark scripts and CSV.
- GRETA installer bundles torch, triton, and jax (no extra installs required).

## Pass/Fail Gates
- **Same code** runs on Radeon and MI300X without edits.
- **Install steps** ≤ 3 commands; no Docker required.
- **Framework parity**: key kernels run via Triton/PyTorch/JAX path.
- **Reproducible results**: benchmarks archived with metadata.
- **Version lock**: pinned versions defined in `docs/en/strategy/framework_versions.md`.

## Prototype Artifacts
- `tools/compat/triton/vec_add.py`
- `tools/compat/pytorch/greta_extension_hello.py`
- `tools/compat/jax/jax_custom_call_hello.py`

## Prototype Status Criteria
- Triton: `STATUS=OK` on ROCm-enabled AMD backend.
- PyTorch: `STATUS=OK` for custom op (CPU baseline); GPU path follows LOE-5.
- JAX: `STATUS=OK` on backend; custom call remains TODO.
- Triton CPU fallback is acceptable for notebook dev (no ROCm).
