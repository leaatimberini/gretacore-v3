# GRETA CORE – Framework Compatibility Prototypes

Purpose: minimal, reproducible scripts that validate the "same code" path for Triton, PyTorch, and JAX without Docker-only workflows.

## Layout
- `triton/vec_add.py` – Triton kernel sanity check (vector add)
- `pytorch/greta_extension_hello.py` – PyTorch custom op (CPU) sanity check
- `jax/jax_custom_call_hello.py` – JAX backend sanity check (custom call placeholder)
- `lock/greta_requirements.txt` – pinned versions (TBD)
- `lock/greta_environment.yml` – conda environment (TBD)

## Expected Status Lines
Each script prints a final line:
- `STATUS=OK` when the required runtime is available and the check passes.
- `STATUS=SKIPPED` when the runtime is not installed or the backend is unavailable.

## Notes
- These are prototypes. GPU backends and custom calls are expanded in LOE-5.
- No Docker is required; GRETA install bundles torch/triton/jax.
- Local dev uses `.venv` to avoid system Python restrictions.
- Triton prototype requires a ROCm-enabled torch build.
- CPU fallback is enabled for Triton when ROCm is unavailable.
