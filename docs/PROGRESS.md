# GRETA Core Progress Index

## Phase Index

| Phase | Date | HEAD Hash | Objective | Root Cause | Result | Artifacts | AMD Report |
|-------|------|-----------|-----------|------------|--------|-----------|------------|
| B3.57.1 | 2026-02-04 | `a04fbc7` | RMSNorm divergence | `NORMOUT_SELECTION` | Confirmed | N/A | N/A |
| B3.58 | 2026-02-05 | `a04fbc7` | RMSNorm wiring audit | `UPSTREAM_X_MISMATCH` | X0/Ceros en decode0 | N/A | N/A |
| B3.59 | 2026-02-05 | `8da7ab6` | Embedding/DebugInput audit | TBD | In Progress | [B3.59](artifacts_remote/2026-02-04/b3_59/) | [AMD_B3_59](docs/AMD/2026_02_04_B3_59_embedding_debug_input_audit.md) |

---

## Technical Details (B3.59)
- **Objective**: Identify why `embedding_out/x_in` is zeroed in `decode0` when `GRETA_TRACE_STAGE_DEBUG_INPUT=1` is active.
- **Flags used**: `GRETA_TRACE_STAGE=1`, `GRETA_TRACE_STAGE_DEBUG_INPUT=1`.
- **Status**: Execution in progress.
